"""SAP patches for OpenEvolve: count only successful eval cycles, restart pool on failure."""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from typing import Dict, List, Optional

from openevolve.database import Program
from openevolve.utils.metrics_utils import safe_numeric_average

logger = logging.getLogger(__name__)

_POOL_FAILURE_MARKERS = (
    "process pool was terminated abruptly",
    "process pool is not usable",
    "child process terminated abruptly",
    "broken process pool",
)


def _is_pool_failure(exc: BaseException | str) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _POOL_FAILURE_MARKERS)


async def _sap_restart_pool(controller) -> None:
    """Recycle the process pool after a worker crash."""
    logger.warning("SAP: restarting process pool after worker failure")
    if controller.executor:
        try:
            controller.executor.shutdown(wait=False, cancel_futures=True)
        except Exception as exc:
            logger.debug("Pool shutdown during restart: %s", exc)
    controller.executor = None
    controller.start()


async def _sap_drain_pending(pending_futures: Dict[int, Future], timeout_seconds: float) -> None:
    """Wait for in-flight evals to finish before tearing down the pool."""
    if not pending_futures:
        return
    logger.info("SAP: draining %d in-flight iteration(s) before shutdown", len(pending_futures))
    deadline = asyncio.get_event_loop().time() + timeout_seconds
    while pending_futures and asyncio.get_event_loop().time() < deadline:
        done = [iteration for iteration, fut in pending_futures.items() if fut.done()]
        for iteration in done:
            fut = pending_futures.pop(iteration)
            try:
                fut.result(timeout=1)
            except Exception as exc:
                logger.warning("SAP: drained iteration %s with error: %s", iteration, exc)
        if pending_futures:
            await asyncio.sleep(0.05)
    if pending_futures:
        logger.warning(
            "SAP: %d iteration(s) still pending after drain timeout; canceling",
            len(pending_futures),
        )
        for fut in pending_futures.values():
            fut.cancel()
        pending_futures.clear()


async def sap_run_evolution(
    self,
    start_iteration: int,
    max_iterations: int,
    target_score: Optional[float] = None,
    checkpoint_callback=None,
):
    """
    Run evolution until `max_iterations` **successful** GPU eval cycles complete.

    Failed LLM mutations, timeouts, and pool crashes are retried and do not count
    toward the iteration budget.
    """
    if not self.executor:
        raise RuntimeError("Process pool not started")

    max_total_attempts = int(
        __import__("os").getenv("SAP_MAX_EVOLUTION_ATTEMPTS", str(max_iterations * 8))
    )
    total_attempt_cap = start_iteration + max_total_attempts

    logger.info(
        "SAP: evolution target = %d successful eval cycles (attempt cap %d, start id %d)",
        max_iterations,
        max_total_attempts,
        start_iteration,
    )

    pending_futures: Dict[int, Future] = {}
    island_pending: Dict[int, List[int]] = {i: [] for i in range(self.num_islands)}
    batch_size = min(self.num_workers * 2, max_iterations)
    batch_per_island = max(1, batch_size // self.num_islands) if batch_size > 0 else 0

    next_attempt_id = start_iteration
    successful_iterations = 0
    last_success_attempt_id = start_iteration - 1
    pool_broken = False

    async def _fill_pipeline() -> None:
        nonlocal next_attempt_id, pool_broken
        for island_id in range(self.num_islands):
            while (
                len(island_pending[island_id]) < batch_per_island
                and len(pending_futures) < batch_size
                and successful_iterations < max_iterations
                and next_attempt_id < total_attempt_cap
                and not self.shutdown_event.is_set()
            ):
                future = self._submit_iteration(next_attempt_id, island_id)
                if future is None:
                    pool_broken = True
                    return
                pending_futures[next_attempt_id] = future
                island_pending[island_id].append(next_attempt_id)
                next_attempt_id += 1

    await _fill_pipeline()

    early_stopping_enabled = self.config.early_stopping_patience is not None
    if early_stopping_enabled:
        best_score = float("-inf")
        iterations_without_improvement = 0
        logger.info(
            "Early stopping enabled: patience=%s, threshold=%s, metric=%s",
            self.config.early_stopping_patience,
            self.config.convergence_threshold,
            self.config.early_stopping_metric,
        )
    else:
        logger.info("Early stopping disabled")

    timeout_seconds = self.config.evaluator.timeout + 30

    while (
        successful_iterations < max_iterations
        and next_attempt_id < total_attempt_cap
        and not self.shutdown_event.is_set()
        and not getattr(self, "early_stopping_triggered", False)
    ):
        await _fill_pipeline()

        if not pending_futures:
            if pool_broken:
                await _sap_restart_pool(self)
                pool_broken = False
                await _fill_pipeline()
            await asyncio.sleep(0.05)
            continue

        completed_iteration = None
        for iteration, future in list(pending_futures.items()):
            if future.done():
                completed_iteration = iteration
                break

        if completed_iteration is None:
            await asyncio.sleep(0.01)
            continue

        future = pending_futures.pop(completed_iteration)
        success = False

        try:
            result = future.result(timeout=timeout_seconds)

            if result.error:
                logger.warning("Iteration %d error: %s", completed_iteration, result.error)
            elif result.child_program_dict:
                child_program = Program(**result.child_program_dict)
                self.database.add(
                    child_program,
                    iteration=completed_iteration,
                    target_island=result.target_island,
                )

                if result.artifacts:
                    self.database.store_artifacts(child_program.id, result.artifacts)

                if self.evolution_tracer:
                    parent_program = (
                        self.database.get(result.parent_id) if result.parent_id else None
                    )
                    if parent_program:
                        island_id = child_program.metadata.get(
                            "island", self.database.current_island
                        )
                        self.evolution_tracer.log_trace(
                            iteration=completed_iteration,
                            parent_program=parent_program,
                            child_program=child_program,
                            prompt=result.prompt,
                            llm_response=result.llm_response,
                            artifacts=result.artifacts,
                            island_id=island_id,
                            metadata={
                                "iteration_time": result.iteration_time,
                                "changes": child_program.metadata.get("changes", ""),
                            },
                        )

                if result.prompt:
                    self.database.log_prompt(
                        template_key=(
                            "full_rewrite_user"
                            if not self.config.diff_based_evolution
                            else "diff_user"
                        ),
                        program_id=child_program.id,
                        prompt=result.prompt,
                        responses=[result.llm_response] if result.llm_response else [],
                    )

                island_id = child_program.metadata.get("island", self.database.current_island)
                self.database.increment_island_generation(island_idx=island_id)

                if self.database.should_migrate():
                    logger.info("Performing migration at iteration %d", completed_iteration)
                    self.database.migrate_programs()
                    self.database.log_island_status()

                logger.info(
                    "Iteration %d: Program %s (parent: %s) completed in %.2fs "
                    "[successful %d/%d]",
                    completed_iteration,
                    child_program.id,
                    result.parent_id,
                    result.iteration_time,
                    successful_iterations + 1,
                    max_iterations,
                )

                if child_program.metrics:
                    metrics_str = ", ".join(
                        f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                        for k, v in child_program.metrics.items()
                    )
                    logger.info("Metrics: %s", metrics_str)

                if self.database.best_program_id == child_program.id:
                    logger.info(
                        "🌟 New best solution at iteration %d: %s",
                        completed_iteration,
                        child_program.id,
                    )

                successful_iterations += 1
                last_success_attempt_id = completed_iteration
                success = True

                if (
                    successful_iterations > 0
                    and successful_iterations % self.config.checkpoint_interval == 0
                    and checkpoint_callback
                ):
                    logger.info(
                        "Checkpoint interval reached at successful iteration %d (attempt %d)",
                        successful_iterations,
                        completed_iteration,
                    )
                    self.database.log_island_status()
                    checkpoint_callback(completed_iteration)

                if target_score is not None and child_program.metrics:
                    if (
                        "combined_score" in child_program.metrics
                        and child_program.metrics["combined_score"] >= target_score
                    ):
                        logger.info(
                            "Target score %s reached at iteration %d",
                            target_score,
                            completed_iteration,
                        )
                        break

                if early_stopping_enabled and child_program.metrics:
                    current_score = None
                    if self.config.early_stopping_metric in child_program.metrics:
                        current_score = child_program.metrics[self.config.early_stopping_metric]
                    elif self.config.early_stopping_metric == "combined_score":
                        current_score = safe_numeric_average(child_program.metrics)
                    else:
                        current_score = safe_numeric_average(child_program.metrics)

                    if current_score is not None and isinstance(current_score, (int, float)):
                        if self.config.early_stopping_patience > 0:
                            improvement = current_score - best_score
                            if improvement >= self.config.convergence_threshold:
                                best_score = current_score
                                iterations_without_improvement = 0
                            else:
                                iterations_without_improvement += 1
                            if (
                                iterations_without_improvement
                                >= self.config.early_stopping_patience
                            ):
                                self.early_stopping_triggered = True
                                logger.info(
                                    "🛑 Early stopping at successful iteration %d",
                                    successful_iterations,
                                )
                                break

        except FutureTimeoutError:
            logger.error(
                "⏰ Iteration %d timed out after %ds; retrying",
                completed_iteration,
                timeout_seconds,
            )
            future.cancel()
        except Exception as exc:
            logger.error(
                "Error processing result from iteration %d: %s",
                completed_iteration,
                exc,
            )
            if _is_pool_failure(exc):
                pool_broken = True
                await _sap_restart_pool(self)

        for island_id, iteration_list in island_pending.items():
            if completed_iteration in iteration_list:
                iteration_list.remove(completed_iteration)
                break

        if not success:
            await _fill_pipeline()

    await _sap_drain_pending(pending_futures, timeout_seconds)

    if self.shutdown_event.is_set():
        logger.info("✅ Evolution completed - Shutdown requested")
    elif getattr(self, "early_stopping_triggered", False):
        logger.info("✅ Evolution completed - Early stopping triggered")
    elif successful_iterations >= max_iterations:
        logger.info(
            "✅ Evolution completed - %d/%d successful eval cycles finished",
            successful_iterations,
            max_iterations,
        )
    else:
        logger.warning(
            "⚠️ Evolution stopped early - only %d/%d successful eval cycles "
            "(attempt cap %d reached)",
            successful_iterations,
            max_iterations,
            max_total_attempts,
        )

    self._sap_last_success_attempt_id = last_success_attempt_id
    self._sap_successful_iterations = successful_iterations
    return self.database.get_best_program()


def patch_controller_final_checkpoint() -> None:
    """Save final checkpoint at last real successful iteration, not a synthetic number."""
    from openevolve.controller import OpenEvolve

    if getattr(OpenEvolve, "_sap_final_checkpoint_patched", False):
        return

    original = OpenEvolve._run_evolution_with_checkpoints

    async def _run_evolution_with_checkpoints(
        self, start_iteration: int, max_iterations: int, target_score: Optional[float]
    ) -> None:
        await original(self, start_iteration, max_iterations, target_score)

        if self.parallel_controller is None:
            return
        if self.parallel_controller.shutdown_event.is_set():
            return
        if self.parallel_controller.early_stopping_triggered:
            pass

        ctrl = self.parallel_controller
        last_id = getattr(ctrl, "_sap_last_success_attempt_id", None)
        successful = getattr(ctrl, "_sap_successful_iterations", 0)
        if last_id is None or last_id < start_iteration:
            return
        if successful > 0 and last_id % self.config.checkpoint_interval == 0:
            return
        if successful > 0:
            logger.info(
                "SAP: saving final checkpoint at attempt %d (%d successful cycles)",
                last_id,
                successful,
            )
            self._save_checkpoint(last_id)

    OpenEvolve._run_evolution_with_checkpoints = _run_evolution_with_checkpoints
    OpenEvolve._sap_final_checkpoint_patched = True


def install_evolution_patch() -> None:
    """Replace OpenEvolve parallel loop with SAP success-counting version."""
    from openevolve.process_parallel import ProcessParallelController

    if getattr(ProcessParallelController, "_sap_evolution_patched", False):
        return

    ProcessParallelController.run_evolution = sap_run_evolution
    ProcessParallelController._sap_evolution_patched = True
    patch_controller_final_checkpoint()
    logger.info("SAP evolution patch installed (successful-cycle counting + pool restart)")
