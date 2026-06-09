"""Serial FLUX GPU render queue — one in-flight render per worker process."""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class RenderJob:
    """One FLUX render request."""

    job_id: int
    idx: int
    prompt: str
    sap_out: dict
    prompt_dir: Path
    run_id: str
    status_path: Path
    record: dict
    seeds: list[int]
    num_inference_steps: int
    render_fn: Callable[..., list[Path]]
    render_kwargs: dict[str, Any] = field(default_factory=dict)


class FluxRenderQueue:
    """
    Dedicated GPU worker thread consuming render jobs FIFO.

    All non-GPU stages (SAP, VL) run elsewhere; only this queue touches FLUX.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[RenderJob | None] = queue.Queue()
        self._results: dict[int, list[Path]] = {}
        self._errors: dict[int, BaseException] = {}
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._depth = 0
        self._worker = threading.Thread(target=self._run, name="flux-render", daemon=True)
        self._worker.start()

    @property
    def depth(self) -> int:
        with self._lock:
            return self._queue.qsize() + (1 if self._depth > 0 else 0)

    def submit(self, job: RenderJob) -> int:
        self._queue.put(job)
        return job.job_id

    def get(self, job_id: int, timeout: float | None = None) -> list[Path]:
        deadline = None if timeout is None else (__import__("time").time() + timeout)
        while True:
            with self._lock:
                if job_id in self._results:
                    return self._results.pop(job_id)
                if job_id in self._errors:
                    raise self._errors.pop(job_id)
            if deadline is not None and __import__("time").time() >= deadline:
                raise TimeoutError(f"Render job {job_id} timed out")
            __import__("time").sleep(0.02)

    def shutdown(self, wait: bool = True) -> None:
        self._queue.put(None)
        if wait:
            self._worker.join(timeout=120)

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                self._queue.task_done()
                break
            with self._lock:
                self._depth += 1
            try:
                paths = job.render_fn(
                    job.idx,
                    job.prompt,
                    job.sap_out,
                    job.prompt_dir,
                    job.run_id,
                    job.status_path,
                    job.record,
                    seeds=job.seeds,
                    num_inference_steps=job.num_inference_steps,
                    **job.render_kwargs,
                )
                with self._lock:
                    self._results[job.job_id] = paths
            except BaseException as exc:
                with self._lock:
                    self._errors[job.job_id] = exc
            finally:
                with self._lock:
                    self._depth = max(0, self._depth - 1)
                self._queue.task_done()


_global_queue: FluxRenderQueue | None = None
_global_lock = threading.Lock()


def get_render_queue() -> FluxRenderQueue:
    """Process-wide singleton render queue (one per evaluator worker)."""
    global _global_queue
    with _global_lock:
        if _global_queue is None:
            _global_queue = FluxRenderQueue()
        return _global_queue


def shutdown_render_queue() -> None:
    global _global_queue
    with _global_lock:
        if _global_queue is not None:
            _global_queue.shutdown(wait=True)
            _global_queue = None
