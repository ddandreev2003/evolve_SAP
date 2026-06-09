"""GPU assignment for OpenEvolve process-pool workers (pickle-safe)."""
from __future__ import annotations

import fcntl
import logging
import os
from pathlib import Path

from openevolve_sap.sap_eval_settings import is_single_gpu_mode, preload_flux_on_worker_start

logger = logging.getLogger(__name__)

_ORIGINAL_WORKER_INIT = None


def _gpu_ids_from_env() -> list[int]:
    raw = os.getenv("SAP_GPU_IDS", "0").strip()
    if not raw:
        return [0]
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _counter_path() -> Path:
    exp = os.getenv("SAP_EXPERIMENT_DIR", "").strip()
    base = Path(exp) if exp else Path(__file__).resolve().parents[1]
    base.mkdir(parents=True, exist_ok=True)
    return base / ".gpu_assign_counter"


def assign_gpu_for_worker() -> int:
    """Pick next GPU from SAP_GPU_IDS using a file lock (spawn-safe)."""
    gpu_ids = _gpu_ids_from_env()
    if is_single_gpu_mode() or len(gpu_ids) == 1:
        physical_gpu = gpu_ids[0]
        index = 0
    else:
        counter_file = _counter_path()
        counter_file.parent.mkdir(parents=True, exist_ok=True)
        with open(counter_file, "a+", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.seek(0)
            content = f.read().strip()
            index = int(content) if content else 0
            physical_gpu = gpu_ids[index % len(gpu_ids)]
            f.seek(0)
            f.truncate()
            f.write(str(index + 1))
            f.flush()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    os.environ["SAP_PHYSICAL_GPU_ID"] = str(physical_gpu)
    os.environ["SAP_CUDA_DEVICE"] = "0"
    os.environ["SAP_WORKER_ID"] = f"worker_{index}"
    os.environ["SAP_RELEASE_MODEL_AFTER_EVAL"] = "0"
    if is_single_gpu_mode() or len(gpu_ids) == 1:
        os.environ["SAP_KEEP_MODEL_LOADED"] = "1"
    return physical_gpu


def _preload_flux_in_worker() -> None:
    if not preload_flux_on_worker_start():
        return
    try:
        from openevolve_sap.evaluator import warmup_flux_model

        warmup_flux_model()
        logger.info(
            "SAP: FLUX preloaded on GPU %s (worker=%s)",
            os.getenv("SAP_PHYSICAL_GPU_ID"),
            os.getenv("SAP_WORKER_ID"),
        )
    except Exception as exc:
        logger.warning("SAP: FLUX preload failed: %s", exc)


def _resolve_original_worker_init():
    """
    Return OpenEvolve's real _worker_init.

    Parent process: captured in patch_openevolve_worker_init().
    Spawned children: fresh import still has the original on process_parallel.
    """
    global _ORIGINAL_WORKER_INIT
    if _ORIGINAL_WORKER_INIT is not None:
        return _ORIGINAL_WORKER_INIT
    import openevolve.process_parallel as pp

    fn = pp._worker_init
    if fn is sap_worker_init:
        raise RuntimeError(
            "GPU worker patch not installed in parent: call install_gpu_worker_patch() "
            "before starting the process pool"
        )
    _ORIGINAL_WORKER_INIT = fn
    return fn


def sap_worker_init(config_dict, evaluation_file, parent_env=None):
    """
    Top-level initializer for ProcessPoolExecutor (must be picklable).
    Pins one GPU per worker, then runs OpenEvolve worker setup.
    """
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    assign_gpu_for_worker()
    if parent_env:
        # Do not let parent env override per-worker GPU pinning or release policy
        skip_keys = {
            "CUDA_VISIBLE_DEVICES",
            "SAP_RELEASE_MODEL_AFTER_EVAL",
            "SAP_KEEP_MODEL_LOADED",
        }
        safe_env = {k: v for k, v in parent_env.items() if k not in skip_keys}
        os.environ.update(safe_env)
    result = _resolve_original_worker_init()(config_dict, evaluation_file, parent_env)
    _preload_flux_in_worker()
    return result


def patch_openevolve_worker_init() -> None:
    """Replace openevolve.process_parallel._worker_init with sap_worker_init."""
    global _ORIGINAL_WORKER_INIT
    import openevolve.process_parallel as pp

    if pp._worker_init is sap_worker_init:
        return

    if _ORIGINAL_WORKER_INIT is None:
        _ORIGINAL_WORKER_INIT = pp._worker_init

    pp._worker_init = sap_worker_init


def patch_pool_start_release_model() -> None:
    """Release FLUX from parent process before worker pool starts."""
    from openevolve.process_parallel import ProcessParallelController

    if getattr(ProcessParallelController, "_sap_release_patched", False):
        return

    original_start = ProcessParallelController.start

    def start_with_release(self):
        from openevolve_sap.evaluator import release_cached_model

        release_cached_model()
        logger.info("SAP: released FLUX from parent before worker pool start")
        return original_start(self)

    ProcessParallelController.start = start_with_release
    ProcessParallelController._sap_release_patched = True


def install_gpu_worker_patch() -> None:
    if os.getenv("SAP_ENABLE_GPU_PATCH", "1").strip().lower() in {"0", "false", "no"}:
        return
    patch_openevolve_worker_init()
    patch_pool_start_release_model()
