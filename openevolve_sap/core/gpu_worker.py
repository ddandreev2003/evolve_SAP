"""GPU assignment for OpenEvolve process-pool workers (pickle-safe)."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path

_ORIGINAL_WORKER_INIT = None


def _gpu_ids_from_env() -> list[int]:
    raw = os.getenv("SAP_GPU_IDS", "0,1,2,3").strip()
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
    # Workers keep the model loaded across evals in the same process
    os.environ["SAP_RELEASE_MODEL_AFTER_EVAL"] = "0"
    return physical_gpu


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
        skip_keys = {"CUDA_VISIBLE_DEVICES", "SAP_RELEASE_MODEL_AFTER_EVAL"}
        safe_env = {k: v for k, v in parent_env.items() if k not in skip_keys}
        os.environ.update(safe_env)
    return _resolve_original_worker_init()(config_dict, evaluation_file, parent_env)


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
        from run_SAP_flux import release_model

        release_model()
        return original_start(self)

    ProcessParallelController.start = start_with_release
    ProcessParallelController._sap_release_patched = True


def install_gpu_worker_patch() -> None:
    if os.getenv("SAP_ENABLE_GPU_PATCH", "1").strip().lower() in {"0", "false", "no"}:
        return
    patch_openevolve_worker_init()
    patch_pool_start_release_model()
