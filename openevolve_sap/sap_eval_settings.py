"""Shared SAP evaluation settings (env + optional YAML overrides)."""
from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return int(raw)


def get_image_height() -> int:
    return _env_int("SAP_IMAGE_HEIGHT", 512)


def get_image_width() -> int:
    return _env_int("SAP_IMAGE_WIDTH", 512)


def get_num_inference_steps() -> int:
    """Default inference steps (stage2 / full eval)."""
    return _env_int("SAP_NUM_INFERENCE_STEPS", 20)


def get_stage1_num_inference_steps() -> int:
    """Fast cascade stage1 filter."""
    return _env_int("SAP_STAGE1_NUM_INFERENCE_STEPS", 15)


def get_stage2_num_inference_steps() -> int:
    raw = os.getenv("SAP_STAGE2_NUM_INFERENCE_STEPS", "").strip()
    if raw:
        return int(raw)
    return get_num_inference_steps()


def get_seeds_list() -> list[int]:
    """Comma-separated seeds for FLUX generation (stage2 / full eval)."""
    raw = os.getenv("SAP_SEEDS_LIST", "30498").strip()
    if not raw:
        return [30498]
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def get_stage1_seeds_list() -> list[int]:
    """Single seed for fast cascade stage1."""
    raw = os.getenv("SAP_STAGE1_SEEDS_LIST", "").strip()
    if raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    return [get_seeds_list()[0]]


def get_stage2_seeds_list() -> list[int]:
    raw = os.getenv("SAP_STAGE2_SEEDS_LIST", "").strip()
    if raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    return get_seeds_list()


def get_eval_seeds_for_profile(eval_profile: str) -> list[int]:
    if eval_profile == "stage1":
        return get_stage1_seeds_list()
    return get_stage2_seeds_list()


def get_eval_steps_for_profile(eval_profile: str) -> int:
    if eval_profile == "stage1":
        return get_stage1_num_inference_steps()
    return get_stage2_num_inference_steps()


def get_vl_max_tokens() -> int:
    return max(256, _env_int("SAP_VL_MAX_TOKENS", 1024))


def system_ram_gb() -> float:
    """Total system RAM from /proc/meminfo (GiB)."""
    with open("/proc/meminfo", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                kib = int(line.split()[1])
                return kib / (1024 * 1024)
    return 128.0


def get_ram_limit_gb() -> float:
    """
    Per-process RSS limit (GiB).

    Default: 75% of MemTotal, but always kept a few GiB below physical RAM so the
    soft guard can actually fire on small hosts (e.g. ~23 GiB on a 31 GiB box).
    Override via SAP_RAM_LIMIT_GB.
    """
    raw = os.getenv("SAP_RAM_LIMIT_GB", "").strip()
    if raw:
        return float(raw)
    total = system_ram_gb()
    return min(max(8.0, total * 0.75), total - 4.0)


def get_cuda_device_index() -> int:
    """Local CUDA index inside CUDA_VISIBLE_DEVICES (usually 0)."""
    return _env_int("SAP_CUDA_DEVICE", 0)


def get_physical_gpu_id() -> str:
    return os.getenv("SAP_PHYSICAL_GPU_ID", os.getenv("CUDA_VISIBLE_DEVICES", "0"))


def get_worker_id() -> str:
    return os.getenv("SAP_WORKER_ID", "worker_unknown")


def get_vl_max_concurrent() -> int:
    """Max parallel VL alignment API calls (RouterAI)."""
    return max(1, _env_int("SAP_VL_MAX_CONCURRENT", 3))


def get_sap_max_concurrent() -> int:
    """Max parallel SAP decompose API calls (RouterAI)."""
    return max(1, _env_int("SAP_SAP_MAX_CONCURRENT", 4))


def use_pipeline_parallel_sap() -> bool:
    """
    Per-prompt parallel SAP decompose (overlaps with FLUX render).

    When false, uses batch SAP API call before any rendering.
    """
    raw = os.getenv("SAP_PIPELINE_PARALLEL_SAP", "1").strip().lower()
    return raw not in {"0", "false", "no"}


def use_batch_sap() -> bool:
    raw = os.getenv("SAP_BATCH_SAP", "1").strip().lower()
    return raw not in {"0", "false", "no"}


def use_eval_pipeline_overlap() -> bool:
    """Score prompt i while FLUX renders prompt i+1 (same worker thread pool)."""
    raw = os.getenv("SAP_EVAL_PIPELINE", "1").strip().lower()
    return raw not in {"0", "false", "no"}


def get_cleanup_every_n_prompts() -> int:
    """Run gc/cuda cleanup every N prompts (0 = after each prompt)."""
    return max(0, _env_int("SAP_CLEANUP_EVERY_N_PROMPTS", 3))


def get_primary_fitness_metric() -> str:
    """Metric used for evolution selection, early stopping, and cascade."""
    return os.getenv("SAP_PRIMARY_METRIC", "alignment_score").strip() or "alignment_score"


def get_cascade_stage1_threshold() -> float:
    """Min alignment_score (1–5) to pass cascade stage1."""
    raw = os.getenv("SAP_CASCADE_STAGE1_THRESHOLD", "1.75").strip()
    try:
        return float(raw)
    except ValueError:
        return 1.75


def aggregate_alignment_scores(values: list[float]) -> float:
    """
    Combine per-prompt alignment scores into overall fitness.

    Uses harmonic mean so a single weak prompt penalizes the total more than
    arithmetic mean (e.g. 5+5+2 → 3.33 vs 4.0).
    """
    if not values:
        return 0.0
    if any(v <= 0 for v in values):
        return 0.0
    return len(values) / sum(1.0 / v for v in values)


def fitness_from_metrics(metrics: dict) -> float:
    """Primary fitness value from evaluator metrics dict."""
    key = get_primary_fitness_metric()
    if key in metrics:
        return float(metrics[key])
    return float(metrics.get("alignment_score", metrics.get("combined_score", 0.0)))


def use_cascade_eval() -> bool:
    raw = os.getenv("SAP_CASCADE_EVAL", "1").strip().lower()
    return raw not in {"0", "false", "no"}


def use_gemma_judge() -> bool:
    """Run Gemini meta-judge on SYSTEM_PROMPT (disabled by default)."""
    raw = os.getenv("SAP_ENABLE_GEMMA_JUDGE", "0").strip().lower()
    return raw in {"1", "true", "yes"}


def is_single_gpu_mode() -> bool:
    raw = os.getenv("SAP_SINGLE_GPU", "").strip().lower()
    if raw in {"1", "true", "yes"}:
        return True
    gpu_ids = os.getenv("SAP_GPU_IDS", "0").strip()
    if not gpu_ids:
        return True
    return len([x for x in gpu_ids.split(",") if x.strip()]) == 1


def keep_model_loaded() -> bool:
    """When true, FLUX is never released between evals (single-GPU default)."""
    raw = os.getenv("SAP_KEEP_MODEL_LOADED", "").strip().lower()
    if raw in {"1", "true", "yes"}:
        return True
    if raw in {"0", "false", "no"}:
        return False
    if is_single_gpu_mode():
        return True
    return os.getenv("SAP_RELEASE_MODEL_AFTER_EVAL", "0").strip() != "1"


def preload_flux_on_worker_start() -> bool:
    raw = os.getenv("SAP_PRELOAD_FLUX", "").strip().lower()
    if raw in {"0", "false", "no"}:
        return False
    if raw in {"1", "true", "yes"}:
        return True
    return is_single_gpu_mode()
