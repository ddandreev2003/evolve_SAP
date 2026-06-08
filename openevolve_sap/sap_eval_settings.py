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
    return _env_int("SAP_NUM_INFERENCE_STEPS", 30)


def get_seeds_list() -> list[int]:
    """Comma-separated seeds for FLUX generation (e.g. SAP_SEEDS_LIST=30498,30499)."""
    raw = os.getenv("SAP_SEEDS_LIST", "30498,30499").strip()
    if not raw:
        return [30498, 30499]
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


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

    Default: 75% of MemTotal (e.g. 96 GiB on 128 GiB hosts). Override via SAP_RAM_LIMIT_GB.
    """
    raw = os.getenv("SAP_RAM_LIMIT_GB", "").strip()
    if raw:
        return float(raw)
    return max(32.0, system_ram_gb() * 0.75)


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


def use_batch_sap() -> bool:
    raw = os.getenv("SAP_BATCH_SAP", "1").strip().lower()
    return raw not in {"0", "false", "no"}


def use_eval_pipeline_overlap() -> bool:
    """Score prompt i while FLUX renders prompt i+1 (same worker thread pool)."""
    raw = os.getenv("SAP_EVAL_PIPELINE", "1").strip().lower()
    return raw not in {"0", "false", "no"}


def get_cleanup_every_n_prompts() -> int:
    """Run gc/cuda cleanup every N prompts (0 = after each prompt)."""
    return max(0, _env_int("SAP_CLEANUP_EVERY_N_PROMPTS", 0))


def get_cascade_stage1_threshold() -> float:
    raw = os.getenv("SAP_CASCADE_STAGE1_THRESHOLD", "0.35").strip()
    try:
        return float(raw)
    except ValueError:
        return 0.35
