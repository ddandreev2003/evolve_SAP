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
