"""Background GPU metrics collector (pynvml with nvidia-smi fallback)."""
from __future__ import annotations

import csv
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class GPUMonitor:
    def __init__(
        self,
        experiment_dir: Path,
        gpu_ids: list[int],
        interval_sec: float = 5.0,
    ):
        self.experiment_dir = Path(experiment_dir)
        self.gpu_ids = gpu_ids
        self.interval_sec = interval_sec
        self.csv_path = self.experiment_dir / "gpu_metrics.csv"
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._use_pynvml = False
        try:
            try:
                import nvidia_ml_py  # noqa: F401
            except ImportError:
                import pynvml  # noqa: F401
            self._use_pynvml = True
        except ImportError:
            self._use_pynvml = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._write_header()
        self._thread = threading.Thread(target=self._run, name="gpu_monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval_sec + 2)

    def _write_header(self) -> None:
        if self.csv_path.exists():
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "worker_id",
                    "gpu_id",
                    "util_percent",
                    "memory_used_mb",
                    "memory_total_mb",
                    "temp_c",
                    "power_w",
                ]
            )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            rows = self._sample()
            if not rows:
                continue
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(rows)

    def _sample(self) -> list[list]:
        ts = datetime.now(timezone.utc).isoformat()
        worker_id = os.getenv("SAP_WORKER_ID", "scheduler")
        if self._use_pynvml:
            return self._sample_pynvml(ts, worker_id)
        return self._sample_nvidia_smi(ts, worker_id)

    def _sample_pynvml(self, ts: str, worker_id: str) -> list[list]:
        import pynvml

        pynvml.nvmlInit()
        rows = []
        try:
            for gpu_id in self.gpu_ids:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except Exception:
                    temp = ""
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except Exception:
                    power = ""
                rows.append(
                    [
                        ts,
                        worker_id,
                        gpu_id,
                        util.gpu,
                        round(mem.used / (1024 * 1024), 1),
                        round(mem.total / (1024 * 1024), 1),
                        temp,
                        power,
                    ]
                )
        finally:
            pynvml.nvmlShutdown()
        return rows

    def _sample_nvidia_smi(self, ts: str, worker_id: str) -> list[list]:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=10,
            )
        except Exception:
            return []
        rows = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            idx = int(parts[0])
            if idx not in self.gpu_ids:
                continue
            rows.append(
                [
                    ts,
                    worker_id,
                    idx,
                    parts[1],
                    parts[2],
                    parts[3],
                    parts[4],
                    parts[5],
                ]
            )
        return rows
