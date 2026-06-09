"""Tail-friendly live logging for evolution eval pipeline."""
from __future__ import annotations

import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class LiveEvolutionLogger:
    """Structured stdout + live.log events for real-time monitoring."""

    def __init__(self, run_id: str, experiment_dir: Path | None = None):
        self.run_id = run_id
        self._short_id = run_id[:16] if len(run_id) > 16 else run_id
        self._lock = threading.Lock()
        self._render_q_depth = 0
        self._vl_inflight = 0
        self._sap_inflight = 0
        self._live_path: Path | None = None
        if experiment_dir is None:
            exp = os.getenv("SAP_EXPERIMENT_DIR", "").strip()
            experiment_dir = Path(exp) if exp else None
        if experiment_dir is not None:
            logs_dir = Path(experiment_dir) / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            self._live_path = logs_dir / "live.log"

    def set_queue_depth(self, render: int | None = None, vl: int | None = None, sap: int | None = None) -> None:
        if render is not None:
            self._render_q_depth = render
        if vl is not None:
            self._vl_inflight = vl
        if sap is not None:
            self._sap_inflight = sap

    def event(
        self,
        stage: str,
        message: str,
        *,
        prompt_index: int | None = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        parts = [ts, f"{stage:10s}", self._short_id, message]
        if prompt_index is not None:
            parts.append(f"p={prompt_index + 1}")
        qparts = []
        if self._render_q_depth:
            qparts.append(f"render_q={self._render_q_depth}")
        if self._vl_inflight:
            qparts.append(f"vl={self._vl_inflight}")
        if self._sap_inflight:
            qparts.append(f"sap={self._sap_inflight}")
        if qparts:
            parts.append(" ".join(qparts))
        if extra:
            for k, v in extra.items():
                if v is not None:
                    parts.append(f"{k}={v}")
        line = " | ".join(parts)
        with self._lock:
            print(f"[evolve] {line}", flush=True)
            if self._live_path is not None:
                with open(self._live_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")

    def info(self, message: str, **kwargs: Any) -> None:
        self.event("INFO", message, **kwargs)

    def stage(self, stage_name: str, message: str, **kwargs: Any) -> None:
        self.event(stage_name.upper(), message, **kwargs)

    def done(self, message: str, **kwargs: Any) -> None:
        self.event("DONE", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self.event("ERROR", message, **kwargs)


_loggers: dict[str, LiveEvolutionLogger] = {}
_loggers_lock = threading.Lock()


def get_live_logger(run_id: str) -> LiveEvolutionLogger:
    with _loggers_lock:
        if run_id not in _loggers:
            _loggers[run_id] = LiveEvolutionLogger(run_id)
        return _loggers[run_id]


def clear_live_logger(run_id: str) -> None:
    with _loggers_lock:
        _loggers.pop(run_id, None)
