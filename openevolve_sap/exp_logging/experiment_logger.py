"""Structured JSONL experiment logging."""
from __future__ import annotations

import json
import logging
import os
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional


class ExperimentLogger:
    def __init__(self, experiment_dir: Path, level: str = "INFO"):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.experiment_dir / "experiment.jsonl"

        self._logger = logging.getLogger("openevolve_sap.experiment")
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self._logger.handlers.clear()

        text_log = self.experiment_dir / "logs" / "experiment.log"
        text_log.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            text_log, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        self._logger.addHandler(handler)
        self._logger.propagate = False

    def log(
        self,
        level: str,
        source: str,
        event: str,
        data: Optional[dict[str, Any]] = None,
        gpu_id: Optional[int | str] = None,
    ) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level.upper(),
            "source": source,
            "gpu_id": gpu_id if gpu_id is not None else os.getenv("SAP_PHYSICAL_GPU_ID"),
            "event": event,
            "data": data or {},
        }
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        msg = f"{event} {json.dumps(data or {}, ensure_ascii=False)[:500]}"
        log_fn = getattr(self._logger, level.lower(), self._logger.info)
        log_fn("[%s] %s", source, msg)

    def exception(self, source: str, event: str, exc: BaseException) -> None:
        self.log(
            "ERROR",
            source,
            event,
            {"detail": str(exc), "traceback": traceback.format_exc()},
        )


_global: Optional[ExperimentLogger] = None


def get_experiment_logger() -> Optional[ExperimentLogger]:
    global _global
    if _global is not None:
        return _global
    exp = os.getenv("SAP_EXPERIMENT_DIR", "").strip()
    if not exp:
        return None
    level = os.getenv("SAP_LOG_LEVEL", "INFO")
    _global = ExperimentLogger(Path(exp), level=level)
    return _global


def log_event(
    level: str,
    source: str,
    event: str,
    data: Optional[dict[str, Any]] = None,
    gpu_id: Optional[int | str] = None,
) -> None:
    logger = get_experiment_logger()
    if logger:
        logger.log(level, source, event, data=data, gpu_id=gpu_id)
