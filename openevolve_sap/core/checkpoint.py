"""Enhanced checkpoint metadata, RNG state, manifest."""
from __future__ import annotations

import csv
import json
import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def save_rng_state(path: Path, iteration: int, seed: int) -> None:
    state: dict[str, Any] = {
        "iteration": iteration,
        "seed": seed,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if torch is not None:
        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        torch.save(state, path / "rng_state.pt")
    else:
        with open(path / "rng_state.json", "w", encoding="utf-8") as f:
            json.dump({"iteration": iteration, "seed": seed}, f, indent=2)


def append_evolution_stats(
    experiment_dir: Path,
    iteration: int,
    best_score: float,
    avg_score: float,
    num_programs: int,
) -> None:
    stats_path = experiment_dir / "evolution_stats.csv"
    write_header = not stats_path.exists()
    with open(stats_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "timestamp",
                    "iteration",
                    "best_combined_score",
                    "avg_combined_score",
                    "num_programs",
                ]
            )
        writer.writerow(
            [
                datetime.now(timezone.utc).isoformat(),
                iteration,
                best_score,
                avg_score,
                num_programs,
            ]
        )


def update_manifest(experiment_dir: Path, checkpoint_path: Path, iteration: int) -> None:
    manifest_path = experiment_dir / "checkpoints_manifest.json"
    entries: list = []
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            entries = json.load(f)
    entries.append(
        {
            "iteration": iteration,
            "path": str(checkpoint_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def enrich_checkpoint(
    checkpoint_path: Path,
    experiment_dir: Path,
    config_path: Optional[Path],
    iteration: int,
    seed: int,
    database_stats: Optional[dict[str, Any]] = None,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    experiment_dir = Path(experiment_dir)

    if config_path and config_path.exists():
        shutil.copy2(config_path, checkpoint_path / "config.yaml")

    save_rng_state(checkpoint_path, iteration, seed)

    stats = database_stats or {}
    append_evolution_stats(
        experiment_dir,
        iteration,
        float(stats.get("best_score", 0.0)),
        float(stats.get("avg_score", 0.0)),
        int(stats.get("num_programs", 0)),
    )
    update_manifest(experiment_dir, checkpoint_path, iteration)

    (checkpoint_path / "README.md").write_text(
        f"""# Checkpoint {iteration}

## Resume

```bash
source /home/ubuntu/venv/bin/activate
python scripts/run_evolution.py --checkpoint {checkpoint_path} --experiment-dir {experiment_dir}
```

## Visualize

```bash
python openevolve_sap/visualization/visualizer.py --checkpoint {checkpoint_path}
```
""",
        encoding="utf-8",
    )

    with open(checkpoint_path / "sap_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "sap_enriched": True,
                "iteration": iteration,
                "enriched_at": datetime.now(timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )


def patch_controller_checkpoints() -> None:
    """Extend OpenEvolve._save_checkpoint with SAP enrichment."""
    from openevolve.controller import OpenEvolve

    if getattr(OpenEvolve, "_sap_checkpoint_patched", False):
        return

    original = OpenEvolve._save_checkpoint

    def _save_checkpoint(self, iteration: int) -> None:
        original(self, iteration)
        checkpoint_path = Path(self.output_dir) / "checkpoints" / f"checkpoint_{iteration}"
        if not checkpoint_path.is_dir():
            return
        exp_dir = Path(os.getenv("SAP_EXPERIMENT_DIR", self.output_dir))
        cfg = os.getenv("SAP_CONFIG_PATH", "")
        best = self.database.get_best_program()
        best_score = 0.0
        if best and best.metrics:
            best_score = float(best.metrics.get("combined_score", 0.0))
        scores = [
            float(p.metrics.get("combined_score", 0.0))
            for p in self.database.programs.values()
            if p.metrics
        ]
        stats = {
            "best_score": best_score,
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "num_programs": len(self.database.programs),
        }
        enrich_checkpoint(
            checkpoint_path,
            exp_dir,
            Path(cfg) if cfg else None,
            iteration,
            getattr(self.config, "random_seed", 42),
            stats,
        )

    OpenEvolve._save_checkpoint = _save_checkpoint
    OpenEvolve._sap_checkpoint_patched = True
