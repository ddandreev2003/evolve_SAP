"""Utilities for loading OpenEvolve checkpoint data."""
from __future__ import annotations

import csv
import glob
import json
import math
import os
from numbers import Number
from pathlib import Path
from typing import Any, Optional


def find_latest_checkpoint(base_folder: str | Path) -> Optional[str]:
    base = str(base_folder)
    if os.path.basename(base).startswith("checkpoint_"):
        return base
    checkpoint_folders = glob.glob("**/checkpoint_*", root_dir=base, recursive=True)
    if not checkpoint_folders:
        return None
    paths = [os.path.join(base, folder) for folder in checkpoint_folders]
    paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return paths[0]


def check_json_float(v: Optional[float]) -> bool:
    return isinstance(v, Number) and not (math.isinf(v) or math.isnan(v))


def sanitize_program_for_visualization(program: dict[str, Any]) -> None:
    for k, v in list(program.get("metrics", {}).items()):
        if not check_json_float(v):
            program["metrics"][k] = None
    meta = program.get("metadata") or {}
    if "parent_metrics" in meta:
        for k, v in list(meta["parent_metrics"].items()):
            if not check_json_float(v):
                meta["parent_metrics"][k] = None


def load_evolution_data(checkpoint_folder: str | Path) -> dict[str, Any]:
    checkpoint_folder = str(checkpoint_folder)
    meta_path = os.path.join(checkpoint_folder, "metadata.json")
    programs_dir = os.path.join(checkpoint_folder, "programs")
    if not os.path.exists(meta_path) or not os.path.exists(programs_dir):
        return {"archive": [], "nodes": [], "edges": [], "checkpoint_dir": checkpoint_folder}
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    nodes = []
    id_to_program = {}
    pids: set[str] = set()
    for island_idx, id_list in enumerate(meta.get("islands", [])):
        for pid in id_list:
            prog_path = os.path.join(programs_dir, f"{pid}.json")
            if pid in pids:
                base_pid = pid.rsplit("-copy", 1)[0] if "-copy" in pid else pid
                copy_num = 1
                while f"{base_pid}-copy{copy_num}" in pids:
                    copy_num += 1
                pid = f"{base_pid}-copy{copy_num}"
            pids.add(pid)
            if os.path.exists(prog_path):
                with open(prog_path, encoding="utf-8") as pf:
                    prog = json.load(pf)
                sanitize_program_for_visualization(prog)
                prog["id"] = pid
                prog["island"] = island_idx
                nodes.append(prog)
                id_to_program[pid] = prog

    edges = []
    for prog in nodes:
        parent_id = prog.get("parent_id")
        if parent_id and parent_id in id_to_program:
            edges.append({"source": parent_id, "target": prog["id"]})

    return {
        "archive": meta.get("archive", []),
        "nodes": nodes,
        "edges": edges,
        "checkpoint_dir": checkpoint_folder,
    }


def load_gpu_metrics(experiment_dir: Path) -> list[dict[str, Any]]:
    csv_path = experiment_dir / "gpu_metrics.csv"
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def aggregate_metrics(nodes: list[dict]) -> dict[str, Any]:
    by_gen: dict[int, list[float]] = {}
    for node in nodes:
        gen = node.get("generation", 0)
        score = node.get("metrics", {}).get("combined_score")
        if score is None:
            continue
        by_gen.setdefault(gen, []).append(float(score))
    generations = sorted(by_gen.keys())
    return {
        "generations": generations,
        "best_scores": [max(by_gen[g]) for g in generations],
        "avg_scores": [sum(by_gen[g]) / len(by_gen[g]) for g in generations],
    }
