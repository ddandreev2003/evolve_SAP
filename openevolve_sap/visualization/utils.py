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


def parse_artifacts_json(artifacts: Any) -> dict[str, Any]:
    """Checkpoint programs store artifacts_json as a JSON string."""
    if isinstance(artifacts, dict):
        return artifacts
    if isinstance(artifacts, str) and artifacts.strip():
        try:
            parsed = json.loads(artifacts)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def extract_run_id_from_artifacts(artifacts: Any) -> str | None:
    """Parse eval run_id from checkpoint artifacts paths."""
    artifacts = parse_artifacts_json(artifacts)
    if not artifacts:
        return None
    for key in ("manifest_path", "score_records_path"):
        path = artifacts.get(key)
        if path:
            parts = Path(str(path)).parts
            if "eval_results" in parts:
                idx = parts.index("eval_results")
                if idx + 1 < len(parts):
                    return parts[idx + 1]
    records = artifacts.get("prompt_records") or []
    if records:
        prompt_dir = records[0].get("prompt_dir")
        if prompt_dir:
            parts = Path(str(prompt_dir)).parts
            if "eval_results" in parts:
                idx = parts.index("eval_results")
                if idx + 1 < len(parts):
                    return parts[idx + 1]
    return None


def eval_images_from_artifacts(artifacts: Any) -> list[dict[str, Any]]:
    """Build eval_images from OpenEvolve checkpoint artifacts_json."""
    artifacts = parse_artifacts_json(artifacts)
    if not artifacts:
        return []
    run_id = extract_run_id_from_artifacts(artifacts)
    if not run_id:
        return []
    images: list[dict[str, Any]] = []
    for rec in artifacts.get("prompt_records") or []:
        image_path = rec.get("image_path")
        if not image_path and rec.get("images"):
            image_path = rec["images"][0].get("image_path")
        if not image_path or not Path(str(image_path)).is_file():
            continue
        idx = rec.get("prompt_index", 0)
        score = rec.get("score") or {}
        images.append(
            {
                "prompt_index": idx,
                "url": f"/api/eval_image/{run_id}/{idx}",
                "original_prompt": rec.get("original_prompt") or rec.get("prompt", ""),
                "alignment_score": rec.get("alignment_score")
                or score.get("alignment score")
                or score.get("alignment_score"),
                "alignment_explanation": score.get("alignment explanation", ""),
            }
        )
    return images


def _eval_images_from_prompts(run_id: str, prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    images = []
    for p in prompts:
        idx = p.get("prompt_index", 0)
        if p.get("image_path") and not Path(str(p["image_path"])).is_file():
            continue
        images.append(
            {
                "prompt_index": idx,
                "url": f"/api/eval_image/{run_id}/{idx}",
                "original_prompt": p.get("original_prompt", ""),
                "alignment_score": p.get("alignment_score"),
                "alignment_explanation": p.get("alignment_explanation", ""),
            }
        )
    return images


def _prompts_from_run_dir(run_dir: Path, run_id: str) -> list[dict[str, Any]]:
    """Read prompts/images from disk (works before manifest.json exists)."""
    prompts: list[dict[str, Any]] = []
    if not run_dir.is_dir():
        return prompts
    for prompt_dir in sorted(run_dir.glob("prompt_*")):
        if not prompt_dir.is_dir():
            continue
        try:
            idx = int(prompt_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        image_path = prompt_dir / "image_00.png"
        if not image_path.is_file():
            continue
        original_path = prompt_dir / "original_prompt.txt"
        original = (
            original_path.read_text(encoding="utf-8").strip()
            if original_path.is_file()
            else ""
        )
        alignment_score = None
        alignment_explanation = ""
        score_path = prompt_dir / "score.json"
        if score_path.is_file():
            try:
                score = json.loads(score_path.read_text(encoding="utf-8"))
                alignment_score = score.get("alignment score", score.get("alignment_score"))
                alignment_explanation = score.get("alignment explanation", "")
            except (OSError, json.JSONDecodeError):
                pass
        prompts.append(
            {
                "prompt_index": idx,
                "original_prompt": original,
                "alignment_score": alignment_score,
                "alignment_explanation": alignment_explanation,
                "image_path": str(image_path),
            }
        )
    return prompts


def _run_entry_from_manifest(manifest: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    run_id = manifest.get("run_id") or run_dir.name
    metrics = manifest.get("metrics") or {}
    prompts_out = []
    for p in manifest.get("prompts") or []:
        images = p.get("images") or []
        image_path = images[0].get("image_path") if images else None
        score = p.get("score") or {}
        prompts_out.append(
            {
                "prompt_index": p.get("prompt_index"),
                "original_prompt": p.get("original_prompt"),
                "alignment_score": p.get("alignment_score"),
                "alignment_explanation": score.get("alignment explanation", ""),
                "image_path": image_path,
            }
        )
    if not prompts_out:
        prompts_out = _prompts_from_run_dir(run_dir, run_id)
    return {
        "run_id": run_id,
        "program_id": manifest.get("program_id"),
        "timestamp": manifest.get("timestamp"),
        "combined_score": metrics.get("combined_score"),
        "alignment_score": metrics.get("alignment_score"),
        "gemma_score": metrics.get("gemma_score"),
        "prompts": prompts_out,
        "eval_images": _eval_images_from_prompts(run_id, prompts_out),
    }


def load_eval_runs(experiment_dir: str | Path) -> list[dict[str, Any]]:
    """Load eval runs from manifests and/or prompt_* folders with image_00.png."""
    experiment_dir = Path(experiment_dir)
    eval_root = experiment_dir / "eval_results"
    if not eval_root.is_dir():
        return []

    runs_by_id: dict[str, dict[str, Any]] = {}

    for run_dir in sorted(eval_root.iterdir()):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        manifest_path = run_dir / "manifest.json"
        if manifest_path.is_file():
            try:
                with open(manifest_path, encoding="utf-8") as f:
                    manifest = json.load(f)
                runs_by_id[run_id] = _run_entry_from_manifest(manifest, run_dir)
            except (OSError, json.JSONDecodeError):
                continue
        else:
            prompts_out = _prompts_from_run_dir(run_dir, run_id)
            if prompts_out:
                runs_by_id[run_id] = {
                    "run_id": run_id,
                    "program_id": None,
                    "timestamp": run_dir.stat().st_mtime,
                    "combined_score": None,
                    "alignment_score": None,
                    "gemma_score": None,
                    "prompts": prompts_out,
                    "eval_images": _eval_images_from_prompts(run_id, prompts_out),
                }

    runs = list(runs_by_id.values())
    runs.sort(key=lambda r: r.get("timestamp") or 0, reverse=True)
    return runs


def load_live_experiment_data(experiment_dir: str | Path) -> dict[str, Any]:
    """Build graph nodes from experiment.jsonl + eval manifests (no checkpoint required)."""
    experiment_dir = Path(experiment_dir)
    jsonl_path = experiment_dir / "experiment.jsonl"
    eval_runs = load_eval_runs(experiment_dir)
    run_by_program: dict[str, dict[str, Any]] = {}
    run_by_id: dict[str, dict[str, Any]] = {r["run_id"]: r for r in eval_runs}
    for run in eval_runs:
        pid = run.get("program_id")
        if pid and pid not in run_by_program:
            run_by_program[pid] = run

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    prev_id: str | None = None
    gen_idx = 0

    if jsonl_path.is_file():
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") != "evaluation_complete":
                    continue
                payload = event.get("data") or {}
                program_id = payload.get("program_id")
                if not program_id:
                    continue
                metrics = {
                    "alignment_score": payload.get("alignment_score"),
                    "gemma_score": payload.get("gemma_score"),
                    "combined_score": payload.get("combined_score"),
                    "num_eval_prompts": payload.get("num_eval_prompts"),
                }
                run_id = payload.get("run_id")
                run = run_by_id.get(run_id) if run_id else run_by_program.get(program_id)
                if not run_id and run:
                    run_id = run["run_id"]
                eval_images = run.get("eval_images", []) if run else []
                node = {
                    "id": program_id,
                    "generation": gen_idx,
                    "island": 0,
                    "metrics": metrics,
                    "eval_images": eval_images,
                    "artifacts_json": {
                        "run_id": run_id,
                        "eval_dir": str(experiment_dir / "eval_results" / run_id)
                        if run_id
                        else None,
                    },
                }
                sanitize_program_for_visualization(node)
                nodes.append(node)
                if prev_id:
                    edges.append({"source": prev_id, "target": program_id})
                prev_id = program_id
                gen_idx += 1

    if not nodes and eval_runs:
        for idx, run in enumerate(reversed(eval_runs)):
            program_id = run.get("program_id") or run["run_id"]
            node = {
                "id": program_id,
                "generation": idx,
                "island": 0,
                "metrics": {
                    "alignment_score": run.get("alignment_score"),
                    "gemma_score": run.get("gemma_score"),
                    "combined_score": run.get("combined_score"),
                },
                "eval_images": run.get("eval_images", []),
                "artifacts_json": {"run_id": run["run_id"]},
            }
            sanitize_program_for_visualization(node)
            nodes.append(node)
            if idx > 0:
                edges.append(
                    {
                        "source": nodes[idx - 1]["id"],
                        "target": program_id,
                    }
                )

    best_id = None
    best_score = -1.0
    for n in nodes:
        cs = n.get("metrics", {}).get("combined_score")
        if cs is not None and float(cs) > best_score:
            best_score = float(cs)
            best_id = n["id"]

    return {
        "archive": [best_id] if best_id else [],
        "nodes": nodes,
        "edges": edges,
        "eval_runs": eval_runs,
        "checkpoint_dir": str(experiment_dir),
        "data_source": "live_experiment",
    }


def attach_eval_images_to_nodes(
    nodes: list[dict[str, Any]], eval_runs: list[dict[str, Any]]
) -> None:
    """Add eval_images to checkpoint nodes when experiment eval_results exist."""
    run_by_program: dict[str, dict[str, Any]] = {}
    run_by_id = {r["run_id"]: r for r in eval_runs}
    for run in eval_runs:
        pid = run.get("program_id")
        if pid and pid not in run_by_program:
            run_by_program[pid] = run
    for node in nodes:
        if node.get("eval_images"):
            continue
        artifacts_raw = node.get("artifacts_json")
        artifacts = parse_artifacts_json(artifacts_raw)
        run_id = artifacts.get("run_id")
        run = run_by_id.get(run_id) if run_id else run_by_program.get(node.get("id"))
        if run:
            node["eval_images"] = run.get("eval_images", [])
            continue
        from_artifacts = eval_images_from_artifacts(artifacts_raw)
        if from_artifacts:
            node["eval_images"] = from_artifacts


def merge_evolution_data(
    checkpoint_folder: str | Path | None,
    experiment_dir: str | Path | None,
) -> dict[str, Any]:
    """Prefer checkpoint programs; fall back to live experiment data."""
    ckpt_data: dict[str, Any] = {
        "archive": [],
        "nodes": [],
        "edges": [],
        "checkpoint_dir": str(checkpoint_folder or ""),
        "data_source": "none",
        "message": "",
    }
    if checkpoint_folder:
        ckpt_data = load_evolution_data(checkpoint_folder)
        if ckpt_data.get("nodes"):
            ckpt_data["data_source"] = "checkpoint"
            if experiment_dir:
                eval_runs = load_eval_runs(experiment_dir)
                ckpt_data["eval_runs"] = eval_runs
                attach_eval_images_to_nodes(ckpt_data["nodes"], eval_runs)
            return ckpt_data

    if experiment_dir:
        live = load_live_experiment_data(experiment_dir)
        if live.get("nodes"):
            live["message"] = (
                "Live mode: checkpoint not ready yet (saved every 50 iterations). "
                "Showing evaluations from experiment.jsonl / eval_results."
            )
            return live
        live["message"] = (
            "No evaluations yet. Wait for the first eval to finish or pass --experiment-dir."
        )
        return live

    ckpt_data["message"] = (
        "No checkpoint data and no --experiment-dir. "
        "Example: --experiment-dir openevolve_sap/experiments/experiment_YYYYMMDD_HHMMSS"
    )
    return ckpt_data


def resolve_eval_image_path(experiment_dir: Path, run_id: str, prompt_index: int) -> Path | None:
    """Safe path to eval image; returns None if invalid."""
    if not run_id or ".." in run_id or "/" in run_id or "\\" in run_id:
        return None
    if prompt_index < 0 or prompt_index > 20:
        return None
    base = (experiment_dir / "eval_results" / run_id).resolve()
    eval_root = (experiment_dir / "eval_results").resolve()
    if not str(base).startswith(str(eval_root)):
        return None
    for name in (f"prompt_{prompt_index:02d}/image_00.png", f"prompt_{prompt_index}/image_00.png"):
        candidate = base / name
        if candidate.is_file():
            return candidate
    return None


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
