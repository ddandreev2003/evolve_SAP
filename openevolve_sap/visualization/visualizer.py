#!/usr/bin/env python3
"""Standalone Flask visualizer for OpenEvolve SAP checkpoints."""
from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, send_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openevolve_sap.visualization.utils import (
    aggregate_metrics,
    find_latest_checkpoint,
    load_eval_runs,
    load_evolution_data,
    load_gpu_metrics,
    merge_evolution_data,
    resolve_eval_image_path,
)

logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(HERE / "templates"), static_folder=str(HERE / "static"))

_checkpoint_dir: str | None = None
_experiment_dir: Path | None = None


def _get_data() -> dict:
    return merge_evolution_data(_checkpoint_dir, _experiment_dir)


@app.route("/")
def index():
    return render_template("index.html", checkpoint_dir=_checkpoint_dir or "")


@app.route("/api/data")
def api_data():
    return jsonify(_get_data())


@app.route("/api/metrics")
def api_metrics():
    data = _get_data()
    metrics = aggregate_metrics(data.get("nodes") or [])
    if _experiment_dir:
        metrics["gpu_samples"] = load_gpu_metrics(_experiment_dir)
    metrics["data_source"] = data.get("data_source", "none")
    return jsonify(metrics)


@app.route("/api/eval_runs")
def api_eval_runs():
    if not _experiment_dir:
        return jsonify({"runs": [], "message": "No --experiment-dir configured"})
    runs = load_eval_runs(_experiment_dir)
    return jsonify({"runs": runs, "count": len(runs)})


@app.route("/api/eval_image/<run_id>/<int:prompt_index>")
def api_eval_image(run_id: str, prompt_index: int):
    if not _experiment_dir:
        return jsonify({"error": "no experiment dir"}), 404
    image_path = resolve_eval_image_path(_experiment_dir, run_id, prompt_index)
    if not image_path:
        return jsonify({"error": "image not found"}), 404
    return send_file(image_path, mimetype="image/png")


@app.route("/api/program/<program_id>")
def api_program(program_id: str):
    data = _get_data()
    program = next((p for p in data.get("nodes", []) if p["id"] == program_id), None)
    if not program:
        return jsonify({"error": "not found"}), 404
    return jsonify(program)


@app.route("/program/<program_id>")
def program_page(program_id: str):
    data = _get_data()
    program_data = next((p for p in data.get("nodes", []) if p["id"] == program_id), None)
    program_data = {"code": "", "prompts": {}, **(program_data or {})}
    return render_template(
        "program_page.html",
        program_data=program_data,
        checkpoint_dir=_checkpoint_dir or "",
        artifacts_json=program_data.get("artifacts_json"),
    )


def run_static_export(checkpoint: str, output_dir: str) -> None:
    global _checkpoint_dir
    _checkpoint_dir = checkpoint
    data = _get_data()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with app.app_context():
        data_json = jsonify(data).get_data(as_text=True)
    inlined = f"<script>window.STATIC_DATA = {data_json};</script>"
    html = (HERE / "templates" / "index.html").read_text(encoding="utf-8")
    html = re.sub(r"\{\{\s*url_for\('static', filename='([^']+)'\)\s*\}\}", r"static/\1", html)
    idx = html.find('<script type="module"')
    if idx != -1:
        html = html[:idx] + inlined + "\n" + html[idx:]
    else:
        html = html.replace("</body>", inlined + "\n</body>")
    (out / "index.html").write_text(html, encoding="utf-8")
    static_dst = out / "static"
    if static_dst.exists():
        shutil.rmtree(static_dst)
    shutil.copytree(HERE / "static", static_dst)
    logger.info("Static export: %s", out)


def main() -> None:
    global _checkpoint_dir, _experiment_dir
    parser = argparse.ArgumentParser(description="SAP OpenEvolve checkpoint visualizer")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir or parent output dir")
    parser.add_argument("--experiment-dir", default=None, help="Experiment dir for live eval + GPU")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--static-output", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    path = Path(args.checkpoint).resolve()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.name.startswith("checkpoint_"):
        _checkpoint_dir = str(path)
    else:
        found = find_latest_checkpoint(path)
        _checkpoint_dir = found or str(path)

    if args.experiment_dir:
        exp = Path(args.experiment_dir)
        _experiment_dir = exp if exp.is_absolute() else PROJECT_ROOT / exp
    else:
        parent = path.parent
        if (parent / "gpu_metrics.csv").exists():
            _experiment_dir = parent
        elif (parent / "experiment.jsonl").exists():
            _experiment_dir = parent

    logger.info("Checkpoint: %s", _checkpoint_dir)
    logger.info("Experiment dir: %s", _experiment_dir)
    preview = _get_data()
    logger.info(
        "Data source: %s, nodes: %d, eval_runs: %d",
        preview.get("data_source"),
        len(preview.get("nodes") or []),
        len(preview.get("eval_runs") or []),
    )
    if preview.get("message"):
        logger.info("Message: %s", preview["message"])

    if args.static_output:
        run_static_export(_checkpoint_dir, args.static_output)
        return
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
