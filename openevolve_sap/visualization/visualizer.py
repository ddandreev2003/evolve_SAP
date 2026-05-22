#!/usr/bin/env python3
"""Standalone Flask visualizer for OpenEvolve SAP checkpoints."""
from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
from pathlib import Path

from flask import Flask, jsonify, render_template

from openevolve_sap.visualization.utils import (
    aggregate_metrics,
    find_latest_checkpoint,
    load_evolution_data,
    load_gpu_metrics,
)

logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(HERE / "templates"), static_folder=str(HERE / "static"))

_checkpoint_dir: str | None = None
_experiment_dir: Path | None = None


@app.route("/")
def index():
    return render_template("index.html", checkpoint_dir=_checkpoint_dir or "")


@app.route("/api/data")
def api_data():
    if not _checkpoint_dir:
        return jsonify({"archive": [], "nodes": [], "edges": [], "checkpoint_dir": ""})
    data = load_evolution_data(_checkpoint_dir)
    return jsonify(data)


@app.route("/api/metrics")
def api_metrics():
    if not _checkpoint_dir:
        return jsonify({})
    data = load_evolution_data(_checkpoint_dir)
    metrics = aggregate_metrics(data["nodes"])
    if _experiment_dir:
        metrics["gpu_samples"] = load_gpu_metrics(_experiment_dir)
    return jsonify(metrics)


@app.route("/api/program/<program_id>")
def api_program(program_id: str):
    if not _checkpoint_dir:
        return jsonify({"error": "no checkpoint"}), 404
    data = load_evolution_data(_checkpoint_dir)
    program = next((p for p in data["nodes"] if p["id"] == program_id), None)
    if not program:
        return jsonify({"error": "not found"}), 404
    return jsonify(program)


@app.route("/program/<program_id>")
def program_page(program_id: str):
    if not _checkpoint_dir:
        return "No checkpoint loaded", 500
    data = load_evolution_data(_checkpoint_dir)
    program_data = next((p for p in data["nodes"] if p["id"] == program_id), None)
    program_data = {"code": "", "prompts": {}, **(program_data or {})}
    return render_template(
        "program_page.html",
        program_data=program_data,
        checkpoint_dir=_checkpoint_dir,
        artifacts_json=program_data.get("artifacts_json"),
    )


def run_static_export(checkpoint: str, output_dir: str) -> None:
    global _checkpoint_dir
    _checkpoint_dir = checkpoint
    data = load_evolution_data(checkpoint)
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
    parser.add_argument("--experiment-dir", default=None, help="Experiment dir for gpu_metrics.csv")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--static-output", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    path = Path(args.checkpoint).resolve()
    if path.name.startswith("checkpoint_"):
        _checkpoint_dir = str(path)
    else:
        found = find_latest_checkpoint(path)
        _checkpoint_dir = found or str(path)

    if args.experiment_dir:
        _experiment_dir = Path(args.experiment_dir)
    else:
        parent = path.parent
        if (parent / "gpu_metrics.csv").exists():
            _experiment_dir = parent

    logger.info("Checkpoint: %s", _checkpoint_dir)
    if args.static_output:
        run_static_export(_checkpoint_dir, args.static_output)
        return
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
