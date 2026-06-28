#!/usr/bin/env python3
"""Whoops-Hard benchmark: evolved LLM SAP vs pre-mapped JSON SAP."""
from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.gpt_eval import evaluate_image_with_gpt
from llm_interface.llm_SAP import LLM_SAP
from run_SAP_flux import load_model, release_model

LOGGER = logging.getLogger("benchmark_whoops_hard")

METHODS = ("evolved_llm", "json_mapping")
DEFAULT_PROMPTS_FILE = PROJECT_ROOT / "benchmarks/original_prompts/Whoops_Hard.txt"
DEFAULT_MAPPING_JSON = PROJECT_ROOT / "benchmarks/SAP_prompts/Whoops_Hard_prompt_mapping.json"
DEFAULT_SEEDS_JSON = PROJECT_ROOT / "benchmarks/evaluated_seeds/Whoops_Hard_prompt_seed_map.json"
DEFAULT_EVOLVED_PROMPT = PROJECT_ROOT / "openevolve_sap/best_evolved_system_prompt.txt"


@dataclass
class BenchmarkConfig:
    prompts_file: Path
    mapping_json: Path
    seeds_json: Path
    evolved_prompt: Path
    out_dir: Path
    methods: list[str]
    limit: int | None
    indices: list[int] | None
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    resume: bool
    vl_workers: int
    phase: str


def slugify(text: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return slug[:max_len] if len(slug) > max_len else slug


def _normalize_prompt_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().strip('"').strip("'"))


def _build_lookup(data: dict[str, Any]) -> dict[str, str]:
    return {_normalize_prompt_key(k): k for k in data}


def lookup_in_mapping(prompt: str, data: dict[str, Any]) -> tuple[str, Any]:
    norm = _normalize_prompt_key(prompt)
    lookup = _build_lookup(data)
    if norm in lookup:
        key = lookup[norm]
        return key, data[key]
    matches = difflib.get_close_matches(norm, list(lookup.keys()), n=1, cutoff=0.92)
    if matches:
        key = lookup[matches[0]]
        LOGGER.warning("Fuzzy matched prompt %r -> %r", prompt, key)
        return key, data[key]
    raise KeyError(f"Prompt not found in mapping: {prompt!r}")


def load_prompts(path: Path) -> list[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return lines


def select_prompts(all_prompts: list[str], limit: int | None, indices: list[int] | None) -> list[str]:
    if indices is not None:
        return [all_prompts[i] for i in indices if 0 <= i < len(all_prompts)]
    if limit is not None:
        return all_prompts[:limit]
    return all_prompts


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Whoops-Hard: evolved SAP vs JSON mapping benchmark")
    parser.add_argument("--prompts-file", type=Path, default=DEFAULT_PROMPTS_FILE)
    parser.add_argument("--mapping-json", type=Path, default=DEFAULT_MAPPING_JSON)
    parser.add_argument("--seeds-json", type=Path, default=DEFAULT_SEEDS_JSON)
    parser.add_argument("--evolved-prompt", type=Path, default=DEFAULT_EVOLVED_PROMPT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--indices", type=str, default=None, help="Comma-separated 0-based indices")
    parser.add_argument("--methods", type=str, default="evolved_llm,json_mapping")
    parser.add_argument(
        "--phase",
        choices=("generate", "evaluate", "plot", "all"),
        default="all",
    )
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--vl-workers", type=int, default=int(os.getenv("SAP_VL_MAX_CONCURRENT", "3")))
    args = parser.parse_args()

    indices = None
    if args.indices:
        indices = [int(x.strip()) for x in args.indices.split(",") if x.strip()]

    out_dir = args.out_dir
    if out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = PROJECT_ROOT / "benchmarks/results" / f"whoops_hard_{ts}"
    elif not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHODS:
            raise ValueError(f"Unknown method {m!r}; expected one of {METHODS}")

    return BenchmarkConfig(
        prompts_file=args.prompts_file if args.prompts_file.is_absolute() else PROJECT_ROOT / args.prompts_file,
        mapping_json=args.mapping_json if args.mapping_json.is_absolute() else PROJECT_ROOT / args.mapping_json,
        seeds_json=args.seeds_json if args.seeds_json.is_absolute() else PROJECT_ROOT / args.seeds_json,
        evolved_prompt=args.evolved_prompt if args.evolved_prompt.is_absolute() else PROJECT_ROOT / args.evolved_prompt,
        out_dir=out_dir,
        methods=methods,
        limit=args.limit,
        indices=indices,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        resume=args.resume,
        vl_workers=args.vl_workers,
        phase=args.phase,
    )


def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "benchmark.log"
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    LOGGER.addHandler(fh)
    LOGGER.addHandler(sh)


def get_gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update(
            {
                "device_index": idx,
                "device_name": torch.cuda.get_device_name(idx),
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
            }
        )
    return info


def write_manifest(cfg: BenchmarkConfig, prompts: list[str]) -> None:
    manifest_path = cfg.out_dir / "manifest.json"
    if manifest_path.exists() and cfg.resume:
        return
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompts_file": str(cfg.prompts_file),
        "mapping_json": str(cfg.mapping_json),
        "seeds_json": str(cfg.seeds_json),
        "evolved_prompt": str(cfg.evolved_prompt),
        "methods": cfg.methods,
        "num_prompts": len(prompts),
        "prompts": prompts,
        "generation": {
            "height": cfg.height,
            "width": cfg.width,
            "num_inference_steps": cfg.num_inference_steps,
            "guidance_scale": cfg.guidance_scale,
        },
        "gpu": get_gpu_info(),
        "env": {
            "SAP_LOW_VRAM": os.getenv("SAP_LOW_VRAM", ""),
            "SAP_KEEP_MODEL_LOADED": os.getenv("SAP_KEEP_MODEL_LOADED", ""),
        },
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_flux_params(cfg: BenchmarkConfig, seeds: list[int], sap_out: dict[str, Any]) -> dict[str, Any]:
    generators = [torch.Generator().manual_seed(seed) for seed in seeds]
    return {
        "height": cfg.height,
        "width": cfg.width,
        "num_inference_steps": cfg.num_inference_steps,
        "generator": generators,
        "num_images_per_prompt": len(seeds),
        "guidance_scale": cfg.guidance_scale,
        "sap_prompts": sap_out,
    }


def load_decomposition(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    if "prompts_list" not in data or "switch_prompts_steps" not in data:
        return None
    return data


def save_decomposition(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_evolved_decomposition(prompt: str, cfg: BenchmarkConfig, api_key: str, decomp_path: Path) -> dict[str, Any]:
    cached = load_decomposition(decomp_path)
    if cached is not None and cfg.resume:
        LOGGER.info("Using cached LLM decomposition: %s", decomp_path)
        return cached

    os.environ["SAP_SYSTEM_PROMPT_PATH"] = str(cfg.evolved_prompt.resolve())
    for attempt in range(2):
        result = LLM_SAP(prompt, llm="GPT", key=api_key)
        sap_out = result[0] if result else None
        if sap_out is not None:
            save_decomposition(decomp_path, sap_out)
            return sap_out
        LOGGER.warning("LLM SAP failed (attempt %d/2) for: %s", attempt + 1, prompt[:60])
        time.sleep(2)

    fallback = {"prompts_list": [prompt], "switch_prompts_steps": [], "explanation": "LLM fallback single prompt"}
    save_decomposition(decomp_path, fallback)
    return fallback


def get_json_decomposition(prompt: str, mapping: dict[str, Any], decomp_path: Path, cfg: BenchmarkConfig) -> dict[str, Any]:
    if cfg.resume:
        cached = load_decomposition(decomp_path)
        if cached is not None:
            return cached

    _key, entry = lookup_in_mapping(prompt, mapping)
    sap_out = {
        "explanation": entry.get("explanation", ""),
        "prompts_list": entry["prompts_list"],
        "switch_prompts_steps": entry["switch_prompts_steps"],
    }
    save_decomposition(decomp_path, sap_out)
    return sap_out


def all_images_exist(prompt_dir: Path, seeds: list[int]) -> bool:
    return all((prompt_dir / f"seed_{seed}.png").exists() for seed in seeds)


def generate_images_for_prompt(
    model,
    cfg: BenchmarkConfig,
    prompt: str,
    method: str,
    seeds: list[int],
    mapping: dict[str, Any],
    api_key: str,
) -> None:
    slug = slugify(prompt)
    prompt_dir = cfg.out_dir / method / slug
    prompt_dir.mkdir(parents=True, exist_ok=True)
    decomp_path = prompt_dir / "decomposition.json"
    meta_path = prompt_dir / "meta.json"
    meta_path.write_text(
        json.dumps({"prompt": prompt, "method": method, "seeds": seeds}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if cfg.resume and all_images_exist(prompt_dir, seeds):
        LOGGER.info("[%s] skip (exists): %s", method, prompt[:50])
        return

    if method == "evolved_llm":
        sap_out = get_evolved_decomposition(prompt, cfg, api_key, decomp_path)
    else:
        sap_out = get_json_decomposition(prompt, mapping, decomp_path, cfg)

    params = build_flux_params(cfg, seeds, sap_out)
    LOGGER.info("[%s] generating %d seeds: %s", method, len(seeds), prompt[:60])
    output = model(**params)
    images = list(output.images or [])
    if len(images) != len(seeds):
        raise RuntimeError(f"Expected {len(seeds)} images, got {len(images)} for {prompt!r}")

    for seed, image in zip(seeds, images):
        out_path = prompt_dir / f"seed_{seed}.png"
        image.save(out_path)
        LOGGER.info("Saved %s", out_path)


def phase_generate(cfg: BenchmarkConfig, prompts: list[str]) -> None:
    api_key = os.getenv("ROUTERAI_API_KEY", "")
    if "evolved_llm" in cfg.methods and not api_key:
        raise ValueError("ROUTERAI_API_KEY required for evolved_llm method")

    mapping = json.loads(cfg.mapping_json.read_text(encoding="utf-8"))
    seeds_map = json.loads(cfg.seeds_json.read_text(encoding="utf-8"))

    os.environ.setdefault("SAP_LOW_VRAM", "0")
    os.environ.setdefault("SAP_KEEP_MODEL_LOADED", "1")

    LOGGER.info("Loading FLUX (SAP_LOW_VRAM=%s)", os.getenv("SAP_LOW_VRAM"))
    model = load_model()

    total = len(prompts) * len(cfg.methods)
    done = 0
    for prompt in prompts:
        _seed_key, seed_entry = lookup_in_mapping(prompt, seeds_map)
        seeds = list(seed_entry) if isinstance(seed_entry, list) else list(seed_entry.values())

        for method in cfg.methods:
            done += 1
            LOGGER.info("Progress %d/%d", done, total)
            try:
                generate_images_for_prompt(model, cfg, prompt, method, seeds, mapping, api_key)
            except Exception as exc:
                LOGGER.exception("Generation failed [%s] %s: %s", method, prompt[:40], exc)

    if os.getenv("SAP_KEEP_MODEL_LOADED", "1").strip().lower() not in {"1", "true", "yes"}:
        release_model()
    else:
        LOGGER.info("Keeping FLUX loaded (SAP_KEEP_MODEL_LOADED=1); release in evaluate phase")


def load_scores(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_scores(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_one(image_path: Path, prompt: str, seed: int, api_key: str) -> dict[str, Any]:
    score = evaluate_image_with_gpt(str(image_path), prompt, api_key)
    return {"seed": seed, "image_path": str(image_path), "score": score}


def phase_evaluate(cfg: BenchmarkConfig, prompts: list[str]) -> dict[str, Any]:
    api_key = os.getenv("ROUTERAI_API_KEY", "")
    if not api_key:
        raise ValueError("ROUTERAI_API_KEY required for evaluation")

    release_model()
    LOGGER.info("FLUX released before VL evaluation")

    tasks: list[tuple[str, str, int, Path, Path]] = []
    for prompt in prompts:
        slug = slugify(prompt)
        for method in cfg.methods:
            prompt_dir = cfg.out_dir / method / slug
            meta = json.loads((prompt_dir / "meta.json").read_text(encoding="utf-8"))
            seeds = meta["seeds"]
            scores_path = prompt_dir / "scores.json"
            existing = load_scores(scores_path) or {"prompt": prompt, "method": method, "per_seed": {}}
            per_seed: dict[str, Any] = existing.get("per_seed", {})

            for seed in seeds:
                img_path = prompt_dir / f"seed_{seed}.png"
                if not img_path.exists():
                    LOGGER.warning("Missing image: %s", img_path)
                    continue
                if cfg.resume and str(seed) in per_seed:
                    continue
                tasks.append((method, prompt, seed, img_path, scores_path))

    LOGGER.info("VL tasks to run: %d (workers=%d)", len(tasks), cfg.vl_workers)

    def _run_task(task: tuple[str, str, int, Path, Path]) -> tuple[str, str, int, dict[str, Any], Path]:
        method, prompt, seed, img_path, scores_path = task
        for attempt in range(3):
            try:
                result = evaluate_one(img_path, prompt, seed, api_key)
                return method, prompt, seed, result, scores_path
            except Exception as exc:
                wait = 2 ** attempt
                LOGGER.warning("VL failed (attempt %d): %s — %s", attempt + 1, img_path.name, exc)
                time.sleep(wait)
        return method, prompt, seed, {"seed": seed, "error": "vl_failed"}, scores_path

    scores_by_path: dict[Path, dict[str, Any]] = {}
    for prompt in prompts:
        slug = slugify(prompt)
        for method in cfg.methods:
            scores_path = cfg.out_dir / method / slug / "scores.json"
            if scores_path.exists():
                scores_by_path[scores_path] = load_scores(scores_path) or {
                    "prompt": prompt,
                    "method": method,
                    "per_seed": {},
                }
            else:
                scores_by_path[scores_path] = {"prompt": prompt, "method": method, "per_seed": {}}

    with ThreadPoolExecutor(max_workers=cfg.vl_workers) as pool:
        futures = [pool.submit(_run_task, t) for t in tasks]
        for fut in as_completed(futures):
            method, prompt, seed, result, scores_path = fut.result()
            entry = scores_by_path.setdefault(
                scores_path,
                {"prompt": prompt, "method": method, "per_seed": {}},
            )
            entry["per_seed"][str(seed)] = result
            align = None
            if "score" in result and isinstance(result["score"], dict):
                align = result["score"].get("alignment score")
            LOGGER.info("Scored [%s] seed=%s alignment=%s | %s", method, seed, align, prompt[:40])
            save_scores(scores_path, entry)

    return build_summary(cfg, prompts)


def _mean_alignment(per_seed: dict[str, Any]) -> float | None:
    values = []
    for entry in per_seed.values():
        if not isinstance(entry, dict):
            continue
        score = entry.get("score")
        if isinstance(score, dict) and "alignment score" in score:
            values.append(float(score["alignment score"]))
    if not values:
        return None
    return sum(values) / len(values)


def build_summary(cfg: BenchmarkConfig, prompts: list[str]) -> dict[str, Any]:
    per_method: dict[str, list[float]] = {m: [] for m in cfg.methods}
    per_prompt: list[dict[str, Any]] = []

    for i, prompt in enumerate(prompts):
        slug = slugify(prompt)
        row: dict[str, Any] = {"index": i, "prompt": prompt, "methods": {}}
        for method in cfg.methods:
            scores_path = cfg.out_dir / method / slug / "scores.json"
            data = load_scores(scores_path) or {"per_seed": {}}
            mean = _mean_alignment(data.get("per_seed", {}))
            row["methods"][method] = mean
            if mean is not None:
                per_method[method].append(mean)
        if len(cfg.methods) == 2:
            a, b = cfg.methods
            va, vb = row["methods"].get(a), row["methods"].get(b)
            if va is not None and vb is not None:
                row["delta_evolved_minus_json"] = va - vb if a == "evolved_llm" else vb - va
        per_prompt.append(row)

    summary: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_prompts": len(prompts),
        "methods": {},
        "per_prompt": per_prompt,
    }

    for method, values in per_method.items():
        if values:
            summary["methods"][method] = {
                "mean_alignment": sum(values) / len(values),
                "median_alignment": statistics.median(values),
                "std_alignment": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "n_prompts": len(values),
            }

    if "evolved_llm" in cfg.methods and "json_mapping" in cfg.methods:
        deltas = [r["delta_evolved_minus_json"] for r in per_prompt if "delta_evolved_minus_json" in r]
        wins = sum(1 for d in deltas if d > 0)
        ties = sum(1 for d in deltas if d == 0)
        summary["comparison"] = {
            "mean_delta_evolved_minus_json": sum(deltas) / len(deltas) if deltas else None,
            "evolved_win_rate": wins / len(deltas) if deltas else None,
            "ties": ties,
            "n_compared": len(deltas),
        }

    summary_path = cfg.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved summary: %s", summary_path)
    return summary


def phase_plot(cfg: BenchmarkConfig, summary: dict[str, Any] | None = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting: pip install matplotlib") from exc

    if summary is None:
        summary_path = cfg.out_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"No summary.json at {summary_path}; run evaluate first")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    per_prompt = summary.get("per_prompt", [])
    if not per_prompt:
        raise ValueError("summary has no per_prompt data")

    methods = cfg.methods
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [1, 2]})

    # Bar chart: mean alignment per method
    ax0 = axes[0]
    means = []
    stds = []
    labels = []
    for method in methods:
        info = summary.get("methods", {}).get(method, {})
        means.append(info.get("mean_alignment", 0))
        stds.append(info.get("std_alignment", 0))
        labels.append(method.replace("_", "\n"))

    colors = ["#4C72B0", "#DD8452"][: len(methods)]
    bars = ax0.bar(labels, means, yerr=stds, capsize=6, color=colors, edgecolor="black", linewidth=0.8)
    ax0.set_ylabel("Mean alignment score")
    ax0.set_title("Whoops-Hard: evolved SAP vs JSON mapping")
    ax0.set_ylim(0, 5.5)
    for bar, val in zip(bars, means):
        ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08, f"{val:.2f}", ha="center", fontsize=10)

    comp = summary.get("comparison", {})
    if comp:
        subtitle = (
            f"delta={comp.get('mean_delta_evolved_minus_json', 0):+.2f}  "
            f"evolved wins={comp.get('evolved_win_rate', 0):.0%}  n={comp.get('n_compared', 0)}"
        )
        ax0.text(0.5, 1.02, subtitle, transform=ax0.transAxes, ha="center", fontsize=9)

    # Paired dot plot
    ax1 = axes[1]
    indices = [r["index"] for r in per_prompt]
    for method, color in zip(methods, colors):
        ys = [r["methods"].get(method) for r in per_prompt]
        valid_x = [i for i, y in zip(indices, ys) if y is not None]
        valid_y = [y for y in ys if y is not None]
        ax1.scatter(valid_x, valid_y, label=method, color=color, alpha=0.75, s=40)

    for r in per_prompt:
        if len(methods) == 2:
            y0 = r["methods"].get(methods[0])
            y1 = r["methods"].get(methods[1])
            if y0 is not None and y1 is not None:
                ax1.plot([r["index"], r["index"]], [y0, y1], color="#888888", linewidth=0.8, alpha=0.5)

    ax1.set_xlabel("Prompt index")
    ax1.set_ylabel("Mean alignment (3 seeds)")
    ax1.set_title("Per-prompt paired comparison")
    ax1.set_ylim(0, 5.5)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    png_path = cfg.out_dir / "comparison.png"
    pdf_path = cfg.out_dir / "comparison.pdf"
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)
    LOGGER.info("Saved plot: %s and %s", png_path, pdf_path)


def main() -> None:
    cfg = parse_args()
    setup_logging(cfg.out_dir)

    all_prompts = load_prompts(cfg.prompts_file)
    prompts = select_prompts(all_prompts, cfg.limit, cfg.indices)
    LOGGER.info("Selected %d prompts", len(prompts))

    write_manifest(cfg, prompts)

    run_phase = os.getenv("BENCHMARK_PHASE", cfg.phase)
    LOGGER.info("Phase: %s | out_dir: %s", run_phase, cfg.out_dir)

    summary = None
    if run_phase in ("generate", "all"):
        phase_generate(cfg, prompts)
    if run_phase in ("evaluate", "all"):
        summary = phase_evaluate(cfg, prompts)
    if run_phase in ("plot", "all"):
        phase_plot(cfg, summary)

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
