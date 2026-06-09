"""Eval result cache keyed by SYSTEM_PROMPT hash and eval configuration."""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Optional

from openevolve.evaluation_result import EvaluationResult

from benchmarks.gpt_eval import get_judge_version

_LOCK = threading.Lock()


def use_eval_cache() -> bool:
    raw = os.getenv("SAP_EVAL_CACHE", "1").strip().lower()
    return raw not in {"0", "false", "no"}


def _cache_path() -> Path:
    experiment_dir = os.getenv("SAP_EXPERIMENT_DIR", "").strip()
    if experiment_dir:
        return Path(experiment_dir) / "eval_cache.json"
    return Path(__file__).resolve().parent / "eval_cache.json"


def _prompt_set_hash() -> str:
    prompt_set = Path(__file__).resolve().parent / "prompt_set.json"
    if not prompt_set.is_file():
        return "no_prompt_set"
    return hashlib.sha256(prompt_set.read_bytes()).hexdigest()[:16]


def make_cache_key(
    system_prompt: str,
    *,
    eval_profile: str,
    prompt_indices: Optional[list[int]],
    enable_gemma: bool,
    num_inference_steps: int,
    seeds: list[int],
    image_height: int,
    image_width: int,
) -> str:
    indices_key = "all" if prompt_indices is None else ",".join(str(i) for i in prompt_indices)
    payload = "|".join(
        [
            hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
            _prompt_set_hash(),
            eval_profile,
            indices_key,
            "gemma" if enable_gemma else "no_gemma",
            str(num_inference_steps),
            ",".join(str(s) for s in seeds),
            f"{image_height}x{image_width}",
            get_judge_version(),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_store() -> dict[str, Any]:
    path = _cache_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_store(store: dict[str, Any]) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


def lookup(
    system_prompt: str,
    *,
    eval_profile: str,
    prompt_indices: Optional[list[int]],
    enable_gemma: bool,
    num_inference_steps: int,
    seeds: list[int],
    image_height: int,
    image_width: int,
) -> Optional[EvaluationResult]:
    if not use_eval_cache():
        return None
    key = make_cache_key(
        system_prompt,
        eval_profile=eval_profile,
        prompt_indices=prompt_indices,
        enable_gemma=enable_gemma,
        num_inference_steps=num_inference_steps,
        seeds=seeds,
        image_height=image_height,
        image_width=image_width,
    )
    with _LOCK:
        entry = _load_store().get(key)
    if not entry:
        return None
    metrics = entry.get("metrics", {})
    artifacts = dict(entry.get("artifacts", {}))
    artifacts["cache_hit"] = True
    artifacts["cache_key"] = key
    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def store(
    system_prompt: str,
    result: EvaluationResult,
    *,
    eval_profile: str,
    prompt_indices: Optional[list[int]],
    enable_gemma: bool,
    num_inference_steps: int,
    seeds: list[int],
    image_height: int,
    image_width: int,
) -> None:
    if not use_eval_cache():
        return
    key = make_cache_key(
        system_prompt,
        eval_profile=eval_profile,
        prompt_indices=prompt_indices,
        enable_gemma=enable_gemma,
        num_inference_steps=num_inference_steps,
        seeds=seeds,
        image_height=image_height,
        image_width=image_width,
    )
    entry = {
        "metrics": dict(result.metrics or {}),
        "artifacts": {
            k: v
            for k, v in (result.artifacts or {}).items()
            if k not in {"cache_hit", "cache_key"}
        },
    }
    with _LOCK:
        store_data = _load_store()
        store_data[key] = entry
        _save_store(store_data)
