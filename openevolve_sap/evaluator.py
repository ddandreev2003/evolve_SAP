import importlib.util
import gc
import json
import os
import sys
import tempfile
import time
import traceback
import uuid
import fcntl
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import torch
from openai import OpenAI
from openevolve.evaluation_result import EvaluationResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.gpt_eval import evaluate_image_with_gpt
from llm_interface.llm_SAP import LLM_SAP
from openevolve_sap.exp_logging.experiment_logger import log_event
from openevolve_sap.sap_eval_settings import (
    get_cleanup_every_n_prompts,
    get_image_height,
    get_image_width,
    get_num_inference_steps,
    get_physical_gpu_id,
    get_ram_limit_gb,
    get_cascade_stage1_threshold,
    get_seeds_list,
    get_vl_max_concurrent,
    get_worker_id,
    use_batch_sap,
    use_eval_pipeline_overlap,
)
from run_SAP_flux import load_model, release_model

PROMPT_SET_PATH = Path(__file__).with_name("prompt_set.json")
RESULTS_DIR = Path(
    os.getenv(
        "SAP_EVOLUTION_RESULTS_DIR",
        str(Path(__file__).with_name("evolution_eval_results")),
    )
)
TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "llm_interface" / "template"
_MODEL = None


def _generation_lock_path() -> Path:
    gpu = get_physical_gpu_id().replace(",", "_")
    return PROJECT_ROOT / "openevolve_sap" / f".generation.gpu{gpu}.lock"


def _maybe_release_model_after_eval() -> None:
    """Free GPU memory in parent process after initial OpenEvolve evaluation."""
    if os.getenv("SAP_RELEASE_MODEL_AFTER_EVAL", "0").strip() == "1":
        release_model()
        _cleanup_memory()


def _load_prompt_set():
    with open(PROMPT_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_system_prompt(program_path: str) -> str:
    spec = importlib.util.spec_from_file_location("candidate_program", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "SYSTEM_PROMPT") and isinstance(module.SYSTEM_PROMPT, str):
        return module.SYSTEM_PROMPT
    if hasattr(module, "get_system_prompt"):
        prompt = module.get_system_prompt()
        if isinstance(prompt, str):
            return prompt
    raise ValueError("Candidate program must define SYSTEM_PROMPT string or get_system_prompt().")


def _load_template_context() -> dict:
    context = {}
    if TEMPLATE_DIR.exists():
        for path in sorted(TEMPLATE_DIR.glob("*.txt")):
            try:
                context[path.name] = path.read_text(encoding="utf-8")
            except Exception:
                context[path.name] = ""
    return context


def _status_print(run_id: str, message: str):
    print(f"[openevolve_sap][{run_id}] {message}", flush=True)


def _append_jsonl(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _prompt_dir(run_dir: Path, prompt_index: int) -> Path:
    d = run_dir / f"prompt_{prompt_index:02d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_decomposition(prompt_dir: Path, original_prompt: str, sap_out: dict) -> Path:
    """Persist SAP decomposition next to images for this test prompt."""
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "original_prompt.txt").write_text(original_prompt, encoding="utf-8")
    payload = {
        "original_prompt": original_prompt,
        "explanation": sap_out.get("explanation", ""),
        "prompts_list": sap_out.get("prompts_list", []),
        "switch_prompts_steps": sap_out.get("switch_prompts_steps", []),
    }
    path = prompt_dir / "decomposition.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _save_score(prompt_dir: Path, score: dict) -> Path:
    path = prompt_dir / "score.json"
    path.write_text(json.dumps(score, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_run_manifest(
    run_dir: Path,
    run_id: str,
    program_path: str,
    prompt_records: list[dict],
    metrics: dict | None = None,
) -> Path:
    manifest = {
        "run_id": run_id,
        "program_path": program_path,
        "program_id": Path(program_path).stem,
        "timestamp": time.time(),
        "results_dir": str(run_dir),
        "prompts": prompt_records,
    }
    if metrics:
        manifest["metrics"] = metrics
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _failure_extra(exc: BaseException | None = None) -> dict:
    """Worker/GPU context and OOM hints for status.jsonl (pool crash triage)."""
    extra: dict = {
        "worker_id": get_worker_id(),
        "physical_gpu": get_physical_gpu_id(),
    }
    if exc is None:
        return extra
    msg = str(exc).lower()
    cuda_oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if (cuda_oom_type is not None and isinstance(exc, cuda_oom_type)) or "out of memory" in msg:
        extra["cuda_oom"] = True
        extra["cuda"] = _cuda_mem_stats()
    if isinstance(exc, MemoryError) or "ram limit exceeded" in msg:
        extra["ram_limit"] = True
        extra["rss_bytes"] = _current_rss_bytes()
    return extra


def _final_error(
    run_id: str,
    status_path: Path,
    reason: str,
    detail: str | None = None,
    exc: BaseException | None = None,
):
    payload = {
        "event": "final_error",
        "run_id": run_id,
        "reason": reason,
        "timestamp": time.time(),
        **_failure_extra(exc),
    }
    if detail:
        payload["detail"] = detail
        if exc is None and "out of memory" in detail.lower():
            payload["cuda_oom"] = True
            payload["cuda"] = _cuda_mem_stats()
    _append_jsonl(status_path, payload)


def _cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _maybe_cleanup_after_prompt(prompt_count: int, idx: int) -> None:
    every = get_cleanup_every_n_prompts()
    if every == 0 or (idx + 1) % every == 0 or idx >= prompt_count - 1:
        _cleanup_memory()


def _get_ram_limit_bytes() -> int:
    raw = os.getenv("SAP_RAM_LIMIT_GB", str(get_ram_limit_gb())).strip()
    try:
        gb = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid SAP_RAM_LIMIT_GB value: {raw}") from exc
    if gb <= 0:
        raise RuntimeError("SAP_RAM_LIMIT_GB must be > 0")
    return int(gb * 1024 * 1024 * 1024)


def _current_rss_bytes() -> int:
    with open("/proc/self/status", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                parts = line.split()
                return int(parts[1]) * 1024
    return 0


def _check_ram_limit(run_id: str, status_path: Path, stage: str):
    rss = _current_rss_bytes()
    limit = _get_ram_limit_bytes()
    _append_jsonl(
        status_path,
        {
            "event": "ram_check",
            "run_id": run_id,
            "stage": stage,
            "rss_bytes": rss,
            "limit_bytes": limit,
            "timestamp": time.time(),
        },
    )
    if rss > limit:
        _append_jsonl(
            status_path,
            {
                "event": "ram_limit_exceeded",
                "run_id": run_id,
                "stage": stage,
                "rss_bytes": rss,
                "limit_bytes": limit,
                "worker_id": get_worker_id(),
                "physical_gpu": get_physical_gpu_id(),
                "timestamp": time.time(),
            },
        )
        raise MemoryError(f"RAM limit exceeded at {stage}: rss={rss}, limit={limit}")


def _cuda_mem_stats() -> dict:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    device = torch.cuda.current_device()
    return {
        "cuda_available": True,
        "device": device,
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _log_mem_snapshot(run_id: str, status_path: Path, stage: str):
    _append_jsonl(
        status_path,
        {
            "event": "mem_snapshot",
            "run_id": run_id,
            "stage": stage,
            "rss_bytes": _current_rss_bytes(),
            "cuda": _cuda_mem_stats(),
            "timestamp": time.time(),
        },
    )


def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL


def _gemma_judge(
    system_prompt: str,
    sampled_outputs: list[dict],
    template_context: dict,
    key: str,
    run_id: str,
    status_path: Path,
) -> float:
    base_url = os.getenv("ROUTERAI_BASE_URL", "https://routerai.ru/api/v1")
    client = OpenAI(api_key=key, base_url=base_url)
    judge_request = {
        "system_prompt": system_prompt,
        "sampled_outputs": sampled_outputs[:2],
        "template_context": template_context,
        "instruction": (
            "Rate this SAP system prompt from 1 to 5 for helping diffusion-stage "
            "prompt decomposition. Return only a JSON object with key score."
        ),
    }
    response = client.chat.completions.create(
        model="google/gemini-3.1-pro-preview",
        messages=[{"role": "user", "content": json.dumps(judge_request, ensure_ascii=False)}],
        max_tokens=256,
    )
    choices = getattr(response, "choices", None)
    if not choices:
        _append_jsonl(
            status_path,
            {
                "event": "gemma_judge_fallback",
                "run_id": run_id,
                "reason": "empty_choices",
                "score": 1.0,
                "timestamp": time.time(),
            },
        )
        return 1.0
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if content is None:
        _append_jsonl(
            status_path,
            {
                "event": "gemma_judge_fallback",
                "run_id": run_id,
                "reason": "empty_content",
                "score": 1.0,
                "timestamp": time.time(),
            },
        )
        return 1.0
    if not isinstance(content, str):
        content = str(content)
    text = content.strip()
    if not text:
        _append_jsonl(
            status_path,
            {
                "event": "gemma_judge_fallback",
                "run_id": run_id,
                "reason": "blank_content",
                "score": 1.0,
                "timestamp": time.time(),
            },
        )
        return 1.0
    try:
        obj = json.loads(text)
        score = float(obj.get("score", 1.0))
    except Exception:
        _append_jsonl(
            status_path,
            {
                "event": "gemma_judge_fallback",
                "run_id": run_id,
                "reason": "invalid_json_content",
                "score": 1.0,
                "raw_preview": text[:200],
                "timestamp": time.time(),
            },
        )
        score = 1.0
    return max(1.0, min(5.0, score))


def _decompose_prompts(
    prompts: list[str],
    api_key: str,
    run_id: str,
    status_path: Path,
) -> dict[int, dict | None]:
    """SAP decomposition for all prompts; batch API with per-prompt fallback."""
    out: dict[int, dict | None] = {i: None for i in range(len(prompts))}
    if not prompts:
        return out

    if use_batch_sap() and len(prompts) > 1:
        _status_print(run_id, f"batch SAP decompose ({len(prompts)} prompts)")
        try:
            batch_results = LLM_SAP(list(prompts), llm="GPT", key=api_key)
            if batch_results and len(batch_results) >= len(prompts):
                ok = True
                for i, sap in enumerate(batch_results[: len(prompts)]):
                    if sap is None:
                        ok = False
                        break
                    out[i] = sap
                if ok:
                    _append_jsonl(
                        status_path,
                        {
                            "event": "sap_batch_ok",
                            "run_id": run_id,
                            "num_prompts": len(prompts),
                            "timestamp": time.time(),
                        },
                    )
                    return out
        except Exception as e:
            _append_jsonl(
                status_path,
                {
                    "event": "sap_batch_failed",
                    "run_id": run_id,
                    "error": str(e),
                    "timestamp": time.time(),
                },
            )

    for i, prompt in enumerate(prompts):
        _status_print(run_id, f"prompt {i + 1}/{len(prompts)}: decompose (fallback)")
        try:
            sap = LLM_SAP(prompt, llm="GPT", key=api_key)[0]
            out[i] = sap
        except Exception as e:
            _append_jsonl(
                status_path,
                {
                    "event": "sap_parse_exception",
                    "run_id": run_id,
                    "prompt_index": i,
                    "prompt": prompt,
                    "error": str(e),
                    "timestamp": time.time(),
                },
            )
    return out


def _render_flux_image(
    model,
    idx: int,
    prompt: str,
    sap_out: dict,
    prompt_dir: Path,
    run_id: str,
    status_path: Path,
    record: dict,
) -> list[Path]:
    """Run FLUX for one prompt; return saved image paths (one per seed) or empty list."""
    seeds = get_seeds_list()
    generators = [torch.Generator().manual_seed(seed) for seed in seeds]
    params = {
        "height": get_image_height(),
        "width": get_image_width(),
        "num_inference_steps": get_num_inference_steps(),
        "generator": generators,
        "num_images_per_prompt": len(seeds),
        "guidance_scale": 3.5,
        "sap_prompts": sap_out,
    }
    _status_print(run_id, f"prompt {idx + 1}: render start")
    _check_ram_limit(run_id, status_path, f"prompt_{idx}_before_render")
    _log_mem_snapshot(run_id, status_path, f"prompt_{idx}_before_render")
    try:
        lock_path = _generation_lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            generation_output = model(**params)
    except Exception as e:
        _append_jsonl(
            status_path,
            {
                "event": "generation_failed",
                "run_id": run_id,
                "prompt_index": idx,
                "prompt": prompt,
                "error": str(e),
                "timestamp": time.time(),
                **_failure_extra(e),
            },
        )
        return []

    generated_images = list(generation_output.images or [])
    if not generated_images:
        _append_jsonl(
            status_path,
            {
                "event": "generation_failed",
                "run_id": run_id,
                "prompt_index": idx,
                "prompt": prompt,
                "error": "generation produced no images",
                "timestamp": time.time(),
            },
        )
        return []

    image_paths: list[Path] = []
    for image_idx, (image, seed) in enumerate(zip(generated_images, seeds)):
        image_path = prompt_dir / f"image_{image_idx:02d}.png"
        try:
            image.save(image_path)
        except Exception as e:
            _append_jsonl(
                status_path,
                {
                    "event": "image_save_failed",
                    "run_id": run_id,
                    "prompt_index": idx,
                    "image_index": image_idx,
                    "seed": seed,
                    "error": str(e),
                    "timestamp": time.time(),
                },
            )
            continue
        record["images"].append(
            {"image_index": image_idx, "seed": seed, "image_path": str(image_path)}
        )
        _append_jsonl(
            status_path,
            {
                "event": "image_saved",
                "run_id": run_id,
                "prompt_index": idx,
                "image_index": image_idx,
                "seed": seed,
                "image_path": str(image_path),
                "timestamp": time.time(),
            },
        )
        image_paths.append(image_path)
    del generated_images
    _check_ram_limit(run_id, status_path, f"prompt_{idx}_after_render")
    _log_mem_snapshot(run_id, status_path, f"prompt_{idx}_after_render")
    return image_paths


def _apply_vl_score(
    idx: int,
    prompt: str,
    image_paths: list[Path],
    prompt_dir: Path,
    decomp_path: Path,
    api_key: str,
    run_id: str,
    status_path: Path,
    record: dict,
) -> float | None:
    """VL alignment judge for all seed images; updates record. Returns mean alignment or None."""
    if not image_paths:
        return None
    _status_print(run_id, f"prompt {idx + 1}: score start ({len(image_paths)} seeds)")
    _check_ram_limit(run_id, status_path, f"prompt_{idx}_before_score")
    per_seed_scores: list[dict] = []
    alignments: list[float] = []
    for image_idx, image_path in enumerate(image_paths):
        if not image_path.exists():
            _append_jsonl(
                status_path,
                {
                    "event": "image_file_missing",
                    "run_id": run_id,
                    "prompt_index": idx,
                    "image_index": image_idx,
                    "image_path": str(image_path),
                    "timestamp": time.time(),
                },
            )
            continue
        try:
            score = evaluate_image_with_gpt(str(image_path), prompt, api_key)
        except Exception as e:
            _append_jsonl(
                status_path,
                {
                    "event": "score_failed",
                    "run_id": run_id,
                    "prompt_index": idx,
                    "image_index": image_idx,
                    "prompt": prompt,
                    "error": str(e),
                    "timestamp": time.time(),
                },
            )
            continue
        alignment_value = float(score.get("alignment score", 0.0))
        per_seed_scores.append(
            {
                "image_index": image_idx,
                "image_path": str(image_path),
                "alignment score": alignment_value,
                "score": score,
            }
        )
        alignments.append(alignment_value)
        _append_jsonl(
            status_path,
            {
                "event": "score_seed_done",
                "run_id": run_id,
                "prompt_index": idx,
                "image_index": image_idx,
                "alignment_score": alignment_value,
                "timestamp": time.time(),
            },
        )
    if not alignments:
        return None
    alignment_value = sum(alignments) / len(alignments)
    aggregate_score = {
        "alignment score": alignment_value,
        "per_seed": per_seed_scores,
        "num_seeds_scored": len(alignments),
    }
    score_path = _save_score(prompt_dir, aggregate_score)
    record["score"] = aggregate_score
    record["score_path"] = str(score_path)
    record["alignment_score"] = alignment_value
    _append_jsonl(
        status_path,
        {
            "event": "score_done",
            "run_id": run_id,
            "prompt_index": idx,
            "alignment_score": alignment_value,
            "num_seeds_scored": len(alignments),
            "timestamp": time.time(),
        },
    )
    _status_print(run_id, f"prompt {idx + 1}: alignment={alignment_value:.3f} ({len(alignments)} seeds)")
    _check_ram_limit(run_id, status_path, f"prompt_{idx}_after_score")
    return alignment_value


def _collect_vl_future(
    future: Future,
    idx: int,
    prompt: str,
    prompt_dir: Path,
    decomp_path: Path,
    run_id: str,
    status_path: Path,
    record: dict,
    alignments: list[float],
    score_records: list[dict],
) -> None:
    try:
        alignment_value = future.result()
    except Exception as e:
        _append_jsonl(
            status_path,
            {
                "event": "score_failed",
                "run_id": run_id,
                "prompt_index": idx,
                "error": str(e),
                "timestamp": time.time(),
            },
        )
        return
    if alignment_value is None:
        return
    alignments.append(alignment_value)
    score_records.append(
        {
            "prompt_index": idx,
            "prompt": prompt,
            "prompt_dir": str(prompt_dir),
            "decomposition_path": str(decomp_path),
            "image_path": str(prompt_dir / "image_00.png"),
            "score_path": record.get("score_path"),
            "score": record.get("score"),
        }
    )


def _run_evaluation_core(
    program_path: str,
    *,
    visualization_only: bool = False,
    prompt_indices: list[int] | None = None,
    enable_gemma: bool = True,
) -> EvaluationResult:
    """Shared eval loop: batch SAP, pipeline VL overlap, optional subset of prompts."""
    api_key = os.getenv("ROUTERAI_API_KEY", "")
    run_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.jsonl"
    mode_label = "visualization" if visualization_only else "evaluation"
    _status_print(run_id, f"{mode_label} started")
    log_event(
        "INFO",
        get_worker_id(),
        "evaluation_started",
        {
            "run_id": run_id,
            "program_path": program_path,
            "mode": mode_label,
            "prompt_indices": prompt_indices,
        },
    )
    if not api_key:
        _final_error(run_id, status_path, "missing_api_key", "ROUTERAI_API_KEY is not set")
        return EvaluationResult(
            metrics={"alignment_score": 0.0, "gemma_score": 0.0, "combined_score": 0.0},
            artifacts={"error": "Missing ROUTERAI_API_KEY"},
        )

    try:
        system_prompt = _extract_system_prompt(program_path)
    except Exception as e:
        _final_error(run_id, status_path, "extract_system_prompt_failed", str(e))
        return EvaluationResult(
            metrics={"alignment_score": 0.0, "gemma_score": 0.0, "combined_score": 0.0},
            artifacts={"error": f"extract_system_prompt_failed: {e}", "traceback": traceback.format_exc()},
        )

    all_prompts = _load_prompt_set()
    if prompt_indices is not None:
        prompts = [all_prompts[i] for i in prompt_indices]
        index_map = {local_i: prompt_indices[local_i] for local_i in range(len(prompt_indices))}
    else:
        prompts = all_prompts
        index_map = {i: i for i in range(len(prompts))}

    template_context = _load_template_context()
    sampled_outputs: list[dict] = []
    alignments: list[float] = []
    score_records: list[dict] = []
    saved_images: list[dict] = []
    prompt_records: list[dict] = []

    try:
        model = _get_model()
    except Exception as e:
        _final_error(run_id, status_path, "model_load_failed", str(e))
        _maybe_release_model_after_eval()
        return EvaluationResult(
            metrics={"alignment_score": 0.0, "gemma_score": 0.0, "combined_score": 0.0},
            artifacts={"error": f"model_load_failed: {e}", "traceback": traceback.format_exc()},
        )

    temp_system_prompt_path = None
    try:
        _log_mem_snapshot(run_id, status_path, "after_model_load")
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
            tmp.write(system_prompt)
            temp_system_prompt_path = tmp.name
        original_path = os.getenv("SAP_SYSTEM_PROMPT_PATH")
        os.environ["SAP_SYSTEM_PROMPT_PATH"] = temp_system_prompt_path

        _append_jsonl(
            status_path,
            {
                "event": "start_loop",
                "run_id": run_id,
                "num_prompts": len(prompts),
                "batch_sap": use_batch_sap(),
                "pipeline_overlap": use_eval_pipeline_overlap(),
                "seeds": get_seeds_list(),
                "num_inference_steps": get_num_inference_steps(),
                "timestamp": time.time(),
            },
        )
        _check_ram_limit(run_id, status_path, "start_loop")

        sap_by_local = _decompose_prompts(prompts, api_key, run_id, status_path)
        pending_vl: Future | None = None
        pending_vl_ctx: dict | None = None
        vl_workers = get_vl_max_concurrent()
        pipeline = use_eval_pipeline_overlap() and not visualization_only
        gemma_future: Future | None = None
        deferred_vl: list[dict] = []

        with ThreadPoolExecutor(max_workers=max(vl_workers, 1)) as vl_pool:
            for local_i, prompt in enumerate(prompts):
                global_idx = index_map[local_i]
                sap_out = sap_by_local.get(local_i)
                if sap_out is None:
                    _append_jsonl(
                        status_path,
                        {
                            "event": "sap_parse_failed",
                            "run_id": run_id,
                            "prompt_index": global_idx,
                            "prompt": prompt,
                            "timestamp": time.time(),
                        },
                    )
                    continue

                sampled_outputs.append(sap_out)
                prompt_dir = _prompt_dir(run_dir, global_idx)
                decomp_path = _save_decomposition(prompt_dir, prompt, sap_out)
                record = {
                    "prompt_index": global_idx,
                    "original_prompt": prompt,
                    "prompt_dir": str(prompt_dir),
                    "decomposition_path": str(decomp_path),
                    "images": [],
                    "score": None,
                    "alignment_score": None,
                }
                prompt_records.append(record)
                _append_jsonl(
                    status_path,
                    {
                        "event": "decomposition_saved",
                        "run_id": run_id,
                        "prompt_index": global_idx,
                        "prompt": prompt,
                        "decomposition_path": str(decomp_path),
                        "prompts_list": sap_out.get("prompts_list", []),
                        "switch_prompts_steps": sap_out.get("switch_prompts_steps", []),
                        "timestamp": time.time(),
                    },
                )

                if pipeline and pending_vl is not None and pending_vl_ctx is not None:
                    _collect_vl_future(
                        pending_vl,
                        pending_vl_ctx["idx"],
                        pending_vl_ctx["prompt"],
                        pending_vl_ctx["prompt_dir"],
                        pending_vl_ctx["decomp_path"],
                        run_id,
                        status_path,
                        pending_vl_ctx["record"],
                        alignments,
                        score_records,
                    )
                    pending_vl = None
                    pending_vl_ctx = None

                image_paths = _render_flux_image(
                    model, global_idx, prompt, sap_out, prompt_dir, run_id, status_path, record
                )
                _maybe_cleanup_after_prompt(len(prompts), local_i)

                if not image_paths:
                    continue

                saved_images.append(
                    {
                        "prompt_index": global_idx,
                        "prompt": prompt,
                        "prompt_dir": str(prompt_dir),
                        "decomposition_path": str(decomp_path),
                        "image_path": str(image_paths[0]),
                        "image_paths": [str(p) for p in image_paths],
                        "images": list(record["images"]),
                    }
                )

                if visualization_only:
                    _status_print(run_id, f"prompt {local_i + 1}/{len(prompts)}: image saved")
                    continue

                vl_ctx = {
                    "idx": global_idx,
                    "prompt": prompt,
                    "prompt_dir": prompt_dir,
                    "decomp_path": decomp_path,
                    "record": record,
                    "image_paths": image_paths,
                }

                if pipeline:
                    pending_vl = vl_pool.submit(
                        _apply_vl_score,
                        global_idx,
                        prompt,
                        image_paths,
                        prompt_dir,
                        decomp_path,
                        api_key,
                        run_id,
                        status_path,
                        record,
                    )
                    pending_vl_ctx = vl_ctx
                else:
                    deferred_vl.append(vl_ctx)

                if (
                    enable_gemma
                    and not visualization_only
                    and local_i == len(prompts) - 1
                    and sampled_outputs
                ):
                    gemma_future = vl_pool.submit(
                        _gemma_judge,
                        system_prompt,
                        sampled_outputs,
                        template_context,
                        api_key,
                        run_id,
                        status_path,
                    )

            if pipeline and pending_vl is not None and pending_vl_ctx is not None:
                _collect_vl_future(
                    pending_vl,
                    pending_vl_ctx["idx"],
                    pending_vl_ctx["prompt"],
                    pending_vl_ctx["prompt_dir"],
                    pending_vl_ctx["decomp_path"],
                    run_id,
                    status_path,
                    pending_vl_ctx["record"],
                    alignments,
                    score_records,
                )

            if not pipeline and not visualization_only and deferred_vl:
                futures = []
                for ctx in deferred_vl:
                    futures.append(
                        (
                            ctx,
                            vl_pool.submit(
                                _apply_vl_score,
                                ctx["idx"],
                                ctx["prompt"],
                                ctx["image_paths"],
                                ctx["prompt_dir"],
                                ctx["decomp_path"],
                                api_key,
                                run_id,
                                status_path,
                                ctx["record"],
                            ),
                        )
                    )
                for ctx, fut in futures:
                    try:
                        alignment_value = fut.result()
                    except Exception as e:
                        _append_jsonl(
                            status_path,
                            {
                                "event": "score_failed",
                                "run_id": run_id,
                                "prompt_index": ctx["idx"],
                                "error": str(e),
                                "timestamp": time.time(),
                            },
                        )
                        continue
                    if alignment_value is not None:
                        alignments.append(alignment_value)
                        score_records.append(
                            {
                                "prompt_index": ctx["idx"],
                                "prompt": ctx["prompt"],
                                "prompt_dir": str(ctx["prompt_dir"]),
                                "decomposition_path": str(ctx["decomp_path"]),
                                "image_path": str(ctx["image_paths"][0]),
                                "image_paths": [str(p) for p in ctx["image_paths"]],
                                "score_path": ctx["record"].get("score_path"),
                                "score": ctx["record"].get("score"),
                            }
                        )

    except Exception as e:
        _final_error(run_id, status_path, "evaluation_failed", str(e), exc=e)
        if prompt_records:
            _write_run_manifest(run_dir, run_id, program_path, prompt_records)
        _maybe_release_model_after_eval()
        return EvaluationResult(
            metrics={"alignment_score": 0.0, "gemma_score": 0.0, "combined_score": 0.0},
            artifacts={
                "error": f"evaluation_failed: {e}",
                "traceback": traceback.format_exc(),
                "prompt_records": prompt_records,
                "manifest_path": str(run_dir / "manifest.json"),
            },
        )
    finally:
        if "original_path" in locals():
            if original_path:
                os.environ["SAP_SYSTEM_PROMPT_PATH"] = original_path
            else:
                os.environ.pop("SAP_SYSTEM_PROMPT_PATH", None)
        if temp_system_prompt_path and os.path.exists(temp_system_prompt_path):
            os.unlink(temp_system_prompt_path)
        _maybe_release_model_after_eval()

    if visualization_only:
        if not saved_images:
            _final_error(run_id, status_path, "no_images_generated", "No images were generated")
            if prompt_records:
                _write_run_manifest(run_dir, run_id, program_path, prompt_records)
            return EvaluationResult(
                metrics={"num_images": 0.0},
                artifacts={
                    "error": "No images were generated",
                    "sampled_outputs": sampled_outputs,
                    "manifest_path": str(run_dir / "manifest.json"),
                },
            )
        manifest_path = _write_run_manifest(
            run_dir,
            run_id,
            program_path,
            prompt_records,
            metrics={"num_images": len(saved_images)},
        )
        _append_jsonl(
            status_path,
            {
                "event": "visualization_complete",
                "run_id": run_id,
                "num_images": len(saved_images),
                "manifest_path": str(manifest_path),
                "timestamp": time.time(),
            },
        )
        return EvaluationResult(
            metrics={"num_images": float(len(saved_images))},
            artifacts={
                "images": saved_images,
                "sampled_outputs": sampled_outputs,
                "prompt_records": prompt_records,
                "manifest_path": str(manifest_path),
            },
        )

    if not alignments:
        _final_error(run_id, status_path, "no_valid_alignments", "No valid alignments produced")
        if prompt_records:
            _write_run_manifest(run_dir, run_id, program_path, prompt_records)
        return EvaluationResult(
            metrics={"alignment_score": 0.0, "gemma_score": 0.0, "combined_score": 0.0},
            artifacts={
                "error": "No valid alignments produced",
                "sampled_outputs": sampled_outputs,
                "prompt_records": prompt_records,
                "manifest_path": str(run_dir / "manifest.json"),
            },
        )

    alignment_score = sum(alignments) / len(alignments)
    gemma_score = 1.0
    if enable_gemma:
        _status_print(run_id, "running gemma judge")
        try:
            if gemma_future is not None:
                gemma_score = gemma_future.result()
            else:
                gemma_score = _gemma_judge(
                    system_prompt,
                    sampled_outputs,
                    template_context,
                    api_key,
                    run_id,
                    status_path,
                )
        except Exception as e:
            _final_error(run_id, status_path, "gemma_judge_failed", str(e))
            return EvaluationResult(
                metrics={"alignment_score": 0.0, "gemma_score": 0.0, "combined_score": 0.0},
                artifacts={"error": f"gemma_judge_failed: {e}", "traceback": traceback.format_exc()},
            )
    else:
        gemma_score = 5.0

    combined_score = 0.8 * (alignment_score / 5.0) + 0.2 * (gemma_score / 5.0)
    manifest_path = _write_run_manifest(
        run_dir,
        run_id,
        program_path,
        prompt_records,
        metrics={
            "alignment_score": alignment_score,
            "gemma_score": gemma_score,
            "combined_score": combined_score,
        },
    )
    _append_jsonl(
        status_path,
        {
            "event": "final_scores",
            "run_id": run_id,
            "alignment_score": alignment_score,
            "gemma_score": gemma_score,
            "combined_score": combined_score,
            "manifest_path": str(manifest_path),
            "timestamp": time.time(),
        },
    )
    _status_print(
        run_id,
        f"done alignment={alignment_score:.3f} gemma={gemma_score:.3f} combined={combined_score:.3f}",
    )
    log_event(
        "INFO",
        get_worker_id(),
        "evaluation_complete",
        {
            "run_id": run_id,
            "program_id": Path(program_path).stem,
            "alignment_score": alignment_score,
            "gemma_score": gemma_score,
            "combined_score": combined_score,
            "num_eval_prompts": len(alignments),
        },
    )
    return EvaluationResult(
        metrics={
            "alignment_score": float(alignment_score),
            "gemma_score": float(gemma_score),
            "combined_score": float(combined_score),
            "num_eval_prompts": float(len(alignments)),
        },
        artifacts={
            "alignment_values": alignments,
            "sampled_outputs": sampled_outputs,
            "prompt_records": prompt_records,
            "score_records": score_records,
            "score_records_path": str(status_path),
            "manifest_path": str(manifest_path),
            "template_context_files": sorted(template_context.keys()),
        },
    )


def find_latest_checkpoint_program(output_dir: Path | None = None) -> Path:
    """Return best_program.py from the latest OpenEvolve checkpoint, or output/best."""
    root_output = output_dir or (PROJECT_ROOT / "openevolve_sap" / "output")
    checkpoints_dir = root_output / "checkpoints"
    if checkpoints_dir.is_dir():
        checkpoint_dirs = sorted(
            (p for p in checkpoints_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint_")),
            key=lambda p: int(p.name.rsplit("_", 1)[-1]) if p.name.rsplit("_", 1)[-1].isdigit() else -1,
        )
        for checkpoint_dir in reversed(checkpoint_dirs):
            program_path = checkpoint_dir / "best_program.py"
            if program_path.is_file():
                return program_path
    best_program = root_output / "best" / "best_program.py"
    if best_program.is_file():
        return best_program
    raise FileNotFoundError(f"No checkpoint program found under {root_output}")


def evaluate(program_path: str, visualization_only: bool = False):
    return _run_evaluation_core(
        program_path,
        visualization_only=visualization_only,
        prompt_indices=None,
        enable_gemma=True,
    )


def evaluate_stage1(program_path: str):
    """Cascade stage 1: full SAP+FLUX+VL on first benchmark prompt only."""
    result = _run_evaluation_core(
        program_path,
        visualization_only=False,
        prompt_indices=[0],
        enable_gemma=False,
    )
    metrics = dict(result.metrics or {})
    metrics["eval_stage"] = 1.0
    metrics["stage1_passed"] = float(metrics.get("combined_score", 0.0) >= get_cascade_stage1_threshold())
    artifacts = dict(result.artifacts or {})
    artifacts["cascade_stage"] = "stage1"
    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def evaluate_stage2(program_path: str):
    """Cascade stage 2: full evaluation on all benchmark prompts."""
    result = _run_evaluation_core(
        program_path,
        visualization_only=False,
        prompt_indices=None,
        enable_gemma=True,
    )
    metrics = dict(result.metrics or {})
    metrics["eval_stage"] = 2.0
    artifacts = dict(result.artifacts or {})
    artifacts["cascade_stage"] = "stage2"
    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def evaluate_visualization_only(program_path: str):
    """Generate and save images only (no GPT alignment scoring or Gemma judge)."""
    return evaluate(program_path, visualization_only=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SAP visualization for a checkpoint program.")
    parser.add_argument(
        "--program",
        type=str,
        default="",
        help="Path to evolved program .py (default: latest checkpoint best_program.py).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "openevolve_sap" / "output"),
        help="OpenEvolve output directory used to resolve latest checkpoint.",
    )
    cli_args = parser.parse_args()
    program_path = (
        Path(cli_args.program)
        if cli_args.program
        else find_latest_checkpoint_program(Path(cli_args.output_dir))
    )
    print(f"Running visualization for: {program_path}")
    result = evaluate_visualization_only(str(program_path))
    print("Metrics:", result.metrics)
    if result.artifacts.get("images"):
        print("Saved images:")
        for item in result.artifacts["images"]:
            print(f"  - {item['image_path']}")
