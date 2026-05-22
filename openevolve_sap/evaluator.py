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
    get_image_height,
    get_image_width,
    get_num_inference_steps,
    get_physical_gpu_id,
    get_ram_limit_gb,
    get_worker_id,
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
DEFAULT_SEED = 30498
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


def _final_error(run_id: str, status_path: Path, reason: str, detail: str | None = None):
    payload = {
        "event": "final_error",
        "run_id": run_id,
        "reason": reason,
        "timestamp": time.time(),
    }
    if detail:
        payload["detail"] = detail
    _append_jsonl(status_path, payload)


def _cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


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
        model="google/gemma-4-26b-a4b-it",
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
        {"run_id": run_id, "program_path": program_path, "mode": mode_label},
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

    prompts = _load_prompt_set()
    template_context = _load_template_context()

    sampled_outputs = []
    alignments = []
    score_records = []
    saved_images = []
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
                "timestamp": time.time(),
            },
        )
        _check_ram_limit(run_id, status_path, "start_loop")

        for idx, prompt in enumerate(prompts):
            _check_ram_limit(run_id, status_path, f"prompt_{idx}_before_decompose")
            _status_print(run_id, f"prompt {idx + 1}/{len(prompts)}: decompose")
            try:
                sap_out = LLM_SAP(prompt, llm="GPT", key=api_key)[0]
            except Exception as e:
                _append_jsonl(
                    status_path,
                    {
                        "event": "sap_parse_exception",
                        "run_id": run_id,
                        "prompt_index": idx,
                        "prompt": prompt,
                        "error": str(e),
                        "timestamp": time.time(),
                    },
                )
                continue
            if sap_out is None:
                _append_jsonl(
                    status_path,
                    {
                        "event": "sap_parse_failed",
                        "run_id": run_id,
                        "prompt_index": idx,
                        "prompt": prompt,
                        "timestamp": time.time(),
                    },
                )
                continue
            sampled_outputs.append(sap_out)
            prompt_dir = _prompt_dir(run_dir, idx)
            decomp_path = _save_decomposition(prompt_dir, prompt, sap_out)
            record = {
                "prompt_index": idx,
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
                    "prompt_index": idx,
                    "prompt": prompt,
                    "decomposition_path": str(decomp_path),
                    "prompts_list": sap_out.get("prompts_list", []),
                    "switch_prompts_steps": sap_out.get("switch_prompts_steps", []),
                    "timestamp": time.time(),
                },
            )

            generator = [torch.Generator().manual_seed(DEFAULT_SEED)]
            params = {
                "height": get_image_height(),
                "width": get_image_width(),
                "num_inference_steps": get_num_inference_steps(),
                "generator": generator,
                "num_images_per_prompt": 1,
                "guidance_scale": 3.5,
                "sap_prompts": sap_out,
            }
            _status_print(run_id, f"prompt {idx + 1}/{len(prompts)}: render start")
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
                    },
                )
                continue
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
                continue
            first_image_path = None
            for image_idx, image in enumerate(generated_images):
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
                            "prompt": prompt,
                            "image_path": str(image_path),
                            "error": str(e),
                            "timestamp": time.time(),
                        },
                    )
                    continue
                img_entry = {
                    "image_index": image_idx,
                    "image_path": str(image_path),
                }
                record["images"].append(img_entry)
                _append_jsonl(
                    status_path,
                    {
                        "event": "image_saved",
                        "run_id": run_id,
                        "prompt_index": idx,
                        "image_index": image_idx,
                        "prompt": prompt,
                        "image_path": str(image_path),
                        "decomposition_path": str(decomp_path),
                        "timestamp": time.time(),
                    },
                )
                if first_image_path is None:
                    first_image_path = image_path
            if first_image_path is None:
                continue
            saved_images.append(
                {
                    "prompt_index": idx,
                    "prompt": prompt,
                    "prompt_dir": str(prompt_dir),
                    "decomposition_path": str(decomp_path),
                    "image_path": str(first_image_path),
                    "images": list(record["images"]),
                }
            )
            del generated_images
            _cleanup_memory()
            _check_ram_limit(run_id, status_path, f"prompt_{idx}_after_render")
            _log_mem_snapshot(run_id, status_path, f"prompt_{idx}_after_render")
            if visualization_only:
                _status_print(run_id, f"prompt {idx + 1}/{len(prompts)}: image saved")
                continue
            if not first_image_path.exists():
                _append_jsonl(
                    status_path,
                    {
                        "event": "image_file_missing",
                        "run_id": run_id,
                        "prompt_index": idx,
                        "prompt": prompt,
                        "image_path": str(first_image_path),
                        "timestamp": time.time(),
                    },
                )
                continue
            _status_print(run_id, f"prompt {idx + 1}/{len(prompts)}: score start")
            _check_ram_limit(run_id, status_path, f"prompt_{idx}_before_score")
            _log_mem_snapshot(run_id, status_path, f"prompt_{idx}_before_score")
            try:
                score = evaluate_image_with_gpt(str(first_image_path), prompt, api_key)
            except Exception as e:
                _append_jsonl(
                    status_path,
                    {
                        "event": "score_failed",
                        "run_id": run_id,
                        "prompt_index": idx,
                        "prompt": prompt,
                        "image_path": str(first_image_path),
                        "error": str(e),
                        "timestamp": time.time(),
                    },
                )
                continue
            alignment_value = float(score.get("alignment score", 0.0))
            alignments.append(alignment_value)
            score_path = _save_score(prompt_dir, score)
            record["score"] = score
            record["score_path"] = str(score_path)
            record["alignment_score"] = alignment_value
            score_records.append(
                {
                    "prompt_index": idx,
                    "prompt": prompt,
                    "prompt_dir": str(prompt_dir),
                    "decomposition_path": str(decomp_path),
                    "image_path": str(first_image_path),
                    "score_path": str(score_path),
                    "score": score,
                }
            )
            _append_jsonl(
                status_path,
                {
                    "event": "score_done",
                    "run_id": run_id,
                    "prompt_index": idx,
                    "alignment_score": alignment_value,
                    "timestamp": time.time(),
                },
            )
            _status_print(run_id, f"prompt {idx + 1}/{len(prompts)}: alignment={alignment_value:.3f}")
            _cleanup_memory()
            _check_ram_limit(run_id, status_path, f"prompt_{idx}_after_score")
            _log_mem_snapshot(run_id, status_path, f"prompt_{idx}_after_score")
    except Exception as e:
        _final_error(run_id, status_path, "evaluation_failed", str(e))
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
                "images": saved_images,
                "manifest_path": str(manifest_path),
                "timestamp": time.time(),
            },
        )
        _status_print(run_id, f"visualization done images={len(saved_images)}")
        return EvaluationResult(
            metrics={"num_images": float(len(saved_images))},
            artifacts={
                "images": saved_images,
                "sampled_outputs": sampled_outputs,
                "prompt_records": prompt_records,
                "status_path": str(status_path),
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
    _status_print(run_id, "running gemma judge")
    try:
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
            "score_records_path": str(run_dir / "status.jsonl"),
            "manifest_path": str(manifest_path),
            "template_context_files": sorted(list(template_context.keys())),
        },
    )


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
