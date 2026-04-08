import ast
import importlib.util
import json
import os
import tempfile
import traceback
from pathlib import Path

from openai import OpenAI
from openevolve.evaluation_result import EvaluationResult

from benchmarks.gpt_eval import evaluate_image_with_gpt
from llm_interface.llm_SAP import LLM_SAP
from run_SAP_flux import load_model

PROMPT_SET_PATH = Path(__file__).with_name("prompt_set.json")
RESULTS_DIR = Path(__file__).with_name("evolution_eval_results")
DEFAULT_SEED = 30498

_MODEL = None


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


def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL


def _gemma_judge(system_prompt: str, sampled_outputs: list[dict], key: str) -> float:
    base_url = os.getenv("ROUTERAI_BASE_URL", "https://routerai.ru/api/v1")
    client = OpenAI(api_key=key, base_url=base_url)
    judge_request = {
        "system_prompt": system_prompt,
        "sampled_outputs": sampled_outputs[:2],
        "instruction": (
            "Rate this SAP system prompt from 1 to 5 for helping diffusion-stage "
            "prompt decomposition. Return only a JSON object with key score."
        ),
    }
    response = client.chat.completions.create(
        model="google/gemma-4-31b-it",
        messages=[{"role": "user", "content": json.dumps(judge_request, ensure_ascii=False)}],
        max_tokens=256,
    )
    text = response.choices[0].message.content.strip()
    try:
        obj = json.loads(text)
        score = float(obj.get("score", 1.0))
    except Exception:
        score = 1.0
    return max(1.0, min(5.0, score))


def evaluate(program_path: str):
    api_key = os.getenv("ROUTERAI_API_KEY", "")
    if not api_key:
        return EvaluationResult(
            metrics={"alignment_score": 0.0, "gemma_score": 0.0, "combined_score": 0.0},
            artifacts={"error": "Missing ROUTERAI_API_KEY"},
        )

    try:
        system_prompt = _extract_system_prompt(program_path)
    except Exception as e:
        return EvaluationResult(
            metrics={"alignment_score": 0.0, "gemma_score": 0.0, "combined_score": 0.0},
            artifacts={"error": f"extract_system_prompt_failed: {e}", "traceback": traceback.format_exc()},
        )

    prompts = _load_prompt_set()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sampled_outputs = []
    alignments = []
    model = _get_model()

    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
            tmp.write(system_prompt)
            temp_system_prompt_path = tmp.name

        original_path = os.getenv("SAP_SYSTEM_PROMPT_PATH")
        os.environ["SAP_SYSTEM_PROMPT_PATH"] = temp_system_prompt_path

        for idx, prompt in enumerate(prompts):
            sap_out = LLM_SAP(prompt, llm="GPT", key=api_key)[0]
            if sap_out is None:
                continue
            sampled_outputs.append(sap_out)

            generator = [__import__("torch").Generator().manual_seed(DEFAULT_SEED)]
            params = {
                "height": 1024,
                "width": 1024,
                "num_inference_steps": 50,
                "generator": generator,
                "num_images_per_prompt": 1,
                "guidance_scale": 3.5,
                "sap_prompts": sap_out,
            }
            image = model(**params).images[0]
            image_path = RESULTS_DIR / f"candidate_{Path(program_path).stem}_{idx}.png"
            image.save(image_path)
            score = evaluate_image_with_gpt(str(image_path), prompt, api_key)
            alignments.append(float(score.get("alignment score", 0.0)))

        if original_path:
            os.environ["SAP_SYSTEM_PROMPT_PATH"] = original_path
        else:
            os.environ.pop("SAP_SYSTEM_PROMPT_PATH", None)
        os.unlink(temp_system_prompt_path)
    except Exception as e:
        return EvaluationResult(
            metrics={"alignment_score": 0.0, "gemma_score": 0.0, "combined_score": 0.0},
            artifacts={"error": f"evaluation_failed: {e}", "traceback": traceback.format_exc()},
        )

    if not alignments:
        return EvaluationResult(
            metrics={"alignment_score": 0.0, "gemma_score": 0.0, "combined_score": 0.0},
            artifacts={"error": "No valid alignments produced", "sampled_outputs": sampled_outputs},
        )

    alignment_score = sum(alignments) / len(alignments)
    gemma_score = _gemma_judge(system_prompt, sampled_outputs, api_key)
    combined_score = 0.8 * (alignment_score / 5.0) + 0.2 * (gemma_score / 5.0)

    return EvaluationResult(
        metrics={
            "alignment_score": float(alignment_score),
            "gemma_score": float(gemma_score),
            "combined_score": float(combined_score),
            "num_eval_prompts": float(len(alignments)),
        },
        artifacts={
            "alignment_values": alignments,
            "sampled_outputs": sampled_outputs[:2],
        },
    )
