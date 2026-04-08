import argparse
import gc
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmarks.gpt_eval import evaluate_image_with_gpt
from llm_interface.llm_SAP import LLM_SAP
from run_SAP_flux import load_model


LOGGER = logging.getLogger("compare_sap")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare SAP vs no-SAP generation and scoring.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to generate and evaluate.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed shared by both runs.")
    parser.add_argument("--height", type=int, default=1024, help="Image height.")
    parser.add_argument("--width", type=int, default=1024, help="Image width.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps.")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale.")
    parser.add_argument("--out_dir", type=str, default="results_compare", help="Output directory root.")
    return parser.parse_args()


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")


def build_params(args, sap_prompts):
    return {
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "generator": [torch.Generator().manual_seed(args.seed)],
        "num_images_per_prompt": 1,
        "guidance_scale": args.guidance_scale,
        "sap_prompts": sap_prompts,
    }


def generate_image(model, params, save_path: Path):
    LOGGER.info("Generating image -> %s", save_path)
    image = model(**params).images[0]
    image.save(save_path)
    LOGGER.info("Saved image -> %s", save_path)
    return image


def clear_gpu_and_collect_garbage():
    LOGGER.info("Running garbage collection.")
    gc.collect()
    if torch.cuda.is_available():
        LOGGER.info("Clearing CUDA cache.")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def setup_logging(output_dir: Path):
    launch_minute = datetime.now().strftime("%Y%m%d_%H%M")
    log_path = output_dir / f"log_{launch_minute}.log"
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)

    LOGGER.info("Logging initialized. Log file: %s", log_path)
    return log_path


def get_system_status():
    status = {
        "python_pid": os.getpid(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        device_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_idx)
        status.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(device_idx),
                "cuda_total_memory_mb": round(props.total_memory / (1024**2), 2),
                "cuda_allocated_mb": round(torch.cuda.memory_allocated(device_idx) / (1024**2), 2),
                "cuda_reserved_mb": round(torch.cuda.memory_reserved(device_idx) / (1024**2), 2),
            }
        )

    mem_total_kb = None
    mem_available_kb = None
    if os.path.exists("/proc/meminfo"):
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    mem_available_kb = int(line.split()[1])
                if mem_total_kb is not None and mem_available_kb is not None:
                    break
    if mem_total_kb is not None and mem_available_kb is not None:
        status["ram_total_mb"] = round(mem_total_kb / 1024, 2)
        status["ram_available_mb"] = round(mem_available_kb / 1024, 2)
    return status


def log_system_status(stage: str):
    status = get_system_status()
    LOGGER.info("System status @ %s: %s", stage, status)


def get_process_memory_status():
    status = {}
    if os.path.exists("/proc/self/status"):
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    status["vm_rss_kb"] = int(line.split()[1])
                elif line.startswith("VmHWM:"):
                    status["vm_hwm_kb"] = int(line.split()[1])
                elif line.startswith("VmSwap:"):
                    status["vm_swap_kb"] = int(line.split()[1])
    return status


def main():
    args = parse_args()
    api_key = os.getenv("ROUTERAI_API_KEY", "")
    if not api_key:
        raise ValueError("Missing API key. Set ROUTERAI_API_KEY environment variable.")

    output_dir = Path(args.out_dir) / slugify(args.prompt) / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(output_dir)
    LOGGER.info("Run started.")
    LOGGER.info("Arguments: %s", vars(args))
    log_system_status("startup")

    no_sap_image_path = output_dir / "no_sap.png"
    sap_image_path = output_dir / "sap.png"
    scores_json_path = output_dir / "scores.json"

    LOGGER.info("Preparing SAP decomposition prompts.")
    no_sap_prompts = {"prompts_list": [args.prompt], "switch_prompts_steps": []}
    sap_prompts = LLM_SAP(args.prompt, llm="GPT", key=api_key)[0]
    if sap_prompts is None:
        raise ValueError("LLM SAP decomposition failed. Cannot run SAP comparison.")
    LOGGER.info("SAP decomposition ready: %s", sap_prompts)

    no_sap_params = build_params(args, no_sap_prompts)
    sap_params = build_params(args, sap_prompts)
    LOGGER.info("Generation params (shared): %s", {k: v for k, v in no_sap_params.items() if k != "sap_prompts"})

    LOGGER.info("Loading model for no-SAP generation.")
    log_system_status("before_no_sap_generate")
    model = load_model()
    generate_image(model, no_sap_params, no_sap_image_path)
    LOGGER.info("no-SAP generation complete. Releasing model and memory.")
    del model
    clear_gpu_and_collect_garbage()
    log_system_status("after_no_sap_cleanup")
    print("Закончил генерацию no-SAP изображения")

    LOGGER.info("Loading model from scratch for SAP generation.")
    log_system_status("before_sap_generate")
    model = load_model()
    generate_image(model, sap_params, sap_image_path)
    LOGGER.info("SAP generation complete. Releasing model and memory.")
    del model
    print("Закончил генерацию SAP изображения")
    clear_gpu_and_collect_garbage()
    log_system_status("after_sap_cleanup")

    LOGGER.info("Starting evaluation for no-SAP image.")
    no_sap_score = evaluate_image_with_gpt(str(no_sap_image_path), args.prompt, api_key)
    LOGGER.info("no-SAP score: %s", no_sap_score)
    LOGGER.info("Starting evaluation for SAP image.")
    sap_score = evaluate_image_with_gpt(str(sap_image_path), args.prompt, api_key)
    LOGGER.info("SAP score: %s", sap_score)

    payload = {
        "prompt": args.prompt,
        "seed": args.seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "generation": {
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
        },
        "images": {
            "no_sap": str(no_sap_image_path),
            "sap": str(sap_image_path),
        },
        "scores": {
            "no_sap": no_sap_score,
            "sap": sap_score,
        },
    }

    with open(scores_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    LOGGER.info("Saved scores JSON: %s", scores_json_path)
    log_system_status("final")
    LOGGER.info("Run finished successfully.")

    print(f"Saved no-SAP image: {no_sap_image_path}")
    print(f"Saved SAP image: {sap_image_path}")
    print(f"Saved scores JSON: {scores_json_path}")
    print(f"Saved log file: {log_path}")


if __name__ == "__main__":
    main()
