import os
import json
import torch
import argparse
from pathlib import Path
from SAP_pipeline_flux import SapFlux
from llm_interface.llm_SAP import LLM_SAP
from benchmarks.gpt_eval import evaluate_image_with_gpt
BASE_FOLDER = os.getcwd()
DEFAULT_FLUX_MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"
DEFAULT_FLUX_CACHE_DIR = Path.home() / ".cache" / "sap_flux_models"
_LOADED_MODEL = None
_LOADED_FROM = None


def str2bool(v):
    if isinstance(v, bool):
        return v
    value = str(v).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")

def parse_input_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--height', type=int, default=1024, help="define the generated image height")
    parser.add_argument('--width', type=int, default=1024, help="define the generated image width")
    parser.add_argument('--seeds_list', nargs='+', type=int, default=[30498], help="define the list of seeds for the prompt generated images")
    parser.add_argument('--prompt', type=str, default="A bear is performing a handstand in the park")
    parser.add_argument('--llm', type=str, default="GPT", help="define the llm to be used, support GPT (RouterAI Qwen) and Zephyr")
    parser.add_argument('--use_sap', type=str2bool, default=True, help="use SAP prompt decomposition (true/false)")
    parser.add_argument('--score', type=str2bool, default=False, help="evaluate generated images with gpt_eval (true/false)")
    parser.add_argument('--sap_system_prompt_path', type=str, default="", help="optional path to evolved SAP system prompt template")
    args = parser.parse_args()
    return args

def _resolve_local_model_path() -> Path:
    local_model_path = os.getenv("SAP_FLUX_MODEL_PATH", "").strip()
    if not local_model_path:
        raise RuntimeError(
            "SAP_FLUX_MODEL_PATH is required in strict local mode. "
            "Set it to a local FLUX model directory."
        )
    local_path = Path(local_model_path).expanduser().resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"SAP_FLUX_MODEL_PATH does not exist: {local_path}")
    return local_path


def _get_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required in strict mode, but no GPU is available.")
    local_idx = int(os.getenv("SAP_CUDA_DEVICE", "0").strip() or "0")
    device = torch.device(f"cuda:{local_idx}")
    torch.cuda.set_device(device)
    return device


def release_model() -> None:
    """Unload FLUX and free GPU memory (needed before spawning worker processes)."""
    global _LOADED_MODEL, _LOADED_FROM
    import gc

    model = _LOADED_MODEL
    _LOADED_MODEL = None
    _LOADED_FROM = None
    if model is not None:
        try:
            if hasattr(model, "to"):
                model.to("cpu")
        except Exception:
            pass
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_model():
    global _LOADED_MODEL, _LOADED_FROM
    if _LOADED_MODEL is not None:
        return _LOADED_MODEL

    local_path = _resolve_local_model_path()
    device = _get_cuda_device()
    local_idx = int(os.getenv("SAP_CUDA_DEVICE", "0").strip() or "0")
    physical = os.getenv("SAP_PHYSICAL_GPU_ID", os.getenv("CUDA_VISIBLE_DEVICES", "?"))
    low_vram = os.getenv("SAP_LOW_VRAM", "1").strip().lower() not in {"0", "false", "no"}
    print(
        f"[SAP] Loading FLUX from {local_path} -> device {device} "
        f"(physical GPU {physical}, low_vram={low_vram})"
    )
    model = SapFlux.from_pretrained(
        str(local_path),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    if low_vram and hasattr(model, "enable_model_cpu_offload"):
        model.enable_model_cpu_offload(gpu_id=local_idx)
        print(f"[SAP] FLUX using CPU offload on GPU {local_idx}")
    else:
        model.to(device)
        print(f"[SAP] FLUX model ready on {device}")

    _LOADED_MODEL = model
    _LOADED_FROM = str(local_path)
    return _LOADED_MODEL

def save_results(images, prompt, seeds_list):
    prompt_model_path = os.path.join(BASE_FOLDER, "results", prompt)
    Path(prompt_model_path).mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for i, seed in enumerate(seeds_list):
        image_path = os.path.join(prompt_model_path, f"Seed{seed}.png")
        images[i].save(image_path)
        saved_paths.append(image_path)
    return saved_paths

def generate_models_params(args, SAP_prompts):
    generators_lst = []
    for seed in args.seeds_list:
        generator = torch.Generator()
        generator.manual_seed(seed)
        generators_lst.append(generator)
    params = {"height": args.height, 
              "width": args.width,
              "num_inference_steps": 50,
              "generator": generators_lst,
              "num_images_per_prompt": len(generators_lst),
              "guidance_scale": 3.5, 
              "sap_prompts": SAP_prompts}
    return params

def run(args):
    api_key = os.getenv("ROUTERAI_API_KEY", "")
    if args.sap_system_prompt_path:
        os.environ["SAP_SYSTEM_PROMPT_PATH"] = args.sap_system_prompt_path
    if args.use_sap:
        # generate prompt decomposition
        SAP_prompts = LLM_SAP(args.prompt, llm=args.llm, key=api_key)[0] # using [0] because of a single prompt decomposition
    else:
        SAP_prompts = {"prompts_list": [args.prompt], "switch_prompts_steps": []}
    params = generate_models_params(args, SAP_prompts)
    # Load model
    model = load_model()
    # Run model
    images = model(**params).images
    # Save results
    saved_paths = save_results(images, args.prompt, args.seeds_list)

    if args.score:
        scores_by_seed = {}
        for seed, image_path in zip(args.seeds_list, saved_paths):
            score_dict = evaluate_image_with_gpt(image_path, args.prompt, api_key)
            scores_by_seed[str(seed)] = score_dict
        output_dir = os.path.join(BASE_FOLDER, "results", args.prompt)
        scores_path = os.path.join(output_dir, "scores.json")
        with open(scores_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "prompt": args.prompt,
                    "use_sap": args.use_sap,
                    "scores": scores_by_seed,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved scores to: {scores_path}")

def main():
    args = parse_input_arguments()
    # pass update args with defualts
    run(args)
    
if __name__ == "__main__":
    main()