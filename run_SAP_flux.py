import os
import json
import torch
import argparse
from pathlib import Path
from SAP_pipeline_flux import SapFlux
from llm_interface.llm_SAP import LLM_SAP
from benchmarks.gpt_eval import evaluate_image_with_gpt
BASE_FOLDER = os.getcwd()


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

def load_model():
    model = SapFlux.from_pretrained("black-forest-labs/FLUX.2-klein-base-4B", torch_dtype=torch.bfloat16)
    model.enable_model_cpu_offload()
    return model

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