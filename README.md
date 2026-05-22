# Image Generation from Contextually-Contradictory Prompts

> **Saar Huberman, Or Patashnik, Omer Dahary, Ron Mokady, Daniel Cohen-Or**
> 
> Text-to-image diffusion models excel at generating high-quality, diverse images from natural language prompts. However, they often fail to produce semantically accurate results when the prompt contains concept combinations that contradict their learned priors. We define this failure mode as contextual contradiction, where one concept implicitly negates another due to entangled associations learned during training. To address this, we propose a stage-aware prompt decomposition framework that guides the denoising process
using a sequence of proxy prompts. Each proxy prompt is constructed to match the semantic content expected to emerge at a specific
stage of denoising, while ensuring contextual coherence. To construct these proxy prompts, we leverage a large language model (LLM) to analyze the target prompt, identify contradictions, and generate alternative expressions that preserve the original intent while resolving contextual conflicts. By aligning prompt information with the denoising progression, our method enables fine-grained semantic control and accurate image generation in the presence of contextual contradictions. Experiments across a variety of challenging prompts show substantial improvements in alignment to the textual prompt.

<a href="https://tdpc2025.github.io/SAP/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a> 
<a href="https://arxiv.org/abs/2506.01929"><img src="https://img.shields.io/badge/arXiv-SAP-b31b1b.svg" height=20.5></a>
<!-- [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/omer11a/bounded-attention) -->

<p align="center">
<img src="images/teaser.jpg" width="800px"/>
</p>

## Description  
Official implementation of our "Image Generation from Contextually-Contradictory Prompts" paper.

This repository also includes **OpenEvolve integration** (`openevolve_sap/`) for evolving the SAP LLM system prompt against image-alignment metrics.

---

## Project layout

```
evolve_SAP/
├── run_SAP_flux.py              # Main SAP image generation (FLUX + stage-aware prompts)
├── llm_interface/
│   ├── llm_SAP.py               # Prompt decomposition (RouterAI Qwen or local Zephyr)
│   └── template/                # System/user templates for decomposition
├── benchmarks/
│   ├── gpt_eval.py              # VL alignment scoring (RouterAI)
│   ├── original_prompts/        # Whoops!, Whoops-Hard, ContraBench
│   ├── SAP_prompts/
│   └── evaluated_seeds/
├── scripts/run_evolution.py     # Multi-GPU entry point
├── openevolve_sap/
│   ├── configs/multi_gpu.yaml   # 4 workers, 4 islands, 30-step eval
│   ├── core/                    # scheduler, gpu_worker, checkpoint
│   ├── exp_logging/             # experiment.jsonl, gpu_metrics.csv
│   ├── visualization/         # Flask checkpoint visualizer
│   ├── initial_program.py
│   ├── evaluator.py
│   ├── config.yaml
│   ├── run_openevolve_sap.py
│   ├── prompt_set.json
│   ├── output/
│   └── experiments/
├── requirements.txt
└── README.md
```

### Pipeline overview

1. **SAP decomposition** — `LLM_SAP` calls RouterAI (`qwen/qwen3.5-35b-a3b`) with a system prompt and returns `prompts_list` + `switch_prompts_steps`.
2. **Image generation** — `run_SAP_flux.py` runs local **FLUX.2-klein-base-4B** with stage switching during denoising.
3. **Scoring** — `benchmarks/gpt_eval.py` scores alignment via RouterAI (`qwen/qwen3-vl-235b-a22b-thinking`).
4. **Evolution (optional)** — `openevolve_sap` mutates the system prompt (OpenEvolve + `google/gemini-3.1-pro-preview`, meta prompt in `openevolve_sap/prompts/evolution_system_message.md`), re-runs steps 1–3, and optimizes `combined_score` (80% alignment + 20% Gemini judge).

All cloud LLM calls use the **OpenAI-compatible RouterAI API** (`ROUTERAI_API_KEY`, default base `https://routerai.ru/api/v1`). Set `llm=Zephyr` in code to use a local Hugging Face model instead (no RouterAI for decomposition).

---

## Project specifics

### What this repo adds beyond the paper

| Component | Purpose |
|-----------|---------|
| `run_SAP_flux.py` + `SAP_pipeline_flux.py` | SAP on **FLUX.2-klein-base-4B** (local only, no Hub download in strict mode) |
| `openevolve_sap/` | Evolve the **SAP system prompt** (`SYSTEM_PROMPT` in Python) via OpenEvolve |
| `scripts/run_evolution.py` | Multi-GPU entry (spawn-safe, does not re-import scheduler in workers) |
| `openevolve_sap/visualization/` | Flask UI: checkpoint tree + **live** `eval_results` before iter 50 |
| `benchmarks/gpt_eval.py` | VL alignment judge (used in paper eval and evolution) |

### SAP decomposition contract

The LLM must return parseable text; `get_params_dict_SAP()` in `llm_interface/llm_SAP.py` extracts:

```python
{
  "explanation": "<why this decomposition>",
  "prompts_list": ["<stage0>", "<stage1>", ...],      # 1–3 prompts
  "switch_prompts_steps": [<step>, ...],              # len == len(prompts_list) - 1
}
```

Rules enforced in `SAP_pipeline_flux.map_SAP_dict()` / `verify_SAP_prompts()`:

- `len(switch_prompts_steps) == len(prompts_list) - 1`
- Switch steps are **sorted** and within `[0, num_inference_steps)`
- At denoising step `i`, active prompt index increments when `i >= switch_prompts_steps[k]`

### Denoising stage switching (FLUX)

`SapFlux.__call__()` (`SAP_pipeline_flux.py`) extends `Flux2KleinPipeline`:

1. `map_SAP_dict(sap_prompts, num_inference_steps)` → `prompts_list` + per-step index map
2. Each step encodes the **current** sub-prompt from `prompts_list`
3. Manual runs default to **50** steps (`run_SAP_flux.generate_models_params`); evolution eval uses **30** steps and **512×512** via env (see below)

### Evolution fitness

```
combined_score = 0.8 × (alignment_score / 5) + 0.2 × (gemma_score / 5)
```

- **alignment_score** — mean of per-prompt VL scores from `evaluate_image_with_gpt()` (`benchmarks/gpt_eval.py`)
- **gemma_score** — `google/gemini-3.1-pro-preview` rates the candidate `SYSTEM_PROMPT` (`_gemma_judge()` in `evaluator.py`; metric name kept for compatibility)
- OpenEvolve **maximizes** `combined_score`

### Multi-GPU evolution (important)

- **4 worker processes** = 4 GPUs (`parallel_evaluations: 4` in `configs/multi_gpu.yaml`)
- `install_gpu_worker_patch()` replaces OpenEvolve `process_parallel._worker_init` with `sap_worker_init()` (`core/gpu_worker.py`): round-robin `CUDA_VISIBLE_DEVICES` via file lock `.gpu_assign_counter`
- Each worker loads **one FLUX** instance; **per-GPU file lock** `.generation.gpu{N}.lock` serializes generation on that GPU
- Parent scheduler sets `SAP_RELEASE_MODEL_AFTER_EVAL=1` for the initial eval; workers set `SAP_RELEASE_MODEL_AFTER_EVAL=0` to keep FLUX loaded across evals
- Checkpoints every **50** iterations → `output/checkpoints/checkpoint_<N>/` with `metadata.json`, `programs/*.json`
- Until the first checkpoint, use the visualizer **live mode** (`experiment.jsonl` + `eval_results/`)

### Naming note: `exp_logging/`

The package `openevolve_sap/exp_logging/` exists because a folder named `logging/` shadowed Python’s stdlib `logging` when worker processes re-imported the project under `spawn`.

---

## Function call reference

### A. Manual image generation (single prompt)

```
main()  [run_SAP_flux.py]
  └─ parse_input_arguments()
  └─ run(args)
       ├─ LLM_SAP(prompt, llm, key)           [llm_interface/llm_SAP.py]
       │    ├─ LLM_SAP_batch_gpt()  → RouterAI qwen/qwen3.5-35b-a3b
       │    └─ parse_batched_llm_output() → get_params_dict_SAP()
       ├─ generate_models_params(args, SAP_prompts)
       ├─ load_model() → SapFlux.from_pretrained(local_path)
       ├─ model(**params).images              [SapFlux.__call__ → map_SAP_dict → denoise loop]
       ├─ save_results() → results/<prompt>/Seed<seed>.png
       └─ [optional] evaluate_image_with_gpt()  [benchmarks/gpt_eval.py]
```

**CLI:**

```bash
python run_SAP_flux.py --prompt "..." --seeds_list 30498 [--score] [--use_sap true] [--llm GPT|Zephyr]
```

| Argument | Default | Effect |
|----------|---------|--------|
| `--use_sap` | `true` | If false: single prompt, no decomposition |
| `--llm` | `GPT` | `GPT` = RouterAI; `Zephyr` = local HF pipeline |
| `--score` | `false` | Run VL alignment after save |
| `--sap_system_prompt_path` | `""` | Sets `SAP_SYSTEM_PROMPT_PATH` for decomposition |

---

### B. OpenEvolve evolution (multi-GPU)

```
main()  [scripts/run_evolution.py, MainProcess only]
  └─ openevolve_sap.core.scheduler.main()
       └─ asyncio.run(run_evolution_async(args))
            ├─ preflight_gpus()
            ├─ build_experiment_dir() → experiments/experiment_<timestamp>/
            ├─ prepare_env()            # SAP_* env vars
            ├─ install_gpu_worker_patch()
            ├─ patch_controller_checkpoints()
            ├─ load_config() + config.prompt.system_message = load_evolution_system_message()
            ├─ GPUMonitor.start()
            └─ OpenEvolve(...).run(iterations=)
                 ├─ LLM mutations: google/gemini-3.1-pro-preview (RouterAI)
                 └─ evaluate(program_path)  [openevolve_sap/evaluator.py] per candidate
```

**`evaluate(program_path)`** (OpenEvolve entry point):

```
evaluate(program_path, visualization_only=False)
  ├─ _extract_system_prompt(program_path)   # import SYSTEM_PROMPT from temp .py
  ├─ _load_prompt_set()                     # openevolve_sap/prompt_set.json (3 prompts)
  ├─ _get_model() → load_model()            # FLUX on worker GPU
  ├─ os.environ["SAP_SYSTEM_PROMPT_PATH"] = temp file with evolved prompt
  └─ for each test prompt:
       ├─ LLM_SAP(prompt) → decomposition
       ├─ _save_decomposition() → eval_results/<run_id>/prompt_XX/
       ├─ SapFlux(...) under generation lock
       ├─ evaluate_image_with_gpt(image, prompt) → alignment 1–5
       └─ _save_score(), images → prompt_XX/image_00.png
  ├─ _gemma_judge(system_prompt, ...)       # gemini-3.1-pro-preview
  ├─ combined_score
  └─ _write_run_manifest() → manifest.json
```

**CLI (`scripts/run_evolution.py` / `openevolve_sap/run_openevolve_sap.py`):**

| Flag | Description |
|------|-------------|
| `--config` | YAML (default `openevolve_sap/configs/multi_gpu.yaml`) |
| `--gpus 0 1 2 3` | Physical GPU indices |
| `--iterations` / `-i` | Override `max_iterations` |
| `--checkpoint-interval` | Override checkpoint period |
| `--checkpoint` | Resume from `checkpoint_<N>` dir |
| `--experiment-dir` | Fixed experiment folder (else auto timestamp) |
| `--ram-limit-gb` | Per-process RSS cap (default 75% RAM) |
| `--export-best` | Write best `SYSTEM_PROMPT` text after run |

---

### C. Standalone evaluator / render-only

```
python openevolve_sap/evaluator.py [--program PATH] [--output-dir DIR]
  └─ find_latest_checkpoint_program()  # if --program omitted
  └─ evaluate_visualization_only(program_path)
       └─ evaluate(..., visualization_only=True)  # images only, no judges
```

---

### D. Visualization server

```
python openevolve_sap/visualization/visualizer.py --checkpoint ... --experiment-dir ...
  ├─ merge_evolution_data(checkpoint, experiment_dir)  [utils.py]
  │    ├─ load_evolution_data() if checkpoint has metadata.json
  │    └─ else load_live_experiment_data() from experiment.jsonl + manifests
  ├─ GET /api/data, /api/metrics
  ├─ GET /api/eval_runs, /api/eval_image/<run_id>/<prompt_index>
  └─ Flask static: Branching / Performance / List / Evals tabs
```

---

### E. `llm_interface/llm_SAP.py` API

| Function | Role |
|----------|------|
| `load_sap_system_prompt_text()` | Reads `SAP_SYSTEM_PROMPT_PATH` or `template/template_SAP_system.txt` |
| `LLM_SAP(prompts_list, llm, key)` | Dispatcher: `GPT` → RouterAI, `Zephyr` → local |
| `LLM_SAP_batch_gpt(prompts_list, key)` | Batched decomposition via `template_SAP_user.txt` |
| `LLM_SAP_batch_Zephyr(...)` | Local Zephyr pipeline |
| `parse_batched_llm_output(text, originals)` | Split `### Input N:` blocks |
| `get_params_dict_SAP(response)` | Parse explanation + `{prompts_list, switch_prompts_steps}` |

---

### F. `benchmarks/gpt_eval.py`

| Function | Role |
|----------|------|
| `evaluate_image_with_gpt(image_path, prompt, key)` | VL model scores alignment + quality; returns dict with `alignment score`, `quality score`, explanations |

Model: `qwen/qwen3-vl-235b-a22b-thinking` via RouterAI.

---

### G. Core modules (`openevolve_sap/core/`)

| Module | Key symbols |
|--------|-------------|
| `scheduler.py` | `main()`, `run_evolution_async()`, `prepare_env()`, `load_evolution_system_message()` |
| `gpu_worker.py` | `install_gpu_worker_patch()`, `sap_worker_init()`, `assign_gpu_for_worker()` |
| `checkpoint.py` | `patch_controller_checkpoints()`, `enrich_checkpoint()`, `save_rng_state()` |

---

## Environment variables

| Variable | Used by | Meaning |
|----------|---------|---------|
| `ROUTERAI_API_KEY` | LLM + judges | RouterAI / OpenAI-compatible API key |
| `ROUTERAI_BASE_URL` | LLM + judges | Default `https://routerai.ru/api/v1` |
| `OPENAI_API_KEY` | OpenEvolve | Copied from `ROUTERAI_API_KEY` in scheduler if unset |
| `SAP_FLUX_MODEL_PATH` | **Required** | Local directory with FLUX.2-klein-base-4B weights |
| `SAP_SYSTEM_PROMPT_PATH` | `LLM_SAP`, evaluator | Override decomposition system prompt (evolved programs write a temp file) |
| `SAP_CUDA_DEVICE` | FLUX | Local CUDA index inside `CUDA_VISIBLE_DEVICES` (usually `0`) |
| `SAP_PHYSICAL_GPU_ID` | Locks, logging | Physical GPU id string |
| `SAP_GPU_IDS` | Worker assignment | Comma-separated list, e.g. `0,1,2,3` |
| `SAP_EXPERIMENT_DIR` | Logging, eval output | Active `experiments/experiment_*` path |
| `SAP_EVOLUTION_RESULTS_DIR` | evaluator | Defaults to `<experiment_dir>/eval_results` |
| `SAP_CONFIG_PATH` | checkpoint enrich | Path to YAML copied into checkpoints |
| `SAP_NUM_INFERENCE_STEPS` | Evolution eval | Default `30` |
| `SAP_IMAGE_HEIGHT` / `SAP_IMAGE_WIDTH` | Evolution eval | Default `512` |
| `SAP_RAM_LIMIT_GB` | evaluator | Per-process RSS limit (default ~75% system RAM) |
| `SAP_LOW_VRAM` | `load_model()` | `1` = CPU offload on GPU (default for 10GB cards) |
| `SAP_RELEASE_MODEL_AFTER_EVAL` | Parent vs worker | `1` in scheduler parent, `0` in pool workers |
| `SAP_ENABLE_GPU_PATCH` | startup | Must be `1` for multi-GPU worker pinning |
| `SAP_WORKER_ID` | Logs | e.g. `worker_0` |
| `SAP_CHECKPOINT_INTERVAL` | OpenEvolve | Set from `--checkpoint-interval` |
| `SAP_LOG_LEVEL` | experiment logger | e.g. `INFO` |
| `PYTHONPATH` | visualizer, imports | Project root when running `visualization/visualizer.py` |

---

## Config files

| File | Role |
|------|------|
| `openevolve_sap/configs/multi_gpu.yaml` | Production: 80 iter, 4 islands, 4 parallel evals, checkpoint 50 |
| `openevolve_sap/config.yaml` | Shorter defaults (1 island, 1 parallel eval) |
| `openevolve_sap/prompt_set.json` | Fixed 3 contradictory test prompts for evolution |
| `openevolve_sap/prompts/evolution_system_message.md` | Full meta-prompt for Gemini (loaded at runtime, not inlined in YAML) |
| `openevolve_sap/initial_program.py` | Seed `SYSTEM_PROMPT` for generation 0 |
| `llm_interface/template/template_SAP_system.txt` | Default decomposition instructions |
| `llm_interface/template/template_SAP_user.txt` | User batch template for decomposition |

---

## Output directories

| Path | Contents |
|------|----------|
| `results/<prompt>/Seed*.png` | Manual `run_SAP_flux.py` outputs |
| `openevolve_sap/output/best/` | Best program after evolution (`best_program.py`, `best_program_info.json`) |
| `openevolve_sap/output/checkpoints/checkpoint_<N>/` | OpenEvolve DB snapshot + `programs/` |
| `openevolve_sap/experiments/experiment_<ts>/` | Per-run logs, GPU CSV, `eval_results/`, copied `config.yaml` |
| `openevolve_sap/experiments/.../eval_results/<run_id>/` | `manifest.json`, `status.jsonl`, `prompt_XX/{decomposition.json, image_00.png, score.json}` |

---

## Other entry points

| Script | Purpose |
|--------|---------|
| `app.py` | Gradio demo (FLUX baseline vs SAP + Zephyr/GPT); uses `run_SAP_flux` helpers |
| `compare_sap.py` | Batch comparison utilities (paper experiments) |
| `test.py` | Ad-hoc tests |
| `SAP_pipeline_flux.py` | `SapFlux` pipeline only (imported by `run_SAP_flux`) |

---

## Setup

### Environment

On this machine, dependencies are installed in a **venv** at `/home/ubuntu/venv` (Python 3.12). Activate it before any run:

```bash
source /home/ubuntu/venv/bin/activate
```

Verify the environment:

```bash
which python   # should be /home/ubuntu/venv/bin/python
python -c "import torch, diffusers, openevolve, openai; print('OK')"
```

**Alternative (from upstream paper repo):** conda with Python 3.10:

```bash
conda create -n sap python=3.10 -y
conda activate sap
pip install -r requirements.txt
```

Clone and install (if setting up from scratch):

```bash
git clone https://github.com/TDPC2025/SAP.git
cd SAP   # or evolve_SAP
source /home/ubuntu/venv/bin/activate   # or: conda activate sap
pip install -r requirements.txt
```

---

## Usage

Activate the venv first:

```bash
source /home/ubuntu/venv/bin/activate
```

### Generate images (SAP)

Export credentials and the local FLUX path, then run:

```bash
export ROUTERAI_API_KEY="YOUR_API_KEY"
export ROUTERAI_BASE_URL="https://routerai.ru/api/v1"
export SAP_FLUX_MODEL_PATH="/absolute/path/to/local/FLUX.2-klein-base-4B"

python run_SAP_flux.py --prompt "your prompt" --seeds_list seed1 seed2 seed3

# Example:
python run_SAP_flux.py --prompt "A bear is performing a handstand in the park" --seeds_list 30498
```

**Models used:**

| Role | Model |
|------|--------|
| Image generation (local) | `black-forest-labs/FLUX.2-klein-base-4B` |
| Prompt decomposition (RouterAI) | `qwen/qwen3.5-35b-a3b` |
| Alignment scoring (RouterAI) | `qwen/qwen3-vl-235b-a22b-thinking` |
| Evolution mutations (RouterAI, OpenEvolve only) | `google/gemini-3.1-pro-preview` |
| Prompt-quality judge (RouterAI, evaluator only) | `google/gemini-3.1-pro-preview` |

### Multi-GPU evolution (4× RTX 3080)

Target hardware: **4× RTX 3080 (10240 MiB each)**. One OpenEvolve worker process is pinned per GPU via `CUDA_VISIBLE_DEVICES`.

```bash
source /home/ubuntu/venv/bin/activate
export ROUTERAI_API_KEY="YOUR_API_KEY"
export ROUTERAI_BASE_URL="https://routerai.ru/api/v1"
export SAP_FLUX_MODEL_PATH="/absolute/path/to/local/FLUX.2-klein-base-4B"

# Full multi-GPU run (config: openevolve_sap/configs/multi_gpu.yaml)
python scripts/run_evolution.py \
  --config openevolve_sap/configs/multi_gpu.yaml \
  --gpus 0 1 2 3 \
  --checkpoint-interval 50 \
  --log-level INFO

# Smoke test (few iterations)
python scripts/run_evolution.py --iterations 4 --gpus 0 1 2 3

# Resume from checkpoint
python scripts/run_evolution.py \
  --checkpoint openevolve_sap/output/checkpoints/checkpoint_50 \
  --gpus 0 1 2 3
```

**Evolution eval defaults:** 512×512 images, **30 inference steps** (`SAP_NUM_INFERENCE_STEPS`, `SAP_IMAGE_HEIGHT`, `SAP_IMAGE_WIDTH`).

**Experiment artifacts** (under `openevolve_sap/experiments/experiment_<timestamp>/`):

| File | Description |
|------|-------------|
| `experiment.jsonl` | Structured evolution log (JSON lines) |
| `gpu_metrics.csv` | GPU util / memory / temp every 5s |
| `evolution_stats.csv` | Best/avg score per checkpoint |
| `checkpoints_manifest.json` | Checkpoint index |
| `eval_results/<run_id>/` | Per run: `manifest.json`, `status.jsonl`, `prompt_00/` … (`original_prompt.txt`, `decomposition.json`, `image_00.png`, `score.json`) |

**Visualize evolution (live, without stopping `run_evolution.py`):**

While checkpoints are saved every 50 iterations, the UI can show **live** eval data from `experiment.jsonl` and `eval_results/`:

```bash
cd /home/ubuntu/evolve_SAP
export PYTHONPATH=.
python openevolve_sap/visualization/visualizer.py \
  --checkpoint openevolve_sap/output \
  --experiment-dir openevolve_sap/experiments/experiment_YYYYMMDD_HHMMSS \
  --host 0.0.0.0 --port 8050
```

Replace `experiment_YYYYMMDD_HHMMSS` with the active run folder under `openevolve_sap/experiments/`.

**Restart visualizer only** (does not stop evolution):

```bash
pkill -f "openevolve_sap/visualization/visualizer.py"   # only Flask UI
# then run the python command above again
```

Tabs: **Branching** (program graph), **Performance** (GPU), **List**, **Evals** (image gallery from `eval_results`).

After checkpoint 50+ exists, the same command also loads full OpenEvolve program trees from `output/checkpoints/`.

### OpenEvolve runtime policy

- FLUX loading is **local-only** (`SAP_FLUX_MODEL_PATH` required);
- **one generation stream per GPU** (per-GPU file lock, not global);
- **RAM limit per process** (default **75% of MemTotal**, ~96 GiB on 128 GiB hosts; override via `--ram-limit-gb` / `SAP_RAM_LIMIT_GB`);

Manual SAP runs save images to:

```
results/<prompt>/Seed<seed>.png
```

### Single-GPU / legacy launcher

`python openevolve_sap/run_openevolve_sap.py` is an alias for `scripts/run_evolution.py` (defaults to multi-GPU config if you pass `--config openevolve_sap/configs/multi_gpu.yaml`).

**Evaluator metrics:** `alignment_score`, `gemma_score`, `combined_score = 0.8·alignment/5 + 0.2·gemma/5`. Test prompts in `openevolve_sap/prompt_set.json`:

1. A bouquet of flowers is upside down in a vase  
2. A white glove has 6 fingers  
3. The shadow of a cat is facing the opposite direction  

**Render-only** (no RouterAI scoring):

```bash
python openevolve_sap/evaluator.py --program path/to/best_program.py
```

---

## 📊 Benchmarks

We evaluate our method using three benchmarks designed to challenge text-to-image models with **contextually contradictory prompts**:

- **Whoops!**  
  A dataset of 500 prompts designed to expose failures in visual reasoning when faced with commonsense-defying descriptions.

- **Whoops-Hard** (✨ introduced in this paper)  
  A curated subset of 100 particularly challenging prompts from Whoops! where existing models often fail to preserve semantic intent.

- **ContraBench** (🆕 introduced in this paper)  
  A novel benchmark of 40 prompts carefully constructed to include **Contextual contradictions**.

### 🧪 Evaluation

We include `gpt_eval.py`, the automatic evaluator used in the paper.  
It uses `qwen/qwen3-vl-235b-a22b-thinking` (via RouterAI) to assess image-text alignment by scoring how well generated images reflect the semantics of the prompt.


### 📁 Benchmarks Structure

All benchmark-related resources are organized under the `benchmarks/` folder:

```
benchmarks/
├── original_prompts/ # Raw prompts for Whoops!, Whoops-Hard, and ContraBench
├── SAP_prompts/ # Decomposed proxy prompts from our method
├── evaluated_seeds/ # Fixed seeds used for reproducibility
└── gpt_eval.py # GPT-based evaluator for semantic alignment
```

## Acknowledgements 

This code was built using the code from the following repositories:
- [diffusers](https://github.com/huggingface/diffusers)

## Citation

If you use this code for your research, please cite our paper:

```
@article{huberman2025image,
  title={Image Generation from Contextually-Contradictory Prompts},
  author={Huberman, Saar and Patashnik, Or and Dahary, Omer and Mokady, Ron and Cohen-Or, Daniel},
  journal={arXiv preprint arXiv:2506.01929},
  year={2025}
}
```