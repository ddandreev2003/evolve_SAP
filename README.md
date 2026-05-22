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
4. **Evolution (optional)** — `openevolve_sap` mutates the system prompt (OpenEvolve + `google/gemini-3.1-pro-preview`), re-runs steps 1–3, and optimizes `combined_score` (80% alignment + 20% Gemma judge).

All cloud LLM calls use the **OpenAI-compatible RouterAI API** (`ROUTERAI_API_KEY`, default base `https://routerai.ru/api/v1`). Set `llm=Zephyr` in code to use a local Hugging Face model instead (no RouterAI for decomposition).

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
| Prompt-quality judge (RouterAI, evaluator only) | `google/gemma-4-26b-a4b-it` |

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

**Visualize checkpoint (no evolution run):**

```bash
python openevolve_sap/visualization/visualizer.py \
  --checkpoint openevolve_sap/output/checkpoints/checkpoint_50 \
  --port 8050
```

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