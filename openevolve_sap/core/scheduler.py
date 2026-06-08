"""Multi-GPU evolution scheduler: preflight, env, OpenEvolve run, monitoring."""
from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch

from openevolve_sap.core.checkpoint import patch_controller_checkpoints
from openevolve_sap.core.evolution_patch import install_evolution_patch
from openevolve_sap.core.gpu_worker import install_gpu_worker_patch
from openevolve_sap.exp_logging.experiment_logger import ExperimentLogger
from openevolve_sap.exp_logging.gpu_monitor import GPUMonitor
from openevolve_sap.export_utils import extract_system_prompt_from_program
from openevolve_sap.sap_eval_settings import get_ram_limit_gb, system_ram_gb

EVOLUTION_META_PROMPT_PATH = PROJECT_ROOT / "openevolve_sap" / "prompts" / "evolution_system_message.md"


def load_evolution_system_message() -> str:
    if not EVOLUTION_META_PROMPT_PATH.is_file():
        raise FileNotFoundError(f"Evolution meta prompt not found: {EVOLUTION_META_PROMPT_PATH}")
    return EVOLUTION_META_PROMPT_PATH.read_text(encoding="utf-8").strip()


def preflight_gpus(gpu_ids: list[int], min_free_mib: int = 500) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    count = torch.cuda.device_count()
    if count < len(gpu_ids):
        raise RuntimeError(f"Need {len(gpu_ids)} GPUs, found {count}")
    try:
        try:
            import pynvml
        except ImportError:
            import nvidia_ml_py as pynvml  # noqa: F401

        pynvml.nvmlInit()
        for gid in gpu_ids:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gid)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mib = mem.total / (1024 * 1024)
            used_mib = mem.used / (1024 * 1024)
            if total_mib < 10000:
                raise RuntimeError(f"GPU {gid}: expected ~10240 MiB, got {total_mib:.0f}")
            if used_mib > min_free_mib:
                raise RuntimeError(
                    f"GPU {gid}: {used_mib:.0f} MiB in use (need < {min_free_mib} free)"
                )
        pynvml.nvmlShutdown()
    except ImportError:
        for gid in gpu_ids:
            props = torch.cuda.get_device_properties(gid)
            if props.total_memory < 10 * 1024**3:
                raise RuntimeError(f"GPU {gid}: insufficient VRAM")


def build_experiment_dir(base: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp = base / f"experiment_{stamp}"
    exp.mkdir(parents=True, exist_ok=True)
    (exp / "logs").mkdir(exist_ok=True)
    return exp


def prepare_env(
    gpu_ids: list[int],
    experiment_dir: Path,
    config_path: Path,
    output_dir: Path,
    ram_limit_gb: float,
    log_level: str,
) -> dict[str, str]:
    env = os.environ.copy()
    if not env.get("OPENAI_API_KEY") and env.get("ROUTERAI_API_KEY"):
        env["OPENAI_API_KEY"] = env["ROUTERAI_API_KEY"]

    env["SAP_GPU_IDS"] = ",".join(str(g) for g in gpu_ids)
    env["SAP_EXPERIMENT_DIR"] = str(experiment_dir)
    env["SAP_CONFIG_PATH"] = str(config_path)
    env["SAP_EVOLUTION_RESULTS_DIR"] = str(experiment_dir / "eval_results")
    env["SAP_RAM_LIMIT_GB"] = str(ram_limit_gb)
    env["SAP_NUM_INFERENCE_STEPS"] = env.get("SAP_NUM_INFERENCE_STEPS", "30")
    env["SAP_SEEDS_LIST"] = env.get("SAP_SEEDS_LIST", "30498,30499")
    env["SAP_IMAGE_HEIGHT"] = env.get("SAP_IMAGE_HEIGHT", "512")
    env["SAP_IMAGE_WIDTH"] = env.get("SAP_IMAGE_WIDTH", "512")
    env["SAP_LOG_LEVEL"] = log_level
    env["SAP_ENABLE_GPU_PATCH"] = "1"
    env["SAP_RELEASE_MODEL_AFTER_EVAL"] = "1"
    env["SAP_LOW_VRAM"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    if not env.get("SAP_FLUX_MODEL_PATH", "").strip():
        raise RuntimeError("SAP_FLUX_MODEL_PATH must be set")

    Path(env["SAP_EVOLUTION_RESULTS_DIR"]).mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    counter = experiment_dir / ".gpu_assign_counter"
    if counter.exists():
        counter.unlink()

    return env


async def run_evolution_async(args: argparse.Namespace) -> int:
    root = PROJECT_ROOT
    gpu_ids = [int(x) for x in args.gpus]
    preflight_gpus(gpu_ids)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    experiment_dir = Path(args.experiment_dir) if args.experiment_dir else None
    if experiment_dir is None:
        experiment_dir = build_experiment_dir(root / "openevolve_sap/experiments")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(config_path, experiment_dir / "config.yaml")

    ram_limit_gb = args.ram_limit_gb if args.ram_limit_gb is not None else get_ram_limit_gb()
    env = prepare_env(
        gpu_ids,
        experiment_dir,
        config_path,
        output_dir,
        ram_limit_gb,
        args.log_level,
    )
    for key, value in env.items():
        os.environ[key] = value

    logger = ExperimentLogger(experiment_dir, level=args.log_level)
    meta_prompt = load_evolution_system_message()
    logger.log(
        "INFO",
        "scheduler",
        "experiment_start",
        {
            "gpu_ids": gpu_ids,
            "output": str(output_dir),
            "ram_limit_gb": ram_limit_gb,
            "system_ram_gb": system_ram_gb(),
            "meta_prompt_chars": len(meta_prompt),
            "meta_prompt_path": str(EVOLUTION_META_PROMPT_PATH),
        },
    )

    monitor = GPUMonitor(experiment_dir, gpu_ids, interval_sec=args.gpu_monitor_interval)
    monitor.start()

    install_gpu_worker_patch()
    install_evolution_patch()
    patch_controller_checkpoints()

    initial_program = root / "openevolve_sap/initial_program.py"
    evaluator_file = root / "openevolve_sap/evaluator.py"

    from openevolve import OpenEvolve
    from openevolve.config import load_config

    config = load_config(str(config_path))
    config.prompt.system_message = load_evolution_system_message()
    if args.iterations is not None:
        config.max_iterations = args.iterations
    ckpt_int = args.checkpoint_interval or int(os.getenv("SAP_CHECKPOINT_INTERVAL", "0") or 0)
    if ckpt_int > 0:
        config.checkpoint_interval = ckpt_int
    config.evaluator.parallel_evaluations = len(gpu_ids)
    if os.getenv("SAP_CASCADE_EVAL", "").strip().lower() in {"1", "true", "yes"}:
        config.evaluator.cascade_evaluation = True
    cascade_thresh = os.getenv("SAP_CASCADE_THRESHOLDS", "").strip()
    if cascade_thresh:
        config.evaluator.cascade_thresholds = [
            float(x.strip()) for x in cascade_thresh.split(",") if x.strip()
        ]

    openevolve = OpenEvolve(
        initial_program_path=str(initial_program),
        evaluation_file=str(evaluator_file),
        config=config,
        output_dir=str(output_dir),
    )

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        openevolve.database.load(str(ckpt))
        logger.log("INFO", "scheduler", "checkpoint_loaded", {"path": str(ckpt)})

    try:
        best = await openevolve.run(
            iterations=args.iterations,
            target_score=args.target_score,
            checkpoint_path=args.checkpoint,
        )
        logger.log(
            "INFO",
            "scheduler",
            "experiment_complete",
            {"metrics": dict(best.metrics) if best else {}},
        )
    finally:
        monitor.stop()

    export_path = root / args.export_best
    best_program_path = output_dir / "best/best_program.py"
    if best_program_path.exists():
        prompt_text = extract_system_prompt_from_program(best_program_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(prompt_text, encoding="utf-8")
        print(f"Exported best prompt: {export_path}")

    summary = {
        "experiment_dir": str(experiment_dir),
        "output_dir": str(output_dir),
        "gpu_ids": gpu_ids,
        "iterations": args.iterations,
    }
    (experiment_dir / "run_summary.json").write_text(
        __import__("json").dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Experiment dir: {experiment_dir}")
    return 0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAP OpenEvolve on multiple GPUs")
    parser.add_argument("--config", default="openevolve_sap/configs/multi_gpu.yaml")
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--iterations", "-i", type=int, default=None)
    parser.add_argument("--output", "-o", default="openevolve_sap/output")
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--target-score", type=float, default=None)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--ram-limit-gb",
        type=float,
        default=None,
        help="Per-process RSS cap in GiB (default: 75%% of system RAM)",
    )
    parser.add_argument("--gpu-monitor-interval", type=float, default=5.0)
    parser.add_argument(
        "--export-best",
        default="openevolve_sap/best_evolved_system_prompt.txt",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.checkpoint_interval is not None:
        os.environ["SAP_CHECKPOINT_INTERVAL"] = str(args.checkpoint_interval)
    try:
        return asyncio.run(run_evolution_async(args))
    except KeyboardInterrupt:
        print("Shutdown requested")
        return 130


if __name__ == "__main__":
    sys.exit(main())
