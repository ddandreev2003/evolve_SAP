import argparse
import ast
import json
import os
import subprocess
import sys
from shutil import which
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run OpenEvolve for SAP system prompt evolution.")
    parser.add_argument("--iterations", type=int, default=80, help="Number of OpenEvolve iterations.")
    parser.add_argument("--output", type=str, default="openevolve_sap/output", help="Output directory.")
    parser.add_argument(
        "--config",
        type=str,
        default="openevolve_sap/config.yaml",
        help="Path to OpenEvolve config YAML.",
    )
    parser.add_argument(
        "--export-best",
        type=str,
        default="openevolve_sap/best_evolved_system_prompt.txt",
        help="Export path for best evolved SAP system prompt.",
    )
    return parser.parse_args()


def extract_system_prompt_from_program(program_path: Path) -> str:
    source = program_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "SYSTEM_PROMPT":
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        return node.value.value
    raise ValueError(f"SYSTEM_PROMPT was not found in {program_path}")


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    initial_program = root / "openevolve_sap/initial_program.py"
    evaluator_file = root / "openevolve_sap/evaluator.py"
    config_file = root / args.config
    output_dir = root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    child_env = os.environ.copy()
    if not child_env.get("OPENAI_API_KEY") and child_env.get("ROUTERAI_API_KEY"):
        child_env["OPENAI_API_KEY"] = child_env["ROUTERAI_API_KEY"]

    if which("openevolve-run.py"):
        cmd = [
            "openevolve-run.py",
            str(initial_program),
            str(evaluator_file),
            "--config",
            str(config_file),
            "--iterations",
            str(args.iterations),
            "--output",
            str(output_dir),
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "openevolve.cli",
            str(initial_program),
            str(evaluator_file),
            "--config",
            str(config_file),
            "--iterations",
            str(args.iterations),
            "--output",
            str(output_dir),
        ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=child_env)

    best_program_path = output_dir / "best/best_program.py"
    export_path = root / args.export_best
    if best_program_path.exists():
        prompt_text = extract_system_prompt_from_program(best_program_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(prompt_text, encoding="utf-8")
        print(f"Exported best evolved prompt to: {export_path}")
    else:
        print(f"Best program not found at: {best_program_path}")
        print("Evolution finished, but export step was skipped.")

    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "iterations": args.iterations,
                "output_dir": str(output_dir),
                "best_program_path": str(best_program_path),
                "export_path": str(export_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
