"""Utilities for exporting evolved programs."""
from __future__ import annotations

import ast
from pathlib import Path


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
