#!/usr/bin/env python3
"""Static SOLID boundary guard for SmartBOM.

Find objective architecture violations and report complexity signals without
pretending to prove semantic SOLID compliance.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Iterable


ROOT = Path(__file__).resolve().parents[3]
PRODUCTION_PREFIXES = (
    "smartbom_bom/",
    "connectivity_bom_main.py",
    "connectivity_bom_batch.py",
    "reanalyze_batch_bom.py",
    "dxf_reader.py",
)
DOMAIN_MODULES = {
    "config.py", "models.py", "geometry.py", "grouping.py", "text.py",
    "predicates.py", "optimizer.py", "rules.py", "reporting.py", "review.py",
    "service.py", "batch.py",
}
MAX_MODULE_LINES = 500
MAX_FUNCTION_LINES = 100
MAX_CLASS_METHODS = 12


@dataclass(frozen=True)
class Finding:
    severity: str
    path: Path
    line: int
    principle: str
    message: str


def _git_lines(*args: str) -> set[str]:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _select_paths(names: Iterable[str]) -> list[Path]:
    selected = []
    for name in names:
        normalized = Path(name).as_posix()
        if not normalized.endswith(".py"):
            continue
        if not any(normalized == prefix or normalized.startswith(prefix) for prefix in PRODUCTION_PREFIXES):
            continue
        path = (ROOT / normalized).resolve()
        if path.is_file() and ROOT in path.parents:
            selected.append(path)
    return sorted(set(selected))


def changed_python_files() -> list[Path]:
    names = _git_lines("diff", "--name-only", "--diff-filter=ACMR", "HEAD")
    names |= _git_lines("ls-files", "--others", "--exclude-standard")
    return _select_paths(names)


def all_python_files() -> list[Path]:
    return _select_paths(path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*.py"))


def import_roots(tree: ast.AST) -> list[tuple[str, int]]:
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.module, node.lineno))
    return imports


def function_length(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    return int(getattr(node, "end_lineno", node.lineno)) - node.lineno + 1


def inspect_file(path: Path) -> list[Finding]:
    relative = path.relative_to(ROOT)
    findings: list[Finding] = []
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(relative))
    except SyntaxError as exc:
        return [Finding("ERROR", relative, exc.lineno or 1, "Correctness", exc.msg)]

    line_count = len(source.splitlines())
    if line_count > MAX_MODULE_LINES:
        findings.append(Finding(
            "WARN", relative, 1, "SRP",
            f"module has {line_count} lines (threshold {MAX_MODULE_LINES}); verify responsibilities",
        ))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            length = function_length(node)
            if length > MAX_FUNCTION_LINES:
                findings.append(Finding(
                    "WARN", relative, node.lineno, "SRP",
                    f"function {node.name} has {length} lines (threshold {MAX_FUNCTION_LINES})",
                ))
        elif isinstance(node, ast.ClassDef):
            methods = [item for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if len(methods) > MAX_CLASS_METHODS:
                findings.append(Finding(
                    "WARN", relative, node.lineno, "SRP/ISP",
                    f"class {node.name} exposes {len(methods)} methods (threshold {MAX_CLASS_METHODS})",
                ))

    package_file = relative.parent.as_posix() == "smartbom_bom"
    is_visualization = relative.as_posix() == "smartbom_bom/visualization.py"
    is_io = relative.as_posix() == "smartbom_bom/io.py"
    for module, line in import_roots(tree):
        if package_file and module.startswith(("connectivity_bom_main", "connectivity_bom_batch")):
            findings.append(Finding(
                "ERROR", relative, line, "DIP", "package code must not import a UI/batch entry point",
            ))
        if package_file and relative.name in DOMAIN_MODULES and module.startswith("gradio"):
            findings.append(Finding(
                "ERROR", relative, line, "SRP/DIP", "domain/application module must not import Gradio",
            ))
        if package_file and relative.name in DOMAIN_MODULES and module.startswith("plotly"):
            findings.append(Finding(
                "ERROR", relative, line, "SRP/DIP", "domain/application module must not import Plotly",
            ))
        if package_file and module == "dxf_reader" and not is_io:
            findings.append(Finding(
                "ERROR", relative, line, "SRP/DIP", "only smartbom_bom/io.py may import dxf_reader",
            ))
        if module.startswith("plotly") and package_file and not is_visualization:
            findings.append(Finding(
                "ERROR", relative, line, "SRP", "Plotly belongs in visualization.py",
            ))

    if package_file and not is_io:
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {"read_csv", "to_csv"}:
                findings.append(Finding(
                    "WARN", relative, node.lineno, "SRP",
                    f"{node.attr} outside io.py; verify persistence has not leaked into domain logic",
                ))
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="specific repository-relative Python files")
    parser.add_argument("--all", action="store_true", help="check every production Python file")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.paths:
        files = _select_paths(args.paths)
    elif args.all:
        files = all_python_files()
    else:
        files = changed_python_files()

    if not files:
        print("SOLID guard: no changed production Python files to inspect.")
        return 0

    findings = [finding for path in files for finding in inspect_file(path)]
    for finding in sorted(findings, key=lambda item: (str(item.path), item.line, item.severity)):
        print(
            f"{finding.severity} {finding.path}:{finding.line} "
            f"[{finding.principle}] {finding.message}"
        )
    errors = sum(item.severity == "ERROR" for item in findings)
    warnings = sum(item.severity == "WARN" for item in findings)
    print(f"SOLID guard: checked {len(files)} file(s), {errors} error(s), {warnings} warning(s).")
    return 1 if errors or (args.strict and warnings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
