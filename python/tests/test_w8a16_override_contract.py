from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "third_party/sparkinfer/b12x/moe/_shared/kernels/w4a16/kernel.py"
OVERRIDE = ROOT / "python/tools/tune_w8a16_cute_packed_prefill.py"


def _method_arity(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    assert len(matches) == 1, (path, name, len(matches))
    return len(matches[0].args.args) - 1


def _inherited_call_arities(path: Path, name: str) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        len(node.args)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == name
    ]


def test_packed_prefill_override_tracks_sparkinfer_output_drain_abi() -> None:
    for name in ("_drain_output_smem", "_drain_output_smem_tail"):
        expected = _method_arity(BASE, name)
        calls = _inherited_call_arities(OVERRIDE, name)
        assert calls, name
        assert calls == [expected] * len(calls), (name, expected, calls)
