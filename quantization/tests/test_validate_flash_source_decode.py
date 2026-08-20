from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "validate_flash_source_decode.py"
SPEC = importlib.util.spec_from_file_location("validate_flash_source_decode", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_default_cases_cover_target_and_every_mtp_block() -> None:
    cases = MODULE.parse_cases(MODULE.DEFAULT_CASES)
    assert cases[:3] == [
        ("layers", 0, 0),
        ("layers", 21, 127),
        ("layers", 42, 255),
    ]
    assert cases[3:] == [("mtp", 0, 0), ("mtp", 1, 127), ("mtp", 2, 255)]


@pytest.mark.parametrize("value", ["", "layer.0:0", "layers.-1:0", "mtp.0", "mtp.0:-1"])
def test_case_parser_rejects_malformed_values(value: str) -> None:
    with pytest.raises(MODULE.ValidationError):
        MODULE.parse_cases([value])


def test_case_parser_deduplicates_without_reordering() -> None:
    assert MODULE.parse_cases(["mtp.1:7", "layers.2:3", "mtp.1:7"]) == [
        ("mtp", 1, 7),
        ("layers", 2, 3),
    ]


def test_projection_names_bind_checkpoint_and_runtime_aliases() -> None:
    assert MODULE.checkpoint_base("layers", 4, 9, "w3") == (
        "layers.4.ffn.experts.9.w3"
    )
    assert MODULE.runtime_module_path("layers", 4, 9, "w3") == (
        "model.layers.4.mlp.experts.9.up_proj"
    )
    assert MODULE.runtime_module_path("mtp", 1, 9, "w3") is None


def test_geometry_validation_includes_mtp_and_expert_bounds() -> None:
    config = {
        "num_hidden_layers": 43,
        "dspark_target_layer_ids": [40, 41, 42],
        "n_routed_experts": 256,
    }
    MODULE.validate_case_geometry([("layers", 42, 255), ("mtp", 2, 0)], config)
    with pytest.raises(MODULE.ValidationError, match="outside"):
        MODULE.validate_case_geometry([("mtp", 3, 0)], config)
