from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import validate_ds4_exl3_retained_native as validator  # noqa: E402


def test_legacy_retained_native_report_preserves_declared_recipe(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "legacy"
    artifact.mkdir()
    (artifact / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "exl3",
                    "ds4rt": {
                        "recipe": (
                            "deepseek_v4_exl3_trellis_2bpw_v4_flash_natural_route"
                        )
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    recipe, output = validator.artifact_recipe_and_output(artifact, None)

    assert recipe == "deepseek_v4_exl3_trellis_2bpw_v4_flash_natural_route"
    assert output == artifact / "ds4rt-exl3-retained-native.json"


def test_gptqmodel_retained_native_report_must_remain_external(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "gptqmodel"
    artifact.mkdir()
    (artifact / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "exl3",
                    "bits": 2.0,
                    "meta": {"ds4rt_error_ledger": {}},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        validator,
        "validate_gptqmodel_publication",
        lambda *_args, **_kwargs: {},
    )

    with pytest.raises(ValueError, match="--output"):
        validator.artifact_recipe_and_output(artifact, None)
    with pytest.raises(ValueError, match="outside the immutable artifact"):
        validator.artifact_recipe_and_output(
            artifact,
            artifact / "retained-native.json",
        )

    output = tmp_path / "reports" / "retained-native.json"
    recipe, resolved = validator.artifact_recipe_and_output(artifact, output)
    assert recipe == "deepseek_v4_exl3_trellis_2bpw_v4_flash_natural_route"
    assert resolved == output


def test_gptqmodel_k3_retained_native_report_uses_k3_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "gptqmodel-k3"
    artifact.mkdir()
    (artifact / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "exl3",
                    "bits": 3.0,
                    "meta": {"ds4rt_error_ledger": {}},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        validator,
        "validate_gptqmodel_publication",
        lambda *_args, **_kwargs: {},
    )

    recipe, _ = validator.artifact_recipe_and_output(
        artifact, tmp_path / "reports" / "retained-native.json"
    )

    assert recipe == "deepseek_v4_exl3_trellis_3bpw_v4_flash_natural_route"


@pytest.mark.parametrize(
    ("quantization_config", "expected_layout", "expected_bits"),
    [
        (
            {
                "quant_method": "exl3",
                "bits": 2.0,
                "meta": {"ds4rt_error_ledger": {}},
            },
            validator.EXPERT_TENSOR_LAYOUT_GPTQMODEL,
            2,
        ),
        (
            {
                "quant_method": "exl3",
                "bits": 3.0,
                "meta": {"ds4rt_error_ledger": {}},
            },
            validator.EXPERT_TENSOR_LAYOUT_GPTQMODEL,
            3,
        ),
        (
            {
                "quant_method": "exl3",
                "ds4rt": {"recipe": "legacy"},
            },
            validator.EXPERT_TENSOR_LAYOUT_CHECKPOINT_NATIVE,
            2,
        ),
    ],
)
def test_retained_native_plan_uses_the_published_expert_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quantization_config: dict,
    expected_layout: str,
    expected_bits: int,
) -> None:
    captured = {}
    expected = object()

    def build(snapshot: Path, *, expert_tensor_layout: str, exl3_bits: int):
        captured["snapshot"] = snapshot
        captured["layout"] = expert_tensor_layout
        captured["bits"] = exl3_bits
        return expected

    monkeypatch.setattr(validator, "build_artifact_plan", build)
    native = tmp_path / "native"

    actual = validator.retained_native_plan(
        native,
        {"quantization_config": quantization_config},
    )

    assert actual is expected
    assert captured == {
        "snapshot": native,
        "layout": expected_layout,
        "bits": expected_bits,
    }
