from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from compare_ds4_flash_exl3_control import (  # noqa: E402
    control_tensor_prefix,
    output_metrics,
    parse_expert_ids,
    read_control_config,
    resolve_control_layout,
    stratified_expert_ids,
)
from ds4rt_runtime.native_experts import NativeExpertConfig  # noqa: E402


def native_config() -> NativeExpertConfig:
    return NativeExpertConfig(
        hidden_size=4096,
        intermediate_size=2048,
        num_hidden_layers=40,
        dspark_blocks=1,
        global_experts=256,
        top_k=6,
        swiglu_limit=10.0,
    )


def test_control_stratifies_and_validates_expert_ids() -> None:
    assert stratified_expert_ids(256) == (0, 51, 102, 153, 204, 255)
    assert parse_expert_ids("auto", 256) == (0, 51, 102, 153, 204, 255)
    assert parse_expert_ids("0,1,2,3,4,5", 256) == (0, 1, 2, 3, 4, 5)
    with pytest.raises(ValueError, match="six unique sorted"):
        parse_expert_ids("0,1,2,3,5,4", 256)


def test_control_config_requires_matching_exl3_codebook_and_deepseek_v4(
    tmp_path: Path,
) -> None:
    config = native_config()
    raw = {
        "model_type": "deepseek_v4",
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
        "num_hidden_layers": 40,
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "swiglu_limit": 10.0,
        "quantization_config": {
            "quant_method": "exl3",
            "version": "1.3.0",
            "bits": 2.04,
            "codebook": "mul1",
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    _, quant = read_control_config(tmp_path, config)
    assert quant["bits"] == 2.04

    raw["quantization_config"]["codebook"] = "mcg"
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    _, quant = read_control_config(tmp_path, config)
    assert quant["codebook"] == "mcg"

    raw["quantization_config"]["codebook"] = "unknown"
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="mcg or mul1"):
        read_control_config(tmp_path, config)


def test_control_tensor_prefix_and_metrics() -> None:
    assert (
        control_tensor_prefix(0, 255, "w3")
        == "layers.0.ffn.experts.255.w3"
    )
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    exact = output_metrics(reference, reference)
    assert exact["cosine"] == pytest.approx(1.0)
    assert exact["relative_l2"] == 0.0


def test_control_layout_detects_unsharded_and_strict_tp4(tmp_path: Path) -> None:
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps({"weight_map": {"tensor": "model.safetensors"}}),
        encoding="utf-8",
    )
    unsharded = resolve_control_layout(
        tmp_path,
        {"codebook": "mul1"},
        layer_id=0,
        intermediate_size=2048,
        expert_count=256,
    )
    assert unsharded.name == "unsharded_full_model"
    assert unsharded.ranks == 1
    assert unsharded.tensor_prefix(0, 1, "w2", 0) == "layers.0.ffn.experts.1.w2"

    index.unlink()
    files = [
        {
            "name": f"exl3-layer-000-tp4-rank{rank}.safetensors",
            "sha256": f"{rank + 1:064x}",
        }
        for rank in range(4)
    ]
    (tmp_path / "EXL3_MANIFEST.json").write_text(
        json.dumps(
            {
                "tensor_parallel_size": 4,
                "expert_parallel": False,
                "source_revision": "native-revision",
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    sliced = resolve_control_layout(
        tmp_path,
        {"codebook": "mcg", "version": "rank-sliced-deepseek-v4-v1"},
        layer_id=0,
        intermediate_size=2048,
        expert_count=2,
        native_revision="native-revision",
    )
    assert sliced.name == "strict_tp4_rank_sliced"
    assert sliced.ranks == 4
    assert sliced.local_intermediate_size == 512
    assert (
        sliced.tensor_prefix(0, 1, "w2", 3)
        == "layers.0.ffn.experts.1.w2.rank3"
    )
