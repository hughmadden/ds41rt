from __future__ import annotations

from pathlib import Path
import sys

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from compose_ds4_exl3_k2_dspark_control import (  # noqa: E402
    combine_quantization_config,
    final_draft_shard,
)


def quant_config(bits: int) -> dict:
    storage = {}
    for prefix in ("model.layers.0", "model.layers.1", "mtp.0"):
        for expert_id in range(2):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                storage[f"{prefix}.mlp.experts.{expert_id}.{projection}"] = {
                    "bits_per_weight": bits,
                }
    return {
        "bits": float(bits),
        "method": "exl3",
        "quant_method": "exl3",
        "tensor_storage": storage,
    }


def test_combiner_keeps_target_base_and_forces_dspark_to_k2() -> None:
    target = quant_config(3)
    k2 = quant_config(2)

    mixed, plan = combine_quantization_config(
        target=target,
        k2=k2,
        hidden_layers=2,
        mtp_layers=1,
        experts=2,
    )

    assert plan.k3_experts_by_layer == ((0, 1), (0, 1), ())
    assert plan.realized_bpw == pytest.approx(2 + 4 / 6)
    assert mixed["bits"] == pytest.approx(2 + 4 / 6)
    assert (
        mixed["tensor_storage"]["model.layers.0.mlp.experts.0.gate_proj"][
            "bits_per_weight"
        ]
        == 3
    )
    assert (
        mixed["tensor_storage"]["mtp.0.mlp.experts.0.gate_proj"][
            "bits_per_weight"
        ]
        == 2
    )


def test_final_draft_shard_requires_complete_standard_sequence() -> None:
    weight_map = {
        "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
        "mtp.0.mlp.experts.0.gate_proj.trellis": "model-00002-of-00002.safetensors",
    }

    final_name, final_names = final_draft_shard(weight_map)
    assert final_name == "model-00002-of-00002.safetensors"
    assert final_names == ("mtp.0.mlp.experts.0.gate_proj.trellis",)

    weight_map["mtp.0.norm.weight"] = final_name
    with pytest.raises(ValueError, match="not exclusively routed dSpark experts"):
        final_draft_shard(weight_map)
