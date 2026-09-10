from __future__ import annotations

import json

import pytest

from ds41rt_runtime.native_experts import (
    EXPERT_TP_WORLD_SIZE,
    checkpoint_tensor_names,
    load_native_expert_tp_layer,
    read_native_expert_config,
    replicated_expert_ids,
    tp_intermediate_slice,
)


def _write_config(path):
    path.write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "expert_dtype": "fp4",
                "quantization_config": {"quant_method": "fp8"},
                "hidden_size": 4096,
                "moe_intermediate_size": 2048,
                "num_hidden_layers": 43,
                "dspark_target_layer_ids": [40, 41, 42],
                "n_routed_experts": 256,
                "num_experts_per_tok": 6,
                "swiglu_limit": 10.0,
            }
        ),
        encoding="utf-8",
    )


def test_native_checkpoint_names_and_w13_semantics(tmp_path):
    _write_config(tmp_path / "config.json")
    config = read_native_expert_config(tmp_path)

    target = checkpoint_tensor_names(config, 42, 17)
    assert target["gate_weight"] == "layers.42.ffn.experts.17.w1.weight"
    assert target["up_weight"] == "layers.42.ffn.experts.17.w3.weight"
    assert target["down_scale"] == "layers.42.ffn.experts.17.w2.scale"

    dspark = checkpoint_tensor_names(config, 45, 17)
    assert dspark["up_weight"] == "mtp.2.ffn.experts.17.w3.weight"


def test_tp4_replicates_expert_ids_and_partitions_intermediate_dimension():
    expert_ids = replicated_expert_ids(256)
    slices = [
        tp_intermediate_slice(2048, rank)
        for rank in range(EXPERT_TP_WORLD_SIZE)
    ]

    assert expert_ids == tuple(range(256))
    assert all(tp_slice.size == 512 for tp_slice in slices)
    assert [(tp_slice.start, tp_slice.stop) for tp_slice in slices] == [
        (0, 512),
        (512, 1024),
        (1024, 1536),
        (1536, 2048),
    ]
    assert [(tp_slice.packed_byte_start, tp_slice.packed_byte_stop) for tp_slice in slices] == [
        (0, 256),
        (256, 512),
        (512, 768),
        (768, 1024),
    ]
    assert [(tp_slice.scale_start, tp_slice.scale_stop) for tp_slice in slices] == [
        (0, 16),
        (16, 32),
        (32, 48),
        (48, 64),
    ]


def test_pro_tp4_intermediate_geometry_is_k32_aligned():
    slices = [tp_intermediate_slice(3072, rank) for rank in range(4)]
    assert all(tp_slice.size == 768 for tp_slice in slices)
    assert slices[-1].scale_stop == 96


def test_tp_slice_rejects_non_k32_aligned_partition():
    with pytest.raises(ValueError, match="K/32 scale block"):
        tp_intermediate_slice(2050, 0)


def test_native_loader_rejects_gpu1_before_importing_torch(tmp_path):
    with pytest.raises(ValueError, match="physical GPU 0"):
        load_native_expert_tp_layer(
            tmp_path,
            0,
            tp_rank=0,
            expert_ids=[0],
            device="cuda:1",
        )
