from __future__ import annotations

import hashlib
import json

import pytest

from ds41rt_runtime.exl3_experts import (
    EXL3_W13_PROJECTION_LAYOUT,
    checkpoint_tensor_names,
    load_exl3_expert_tp_layer,
    read_exl3_expert_config,
)
from ds41rt_runtime.exl3_quantizer import (
    EXL3_ACTIVATION_RECIPE,
    EXL3_FORCED_ACTIVATION_RECIPE,
    EXL3_ROUTE_REPLAY_ARTIFACT_FILENAME,
    exl3_quantization_config,
)
from ds41rt_runtime.native_experts import (
    GPTQMODEL_EXPERT_LAYOUT,
    NativeExpertConfig,
)


def _write_config(path):
    path.write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "hidden_size": 4096,
                "moe_intermediate_size": 2048,
                "num_hidden_layers": 43,
                "dspark_target_layer_ids": [40, 41, 42],
                "n_routed_experts": 256,
                "num_experts_per_tok": 6,
                "swiglu_limit": 10.0,
                "quantization_config": exl3_quantization_config(
                    calibration_rows=512,
                    seed=17,
                    swiglu_limit=10.0,
                ),
            }
        ),
        encoding="utf-8",
    )


def test_exl3_checkpoint_names_preserve_w13_and_dspark_storage_semantics(tmp_path):
    _write_config(tmp_path / "config.json")
    config = read_exl3_expert_config(tmp_path)

    target = checkpoint_tensor_names(config, 42, 17)
    assert target["gate_trellis"] == "layers.42.ffn.experts.17.w1.trellis"
    assert target["up_suh"] == "layers.42.ffn.experts.17.w3.suh"
    assert target["down_svh"] == "layers.42.ffn.experts.17.w2.svh"

    dspark = checkpoint_tensor_names(config, 45, 17)
    assert dspark["up_trellis"] == "mtp.2.ffn.experts.17.w3.trellis"


def test_gptqmodel_exl3_checkpoint_names_use_published_module_namespace():
    config = NativeExpertConfig(
        hidden_size=4096,
        intermediate_size=2048,
        num_hidden_layers=43,
        dspark_blocks=3,
        global_experts=256,
        top_k=6,
        swiglu_limit=10.0,
        expert_tensor_layout=GPTQMODEL_EXPERT_LAYOUT,
    )

    target = checkpoint_tensor_names(config, 42, 17)
    assert (
        target["gate_trellis"]
        == "model.layers.42.mlp.experts.17.gate_proj.trellis"
    )
    assert target["up_suh"] == "model.layers.42.mlp.experts.17.up_proj.suh"
    assert (
        target["down_svh"]
        == "model.layers.42.mlp.experts.17.down_proj.svh"
    )

    dspark = checkpoint_tensor_names(config, 45, 17)
    assert dspark["up_trellis"] == "mtp.2.mlp.experts.17.up_proj.trellis"


def test_projection_major_exl3_uses_kernel_native_gate_then_up_planes():
    assert EXL3_W13_PROJECTION_LAYOUT == (("gate", 0, 0), ("up", 1, 1))


def test_exl3_config_fails_closed_on_noncalibrated_or_non_tp4_artifact(tmp_path):
    _write_config(tmp_path / "config.json")
    raw = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    raw["quantization_config"]["ds41rt"]["calibrated"] = False
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="calibrated"):
        read_exl3_expert_config(tmp_path)

    raw["quantization_config"]["ds41rt"]["calibrated"] = True
    raw["quantization_config"]["ds41rt"]["expert_tp_world_size"] = 2
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="expert_tp_world_size"):
        read_exl3_expert_config(tmp_path)


def test_exl3_config_rejects_pre_clamp_calibration_contract(tmp_path):
    _write_config(tmp_path / "config.json")
    raw = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    del raw["quantization_config"]["calibration"]["gate_clamp"]
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="bounded DeepSeek SwiGLU"):
        read_exl3_expert_config(tmp_path)


def test_exl3_config_accepts_checkpoint_bound_forced_activation_pilot(tmp_path):
    _write_config(tmp_path / "config.json")
    raw = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    raw["quantization_config"] = exl3_quantization_config(
        calibration_rows=512,
        seed=17,
        swiglu_limit=10.0,
        recipe=EXL3_FORCED_ACTIVATION_RECIPE,
        calibration_override={
            "method": "layerwise_native_activation_forced_down",
            "device": "cuda:0",
            "rows": 59_138,
            "seed": 17,
            "hessian": "native_sample_covariance_with_forced_down_shrinkage",
            "distribution": "checkpoint_bound_native_expert_inputs",
            "activation_corpus_sha256": "a" * 64,
            "activation_base_layers": 43,
            "forced_down_rows_per_expert": 512,
            "forced_down_expert_weight": 0.25,
            "forced_down_pool_weight": 0.75,
            "mtp_calibration": "analytic_identity_pilot_only",
        },
    )
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")

    config = read_exl3_expert_config(tmp_path)
    assert config.hidden_size == 4096

    raw["quantization_config"]["calibration"]["activation_base_layers"] = 42
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint-bound"):
        read_exl3_expert_config(tmp_path)


def test_exl3_config_accepts_checkpoint_bound_natural_route_calibration(tmp_path):
    _write_config(tmp_path / "config.json")
    replay_payload = b'{"schema":"ds41rt-flash-route-replay-v1","status":"exact"}\n'
    (tmp_path / EXL3_ROUTE_REPLAY_ARTIFACT_FILENAME).write_bytes(replay_payload)
    raw = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    raw["quantization_config"] = exl3_quantization_config(
        calibration_rows=1024,
        seed=17,
        swiglu_limit=10.0,
        recipe=EXL3_ACTIVATION_RECIPE,
        calibration_override={
            "method": "layerwise_native_natural_routes",
            "device": "cuda:0",
            "rows": 43_691,
            "seed": 17,
            "hessian": "per_expert_natural_route_gate_squared_covariance",
            "distribution": "checkpoint_bound_native_expert_inputs",
            "activation_corpus_sha256": "b" * 64,
            "activation_base_layers": 43,
            "natural_routing": True,
            "forced_expert_activation": False,
            "route_gate_weighting": "squared_unit_rms",
            "minimum_natural_routes_per_expert": 1024,
            "route_replay_report": EXL3_ROUTE_REPLAY_ARTIFACT_FILENAME,
            "route_replay_report_sha256": hashlib.sha256(replay_payload).hexdigest(),
            "mtp_calibration": "analytic_identity_pilot_only",
        },
    )
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")

    assert read_exl3_expert_config(tmp_path).hidden_size == 4096

    raw["quantization_config"]["calibration"]["forced_expert_activation"] = True
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint-bound"):
        read_exl3_expert_config(tmp_path)

    raw["quantization_config"]["calibration"]["forced_expert_activation"] = False
    raw["quantization_config"]["calibration"][
        "minimum_natural_routes_per_expert"
    ] = 1023
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint-bound"):
        read_exl3_expert_config(tmp_path)


def test_exl3_loader_rejects_gpu1_before_importing_torch(tmp_path):
    with pytest.raises(ValueError, match="physical GPU 0"):
        load_exl3_expert_tp_layer(
            tmp_path,
            0,
            tp_rank=0,
            expert_ids=[0],
            device="cuda:1",
        )
