from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from ds4rt_runtime.exl3_quantizer import (
    ACTIVATION_CORPUS_SCHEMA,
    EXL3_ACTIVATION_RECIPE,
    EXL3_RECIPE,
    EXPERT_TENSOR_LAYOUT_GPTQMODEL,
    MIN_NATURAL_ROUTES_PER_EXPERT,
    ROUTED_ACTIVATION_CORPUS_SCHEMA,
    ROUTED_ACTIVATION_ROUTE_RECORD,
    ROUTED_ACTIVATION_ROUTE_RECORD_FORMAT,
    ROUTE_REPLAY_REPORT_FILENAME,
    ROUTE_REPLAY_REPORT_SCHEMA,
    RoutedActivationLayer,
    SafetensorsArtifactWriter,
    accumulate_activation_hessian,
    accumulate_routed_activation_hessian,
    analytic_isotropic_hessian,
    blended_forced_down_hessian,
    build_artifact_plan,
    calibration_report_for_run,
    deepseek_v4_swiglu_activation,
    dequantize_native_fp4_projection,
    deterministic_expert_row_indices,
    exl3_quantization_config,
    gptqmodel_tensor_storage_for_plan,
    load_activation_corpus,
    load_activation_layer_samples,
    load_routed_activation_layer,
    routed_expert_activation_rows,
    plan_summary,
    quantization_progress,
    simulated_hidden_states,
    validate_projection_quantization_diagnostic,
    verify_retained_native_tensors,
)


def make_activation_corpus(
    tmp_path: Path,
    *,
    snapshot: Path,
    samples: torch.Tensor,
    routed: bool = False,
    minimum_natural_routes_per_expert: int = MIN_NATURAL_ROUTES_PER_EXPERT,
) -> Path:
    root = tmp_path / "activation-corpus"
    capture = root / "captures" / "000-test" / "layer_00_rows_3_expert_input.bf16"
    capture.parent.mkdir(parents=True)
    payload = samples.to(torch.bfloat16).view(torch.int16).numpy().tobytes()
    capture.write_bytes(payload)
    corpus_path = root / "source.jsonl"
    corpus_path.write_text('{"id":"test","prompt":"test"}\n', encoding="utf-8")
    capture_record = {
        "path": str(capture.relative_to(root)),
        "layer_id": 0,
        "rows": int(samples.shape[0]),
        "hidden_size": int(samples.shape[1]),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if routed:
        route_path = capture.with_name(
            capture.name.removesuffix(".bf16") + "_routes_u16_f32.bin"
        )
        route_payload = b"".join(
            ROUTED_ACTIVATION_ROUTE_RECORD.pack(0, 1.5)
            for _ in range(int(samples.shape[0]))
        )
        route_path.write_bytes(route_payload)
        capture_record.update(
            {
                "capture_id": "test",
                "route_path": str(route_path.relative_to(root)),
                "route_bytes": len(route_payload),
                "route_sha256": hashlib.sha256(route_payload).hexdigest(),
                "route_record_format": ROUTED_ACTIVATION_ROUTE_RECORD_FORMAT,
                "routes_per_row": 1,
            }
        )
    manifest = {
        "schema": (
            ROUTED_ACTIVATION_CORPUS_SCHEMA if routed else ACTIVATION_CORPUS_SCHEMA
        ),
        "checkpoint": str(snapshot.resolve()),
        "corpus_path": str(corpus_path.resolve()),
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "hidden_size": int(samples.shape[1]),
        "layer_count": 1,
        "dspark_layer_count": 0,
        **(
            {
                "routed_experts": 1,
                "top_k": 1,
                "routed_scaling_factor": 1.5,
                "route_record_format": ROUTED_ACTIVATION_ROUTE_RECORD_FORMAT,
                "retention": {
                    "policy": "first_joint_rows_until_per_layer_expert_route_quota",
                    "routes_per_expert_target": MIN_NATURAL_ROUTES_PER_EXPERT,
                    "minimum_natural_routes_per_expert": (
                        minimum_natural_routes_per_expert
                    ),
                },
                "route_distribution": [
                    {
                        "layer_id": 0,
                        "rows": int(samples.shape[0]),
                        "routes": int(samples.shape[0]),
                        "zero_hit_experts": 0,
                        "min_hits": int(samples.shape[0]),
                        "p50_hits": int(samples.shape[0]),
                        "max_hits": int(samples.shape[0]),
                        "expert_route_counts": [int(samples.shape[0])],
                    }
                ],
                "retained_route_distribution": [
                    {
                        "layer_id": 0,
                        "rows": int(samples.shape[0]),
                        "routes": int(samples.shape[0]),
                        "zero_hit_experts": 0,
                        "min_hits": int(samples.shape[0]),
                        "p50_hits": int(samples.shape[0]),
                        "max_hits": int(samples.shape[0]),
                        "expert_route_counts": [int(samples.shape[0])],
                    }
                ],
                "summary": {
                    "weakest_natural_route_coverage": int(samples.shape[0]),
                    "covered_layers": [0],
                },
            }
            if routed
            else {}
        ),
        "prompts": [
            {
                "capture_files": [capture_record]
            }
        ],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    if routed:
        replay = {
            "schema": ROUTE_REPLAY_REPORT_SCHEMA,
            "status": "exact",
            "metadata_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "snapshot": str(snapshot.resolve()),
            "layers": [],
            "max_capture_rows_per_layer": -1,
            "repeats": 2,
            "rows": 0,
            "routes": 0,
            "ranked_mismatch_rows": 0,
            "set_mismatch_rows": 0,
            "route_entry_mismatches": 0,
            "weight_bit_mismatches": 0,
            "native_library_sha256": "0" * 64,
            "layer_summaries": [],
        }
        (root / ROUTE_REPLAY_REPORT_FILENAME).write_text(
            json.dumps(replay), encoding="utf-8"
        )
    return root


def test_quantization_progress_uses_durable_layer_timings_for_eta() -> None:
    progress = quantization_progress(
        [
            {"layer_id": 0, "elapsed_seconds": 4.0},
            {"layer_id": 1, "elapsed_seconds": 6.0},
        ],
        total_layers=5,
    )

    assert progress == {
        "completed_layers": 2,
        "total_layers": 5,
        "completed_fraction": 0.4,
        "timed_layers": 2,
        "quantized_layer_seconds": 10.0,
        "mean_layer_seconds": 5.0,
        "estimated_remaining_seconds": 15.0,
    }


def test_routed_expert_hessian_uses_only_natural_rows_and_squared_gates() -> None:
    layer = RoutedActivationLayer(
        samples=torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [50.0, 60.0]],
            dtype=torch.bfloat16,
        ),
        expert_ids=torch.tensor([[0, 1], [2, 0], [1, 2]]),
        gate_weights=torch.tensor([[0.75, 0.25], [0.75, 0.25], [0.6, 0.4]]),
    )

    samples, gates = routed_expert_activation_rows(layer, expert_id=0)
    hessian = accumulate_routed_activation_hessian(
        samples,
        gates,
        key="expert.0.w1",
        device="cpu",
    )

    assert torch.equal(samples, layer.samples[:2])
    assert torch.equal(gates, torch.tensor([0.75, 0.25]))
    normalized = gates / gates.square().mean().sqrt()
    expected = (samples.float() * normalized[:, None]).T @ (
        samples.float() * normalized[:, None]
    )
    assert hessian["count"] == 2
    assert torch.allclose(hessian["H"], expected)


def test_quantization_progress_handles_legacy_untimed_resume_layers() -> None:
    progress = quantization_progress(
        [{"layer_id": 0}, {"layer_id": 1, "elapsed_seconds": 3.0}],
        total_layers=3,
    )

    assert progress["completed_layers"] == 2
    assert progress["timed_layers"] == 1
    assert progress["estimated_remaining_seconds"] == 3.0


def test_projection_diagnostic_rejects_scale_boundaries_and_corrupt_proxy() -> None:
    validate_projection_quantization_diagnostic(
        name="layers.0.ffn.experts.0.w1",
        proxy_error=0.07,
        g_scale=0.99,
    )
    for g_scale in (0.01, 2.05, float("nan")):
        with pytest.raises(RuntimeError, match="global scale"):
            validate_projection_quantization_diagnostic(
                name="layers.0.ffn.experts.0.w1",
                proxy_error=0.07,
                g_scale=g_scale,
            )
    for proxy_error in (-0.1, 0.5, float("inf")):
        with pytest.raises(RuntimeError, match="proxy error"):
            validate_projection_quantization_diagnostic(
                name="layers.0.ffn.experts.0.w1",
                proxy_error=proxy_error,
                g_scale=0.99,
            )


@pytest.mark.parametrize("total_layers", (0, -1))
def test_quantization_progress_rejects_empty_plan(total_layers: int) -> None:
    with pytest.raises(ValueError, match="positive layer count"):
        quantization_progress([], total_layers=total_layers)


def make_native_snapshot(tmp_path: Path) -> Path:
    snapshot = tmp_path / "native"
    snapshot.mkdir(parents=True)
    config = {
        "model_type": "deepseek_v4",
        "architectures": ["DeepseekV4ForCausalLM"],
        "hidden_size": 128,
        "moe_intermediate_size": 512,
        "num_hidden_layers": 1,
        "dspark_target_layer_ids": [],
        "n_routed_experts": 1,
        "num_experts_per_tok": 1,
        "routed_scaling_factor": 1.5,
        "swiglu_limit": 10.0,
        "rms_norm_eps": 1.0e-6,
        "expert_dtype": "fp4",
        "quantization_config": {"quant_method": "fp8"},
    }
    (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (snapshot / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    tensors = {"norm.weight": torch.ones(128, dtype=torch.float16)}
    for stem in ("w1", "w3"):
        base = f"layers.0.ffn.experts.0.{stem}"
        tensors[f"{base}.weight"] = torch.full((512, 64), 0x21, dtype=torch.int8)
        tensors[f"{base}.scale"] = torch.full((512, 4), 127, dtype=torch.uint8)
    base = "layers.0.ffn.experts.0.w2"
    tensors[f"{base}.weight"] = torch.full((128, 256), 0x21, dtype=torch.int8)
    tensors[f"{base}.scale"] = torch.full((128, 16), 127, dtype=torch.uint8)
    shard_name = "model.safetensors"
    save_file(tensors, snapshot / shard_name)
    index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
        "weight_map": {name: shard_name for name in tensors},
    }
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    return snapshot


def test_flash_plan_replaces_only_routed_experts_with_k2_trellis(tmp_path: Path) -> None:
    plan = build_artifact_plan(make_native_snapshot(tmp_path), max_shard_bytes=32 * 1024)
    names = {tensor.name for tensor in plan.tensors}
    assert "norm.weight" in names
    assert "layers.0.ffn.experts.0.w1.weight" not in names
    assert "layers.0.ffn.experts.0.w1.scale" not in names
    trellis = next(
        tensor
        for tensor in plan.tensors
        if tensor.name == "layers.0.ffn.experts.0.w1.trellis"
    )
    assert trellis.dtype == "I16"
    assert trellis.shape == (8, 32, 32)
    summary = plan_summary(plan)
    assert summary["strict_expert_tp"] == 4
    assert summary["local_intermediate_size"] == 128
    assert summary["generated_exl3_tensors"] == 12
    assert summary["quantization_scope"] == "routed_experts_only"
    assert summary["retained_native_tensor_bytes"] == plan.source_bytes
    assert summary["routed_expert_exl3_bytes"] == plan.trellis_bytes


def test_gptqmodel_hybrid_plan_keeps_native_coordinator_namespace(tmp_path: Path) -> None:
    plan = build_artifact_plan(
        make_native_snapshot(tmp_path),
        max_shard_bytes=32 * 1024,
        expert_tensor_layout=EXPERT_TENSOR_LAYOUT_GPTQMODEL,
    )
    names = {tensor.name for tensor in plan.tensors}

    assert plan.expert_tensor_layout == EXPERT_TENSOR_LAYOUT_GPTQMODEL
    assert "norm.weight" in names
    assert "layers.0.ffn.experts.0.w1.weight" not in names
    assert "layers.0.ffn.experts.0.w1.scale" not in names
    assert "layers.0.ffn.experts.0.w1.trellis" not in names
    assert "model.layers.0.mlp.experts.0.gate_proj.trellis" in names
    assert "model.layers.0.mlp.experts.0.up_proj.trellis" in names
    assert "model.layers.0.mlp.experts.0.down_proj.trellis" in names
    assert plan_summary(plan)["expert_tensor_layout"] == "gptqmodel"


def test_gptqmodel_hybrid_plan_supports_k3_geometry(tmp_path: Path) -> None:
    plan = build_artifact_plan(
        make_native_snapshot(tmp_path),
        max_shard_bytes=32 * 1024,
        expert_tensor_layout=EXPERT_TENSOR_LAYOUT_GPTQMODEL,
        exl3_bits=3,
    )
    trellis = next(
        tensor
        for tensor in plan.generated_tensors
        if tensor.name == "model.layers.0.mlp.experts.0.gate_proj.trellis"
    )
    assert trellis.shape == (8, 32, 48)
    assert plan_summary(plan)["trellis_bits"] == 3
    storage = gptqmodel_tensor_storage_for_plan(plan)
    assert storage["model.layers.0.mlp.experts.0.gate_proj"]["bits_per_weight"] == 3


def test_hybrid_plan_rejects_unknown_expert_namespace(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported EXL3 expert tensor layout"):
        build_artifact_plan(
            make_native_snapshot(tmp_path),
            expert_tensor_layout="transformed-coordinator",
        )


def gptqmodel_quant_config_for_plan(plan) -> dict[str, object]:
    dtype_names = {"I16": "int16", "F16": "float16", "I32": "int32"}
    by_base: dict[str, dict[str, object]] = {}
    for tensor in plan.generated_tensors:
        base, _, _ = tensor.name.rpartition(".")
        entry = by_base.setdefault(
            base,
            {
                "quant_format": "exl3",
                "bits_per_weight": 2,
                "mcg_multiplier": 0xCBAC1FED,
                "stored_tensors": {},
            },
        )
        entry["stored_tensors"][tensor.name] = {
            "shape": list(tensor.shape),
            "torch_dtype": dtype_names[tensor.dtype],
        }
    return {
        "quant_method": "exl3",
        "method": "exl3",
        "format": "exl3",
        "checkpoint_format": "exl3",
        "bits": 2.0,
        "tensor_storage": by_base,
    }


def test_streaming_writer_preserves_gptqmodel_storage_contract(tmp_path: Path) -> None:
    snapshot = make_native_snapshot(tmp_path)
    output = tmp_path / "gptqmodel-hybrid"
    plan = build_artifact_plan(
        snapshot,
        max_shard_bytes=32 * 1024,
        expert_tensor_layout=EXPERT_TENSOR_LAYOUT_GPTQMODEL,
    )
    qconfig = gptqmodel_quant_config_for_plan(plan)
    writer = SafetensorsArtifactWriter(
        plan,
        output,
        calibration_rows=128,
        seed=7,
        quant_config_override=qconfig,
        quant_config_filename="quantize_config.json",
    )
    writer.copy_native_tensors(checkpoint_interval=1)
    dtype_map = {"I16": torch.int16, "F16": torch.float16, "I32": torch.int32}
    for tensor in plan.generated_tensors:
        value = torch.zeros(tensor.shape, dtype=dtype_map[tensor.dtype])
        if tensor.name.endswith(".mcg"):
            value = torch.tensor(0xCBAC1FED, dtype=torch.uint32).view(torch.int32)
        writer.write_generated_tensor(tensor.name, value)
    writer.finish({"schema": "test", "layers": []})

    assert not (output / "quantization_config.json").exists()
    assert json.loads((output / "quantize_config.json").read_text()) == qconfig
    config = json.loads((output / "config.json").read_text())
    assert config["quantization_config"] == qconfig
    index = json.loads((output / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    assert "norm.weight" in index
    assert "model.layers.0.mlp.experts.0.gate_proj.trellis" in index
    assert not any(".ffn.experts." in name for name in index)


def test_streaming_writer_rejects_incomplete_gptqmodel_storage(tmp_path: Path) -> None:
    plan = build_artifact_plan(
        make_native_snapshot(tmp_path),
        max_shard_bytes=32 * 1024,
        expert_tensor_layout=EXPERT_TENSOR_LAYOUT_GPTQMODEL,
    )
    qconfig = gptqmodel_quant_config_for_plan(plan)
    qconfig["tensor_storage"].pop(
        "model.layers.0.mlp.experts.0.gate_proj"
    )
    with pytest.raises(ValueError, match="differs from the artifact plan"):
        SafetensorsArtifactWriter(
            plan,
            tmp_path / "invalid-hybrid",
            calibration_rows=128,
            seed=7,
            quant_config_override=qconfig,
            quant_config_filename="quantize_config.json",
        )


def test_streaming_writer_publishes_standard_hybrid_hf_snapshot(tmp_path: Path) -> None:
    snapshot = make_native_snapshot(tmp_path)
    output = tmp_path / "exl3"
    plan = build_artifact_plan(snapshot, max_shard_bytes=32 * 1024)
    writer = SafetensorsArtifactWriter(
        plan,
        output,
        calibration_rows=128,
        seed=7,
    )
    assert not (output / "config.json").exists()
    writer.copy_native_tensors(checkpoint_interval=1)
    dtype_map = {"I16": torch.int16, "F16": torch.float16, "I32": torch.int32}
    for tensor in plan.generated_tensors:
        value = torch.zeros(tensor.shape, dtype=dtype_map[tensor.dtype])
        if tensor.name.endswith(".mcg"):
            value = torch.tensor(0xCBAC1FED, dtype=torch.uint32).view(torch.int32)
        writer.write_generated_tensor(tensor.name, value)
    evidence = tmp_path / "private-ledger.jsonl"
    evidence.write_text('{"record":1}\n', encoding="utf-8")
    evidence.chmod(0o600)
    writer.finish(
        {
            "schema": "ds4rt-exl3-calibration-report-v1",
            "recipe": EXL3_RECIPE,
            "layers": [],
        },
        evidence_files={"published-ledger.jsonl": evidence},
    )

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["quantization_config"]["ds4rt"]["recipe"] == EXL3_RECIPE
    calibration = config["quantization_config"]["calibration"]
    assert calibration["method"] == "layerwise_simulated_isotropic"
    assert calibration["hessian"] == "analytic_identity"
    assert calibration["distribution"] == "rms_normalized_isotropic"
    assert calibration["activation"] == "silu"
    assert calibration["swiglu_limit"] == 10.0
    assert calibration["gate_clamp"] == [None, 10.0]
    assert calibration["up_clamp"] == [-10.0, 10.0]
    assert (output / "published-ledger.jsonl").stat().st_mode & 0o777 == 0o644
    index = json.loads(
        (output / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    assert "norm.weight" in index
    assert "layers.0.ffn.experts.0.w1.trellis" in index
    assert "layers.0.ffn.experts.0.w1.weight" not in index
    with safe_open(output / index["norm.weight"], framework="pt") as shard:
        assert torch.equal(shard.get_tensor("norm.weight"), torch.ones(128, dtype=torch.float16))
    trellis_name = "layers.0.ffn.experts.0.w1.trellis"
    with safe_open(output / index[trellis_name], framework="pt") as shard:
        assert tuple(shard.get_tensor(trellis_name).shape) == (8, 32, 32)
    assert not (output / ".ds4rt-exl3-state.json").exists()
    integrity = json.loads(
        (output / "ds4rt-exl3-retained-native.json").read_text(encoding="utf-8")
    )
    assert integrity["quantization_scope"] == "routed_experts_only"
    assert integrity["retained_tensor_count"] == 1
    assert integrity["retained_bytes"] == 256
    assert integrity["generated_exl3_tensor_count"] == 12
    assert integrity["generated_exl3_metadata_verified"] is True
    assert integrity["generated_exl3_mcg_tensor_count"] == 3
    assert integrity["generated_exl3_mcg_markers_verified"] is True
    assert integrity["strict_tp4_source_layout"]["rank_source_bytes_per_block"] == [
        13_836
    ] * 4
    assert integrity["tensors"][0]["name"] == "norm.weight"


def test_streaming_writer_finish_reenters_after_metadata_rename(
    tmp_path: Path,
) -> None:
    snapshot = make_native_snapshot(tmp_path)
    output = tmp_path / "exl3-finish-resume"
    plan = build_artifact_plan(snapshot, max_shard_bytes=32 * 1024)
    writer = SafetensorsArtifactWriter(
        plan,
        output,
        calibration_rows=128,
        seed=7,
    )
    writer.copy_native_tensors(checkpoint_interval=1)
    dtype_map = {"I16": torch.int16, "F16": torch.float16, "I32": torch.int32}
    for tensor in plan.generated_tensors:
        value = torch.zeros(tensor.shape, dtype=dtype_map[tensor.dtype])
        if tensor.name.endswith(".mcg"):
            value = torch.tensor(0xCBAC1FED, dtype=torch.uint32).view(torch.int32)
        writer.write_generated_tensor(tensor.name, value)

    # Model the last crash window in finish(): the final HF metadata names are
    # already visible, but the durable writer state has not yet been removed.
    (output / "config.json.incomplete").replace(output / "config.json")
    (output / "model.safetensors.index.json.incomplete").replace(
        output / "model.safetensors.index.json"
    )
    writer.finish({"schema": "test", "layers": []})

    assert (output / "config.json").is_file()
    assert (output / "model.safetensors.index.json").is_file()
    assert not (output / ".ds4rt-exl3-state.json").exists()


def test_retained_native_integrity_rejects_changed_bytes(tmp_path: Path) -> None:
    snapshot = make_native_snapshot(tmp_path)
    output = tmp_path / "exl3"
    plan = build_artifact_plan(snapshot, max_shard_bytes=32 * 1024)
    writer = SafetensorsArtifactWriter(
        plan,
        output,
        calibration_rows=128,
        seed=7,
    )
    writer.copy_native_tensors(checkpoint_interval=1)

    report = verify_retained_native_tensors(plan, output, allow_incomplete=True)
    assert report["retained_tensor_count"] == 1
    assert report["generated_exl3_metadata_verified"] is True
    assert report["generated_exl3_mcg_markers_verified"] is False
    location = writer.locations["norm.weight"]
    descriptor = os.open(output / location.file_name, os.O_WRONLY)
    try:
        os.pwrite(descriptor, b"\xff", location.absolute_offset)
    finally:
        os.close(descriptor)
    with pytest.raises(ValueError, match="retained native tensor bytes changed"):
        verify_retained_native_tensors(plan, output, allow_incomplete=True)


def test_integrity_rejects_changed_generated_tensor_metadata(tmp_path: Path) -> None:
    snapshot = make_native_snapshot(tmp_path)
    output = tmp_path / "exl3"
    plan = build_artifact_plan(snapshot, max_shard_bytes=32 * 1024)
    writer = SafetensorsArtifactWriter(plan, output, calibration_rows=128, seed=7)
    trellis_name = "layers.0.ffn.experts.0.w1.trellis"
    shard = output / writer.locations[trellis_name].file_name
    with shard.open("r+b") as handle:
        header_len = int.from_bytes(handle.read(8), "little")
        header = handle.read(header_len)
        changed = header.replace(b'"dtype":"I16"', b'"dtype":"F16"', 1)
        assert changed != header
        handle.seek(8)
        handle.write(changed)
    with pytest.raises(ValueError, match="artifact tensor metadata changed"):
        verify_retained_native_tensors(plan, output, allow_incomplete=True)


def test_native_fp4_decoder_preserves_low_first_nibbles_and_e8m0_scale() -> None:
    codes = torch.tensor(list(range(16)) * 2, dtype=torch.uint8)
    packed = (codes[0::2] | (codes[1::2] << 4)).view(1, 16)
    scales = torch.tensor([[127]], dtype=torch.uint8)
    decoded = dequantize_native_fp4_projection(packed, scales)
    expected = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ]
        * 2
    ).view(32, 1)
    assert torch.equal(decoded, expected)


def test_simulated_calibration_is_deterministic_and_rms_normalized() -> None:
    first = simulated_hidden_states(
        rows=32,
        hidden_size=128,
        layer_id=3,
        seed=17,
        device=torch.device("cpu"),
        rms_norm_eps=1.0e-6,
    )
    second = simulated_hidden_states(
        rows=32,
        hidden_size=128,
        layer_id=3,
        seed=17,
        device=torch.device("cpu"),
        rms_norm_eps=1.0e-6,
    )
    assert torch.equal(first, second)
    assert torch.allclose(first.square().mean(dim=1), torch.ones(32), atol=2.0e-6)


def test_analytic_isotropic_hessian_is_full_rank_and_exact() -> None:
    calibration = analytic_isotropic_hessian(
        features=4,
        equivalent_rows=32,
        key="layers.0.ffn.experts.*.w1",
        device=torch.device("cpu"),
    )
    assert torch.equal(calibration["H"], torch.eye(4) * 32)
    assert torch.linalg.matrix_rank(calibration["H"]) == 4
    assert calibration["count"] == 32
    assert calibration["num_total"] == 128
    assert not calibration["finalized"]


def test_analytic_isotropic_hessian_rejects_empty_geometry() -> None:
    for features, rows in ((0, 32), (4, 0)):
        with pytest.raises(ValueError, match="dimensions must be positive"):
            analytic_isotropic_hessian(
                features=features,
                equivalent_rows=rows,
                key="invalid",
                device=torch.device("cpu"),
            )


def test_activation_corpus_is_exactly_checkpoint_bound_and_byte_verified(
    tmp_path: Path,
) -> None:
    snapshot = make_native_snapshot(tmp_path)
    plan = build_artifact_plan(snapshot, max_shard_bytes=32 * 1024)
    expected = torch.arange(3 * 128, dtype=torch.float32).reshape(3, 128) / 17
    root = make_activation_corpus(tmp_path, snapshot=snapshot, samples=expected)

    corpus = load_activation_corpus(root, snapshot=snapshot, shape=plan.shape)
    observed = load_activation_layer_samples(corpus, 0)

    assert corpus.checkpoint == snapshot.resolve()
    assert corpus.rows_for_layer(0) == 3
    assert torch.equal(observed, expected.to(torch.bfloat16))

    other_snapshot = make_native_snapshot(tmp_path / "other")
    with pytest.raises(ValueError, match="bound to a different checkpoint"):
        load_activation_corpus(root, snapshot=other_snapshot, shape=plan.shape)

    corpus.captures_by_layer[0][0].path.write_bytes(bytes(3 * 128 * 2))
    with pytest.raises(ValueError, match="SHA-256 changed"):
        load_activation_layer_samples(corpus, 0)


def test_routed_activation_corpus_loads_joint_router_plane_and_verifies_it(
    tmp_path: Path,
) -> None:
    snapshot = make_native_snapshot(tmp_path)
    plan = build_artifact_plan(snapshot, max_shard_bytes=32 * 1024)
    expected = torch.arange(
        MIN_NATURAL_ROUTES_PER_EXPERT * 128, dtype=torch.float32
    ).reshape(MIN_NATURAL_ROUTES_PER_EXPERT, 128) / 17
    root = make_activation_corpus(
        tmp_path, snapshot=snapshot, samples=expected, routed=True
    )

    corpus = load_activation_corpus(root, snapshot=snapshot, shape=plan.shape)
    layer = load_routed_activation_layer(corpus, 0)

    assert corpus.route_aware is True
    assert corpus.top_k == 1
    assert torch.equal(layer.samples, expected.to(torch.bfloat16))
    assert torch.equal(
        layer.expert_ids,
        torch.zeros((MIN_NATURAL_ROUTES_PER_EXPERT, 1), dtype=torch.int64),
    )
    assert torch.equal(
        layer.gate_weights,
        torch.full((MIN_NATURAL_ROUTES_PER_EXPERT, 1), 1.5),
    )

    corpus.captures_by_layer[0][0].route_path.write_bytes(
        bytes(corpus.captures_by_layer[0][0].route_nbytes)
    )
    with pytest.raises(ValueError, match="SHA-256 changed"):
        load_routed_activation_layer(corpus, 0)


def test_routed_activation_corpus_requires_exact_replay_report(tmp_path: Path) -> None:
    snapshot = make_native_snapshot(tmp_path)
    plan = build_artifact_plan(snapshot, max_shard_bytes=32 * 1024)
    samples = torch.ones((MIN_NATURAL_ROUTES_PER_EXPERT, 128))
    root = make_activation_corpus(
        tmp_path, snapshot=snapshot, samples=samples, routed=True
    )
    (root / ROUTE_REPLAY_REPORT_FILENAME).unlink()

    with pytest.raises(ValueError, match="replay report is missing"):
        load_activation_corpus(root, snapshot=snapshot, shape=plan.shape)
    corpus = load_activation_corpus(
        root,
        snapshot=snapshot,
        shape=plan.shape,
        require_exact_route_replay=False,
    )
    assert corpus.route_replay_report_path is None
    assert corpus.route_replay_report_sha256 is None


def test_routed_activation_corpus_rejects_undercovered_manifest(tmp_path: Path) -> None:
    snapshot = make_native_snapshot(tmp_path)
    plan = build_artifact_plan(snapshot, max_shard_bytes=32 * 1024)
    samples = torch.ones((MIN_NATURAL_ROUTES_PER_EXPERT - 1, 128))
    root = make_activation_corpus(
        tmp_path, snapshot=snapshot, samples=samples, routed=True
    )

    with pytest.raises(ValueError, match="does not satisfy its route floor"):
        load_activation_corpus(root, snapshot=snapshot, shape=plan.shape)


def test_routed_activation_corpus_accepts_explicit_heldout_floor(tmp_path: Path) -> None:
    snapshot = make_native_snapshot(tmp_path)
    plan = build_artifact_plan(snapshot, max_shard_bytes=32 * 1024)
    samples = torch.ones((16, 128))
    root = make_activation_corpus(
        tmp_path,
        snapshot=snapshot,
        samples=samples,
        routed=True,
        minimum_natural_routes_per_expert=0,
    )

    with pytest.raises(ValueError, match="1024-route floor"):
        load_activation_corpus(root, snapshot=snapshot, shape=plan.shape)
    corpus = load_activation_corpus(
        root,
        snapshot=snapshot,
        shape=plan.shape,
        required_natural_routes_per_expert=0,
    )
    assert corpus.minimum_natural_routes_per_expert == 0
    assert load_routed_activation_layer(corpus, 0).samples.shape == (16, 128)


def test_activation_hessian_matches_direct_covariance() -> None:
    samples = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, -1.0, 2.0], [0.5, 1.5, -2.0]],
        dtype=torch.bfloat16,
    )
    result = accumulate_activation_hessian(
        samples,
        key="layers.0.ffn.experts.*.w1",
        device=torch.device("cpu"),
        chunk_rows=2,
    )

    assert torch.equal(result["H"], samples.float().T @ samples.float())
    assert result["count"] == 3
    assert result["first_key"] == "layers.0.ffn.experts.*.w1"


def test_forced_expert_row_selection_is_deterministic_and_non_repeating() -> None:
    first = deterministic_expert_row_indices(
        total_rows=97,
        rows=64,
        layer_id=18,
        expert_id=42,
        seed=7,
    )
    repeated = deterministic_expert_row_indices(
        total_rows=97,
        rows=64,
        layer_id=18,
        expert_id=42,
        seed=7,
    )
    other = deterministic_expert_row_indices(
        total_rows=97,
        rows=64,
        layer_id=18,
        expert_id=43,
        seed=7,
    )

    assert torch.equal(first, repeated)
    assert first.unique().numel() == 64
    assert not torch.equal(first, other)


def test_forced_down_hessian_shrinks_expert_covariance_toward_equal_pool() -> None:
    activations = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    pooled = torch.tensor([[2.0, 0.5], [0.5, 3.0]])
    result = blended_forced_down_hessian(
        activations,
        pooled_covariance=pooled,
        expert_weight=0.25,
        key="layers.0.ffn.experts.0.w2",
    )
    expert = activations.T @ activations / 2
    expected_covariance = pooled.lerp(expert, 0.25)

    assert torch.allclose(result["H"] / result["count"], expected_covariance)
    assert result["count"] == 2


def test_activation_recipe_metadata_is_distinct_and_model_bound() -> None:
    config = exl3_quantization_config(
        calibration_rows=512,
        seed=11,
        swiglu_limit=10.0,
        recipe=EXL3_ACTIVATION_RECIPE,
        calibration_override={
            "method": "layerwise_native_natural_routes",
            "hessian": "per_expert_natural_route_gate_squared_covariance",
            "distribution": "checkpoint_bound_native_expert_inputs",
            "activation_corpus_sha256": "a" * 64,
            "natural_routing": True,
            "forced_expert_activation": False,
            "route_gate_weighting": "squared_unit_rms",
        },
    )

    assert config["ds4rt"]["recipe"] == EXL3_ACTIVATION_RECIPE
    assert config["ds4rt"]["recipe"] != EXL3_RECIPE
    assert config["calibration"]["activation_corpus_sha256"] == "a" * 64
    assert config["calibration"]["gate_clamp"] == [None, 10.0]


def test_calibration_swiglu_matches_serving_clamp_order() -> None:
    gate = torch.tensor([-20.0, 20.0, 2.0])
    up = torch.tensor([-20.0, 20.0, 1.0])
    actual = deepseek_v4_swiglu_activation(gate, up, limit=10.0)
    expected_gate = torch.tensor([-20.0, 10.0, 2.0])
    expected_up = torch.tensor([-10.0, 10.0, 1.0])
    expected = torch.nn.functional.silu(expected_gate) * expected_up
    assert torch.equal(actual, expected)


def test_calibration_swiglu_rejects_missing_limit() -> None:
    values = torch.ones(1)
    for limit in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="positive and finite"):
            deepseek_v4_swiglu_activation(values, values, limit=limit)


def test_resume_rejects_pre_clamp_calibration_report(tmp_path: Path) -> None:
    snapshot = make_native_snapshot(tmp_path)
    plan = build_artifact_plan(snapshot, max_shard_bytes=32 * 1024)
    writer = SafetensorsArtifactWriter(
        plan,
        tmp_path / "exl3",
        calibration_rows=128,
        seed=7,
    )
    old_report = {
        "schema": "ds4rt-exl3-calibration-report-v1",
        "recipe": EXL3_RECIPE,
        "source_snapshot": str(plan.snapshot),
        "calibration_rows": 128,
        "seed": 7,
        "layers": [],
    }
    writer.partial_report_path.write_text(json.dumps(old_report), encoding="utf-8")
    new_template = {
        **old_report,
        "activation": "silu",
        "swiglu_limit": 10.0,
        "hessian": "analytic_identity",
        "distribution": "rms_normalized_isotropic",
    }
    with pytest.raises(ValueError, match="changed at activation"):
        writer.load_partial_report(new_template)


def test_resume_clears_previous_development_stop_marker(tmp_path: Path) -> None:
    snapshot = make_native_snapshot(tmp_path)
    plan = build_artifact_plan(snapshot, max_shard_bytes=32 * 1024)
    writer = SafetensorsArtifactWriter(
        plan,
        tmp_path / "exl3",
        calibration_rows=128,
        seed=7,
    )
    template = {
        "schema": "ds4rt-exl3-calibration-report-v1",
        "recipe": EXL3_RECIPE,
        "source_snapshot": str(plan.snapshot),
        "calibration_rows": 128,
        "seed": 7,
        "activation": "silu",
        "swiglu_limit": 10.0,
        "hessian": "analytic_identity",
        "distribution": "rms_normalized_isotropic",
        "layers": [{"layer_id": 0, "projections": []}],
    }
    writer.partial_report_path.write_text(
        json.dumps({**template, "incomplete": True}), encoding="utf-8"
    )
    resumed = calibration_report_for_run(writer, template)
    assert "incomplete" not in resumed
    assert resumed["layers"] == template["layers"]


def test_resume_rejects_changed_hessian_recipe(tmp_path: Path) -> None:
    snapshot = make_native_snapshot(tmp_path)
    plan = build_artifact_plan(snapshot, max_shard_bytes=32 * 1024)
    writer = SafetensorsArtifactWriter(
        plan,
        tmp_path / "exl3",
        calibration_rows=128,
        seed=7,
    )
    template = {
        "schema": "ds4rt-exl3-calibration-report-v1",
        "recipe": EXL3_RECIPE,
        "source_snapshot": str(plan.snapshot),
        "calibration_rows": 128,
        "seed": 7,
        "activation": "silu",
        "swiglu_limit": 10.0,
        "hessian": "analytic_identity",
        "distribution": "rms_normalized_isotropic",
        "layers": [],
    }
    writer.partial_report_path.write_text(
        json.dumps({**template, "hessian": "sample_covariance"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="changed at hessian"):
        writer.load_partial_report(template)
