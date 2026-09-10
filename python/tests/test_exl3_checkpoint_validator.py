from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import validate_ds4_exl3_checkpoint_layer as validator  # noqa: E402
from validate_ds4_exl3_checkpoint_layer import (  # noqa: E402
    aggregate_results,
    calibration_disjointness,
    calibration_summary,
    enforce_thresholds,
    expert_geometry,
    layer_sampling_mode,
    load_progress,
    natural_route_event_inputs,
    natural_stratified_expert_inputs,
    output_metrics,
    parse_expert_ids,
    progress_path,
    proxy_summary,
    readable_exl3_snapshot,
    stratified_expert_ids,
    write_progress,
)


def test_validator_compares_geometry_independently_of_tensor_layout() -> None:
    native = SimpleNamespace(
        hidden_size=4096,
        intermediate_size=2048,
        num_hidden_layers=43,
        dspark_blocks=3,
        global_experts=256,
        top_k=6,
        swiglu_limit=10.0,
        expert_tensor_layout="checkpoint-native",
    )
    exl3 = SimpleNamespace(**vars(native))
    exl3.expert_tensor_layout = "gptqmodel"

    assert expert_geometry(native) == expert_geometry(exl3)
    exl3.dspark_blocks = 1
    assert expert_geometry(native) != expert_geometry(exl3)


def test_validator_reuses_one_fail_closed_snapshot_proof_per_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator.cached_validated_exl3_snapshot.cache_clear()
    calls: list[Path] = []
    proof = SimpleNamespace(path=tmp_path.resolve(), config=object())

    def validate(snapshot: Path):
        calls.append(snapshot)
        return proof

    monkeypatch.setattr(validator, "validate_exl3_expert_snapshot", validate)
    try:
        first = validator.cached_validated_exl3_snapshot(tmp_path.resolve())
        second = validator.cached_validated_exl3_snapshot(tmp_path.resolve())
    finally:
        validator.cached_validated_exl3_snapshot.cache_clear()

    assert first is proof
    assert second is proof
    assert calls == [tmp_path.resolve()]


def test_natural_stratified_sampling_uses_only_naturally_selected_expert_rows() -> None:
    captured = torch.arange(32 * 4, dtype=torch.float32).reshape(32, 4).to(torch.bfloat16)
    router_weight = torch.zeros((8, 4), dtype=torch.bfloat16)
    router_bias = torch.tensor([6, 5, 4, 3, 2, 1, -1, -2], dtype=torch.float32)

    hidden, topk_ids, topk_weights, evidence, slices = natural_stratified_expert_inputs(
        captured,
        router_weight=router_weight,
        router_bias=router_bias,
        expert_ids=(0, 1, 2, 3, 4, 5),
        rows_per_expert=2,
        layer_id=7,
        seed=19,
        top_k=6,
        routed_scaling_factor=1.5,
        device=torch.device("cpu"),
    )

    assert hidden.shape == (12, 4)
    assert torch.equal(topk_ids, torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32).repeat(12, 1))
    assert slices == (
        (0, 0, 2),
        (1, 2, 4),
        (2, 4, 6),
        (3, 6, 8),
        (4, 8, 10),
        (5, 10, 12),
    )
    assert torch.count_nonzero(topk_weights).item() == 12
    assert torch.allclose(topk_weights.sum(dim=1), torch.full((12,), 0.25))
    assert evidence["natural_candidate_rows"] == {str(expert): 32 for expert in range(6)}
    assert evidence["execution_rows"] == 12


def test_captured_stratified_sampling_supports_hash_router_sidecars() -> None:
    captured = torch.arange(32 * 4, dtype=torch.float32).reshape(32, 4).to(torch.bfloat16)
    route_ids = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64).repeat(32, 1)
    route_weights = torch.full((32, 6), 0.25, dtype=torch.float32)
    routed = SimpleNamespace(
        samples=captured,
        expert_ids=route_ids,
        gate_weights=route_weights,
    )

    hidden, topk_ids, topk_weights, evidence, slices = (
        validator.captured_stratified_expert_inputs(
            routed,
            expert_ids=(0, 1, 2, 3, 4, 5),
            rows_per_expert=2,
            layer_id=0,
            seed=19,
            top_k=6,
            device=torch.device("cpu"),
        )
    )

    assert hidden.shape == (12, 4)
    assert topk_ids.shape == (12, 6)
    assert torch.count_nonzero(topk_weights).item() == 12
    assert torch.allclose(topk_weights.sum(dim=1), torch.full((12,), 0.25))
    assert slices == tuple((expert, expert * 2, expert * 2 + 2) for expert in range(6))
    assert evidence["route_source"] == "native_capture_sidecar"
    assert evidence["natural_candidate_rows"] == {
        str(expert): 32 for expert in range(6)
    }


def test_captured_route_events_preserve_native_ids_and_weights() -> None:
    captured = torch.arange(20 * 4, dtype=torch.float32).reshape(20, 4).to(torch.bfloat16)
    route_ids = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64).repeat(20, 1)
    route_weights = torch.full((20, 6), 0.25, dtype=torch.float32)
    routed = SimpleNamespace(
        samples=captured,
        expert_ids=route_ids,
        gate_weights=route_weights,
    )

    hidden, topk_ids, topk_weights, expert_ids, evidence = (
        validator.captured_route_event_inputs(
            routed,
            rows=2,
            layer_id=0,
            seed=19,
            top_k=6,
            routed_experts=8,
            device=torch.device("cpu"),
        )
    )

    assert hidden.shape == (12, 4)
    assert topk_ids.shape == (12, 6)
    assert torch.count_nonzero(topk_weights).item() == 12
    assert expert_ids == (0, 1, 2, 3, 4, 5)
    assert torch.allclose(
        topk_weights.reshape(2, 6, 6).sum(dim=(1, 2)),
        torch.full((2,), 1.5),
    )
    assert evidence["route_source"] == "native_capture_sidecar"


def test_natural_route_events_retain_exact_routes_and_isolate_each_event() -> None:
    captured = torch.arange(20 * 4, dtype=torch.float32).reshape(20, 4).to(torch.bfloat16)
    router_weight = torch.zeros((8, 4), dtype=torch.bfloat16)
    router_bias = torch.tensor([6, 5, 4, 3, 2, 1, -1, -2], dtype=torch.float32)

    hidden, topk_ids, topk_weights, expert_ids, evidence = natural_route_event_inputs(
        captured,
        router_weight=router_weight,
        router_bias=router_bias,
        rows=2,
        layer_id=7,
        seed=19,
        top_k=6,
        routed_scaling_factor=1.5,
        device=torch.device("cpu"),
    )

    assert hidden.shape == (12, 4)
    assert topk_ids.shape == (12, 6)
    assert topk_weights.shape == (12, 6)
    assert expert_ids == (0, 1, 2, 3, 4, 5)
    assert torch.count_nonzero(topk_weights).item() == 12
    assert torch.allclose(
        topk_weights.reshape(2, 6, 6).sum(dim=(1, 2)),
        torch.full((2,), 1.5),
    )
    assert evidence["selected_rows"] == 2
    assert evidence["execution_rows"] == 12
    assert evidence["unique_routed_experts"] == 6


def test_validator_metrics_distinguish_exact_and_perturbed_outputs() -> None:
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    exact = output_metrics(reference, reference)
    assert exact["cosine"] == pytest.approx(1.0)
    assert exact["relative_l2"] == 0.0

    perturbed = output_metrics(reference + 0.5, reference)
    assert perturbed["cosine"] < 1.0
    assert perturbed["relative_l2"] > 0.0


def test_validator_stratifies_default_experts_for_flash_and_pro() -> None:
    assert stratified_expert_ids(256) == (0, 51, 102, 153, 204, 255)
    assert stratified_expert_ids(384) == (0, 76, 153, 229, 306, 383)
    assert parse_expert_ids("auto", 256) == (0, 51, 102, 153, 204, 255)
    with pytest.raises(ValueError, match="checkpoint geometry"):
        parse_expert_ids("auto")


def test_hybrid_sampling_uses_natural_target_and_isotropic_dspark_rows() -> None:
    policy = validator.NATURAL_TARGET_SYNTHETIC_MTP

    assert layer_sampling_mode(
        policy, layer_id=42, captured_layer_count=43
    ) == "natural-stratified"
    assert layer_sampling_mode(
        policy, layer_id=43, captured_layer_count=43
    ) == "synthetic-stratified"
    assert layer_sampling_mode(
        "natural-route-events", layer_id=43, captured_layer_count=43
    ) == "natural-route-events"


def test_quality_gpu_is_bound_to_planned_physical_gpu0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "artifact"
    snapshot.mkdir()
    uuid = "GPU-95f8f212-9131-df99-fd53-7535965197d7"
    (snapshot / "ds41rt-gptqmodel-plan.json").write_text(
        json.dumps(
            {
                "preflight": {
                    "gpus": [
                        {
                            "index": 0,
                            "uuid": uuid,
                            "name": (
                                "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
                            ),
                            "driver_version": "595.84",
                            "compute_capability": [12, 0],
                        },
                        {
                            "index": 1,
                            "uuid": "GPU-fe5b6dd0-a77c-c8fb-6360-e1b9d9918ac0",
                            "name": (
                                "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
                            ),
                            "driver_version": "595.84",
                            "compute_capability": [12, 0],
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", uuid)
    monkeypatch.setattr(
        validator.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=(
                f"{uuid}, 00000000:11:00.0, "
                "NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 595.84, 12.0\n"
            )
        ),
    )

    identity = validator.quality_gpu_identity(
        snapshot,
        gptqmodel_native=True,
        requested_uuid=uuid,
    )

    assert identity is not None
    assert identity["physical_role"] == "coordinator-gpu0"
    assert identity["quantization_preflight"]["index"] == 0
    assert identity["validation_inventory"]["pci_bus_id"] == "00000000:11:00.0"
    assert len(identity["sha256"]) == 64

    with pytest.raises(ValueError, match="differs from quantization preflight GPU0"):
        validator.quality_gpu_identity(
            snapshot,
            gptqmodel_native=True,
            requested_uuid="GPU-00000000-0000-0000-0000-000000000000",
        )

    monkeypatch.setenv(
        "CUDA_VISIBLE_DEVICES", "GPU-fe5b6dd0-a77c-c8fb-6360-e1b9d9918ac0"
    )
    with pytest.raises(RuntimeError, match="exactly physical GPU UUID"):
        validator.quality_gpu_identity(
            snapshot,
            gptqmodel_native=True,
            requested_uuid=uuid,
        )


def test_validator_rejects_catastrophic_quantization_even_when_tp4_is_exact() -> None:
    result = {
        "native_to_exl3": {"cosine": 0.60, "relative_l2": 0.88},
        "exl3_tp4_to_unsharded": {"cosine": 1.0, "relative_l2": 0.0},
        "tp4_equal_rank_source_bytes": True,
    }
    args = SimpleNamespace(
        min_quant_cosine=0.98,
        max_quant_relative_l2=0.20,
        min_tp_cosine=0.999,
        max_tp_relative_l2=0.03,
    )
    with pytest.raises(RuntimeError, match="native-to-EXL3 cosine"):
        enforce_thresholds(result, args)


def test_validator_uses_separate_target_and_dspark_proxy_thresholds() -> None:
    args = SimpleNamespace(
        min_quant_cosine=0.88,
        max_quant_relative_l2=0.50,
        min_dspark_quant_cosine=0.84,
        max_dspark_quant_relative_l2=0.56,
    )

    target = validator.threshold_args_for_layer(args, 42, 43)
    dspark = validator.threshold_args_for_layer(args, 43, 43)

    assert target.min_quant_cosine == 0.88
    assert target.max_quant_relative_l2 == 0.50
    assert dspark.min_quant_cosine == 0.84
    assert dspark.max_quant_relative_l2 == 0.56


def test_validator_aggregates_worst_layer_quality() -> None:
    def result(quant_cosine, quant_relative_l2, tp_cosine, tp_relative_l2):
        return {
            "native_to_exl3": {
                "cosine": quant_cosine,
                "min_row_cosine": quant_cosine - 0.01,
                "relative_l2": quant_relative_l2,
                "rmse": quant_relative_l2 / 2,
                "max_abs": quant_relative_l2 * 2,
            },
            "exl3_tp4_to_unsharded": {
                "cosine": tp_cosine,
                "min_row_cosine": tp_cosine - 0.0001,
                "relative_l2": tp_relative_l2,
                "rmse": tp_relative_l2 / 2,
                "max_abs": tp_relative_l2 * 2,
            },
            "tp4_equal_rank_source_bytes": True,
        }

    summary = aggregate_results(
        [result(0.99, 0.14, 0.9999999, 0.0002), result(0.985, 0.17, 0.9999998, 0.0003)]
    )
    assert summary["layers"] == 2
    assert summary["native_to_exl3"]["min_cosine"] == pytest.approx(0.985)
    assert summary["native_to_exl3"]["max_relative_l2"] == pytest.approx(0.17)
    assert summary["exl3_tp4_to_unsharded"]["min_cosine"] == pytest.approx(0.9999998)
    assert summary["all_tp4_ranks_equal_source_bytes"] is True


def test_validator_aggregates_natural_routed_sum_quality() -> None:
    def result(cosine: float, relative_l2: float) -> dict:
        metric = {
            "cosine": cosine,
            "min_row_cosine": cosine - 0.02,
            "relative_l2": relative_l2,
            "rmse": relative_l2 / 2,
            "max_abs": relative_l2 * 2,
        }
        tp = {
            "cosine": 1.0,
            "min_row_cosine": 0.9999,
            "relative_l2": 0.0003,
            "rmse": 0.0001,
            "max_abs": 0.001,
        }
        return {
            "native_to_exl3": metric,
            "exl3_tp4_to_unsharded": tp,
            "natural_routed_sum": {
                "native_to_exl3": metric,
                "exl3_tp4_to_unsharded": tp,
            },
            "tp4_equal_rank_source_bytes": True,
        }

    summary = aggregate_results([result(0.93, 0.36), result(0.91, 0.41)])
    natural = summary["natural_routed_sum"]["native_to_exl3"]
    assert natural["min_cosine"] == pytest.approx(0.91)
    assert natural["max_relative_l2"] == pytest.approx(0.41)


def test_validator_selects_exact_mtp_expert_proxy_errors() -> None:
    expert_ids = parse_expert_ids("5,4,3,2,1,0")
    projections = []
    for expert_id in expert_ids:
        for stem, error in (("w1", 0.1), ("w2", 0.2), ("w3", 0.3)):
            projections.append(
                {
                    "name": f"mtp.0.ffn.experts.{expert_id}.{stem}",
                    "proxy_error": error,
                }
            )
    report = {"layers": [{"layer_id": 43, "projections": projections}]}
    summary = proxy_summary(report, 43, 43, expert_ids)
    assert summary["count"] == 18
    assert summary["mean"] == pytest.approx(0.2)


def test_validator_reads_gptqmodel_projection_metrics_without_legacy_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    family = {
        "corpus": {
            "examples": 1421,
            "normalized_stream_sha256": "a" * 64,
        },
        "quantizer_seed": 787,
        "quantizer_numerics": {
            "hessian_numerical": "signed-block-hadamard-congruence-fp64-v1"
        },
    }
    config = {
        "num_hidden_layers": 1,
        "dspark_target_layer_ids": [],
        "n_routed_experts": 6,
        "quantization_config": {"meta": {"ds41rt_error_ledger": {}}},
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    records = []
    for expert in range(6):
        for projection, error in (("w1", 0.1), ("w2", 0.2), ("w3", 0.3)):
            records.append(
                {
                    "record_kind": "projection",
                    "block_namespace": "base",
                    "logical_layer": 0,
                    "expert": expert,
                    "projection": projection,
                    "sample_count": 1000 + expert,
                    "quantizer_metrics": {
                        "hessian_weighted_relative_error": error
                    },
                }
            )
    (tmp_path / "ds41rt-exl3-error-ledger.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    validator.cached_validated_exl3_snapshot.cache_clear()
    validator.cached_gptqmodel_calibration_report.cache_clear()
    monkeypatch.setattr(
        validator,
        "validate_exl3_expert_snapshot",
        lambda snapshot: SimpleNamespace(path=Path(snapshot), config=object()),
    )
    monkeypatch.setattr(
        validator,
        "validate_gptqmodel_native_exl3",
        lambda *_args, **_kwargs: family,
    )

    try:
        with readable_exl3_snapshot(
            tmp_path, layer_id=0, allow_incomplete=False
        ) as (view, report):
            assert view == tmp_path
            summary = proxy_summary(report, 0, 1, tuple(range(6)))
            calibration = calibration_summary(report, 0, 1, tuple(range(6)))
    finally:
        validator.cached_validated_exl3_snapshot.cache_clear()
        validator.cached_gptqmodel_calibration_report.cache_clear()

    assert summary["count"] == 18
    assert summary["mean"] == pytest.approx(0.2)
    assert calibration["format"] == "gptqmodel-native-exl3-v4"
    assert calibration["corpus_examples"] == 1421
    assert calibration["selected_route_rows"] == {
        "min": 1000,
        "max": 1005,
        "mean": 1002.5,
    }


def test_gptqmodel_quality_requires_artifact_bound_disjoint_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact"
    activations = tmp_path / "activations"
    artifact.mkdir()
    activations.mkdir()
    calibration = tmp_path / "calibration.jsonl"
    calibration_prompts = [
        "calibration one",
        "calibration two",
        "calibration one",
    ]
    calibration.write_text(
        "".join(json.dumps({"prompt": prompt}) + "\n" for prompt in calibration_prompts),
        encoding="utf-8",
    )
    (artifact / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "meta": {"ds41rt_error_ledger": {}}
                }
            }
        ),
        encoding="utf-8",
    )
    heldout_prompt = "held out"
    manifest = activations / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "prompts": [
                    {
                        "prompt_sha256": hashlib.sha256(
                            heldout_prompt.encode()
                        ).hexdigest()
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    family = {
        "corpus": {
            "examples": 3,
            "file_sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
        }
    }
    monkeypatch.setattr(
        validator,
        "validate_gptqmodel_native_exl3",
        lambda *_args, **_kwargs: family,
    )
    args = SimpleNamespace(
        exl3_snapshot=artifact,
        calibration_jsonl=calibration,
        activation_corpus=activations,
        development_diagnostic=False,
        sampling_mode=validator.NATURAL_TARGET_SYNTHETIC_MTP,
    )

    evidence = calibration_disjointness(args)

    assert evidence is not None
    assert evidence["prompt_sha256_overlap"] == 0
    assert evidence["calibration_jsonl"]["prompts"] == 3
    assert evidence["calibration_jsonl"]["unique_prompts"] == 2
    assert evidence["heldout_activation_manifest"]["prompts"] == 1

    manifest.write_text(
        json.dumps(
            {
                "prompts": [
                    {
                        "prompt_sha256": hashlib.sha256(
                            calibration_prompts[0].encode()
                        ).hexdigest()
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overlap 1 calibration"):
        calibration_disjointness(args)


def test_development_diagnostic_allows_only_synthetic_without_heldout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "config.json").write_text(
        json.dumps({"quantization_config": {"meta": {"ds41rt_error_ledger": {}}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        validator,
        "validate_gptqmodel_native_exl3",
        lambda *_args, **_kwargs: {},
    )
    args = SimpleNamespace(
        exl3_snapshot=artifact,
        calibration_jsonl=None,
        activation_corpus=None,
        development_diagnostic=True,
        sampling_mode="synthetic-stratified",
    )

    assert calibration_disjointness(args) is None

    args.sampling_mode = validator.NATURAL_TARGET_SYNTHETIC_MTP
    with pytest.raises(ValueError, match="requires --sampling-mode synthetic"):
        calibration_disjointness(args)


def test_incomplete_snapshot_view_requires_checkpointed_layer(tmp_path: Path) -> None:
    (tmp_path / "config.json.incomplete").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json.incomplete").write_text(
        "{}", encoding="utf-8"
    )
    (tmp_path / "model.safetensors").write_bytes(b"test")
    (tmp_path / "ds41rt-exl3-calibration.json.incomplete").write_text(
        json.dumps({"layers": [{"layer_id": 0, "projections": []}]}),
        encoding="utf-8",
    )

    with readable_exl3_snapshot(
        tmp_path, layer_id=0, allow_incomplete=True
    ) as (view, report):
        assert (view / "config.json").resolve() == (
            tmp_path / "config.json.incomplete"
        ).resolve()
        assert (view / "model.safetensors").resolve() == (
            tmp_path / "model.safetensors"
        ).resolve()
        assert report["layers"][0]["layer_id"] == 0

    with pytest.raises(ValueError, match="has not checkpointed"):
        with readable_exl3_snapshot(tmp_path, layer_id=1, allow_incomplete=True):
            pass


def quality_args() -> SimpleNamespace:
    return SimpleNamespace(
        min_quant_cosine=0.98,
        max_quant_relative_l2=0.20,
        min_tp_cosine=0.999,
        max_tp_relative_l2=0.03,
    )


def passing_layer(layer_id: int) -> dict:
    return {
        "layer_id": layer_id,
        "native_to_exl3": {"cosine": 0.99, "relative_l2": 0.15},
        "exl3_tp4_to_unsharded": {"cosine": 0.9999, "relative_l2": 0.001},
        "tp4_equal_rank_source_bytes": True,
    }


def test_quality_resume_journal_is_atomic_and_contract_bound(tmp_path: Path) -> None:
    output = tmp_path / "quality.json"
    journal = progress_path(output)
    contract = {"layer_ids": [0, 1], "sha256": "contract-a"}
    write_progress(journal, contract, [passing_layer(1), passing_layer(0)])

    loaded = load_progress(journal, contract, quality_args())
    assert [result["layer_id"] for result in loaded] == [0, 1]
    assert not list(tmp_path.glob(".*.tmp"))

    with pytest.raises(ValueError, match="active validation contract"):
        load_progress(
            journal,
            {"layer_ids": [0, 1], "sha256": "contract-b"},
            quality_args(),
        )


def test_quality_resume_journal_rechecks_saved_thresholds(tmp_path: Path) -> None:
    journal = tmp_path / "quality.json.incomplete"
    contract = {"layer_ids": [0], "sha256": "contract"}
    failed = passing_layer(0)
    failed["native_to_exl3"]["relative_l2"] = 0.21
    write_progress(journal, contract, [failed])

    with pytest.raises(RuntimeError, match="relative L2"):
        load_progress(journal, contract, quality_args())
