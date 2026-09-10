from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch


QUANTIZATION = Path(__file__).parents[1]
if str(QUANTIZATION) not in sys.path:
    sys.path.insert(0, str(QUANTIZATION))
SCRIPT = QUANTIZATION / "validate_projection_checkpoint_block.py"
SPEC = importlib.util.spec_from_file_location(
    "validate_projection_checkpoint_block",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CANONICAL_SCRIPT = QUANTIZATION / "canonicalize_gptqmodel_artifact.py"
CANONICAL_SPEC = importlib.util.spec_from_file_location(
    "canonicalize_gptqmodel_artifact",
    CANONICAL_SCRIPT,
)
assert CANONICAL_SPEC is not None and CANONICAL_SPEC.loader is not None
CANONICAL = importlib.util.module_from_spec(CANONICAL_SPEC)
CANONICAL_SPEC.loader.exec_module(CANONICAL)

from ds41rt_runtime.exl3_quantizer import OutputTensor  # noqa: E402

from gptqmodel.utils.exl3_error_ledger import (  # noqa: E402
    append_exl3_error_journal,
    build_projection_record,
)
from gptqmodel.utils.exl3_projection_checkpoint import (  # noqa: E402
    EXL3ProjectionCheckpointStore,
    build_projection_request,
    canonical_json_bytes,
)
from gptqmodel.utils.exl3_inline_mixed import (  # noqa: E402
    INLINE_MIXED_META_KEY,
    InlineMixedTierPlanStore,
    build_layer_tier_plan,
    inline_mixed_policy,
)


def test_canonical_raw_config_completion_preserves_quant_and_source_extensions():
    raw = {
        "model_type": "deepseek_v4",
        "quantization_config": {"bits": 2, "format": "exl3"},
    }
    source = {
        "model_type": "deepseek_v4",
        "num_hash_layers": 3,
        "quantization_config": {"quant_method": "fp8"},
        "attn_implementation": "flash_attention_2",
    }

    completed = CANONICAL._source_complete_raw_config(raw, source)

    assert completed == {
        "model_type": "deepseek_v4",
        "num_hash_layers": 3,
        "quantization_config": {"bits": 2, "format": "exl3"},
    }
    assert raw == {
        "model_type": "deepseek_v4",
        "quantization_config": {"bits": 2, "format": "exl3"},
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _route_evidence(
    expert: int,
    *,
    block_namespace: str = "base",
    logical_layer: int = 0,
) -> dict[str, object]:
    return {
        "schema": "ds41rt.exl3-natural-route",
        "schema_version": 1,
        "block_namespace": block_namespace,
        "logical_layer": logical_layer,
        "expert": expert,
        "router_calls": 1,
        "router_token_count": 4,
        "router_selected_route_count": 8,
        "router_top_k": 2,
        "router_weight_dtypes": ["torch.float32"],
        "mask_modes": ["attention_mask"],
        "expert_route_count": 4,
        "expert_gate_weight_sum": 2.0,
        "expert_gate_squared_mass": 1.0,
        "total_gate_weight_sum": 4.0,
        "total_gate_squared_mass": 2.0,
        "expert_route_fraction": 0.5,
        "expert_gate_weight_mass_fraction": 0.5,
        "expert_gate_squared_mass_fraction": 0.5,
        "expert_gate_weight_mean": 0.5,
        "expert_gate_weight_rms": 0.5,
    }


def _metrics(value: float, *, sample_count: int = 4) -> dict[str, object]:
    return {
        "schema": "gptqmodel.exl3-trellis-error",
        "schema_version": 1,
        "quantizer_path": "hessian_ldlq",
        "hessian_domain": "regularized_exl3_search_space",
        "hessian_metric_status": "ok",
        "hessian_regularization_sigma": 0.025,
        "hessian_sample_count": sample_count,
        "hessian_numerical_contract": ("signed-block-hadamard-congruence-fp64-v1"),
        "hessian_transform_compute_dtype": "torch.float64",
        "hessian_storage_dtype": "torch.float32",
        "hessian_regularization_placement": "before-fp64-congruence",
        "hessian_regularization_diagonal_addend": 0.00018,
        "hessian_symmetry_restoration": "mean-with-transpose-fp64",
        "hessian_symmetry_correction_max_abs": value / 10,
        "hessian_weighted_error_numerator": value,
        "hessian_weighted_reference_denominator": 1.0,
        "hessian_weighted_relative_error": value,
        "reported_metric_kind": "hessian_weighted_relative_error",
        "reported_metric_value": value,
        "scale_search_mse": value * 2,
        "selected_global_scale": 1.0,
        "apply_out_scales": True,
        "reconstruction": {
            "domain": "regularized_exl3_search_space",
            "element_count": 512,
            "error_sum_sq": value,
            "reference_sum_sq": 1.0,
            "mse": value / 512,
            "nmse": value,
            "relative_frobenius": value**0.5,
            "max_abs_error": value,
            "error_finite": True,
            "reference_finite": True,
        },
    }


def _projection_tensors(projection: str, *, bits: int = 2) -> dict[str, torch.Tensor]:
    if projection in {"w1", "w3"}:
        trellis_shape = (2, 1, bits * 16)
        suh = torch.ones(32, dtype=torch.float16)
        svh = torch.ones(16, dtype=torch.float16)
    else:
        trellis_shape = (1, 2, bits * 16)
        suh = torch.ones(16, dtype=torch.float16)
        svh = torch.ones(32, dtype=torch.float16)
    return {
        "trellis": torch.arange(
            torch.tensor(trellis_shape).prod().item(),
            dtype=torch.int16,
        ).reshape(trellis_shape),
        "suh": suh,
        "svh": svh,
        "mcg": torch.tensor(-877912083, dtype=torch.int32),
    }


def _make_run_state(
    tmp_path: Path,
    *,
    omit: str | None = None,
    include_mtp: bool = False,
    bits: int = 2,
    overlay: bool = False,
    recover_expert: int | None = None,
    router_recovery: bool = False,
    mixed_recovery: bool = False,
    recover_namespace: str = "mtp",
) -> Path:
    run_state = (tmp_path / "run-state").resolve()
    run_state.mkdir()
    checkpoint_root = run_state / "projection-checkpoints"
    assignment_root = run_state / "dynamic-projection-assignments"
    family_join = {
        "recipe": "deepseek_v4_exl3_trellis_2bpw_v4_flash_natural_route",
        "bits": bits,
        "codebook": "mcg",
        "route_evidence_contract": "ds41rt.exl3-natural-route",
        "quantizer_numerics": {
            "sigma_reg": 0.025,
            "hessian_capture": "raw-xtx-sum-fp32-v1",
            "hessian_numerical": "signed-block-hadamard-congruence-fp64-v1",
            "hessian_symmetry": "mean-with-transpose-fp64",
        },
    }
    mtp_anchor_selection = {
        "contract": "ds41rt-mtp-anchor-stratified-v1",
        "count": 8,
        "seed": 20260809,
    }
    mtp_replay_batching = {
        "contract": "ds41rt-mtp-source-sequence-anchor-batches-v1",
        "proposal_rows_per_anchor": 5,
        "source_sequence_anchor_cap": None,
    }
    family_join["mtp_anchor_selection"] = mtp_anchor_selection
    family_join["mtp_replay_batching"] = mtp_replay_batching
    if recover_expert is not None:
        family_join["zero_route_recovery_contract"] = "ds41rt.exl3-zero-route-recovery"
    plan = {
        "schema": (
            "ds41rt-deepseek-v4-dspark-overlay-plan-v2"
            if overlay
            else "ds41rt-deepseek-v4-gptqmodel-plan-v5"
        ),
        "source": {
            "geometry": {
                "num_hidden_layers": 1,
                "n_routed_experts": 2,
                "hidden_size": 32,
                "moe_intermediate_size": 16,
                "dspark_target_layer_ids": [0] if include_mtp or overlay else [],
            }
        },
        "run_state_dir": str(run_state),
        "projection_checkpoint": {
            "contract": "ds41rt.exl3-projection-checkpoint-v1",
            "root": str(checkpoint_root),
        },
        "remote_workers": {
            "scheduler": "dynamic-pipelined-slot-projection-v2",
            "assignment_store": str(assignment_root),
        },
        "mtp_execution_mode": "integrated",
        "mtp_anchor_selection": mtp_anchor_selection,
        "mtp_replay_batching": mtp_replay_batching,
        "exl3": {"bits": bits, "codebook": "mcg"},
        "ledger_provenance": {
            "family_join": family_join,
            "run": {
                "mtp_anchor_selection": mtp_anchor_selection,
                "mtp_replay_batching": mtp_replay_batching,
            },
        },
    }
    if overlay:
        plan["scope"] = "mtp-routed-experts-only"
    plan["plan_sha256"] = hashlib.sha256(canonical_json_bytes(plan)).hexdigest()
    _write_json(run_state / "ds41rt-gptqmodel-plan.json", plan)

    store = EXL3ProjectionCheckpointStore(checkpoint_root)
    journal = run_state / ".ds41rt-exl3-error-journal.jsonl"
    execution = {
        "kind": "coordinator",
        "device": "cuda:0",
        "gpu_uuid": "GPU-test",
        "preflight_sha256": "a" * 64,
        "image_digest": "sha256:" + "b" * 64,
        "remote_contract": "ds41rt.exl3-remote-worker-v1",
    }
    projection_names = {
        "w1": "gate_proj",
        "w2": "down_proj",
        "w3": "up_proj",
    }
    blocks = [] if overlay else [("base", 0, "model.layers.0")]
    if include_mtp or overlay:
        blocks.append(("mtp", 0, "mtp.0"))
    for block_namespace, logical_layer, block_prefix in blocks:
        for expert in range(2):
            route = _route_evidence(
                expert,
                block_namespace=block_namespace,
                logical_layer=logical_layer,
            )
            recovery = None
            if expert == recover_expert and block_namespace == recover_namespace:
                if not router_recovery and not mixed_recovery:
                    route.update(
                        {
                            "expert_route_count": 0,
                            "expert_gate_weight_sum": 0.0,
                            "expert_gate_squared_mass": 0.0,
                            "expert_route_fraction": 0.0,
                            "expert_gate_weight_mass_fraction": 0.0,
                            "expert_gate_squared_mass_fraction": 0.0,
                            "expert_gate_weight_mean": 0.0,
                            "expert_gate_weight_rms": 0.0,
                        }
                    )
                family_digest = hashlib.sha256(
                    canonical_json_bytes(family_join)
                ).hexdigest()
                natural_count = int(route["expert_route_count"])
                router_augmented_count = (
                    5
                    if mixed_recovery
                    else 1024 - natural_count
                    if router_recovery
                    else 0
                )
                identity_count = (
                    1024 - natural_count - router_augmented_count
                    if mixed_recovery
                    else 0
                    if router_recovery
                    else 1024
                )
                recovery = {
                    "schema": MODULE.ZERO_ROUTE_RECOVERY_SCHEMA,
                    "schema_version": 1,
                    "trigger": MODULE.ZERO_ROUTE_RECOVERY_TRIGGER,
                    "sample_source": MODULE.ZERO_ROUTE_RECOVERY_SAMPLE_SOURCE,
                    "capture_method": MODULE.ZERO_ROUTE_RECOVERY_CAPTURE_METHOD,
                    "selection_policy": MODULE.ZERO_ROUTE_RECOVERY_SELECTION_POLICY,
                    "candidate_rank_min": MODULE.ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MIN,
                    "candidate_rank_max": MODULE.ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MAX,
                    "selection_cap": MODULE.ZERO_ROUTE_RECOVERY_TARGET_SAMPLE_COUNT,
                    "target_sample_count": MODULE.ZERO_ROUTE_RECOVERY_TARGET_SAMPLE_COUNT,
                    "identity_calibration_policy": MODULE.ZERO_ROUTE_RECOVERY_IDENTITY_POLICY,
                    "block_namespace": block_namespace,
                    "logical_layer": logical_layer,
                    "expert": expert,
                    "natural_sample_count": natural_count,
                    "router_augmented_sample_count": router_augmented_count,
                    "identity_calibration_count": identity_count,
                    "total_sample_count": MODULE.ZERO_ROUTE_RECOVERY_TARGET_SAMPLE_COUNT,
                    "forced_pass_count": 1,
                    "recovery_mode": (
                        "empirical-plus-identity-hessian"
                        if mixed_recovery
                        else "router-near-rows"
                        if router_recovery
                        else "identity-hessian"
                    ),
                    "candidate_rows_observed": router_augmented_count,
                    "candidate_rows_selected": router_augmented_count,
                    "candidate_rank_histogram": {
                        str(rank): router_augmented_count if rank == 7 else 0
                        for rank in range(
                            MODULE.ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MIN,
                            MODULE.ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MAX + 1,
                        )
                    },
                    "candidate_score_gap": (
                        {"min": 0.01, "mean": 0.02, "max": 0.03}
                        if router_recovery or mixed_recovery
                        else None
                    ),
                    "authorization": {
                        "schema": ("ds41rt.exl3-zero-route-recovery-authorization"),
                        "schema_version": 1,
                        "kind": "immutable-family-join",
                        "recovery_contract": MODULE.ZERO_ROUTE_RECOVERY_SCHEMA,
                        "trigger": MODULE.ZERO_ROUTE_RECOVERY_TRIGGER,
                        "sample_source": MODULE.ZERO_ROUTE_RECOVERY_SAMPLE_SOURCE,
                        "capture_method": MODULE.ZERO_ROUTE_RECOVERY_CAPTURE_METHOD,
                        "selection_policy": MODULE.ZERO_ROUTE_RECOVERY_SELECTION_POLICY,
                        "candidate_rank_min": MODULE.ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MIN,
                        "candidate_rank_max": MODULE.ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MAX,
                        "target_sample_count": MODULE.ZERO_ROUTE_RECOVERY_TARGET_SAMPLE_COUNT,
                        "identity_calibration_policy": MODULE.ZERO_ROUTE_RECOVERY_IDENTITY_POLICY,
                        "family_join_sha256": family_digest,
                        "authorization_sha256": family_digest,
                    },
                }
            for projection, projection_name in projection_names.items():
                module = f"{block_prefix}.mlp.experts.{expert}.{projection_name}"
                if module == omit:
                    continue
                tensors = _projection_tensors(projection, bits=bits)
                encoded_bytes = sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in tensors.values()
                )
                effective_sample_count = (
                    MODULE.ZERO_ROUTE_RECOVERY_TARGET_SAMPLE_COUNT
                    if recovery is not None
                    else 4
                )
                metrics = _metrics(
                    0.1 + expert / 100 + len(projection) / 1000,
                    sample_count=effective_sample_count,
                )
                provenance = {"family_join": family_join, "execution": execution}
                ledger = build_projection_record(
                    module_full_name=module,
                    layer_index=logical_layer,
                    bits=bits,
                    codebook="mcg",
                    sample_count=effective_sample_count,
                    duration_seconds=1.0,
                    encoded_bytes=encoded_bytes,
                    device_names=["cuda:0"],
                    quantizer_metrics=metrics,
                    provenance=provenance,
                    route_evidence=route,
                    zero_route_recovery=recovery,
                )
                request = build_projection_request(
                    module_full_name=module,
                    layer_index=logical_layer,
                    input_weight=torch.ones(16, 32),
                    hessian=torch.eye(32),
                    sample_count=effective_sample_count,
                    quantizer_contract={
                        "bits": bits,
                        "codebook": "mcg",
                        "apply_out_scales": None,
                        "sigma_reg": 0.025,
                        "seed": 787,
                        "hessian_capture": "raw-xtx-sum-fp32-v1",
                        "hessian_numerical": (
                            "signed-block-hadamard-congruence-fp64-v1"
                        ),
                        "hessian_symmetry": "mean-with-transpose-fp64",
                        "execution": execution,
                    },
                    family_join=family_join,
                    route_evidence=route,
                    zero_route_recovery=recovery,
                )
                result = {
                    "duration_seconds": 1.0,
                    "proxy_error": metrics["reported_metric_value"],
                    "device_names": ["cuda:0"],
                    "quantizer_metrics": metrics,
                    "ledger_record": ledger,
                    "execution_contract": execution,
                    "execution_result": {
                        "kind": "coordinator",
                        "scheduler_assignment_key": module,
                        "scheduler_new_assignment": True,
                        "scheduler_wait_seconds": 0.0,
                        "coordinator_quant_lock_wait_seconds": 0.0,
                    },
                }
                store.commit(request, tensors, result)
                append_exl3_error_journal(journal, ledger)

                key_sha256 = hashlib.sha256(module.encode()).hexdigest()
                assignment_body = {
                    "schema": "ds41rt.exl3-dynamic-projection-assignments",
                    "schema_version": 1,
                    "scheduler": "dynamic-pipelined-slot-projection-v2",
                    "topology_sha256": "c" * 64,
                    "assignment_key": module,
                    "assignment_key_sha256": key_sha256,
                    "slot_id": "coordinator:cuda:0",
                    "execution": execution,
                }
                assignment = {
                    **assignment_body,
                    "record_sha256": hashlib.sha256(
                        canonical_json_bytes(assignment_body)
                    ).hexdigest(),
                }
                _write_json(
                    assignment_root
                    / key_sha256[:2]
                    / key_sha256[2:4]
                    / f"{key_sha256}.json",
                    assignment,
                )
    return run_state


def _make_inline_mixed_run_state(
    tmp_path: Path,
    *,
    include_mtp: bool = True,
) -> Path:
    """Rewrite the small uniform fixture as an authenticated K2/K3 run."""

    run_state = _make_run_state(tmp_path, include_mtp=include_mtp)
    plan_path = run_state / "ds41rt-gptqmodel-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    checkpoint_root = run_state / "projection-checkpoints"
    old_store = EXL3ProjectionCheckpointStore(checkpoint_root)
    old_entries = []
    for request, _result in old_store.inspect_committed_manifests():
        loaded = old_store.load(request)
        assert loaded is not None
        tensors, result = loaded
        old_entries.append((request, tensors, result))

    shutil.rmtree(checkpoint_root)
    (run_state / ".ds41rt-exl3-error-journal.jsonl").unlink()
    candidate_journal = run_state / ".ds41rt-exl3-error-journal.jsonl.k2-candidates"
    tier_root = run_state / "inline-mixed-tier-plans"
    namespaces = ["base", *(["mtp"] if include_mtp else [])]
    policies = {
        namespace: {
            "schema": "gptqmodel.exl3-inline-mixed",
            "schema_version": 1,
            "namespace": namespace,
            "base_bits": 2,
            "upgrade_bits": 3,
            "extra_bits": {"numerator": 1, "denominator": 2},
            "target_bpw": "5/2",
            "projection_ratio": {"w1": 1, "w3": 1, "w2": 1},
            "score_kind": (
                "k2-hessian-weighted-relative-error-times-natural-gate-squared-mass-v1"
            ),
            "tier_plan_root": str(tier_root),
        }
        for namespace in namespaces
    }
    family_join = plan["ledger_provenance"]["family_join"]
    family_join["inline_mixed"] = {
        namespace: {
            key: value for key, value in policy.items() if key != "tier_plan_root"
        }
        for namespace, policy in policies.items()
    }
    plan["inline_mixed"] = policies
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    plan = {
        **body,
        "plan_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }
    _write_json(plan_path, plan)

    store = EXL3ProjectionCheckpointStore(checkpoint_root)
    candidates: dict[tuple[str, int], list[dict[str, object]]] = {}
    candidate_state: dict[str, tuple[dict, dict, dict, dict]] = {}
    for old_request, tensors, old_result in old_entries:
        module = old_request["module"]
        identity = MODULE.routed_expert_identity(module)
        assert identity is not None
        namespace = identity["block_namespace"]
        policy = inline_mixed_policy({INLINE_MIXED_META_KEY: policies[namespace]})
        assert policy is not None
        inline = {
            "base_bits": 2,
            "upgrade_bits": 3,
            "policy_sha256": policy.policy_sha256,
            "role": "candidate_k2",
        }
        quantizer_contract = {
            **old_request["quantizer_contract"],
            "bits": 2,
            "inline_mixed": inline,
        }
        old_ledger = old_result["ledger_record"]
        provenance = {
            "family_join": family_join,
            "execution": old_ledger["provenance"]["execution"],
        }
        ledger = build_projection_record(
            module_full_name=module,
            layer_index=identity["logical_layer"],
            bits=2,
            codebook="mcg",
            sample_count=old_ledger["sample_count"],
            duration_seconds=old_result["duration_seconds"],
            encoded_bytes=sum(
                tensor.numel() * tensor.element_size() for tensor in tensors.values()
            ),
            device_names=old_result["device_names"],
            quantizer_metrics=old_result["quantizer_metrics"],
            provenance=provenance,
            route_evidence=old_request["route_evidence"],
            zero_route_recovery=old_request.get("zero_route_recovery"),
        )
        request = build_projection_request(
            module_full_name=module,
            layer_index=identity["logical_layer"],
            input_weight=torch.ones(16, 32),
            hessian=torch.eye(32),
            sample_count=old_ledger["sample_count"],
            quantizer_contract=quantizer_contract,
            family_join=family_join,
            route_evidence=old_request["route_evidence"],
            zero_route_recovery=old_request.get("zero_route_recovery"),
        )
        result = {
            **old_result,
            "ledger_record": ledger,
            "quantizer_metrics": ledger["quantizer_metrics"],
            "proxy_error": ledger["quantizer_metrics"]["reported_metric_value"],
        }
        store.commit(request, tensors, result)
        append_exl3_error_journal(candidate_journal, ledger)
        candidates.setdefault((namespace, identity["logical_layer"]), []).append(ledger)
        candidate_state[module] = (request, result, identity, quantizer_contract)

    selected_by_module: dict[str, dict[str, object]] = {}
    for (namespace, logical_layer), records in candidates.items():
        policy = inline_mixed_policy({INLINE_MIXED_META_KEY: policies[namespace]})
        assert policy is not None
        tier_plan = build_layer_tier_plan(
            policy=policy,
            layer_index=logical_layer,
            layer_count=1,
            candidate_records=records,
        )
        InlineMixedTierPlanStore(policy).commit(tier_plan)
        selected_by_module.update(
            {entry["module"]: tier_plan for entry in tier_plan["selected"]}
        )

    journal = run_state / ".ds41rt-exl3-error-journal.jsonl"
    for module, (
        candidate_request,
        candidate_result,
        identity,
        candidate_contract,
    ) in candidate_state.items():
        tier_plan = selected_by_module.get(module)
        if tier_plan is None:
            append_exl3_error_journal(journal, candidate_result["ledger_record"])
            continue
        projection = identity["projection"]
        tensors = _projection_tensors(projection, bits=3)
        metrics = candidate_result["quantizer_metrics"]
        provenance = candidate_result["ledger_record"]["provenance"]
        ledger = build_projection_record(
            module_full_name=module,
            layer_index=identity["logical_layer"],
            bits=3,
            codebook="mcg",
            sample_count=candidate_result["ledger_record"]["sample_count"],
            duration_seconds=1.1,
            encoded_bytes=sum(
                tensor.numel() * tensor.element_size() for tensor in tensors.values()
            ),
            device_names=candidate_result["device_names"],
            quantizer_metrics=metrics,
            provenance=provenance,
            route_evidence=candidate_request["route_evidence"],
            zero_route_recovery=candidate_request.get("zero_route_recovery"),
        )
        selected_inline = {
            **candidate_contract["inline_mixed"],
            "role": "selected_k3",
            "candidate_request_sha256": candidate_request["request_sha256"],
            "tier_plan_sha256": tier_plan["tier_plan_sha256"],
        }
        request = build_projection_request(
            module_full_name=module,
            layer_index=identity["logical_layer"],
            input_weight=torch.ones(16, 32),
            hessian=torch.eye(32),
            sample_count=ledger["sample_count"],
            quantizer_contract={
                **candidate_contract,
                "bits": 3,
                "inline_mixed": selected_inline,
            },
            family_join=family_join,
            route_evidence=candidate_request["route_evidence"],
            zero_route_recovery=candidate_request.get("zero_route_recovery"),
        )
        result = {
            **candidate_result,
            "duration_seconds": 1.1,
            "ledger_record": ledger,
        }
        store.commit(request, tensors, result)
        append_exl3_error_journal(journal, ledger)
    return run_state


def test_audit_complete_projection_block(tmp_path: Path) -> None:
    run_state = _make_run_state(tmp_path)

    report = MODULE.audit_block(
        run_state,
        block_namespace="base",
        logical_layer=0,
    )

    assert report["status"] == "complete"
    assert report["projection_count"] == 6
    assert report["complete_expert_families"] == 2
    assert report["encoded_bytes"] == 6 * 228
    assert report["ownership"] == {"coordinator:cuda:0": 6}
    assert report["route_coverage"] == {
        "experts": 2,
        "zero_count": 0,
        "min": 4,
        "mean": 4.0,
        "max": 4,
    }
    assert len(report["report_sha256"]) == 64


def test_audit_complete_k3_dspark_overlay_block(tmp_path: Path) -> None:
    run_state = _make_run_state(tmp_path, bits=3, overlay=True)

    report = MODULE.audit_block(
        run_state,
        block_namespace="mtp",
        logical_layer=0,
    )

    assert report["status"] == "complete"
    assert report["geometry"]["bits"] == 3
    assert report["projection_count"] == 6
    assert report["encoded_bytes"] == 6 * 292
    with pytest.raises(MODULE.AuditError, match="contain only MTP"):
        MODULE.audit_block(
            run_state,
            block_namespace="base",
            logical_layer=0,
        )


def test_audit_selects_authenticated_inline_mixed_k2_k3_tiers(
    tmp_path: Path,
) -> None:
    run_state = _make_inline_mixed_run_state(tmp_path)

    report = MODULE.audit_block(
        run_state,
        block_namespace="base",
        logical_layer=0,
    )

    mixed = report["geometry"]["inline_mixed"]
    assert mixed["tier_counts"] == {"2": 3, "3": 3}
    assert mixed["quotas"] == {"w1": 1, "w3": 1, "w2": 1}
    assert report["projection_count"] == 6
    assert report["encoded_bytes"] == 3 * 228 + 3 * 292


def test_audit_rejects_drifted_inline_mixed_tier_plan(tmp_path: Path) -> None:
    run_state = _make_inline_mixed_run_state(tmp_path)
    tier_path = run_state / "inline-mixed-tier-plans/base/layer-000000.json"
    tier = json.loads(tier_path.read_text(encoding="utf-8"))
    tier["selected"][0]["score"] += 1.0
    _write_json(tier_path, tier)

    with pytest.raises(MODULE.AuditError, match="immutable contract"):
        MODULE.audit_block(
            run_state,
            block_namespace="base",
            logical_layer=0,
        )


def test_audit_reports_authorized_exact_zero_recovery(tmp_path: Path) -> None:
    run_state = _make_run_state(
        tmp_path,
        overlay=True,
        recover_expert=1,
    )

    report = MODULE.audit_block(
        run_state,
        block_namespace="mtp",
        logical_layer=0,
    )

    assert report["route_coverage"] == {
        "experts": 2,
        "zero_count": 1,
        "min": 0,
        "mean": 2.0,
        "max": 4,
        "augmented_expert_count": 1,
        "augmented_experts": [1],
        "router_augmented_samples": {"min": 0, "mean": 0.0, "max": 0},
        "identity_calibration": {"expert_count": 1, "effective_count": 1024},
    }


def test_audit_reports_authorized_low_positive_router_topup(tmp_path: Path) -> None:
    run_state = _make_run_state(
        tmp_path,
        overlay=True,
        recover_expert=1,
        router_recovery=True,
    )

    report = MODULE.audit_block(
        run_state,
        block_namespace="mtp",
        logical_layer=0,
    )

    assert report["route_coverage"] == {
        "experts": 2,
        "zero_count": 0,
        "min": 4,
        "mean": 4.0,
        "max": 4,
        "augmented_expert_count": 1,
        "augmented_experts": [1],
        "router_augmented_samples": {"min": 1020, "mean": 1020.0, "max": 1020},
        "identity_calibration": {"expert_count": 0, "effective_count": 0},
    }


def test_tp4_residency_matches_runtime_plane_and_rotation_order() -> None:
    hidden_size = 32
    intermediate_size = 64
    expert_count = 2
    selected = {}
    expected = {}
    for expert in range(expert_count):
        for projection, projection_name in MODULE.PROJECTION_NAMES.items():
            module = f"model.layers.0.mlp.experts.{expert}.{projection_name}"
            if projection in {"w1", "w3"}:
                trellis_shape = (hidden_size // 16, intermediate_size // 16, 32)
                input_features = hidden_size
                output_features = intermediate_size
            else:
                trellis_shape = (intermediate_size // 16, hidden_size // 16, 32)
                input_features = intermediate_size
                output_features = hidden_size
            projection_index = tuple(MODULE.PROJECTION_NAMES).index(projection)
            seed = expert * 1_000 + projection_index * 100
            tensors = {
                "trellis": (
                    torch.arange(math.prod(trellis_shape), dtype=torch.int32)
                    .add(seed)
                    .to(torch.int16)
                    .reshape(trellis_shape)
                ),
                "suh": torch.arange(input_features, dtype=torch.float16).add_(seed),
                "svh": torch.arange(output_features, dtype=torch.float16).add_(
                    seed + 17
                ),
                "mcg": torch.tensor(-877912083, dtype=torch.int32),
            }
            selected[module] = (tensors, {}, {})
            expected[module] = (expert, projection)

    report = MODULE._tp4_residency_summary(
        selected,
        expected,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        expert_count=expert_count,
    )

    assert report["schema"] == MODULE.TP4_RESIDENCY_SCHEMA
    assert report["world_size"] == 4
    assert report["placement"] == "strict-tp4-replicated-experts"
    assert report["expert_count_per_rank"] == expert_count
    assert report["local_intermediate_size"] == 16
    assert report["rank_resident_bytes_equal"] is True
    assert [rank["rank"] for rank in report["ranks"]] == [0, 1, 2, 3]
    assert {rank["resident_weight_bytes"] for rank in report["ranks"]} == {768}
    assert {rank["resident_metadata_bytes"] for rank in report["ranks"]} == {612}
    assert {rank["resident_bytes"] for rank in report["ranks"]} == {1_380}
    for component in ("w13_trellis", "w2_trellis", "intermediate_rotations"):
        assert (
            len({rank["components"][component]["sha256"] for rank in report["ranks"]})
            == 4
        )
    for component in ("gate_suh", "up_suh", "down_svh"):
        assert (
            len({rank["components"][component]["sha256"] for rank in report["ranks"]})
            == 1
        )
    assert report["runtime_generated"]["bytes"] == 36
    assert len(report["report_sha256"]) == 64


def test_audit_rejects_incomplete_projection_block(tmp_path: Path) -> None:
    missing = "model.layers.0.mlp.experts.1.up_proj"
    run_state = _make_run_state(tmp_path, omit=missing)

    with pytest.raises(MODULE.AuditError, match="5/6 projection checkpoints"):
        MODULE.audit_block(
            run_state,
            block_namespace="base",
            logical_layer=0,
        )

    partial = MODULE.audit_block(
        run_state,
        block_namespace="base",
        logical_layer=0,
        require_complete=False,
    )
    assert partial["status"] == "partial"
    assert partial["missing_projection_count"] == 1
    assert partial["first_missing_projection"] == missing

    with pytest.raises(MODULE.AuditError, match="requires a complete block"):
        MODULE.audit_block(
            run_state,
            block_namespace="base",
            logical_layer=0,
            require_complete=False,
            include_tp4_residency=True,
        )


def _canonical_generated_tensors(run_state: Path) -> tuple[OutputTensor, ...]:
    plan = json.loads(
        (run_state / "ds41rt-gptqmodel-plan.json").read_text(encoding="utf-8")
    )
    dtype_names = {
        torch.int16: "I16",
        torch.float16: "F16",
        torch.int32: "I32",
    }
    projection_bits, _tiers = CANONICAL._projection_bits_from_tier_plans(
        plan,
        run_state,
    )
    generated = []
    for module, identity in CANONICAL._expected_modules(plan).items():
        for suffix, tensor in _projection_tensors(
            identity["projection"], bits=projection_bits[module]
        ).items():
            generated.append(
                OutputTensor(
                    name=f"{module}.{suffix}",
                    dtype=dtype_names[tensor.dtype],
                    shape=tuple(tensor.shape),
                    nbytes=tensor.numel() * tensor.element_size(),
                    source=None,
                )
            )
    return tuple(generated)


def _write_complete_block_audits(run_state: Path, report_dir: Path) -> Path:
    plan = json.loads(
        (run_state / "ds41rt-gptqmodel-plan.json").read_text(encoding="utf-8")
    )
    geometry = plan["source"]["geometry"]
    for namespace, count in (
        ("base", geometry["num_hidden_layers"]),
        ("mtp", len(geometry["dspark_target_layer_ids"])),
    ):
        for logical_layer in range(count):
            report = MODULE.audit_block(
                run_state,
                block_namespace=namespace,
                logical_layer=logical_layer,
            )
            _write_json(
                report_dir / f"{namespace}-layer-{logical_layer}-projection-audit.json",
                report,
            )
    return report_dir


def test_canonical_block_audit_set_binds_complete_base_and_mtp_reports(
    tmp_path: Path,
) -> None:
    run_state = _make_run_state(tmp_path, include_mtp=True)
    report_dir = _write_complete_block_audits(run_state, tmp_path / "reports")
    plan = json.loads(
        (run_state / "ds41rt-gptqmodel-plan.json").read_text(encoding="utf-8")
    )

    report = CANONICAL.validate_block_audit_set(report_dir, plan)

    assert report["schema"] == CANONICAL.BLOCK_AUDIT_SET_SCHEMA
    assert report["plan_sha256"] == plan["plan_sha256"]
    assert report["base_block_count"] == 1
    assert report["mtp_block_count"] == 1
    assert report["block_count"] == 2
    assert report["projections_per_block"] == 6
    assert report["projection_count"] == 12
    assert report["expert_family_count"] == 4
    assert [item["filename"] for item in report["reports"]] == [
        "base-layer-0-projection-audit.json",
        "mtp-layer-0-projection-audit.json",
    ]
    assert (
        report["reports_sha256"]
        == hashlib.sha256(canonical_json_bytes(report["reports"])).hexdigest()
    )
    assert len(report["report_sha256"]) == 64


def test_canonical_block_audit_set_binds_inline_mixed_tier_plans(
    tmp_path: Path,
) -> None:
    run_state = _make_inline_mixed_run_state(tmp_path)
    report_dir = _write_complete_block_audits(run_state, tmp_path / "reports")
    plan = json.loads(
        (run_state / "ds41rt-gptqmodel-plan.json").read_text(encoding="utf-8")
    )

    report = CANONICAL.validate_block_audit_set(
        report_dir,
        plan,
        run_state=run_state,
    )

    assert report["block_count"] == 2
    assert report["projection_count"] == 12


def test_canonical_block_audit_set_accepts_mtp_identity_recovery(
    tmp_path: Path,
) -> None:
    run_state = _make_run_state(
        tmp_path,
        include_mtp=True,
        recover_expert=1,
    )
    report_dir = _write_complete_block_audits(run_state, tmp_path / "reports")
    plan = json.loads(
        (run_state / "ds41rt-gptqmodel-plan.json").read_text(encoding="utf-8")
    )

    report = CANONICAL.validate_block_audit_set(report_dir, plan)

    assert report["block_count"] == 2
    assert report["projection_count"] == 12


def test_canonical_block_audit_set_accepts_mtp_router_topup(
    tmp_path: Path,
) -> None:
    run_state = _make_run_state(
        tmp_path,
        include_mtp=True,
        recover_expert=1,
        router_recovery=True,
    )
    report_dir = _write_complete_block_audits(run_state, tmp_path / "reports")
    plan = json.loads(
        (run_state / "ds41rt-gptqmodel-plan.json").read_text(encoding="utf-8")
    )

    report = CANONICAL.validate_block_audit_set(report_dir, plan)

    assert report["block_count"] == 2
    assert report["projection_count"] == 12


def test_canonical_block_audit_set_accepts_low_positive_mixed_topup(
    tmp_path: Path,
) -> None:
    run_state = _make_run_state(
        tmp_path,
        include_mtp=True,
        recover_expert=1,
        mixed_recovery=True,
    )
    report_dir = _write_complete_block_audits(run_state, tmp_path / "reports")
    plan = json.loads(
        (run_state / "ds41rt-gptqmodel-plan.json").read_text(encoding="utf-8")
    )

    report = CANONICAL.validate_block_audit_set(report_dir, plan)

    assert report["block_count"] == 2
    assert report["projection_count"] == 12


def test_canonical_block_audit_set_accepts_base_learned_router_topup(
    tmp_path: Path,
) -> None:
    run_state = _make_run_state(
        tmp_path,
        include_mtp=True,
        recover_expert=1,
        router_recovery=True,
        recover_namespace="base",
    )
    report_dir = _write_complete_block_audits(run_state, tmp_path / "reports")
    plan = json.loads(
        (run_state / "ds41rt-gptqmodel-plan.json").read_text(encoding="utf-8")
    )

    report = CANONICAL.validate_block_audit_set(report_dir, plan)

    assert report["block_count"] == 2
    assert report["projection_count"] == 12


def test_composite_block_audits_join_base_and_replacement_mtp_plans(
    tmp_path: Path,
) -> None:
    (tmp_path / "base").mkdir()
    (tmp_path / "mtp").mkdir()
    base_run = _make_run_state(tmp_path / "base", include_mtp=True)
    base_plan = json.loads(
        (base_run / "ds41rt-gptqmodel-plan.json").read_text(encoding="utf-8")
    )
    base_reports = _write_complete_block_audits(
        base_run,
        tmp_path / "base-reports",
    )
    mtp_run = _make_run_state(tmp_path / "mtp", overlay=True)
    mtp_plan_path = mtp_run / "ds41rt-gptqmodel-plan.json"
    mtp_plan = json.loads(mtp_plan_path.read_text(encoding="utf-8"))
    mtp_body = {key: value for key, value in mtp_plan.items() if key != "plan_sha256"}
    mtp_body["target_parent"] = {"plan_sha256": base_plan["plan_sha256"]}
    mtp_plan = {
        **mtp_body,
        "plan_sha256": hashlib.sha256(canonical_json_bytes(mtp_body)).hexdigest(),
    }
    _write_json(mtp_plan_path, mtp_plan)
    mtp_reports = tmp_path / "mtp-reports"
    _write_json(
        mtp_reports / "mtp-layer-0-projection-audit.json",
        MODULE.audit_block(
            mtp_run,
            block_namespace="mtp",
            logical_layer=0,
        ),
    )

    report = CANONICAL.validate_composite_block_audit_set(
        base_report_dir=base_reports,
        base_plan=base_plan,
        mtp_report_dir=mtp_reports,
        mtp_plan=mtp_plan,
    )

    assert report["schema"] == CANONICAL.COMPOSITE_BLOCK_AUDIT_SET_SCHEMA
    assert report["base_block_count"] == 1
    assert report["mtp_block_count"] == 1
    assert report["block_count"] == 2
    assert report["projection_count"] == 12
    assert report["expert_family_count"] == 4
    assert report["base"]["plan_sha256"] == base_plan["plan_sha256"]
    assert report["mtp"]["plan_sha256"] == mtp_plan["plan_sha256"]


def test_canonical_block_audit_set_rejects_missing_or_partial_report(
    tmp_path: Path,
) -> None:
    run_state = _make_run_state(tmp_path, include_mtp=True)
    report_dir = _write_complete_block_audits(run_state, tmp_path / "reports")
    plan = json.loads(
        (run_state / "ds41rt-gptqmodel-plan.json").read_text(encoding="utf-8")
    )
    mtp_report = report_dir / "mtp-layer-0-projection-audit.json"
    mtp_report.unlink()

    with pytest.raises(CANONICAL.AssemblyError, match="not one regular file"):
        CANONICAL.validate_block_audit_set(report_dir, plan)

    partial = MODULE.audit_block(
        run_state,
        block_namespace="mtp",
        logical_layer=0,
        require_complete=False,
    )
    partial["status"] = "partial"
    partial_body = {
        key: value for key, value in partial.items() if key != "report_sha256"
    }
    partial["report_sha256"] = hashlib.sha256(
        canonical_json_bytes(partial_body)
    ).hexdigest()
    _write_json(mtp_report, partial)

    with pytest.raises(CANONICAL.AssemblyError, match="complete contract"):
        CANONICAL.validate_block_audit_set(report_dir, plan)


def test_canonical_assembly_streams_complete_base_and_mtp_run(
    tmp_path: Path,
) -> None:
    run_state = _make_run_state(tmp_path, include_mtp=True)
    plan_path = run_state / "ds41rt-gptqmodel-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    # Exercise an actual host/container bind-mount alias. The immutable plan
    # keeps its container paths while the assembler receives the host path.
    planned_run_state = Path("/artifacts/test-run-state")
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    body["run_state_dir"] = str(planned_run_state)
    body["projection_checkpoint"]["root"] = str(
        planned_run_state / "projection-checkpoints"
    )
    body["remote_workers"]["assignment_store"] = str(
        planned_run_state / "dynamic-projection-assignments"
    )
    plan = {
        **body,
        "plan_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }
    _write_json(plan_path, plan)

    generated = _canonical_generated_tensors(run_state)
    written: dict[str, torch.Tensor] = {}
    checkpoint_calls = 0

    def write_tensor(name: str, value: torch.Tensor) -> None:
        assert name not in written
        written[name] = value.clone()

    def checkpoint() -> None:
        nonlocal checkpoint_calls
        checkpoint_calls += 1

    report = CANONICAL.assemble_projection_checkpoints(
        run_state,
        generated_tensors=generated,
        write_generated_tensor=write_tensor,
        expected_plan_sha256=plan["plan_sha256"],
        checkpoint=checkpoint,
        checkpoint_interval=5,
    )

    assert report["schema"] == CANONICAL.PROJECTION_ASSEMBLY_SCHEMA
    assert report["plan_sha256"] == plan["plan_sha256"]
    assert report["materialized_run_state"] == str(run_state)
    assert report["planned_run_state"] == str(planned_run_state)
    assert report["projection_count"] == 12
    assert report["expert_family_count"] == 4
    assert report["generated_tensor_count"] == 48
    assert report["encoded_bytes"] == 12 * 228
    assert report["ownership"] == {"coordinator:cuda:0": 12}
    assert set(written) == {tensor.name for tensor in generated}
    assert checkpoint_calls == 3
    assert len(report["report_sha256"]) == 64


def test_canonical_assembly_streams_authoritative_inline_mixed_tiers(
    tmp_path: Path,
) -> None:
    run_state = _make_inline_mixed_run_state(tmp_path)
    plan = json.loads(
        (run_state / "ds41rt-gptqmodel-plan.json").read_text(encoding="utf-8")
    )
    generated = _canonical_generated_tensors(run_state)
    written: dict[str, torch.Tensor] = {}

    report = CANONICAL.assemble_projection_checkpoints(
        run_state,
        generated_tensors=generated,
        write_generated_tensor=lambda name, value: written.setdefault(
            name, value.clone()
        ),
        expected_plan_sha256=plan["plan_sha256"],
    )

    assert report["projection_count"] == 12
    assert report["encoded_bytes"] == 2 * (3 * 228 + 3 * 292)
    base_report = MODULE.audit_block(
        run_state,
        block_namespace="base",
        logical_layer=0,
    )
    selected_count = base_report["geometry"]["inline_mixed"]["tier_counts"]["3"]
    assert selected_count == 3
    assert (
        sum(
            tensor.shape[-1] == 48
            for name, tensor in written.items()
            if name.startswith("model.layers.0") and name.endswith(".trellis")
        )
        == selected_count
    )


def test_composite_projection_assembly_ignores_parent_mtp_and_uses_overlay(
    tmp_path: Path,
) -> None:
    (tmp_path / "base").mkdir()
    (tmp_path / "mtp").mkdir()
    base_run = _make_run_state(tmp_path / "base", include_mtp=True)
    base_plan = json.loads(
        (base_run / "ds41rt-gptqmodel-plan.json").read_text(encoding="utf-8")
    )
    mtp_run = _make_run_state(tmp_path / "mtp", overlay=True)
    mtp_plan_path = mtp_run / "ds41rt-gptqmodel-plan.json"
    mtp_plan = json.loads(mtp_plan_path.read_text(encoding="utf-8"))
    mtp_body = {key: value for key, value in mtp_plan.items() if key != "plan_sha256"}
    mtp_body["target_parent"] = {"plan_sha256": base_plan["plan_sha256"]}
    mtp_plan = {
        **mtp_body,
        "plan_sha256": hashlib.sha256(canonical_json_bytes(mtp_body)).hexdigest(),
    }
    _write_json(mtp_plan_path, mtp_plan)
    written: dict[str, torch.Tensor] = {}

    report, records = CANONICAL.assemble_composite_projection_checkpoints(
        base_run_state=base_run,
        mtp_run_state=mtp_run,
        generated_tensors=_canonical_generated_tensors(base_run),
        write_generated_tensor=lambda name, value: written.setdefault(
            name, value.clone()
        ),
        checkpoint_interval=5,
    )

    assert report["schema"] == CANONICAL.COMPOSITE_PROJECTION_ASSEMBLY_SCHEMA
    assert report["base_plan_sha256"] == base_plan["plan_sha256"]
    assert report["mtp_plan_sha256"] == mtp_plan["plan_sha256"]
    assert report["projection_count"] == 12
    assert report["expert_family_count"] == 4
    assert report["generated_tensor_count"] == 48
    assert len(records) == 12
    assert len(written) == 48
    assert report["sources"]["base"]["projection_count"] == 6
    assert report["sources"]["mtp"]["projection_count"] == 6
    assert report["sources"]["base"]["excluded_known_state"] == {
        "checkpoint_count": 6,
        "journal_count": 6,
        "assignment_count": 6,
    }
    assert report["sources"]["mtp"]["excluded_known_state"] == {
        "checkpoint_count": 0,
        "journal_count": 0,
        "assignment_count": 0,
    }


def _canonical_quant_config_inputs(tmp_path: Path) -> tuple[object, dict[str, dict]]:
    run_state = _make_run_state(tmp_path, include_mtp=True)
    plan = SimpleNamespace(
        expert_tensor_layout="gptqmodel",
        generated_tensors=_canonical_generated_tensors(run_state),
    )
    storage = CANONICAL.gptqmodel_tensor_storage_for_plan(plan)
    return plan, storage


def test_canonical_quant_config_closes_exact_base_only_mtp_overlay(
    tmp_path: Path,
) -> None:
    plan, storage = _canonical_quant_config_inputs(tmp_path)
    base_storage = {
        name: entry
        for name, entry in storage.items()
        if name.startswith("model.layers.")
    }
    raw = {"bits": 2, "tensor_storage": base_storage}

    canonical, report = CANONICAL.canonical_quant_config(plan, raw)

    added = sorted(set(storage) - set(base_storage))
    assert canonical == {"bits": 2, "tensor_storage": storage}
    assert report["raw_module_count"] == len(base_storage)
    assert report["canonical_module_count"] == len(storage)
    assert report["added_mtp_module_count"] == len(added)
    assert (
        report["added_mtp_modules_sha256"]
        == hashlib.sha256(canonical_json_bytes(added)).hexdigest()
    )
    assert (
        report["raw_quant_config_sha256"]
        == hashlib.sha256(canonical_json_bytes(raw)).hexdigest()
    )
    assert len(report["report_sha256"]) == 64


def test_canonical_quant_config_accepts_already_complete_storage(
    tmp_path: Path,
) -> None:
    plan, storage = _canonical_quant_config_inputs(tmp_path)
    raw = {"bits": 2, "tensor_storage": storage}

    canonical, report = CANONICAL.canonical_quant_config(plan, raw)

    assert canonical == raw
    assert report["raw_module_count"] == len(storage)
    assert report["canonical_module_count"] == len(storage)
    assert report["added_mtp_module_count"] == 0
    assert (
        report["added_mtp_modules_sha256"]
        == hashlib.sha256(canonical_json_bytes([])).hexdigest()
    )


def test_composite_quant_config_binds_distinct_base_and_mtp_provenance(
    tmp_path: Path,
) -> None:
    (tmp_path / "base").mkdir()
    (tmp_path / "mtp").mkdir()
    base_run = _make_run_state(tmp_path / "base", include_mtp=True)
    mtp_run = _make_run_state(tmp_path / "mtp", overlay=True)
    base_plan_path = base_run / "ds41rt-gptqmodel-plan.json"
    base_plan = json.loads(base_plan_path.read_text(encoding="utf-8"))
    base_body = {key: value for key, value in base_plan.items() if key != "plan_sha256"}
    base_body["ledger_provenance"]["run"] = {"source": "base"}
    base_plan = {
        **base_body,
        "plan_sha256": hashlib.sha256(canonical_json_bytes(base_body)).hexdigest(),
    }
    _write_json(base_plan_path, base_plan)
    mtp_plan_path = mtp_run / "ds41rt-gptqmodel-plan.json"
    mtp_plan = json.loads(mtp_plan_path.read_text(encoding="utf-8"))
    mtp_body = {key: value for key, value in mtp_plan.items() if key != "plan_sha256"}
    mtp_body["ledger_provenance"]["run"] = {"source": "mtp"}
    mtp_body["target_parent"] = {"plan_sha256": base_plan["plan_sha256"]}
    mtp_plan = {
        **mtp_body,
        "plan_sha256": hashlib.sha256(canonical_json_bytes(mtp_body)).hexdigest(),
    }
    _write_json(mtp_plan_path, mtp_plan)
    (tmp_path / "artifact").mkdir()
    artifact_plan, storage = _canonical_quant_config_inputs(tmp_path / "artifact")

    quant, report = CANONICAL.canonical_composite_quant_config(
        artifact_plan,
        base_plan=base_plan,
        mtp_plan=mtp_plan,
    )

    provenance = quant["meta"]["ds41rt_error_ledger"]
    assert quant["tensor_storage"] == storage
    assert quant["module_include"] == [CANONICAL.launcher.BASE_EXPERT_PATTERN]
    assert provenance["family_join"] == base_plan["ledger_provenance"]["family_join"]
    assert (
        provenance["projection_sources"]["base"]["plan_sha256"]
        == base_plan["plan_sha256"]
    )
    assert (
        provenance["projection_sources"]["mtp"]["plan_sha256"]
        == mtp_plan["plan_sha256"]
    )
    assert report["schema"] == CANONICAL.COMPOSITE_QUANT_CONFIG_ASSEMBLY_SCHEMA
    assert report["canonical_module_count"] == len(storage)


def test_canonical_quant_config_rejects_partial_mtp_storage(
    tmp_path: Path,
) -> None:
    plan, storage = _canonical_quant_config_inputs(tmp_path)
    base_storage = {
        name: entry
        for name, entry in storage.items()
        if name.startswith("model.layers.")
    }
    mtp_name = next(name for name in storage if name.startswith("mtp."))
    raw = {
        "bits": 2,
        "tensor_storage": {**base_storage, mtp_name: storage[mtp_name]},
    }

    with pytest.raises(
        CANONICAL.AssemblyError,
        match="neither exact base-only nor complete",
    ):
        CANONICAL.canonical_quant_config(plan, raw)


def test_canonical_quant_config_rejects_base_metadata_drift(
    tmp_path: Path,
) -> None:
    plan, storage = _canonical_quant_config_inputs(tmp_path)
    base_storage = {
        name: entry
        for name, entry in storage.items()
        if name.startswith("model.layers.")
    }
    module = next(iter(base_storage))
    base_storage[module] = {**base_storage[module], "bits_per_weight": 3}

    with pytest.raises(
        CANONICAL.AssemblyError,
        match="differs from packed geometry",
    ):
        CANONICAL.canonical_quant_config(
            plan,
            {"bits": 2, "tensor_storage": base_storage},
        )


def test_canonical_assembly_rejects_another_raw_artifact_plan(
    tmp_path: Path,
) -> None:
    run_state = _make_run_state(tmp_path)

    with pytest.raises(
        CANONICAL.AssemblyError,
        match="run-state plan differs from the raw artifact plan",
    ):
        CANONICAL.assemble_projection_checkpoints(
            run_state,
            generated_tensors=_canonical_generated_tensors(run_state),
            write_generated_tensor=lambda _name, _value: None,
            expected_plan_sha256="f" * 64,
        )
