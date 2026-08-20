from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from stage_ds4_hf_snapshot import (  # noqa: E402
    GPTQMODEL_PRODUCTION_SAMPLING_POLICY,
    gptqmodel_publication_identity,
    model_cache_dir,
    quality_snapshot_identity,
    stage_snapshot,
    validate_error_ledger,
)
from ds4rt_runtime.exl3_artifact_contract import (  # noqa: E402
    GPTQMODEL_SOURCE_SERIALIZED_TRELLIS,
    GPTQMODEL_SOURCE_WITH_ROUTE_RECOVERY,
    PLAN_SCHEMA,
    _expected_source_geometry,
    _validate_canonical_assembly,
    _validate_composite_canonical_assembly,
    _valid_recovery_sample_accounting,
    validate_gptqmodel_native_exl3,
)
from ds4rt_runtime.exl3_experts import read_exl3_expert_config  # noqa: E402
from ds4rt_runtime.native_experts import GPTQMODEL_EXPERT_LAYOUT  # noqa: E402
from sync_ds4_hf_snapshot import load_staged_cache  # noqa: E402
from validate_ds4_flash_generation_ab import (  # noqa: E402
    checkpoint_quantization_recipe,
    validate_staged_exl3_checkpoint,
)


RECIPE = "deepseek_v4_exl3_trellis_2bpw_v2"
V3_RECIPE = "deepseek_v4_exl3_trellis_2bpw_v3_flash_activation_pilot"
V4_RECIPE = "deepseek_v4_exl3_trellis_2bpw_v4_flash_natural_route"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def bind_ledger_record(record: dict) -> dict:
    return {
        **record,
        "record_sha256": hashlib.sha256(canonical_json(record)).hexdigest(),
    }


def test_recovery_sample_accounting_accepts_empirical_identity_residual() -> None:
    family = {"run": "mtp-test"}
    family_sha256 = hashlib.sha256(canonical_json(family)).hexdigest()
    route = {"expert_route_count": 300}
    record = {
        "block_namespace": "mtp",
        "logical_layer": 0,
        "expert": 23,
        "provenance": {"family_join": family},
        "zero_route_recovery": {
            "schema": "ds4rt.exl3-zero-route-recovery",
            "schema_version": 1,
            "trigger": "natural-route-count-below-1024",
            "sample_source": "same-fixed-calibration-selection",
            "capture_method": (
                "direct-expert-router-ranks-7-12-then-identity-residual"
            ),
            "selection_policy": "rank-ascending-then-fixed-replay-order-v1",
            "candidate_rank_min": 7,
            "candidate_rank_max": 12,
            "target_sample_count": 1024,
            "identity_calibration_policy": (
                "normalized-2i-residual-to-effective-count-1024-v2"
            ),
            "block_namespace": "mtp",
            "logical_layer": 0,
            "expert": 23,
            "natural_sample_count": 300,
            "router_augmented_sample_count": 381,
            "identity_calibration_count": 343,
            "total_sample_count": 1024,
            "authorization": {
                "schema": "ds4rt.exl3-zero-route-recovery-authorization",
                "schema_version": 1,
                "kind": "immutable-family-join",
                "family_join_sha256": family_sha256,
                "authorization_sha256": family_sha256,
            },
        },
    }

    assert _valid_recovery_sample_accounting(record, route, 1024)
    record["zero_route_recovery"]["identity_calibration_count"] = 342
    assert not _valid_recovery_sample_accounting(record, route, 1024)


def test_composite_assembly_binds_base_and_mtp_projection_sources(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "composite"
    snapshot.mkdir()
    digest = "d" * 64
    base_plan_sha256 = "a" * 64
    mtp_plan_sha256 = "b" * 64
    overlay_sha256 = "c" * 64
    geometry = {
        "num_hidden_layers": 1,
        "n_routed_experts": 6,
        "hidden_size": 32,
        "moe_intermediate_size": 16,
        "dspark_target_layer_ids": [0],
    }
    base_family = {"source": "base"}
    mtp_family = {"source": "mtp"}
    ledger_sources = {
        "base": {
            "plan_sha256": base_plan_sha256,
            "namespaces": ["base"],
            "family_join": base_family,
        },
        "mtp": {
            "plan_sha256": mtp_plan_sha256,
            "namespaces": ["mtp"],
            "family_join": mtp_family,
        },
    }
    provenance = {
        "family_join": base_family,
        "run": {"source": "base"},
        "projection_sources": ledger_sources,
        "projection_sources_sha256": hashlib.sha256(
            canonical_json(ledger_sources)
        ).hexdigest(),
    }
    quant = {
        "bits": 2.0,
        "tensor_storage": gptqmodel_tensor_storage(geometry),
        "meta": {"ds4rt_error_ledger": provenance},
    }
    projections_per_block = geometry["n_routed_experts"] * 3
    projection_bytes = (
        (geometry["hidden_size"] // 16)
        * (geometry["moe_intermediate_size"] // 16)
        * 32
        * 2
        + (geometry["hidden_size"] + geometry["moe_intermediate_size"]) * 2
        + 4
    )

    def source_report(namespace: str, plan_sha256: str) -> dict:
        return bind_record(
            {
                "schema": "ds4rt-exl3-canonical-hybrid-projection-assembly-v1",
                "plan_sha256": plan_sha256,
                "namespaces": [namespace],
                "materialized_run_state": f"/materialized/{namespace}",
                "planned_run_state": f"/planned/{namespace}",
                "expert_tensor_layout": "gptqmodel",
                "retained_tensor_layout": "source_checkpoint_native_byte_exact",
                "projection_count": projections_per_block,
                "expert_family_count": geometry["n_routed_experts"],
                "generated_tensor_count": projections_per_block * 4,
                "encoded_bytes": projections_per_block * projection_bytes,
                "ownership": {"coordinator:cuda:0": projections_per_block},
                "excluded_known_state": {
                    "checkpoint_count": 0,
                    "journal_count": 0,
                    "assignment_count": 0,
                },
                "content": {
                    "checkpoint_records_sha256": digest,
                    "assignment_records_sha256": digest,
                    "journal_records_sha256": digest,
                },
            },
            "report_sha256",
        )

    source_reports = {
        "base": source_report("base", base_plan_sha256),
        "mtp": source_report("mtp", mtp_plan_sha256),
    }
    projection = bind_record(
        {
            "schema": (
                "ds4rt-exl3-canonical-hybrid-composite-projection-assembly-v1"
            ),
            "base_plan_sha256": base_plan_sha256,
            "mtp_plan_sha256": mtp_plan_sha256,
            "sources": source_reports,
            "expert_tensor_layout": "gptqmodel",
            "retained_tensor_layout": "source_checkpoint_native_byte_exact",
            "projection_count": projections_per_block * 2,
            "expert_family_count": geometry["n_routed_experts"] * 2,
            "generated_tensor_count": projections_per_block * 2 * 4,
            "encoded_bytes": projections_per_block * 2 * projection_bytes,
            "ownership": {"coordinator:cuda:0": projections_per_block * 2},
            "content": {
                "source_reports_sha256": hashlib.sha256(
                    canonical_json(source_reports)
                ).hexdigest(),
                "projection_records_sha256": digest,
            },
        },
        "report_sha256",
    )

    def audit_subset(namespace: str, plan_sha256: str) -> dict:
        filename = f"{namespace}-layer-0-projection-audit.json"
        reports = [
            {
                "ordinal": 1,
                "block_namespace": namespace,
                "logical_layer": 0,
                "filename": filename,
                "file_sha256": digest,
                "report_sha256": digest,
            }
        ]
        return bind_record(
            {
                "schema": "ds4rt-exl3-independent-block-audit-set-v1",
                "plan_sha256": plan_sha256,
                "base_block_count": 1 if namespace == "base" else 0,
                "mtp_block_count": 1 if namespace == "mtp" else 0,
                "block_count": 1,
                "projections_per_block": projections_per_block,
                "projection_count": projections_per_block,
                "expert_family_count": geometry["n_routed_experts"],
                "reports": reports,
                "reports_sha256": hashlib.sha256(canonical_json(reports)).hexdigest(),
            },
            "report_sha256",
        )

    base_audits = audit_subset("base", base_plan_sha256)
    mtp_audits = audit_subset("mtp", mtp_plan_sha256)
    block_audits = bind_record(
        {
            "schema": "ds4rt-exl3-independent-composite-block-audit-set-v1",
            "base": base_audits,
            "mtp": mtp_audits,
            "base_plan_sha256": base_plan_sha256,
            "mtp_plan_sha256": mtp_plan_sha256,
            "base_block_count": 1,
            "mtp_block_count": 1,
            "block_count": 2,
            "projections_per_block": projections_per_block,
            "projection_count": projections_per_block * 2,
            "expert_family_count": geometry["n_routed_experts"] * 2,
        },
        "report_sha256",
    )
    quant_report = bind_record(
        {
            "schema": (
                "ds4rt-exl3-canonical-composite-quant-config-assembly-v1"
            ),
            "base_plan_sha256": base_plan_sha256,
            "mtp_plan_sha256": mtp_plan_sha256,
            "canonical_module_count": projections_per_block * 2,
            "tensor_storage_sha256": hashlib.sha256(
                canonical_json(quant["tensor_storage"])
            ).hexdigest(),
            "ledger_provenance_sha256": hashlib.sha256(
                canonical_json(provenance)
            ).hexdigest(),
            "canonical_quant_config_sha256": hashlib.sha256(
                canonical_json(quant)
            ).hexdigest(),
        },
        "report_sha256",
    )
    projection_sources = {
        "base": {
            "plan_sha256": base_plan_sha256,
            "family_join_sha256": hashlib.sha256(
                canonical_json(base_family)
            ).hexdigest(),
        },
        "mtp": {
            "plan_sha256": mtp_plan_sha256,
            "family_join_sha256": hashlib.sha256(
                canonical_json(mtp_family)
            ).hexdigest(),
            "overlay_sha256": overlay_sha256,
            "overlay_run_sha256": "e" * 64,
        },
    }
    assembly = bind_record(
        {
            "schema": "ds4rt-exl3-canonical-hybrid-assembly-v2",
            "recipe": V4_RECIPE,
            "source_snapshot": {"geometry": geometry},
            "projection_sources": projection_sources,
            "block_audit_set": block_audits,
            "quant_config_assembly": quant_report,
            "projection_assembly": projection,
        },
        "report_sha256",
    )
    write_json(snapshot / "ds4rt-exl3-canonical-assembly.json", assembly)
    declaration = {
        "schema": "ds4rt-exl3-canonical-hybrid-assembly-v2",
        "filename": "ds4rt-exl3-canonical-assembly.json",
        "report_sha256": assembly["report_sha256"],
        "base_plan_sha256": base_plan_sha256,
        "mtp_plan_sha256": mtp_plan_sha256,
        "mtp_overlay_sha256": overlay_sha256,
    }
    plan = {"source": {"geometry": geometry}, "canonical_assembly": declaration}
    assembly_payload = (snapshot / "ds4rt-exl3-canonical-assembly.json").read_bytes()
    artifact = {
        "files": {
            "ds4rt-exl3-canonical-assembly.json": {
                "sha256": hashlib.sha256(assembly_payload).hexdigest()
            }
        }
    }

    assert _validate_composite_canonical_assembly(
        snapshot,
        {**geometry, "quantization_config": quant},
        quant,
        plan,
        artifact,
        declaration,
    ) == {"base": base_family, "mtp": mtp_family}


def write_error_ledger(
    root: Path,
    *,
    family_join: dict | None = None,
) -> None:
    bound_family_join = family_join or {"run": "test-v4"}
    records = []
    for expert in range(6):
        route_evidence = {
            "schema": "ds4rt.exl3-natural-route",
            "schema_version": 1,
            "block_namespace": "base",
            "logical_layer": 0,
            "expert": expert,
            "router_calls": 8,
            "router_token_count": 1024,
            "router_selected_route_count": 6144,
            "router_top_k": 6,
            "expert_route_count": 1024,
            "expert_gate_weight_sum": 128.0,
            "expert_gate_squared_mass": 16.0,
            "total_gate_weight_sum": 768.0,
            "total_gate_squared_mass": 96.0,
            "expert_route_fraction": 1 / 6,
            "expert_gate_weight_mass_fraction": 1 / 6,
            "expert_gate_squared_mass_fraction": 1 / 6,
            "expert_gate_weight_mean": 0.125,
            "expert_gate_weight_rms": 0.125,
            "router_weight_dtypes": ["torch.float32"],
            "mask_modes": ["all-valid"],
        }
        for projection, module_projection in (
            ("w1", "gate_proj"),
            ("w2", "down_proj"),
            ("w3", "up_proj"),
        ):
            records.append(
                bind_ledger_record(
                    {
                        "schema": "ds4rt.exl3-error-ledger",
                        "schema_version": 1,
                        "record_kind": "projection",
                        "module": f"model.layers.0.mlp.experts.{expert}.{module_projection}",
                        "processor_layer_index": 0,
                        "block_namespace": "base",
                        "logical_layer": 0,
                        "expert": expert,
                        "projection": projection,
                        "bits": 2,
                        "codebook": "mcg",
                        "sample_count": 1024,
                        "duration_seconds": 1.0,
                        "encoded_bytes": 128,
                        "devices": ["cuda:0"],
                        "provenance": {"family_join": bound_family_join},
                        "route_evidence": route_evidence,
                        "quantizer_metrics": {
                            "schema": "gptqmodel.exl3-trellis-error",
                            "schema_version": 1,
                            "quantizer_path": "hessian_ldlq",
                            "reported_metric_kind": "hessian_weighted_relative_error",
                            "reported_metric_value": 0.1,
                            "hessian_sample_count": 1024,
                            "hessian_metric_status": "ok",
                            "hessian_regularization_sigma": 0.025,
                            "hessian_numerical_contract": (
                                "signed-block-hadamard-congruence-fp64-v1"
                            ),
                            "hessian_transform_compute_dtype": "torch.float64",
                            "hessian_storage_dtype": "torch.float32",
                            "hessian_regularization_placement": (
                                "before-fp64-congruence"
                            ),
                            "hessian_regularization_diagonal_addend": 0.00018,
                            "hessian_symmetry_restoration": (
                                "mean-with-transpose-fp64"
                            ),
                            "hessian_symmetry_correction_max_abs": 1e-12,
                            "hessian_weighted_error_numerator": 2.0,
                            "hessian_weighted_reference_denominator": 20.0,
                            "hessian_weighted_relative_error": 0.1,
                            "reconstruction": {
                                "domain": "regularized_exl3_search_space",
                                "reference_finite": True,
                                "error_finite": True,
                                "element_count": 256,
                                "error_sum_sq": 4.0,
                                "reference_sum_sq": 40.0,
                                "mse": 0.015625,
                                "nmse": 0.1,
                                "relative_frobenius": 0.316227766,
                                "mean_abs_error": 0.01,
                                "max_abs_error": 0.2,
                            },
                        },
                    }
                )
            )
        records.append(
            bind_ledger_record(
                {
                    "schema": "ds4rt.exl3-error-ledger",
                    "schema_version": 1,
                    "record_kind": "expert_family",
                    "block_namespace": "base",
                    "logical_layer": 0,
                    "expert": expert,
                    "bits": 2,
                    "codebook": "mcg",
                    "projections": ["w1", "w2", "w3"],
                    "provenance": {"family_join": bound_family_join},
                    "route_evidence": route_evidence,
                    "aggregate_metrics": {
                        "hessian_metric_status": "ok",
                        "error_sum_sq": 12.0,
                        "reference_sum_sq": 120.0,
                        "element_count": 768,
                        "mse": 0.015625,
                        "nmse": 0.1,
                        "relative_frobenius": 0.316227766,
                        "hessian_weighted_error_numerator": 6.0,
                        "hessian_weighted_reference_denominator": 60.0,
                        "hessian_weighted_relative_error": 0.1,
                    },
                }
            )
        )
    payload = b"".join(canonical_json(record) + b"\n" for record in records)
    (root / "ds4rt-exl3-error-ledger.jsonl").write_bytes(payload)
    write_json(
        root / "ds4rt-exl3-error-ledger.manifest.json",
        {
            "schema": "ds4rt.exl3-error-ledger",
            "schema_version": 1,
            "ledger": "ds4rt-exl3-error-ledger.jsonl",
            "ledger_sha256": hashlib.sha256(payload).hexdigest(),
            "projection_records": 18,
            "complete_family_records": 6,
            "total_records": 24,
        },
    )


def checkpoint_identity(root: Path) -> dict:
    metadata = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "config.json",
            "model.safetensors.index.json",
            "quantize_config.json",
            "ds4rt-exl3-calibration.json",
            "ds4rt-gptqmodel-plan.json",
            "ds4rt-gptqmodel-run.json",
            "ds4rt-gptqmodel-artifact.json",
            "ds4rt-exl3-error-ledger.manifest.json",
        )
        if (root / name).is_file()
    }
    shards = []
    for path in sorted(root.glob("*.safetensors")):
        file_stat = path.stat()
        shards.append(
            {
                "name": path.name,
                "device": file_stat.st_dev,
                "inode": file_stat.st_ino,
                "size": file_stat.st_size,
                "mtime_ns": file_stat.st_mtime_ns,
            }
        )
    return {"path": str(root.resolve()), "metadata_sha256": metadata, "shards": shards}


def make_artifact(
    root: Path,
    *,
    shard_payload: bytes = b"quantized",
    multigpu: bool = False,
    recipe: str = RECIPE,
) -> Path:
    root.mkdir()
    quantization = {
        "quant_method": "exl3",
        "ds4rt": {"calibrated": True, "recipe": recipe},
    }
    write_json(
        root / "config.json",
        {
            "num_hidden_layers": 1,
            "dspark_target_layer_ids": [],
            "n_routed_experts": 6,
            "hidden_size": 128,
            "moe_intermediate_size": 256,
            "quantization_config": quantization,
        },
    )
    write_json(root / "quantization_config.json", quantization)
    calibration = {
        "schema": "test",
        "recipe": recipe,
        "seed": 20260805,
        "layers": [{"layer_id": 0}],
    }
    if multigpu:
        calibration["multigpu_qualification"] = {
            "schema": "ds4rt-exl3-multigpu-qualification-v1",
            "status": "bit-exact",
            "devices": [0, 1],
            "device_ratios": None,
            "seed": 20260805,
            "batches": 4,
            "tensors_per_batch": 4,
            "shape": [128, 256],
        }
        write_json(
            root / "ds4rt-exl3-multigpu-production.json",
            {
                **calibration["multigpu_qualification"],
                "production_projection_shapes": [[128, 256], [256, 128]],
            },
        )
    write_json(
        root / "ds4rt-exl3-calibration.json",
        calibration,
    )
    quality = {
        "schema": "ds4rt-exl3-checkpoint-quality-v1",
        "expert_ids": [0, 1, 2, 3, 4, 5],
        "rows": 16,
        "seed": 20260805,
        "thresholds": {
            "min_quant_cosine": 0.88,
            "max_quant_relative_l2": 0.50,
            "min_tp_cosine": 0.999,
            "max_tp_relative_l2": 0.03,
        },
        "summary": {"layers": 1, "all_tp4_ranks_equal_source_bytes": True},
        "layers": [
            {
                "layer_id": 0,
                "native_to_exl3": {"cosine": 0.90, "relative_l2": 0.45},
                "exl3_tp4_to_unsharded": {
                    "cosine": 0.9999,
                    "relative_l2": 0.001,
                },
                "tp4_equal_rank_source_bytes": True,
                "tp4_rank_source_bytes": [600, 600, 600, 600],
            }
        ],
    }
    write_json(root / "ds4rt-exl3-quality.json", quality)
    write_json(
        root / "ds4rt-exl3-retained-native.json",
        {
            "schema": "ds4rt-exl3-retained-native-integrity-v1",
            "recipe": recipe,
            "quantization_scope": "routed_experts_only",
            "retained_tensor_count": 1,
            "retained_bytes": 1,
            "generated_exl3_tensor_count": 72,
            "generated_exl3_bytes": 72,
            "generated_exl3_metadata_verified": True,
            "generated_exl3_mcg_tensor_count": 18,
            "generated_exl3_mcg_markers_verified": True,
            "artifact_tensor_count": 73,
            "artifact_bytes": 73,
            "strict_tp4_source_layout": {
                "world_size": 4,
                "blocks": 1,
                "experts_per_block": 6,
                "experts_checked": 6,
                "local_intermediate_size": 128,
                "rank_source_bytes_per_expert": [100, 100, 100, 100],
                "rank_source_bytes_per_block": [600, 600, 600, 600],
                "rank_source_bytes_total": [600, 600, 600, 600],
                "equal_rank_source_bytes": True,
            },
            "aggregate_sha256": "0" * 64,
            "tensors": [
                {
                    "name": "norm.weight",
                    "dtype": "F16",
                    "shape": [1],
                    "bytes": 1,
                    "sha256": "1" * 64,
                }
            ],
        },
    )
    write_json(
        root / "model.safetensors.index.json",
        {"weight_map": {"layers.0.weight": "model-00001-of-00001.safetensors"}},
    )
    (root / "model-00001-of-00001.safetensors").write_bytes(shard_payload)
    (root / "tokenizer.json").write_text('{"version":"test"}\n', encoding="utf-8")
    (root / ".gitattributes").write_text("*.safetensors binary\n", encoding="utf-8")
    if recipe == V4_RECIPE:
        write_error_ledger(root)
    debug = root / ".ds4rt-exl3-debug"
    debug.mkdir()
    (debug / "scratch.bin").write_bytes(b"not staged")
    contract = {
        "native": checkpoint_identity(root),
        "exl3": checkpoint_identity(root),
        "layer_ids": [0],
        "expert_ids": quality["expert_ids"],
        "rows": quality["rows"],
        "seed": quality["seed"],
        "allow_incomplete": False,
        "thresholds": quality["thresholds"],
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    quality["validation_contract"] = {
        **contract,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }
    write_json(root / "ds4rt-exl3-quality.json", quality)
    return root


def bind_record(record: dict, field: str) -> dict:
    return {
        **record,
        field: hashlib.sha256(canonical_json(record)).hexdigest(),
    }


def quality_gpu_identity() -> dict:
    return bind_record(
        {
            "schema": "ds4rt-quality-physical-gpu-v1",
            "physical_role": "coordinator-gpu0",
            "cuda_visible_devices": "GPU-95f8f212-9131-df99-fd53-7535965197d7",
            "quantization_preflight": {
                "index": 0,
                "uuid": "GPU-95f8f212-9131-df99-fd53-7535965197d7",
                "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
                "driver_version": "595.84",
                "compute_capability": [12, 0],
            },
            "validation_inventory": {
                "uuid": "GPU-95f8f212-9131-df99-fd53-7535965197d7",
                "pci_bus_id": "00000000:11:00.0",
                "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
                "driver_version": "595.84",
                "compute_capability": "12.0",
            },
        },
        "sha256",
    )


def gptqmodel_tensor_storage(geometry: dict) -> dict:
    storage = {}
    hidden_size = geometry["hidden_size"]
    intermediate_size = geometry["moe_intermediate_size"]
    blocks = [
        f"model.layers.{layer}"
        for layer in range(geometry["num_hidden_layers"])
    ] + [
        f"mtp.{layer}"
        for layer in range(len(geometry.get("dspark_target_layer_ids", [])))
    ]
    for block in blocks:
        for expert in range(geometry["n_routed_experts"]):
            for projection, (input_size, output_size) in {
                "gate_proj": (hidden_size, intermediate_size),
                "up_proj": (hidden_size, intermediate_size),
                "down_proj": (intermediate_size, hidden_size),
            }.items():
                module = f"{block}.mlp.experts.{expert}.{projection}"
                storage[module] = {
                    "stored_tensors": {
                        f"{module}.trellis": {
                            "shape": [input_size // 16, output_size // 16, 32],
                            "torch_dtype": "int16",
                        },
                        f"{module}.suh": {
                            "shape": [input_size],
                            "torch_dtype": "float16",
                        },
                        f"{module}.svh": {
                            "shape": [output_size],
                            "torch_dtype": "float16",
                        },
                        f"{module}.mcg": {
                            "shape": [],
                            "torch_dtype": "int32",
                        },
                    },
                    "quant_format": "exl3",
                    "bits_per_weight": 2,
                    "mcg_multiplier": 0xCBAC1FED,
                }
    return storage


def make_gptqmodel_artifact(root: Path) -> Path:
    root = make_artifact(root, recipe=V4_RECIPE)
    debug = root / ".ds4rt-exl3-debug"
    (debug / "scratch.bin").unlink()
    debug.rmdir()
    digest = "1" * 64
    coordinator_image = "sha256:" + "2" * 64
    worker_image = "sha256:" + "3" * 64
    geometry = {
        "num_hidden_layers": 1,
        "n_routed_experts": 6,
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
        "dspark_target_layer_ids": [],
    }
    family_join = {
        "recipe": V4_RECIPE,
        "bits": 2,
        "codebook": "mcg",
        "module_include": (
            r"^model\.layers\.\d+\.mlp\.experts\.\d+\."
            r"(?:gate_proj|up_proj|down_proj)$"
        ),
        "quantizer_seed": 787,
        "quantizer_numerics": {
            "sigma_reg": 0.025,
            "hessian_capture": "raw-xtx-sum-fp32-v1",
            "hessian_numerical": "signed-block-hadamard-congruence-fp64-v1",
            "hessian_symmetry": "mean-with-transpose-fp64",
        },
        "operator_contract": "ds4rt-deepseek-v4-target-plus-joint-mtp-v1",
        "route_evidence_contract": "ds4rt.exl3-natural-route",
        "gptqmodel": {
            "schema": 1,
            "repository": "https://github.com/tpurtell/GPTQModel.git",
            "revision": "6f853e66f692cf623a2781f4e4fe05078f57f999",
            "source_tree_sha256": (
                "bc862c87a9b28ee770267e69c663f0b9c0829f7cd1ee57e889ef5bffde7b7e20"
            ),
        },
        "corpus": {
            "examples": 1,
            "utf8_bytes": 4,
            "file_sha256": digest,
            "normalized_stream_sha256": digest,
        },
        "source": {
            "revision": "4" * 40,
            "config_sha256": digest,
            "index_sha256": digest,
            "geometry": geometry,
        },
        "execution_topology": {
            "contract": "ds4rt.exl3-remote-worker-v1",
            "scheduler": "dynamic-pipelined-slot-projection-v2",
            "coordinator": {
                "image_digest": coordinator_image,
                "preflight_sha256": digest,
            },
            "coordinator_slots": [
                {
                    "device": "cuda:0",
                    "gpu_uuid": "GPU-0",
                    "image_digest": coordinator_image,
                    "preflight_sha256": digest,
                },
                {
                    "device": "cuda:1",
                    "gpu_uuid": "GPU-1",
                    "image_digest": coordinator_image,
                    "preflight_sha256": digest,
                },
            ],
            "workers": [
                {
                    "name": name,
                    "image_digest": worker_image,
                    "preflight_sha256": digest,
                }
                for name in ("dodo", "emu", "kiwi", "ostrich")
            ],
        },
    }
    provenance = {"family_join": family_join, "run": {"target_batch_size": 1}}
    quantization = {
        "bits": 2.0,
        "group_size": -1,
        "desc_act": False,
        "method": "exl3",
        "quant_method": "exl3",
        "format": "exl3",
        "checkpoint_format": "exl3",
        "out_scales": "auto",
        "codebook": "mcg",
        "module_include": [family_join["module_include"]],
        "tensor_storage": gptqmodel_tensor_storage(geometry),
        "meta": {"fallback": None, "ds4rt_error_ledger": provenance},
    }
    config = {
        **geometry,
        "model_type": "deepseek_v4",
        "num_experts_per_tok": 6,
        "swiglu_limit": 10.0,
        "quantization_config": quantization,
    }
    write_json(root / "config.json", config)
    write_json(root / "quantize_config.json", quantization)
    write_error_ledger(root, family_join=family_join)
    source_plan_body = {
        "schema": "ds4rt-deepseek-v4-gptqmodel-plan-v5",
        "recipe": V4_RECIPE,
        "source": {"geometry": geometry},
        "exl3": {
            "bits": 2,
            "codebook": "mcg",
            "seed": 787,
            "module_include": [family_join["module_include"]],
            "fallback": None,
            "out_scales": "auto",
            "sigma_reg": 0.025,
            "hessian_capture": "raw-xtx-sum-fp32-v1",
            "hessian_numerical": "signed-block-hadamard-congruence-fp64-v1",
            "hessian_symmetry": "mean-with-transpose-fp64",
        },
        "ledger_provenance": provenance,
    }
    source_plan = bind_record(source_plan_body, "plan_sha256")
    source_artifact_sha256 = "a" * 64
    projection_count = (
        geometry["num_hidden_layers"]
        * geometry["n_routed_experts"]
        * 3
    )
    projection_bytes = (
        (geometry["hidden_size"] // 16)
        * (geometry["moe_intermediate_size"] // 16)
        * 32
        * 2
        + (geometry["hidden_size"] + geometry["moe_intermediate_size"]) * 2
        + 4
    )
    projection_assembly = bind_record(
        {
            "schema": "ds4rt-exl3-canonical-hybrid-projection-assembly-v1",
            "plan_sha256": source_plan["plan_sha256"],
            "materialized_run_state": str((root / "run-state").resolve()),
            "planned_run_state": "/artifacts/run-state",
            "expert_tensor_layout": "gptqmodel",
            "retained_tensor_layout": "source_checkpoint_native_byte_exact",
            "projection_count": projection_count,
            "expert_family_count": projection_count // 3,
            "generated_tensor_count": projection_count * 4,
            "encoded_bytes": projection_count * projection_bytes,
            "ownership": {
                "coordinator:cuda:0": 3,
                "coordinator:cuda:1": 3,
                "remote_worker:dodo": 3,
                "remote_worker:emu": 3,
                "remote_worker:kiwi": 3,
                "remote_worker:ostrich": 3,
            },
            "content": {
                "checkpoint_records_sha256": digest,
                "assignment_records_sha256": digest,
                "journal_records_sha256": digest,
            },
        },
        "report_sha256",
    )
    block_reports = [
        {
            "ordinal": 1,
            "block_namespace": "base",
            "logical_layer": 0,
            "filename": "base-layer-0-projection-audit.json",
            "file_sha256": digest,
            "report_sha256": digest,
        }
    ]
    block_audit_set = bind_record(
        {
            "schema": "ds4rt-exl3-independent-block-audit-set-v1",
            "plan_sha256": source_plan["plan_sha256"],
            "base_block_count": 1,
            "mtp_block_count": 0,
            "block_count": 1,
            "projections_per_block": geometry["n_routed_experts"] * 3,
            "projection_count": projection_count,
            "expert_family_count": projection_count // 3,
            "reports": block_reports,
            "reports_sha256": hashlib.sha256(
                canonical_json(block_reports)
            ).hexdigest(),
        },
        "report_sha256",
    )
    quant_config_sha256 = hashlib.sha256(canonical_json(quantization)).hexdigest()
    quant_config_assembly = bind_record(
        {
            "schema": "ds4rt-exl3-canonical-quant-config-assembly-v1",
            "raw_module_count": projection_count,
            "canonical_module_count": projection_count,
            "added_mtp_module_count": 0,
            "added_mtp_modules_sha256": hashlib.sha256(
                canonical_json([])
            ).hexdigest(),
            "raw_quant_config_sha256": quant_config_sha256,
            "canonical_quant_config_sha256": quant_config_sha256,
        },
        "report_sha256",
    )
    assembly = bind_record(
        {
            "schema": "ds4rt-exl3-canonical-hybrid-assembly-v1",
            "recipe": V4_RECIPE,
            "materialized_raw_artifact": str((root / "raw").resolve()),
            "source_plan_sha256": source_plan["plan_sha256"],
            "source_run_sha256": "b" * 64,
            "source_artifact_manifest_sha256": source_artifact_sha256,
            "source_snapshot": source_plan_body["source"],
            "block_audit_set": block_audit_set,
            "quant_config_assembly": quant_config_assembly,
            "projection_assembly": projection_assembly,
        },
        "report_sha256",
    )
    write_json(root / "ds4rt-exl3-canonical-assembly.json", assembly)
    plan_body = {
        **source_plan_body,
        "canonical_assembly": {
            "schema": "ds4rt-exl3-canonical-hybrid-assembly-v1",
            "filename": "ds4rt-exl3-canonical-assembly.json",
            "report_sha256": assembly["report_sha256"],
            "source_plan_sha256": source_plan["plan_sha256"],
            "source_artifact_manifest_sha256": source_artifact_sha256,
        },
    }
    plan = bind_record(plan_body, "plan_sha256")
    write_json(root / "ds4rt-gptqmodel-plan.json", plan)
    excluded = {"ds4rt-gptqmodel-artifact.json", "ds4rt-gptqmodel-run.json"}
    files = {}
    for path in sorted(root.iterdir()):
        if path.name in excluded:
            continue
        payload = path.read_bytes()
        files[path.name] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    artifact = bind_record(
        {
            "schema": "ds4rt-deepseek-v4-gptqmodel-artifact-v1",
            "plan_sha256": plan["plan_sha256"],
            "files": files,
            "file_count": len(files),
            "total_bytes": sum(record["bytes"] for record in files.values()),
        },
        "manifest_sha256",
    )
    write_json(root / "ds4rt-gptqmodel-artifact.json", artifact)
    run = bind_record(
        {
            "schema": "ds4rt-deepseek-v4-gptqmodel-run-v5",
            "status": "complete",
            "plan_sha256": plan["plan_sha256"],
            "artifact_manifest_sha256": artifact["manifest_sha256"],
            "mtp_replay_batches": 1,
        },
        "run_sha256",
    )
    write_json(root / "ds4rt-gptqmodel-run.json", run)
    return root


def make_gptqmodel_qualification(
    artifact: Path,
    report_root: Path,
) -> tuple[Path, Path]:
    report_root.mkdir()
    retained = json.loads(
        (artifact / "ds4rt-exl3-retained-native.json").read_text()
    )
    retained.update(
        {
            "recipe": V4_RECIPE,
            "native_snapshot": str((report_root / "native").resolve()),
            "exl3_snapshot": str(artifact.resolve()),
            "gptqmodel_publication": gptqmodel_publication_identity(artifact),
        }
    )
    retained_path = report_root / "retained-native.json"
    write_json(retained_path, retained)

    quality = json.loads((artifact / "ds4rt-exl3-quality.json").read_text())
    digest = "1" * 64
    heldout = {
        "path": str((report_root / "heldout" / "manifest.json").resolve()),
        "sha256": digest,
    }
    execution_gpu = quality_gpu_identity()
    for layer in quality["layers"]:
        layer["execution_gpu_sha256"] = execution_gpu["sha256"]
        layer["sampling_mode"] = "natural-stratified"
        layer["sampling_policy"] = GPTQMODEL_PRODUCTION_SAMPLING_POLICY
        layer["heldout_activation_evidence"] = {
            "input_mode": "checkpoint_bound_captured_natural_selected_expert_isolated",
            "route_source": "native_capture_sidecar",
        }
        layer["natural_selected_experts"] = [
            {"expert_id": expert_id, "rows": quality["rows"]}
            for expert_id in quality["expert_ids"]
        ]
    contract = {
        "native": {"path": str((report_root / "native").resolve())},
        "exl3": quality_snapshot_identity(artifact),
        "layer_ids": [0],
        "expert_ids": quality["expert_ids"],
        "rows": quality["rows"],
        "seed": quality["seed"],
        "allow_incomplete": False,
        "sampling_mode": GPTQMODEL_PRODUCTION_SAMPLING_POLICY,
        "execution_gpu": execution_gpu,
        "heldout_activation_manifest": heldout,
        "calibration_disjointness": {
            "calibration_jsonl": {
                "path": str((report_root / "calibration.jsonl").resolve()),
                "sha256": digest,
                "prompts": 1,
            },
            "heldout_activation_manifest": {**heldout, "prompts": 1},
            "prompt_sha256_overlap": 0,
        },
        "thresholds": quality["thresholds"],
    }
    quality["validation_contract"] = bind_record(contract, "sha256")
    quality_path = report_root / "expert-quality.json"
    write_json(quality_path, quality)
    return retained_path, quality_path


def rebind_gptqmodel_publication(root: Path, plan: dict) -> None:
    write_json(root / "ds4rt-gptqmodel-plan.json", plan)
    excluded = {"ds4rt-gptqmodel-artifact.json", "ds4rt-gptqmodel-run.json"}
    files = {}
    for path in sorted(root.iterdir()):
        if path.name in excluded:
            continue
        payload = path.read_bytes()
        files[path.name] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    artifact = bind_record(
        {
            "schema": "ds4rt-deepseek-v4-gptqmodel-artifact-v1",
            "plan_sha256": plan["plan_sha256"],
            "files": files,
            "file_count": len(files),
            "total_bytes": sum(record["bytes"] for record in files.values()),
        },
        "manifest_sha256",
    )
    write_json(root / "ds4rt-gptqmodel-artifact.json", artifact)
    write_json(
        root / "ds4rt-gptqmodel-run.json",
        bind_record(
            {
                "schema": "ds4rt-deepseek-v4-gptqmodel-run-v5",
                "status": "complete",
                "plan_sha256": plan["plan_sha256"],
                "artifact_manifest_sha256": artifact["manifest_sha256"],
                "mtp_replay_batches": 1,
            },
            "run_sha256",
        ),
    )


def test_hardlink_stage_is_resolver_compatible_and_content_addressed(
    tmp_path: Path,
) -> None:
    artifact = make_artifact(tmp_path / "artifact")
    hf_home = tmp_path / "hf"

    result = stage_snapshot(artifact, "tpurtell/flash-exl3", hf_home)

    model_root = model_cache_dir(hf_home.resolve(), "tpurtell/flash-exl3")
    revision = result["revision"]
    assert (model_root / "refs/main").read_text(encoding="utf-8") == f"{revision}\n"
    staged = model_root / "snapshots" / revision
    shard_link = staged / "model-00001-of-00001.safetensors"
    assert shard_link.is_symlink()
    assert not os.path.isabs(os.readlink(shard_link))
    assert shard_link.read_bytes() == b"quantized"
    shard_blob = shard_link.resolve()
    assert shard_blob.parent == model_root / "blobs"
    assert shard_blob.stat().st_ino == (artifact / shard_link.name).stat().st_ino
    assert not (staged / ".ds4rt-exl3-debug").exists()
    assert (model_root / "ds4rt-manifests" / f"{revision}.json").is_file()


def test_gptqmodel_route_recovery_source_requires_its_recovery_contract(
    tmp_path: Path,
) -> None:
    artifact = make_gptqmodel_artifact(tmp_path / "gptqmodel-artifact")
    config = json.loads((artifact / "config.json").read_text())
    quant = config["quantization_config"]
    family = quant["meta"]["ds4rt_error_ledger"]["family_join"]
    family["gptqmodel"] = GPTQMODEL_SOURCE_WITH_ROUTE_RECOVERY
    family["zero_route_recovery_contract"] = "ds4rt.exl3-zero-route-recovery"

    assert validate_gptqmodel_native_exl3(quant) == family

    family.pop("zero_route_recovery_contract")
    with pytest.raises(ValueError, match="qualified natural-route recipe"):
        validate_gptqmodel_native_exl3(quant)


def test_gptqmodel_serialized_trellis_source_is_exactly_qualified(
    tmp_path: Path,
) -> None:
    artifact = make_gptqmodel_artifact(tmp_path / "gptqmodel-artifact")
    config = json.loads((artifact / "config.json").read_text())
    quant = config["quantization_config"]
    family = quant["meta"]["ds4rt_error_ledger"]["family_join"]
    family["gptqmodel"] = GPTQMODEL_SOURCE_SERIALIZED_TRELLIS
    family["zero_route_recovery_contract"] = "ds4rt.exl3-zero-route-recovery"

    assert validate_gptqmodel_native_exl3(quant) == family

    family["gptqmodel"] = {**GPTQMODEL_SOURCE_SERIALIZED_TRELLIS, "revision": "0" * 40}
    with pytest.raises(ValueError, match="qualified natural-route recipe"):
        validate_gptqmodel_native_exl3(quant)


def test_gptqmodel_serialized_trellis_accepts_bound_two_rtx_topology(
    tmp_path: Path,
) -> None:
    artifact = make_gptqmodel_artifact(tmp_path / "gptqmodel-artifact")
    config = json.loads((artifact / "config.json").read_text())
    quant = config["quantization_config"]
    provenance = quant["meta"]["ds4rt_error_ledger"]
    family = provenance["family_join"]
    family["gptqmodel"] = GPTQMODEL_SOURCE_SERIALIZED_TRELLIS
    family["zero_route_recovery_contract"] = "ds4rt.exl3-zero-route-recovery"
    family.pop("execution_topology")
    family["image_digest"] = "sha256:" + "2" * 64
    family["preflight_sha256"] = "1" * 64
    provenance["run"] = {
        "mtp_execution_mode": "external-overlay",
        "coordinator": {
            "image_digest": family["image_digest"],
            "sha256": family["preflight_sha256"],
            "gptqmodel": {
                "revision": GPTQMODEL_SOURCE_SERIALIZED_TRELLIS["revision"],
                "source_tree_sha256": GPTQMODEL_SOURCE_SERIALIZED_TRELLIS[
                    "source_tree_sha256"
                ],
            },
            "gpus": [
                {
                    "index": index,
                    "uuid": f"GPU-{index}",
                    "compute_capability": [12, 0],
                    "total_memory_bytes": 96 * 1024**3,
                }
                for index in (0, 1)
            ],
        },
    }

    assert validate_gptqmodel_native_exl3(quant) == family

    provenance["run"]["coordinator"]["gpus"][1]["uuid"] = "GPU-0"
    with pytest.raises(ValueError, match="qualified tier topology"):
        validate_gptqmodel_native_exl3(quant)


def test_v6_publication_geometry_includes_pro_router_and_mtp_fields() -> None:
    config = {
        "num_hidden_layers": 61,
        "n_routed_experts": 384,
        "hidden_size": 7168,
        "moe_intermediate_size": 3072,
        "dspark_target_layer_ids": [58, 59, 60],
        "num_hash_layers": 3,
        "num_experts_per_tok": 6,
        "hc_mult": 4,
        "num_nextn_predict_layers": 1,
    }

    assert _expected_source_geometry(config, PLAN_SCHEMA) == {
        **config,
        "mtp_block_count": 3,
    }
    assert "mtp_block_count" not in _expected_source_geometry(
        config,
        "ds4rt-deepseek-v4-gptqmodel-plan-v5",
    )


def test_gptqmodel_native_publication_stages_without_rewriting_metadata(
    tmp_path: Path,
) -> None:
    artifact = make_gptqmodel_artifact(tmp_path / "gptqmodel-artifact")
    retained, quality = make_gptqmodel_qualification(
        artifact,
        tmp_path / "qualification",
    )
    original_config = (artifact / "config.json").read_bytes()
    result = stage_snapshot(
        artifact,
        "tpurtell/flash-gptqmodel-exl3",
        tmp_path / "hf-gptqmodel",
        retained_native_report=retained,
        quality_report=quality,
    )
    snapshot = Path(result["snapshot"])
    assert (snapshot / "config.json").read_bytes() == original_config
    assert json.loads((snapshot / "config.json").read_text())["quantization_config"][
        "method"
    ] == "exl3"
    runtime = read_exl3_expert_config(snapshot)
    assert runtime.hidden_size == 4096
    assert runtime.intermediate_size == 2048
    assert runtime.expert_tensor_layout == GPTQMODEL_EXPERT_LAYOUT
    staged_qualification = (
        tmp_path
        / "hf-gptqmodel"
        / "hub"
        / "models--tpurtell--flash-gptqmodel-exl3"
        / "ds4rt-qualifications"
        / result["revision"]
    )
    assert (staged_qualification / "retained-native.json").read_bytes() == retained.read_bytes()
    assert (staged_qualification / "expert-quality.json").read_bytes() == quality.read_bytes()
    assert len(result["qualification"]) == 2
    sync_contract = load_staged_cache(
        tmp_path / "hf-gptqmodel",
        "tpurtell/flash-gptqmodel-exl3",
        verify_hashes=True,
    )
    assert sync_contract.revision == result["revision"]
    assert sync_contract.qualification_reports == 2
    generation_identity = validate_staged_exl3_checkpoint(snapshot)
    assert checkpoint_quantization_recipe(snapshot) == V4_RECIPE
    assert generation_identity["revision"] == result["revision"]
    assert len(generation_identity["qualification"]) == 2


def test_gptqmodel_publication_rejects_unbound_layer_sampling_mode(
    tmp_path: Path,
) -> None:
    artifact = make_gptqmodel_artifact(tmp_path / "gptqmodel-artifact")
    retained, quality = make_gptqmodel_qualification(
        artifact,
        tmp_path / "qualification",
    )
    report = json.loads(quality.read_text(encoding="utf-8"))
    report["layers"][0].pop("sampling_policy")
    write_json(quality, report)

    with pytest.raises(ValueError, match="target/dSpark sampling mode"):
        stage_snapshot(
            artifact,
            "tpurtell/flash-gptqmodel-exl3",
            tmp_path / "hf-gptqmodel",
            retained_native_report=retained,
            quality_report=quality,
        )


def test_development_unqualified_stage_is_explicit_and_never_promotes_quality(
    tmp_path: Path,
) -> None:
    artifact = make_gptqmodel_artifact(tmp_path / "gptqmodel-artifact")
    retained, quality = make_gptqmodel_qualification(
        artifact,
        tmp_path / "qualification",
    )
    diagnostic = json.loads(quality.read_text())
    diagnostic["exl3_snapshot"] = str(artifact.resolve())
    diagnostic["layers"][0]["native_to_exl3"]["cosine"] = 0.10
    write_json(quality, diagnostic)

    with pytest.raises(ValueError, match="outside its thresholds"):
        stage_snapshot(
            artifact,
            "tpurtell/flash-gptqmodel-production",
            tmp_path / "hf-production",
            retained_native_report=retained,
            quality_report=quality,
        )

    result = stage_snapshot(
        artifact,
        "tpurtell/flash-gptqmodel-development",
        tmp_path / "hf-development",
        retained_native_report=retained,
        quality_report=quality,
        development_unqualified=True,
    )
    snapshot = Path(result["snapshot"])
    assert result["qualification"] == []
    assert result["qualification_status"] == "development-unqualified"
    assert [entry["path"] for entry in result["development_evidence"]] == [
        "retained-native.json",
        "diagnostic-quality.json",
    ]

    with pytest.raises(ValueError, match="development-unqualified"):
        load_staged_cache(
            tmp_path / "hf-development",
            "tpurtell/flash-gptqmodel-development",
        )
    sync_contract = load_staged_cache(
        tmp_path / "hf-development",
        "tpurtell/flash-gptqmodel-development",
        allow_development_unqualified=True,
    )
    assert sync_contract.qualification_reports == 0
    assert sync_contract.development_evidence_reports == 2
    assert sync_contract.qualification_status == "development-unqualified"

    with pytest.raises(ValueError, match="development-unqualified"):
        validate_staged_exl3_checkpoint(snapshot)
    runtime_identity = validate_staged_exl3_checkpoint(
        snapshot,
        allow_development_unqualified=True,
    )
    assert runtime_identity["qualification"] == []
    assert runtime_identity["qualification_status"] == "development-unqualified"
    assert len(runtime_identity["development_evidence"]) == 2


def test_staging_rejects_raw_gptqmodel_transformed_export(tmp_path: Path) -> None:
    artifact = make_gptqmodel_artifact(tmp_path / "gptqmodel-artifact")
    plan = json.loads((artifact / "ds4rt-gptqmodel-plan.json").read_text())
    plan_body = {
        key: value
        for key, value in plan.items()
        if key not in {"plan_sha256", "canonical_assembly"}
    }
    raw_plan = bind_record(plan_body, "plan_sha256")
    (artifact / "ds4rt-exl3-canonical-assembly.json").unlink()
    rebind_gptqmodel_publication(artifact, raw_plan)
    retained, quality = make_gptqmodel_qualification(
        artifact,
        tmp_path / "qualification",
    )

    with pytest.raises(ValueError, match="raw GPTQModel export"):
        stage_snapshot(
            artifact,
            "tpurtell/flash-gptqmodel-exl3",
            tmp_path / "hf-gptqmodel",
            retained_native_report=retained,
            quality_report=quality,
        )


def test_runtime_rejects_rebound_incomplete_block_audit_set(tmp_path: Path) -> None:
    artifact_root = make_gptqmodel_artifact(tmp_path / "gptqmodel-artifact")
    assembly_path = artifact_root / "ds4rt-exl3-canonical-assembly.json"
    assembly = json.loads(assembly_path.read_text())
    block_audits = assembly["block_audit_set"]
    block_audit_body = {
        key: value
        for key, value in block_audits.items()
        if key != "report_sha256"
    }
    block_audit_body["block_count"] = 2
    assembly_body = {
        key: value for key, value in assembly.items() if key != "report_sha256"
    }
    assembly_body["block_audit_set"] = bind_record(
        block_audit_body,
        "report_sha256",
    )
    rebound_assembly = bind_record(assembly_body, "report_sha256")
    write_json(assembly_path, rebound_assembly)

    plan = json.loads(
        (artifact_root / "ds4rt-gptqmodel-plan.json").read_text()
    )
    plan["canonical_assembly"]["report_sha256"] = rebound_assembly[
        "report_sha256"
    ]
    config = json.loads((artifact_root / "config.json").read_text())
    publication = json.loads(
        (artifact_root / "ds4rt-gptqmodel-artifact.json").read_text()
    )

    with pytest.raises(ValueError, match="block-audit set is invalid"):
        _validate_canonical_assembly(
            artifact_root,
            config,
            config["quantization_config"],
            plan,
            publication,
        )


def test_gptqmodel_native_publication_requires_external_qualification(
    tmp_path: Path,
) -> None:
    artifact = make_gptqmodel_artifact(tmp_path / "gptqmodel-artifact")
    with pytest.raises(ValueError, match="requires --retained-native-report"):
        stage_snapshot(
            artifact,
            "tpurtell/flash-gptqmodel-exl3",
            tmp_path / "hf-gptqmodel",
        )


def test_gptqmodel_native_publication_rejects_stale_retained_report(
    tmp_path: Path,
) -> None:
    artifact = make_gptqmodel_artifact(tmp_path / "gptqmodel-artifact")
    retained, quality = make_gptqmodel_qualification(
        artifact,
        tmp_path / "qualification",
    )
    report = json.loads(retained.read_text())
    report["gptqmodel_publication"]["metadata_sha256"]["config.json"] = "0" * 64
    write_json(retained, report)

    with pytest.raises(ValueError, match="stale or has the wrong contract"):
        stage_snapshot(
            artifact,
            "tpurtell/flash-gptqmodel-exl3",
            tmp_path / "hf-gptqmodel",
            retained_native_report=retained,
            quality_report=quality,
        )


def test_gptqmodel_native_publication_rejects_calibration_overlap(
    tmp_path: Path,
) -> None:
    artifact = make_gptqmodel_artifact(tmp_path / "gptqmodel-artifact")
    retained, quality = make_gptqmodel_qualification(
        artifact,
        tmp_path / "qualification",
    )
    report = json.loads(quality.read_text())
    contract = report["validation_contract"]
    contract.pop("sha256")
    contract["calibration_disjointness"]["prompt_sha256_overlap"] = 1
    report["validation_contract"] = bind_record(contract, "sha256")
    write_json(quality, report)

    with pytest.raises(ValueError, match="prompt disjointness"):
        stage_snapshot(
            artifact,
            "tpurtell/flash-gptqmodel-exl3",
            tmp_path / "hf-gptqmodel",
            retained_native_report=retained,
            quality_report=quality,
        )


def test_gptqmodel_native_publication_requires_physical_gpu0_quality(
    tmp_path: Path,
) -> None:
    artifact = make_gptqmodel_artifact(tmp_path / "gptqmodel-artifact")
    retained, quality = make_gptqmodel_qualification(
        artifact,
        tmp_path / "qualification",
    )
    report = json.loads(quality.read_text())
    contract = report["validation_contract"]
    contract.pop("sha256")
    contract.pop("execution_gpu")
    report["validation_contract"] = bind_record(contract, "sha256")
    write_json(quality, report)

    with pytest.raises(ValueError, match="physical coordinator GPU0"):
        stage_snapshot(
            artifact,
            "tpurtell/flash-gptqmodel-exl3",
            tmp_path / "hf-gptqmodel",
            retained_native_report=retained,
            quality_report=quality,
        )


def test_gptqmodel_native_publication_rejects_cross_gpu_quality_layer(
    tmp_path: Path,
) -> None:
    artifact = make_gptqmodel_artifact(tmp_path / "gptqmodel-artifact")
    retained, quality = make_gptqmodel_qualification(
        artifact,
        tmp_path / "qualification",
    )
    report = json.loads(quality.read_text())
    report["layers"][0]["execution_gpu_sha256"] = "0" * 64
    write_json(quality, report)

    with pytest.raises(ValueError, match="not validated on physical GPU0"):
        stage_snapshot(
            artifact,
            "tpurtell/flash-gptqmodel-exl3",
            tmp_path / "hf-gptqmodel",
            retained_native_report=retained,
            quality_report=quality,
        )


def test_gptqmodel_native_publication_rejects_quantizer_provenance_drift(
    tmp_path: Path,
) -> None:
    artifact = make_gptqmodel_artifact(tmp_path / "gptqmodel-artifact")
    config = json.loads((artifact / "config.json").read_text())
    config["quantization_config"]["meta"]["ds4rt_error_ledger"]["family_join"][
        "gptqmodel"
    ]["revision"] = "0" * 40
    write_json(artifact / "config.json", config)
    retained, quality = make_gptqmodel_qualification(
        artifact,
        tmp_path / "qualification",
    )
    with pytest.raises(ValueError, match="qualified natural-route recipe"):
        stage_snapshot(
            artifact,
            "tpurtell/flash-gptqmodel-exl3",
            tmp_path / "hf-gptqmodel",
            retained_native_report=retained,
            quality_report=quality,
        )


def test_gptqmodel_native_ledger_rejects_fp32_hessian_record(tmp_path: Path) -> None:
    artifact = make_gptqmodel_artifact(tmp_path / "gptqmodel-artifact")
    config = json.loads((artifact / "config.json").read_text())
    family_join = config["quantization_config"]["meta"]["ds4rt_error_ledger"][
        "family_join"
    ]
    ledger_path = artifact / "ds4rt-exl3-error-ledger.jsonl"
    records = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    projection = records[0]
    projection.pop("record_sha256")
    projection["quantizer_metrics"]["hessian_numerical_contract"] = "legacy-fp32"
    records[0] = bind_ledger_record(projection)
    payload = b"".join(canonical_json(record) + b"\n" for record in records)
    ledger_path.write_bytes(payload)
    manifest_path = artifact / "ds4rt-exl3-error-ledger.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["ledger_sha256"] = hashlib.sha256(payload).hexdigest()
    write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="unusable metrics"):
        validate_error_ledger(
            artifact,
            config,
            expected_family_join=family_join,
        )


def test_gptqmodel_native_ledger_accepts_uniform_k3_records(tmp_path: Path) -> None:
    artifact = make_gptqmodel_artifact(tmp_path / "gptqmodel-artifact")
    config = json.loads((artifact / "config.json").read_text())
    config["quantization_config"]["bits"] = 3.0
    family_join = config["quantization_config"]["meta"]["ds4rt_error_ledger"][
        "family_join"
    ]
    for value in family_join.values():
        if isinstance(value, dict) and "bits" in value:
            value["bits"] = 3

    ledger_path = artifact / "ds4rt-exl3-error-ledger.jsonl"
    records = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    rebound = []
    for record in records:
        record.pop("record_sha256")
        record["bits"] = 3
        provenance = record.get("provenance")
        if isinstance(provenance, dict):
            provenance["family_join"] = family_join
        rebound.append(bind_ledger_record(record))
    payload = b"".join(canonical_json(record) + b"\n" for record in rebound)
    ledger_path.write_bytes(payload)
    manifest_path = artifact / "ds4rt-exl3-error-ledger.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["ledger_sha256"] = hashlib.sha256(payload).hexdigest()
    write_json(manifest_path, manifest)

    validate_error_ledger(
        artifact,
        config,
        expected_family_join=family_join,
    )


@pytest.mark.parametrize("recipe", (V3_RECIPE, V4_RECIPE))
def test_stage_accepts_checkpoint_bound_activation_recipe(
    tmp_path: Path, recipe: str
) -> None:
    artifact = make_artifact(tmp_path / "artifact-activation", recipe=recipe)

    result = stage_snapshot(
        artifact,
        "tpurtell/flash-exl3-activation",
        tmp_path / "hf-activation",
    )

    assert result["revision"]


def test_v4_stage_requires_valid_complete_error_ledger(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path / "artifact-missing", recipe=V4_RECIPE)
    (artifact / "ds4rt-exl3-error-ledger.manifest.json").unlink()
    with pytest.raises(ValueError, match="missing error ledger"):
        stage_snapshot(artifact, "tpurtell/flash-exl3", tmp_path / "hf-missing")

    artifact = make_artifact(tmp_path / "artifact-corrupt", recipe=V4_RECIPE)
    ledger = artifact / "ds4rt-exl3-error-ledger.jsonl"
    ledger.write_bytes(ledger.read_bytes() + b"{}\n")
    with pytest.raises(ValueError, match="manifest or payload digest is invalid"):
        stage_snapshot(artifact, "tpurtell/flash-exl3", tmp_path / "hf-corrupt")

    artifact = make_artifact(tmp_path / "artifact-incomplete", recipe=V4_RECIPE)
    ledger = artifact / "ds4rt-exl3-error-ledger.jsonl"
    records = [json.loads(line) for line in ledger.read_text().splitlines()][1:]
    payload = b"".join(canonical_json(record) + b"\n" for record in records)
    ledger.write_bytes(payload)
    manifest_path = artifact / "ds4rt-exl3-error-ledger.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        ledger_sha256=hashlib.sha256(payload).hexdigest(),
        projection_records=17,
        total_records=23,
    )
    write_json(manifest_path, manifest)
    with pytest.raises(
        ValueError,
        match="incomplete expert family|inconsistent expert family|does not exactly cover",
    ):
        stage_snapshot(artifact, "tpurtell/flash-exl3", tmp_path / "hf-incomplete")

    artifact = make_artifact(tmp_path / "artifact-no-route", recipe=V4_RECIPE)
    ledger = artifact / "ds4rt-exl3-error-ledger.jsonl"
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    projection = records[0]
    projection.pop("record_sha256")
    projection.pop("route_evidence")
    records[0] = bind_ledger_record(projection)
    payload = b"".join(canonical_json(record) + b"\n" for record in records)
    ledger.write_bytes(payload)
    manifest_path = artifact / "ds4rt-exl3-error-ledger.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["ledger_sha256"] = hashlib.sha256(payload).hexdigest()
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="invalid natural-route evidence"):
        stage_snapshot(artifact, "tpurtell/flash-exl3", tmp_path / "hf-no-route")


@pytest.mark.parametrize(
    "unfinished_name",
    [".ds4rt-exl3-state.json", "config.json.incomplete", "shard.tmp"],
)
def test_stage_rejects_incomplete_or_temporary_artifact(
    tmp_path: Path, unfinished_name: str
) -> None:
    artifact = make_artifact(tmp_path / "artifact")
    (artifact / unfinished_name).write_text("unfinished", encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete or temporary"):
        stage_snapshot(artifact, "tpurtell/flash-exl3", tmp_path / "hf")


def test_stage_requires_all_layer_quality_evidence(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path / "artifact")
    (artifact / "ds4rt-exl3-quality.json").unlink()
    with pytest.raises(ValueError, match="missing regular file"):
        stage_snapshot(artifact, "tpurtell/flash-exl3", tmp_path / "hf-missing")

    artifact = make_artifact(tmp_path / "artifact-partial")
    write_json(
        artifact / "ds4rt-exl3-calibration.json",
        {"recipe": RECIPE, "layers": [{"layer_id": 0}, {"layer_id": 1}]},
    )
    with pytest.raises(ValueError, match="every target and dSpark block"):
        stage_snapshot(artifact, "tpurtell/flash-exl3", tmp_path / "hf-partial")


def test_stage_requires_model_bound_multigpu_projection_qualification(
    tmp_path: Path,
) -> None:
    artifact = make_artifact(tmp_path / "artifact-multigpu", multigpu=True)
    stage_snapshot(artifact, "tpurtell/flash-exl3", tmp_path / "hf-valid")

    artifact = make_artifact(tmp_path / "artifact-missing-multigpu", multigpu=True)
    (artifact / "ds4rt-exl3-multigpu-production.json").unlink()
    with pytest.raises(ValueError, match="missing full-shape qualification"):
        stage_snapshot(artifact, "tpurtell/flash-exl3", tmp_path / "hf-missing")

    artifact = make_artifact(tmp_path / "artifact-wrong-shape", multigpu=True)
    qualification_path = artifact / "ds4rt-exl3-multigpu-production.json"
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification["production_projection_shapes"][-1] = [128, 256]
    write_json(qualification_path, qualification)
    with pytest.raises(ValueError, match="does not match the artifact contract"):
        stage_snapshot(artifact, "tpurtell/flash-exl3", tmp_path / "hf-wrong")


def test_stage_rejects_weak_quality_or_non_tp4_evidence(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path / "artifact")
    quality_path = artifact / "ds4rt-exl3-quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["thresholds"]["min_quant_cosine"] = 0.5
    write_json(quality_path, quality)
    with pytest.raises(ValueError, match="weaker than production"):
        stage_snapshot(artifact, "tpurtell/flash-exl3", tmp_path / "hf-weak")

    artifact = make_artifact(tmp_path / "artifact-nontp4")
    quality_path = artifact / "ds4rt-exl3-quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["layers"][0]["tp4_rank_source_bytes"][-1] = 99
    write_json(quality_path, quality)
    with pytest.raises(ValueError, match="strict equal-residency TP4"):
        stage_snapshot(artifact, "tpurtell/flash-exl3", tmp_path / "hf-nontp4")

    artifact = make_artifact(tmp_path / "artifact-unstratified")
    config_path = artifact / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["n_routed_experts"] = 7
    write_json(config_path, config)
    with pytest.raises(ValueError, match="not stratified"):
        stage_snapshot(artifact, "tpurtell/flash-exl3", tmp_path / "hf-unstratified")


def test_stage_requires_full_generated_tensor_and_tp4_layout_proof(
    tmp_path: Path,
) -> None:
    artifact = make_artifact(tmp_path / "artifact-metadata")
    integrity_path = artifact / "ds4rt-exl3-retained-native.json"
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    integrity["generated_exl3_metadata_verified"] = False
    write_json(integrity_path, integrity)
    with pytest.raises(ValueError, match="generated EXL3 tensor metadata"):
        stage_snapshot(artifact, "tpurtell/flash-exl3", tmp_path / "hf-metadata")

    artifact = make_artifact(tmp_path / "artifact-layout")
    integrity_path = artifact / "ds4rt-exl3-retained-native.json"
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    integrity["strict_tp4_source_layout"]["rank_source_bytes_per_block"][-1] = 99
    write_json(integrity_path, integrity)
    with pytest.raises(ValueError, match="full equal-residency TP4"):
        stage_snapshot(artifact, "tpurtell/flash-exl3", tmp_path / "hf-layout")


def test_stage_rejects_stale_or_rewritten_quality_binding(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path / "artifact-stale")
    (artifact / "model-00001-of-00001.safetensors").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="stale or belongs to another"):
        stage_snapshot(artifact, "tpurtell/flash-exl3", tmp_path / "hf-stale")

    artifact = make_artifact(tmp_path / "artifact-rewritten")
    quality_path = artifact / "ds4rt-exl3-quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["validation_contract"]["rows"] = 32
    write_json(quality_path, quality)
    with pytest.raises(ValueError, match="contract digest is invalid"):
        stage_snapshot(artifact, "tpurtell/flash-exl3", tmp_path / "hf-rewritten")


def test_stage_refuses_ref_movement_without_explicit_update(tmp_path: Path) -> None:
    first = make_artifact(tmp_path / "first", shard_payload=b"first")
    second = make_artifact(tmp_path / "second", shard_payload=b"second")
    hf_home = tmp_path / "hf"
    first_result = stage_snapshot(first, "tpurtell/flash-exl3", hf_home)

    with pytest.raises(ValueError, match="--update-ref"):
        stage_snapshot(second, "tpurtell/flash-exl3", hf_home)

    second_result = stage_snapshot(
        second,
        "tpurtell/flash-exl3",
        hf_home,
        update_ref=True,
    )
    assert second_result["revision"] != first_result["revision"]
    model_root = model_cache_dir(hf_home.resolve(), "tpurtell/flash-exl3")
    assert (model_root / "refs/main").read_text(encoding="utf-8").strip() == str(
        second_result["revision"]
    )
    assert (model_root / "snapshots" / str(first_result["revision"])).is_dir()


def test_staged_revision_is_deterministic_across_cache_roots(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path / "artifact")

    first = stage_snapshot(artifact, "tpurtell/flash-exl3", tmp_path / "hf-a")
    second = stage_snapshot(artifact, "tpurtell/flash-exl3", tmp_path / "hf-b")

    assert first["revision"] == second["revision"]


def test_existing_revision_is_validated_and_copy_mode_isolated(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path / "artifact")
    hf_home = tmp_path / "hf"
    result = stage_snapshot(
        artifact,
        "tpurtell/flash-exl3",
        hf_home,
        link_mode="copy",
    )
    model_root = model_cache_dir(hf_home.resolve(), "tpurtell/flash-exl3")
    staged = model_root / "snapshots" / str(result["revision"])
    shard = staged / "model-00001-of-00001.safetensors"
    assert shard.resolve().stat().st_ino != (artifact / shard.name).stat().st_ino

    tokenizer = staged / "tokenizer.json"
    tokenizer.unlink()
    tokenizer.symlink_to("../../blobs/not-the-content-digest")
    with pytest.raises(ValueError, match="incorrect link"):
        stage_snapshot(
            artifact,
            "tpurtell/flash-exl3",
            hf_home,
            link_mode="copy",
        )


def test_stage_rejects_artifact_symlinks_and_missing_index_shards(
    tmp_path: Path,
) -> None:
    artifact = make_artifact(tmp_path / "artifact")
    (artifact / "tokenizer.json").unlink()
    (artifact / "tokenizer.json").symlink_to("config.json")
    with pytest.raises(ValueError, match="must not be a symlink"):
        stage_snapshot(artifact, "tpurtell/flash-exl3", tmp_path / "hf-a")

    artifact = make_artifact(tmp_path / "artifact-missing")
    (artifact / "model-00001-of-00001.safetensors").unlink()
    with pytest.raises(ValueError, match="no safetensor shards|missing shards"):
        stage_snapshot(artifact, "tpurtell/flash-exl3", tmp_path / "hf-b")
