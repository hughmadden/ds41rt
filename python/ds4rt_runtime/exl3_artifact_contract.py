"""Fail-closed contract for DS4RT's GPTQModel-native EXL3 publication."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any


RECIPE = "deepseek_v4_exl3_trellis_2bpw_v4_flash_natural_route"
RECIPE_K3 = "deepseek_v4_exl3_trellis_3bpw_v4_flash_natural_route"
PLAN_SCHEMA = "ds4rt-deepseek-v4-gptqmodel-plan-v6"
LEGACY_PLAN_SCHEMA = "ds4rt-deepseek-v4-gptqmodel-plan-v5"
SUPPORTED_PLAN_SCHEMAS = frozenset((LEGACY_PLAN_SCHEMA, PLAN_SCHEMA))
RUN_SCHEMA = "ds4rt-deepseek-v4-gptqmodel-run-v5"
ARTIFACT_SCHEMA = "ds4rt-deepseek-v4-gptqmodel-artifact-v1"
PLAN_FILE = "ds4rt-gptqmodel-plan.json"
RUN_FILE = "ds4rt-gptqmodel-run.json"
ARTIFACT_FILE = "ds4rt-gptqmodel-artifact.json"
LEDGER_FILE = "ds4rt-exl3-error-ledger.jsonl"
LEDGER_MANIFEST_FILE = "ds4rt-exl3-error-ledger.manifest.json"
CANONICAL_ASSEMBLY_FILE = "ds4rt-exl3-canonical-assembly.json"
CANONICAL_ASSEMBLY_SCHEMA = "ds4rt-exl3-canonical-hybrid-assembly-v1"
COMPOSITE_ASSEMBLY_SCHEMA = "ds4rt-exl3-canonical-hybrid-assembly-v2"
PROJECTION_ASSEMBLY_SCHEMA = (
    "ds4rt-exl3-canonical-hybrid-projection-assembly-v1"
)
COMPOSITE_PROJECTION_ASSEMBLY_SCHEMA = (
    "ds4rt-exl3-canonical-hybrid-composite-projection-assembly-v1"
)
QUANT_CONFIG_ASSEMBLY_SCHEMA = (
    "ds4rt-exl3-canonical-quant-config-assembly-v1"
)
BLOCK_AUDIT_SET_SCHEMA = "ds4rt-exl3-independent-block-audit-set-v1"
COMPOSITE_BLOCK_AUDIT_SET_SCHEMA = (
    "ds4rt-exl3-independent-composite-block-audit-set-v1"
)
COMPOSITE_QUANT_CONFIG_ASSEMBLY_SCHEMA = (
    "ds4rt-exl3-canonical-composite-quant-config-assembly-v1"
)
MCG_MULTIPLIER = 0xCBAC1FED
BASE_EXPERT_PATTERN = (
    r"^model\.layers\.\d+\.mlp\.experts\.\d+\."
    r"(?:gate_proj|up_proj|down_proj)$"
)
GPTQMODEL_SOURCE = {
    "schema": 1,
    "repository": "https://github.com/tpurtell/GPTQModel.git",
    "revision": "6f853e66f692cf623a2781f4e4fe05078f57f999",
    "source_tree_sha256": (
        "bc862c87a9b28ee770267e69c663f0b9c0829f7cd1ee57e889ef5bffde7b7e20"
    ),
}
GPTQMODEL_SOURCE_WITH_ROUTE_RECOVERY = {
    "schema": 1,
    "repository": "https://github.com/tpurtell/GPTQModel.git",
    "revision": "4bbc617f957461ce8de75ef52cd9b27d11fb50f8",
    "source_tree_sha256": (
        "f9e268e06a87e2f3fb572a12903829d320a669403d415bfef86a56402ca92cb4"
    ),
}
GPTQMODEL_SOURCE_SAMPLED_PIPELINE = {
    "schema": 1,
    "repository": "https://github.com/tpurtell/GPTQModel.git",
    "revision": "266fc364f4a4a4f138639f4fad6142e8b8b740c4",
    "source_tree_sha256": (
        "9f0c04ef95f8535ad970f2c0fe7f457de7396f1552e8b9d692b3abc34232d9f1"
    ),
}
GPTQMODEL_SOURCE_SERIALIZED_TRELLIS = {
    "schema": 1,
    "repository": "https://github.com/tpurtell/GPTQModel.git",
    "revision": "fc7b873df814cc54b02facdc999d051deea2c5ca",
    "source_tree_sha256": (
        "05a8d9fda891206fc5b2fd83c68d14101c26101d0d94848fdcaf038e6a8888c0"
    ),
}
OPERATOR_CONTRACT = "ds4rt-deepseek-v4-target-plus-joint-mtp-v1"
REMOTE_CONTRACT = "ds4rt.exl3-remote-worker-v1"
REMOTE_SCHEDULER = "dynamic-pipelined-slot-projection-v2"
ROUTE_EVIDENCE_CONTRACT = "ds4rt.exl3-natural-route"
HESSIAN_NUMERICS = {
    "sigma_reg": 0.025,
    "hessian_capture": "raw-xtx-sum-fp32-v1",
    "hessian_numerical": "signed-block-hadamard-congruence-fp64-v1",
    "hessian_symmetry": "mean-with-transpose-fp64",
}
ZERO_ROUTE_RECOVERY_CONFIG = {
    "contract": "ds4rt.exl3-zero-route-recovery",
    "trigger": "natural-route-count-below-1024",
    "sample_source": "same-fixed-calibration-selection",
    "capture_method": "direct-expert-router-ranks-7-12-then-identity-residual",
    "selection_policy": "rank-ascending-then-fixed-replay-order-v1",
    "candidate_rank_min": 7,
    "candidate_rank_max": 12,
    "target_sample_count": 1024,
    "identity_calibration_policy": (
        "normalized-2i-residual-to-effective-count-1024-v2"
    ),
    "scope": "all-learned-top-k-routers",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
REVISION_RE = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
IMAGE_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _expected_plan_exl3(family: dict[str, Any]) -> dict[str, Any]:
    bits = family.get("bits")
    if isinstance(bits, bool) or bits not in {2, 3}:
        raise ValueError("GPTQModel EXL3 family has an unsupported integer tier")
    expected: dict[str, Any] = {
        "bits": bits,
        "codebook": "mcg",
        "seed": 787,
        "module_include": [BASE_EXPERT_PATTERN],
        "fallback": None,
        "out_scales": "auto",
        "sigma_reg": HESSIAN_NUMERICS["sigma_reg"],
        "hessian_capture": HESSIAN_NUMERICS["hessian_capture"],
        "hessian_numerical": HESSIAN_NUMERICS["hessian_numerical"],
        "hessian_symmetry": HESSIAN_NUMERICS["hessian_symmetry"],
    }
    if family.get("gptqmodel") in (
        GPTQMODEL_SOURCE_WITH_ROUTE_RECOVERY,
        GPTQMODEL_SOURCE_SAMPLED_PIPELINE,
        GPTQMODEL_SOURCE_SERIALIZED_TRELLIS,
    ):
        expected["zero_route_recovery"] = ZERO_ROUTE_RECOVERY_CONFIG
    return expected


def _recipe_for_bits(bits: int) -> str:
    if bits == 2:
        return RECIPE
    if bits == 3:
        return RECIPE_K3
    raise ValueError(f"unsupported EXL3 integer tier K{bits}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read GPTQModel artifact JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"GPTQModel artifact JSON is not an object: {path}")
    return value


def is_gptqmodel_native_exl3(quant: Any) -> bool:
    return (
        isinstance(quant, dict)
        and isinstance(quant.get("meta"), dict)
        and "ds4rt_error_ledger" in quant["meta"]
    )


def _digest(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _validate_bound_record(record: dict[str, Any], field: str, label: str) -> None:
    claimed = record.get(field)
    body = {key: value for key, value in record.items() if key != field}
    if not _digest(claimed) or sha256_bytes(canonical_json_bytes(body)) != claimed:
        raise ValueError(f"{label} digest is invalid")


def _valid_coordinator_only_topology(
    provenance: dict[str, Any],
    family: dict[str, Any],
) -> bool:
    """Validate the two-RTX v6 execution envelope used for Pro quantization."""

    if family.get("gptqmodel") != GPTQMODEL_SOURCE_SERIALIZED_TRELLIS:
        return False
    run = provenance.get("run")
    coordinator = run.get("coordinator") if isinstance(run, dict) else None
    gpus = coordinator.get("gpus") if isinstance(coordinator, dict) else None
    coordinator_source = (
        coordinator.get("gptqmodel") if isinstance(coordinator, dict) else None
    )
    family_source = family["gptqmodel"]
    return (
        isinstance(run, dict)
        and run.get("mtp_execution_mode") == "external-overlay"
        and "remote_workers" not in run
        and isinstance(coordinator, dict)
        and coordinator.get("image_digest") == family.get("image_digest")
        and IMAGE_RE.fullmatch(str(coordinator.get("image_digest", ""))) is not None
        and coordinator.get("sha256") == family.get("preflight_sha256")
        and _digest(coordinator.get("sha256"))
        and isinstance(coordinator_source, dict)
        and coordinator_source.get("revision") == family_source["revision"]
        and coordinator_source.get("source_tree_sha256")
        == family_source["source_tree_sha256"]
        and isinstance(gpus, list)
        and len(gpus) == 2
        and {gpu.get("index") for gpu in gpus if isinstance(gpu, dict)} == {0, 1}
        and len(
            {
                gpu.get("uuid")
                for gpu in gpus
                if isinstance(gpu, dict)
                and str(gpu.get("uuid", "")).startswith("GPU-")
            }
        )
        == 2
        and all(
            isinstance(gpu, dict)
            and gpu.get("compute_capability") == [12, 0]
            and isinstance(gpu.get("total_memory_bytes"), int)
            and not isinstance(gpu.get("total_memory_bytes"), bool)
            and gpu["total_memory_bytes"] > 0
            for gpu in gpus
        )
    )


def validate_gptqmodel_native_exl3(
    quant: Any,
    *,
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and return the family-join provenance for one uniform K2/K3 run."""

    if not isinstance(quant, dict):
        raise ValueError("GPTQModel EXL3 quantization configuration is not an object")
    for field in ("quant_method", "method", "format", "checkpoint_format"):
        if str(quant.get(field, "")).lower() != "exl3":
            raise ValueError(f"GPTQModel EXL3 requires {field}=exl3")
    bits_value = quant.get("bits")
    if (
        isinstance(bits_value, bool)
        or not isinstance(bits_value, (int, float))
        or float(bits_value) not in {2.0, 3.0}
        or quant.get("codebook") != "mcg"
        or quant.get("out_scales") != "auto"
        or quant.get("group_size") != -1
        or quant.get("desc_act") is not False
        or quant.get("module_include") != [BASE_EXPERT_PATTERN]
        or not isinstance(quant.get("tensor_storage"), dict)
        or not quant["tensor_storage"]
    ):
        raise ValueError(
            "GPTQModel EXL3 is not the routed-expert-only integer-K/MCG contract"
        )
    bits = int(bits_value)
    meta = quant.get("meta")
    if not isinstance(meta, dict) or "fallback" not in meta or meta["fallback"] is not None:
        raise ValueError("GPTQModel EXL3 production artifacts forbid fallback quantization")
    provenance = meta.get("ds4rt_error_ledger")
    if (
        not isinstance(provenance, dict)
        or not isinstance(provenance.get("family_join"), dict)
        or not isinstance(provenance.get("run"), dict)
    ):
        raise ValueError("GPTQModel EXL3 has no complete error-ledger provenance")
    family = provenance["family_join"]
    family_source = family.get("gptqmodel")
    recovery_contract = family.get("zero_route_recovery_contract")
    qualified_source = (
        family_source == GPTQMODEL_SOURCE and recovery_contract is None
    ) or (
        family_source
        in (
            GPTQMODEL_SOURCE_WITH_ROUTE_RECOVERY,
            GPTQMODEL_SOURCE_SAMPLED_PIPELINE,
            GPTQMODEL_SOURCE_SERIALIZED_TRELLIS,
        )
        and recovery_contract == ZERO_ROUTE_RECOVERY_CONFIG["contract"]
    )
    if (
        family.get("recipe") != _recipe_for_bits(bits)
        or family.get("bits") != bits
        or family.get("codebook") != "mcg"
        or family.get("module_include") != BASE_EXPERT_PATTERN
        or family.get("quantizer_seed") != 787
        or family.get("quantizer_numerics") != HESSIAN_NUMERICS
        or family.get("operator_contract") != OPERATOR_CONTRACT
        or family.get("route_evidence_contract") != ROUTE_EVIDENCE_CONTRACT
        or not qualified_source
    ):
        raise ValueError("GPTQModel EXL3 provenance is not the qualified natural-route recipe")
    corpus = family.get("corpus")
    if (
        not isinstance(corpus, dict)
        or isinstance(corpus.get("examples"), bool)
        or not isinstance(corpus.get("examples"), int)
        or corpus["examples"] <= 0
        or isinstance(corpus.get("utf8_bytes"), bool)
        or not isinstance(corpus.get("utf8_bytes"), int)
        or corpus["utf8_bytes"] <= 0
        or not _digest(corpus.get("file_sha256"))
        or not _digest(corpus.get("normalized_stream_sha256"))
    ):
        raise ValueError("GPTQModel EXL3 calibration corpus provenance is invalid")
    source = family.get("source")
    geometry = source.get("geometry") if isinstance(source, dict) else None
    if (
        not isinstance(source, dict)
        or REVISION_RE.fullmatch(str(source.get("revision", ""))) is None
        or not _digest(source.get("config_sha256"))
        or not _digest(source.get("index_sha256"))
        or not isinstance(geometry, dict)
    ):
        raise ValueError("GPTQModel EXL3 source-checkpoint provenance is invalid")
    if model_config is not None:
        supported_geometries = (
            _expected_source_geometry(model_config, LEGACY_PLAN_SCHEMA),
            _expected_source_geometry(model_config, PLAN_SCHEMA),
        )
        if not any(geometry == expected for expected in supported_geometries):
            raise ValueError("GPTQModel EXL3 source geometry differs from config.json")
    topology = family.get("execution_topology")
    slots = topology.get("coordinator_slots") if isinstance(topology, dict) else None
    workers = topology.get("workers") if isinstance(topology, dict) else None
    coordinator = topology.get("coordinator") if isinstance(topology, dict) else None
    topology_valid = _valid_coordinator_only_topology(provenance, family) or (
        bits == 3 and topology is None
    ) or (
        bits == 2
        and isinstance(topology, dict)
        and topology.get("contract") == REMOTE_CONTRACT
        and topology.get("scheduler") == REMOTE_SCHEDULER
        and isinstance(coordinator, dict)
        and IMAGE_RE.fullmatch(str(coordinator.get("image_digest", ""))) is not None
        and _digest(coordinator.get("preflight_sha256"))
        and isinstance(slots, list)
        and len(slots) == 2
        and {slot.get("device") for slot in slots if isinstance(slot, dict)}
        == {"cuda:0", "cuda:1"}
        and all(
            isinstance(slot, dict)
            and slot.get("image_digest") == coordinator["image_digest"]
            and str(slot.get("gpu_uuid", "")).startswith("GPU-")
            and _digest(slot.get("preflight_sha256"))
            for slot in slots
        )
        and isinstance(workers, list)
        and len(workers) == 4
        and {worker.get("name") for worker in workers if isinstance(worker, dict)}
        == {"dodo", "emu", "kiwi", "ostrich"}
        and len(
            {
                worker.get("image_digest")
                for worker in workers
                if isinstance(worker, dict)
            }
        )
        == 1
        and all(
            isinstance(worker, dict)
            and IMAGE_RE.fullmatch(str(worker.get("image_digest", ""))) is not None
            and _digest(worker.get("preflight_sha256"))
            for worker in workers
        )
    )
    if not topology_valid:
        raise ValueError("GPTQModel EXL3 does not have its qualified tier topology")
    return family


def _safe_path(raw: Any) -> PurePosixPath:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise ValueError(f"GPTQModel artifact has unsafe path {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"GPTQModel artifact has unsafe path {raw!r}")
    return path


def _canonical_staged_blob(snapshot: Path, path: Path, digest: str | None) -> bool:
    if not path.is_symlink():
        return False
    if snapshot.parent.name != "snapshots":
        raise ValueError(f"artifact symlink is not in an HF snapshot: {path}")
    target = path.resolve(strict=True)
    blob_root = snapshot.parent.parent / "blobs"
    if (
        target.parent != blob_root
        or SHA256_RE.fullmatch(target.name) is None
        or (digest is not None and target.name != digest)
        or target.is_symlink()
        or not target.is_file()
    ):
        raise ValueError(f"artifact symlink is not a canonical content blob: {path}")
    return True


def _finite_number(value: Any, *, positive: bool = False) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and (not positive or float(value) > 0.0)
    )


def _validate_canonical_tensor_storage(
    quant: dict[str, Any],
    model_config: dict[str, Any],
) -> None:
    hidden_size = model_config["hidden_size"]
    intermediate_size = model_config["moe_intermediate_size"]
    layer_count = model_config["num_hidden_layers"]
    expert_count = model_config["n_routed_experts"]
    dspark_targets = model_config["dspark_target_layer_ids"]
    bits = int(quant["bits"])
    storage = quant["tensor_storage"]
    expected: dict[str, dict[str, Any]] = {}
    for namespace, layers in (
        ("base", range(layer_count)),
        ("mtp", range(len(dspark_targets))),
    ):
        for layer in layers:
            block = f"model.layers.{layer}" if namespace == "base" else f"mtp.{layer}"
            for expert in range(expert_count):
                for projection, (input_size, output_size) in {
                    "gate_proj": (hidden_size, intermediate_size),
                    "up_proj": (hidden_size, intermediate_size),
                    "down_proj": (intermediate_size, hidden_size),
                }.items():
                    module = f"{block}.mlp.experts.{expert}.{projection}"
                    expected[module] = {
                        "stored_tensors": {
                            f"{module}.trellis": {
                                "shape": [
                                    input_size // 16,
                                    output_size // 16,
                                    16 * bits,
                                ],
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
                        "bits_per_weight": bits,
                        "mcg_multiplier": MCG_MULTIPLIER,
                    }
    if storage != expected:
        raise ValueError(
            "canonical GPTQModel EXL3 tensor_storage differs from its uniform tier"
        )


def _validate_canonical_assembly(
    snapshot: Path,
    model_config: dict[str, Any],
    quant: dict[str, Any],
    plan: dict[str, Any],
    artifact: dict[str, Any],
) -> dict[str, dict[str, Any]] | None:
    bits = int(quant["bits"])
    recipe = _recipe_for_bits(bits)
    declaration = plan.get("canonical_assembly")
    if not isinstance(declaration, dict):
        raise ValueError(
            "raw GPTQModel export is not a canonical source-native hybrid artifact"
        )
    if declaration.get("schema") == COMPOSITE_ASSEMBLY_SCHEMA:
        return _validate_composite_canonical_assembly(
            snapshot,
            model_config,
            quant,
            plan,
            artifact,
            declaration,
        )
    assembly = read_json_object(snapshot / CANONICAL_ASSEMBLY_FILE)
    _validate_bound_record(assembly, "report_sha256", "canonical assembly report")
    projection = assembly.get("projection_assembly")
    if not isinstance(projection, dict):
        raise ValueError("canonical assembly has no projection-checkpoint report")
    _validate_bound_record(
        projection,
        "report_sha256",
        "canonical projection assembly report",
    )
    quant_config_assembly = assembly.get("quant_config_assembly")
    if not isinstance(quant_config_assembly, dict):
        raise ValueError("canonical assembly has no quantization-config report")
    _validate_bound_record(
        quant_config_assembly,
        "report_sha256",
        "canonical quantization-config assembly report",
    )
    block_audits = assembly.get("block_audit_set")
    if not isinstance(block_audits, dict):
        raise ValueError("canonical assembly has no independent block-audit set")
    _validate_bound_record(
        block_audits,
        "report_sha256",
        "canonical independent block-audit set",
    )

    source_plan_sha = declaration.get("source_plan_sha256")
    source_artifact_sha = declaration.get("source_artifact_manifest_sha256")
    source_snapshot = assembly.get("source_snapshot")
    planned_source = plan.get("source")
    portable_source = (
        {key: value for key, value in planned_source.items() if key != "path"}
        if isinstance(planned_source, dict)
        else None
    )
    if (
        declaration
        != {
            "schema": CANONICAL_ASSEMBLY_SCHEMA,
            "filename": CANONICAL_ASSEMBLY_FILE,
            "report_sha256": assembly["report_sha256"],
            "source_plan_sha256": source_plan_sha,
            "source_artifact_manifest_sha256": source_artifact_sha,
        }
        or not _digest(source_plan_sha)
        or not _digest(source_artifact_sha)
        or assembly.get("schema") != CANONICAL_ASSEMBLY_SCHEMA
        or assembly.get("recipe") != recipe
        or assembly.get("source_plan_sha256") != source_plan_sha
        or assembly.get("source_artifact_manifest_sha256") != source_artifact_sha
        or not _digest(assembly.get("source_run_sha256"))
        or source_snapshot != portable_source
        or projection.get("schema") != PROJECTION_ASSEMBLY_SCHEMA
        or projection.get("plan_sha256") != source_plan_sha
        or projection.get("expert_tensor_layout") != "gptqmodel"
        or projection.get("retained_tensor_layout")
        != "source_checkpoint_native_byte_exact"
    ):
        raise ValueError("canonical GPTQModel assembly provenance is inconsistent")

    hidden_size = model_config["hidden_size"]
    intermediate_size = model_config["moe_intermediate_size"]
    blocks = model_config["num_hidden_layers"] + len(
        model_config["dspark_target_layer_ids"]
    )
    expert_count = model_config["n_routed_experts"]
    family_count = blocks * expert_count
    projection_count = family_count * 3
    base_blocks = model_config["num_hidden_layers"]
    mtp_blocks = len(model_config["dspark_target_layer_ids"])
    projections_per_block = expert_count * 3
    expected_reports = [
        (ordinal, namespace, logical_layer)
        for ordinal, (namespace, logical_layer) in enumerate(
            [
                (namespace, logical_layer)
                for namespace, count in (("base", base_blocks), ("mtp", mtp_blocks))
                for logical_layer in range(count)
            ],
            1,
        )
    ]
    reports = block_audits.get("reports")
    report_records_valid = (
        isinstance(reports, list) and len(reports) == blocks
    )
    if report_records_valid:
        for report, (ordinal, namespace, logical_layer) in zip(
            reports, expected_reports, strict=True
        ):
            filename = f"{namespace}-layer-{logical_layer}-projection-audit.json"
            if (
                not isinstance(report, dict)
                or set(report)
                != {
                    "ordinal",
                    "block_namespace",
                    "logical_layer",
                    "filename",
                    "file_sha256",
                    "report_sha256",
                }
                or type(report.get("ordinal")) is not int
                or report.get("ordinal") != ordinal
                or report.get("block_namespace") != namespace
                or type(report.get("logical_layer")) is not int
                or report.get("logical_layer") != logical_layer
                or report.get("filename") != filename
                or not _digest(report.get("file_sha256"))
                or not _digest(report.get("report_sha256"))
            ):
                report_records_valid = False
                break
    if (
        set(block_audits)
        != {
            "schema",
            "plan_sha256",
            "base_block_count",
            "mtp_block_count",
            "block_count",
            "projections_per_block",
            "projection_count",
            "expert_family_count",
            "reports",
            "reports_sha256",
            "report_sha256",
        }
        or block_audits.get("schema") != BLOCK_AUDIT_SET_SCHEMA
        or block_audits.get("plan_sha256") != source_plan_sha
        or block_audits.get("base_block_count") != base_blocks
        or block_audits.get("mtp_block_count") != mtp_blocks
        or block_audits.get("block_count") != blocks
        or block_audits.get("projections_per_block") != projections_per_block
        or block_audits.get("projection_count") != projection_count
        or block_audits.get("expert_family_count") != family_count
        or not report_records_valid
        or block_audits.get("reports_sha256")
        != sha256_bytes(canonical_json_bytes(reports))
    ):
        raise ValueError("canonical GPTQModel independent block-audit set is invalid")
    base_projection_count = model_config["num_hidden_layers"] * expert_count * 3
    added_mtp_names = sorted(
        f"mtp.{layer}.mlp.experts.{expert}.{projection_name}"
        for layer in range(len(model_config["dspark_target_layer_ids"]))
        for expert in range(expert_count)
        for projection_name in ("gate_proj", "up_proj", "down_proj")
    )
    raw_module_count = quant_config_assembly.get("raw_module_count")
    raw_module_count_valid = (
        isinstance(raw_module_count, int)
        and not isinstance(raw_module_count, bool)
        and raw_module_count in {base_projection_count, projection_count}
    )
    expected_added_names = (
        added_mtp_names if raw_module_count == base_projection_count else []
    )
    expected_added_count = (
        projection_count - raw_module_count if raw_module_count_valid else None
    )
    if (
        quant_config_assembly.get("schema") != QUANT_CONFIG_ASSEMBLY_SCHEMA
        or not raw_module_count_valid
        or quant_config_assembly.get("canonical_module_count") != projection_count
        or quant_config_assembly.get("added_mtp_module_count")
        != expected_added_count
        or quant_config_assembly.get("added_mtp_modules_sha256")
        != sha256_bytes(canonical_json_bytes(expected_added_names))
        or not _digest(quant_config_assembly.get("raw_quant_config_sha256"))
        or quant_config_assembly.get("canonical_quant_config_sha256")
        != sha256_bytes(canonical_json_bytes(quant))
    ):
        raise ValueError("canonical GPTQModel quantization-config assembly is invalid")
    projection_bytes = (
        (hidden_size // 16) * (intermediate_size // 16) * 32 * bits
        + (hidden_size + intermediate_size) * 2
        + 4
    )
    ownership = projection.get("ownership")
    content = projection.get("content")
    if (
        projection.get("projection_count") != projection_count
        or projection.get("expert_family_count") != family_count
        or projection.get("generated_tensor_count") != projection_count * 4
        or projection.get("encoded_bytes") != projection_count * projection_bytes
        or not isinstance(ownership, dict)
        or set(ownership)
        != {
            "coordinator:cuda:0",
            "coordinator:cuda:1",
            "remote_worker:dodo",
            "remote_worker:emu",
            "remote_worker:kiwi",
            "remote_worker:ostrich",
        }
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in ownership.values()
        )
        or sum(ownership.values()) != projection_count
        or not isinstance(content, dict)
        or any(
            not _digest(content.get(field))
            for field in (
                "checkpoint_records_sha256",
                "assignment_records_sha256",
                "journal_records_sha256",
            )
        )
    ):
        raise ValueError("canonical GPTQModel projection assembly is incomplete")

    records = artifact.get("files")
    if (
        not isinstance(records, dict)
        or CANONICAL_ASSEMBLY_FILE not in records
        or sha256_bytes((snapshot / CANONICAL_ASSEMBLY_FILE).read_bytes())
        != records[CANONICAL_ASSEMBLY_FILE].get("sha256")
    ):
        raise ValueError("canonical GPTQModel assembly is not publication-bound")
    _validate_canonical_tensor_storage(quant, model_config)
    return None


def _validate_composite_canonical_assembly(
    snapshot: Path,
    model_config: dict[str, Any],
    quant: dict[str, Any],
    plan: dict[str, Any],
    artifact: dict[str, Any],
    declaration: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    bits = int(quant["bits"])
    recipe = _recipe_for_bits(bits)
    assembly = read_json_object(snapshot / CANONICAL_ASSEMBLY_FILE)
    _validate_bound_record(assembly, "report_sha256", "composite assembly report")
    projection = assembly.get("projection_assembly")
    block_audits = assembly.get("block_audit_set")
    quant_report = assembly.get("quant_config_assembly")
    projection_sources = assembly.get("projection_sources")
    for value, field, label in (
        (projection, "report_sha256", "composite projection assembly"),
        (block_audits, "report_sha256", "composite block-audit set"),
        (quant_report, "report_sha256", "composite quantization config"),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"canonical assembly has no {label}")
        _validate_bound_record(value, field, label)
    if not isinstance(projection_sources, dict) or set(projection_sources) != {
        "base",
        "mtp",
    }:
        raise ValueError("canonical assembly projection sources are incomplete")
    provenance = quant.get("meta", {}).get("ds4rt_error_ledger")
    ledger_sources = (
        provenance.get("projection_sources")
        if isinstance(provenance, dict)
        else None
    )
    if not isinstance(ledger_sources, dict) or set(ledger_sources) != {"base", "mtp"}:
        raise ValueError("composite quantization provenance has incomplete sources")
    if provenance.get("projection_sources_sha256") != sha256_bytes(
        canonical_json_bytes(ledger_sources)
    ):
        raise ValueError("composite quantization source digest is invalid")
    family_joins: dict[str, dict[str, Any]] = {}
    for namespace in ("base", "mtp"):
        source = projection_sources.get(namespace)
        ledger_source = ledger_sources.get(namespace)
        family = ledger_source.get("family_join") if isinstance(ledger_source, dict) else None
        expected_namespaces = [namespace]
        if (
            not isinstance(source, dict)
            or not isinstance(ledger_source, dict)
            or not _digest(source.get("plan_sha256"))
            or ledger_source.get("plan_sha256") != source.get("plan_sha256")
            or ledger_source.get("namespaces") != expected_namespaces
            or not isinstance(family, dict)
            or source.get("family_join_sha256")
            != sha256_bytes(canonical_json_bytes(family))
        ):
            raise ValueError(f"composite {namespace} projection source is inconsistent")
        family_joins[namespace] = family
    base_source = projection_sources["base"]
    mtp_source = projection_sources["mtp"]
    expected_declaration = {
        "schema": COMPOSITE_ASSEMBLY_SCHEMA,
        "filename": CANONICAL_ASSEMBLY_FILE,
        "report_sha256": assembly["report_sha256"],
        "base_plan_sha256": base_source["plan_sha256"],
        "mtp_plan_sha256": mtp_source["plan_sha256"],
        "mtp_overlay_sha256": mtp_source.get("overlay_sha256"),
    }
    portable_source = {
        key: value for key, value in plan.get("source", {}).items() if key != "path"
    }
    if (
        declaration != expected_declaration
        or assembly.get("schema") != COMPOSITE_ASSEMBLY_SCHEMA
        or assembly.get("recipe") != recipe
        or assembly.get("source_snapshot") != portable_source
        or not _digest(mtp_source.get("overlay_sha256"))
        or not _digest(mtp_source.get("overlay_run_sha256"))
    ):
        raise ValueError("canonical composite assembly provenance is inconsistent")

    base_blocks = model_config["num_hidden_layers"]
    mtp_blocks = len(model_config["dspark_target_layer_ids"])
    expert_count = model_config["n_routed_experts"]
    projections_per_block = expert_count * 3
    base_projection_count = base_blocks * projections_per_block
    mtp_projection_count = mtp_blocks * projections_per_block
    projection_count = base_projection_count + mtp_projection_count
    family_count = (base_blocks + mtp_blocks) * expert_count
    projection_bytes = (
        (model_config["hidden_size"] // 16)
        * (model_config["moe_intermediate_size"] // 16)
        * 32
        * bits
        + (model_config["hidden_size"] + model_config["moe_intermediate_size"])
        * 2
        + 4
    )
    source_reports = projection.get("sources") if isinstance(projection, dict) else None
    if not isinstance(source_reports, dict) or set(source_reports) != {"base", "mtp"}:
        raise ValueError("composite projection assembly has incomplete source reports")
    for namespace, expected_count in (
        ("base", base_projection_count),
        ("mtp", mtp_projection_count),
    ):
        report = source_reports[namespace]
        _validate_bound_record(
            report,
            "report_sha256",
            f"composite {namespace} projection source",
        )
        source_ownership = report.get("ownership")
        source_content = report.get("content")
        excluded = report.get("excluded_known_state")
        if (
            report.get("schema") != PROJECTION_ASSEMBLY_SCHEMA
            or report.get("plan_sha256")
            != projection_sources[namespace]["plan_sha256"]
            or report.get("namespaces") != [namespace]
            or report.get("projection_count") != expected_count
            or report.get("expert_family_count") != expected_count // 3
            or report.get("generated_tensor_count") != expected_count * 4
            or report.get("encoded_bytes") != expected_count * projection_bytes
            or report.get("expert_tensor_layout") != "gptqmodel"
            or report.get("retained_tensor_layout")
            != "source_checkpoint_native_byte_exact"
            or not isinstance(source_ownership, dict)
            or not source_ownership
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in source_ownership.values()
            )
            or sum(source_ownership.values()) != expected_count
            or not isinstance(source_content, dict)
            or any(
                not _digest(source_content.get(field))
                for field in (
                    "checkpoint_records_sha256",
                    "assignment_records_sha256",
                    "journal_records_sha256",
                )
            )
            or not isinstance(excluded, dict)
            or set(excluded)
            != {"checkpoint_count", "journal_count", "assignment_count"}
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in excluded.values()
            )
            or (namespace == "mtp" and any(excluded.values()))
        ):
            raise ValueError(f"composite {namespace} projection report is incomplete")
    ownership = projection.get("ownership")
    if (
        projection.get("schema") != COMPOSITE_PROJECTION_ASSEMBLY_SCHEMA
        or projection.get("base_plan_sha256") != base_source["plan_sha256"]
        or projection.get("mtp_plan_sha256") != mtp_source["plan_sha256"]
        or projection.get("projection_count") != projection_count
        or projection.get("expert_family_count") != family_count
        or projection.get("generated_tensor_count") != projection_count * 4
        or projection.get("encoded_bytes") != projection_count * projection_bytes
        or not isinstance(ownership, dict)
        or sum(ownership.values()) != projection_count
    ):
        raise ValueError("canonical composite projection assembly is incomplete")

    base_audits = block_audits.get("base") if isinstance(block_audits, dict) else None
    mtp_audits = block_audits.get("mtp") if isinstance(block_audits, dict) else None
    for subset, namespace, blocks, plan_sha in (
        (base_audits, "base", base_blocks, base_source["plan_sha256"]),
        (mtp_audits, "mtp", mtp_blocks, mtp_source["plan_sha256"]),
    ):
        if not isinstance(subset, dict):
            raise ValueError("canonical composite block-audit subset is missing")
        _validate_bound_record(subset, "report_sha256", f"{namespace} block audits")
        reports = subset.get("reports")
        reports_valid = isinstance(reports, list) and len(reports) == blocks
        if reports_valid:
            for logical_layer, report in enumerate(reports):
                filename = f"{namespace}-layer-{logical_layer}-projection-audit.json"
                if (
                    not isinstance(report, dict)
                    or report.get("ordinal") != logical_layer + 1
                    or report.get("block_namespace") != namespace
                    or report.get("logical_layer") != logical_layer
                    or report.get("filename") != filename
                    or not _digest(report.get("file_sha256"))
                    or not _digest(report.get("report_sha256"))
                ):
                    reports_valid = False
                    break
        if (
            subset.get("schema") != BLOCK_AUDIT_SET_SCHEMA
            or subset.get("plan_sha256") != plan_sha
            or subset.get("base_block_count") != (blocks if namespace == "base" else 0)
            or subset.get("mtp_block_count") != (blocks if namespace == "mtp" else 0)
            or subset.get("block_count") != blocks
            or subset.get("projections_per_block") != projections_per_block
            or subset.get("projection_count") != blocks * projections_per_block
            or subset.get("expert_family_count") != blocks * expert_count
            or not reports_valid
            or subset.get("reports_sha256")
            != sha256_bytes(canonical_json_bytes(reports))
        ):
            raise ValueError(f"canonical composite {namespace} block audits are invalid")
    if (
        block_audits.get("schema") != COMPOSITE_BLOCK_AUDIT_SET_SCHEMA
        or block_audits.get("base_plan_sha256") != base_source["plan_sha256"]
        or block_audits.get("mtp_plan_sha256") != mtp_source["plan_sha256"]
        or block_audits.get("base_block_count") != base_blocks
        or block_audits.get("mtp_block_count") != mtp_blocks
        or block_audits.get("block_count") != base_blocks + mtp_blocks
        or block_audits.get("projection_count") != projection_count
        or block_audits.get("expert_family_count") != family_count
    ):
        raise ValueError("canonical composite block-audit set is invalid")

    storage = quant.get("tensor_storage")
    if (
        quant_report.get("schema") != COMPOSITE_QUANT_CONFIG_ASSEMBLY_SCHEMA
        or quant_report.get("base_plan_sha256") != base_source["plan_sha256"]
        or quant_report.get("mtp_plan_sha256") != mtp_source["plan_sha256"]
        or quant_report.get("canonical_module_count") != projection_count
        or quant_report.get("tensor_storage_sha256")
        != sha256_bytes(canonical_json_bytes(storage))
        or quant_report.get("ledger_provenance_sha256")
        != sha256_bytes(canonical_json_bytes(provenance))
        or quant_report.get("canonical_quant_config_sha256")
        != sha256_bytes(canonical_json_bytes(quant))
    ):
        raise ValueError("canonical composite quantization config is invalid")
    records = artifact.get("files")
    if (
        not isinstance(records, dict)
        or CANONICAL_ASSEMBLY_FILE not in records
        or sha256_bytes((snapshot / CANONICAL_ASSEMBLY_FILE).read_bytes())
        != records[CANONICAL_ASSEMBLY_FILE].get("sha256")
    ):
        raise ValueError("canonical composite assembly is not publication-bound")
    _validate_canonical_tensor_storage(quant, model_config)
    return family_joins


def _valid_recovery_sample_accounting(
    record: dict[str, Any],
    route: dict[str, Any],
    sample_count: int,
) -> bool:
    recovery = record.get("zero_route_recovery")
    if recovery is None:
        return route.get("expert_route_count") == sample_count
    if not isinstance(recovery, dict):
        return False
    natural = recovery.get("natural_sample_count")
    router = recovery.get("router_augmented_sample_count")
    identity = recovery.get("identity_calibration_count")
    authorization = recovery.get("authorization")
    family = record.get("provenance", {}).get("family_join")
    family_digest = (
        sha256_bytes(canonical_json_bytes(family))
        if isinstance(family, dict)
        else None
    )
    return (
        recovery.get("schema") == "ds4rt.exl3-zero-route-recovery"
        and recovery.get("schema_version") == 1
        and recovery.get("trigger") == "natural-route-count-below-1024"
        and recovery.get("sample_source") == "same-fixed-calibration-selection"
        and recovery.get("capture_method")
        == "direct-expert-router-ranks-7-12-then-identity-residual"
        and recovery.get("selection_policy")
        == "rank-ascending-then-fixed-replay-order-v1"
        and recovery.get("candidate_rank_min") == 7
        and recovery.get("candidate_rank_max") == 12
        and recovery.get("target_sample_count") == 1024
        and recovery.get("identity_calibration_policy")
        == "normalized-2i-residual-to-effective-count-1024-v2"
        and (
            recovery.get("block_namespace"),
            recovery.get("logical_layer"),
            recovery.get("expert"),
        )
        == (
            record.get("block_namespace"),
            record.get("logical_layer"),
            record.get("expert"),
        )
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in (natural, router, identity)
        )
        and natural == route.get("expert_route_count")
        and natural < 1024
        and natural + router + identity == sample_count == 1024
        and recovery.get("total_sample_count") == sample_count
        and isinstance(authorization, dict)
        and authorization.get("schema")
        == "ds4rt.exl3-zero-route-recovery-authorization"
        and authorization.get("schema_version") == 1
        and authorization.get("kind") == "immutable-family-join"
        and authorization.get("family_join_sha256") == family_digest
        and authorization.get("authorization_sha256") == family_digest
    )


def _validate_error_ledger(
    snapshot: Path,
    model_config: dict[str, Any],
    family_join: dict[str, Any],
    namespace_family_joins: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Validate complete projection coverage and the FP64 Hessian contract."""

    ledger_payload = (snapshot / LEDGER_FILE).read_bytes()
    manifest = read_json_object(snapshot / LEDGER_MANIFEST_FILE)
    if (
        manifest.get("schema") != "ds4rt.exl3-error-ledger"
        or manifest.get("schema_version") != 1
        or manifest.get("ledger") != LEDGER_FILE
        or manifest.get("ledger_sha256") != sha256_bytes(ledger_payload)
    ):
        raise ValueError("GPTQModel EXL3 error-ledger manifest is invalid")

    bits = family_join.get("bits")
    if isinstance(bits, bool) or bits not in {2, 3}:
        raise ValueError("GPTQModel EXL3 error ledger has an invalid uniform tier")
    records: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(ledger_payload.splitlines(), 1):
            if not line:
                raise ValueError(
                    f"GPTQModel EXL3 error ledger has an empty line at {line_number}"
                )
            record = json.loads(
                line,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON value {value}")
                ),
            )
            if not isinstance(record, dict):
                raise ValueError(
                    f"GPTQModel EXL3 ledger line {line_number} is not an object"
                )
            _validate_bound_record(
                record,
                "record_sha256",
                f"GPTQModel EXL3 ledger line {line_number}",
            )
            if (
                record.get("schema") != "ds4rt.exl3-error-ledger"
                or record.get("schema_version") != 1
                or record.get("record_kind")
                not in {"projection", "expert_family"}
            ):
                raise ValueError(
                    f"GPTQModel EXL3 ledger line {line_number} has a wrong contract"
                )
            records.append(record)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("GPTQModel EXL3 error ledger is not canonical JSONL") from error

    layer_count = model_config.get("num_hidden_layers")
    expert_count = model_config.get("n_routed_experts")
    dspark_targets = model_config.get("dspark_target_layer_ids")
    if (
        isinstance(layer_count, bool)
        or not isinstance(layer_count, int)
        or layer_count <= 0
        or isinstance(expert_count, bool)
        or not isinstance(expert_count, int)
        or expert_count <= 0
        or not isinstance(dspark_targets, list)
    ):
        raise ValueError("GPTQModel EXL3 config cannot define ledger coverage")
    expected_families = {
        (namespace, layer, expert)
        for namespace, layers in (
            ("base", range(layer_count)),
            ("mtp", range(len(dspark_targets))),
        )
        for layer in layers
        for expert in range(expert_count)
    }
    expected_projections = {
        (*family, projection)
        for family in expected_families
        for projection in ("w1", "w2", "w3")
    }

    projections: set[tuple[Any, ...]] = set()
    families: set[tuple[Any, ...]] = set()
    projection_routes: dict[tuple[Any, ...], list[Any]] = {}
    projection_count = 0
    family_count = 0
    for record in records:
        identity = (
            record.get("block_namespace"),
            record.get("logical_layer"),
            record.get("expert"),
        )
        expected_family_join = (
            namespace_family_joins.get(identity[0])
            if namespace_family_joins is not None
            else family_join
        )
        if (
            not isinstance(expected_family_join, dict)
            or record.get("provenance", {}).get("family_join")
            != expected_family_join
        ):
            raise ValueError("GPTQModel EXL3 ledger provenance differs from config")
        if record["record_kind"] == "projection":
            projection_identity = (*identity, record.get("projection"))
            metrics = record.get("quantizer_metrics")
            route = record.get("route_evidence")
            sample_count = record.get("sample_count")
            addend = (
                metrics.get("hessian_regularization_diagonal_addend")
                if isinstance(metrics, dict)
                else None
            )
            correction = (
                metrics.get("hessian_symmetry_correction_max_abs")
                if isinstance(metrics, dict)
                else None
            )
            if (
                projection_identity not in expected_projections
                or projection_identity in projections
                or record.get("bits") != bits
                or record.get("codebook") != "mcg"
                or isinstance(sample_count, bool)
                or not isinstance(sample_count, int)
                or sample_count <= 0
                or not isinstance(metrics, dict)
                or metrics.get("schema") != "gptqmodel.exl3-trellis-error"
                or metrics.get("schema_version") != 1
                or metrics.get("quantizer_path") != "hessian_ldlq"
                or metrics.get("hessian_metric_status") != "ok"
                or metrics.get("hessian_sample_count") != sample_count
                or metrics.get("hessian_regularization_sigma")
                != HESSIAN_NUMERICS["sigma_reg"]
                or metrics.get("hessian_numerical_contract")
                != HESSIAN_NUMERICS["hessian_numerical"]
                or metrics.get("hessian_transform_compute_dtype")
                != "torch.float64"
                or metrics.get("hessian_storage_dtype") != "torch.float32"
                or metrics.get("hessian_regularization_placement")
                != "before-fp64-congruence"
                or metrics.get("hessian_symmetry_restoration")
                != HESSIAN_NUMERICS["hessian_symmetry"]
                or not _finite_number(addend, positive=True)
                or not _finite_number(correction)
                or float(correction) < 0.0
                or not isinstance(route, dict)
                or route.get("schema") != ROUTE_EVIDENCE_CONTRACT
                or route.get("schema_version") != 1
                or (
                    route.get("block_namespace"),
                    route.get("logical_layer"),
                    route.get("expert"),
                )
                != identity
                or not _valid_recovery_sample_accounting(
                    record,
                    route,
                    sample_count,
                )
            ):
                raise ValueError(
                    f"GPTQModel EXL3 ledger has unusable projection {projection_identity}"
                )
            projections.add(projection_identity)
            projection_routes.setdefault(identity, []).append(route)
            projection_count += 1
        else:
            aggregate = record.get("aggregate_metrics")
            if (
                identity not in expected_families
                or identity in families
                or record.get("bits") != bits
                or record.get("codebook") != "mcg"
                or record.get("projections") != ["w1", "w2", "w3"]
                or not isinstance(aggregate, dict)
                or aggregate.get("hessian_metric_status") != "ok"
            ):
                raise ValueError(
                    f"GPTQModel EXL3 ledger has unusable expert family {identity}"
                )
            routes = projection_routes.get(identity)
            if (
                routes is None
                or len(routes) != 3
                or any(route != routes[0] for route in routes[1:])
                or record.get("route_evidence") != routes[0]
            ):
                raise ValueError(
                    f"GPTQModel EXL3 ledger has inconsistent expert family {identity}"
                )
            families.add(identity)
            family_count += 1

    if (
        manifest.get("projection_records") != projection_count
        or manifest.get("complete_family_records") != family_count
        or manifest.get("total_records") != len(records)
        or projections != expected_projections
        or families != expected_families
    ):
        raise ValueError("GPTQModel EXL3 error ledger has incomplete coverage")


def _expected_source_geometry(
    model_config: dict[str, Any],
    plan_schema: Any,
) -> dict[str, Any]:
    geometry = {
        key: model_config.get(key)
        for key in (
            "num_hidden_layers",
            "n_routed_experts",
            "hidden_size",
            "moe_intermediate_size",
            "dspark_target_layer_ids",
        )
    }
    if plan_schema == PLAN_SCHEMA:
        geometry.update(
            {
                "mtp_block_count": len(model_config.get("dspark_target_layer_ids", [])),
                "num_hash_layers": model_config.get("num_hash_layers"),
                "num_experts_per_tok": model_config.get("num_experts_per_tok"),
                "hc_mult": model_config.get("hc_mult"),
                "num_nextn_predict_layers": model_config.get(
                    "num_nextn_predict_layers"
                ),
            }
        )
    return geometry


def validate_gptqmodel_publication(
    snapshot: Path,
    model_config: dict[str, Any],
    *,
    verify_all_hashes: bool,
    require_canonical: bool = False,
) -> dict[str, Any]:
    """Validate the immutable publication envelope without rewriting the artifact."""

    quant = model_config.get("quantization_config")
    family = validate_gptqmodel_native_exl3(quant, model_config=model_config)
    published_quant = read_json_object(snapshot / "quantize_config.json")
    if published_quant != quant:
        raise ValueError(
            "config.json and quantize_config.json contain different EXL3 contracts"
        )
    plan = read_json_object(snapshot / PLAN_FILE)
    artifact = read_json_object(snapshot / ARTIFACT_FILE)
    run = read_json_object(snapshot / RUN_FILE)
    _validate_bound_record(plan, "plan_sha256", "GPTQModel quantization plan")
    _validate_bound_record(artifact, "manifest_sha256", "GPTQModel artifact manifest")
    _validate_bound_record(run, "run_sha256", "GPTQModel run manifest")
    plan_sha = plan["plan_sha256"]
    plan_schema = plan.get("schema")
    expected_geometry = _expected_source_geometry(model_config, plan_schema)
    if (
        plan_schema not in SUPPORTED_PLAN_SCHEMAS
        or plan.get("recipe") != _recipe_for_bits(int(quant["bits"]))
        or plan.get("ledger_provenance") != quant["meta"]["ds4rt_error_ledger"]
        or plan.get("source", {}).get("geometry") != expected_geometry
        or plan.get("exl3") != _expected_plan_exl3(family)
        or artifact.get("schema") != ARTIFACT_SCHEMA
        or artifact.get("plan_sha256") != plan_sha
        or run.get("schema") != RUN_SCHEMA
        or run.get("status") != "complete"
        or run.get("plan_sha256") != plan_sha
        or run.get("artifact_manifest_sha256") != artifact.get("manifest_sha256")
        or isinstance(run.get("mtp_replay_batches"), bool)
        or not isinstance(run.get("mtp_replay_batches"), int)
        or run["mtp_replay_batches"] <= 0
        or family != plan["ledger_provenance"]["family_join"]
    ):
        raise ValueError("GPTQModel EXL3 publication manifests are inconsistent")

    records = artifact.get("files")
    if not isinstance(records, dict) or not records:
        raise ValueError("GPTQModel EXL3 artifact manifest has no file records")
    expected_files = set(records) | {ARTIFACT_FILE, RUN_FILE}
    actual_files: set[str] = set()
    for root, directories, names in os.walk(snapshot, followlinks=False):
        root_path = Path(root)
        if any((root_path / name).is_symlink() for name in directories):
            raise ValueError("GPTQModel artifact contains a symlinked directory")
        for name in names:
            actual_files.add((root_path / name).relative_to(snapshot).as_posix())
    if actual_files != expected_files:
        raise ValueError("GPTQModel artifact file set differs from its manifest")

    total_bytes = 0
    for raw_path, record in records.items():
        relative = _safe_path(raw_path)
        if (
            not isinstance(record, dict)
            or isinstance(record.get("bytes"), bool)
            or not isinstance(record.get("bytes"), int)
            or record["bytes"] < 0
            or not _digest(record.get("sha256"))
        ):
            raise ValueError(f"GPTQModel artifact has invalid record {raw_path!r}")
        path = snapshot.joinpath(*relative.parts)
        staged_blob = _canonical_staged_blob(snapshot, path, record["sha256"])
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ValueError(f"GPTQModel artifact entry differs in size: {raw_path}")
        if verify_all_hashes and not staged_blob:
            if sha256_bytes(path.read_bytes()) != record["sha256"]:
                raise ValueError(f"GPTQModel artifact entry failed hashing: {raw_path}")
        total_bytes += record["bytes"]
    if (
        artifact.get("file_count") != len(records)
        or artifact.get("total_bytes") != total_bytes
        or not any(name.endswith(".safetensors") for name in records)
    ):
        raise ValueError("GPTQModel artifact aggregate file accounting is invalid")
    for required in (
        "config.json",
        "model.safetensors.index.json",
        "quantize_config.json",
        PLAN_FILE,
        LEDGER_FILE,
        LEDGER_MANIFEST_FILE,
    ):
        if required not in records:
            raise ValueError(f"GPTQModel artifact manifest is missing {required}")
        path = snapshot / required
        if sha256_bytes(path.read_bytes()) != records[required]["sha256"]:
            raise ValueError(f"GPTQModel artifact critical file failed hashing: {required}")
    namespace_family_joins = None
    if require_canonical:
        namespace_family_joins = _validate_canonical_assembly(
            snapshot,
            model_config,
            quant,
            plan,
            artifact,
        )
    _validate_error_ledger(
        snapshot,
        model_config,
        family,
        namespace_family_joins=namespace_family_joins,
    )
    return records


__all__ = [
    "ARTIFACT_FILE",
    "BASE_EXPERT_PATTERN",
    "CANONICAL_ASSEMBLY_FILE",
    "LEDGER_FILE",
    "LEDGER_MANIFEST_FILE",
    "PLAN_FILE",
    "RECIPE",
    "RUN_FILE",
    "is_gptqmodel_native_exl3",
    "validate_gptqmodel_native_exl3",
    "validate_gptqmodel_publication",
]
