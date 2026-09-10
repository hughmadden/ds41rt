#!/usr/bin/env python3
"""Stage a complete DS41RT EXL3 artifact as an immutable Hugging Face snapshot."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ds41rt_runtime.exl3_artifact_contract import (  # noqa: E402
    ARTIFACT_FILE as GPTQMODEL_ARTIFACT_FILE,
    LEDGER_FILE as GPTQMODEL_LEDGER_FILE,
    LEDGER_MANIFEST_FILE as GPTQMODEL_LEDGER_MANIFEST_FILE,
    PLAN_FILE as GPTQMODEL_PLAN_FILE,
    RECIPE as GPTQMODEL_RECIPE,
    RECIPE_K3 as GPTQMODEL_RECIPE_K3,
    RUN_FILE as GPTQMODEL_RUN_FILE,
    _valid_recovery_sample_accounting,
    is_gptqmodel_native_exl3,
    validate_gptqmodel_native_exl3,
    validate_gptqmodel_publication,
)
from prepare_ds4_hf_publication import publication_sources  # noqa: E402

MANIFEST_SCHEMA = "ds41rt-hf-staged-snapshot-v1"
EXL3_V2_RECIPE = "deepseek_v4_exl3_trellis_2bpw_v2"
EXL3_V3_RECIPE = "deepseek_v4_exl3_trellis_2bpw_v3_flash_activation_pilot"
EXL3_V4_RECIPE = "deepseek_v4_exl3_trellis_2bpw_v4_flash_natural_route"
EXL3_K3_V4_RECIPE = "deepseek_v4_exl3_trellis_3bpw_v4_flash_natural_route"
EXL3_RECIPES = frozenset(
    (EXL3_V2_RECIPE, EXL3_V3_RECIPE, EXL3_V4_RECIPE, EXL3_K3_V4_RECIPE)
)
DEBUG_DIRECTORY = ".ds41rt-exl3-debug"
QUALITY_REPORT_SCHEMA = "ds41rt-exl3-checkpoint-quality-v1"
QUALITY_GPU_SCHEMA = "ds41rt-quality-physical-gpu-v1"
GPTQMODEL_PRODUCTION_SAMPLING_POLICY = "natural-target-stratified-mtp-rms-isotropic"
DEVELOPMENT_QUALIFICATION_STATUS = "development-unqualified"
DEVELOPMENT_BLOCKERS = (
    "complete token-aware all-layer quality evidence is absent",
    "natural MTP held-out activation quality evidence is absent",
)
MIN_QUANT_COSINE = 0.88
MAX_QUANT_RELATIVE_L2 = 0.50
MIN_DSPARK_QUANT_COSINE = 0.84
MAX_DSPARK_QUANT_RELATIVE_L2 = 0.56
MIN_TP_COSINE = 0.999
MAX_TP_RELATIVE_L2 = 0.03
MULTIGPU_QUALIFICATION_FILE = "ds41rt-exl3-multigpu-production.json"
ERROR_LEDGER_FILE = "ds41rt-exl3-error-ledger.jsonl"
ERROR_LEDGER_MANIFEST_FILE = "ds41rt-exl3-error-ledger.manifest.json"
ERROR_LEDGER_SCHEMA = "ds41rt.exl3-error-ledger"
ROUTE_EVIDENCE_SCHEMA = "ds41rt.exl3-natural-route"
REQUIRED_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "quantization_config.json",
    "ds41rt-exl3-calibration.json",
    "ds41rt-exl3-quality.json",
    "ds41rt-exl3-retained-native.json",
)
GPTQMODEL_REQUIRED_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "quantize_config.json",
    GPTQMODEL_PLAN_FILE,
    GPTQMODEL_RUN_FILE,
    GPTQMODEL_ARTIFACT_FILE,
    GPTQMODEL_LEDGER_FILE,
    GPTQMODEL_LEDGER_MANIFEST_FILE,
)
MODEL_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
GPU_UUID_RE = re.compile(
    r"GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\Z"
)
PCI_BUS_ID_RE = re.compile(r"[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]\Z")
COMPUTE_CAPABILITY_RE = re.compile(r"[0-9]+\.[0-9]+\Z")


@dataclass(frozen=True)
class ArtifactFile:
    source: Path
    relative: PurePosixPath
    size: int
    sha256: str

    def manifest_entry(self) -> dict[str, str | int]:
        return {
            "path": self.relative.as_posix(),
            "sha256": self.sha256,
            "size": self.size,
        }


@dataclass(frozen=True)
class QualificationReport:
    source: Path
    name: str
    size: int
    sha256: str
    schema: str
    contract_sha256: str | None = None

    def manifest_entry(self) -> dict[str, str | int]:
        entry: dict[str, str | int] = {
            "path": self.name,
            "sha256": self.sha256,
            "size": self.size,
            "schema": self.schema,
        }
        if self.contract_sha256 is not None:
            entry["contract_sha256"] = self.contract_sha256
        return entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--model-id",
        required=True,
        help="dedicated Hugging Face cache ID, for example org/model-exl3",
    )
    parser.add_argument("--hf-home", type=Path, default=default_hf_home())
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="hardlink avoids duplicating artifact bytes; copy isolates the cache",
    )
    parser.add_argument(
        "--update-ref",
        action="store_true",
        help="allow refs/main to move from an existing different revision",
    )
    parser.add_argument(
        "--retained-native-report",
        type=Path,
        help=(
            "external retained-native report; required for immutable GPTQModel "
            "publications"
        ),
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        help=(
            "external all-layer held-out quality report; required for immutable "
            "GPTQModel publications"
        ),
    )
    parser.add_argument(
        "--development-unqualified",
        action="store_true",
        help=(
            "stage artifact-bound retained-native and diagnostic quality evidence "
            "for WIP serving without granting production qualification"
        ),
    )
    parser.add_argument(
        "--standard-publication",
        action="store_true",
        help=(
            "stage an already-prepared standard-only Hugging Face publication "
            "without private quantization evidence"
        ),
    )
    return parser.parse_args()


def default_hf_home() -> Path:
    configured = os.environ.get("HF_HOME")
    return (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".cache/huggingface"
    )


def validate_model_id(model_id: str) -> tuple[str, str]:
    components = model_id.split("/")
    if len(components) != 2 or any(
        MODEL_COMPONENT_RE.fullmatch(component) is None for component in components
    ):
        raise ValueError(
            "--model-id must be a two-component Hugging Face ID using letters, "
            "digits, dot, underscore, or hyphen"
        )
    return components[0], components[1]


def model_cache_dir(hf_home: Path, model_id: str) -> Path:
    organization, repository = validate_model_id(model_id)
    return hf_home / "hub" / f"models--{organization}--{repository}"


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot read valid JSON object from {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def validate_quantization_contract(
    quant: Any,
    source: str,
    *,
    model_config: dict[str, Any] | None = None,
) -> str:
    if not isinstance(quant, dict):
        raise ValueError(f"{source} has no quantization configuration object")
    if str(quant.get("quant_method", "")).lower() != "exl3":
        raise ValueError(f"{source} must declare quant_method=exl3")
    if is_gptqmodel_native_exl3(quant):
        validate_gptqmodel_native_exl3(quant, model_config=model_config)
        return EXL3_V4_RECIPE if int(quant["bits"]) == 2 else EXL3_K3_V4_RECIPE
    ds41rt = quant.get("ds41rt")
    if not isinstance(ds41rt, dict):
        raise ValueError(f"{source} has no quantization_config.ds41rt object")
    if ds41rt.get("calibrated") is not True:
        raise ValueError(f"{source} must declare ds41rt.calibrated=true")
    recipe = ds41rt.get("recipe")
    if recipe not in EXL3_RECIPES:
        raise ValueError(
            f"{source} must declare a recognized DS41RT recipe, got {recipe!r}"
        )
    return str(recipe)


def validate_index_path(raw: Any) -> PurePosixPath:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise ValueError(f"invalid shard path in model index: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe shard path in model index: {raw!r}")
    if path.suffix != ".safetensors":
        raise ValueError(f"model index does not reference a safetensors shard: {raw!r}")
    return path


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"EXL3 quality report has invalid {label}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"EXL3 quality report has non-finite {label}")
    return number


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def validate_ledger_route_evidence(
    evidence: Any,
    *,
    identity: tuple[Any, ...],
    sample_count: int,
) -> dict[str, Any]:
    """Validate one projection's natural-route exposure and gate mass."""

    integer_fields = (
        "router_calls",
        "router_token_count",
        "router_selected_route_count",
        "router_top_k",
    )
    if (
        not isinstance(evidence, dict)
        or evidence.get("schema") != ROUTE_EVIDENCE_SCHEMA
        or evidence.get("schema_version") != 1
        or evidence.get("block_namespace") != identity[0]
        or evidence.get("logical_layer") != identity[1]
        or evidence.get("expert") != identity[2]
        or any(
            isinstance(evidence.get(field), bool)
            or not isinstance(evidence.get(field), int)
            or evidence[field] <= 0
            for field in integer_fields
        )
        or isinstance(evidence.get("expert_route_count"), bool)
        or not isinstance(evidence.get("expert_route_count"), int)
        or evidence["expert_route_count"] < 0
        or evidence.get("router_selected_route_count")
        != evidence.get("router_token_count") * evidence.get("router_top_k")
        or not isinstance(evidence.get("router_weight_dtypes"), list)
        or not evidence["router_weight_dtypes"]
        or not all(
            isinstance(value, str) and value
            for value in evidence["router_weight_dtypes"]
        )
        or not isinstance(evidence.get("mask_modes"), list)
        or not evidence["mask_modes"]
        or not all(isinstance(value, str) and value for value in evidence["mask_modes"])
    ):
        raise ValueError(
            f"EXL3 error ledger has invalid natural-route evidence for {identity}"
        )

    values = {
        field: finite_number(evidence.get(field), f"ledger route {field}")
        for field in (
            "expert_gate_weight_sum",
            "expert_gate_squared_mass",
            "total_gate_weight_sum",
            "total_gate_squared_mass",
            "expert_route_fraction",
            "expert_gate_weight_mass_fraction",
            "expert_gate_squared_mass_fraction",
            "expert_gate_weight_mean",
            "expert_gate_weight_rms",
        )
    }
    route_count = evidence["expert_route_count"]
    selected_count = evidence["router_selected_route_count"]
    gate_sum = values["expert_gate_weight_sum"]
    gate_sq = values["expert_gate_squared_mass"]
    total_gate_sum = values["total_gate_weight_sum"]
    total_gate_sq = values["total_gate_squared_mass"]
    if gate_sum < 0 or gate_sq < 0 or total_gate_sum <= 0 or total_gate_sq <= 0:
        raise ValueError(
            f"EXL3 error ledger has inconsistent natural-route evidence for {identity}"
        )
    expected = (
        {
            "expert_route_fraction": route_count / selected_count,
            "expert_gate_weight_mass_fraction": gate_sum / total_gate_sum,
            "expert_gate_squared_mass_fraction": gate_sq / total_gate_sq,
            "expert_gate_weight_mean": gate_sum / route_count,
            "expert_gate_weight_rms": math.sqrt(gate_sq / route_count),
        }
        if route_count > 0
        else {
            "expert_route_fraction": 0.0,
            "expert_gate_weight_mass_fraction": 0.0,
            "expert_gate_squared_mass_fraction": 0.0,
            "expert_gate_weight_mean": 0.0,
            "expert_gate_weight_rms": 0.0,
        }
    )
    if (
        route_count > selected_count
        or (route_count == 0 and (gate_sum != 0 or gate_sq != 0))
        or total_gate_sum < gate_sum
        or total_gate_sq < gate_sq
        or any(
            not math.isclose(values[field], expected_value, rel_tol=1e-9, abs_tol=1e-12)
            for field, expected_value in expected.items()
        )
    ):
        raise ValueError(
            f"EXL3 error ledger has inconsistent natural-route evidence for {identity}"
        )
    return evidence


def validate_error_ledger(
    snapshot: Path,
    config: dict[str, Any],
    *,
    expected_family_join: dict[str, Any] | None = None,
    namespace_family_joins: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Require a complete, content-bound trellis ledger for production v4."""

    ledger_path = snapshot / ERROR_LEDGER_FILE
    manifest_path = snapshot / ERROR_LEDGER_MANIFEST_FILE
    for path in (ledger_path, manifest_path):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"production EXL3 artifact is missing error ledger {path}")

    ledger_payload = ledger_path.read_bytes()
    manifest = read_json_object(manifest_path)
    if (
        manifest.get("schema") != ERROR_LEDGER_SCHEMA
        or manifest.get("schema_version") != 1
        or manifest.get("ledger") != ERROR_LEDGER_FILE
        or manifest.get("ledger_sha256") != hashlib.sha256(ledger_payload).hexdigest()
    ):
        raise ValueError("EXL3 error-ledger manifest or payload digest is invalid")

    records: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(ledger_payload.splitlines(), 1):
            if not line:
                raise ValueError(
                    f"EXL3 error ledger has an empty line at {line_number}"
                )
            record = json.loads(
                line,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON value {value}")
                ),
            )
            if not isinstance(record, dict):
                raise ValueError(
                    f"EXL3 error-ledger line {line_number} is not an object"
                )
            claimed = record.get("record_sha256")
            unbound = {
                key: value for key, value in record.items() if key != "record_sha256"
            }
            if claimed != hashlib.sha256(canonical_json_bytes(unbound)).hexdigest():
                raise ValueError(
                    f"EXL3 error-ledger record {line_number} has an invalid digest"
                )
            if (
                record.get("schema") != ERROR_LEDGER_SCHEMA
                or record.get("schema_version") != 1
                or record.get("record_kind") not in {"projection", "expert_family"}
            ):
                raise ValueError(
                    f"EXL3 error-ledger record {line_number} has a wrong contract"
                )
            records.append(record)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"EXL3 error ledger is not canonical JSONL: {error}"
        ) from error

    projections = [
        record for record in records if record["record_kind"] == "projection"
    ]
    families = [
        record for record in records if record["record_kind"] == "expert_family"
    ]
    if (
        manifest.get("projection_records") != len(projections)
        or manifest.get("complete_family_records") != len(families)
        or manifest.get("total_records") != len(records)
    ):
        raise ValueError("EXL3 error-ledger manifest counts do not match its records")

    layer_count = config.get("num_hidden_layers")
    expert_count = config.get("n_routed_experts")
    dspark_targets = config.get("dspark_target_layer_ids")
    quant = config.get("quantization_config")
    expected_bits = quant.get("bits", 2) if isinstance(quant, dict) else None
    if (
        isinstance(layer_count, bool)
        or not isinstance(layer_count, int)
        or layer_count <= 0
        or isinstance(expert_count, bool)
        or not isinstance(expert_count, int)
        or expert_count <= 0
        or not isinstance(dspark_targets, list)
        or isinstance(expected_bits, bool)
        or expected_bits not in (2, 2.0, 3, 3.0)
    ):
        raise ValueError("EXL3 config cannot define error-ledger coverage")
    expected_bits = int(expected_bits)
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
    projection_storage = ledger_projection_storage_contract(
        quant,
        expected_projections,
        default_bits=expected_bits,
    )
    mixed_projection_storage = any(
        bits != expected_bits for _, bits in projection_storage.values()
    )

    actual_projections: set[tuple[Any, ...]] = set()
    projections_by_family: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in projections:
        identity = (
            record.get("block_namespace"),
            record.get("logical_layer"),
            record.get("expert"),
            record.get("projection"),
        )
        if identity in actual_projections:
            raise ValueError(f"EXL3 error ledger duplicates projection {identity}")
        actual_projections.add(identity)
        projections_by_family.setdefault(identity[:3], []).append(record)
        record_family_join = (
            namespace_family_joins.get(str(identity[0]))
            if namespace_family_joins is not None
            else expected_family_join
        )
        metrics = record.get("quantizer_metrics")
        reconstruction = (
            metrics.get("reconstruction") if isinstance(metrics, dict) else None
        )
        numerator = (
            metrics.get("hessian_weighted_error_numerator")
            if isinstance(metrics, dict)
            else None
        )
        denominator = (
            metrics.get("hessian_weighted_reference_denominator")
            if isinstance(metrics, dict)
            else None
        )
        ratio = (
            metrics.get("hessian_weighted_relative_error")
            if isinstance(metrics, dict)
            else None
        )
        expected_module, projection_bits = projection_storage[identity]
        if (
            record.get("bits") != projection_bits
            or (
                expected_module is not None
                and record.get("module") != expected_module
            )
            or record.get("codebook") != "mcg"
            or not isinstance(record.get("sample_count"), int)
            or record["sample_count"] <= 0
            or not isinstance(record.get("encoded_bytes"), int)
            or record["encoded_bytes"] <= 0
            or not isinstance(record.get("provenance"), dict)
            or not isinstance(metrics, dict)
            or metrics.get("schema") != "gptqmodel.exl3-trellis-error"
            or metrics.get("schema_version") != 1
            or metrics.get("quantizer_path") != "hessian_ldlq"
            or metrics.get("reported_metric_kind") != "hessian_weighted_relative_error"
            or metrics.get("hessian_metric_status") != "ok"
            or metrics.get("hessian_sample_count") != record["sample_count"]
            or (
                record_family_join is not None
                and record["provenance"].get("family_join") != record_family_join
            )
            or (
                record_family_join is not None
                and (
                    metrics.get("hessian_numerical_contract")
                    != "signed-block-hadamard-congruence-fp64-v1"
                    or metrics.get("hessian_transform_compute_dtype") != "torch.float64"
                    or metrics.get("hessian_storage_dtype") != "torch.float32"
                    or metrics.get("hessian_regularization_placement")
                    != "before-fp64-congruence"
                    or metrics.get("hessian_regularization_sigma") != 0.025
                    or metrics.get("hessian_symmetry_restoration")
                    != "mean-with-transpose-fp64"
                )
            )
            or not isinstance(reconstruction, dict)
            or reconstruction.get("domain") != "regularized_exl3_search_space"
            or reconstruction.get("reference_finite") is not True
            or reconstruction.get("error_finite") is not True
        ):
            raise ValueError(
                f"EXL3 error ledger has unusable metrics for projection {identity}"
            )
        if record_family_join is not None:
            if (
                finite_number(
                    metrics.get("hessian_regularization_diagonal_addend"),
                    "ledger Hessian regularization diagonal addend",
                )
                <= 0.0
                or finite_number(
                    metrics.get("hessian_symmetry_correction_max_abs"),
                    "ledger Hessian symmetry correction",
                )
                < 0.0
            ):
                raise ValueError(
                    "EXL3 error ledger has invalid Hessian stabilization terms "
                    f"for projection {identity}"
                )
        route_evidence = validate_ledger_route_evidence(
            record.get("route_evidence"),
            identity=identity,
            sample_count=record["sample_count"],
        )
        if not _valid_recovery_sample_accounting(
            record,
            route_evidence,
            record["sample_count"],
        ):
            raise ValueError(
                f"EXL3 error ledger has invalid recovery sample accounting for {identity}"
            )
        numerator = finite_number(numerator, "ledger Hessian numerator")
        denominator = finite_number(denominator, "ledger Hessian denominator")
        ratio = finite_number(ratio, "ledger Hessian ratio")
        reported = finite_number(
            metrics.get("reported_metric_value"), "ledger reported metric"
        )
        if (
            numerator < 0
            or denominator <= 0
            or ratio < 0
            or not math.isclose(
                ratio, numerator / denominator, rel_tol=1e-6, abs_tol=1e-12
            )
            or not math.isclose(reported, ratio, rel_tol=1e-6, abs_tol=1e-12)
        ):
            raise ValueError(
                f"EXL3 error ledger has inconsistent Hessian terms for {identity}"
            )
        for label in (
            "error_sum_sq",
            "reference_sum_sq",
            "mse",
            "nmse",
            "relative_frobenius",
            "mean_abs_error",
            "max_abs_error",
        ):
            if (
                finite_number(
                    reconstruction.get(label), f"ledger reconstruction {label}"
                )
                < 0
            ):
                raise ValueError(
                    f"EXL3 error ledger has negative reconstruction {label}"
                )
        element_count = reconstruction.get("element_count")
        error_sum_sq = float(reconstruction["error_sum_sq"])
        reference_sum_sq = float(reconstruction["reference_sum_sq"])
        if (
            isinstance(element_count, bool)
            or not isinstance(element_count, int)
            or element_count <= 0
            or reference_sum_sq <= 0
            or not math.isclose(
                float(reconstruction["mse"]),
                error_sum_sq / element_count,
                rel_tol=1e-6,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(reconstruction["nmse"]),
                error_sum_sq / reference_sum_sq,
                rel_tol=1e-6,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(reconstruction["relative_frobenius"]),
                math.sqrt(float(reconstruction["nmse"])),
                rel_tol=1e-6,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                f"EXL3 error ledger has inconsistent reconstruction terms for {identity}"
            )

    actual_families: set[tuple[Any, ...]] = set()
    for record in families:
        identity = (
            record.get("block_namespace"),
            record.get("logical_layer"),
            record.get("expert"),
        )
        if identity in actual_families:
            raise ValueError(f"EXL3 error ledger duplicates expert family {identity}")
        actual_families.add(identity)
        record_family_join = (
            namespace_family_joins.get(str(identity[0]))
            if namespace_family_joins is not None
            else expected_family_join
        )
        if (
            record.get("codebook") != "mcg"
            or record.get("projections") != ["w1", "w2", "w3"]
            or not isinstance(record.get("aggregate_metrics"), dict)
            or (
                record_family_join is not None
                and not isinstance(record.get("provenance"), dict)
            )
            or (
                record_family_join is not None
                and record["provenance"].get("family_join") != record_family_join
            )
        ):
            raise ValueError(
                f"EXL3 error ledger has malformed expert family {identity}"
            )
        components = projections_by_family.get(identity, [])
        if len(components) != 3:
            raise ValueError(
                f"EXL3 error ledger has an incomplete expert family {identity}"
            )
        components_by_projection = {
            component["projection"]: component for component in components
        }
        component_bits = {
            projection: components_by_projection[projection]["bits"]
            for projection in ("w1", "w2", "w3")
        }
        sample_counts = [
            components_by_projection[projection]["sample_count"]
            for projection in ("w1", "w2", "w3")
        ]
        if (
            record.get("bits") != min(component_bits.values())
            or (
                mixed_projection_storage
                and (
                    record.get("projection_bits") != component_bits
                    or record.get("mixed_bits")
                    != (len(set(component_bits.values())) > 1)
                    or record.get("sample_counts") != sample_counts
                )
            )
        ):
            raise ValueError(
                f"EXL3 error ledger has malformed expert family {identity}"
            )
        component_route_evidence = [
            component.get("route_evidence") for component in components
        ]
        if (
            any(
                value != component_route_evidence[0]
                for value in component_route_evidence[1:]
            )
            or record.get("route_evidence") != component_route_evidence[0]
        ):
            raise ValueError(
                f"EXL3 error ledger has inconsistent family route evidence for {identity}"
            )
        aggregate = record["aggregate_metrics"]
        expected_error = sum(
            component["quantizer_metrics"]["reconstruction"]["error_sum_sq"]
            for component in components
        )
        expected_reference = sum(
            component["quantizer_metrics"]["reconstruction"]["reference_sum_sq"]
            for component in components
        )
        expected_elements = sum(
            component["quantizer_metrics"]["reconstruction"]["element_count"]
            for component in components
        )
        expected_hessian_num = sum(
            component["quantizer_metrics"]["hessian_weighted_error_numerator"]
            for component in components
        )
        expected_hessian_den = sum(
            component["quantizer_metrics"]["hessian_weighted_reference_denominator"]
            for component in components
        )
        expected_aggregate = {
            "error_sum_sq": expected_error,
            "reference_sum_sq": expected_reference,
            "mse": expected_error / expected_elements,
            "nmse": expected_error / expected_reference,
            "relative_frobenius": math.sqrt(expected_error / expected_reference),
            "hessian_weighted_error_numerator": expected_hessian_num,
            "hessian_weighted_reference_denominator": expected_hessian_den,
            "hessian_weighted_relative_error": expected_hessian_num
            / expected_hessian_den,
        }
        if aggregate.get("element_count") != expected_elements or any(
            not math.isclose(
                finite_number(aggregate.get(label), f"ledger family {label}"),
                expected,
                rel_tol=1e-6,
                abs_tol=1e-12,
            )
            for label, expected in expected_aggregate.items()
        ):
            raise ValueError(
                f"EXL3 error ledger has inconsistent expert family {identity}"
            )

    route_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for identity, components in projections_by_family.items():
        if components:
            route_groups.setdefault(identity[:2], []).append(
                components[0]["route_evidence"]
            )
    for block, evidence_records in route_groups.items():
        first = evidence_records[0]
        shared_fields = (
            "router_calls",
            "router_token_count",
            "router_selected_route_count",
            "router_top_k",
            "total_gate_weight_sum",
            "total_gate_squared_mass",
            "router_weight_dtypes",
            "mask_modes",
        )
        if any(
            any(record.get(field) != first.get(field) for field in shared_fields)
            for record in evidence_records[1:]
        ):
            raise ValueError(
                f"EXL3 error ledger has inconsistent layer route totals for {block}"
            )
        if (
            sum(record["expert_route_count"] for record in evidence_records)
            != first["router_selected_route_count"]
            or not math.isclose(
                sum(record["expert_gate_weight_sum"] for record in evidence_records),
                first["total_gate_weight_sum"],
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            or not math.isclose(
                sum(record["expert_gate_squared_mass"] for record in evidence_records),
                first["total_gate_squared_mass"],
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                f"EXL3 error ledger route distribution does not close for {block}"
            )

    if (
        actual_projections != expected_projections
        or actual_families != expected_families
    ):
        raise ValueError(
            "EXL3 error ledger does not exactly cover every routed projection and expert family"
        )


def ledger_projection_storage_contract(
    quant: dict[str, Any],
    expected_projections: set[tuple[Any, ...]],
    *,
    default_bits: int,
) -> dict[tuple[Any, ...], tuple[str | None, int]]:
    """Resolve each ledger projection to its physical GPTQModel tier."""

    storage = quant.get("tensor_storage")
    if not isinstance(storage, dict):
        return {
            identity: (None, default_bits) for identity in expected_projections
        }
    projection_names = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
    contract = {}
    for identity in expected_projections:
        namespace, layer, expert, projection = identity
        block = (
            f"model.layers.{layer}" if namespace == "base" else f"mtp.{layer}"
        )
        module = (
            f"{block}.mlp.experts.{expert}.{projection_names[str(projection)]}"
        )
        entry = storage.get(module)
        physical_bits = (
            entry.get("bits_per_weight") if isinstance(entry, dict) else None
        )
        if type(physical_bits) is not int or physical_bits not in {2, 3}:
            raise ValueError(
                f"EXL3 tensor_storage has no physical tier for ledger module {module}"
            )
        contract[identity] = (module, physical_bits)
    return contract


def quality_snapshot_identity(snapshot: Path) -> dict[str, Any]:
    snapshot = snapshot.resolve(strict=True)
    metadata = {}
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "quantize_config.json",
        "ds41rt-exl3-calibration.json",
        GPTQMODEL_PLAN_FILE,
        GPTQMODEL_RUN_FILE,
        GPTQMODEL_ARTIFACT_FILE,
        GPTQMODEL_LEDGER_MANIFEST_FILE,
    ):
        path = snapshot / name
        if path.is_file():
            metadata[name] = hash_file(path)
    shards = []
    for path in sorted(snapshot.glob("*.safetensors")):
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
    if not shards:
        raise ValueError(f"checkpoint has no safetensor shards: {snapshot}")
    return {"path": str(snapshot), "metadata_sha256": metadata, "shards": shards}


def validate_quality_binding(
    quality: dict[str, Any],
    snapshot: Path,
    expected_layer_ids: list[int],
    expert_ids: list[int],
    thresholds: dict[str, Any],
) -> None:
    contract = quality.get("validation_contract")
    if not isinstance(contract, dict):
        raise ValueError("EXL3 quality report has no checkpoint validation contract")
    claimed_sha256 = contract.get("sha256")
    payload = {key: value for key, value in contract.items() if key != "sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    if claimed_sha256 != hashlib.sha256(encoded).hexdigest():
        raise ValueError("EXL3 quality report validation contract digest is invalid")
    if contract.get("allow_incomplete") is not False:
        raise ValueError(
            "EXL3 quality report was not collected from a published artifact"
        )
    if (
        contract.get("layer_ids") != expected_layer_ids
        or contract.get("expert_ids") != expert_ids
        or contract.get("rows") != quality.get("rows")
        or contract.get("seed") != quality.get("seed")
        or contract.get("thresholds") != thresholds
    ):
        raise ValueError("EXL3 quality report does not match its validation contract")
    if contract.get("exl3") != quality_snapshot_identity(snapshot):
        raise ValueError(
            "EXL3 quality report is stale or belongs to another checkpoint"
        )


def validate_quality_execution_gpu(contract: dict[str, Any]) -> str:
    identity = contract.get("execution_gpu")
    if not isinstance(identity, dict):
        raise ValueError(
            "GPTQModel quality report does not bind physical coordinator GPU0"
        )
    claimed_sha256 = identity.get("sha256")
    payload = {key: value for key, value in identity.items() if key != "sha256"}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    quantization = identity.get("quantization_preflight")
    validation = identity.get("validation_inventory")
    if (
        identity.get("schema") != QUALITY_GPU_SCHEMA
        or identity.get("physical_role") != "coordinator-gpu0"
        or not isinstance(claimed_sha256, str)
        or claimed_sha256 != hashlib.sha256(encoded).hexdigest()
        or not isinstance(quantization, dict)
        or not isinstance(validation, dict)
        or quantization.get("index") != 0
    ):
        raise ValueError("GPTQModel quality report has an invalid GPU0 identity")
    uuid = quantization.get("uuid")
    compute = quantization.get("compute_capability")
    if (
        not isinstance(uuid, str)
        or GPU_UUID_RE.fullmatch(uuid) is None
        or identity.get("cuda_visible_devices") != uuid
        or validation.get("uuid") != uuid
        or not isinstance(quantization.get("name"), str)
        or not quantization["name"]
        or validation.get("name") != quantization.get("name")
        or not isinstance(compute, list)
        or len(compute) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in compute
        )
        or validation.get("compute_capability") != f"{compute[0]}.{compute[1]}"
        or not isinstance(quantization.get("driver_version"), str)
        or not quantization["driver_version"]
        or not isinstance(validation.get("driver_version"), str)
        or not validation["driver_version"]
        or not isinstance(validation.get("pci_bus_id"), str)
        or PCI_BUS_ID_RE.fullmatch(validation["pci_bus_id"]) is None
        or not isinstance(validation.get("compute_capability"), str)
        or COMPUTE_CAPABILITY_RE.fullmatch(validation["compute_capability"]) is None
    ):
        raise ValueError(
            "GPTQModel quality report physical GPU differs from planned GPU0"
        )
    return claimed_sha256


def validate_quality_report(
    quality: dict[str, Any],
    calibration: dict[str, Any] | None,
    config: dict[str, Any],
    snapshot: Path,
) -> None:
    if quality.get("schema") != QUALITY_REPORT_SCHEMA:
        raise ValueError("EXL3 quality report has the wrong schema")

    thresholds = quality.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("EXL3 quality report has no threshold contract")
    threshold_values = {
        "min_quant_cosine": finite_number(
            thresholds.get("min_quant_cosine"), "minimum quantized cosine"
        ),
        "max_quant_relative_l2": finite_number(
            thresholds.get("max_quant_relative_l2"), "maximum quantized relative L2"
        ),
        "min_tp_cosine": finite_number(
            thresholds.get("min_tp_cosine"), "minimum TP4 cosine"
        ),
        "max_tp_relative_l2": finite_number(
            thresholds.get("max_tp_relative_l2"), "maximum TP4 relative L2"
        ),
    }
    has_dspark_thresholds = (
        "min_dspark_quant_cosine" in thresholds
        or "max_dspark_quant_relative_l2" in thresholds
    )
    if has_dspark_thresholds:
        threshold_values.update(
            {
                "min_dspark_quant_cosine": finite_number(
                    thresholds.get("min_dspark_quant_cosine"),
                    "minimum dSpark quantized cosine",
                ),
                "max_dspark_quant_relative_l2": finite_number(
                    thresholds.get("max_dspark_quant_relative_l2"),
                    "maximum dSpark quantized relative L2",
                ),
            }
        )
    if (
        threshold_values["min_quant_cosine"] < MIN_QUANT_COSINE
        or threshold_values["max_quant_relative_l2"] > MAX_QUANT_RELATIVE_L2
        or threshold_values["min_tp_cosine"] < MIN_TP_COSINE
        or threshold_values["max_tp_relative_l2"] > MAX_TP_RELATIVE_L2
        or (
            has_dspark_thresholds
            and (
                threshold_values["min_dspark_quant_cosine"] < MIN_DSPARK_QUANT_COSINE
                or threshold_values["max_dspark_quant_relative_l2"]
                > MAX_DSPARK_QUANT_RELATIVE_L2
            )
        )
    ):
        raise ValueError("EXL3 quality report uses weaker than production thresholds")

    calibration_layers = calibration.get("layers") if calibration is not None else None
    quality_layers = quality.get("layers")
    if calibration is not None and (
        not isinstance(calibration_layers, list) or not calibration_layers
    ):
        raise ValueError("EXL3 calibration report has no completed layers")
    if not isinstance(quality_layers, list) or not quality_layers:
        raise ValueError("EXL3 quality report has no layer results")
    try:
        calibration_layer_ids = (
            [layer["layer_id"] for layer in calibration_layers]
            if calibration_layers is not None
            else []
        )
        quality_layer_ids = [layer["layer_id"] for layer in quality_layers]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "EXL3 quality or calibration report has malformed layer IDs"
        ) from error
    if any(
        isinstance(layer_id, bool) or not isinstance(layer_id, int)
        for layer_id in calibration_layer_ids + quality_layer_ids
    ):
        raise ValueError("EXL3 quality or calibration report has malformed layer IDs")
    if len(set(calibration_layer_ids)) != len(calibration_layer_ids):
        raise ValueError("EXL3 calibration report has duplicate layer IDs")
    if len(set(quality_layer_ids)) != len(quality_layer_ids):
        raise ValueError("EXL3 quality report has duplicate layer IDs")
    hidden_layers = config.get("num_hidden_layers")
    dspark_targets = config.get("dspark_target_layer_ids")
    if (
        isinstance(hidden_layers, bool)
        or not isinstance(hidden_layers, int)
        or hidden_layers <= 0
        or not isinstance(dspark_targets, list)
    ):
        raise ValueError("EXL3 config does not declare its target and dSpark blocks")
    expected_layer_ids = list(range(hidden_layers + len(dspark_targets)))
    if calibration is not None and sorted(calibration_layer_ids) != expected_layer_ids:
        raise ValueError(
            "EXL3 calibration report does not cover every target and dSpark block"
        )
    if sorted(quality_layer_ids) != expected_layer_ids:
        raise ValueError(
            "EXL3 quality report does not cover every target and dSpark block"
        )

    expert_ids = quality.get("expert_ids")
    expert_count = config.get("n_routed_experts")
    if (
        not isinstance(expert_ids, list)
        or len(expert_ids) != 6
        or any(
            isinstance(expert, bool) or not isinstance(expert, int)
            for expert in expert_ids
        )
        or len(set(expert_ids)) != len(expert_ids)
        or isinstance(expert_count, bool)
        or not isinstance(expert_count, int)
        or any(expert < 0 or expert >= expert_count for expert in expert_ids)
    ):
        raise ValueError("EXL3 quality report does not cover six valid routed experts")
    stratified_experts = [index * (expert_count - 1) // 5 for index in range(6)]
    if expert_ids != stratified_experts:
        raise ValueError(
            "EXL3 quality report experts are not stratified across the checkpoint"
        )

    validate_quality_binding(
        quality,
        snapshot,
        expected_layer_ids,
        expert_ids,
        thresholds,
    )
    gptqmodel_native = is_gptqmodel_native_exl3(config.get("quantization_config"))
    execution_gpu_sha256 = None
    if gptqmodel_native:
        family = validate_gptqmodel_native_exl3(
            config["quantization_config"],
            model_config=config,
        )
        contract = quality["validation_contract"]
        execution_gpu_sha256 = validate_quality_execution_gpu(contract)
        disjointness = contract.get("calibration_disjointness")
        calibration_identity = (
            disjointness.get("calibration_jsonl")
            if isinstance(disjointness, dict)
            else None
        )
        heldout_identity = (
            disjointness.get("heldout_activation_manifest")
            if isinstance(disjointness, dict)
            else None
        )
        contract_heldout = contract.get("heldout_activation_manifest")
        corpus = family["corpus"]
        if (
            not isinstance(calibration_identity, dict)
            or calibration_identity.get("sha256") != corpus["file_sha256"]
            or calibration_identity.get("prompts") != corpus["examples"]
            or not isinstance(heldout_identity, dict)
            or not isinstance(contract_heldout, dict)
            or heldout_identity.get("path") != contract_heldout.get("path")
            or heldout_identity.get("sha256") != contract_heldout.get("sha256")
            or isinstance(heldout_identity.get("prompts"), bool)
            or not isinstance(heldout_identity.get("prompts"), int)
            or heldout_identity["prompts"] <= 0
            or disjointness.get("prompt_sha256_overlap") != 0
        ):
            raise ValueError(
                "GPTQModel quality report does not prove artifact-bound held-out "
                "prompt disjointness"
            )
        if contract.get("sampling_mode") != GPTQMODEL_PRODUCTION_SAMPLING_POLICY:
            raise ValueError(
                "GPTQModel quality report does not use the production target/dSpark "
                "sampling policy"
            )

    for layer in quality_layers:
        if not isinstance(layer, dict):
            raise ValueError("EXL3 quality report contains a malformed layer result")
        quant = layer.get("native_to_exl3")
        tp = layer.get("exl3_tp4_to_unsharded")
        if not isinstance(quant, dict) or not isinstance(tp, dict):
            raise ValueError("EXL3 quality report layer is missing numerical metrics")
        layer_id = layer.get("layer_id")
        quant_cosine_floor = (
            threshold_values["min_dspark_quant_cosine"]
            if has_dspark_thresholds and layer_id >= hidden_layers
            else threshold_values["min_quant_cosine"]
        )
        quant_relative_l2_ceiling = (
            threshold_values["max_dspark_quant_relative_l2"]
            if has_dspark_thresholds and layer_id >= hidden_layers
            else threshold_values["max_quant_relative_l2"]
        )
        if (
            finite_number(quant.get("cosine"), "quantized cosine") < quant_cosine_floor
            or finite_number(quant.get("relative_l2"), "quantized relative L2")
            > quant_relative_l2_ceiling
            or finite_number(tp.get("cosine"), "TP4 cosine")
            < threshold_values["min_tp_cosine"]
            or finite_number(tp.get("relative_l2"), "TP4 relative L2")
            > threshold_values["max_tp_relative_l2"]
        ):
            raise ValueError(
                "EXL3 quality report contains a layer outside its thresholds"
            )
        rank_bytes = layer.get("tp4_rank_source_bytes")
        if (
            layer.get("tp4_equal_rank_source_bytes") is not True
            or not isinstance(rank_bytes, list)
            or len(rank_bytes) != 4
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in rank_bytes
            )
            or len(set(rank_bytes)) != 1
        ):
            raise ValueError(
                "EXL3 quality report does not prove strict equal-residency TP4"
            )
        if (
            gptqmodel_native
            and layer.get("execution_gpu_sha256") != execution_gpu_sha256
        ):
            raise ValueError(
                "GPTQModel quality layer was not validated on physical GPU0"
            )
        if gptqmodel_native:
            layer_id = layer["layer_id"]
            expected_sampling_mode = (
                "natural-stratified"
                if layer_id < hidden_layers
                else "synthetic-stratified"
            )
            evidence = layer.get("heldout_activation_evidence")
            expected_input_mode = (
                "checkpoint_bound_captured_natural_selected_expert_isolated"
                if layer_id < hidden_layers
                else "uncaptured_mtp_rms_isotropic"
            )
            if (
                layer.get("sampling_policy") != GPTQMODEL_PRODUCTION_SAMPLING_POLICY
                or layer.get("sampling_mode") != expected_sampling_mode
                or not isinstance(evidence, dict)
                or evidence.get("input_mode") != expected_input_mode
                or (
                    layer_id < hidden_layers
                    and evidence.get("route_source") != "native_capture_sidecar"
                )
            ):
                raise ValueError(
                    "GPTQModel quality layer does not prove its target/dSpark "
                    "sampling mode"
                )
            if layer_id < hidden_layers:
                sampled = layer.get("natural_selected_experts")
                if (
                    not isinstance(sampled, list)
                    or [record.get("expert_id") for record in sampled] != expert_ids
                    or any(
                        record.get("rows") != quality.get("rows") for record in sampled
                    )
                ):
                    raise ValueError(
                        "GPTQModel target quality layer lacks natural per-expert rows"
                    )

    summary = quality.get("summary")
    if not isinstance(summary, dict) or summary.get("layers") != len(quality_layers):
        raise ValueError("EXL3 quality report summary does not match its layer results")
    if summary.get("all_tp4_ranks_equal_source_bytes") is not True:
        raise ValueError(
            "EXL3 quality report summary does not prove strict TP4 residency"
        )


def validate_development_quality_report(
    quality: dict[str, Any],
    config: dict[str, Any],
    snapshot: Path,
) -> None:
    """Validate diagnostic evidence without interpreting it as a quality pass."""

    if quality.get("schema") != QUALITY_REPORT_SCHEMA:
        raise ValueError("EXL3 diagnostic quality report has the wrong schema")
    if quality.get("exl3_snapshot") != str(snapshot):
        raise ValueError(
            "EXL3 diagnostic quality report is not bound to this publication"
        )
    layers = quality.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("EXL3 diagnostic quality report has no layer results")
    try:
        layer_ids = [layer["layer_id"] for layer in layers]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "EXL3 diagnostic quality report has malformed layer IDs"
        ) from error
    hidden_layers = config.get("num_hidden_layers")
    dspark_targets = config.get("dspark_target_layer_ids")
    if (
        isinstance(hidden_layers, bool)
        or not isinstance(hidden_layers, int)
        or hidden_layers <= 0
        or not isinstance(dspark_targets, list)
        or any(
            isinstance(layer_id, bool) or not isinstance(layer_id, int)
            for layer_id in layer_ids
        )
        or len(set(layer_ids)) != len(layer_ids)
        or any(
            layer_id < 0 or layer_id >= hidden_layers + len(dspark_targets)
            for layer_id in layer_ids
        )
    ):
        raise ValueError("EXL3 diagnostic quality report has invalid layer coverage")

    contract = quality.get("validation_contract")
    if not isinstance(contract, dict):
        raise ValueError("EXL3 diagnostic quality report has no validation contract")
    claimed_sha256 = contract.get("sha256")
    contract_body = {key: value for key, value in contract.items() if key != "sha256"}
    encoded = json.dumps(
        contract_body,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if claimed_sha256 != hashlib.sha256(encoded).hexdigest():
        raise ValueError("EXL3 diagnostic quality contract digest is invalid")
    if (
        contract.get("allow_incomplete") is not False
        or contract.get("layer_ids") != layer_ids
        or contract.get("exl3") != quality_snapshot_identity(snapshot)
    ):
        raise ValueError(
            "EXL3 diagnostic quality report is stale or belongs to another publication"
        )
    execution_gpu_sha256 = validate_quality_execution_gpu(contract)
    for layer in layers:
        quant = layer.get("native_to_exl3") if isinstance(layer, dict) else None
        tp = layer.get("exl3_tp4_to_unsharded") if isinstance(layer, dict) else None
        rank_bytes = (
            layer.get("tp4_rank_source_bytes") if isinstance(layer, dict) else None
        )
        if (
            not isinstance(quant, dict)
            or not isinstance(tp, dict)
            or layer.get("execution_gpu_sha256") != execution_gpu_sha256
            or layer.get("tp4_equal_rank_source_bytes") is not True
            or not isinstance(rank_bytes, list)
            or len(rank_bytes) != 4
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in rank_bytes
            )
            or len(set(rank_bytes)) != 1
        ):
            raise ValueError(
                "EXL3 diagnostic quality report lacks GPU0/strict-TP4 evidence"
            )
        for metrics in (quant, tp):
            finite_number(metrics.get("cosine"), "diagnostic cosine")
            finite_number(metrics.get("relative_l2"), "diagnostic relative L2")
    summary = quality.get("summary")
    if (
        not isinstance(summary, dict)
        or summary.get("layers") != len(layers)
        or summary.get("all_tp4_ranks_equal_source_bytes") is not True
    ):
        raise ValueError("EXL3 diagnostic quality summary is invalid")


def gptqmodel_publication_identity(snapshot: Path) -> dict[str, Any]:
    snapshot = snapshot.expanduser().resolve(strict=True)
    names = (
        "config.json",
        "model.safetensors.index.json",
        "quantize_config.json",
        GPTQMODEL_PLAN_FILE,
        GPTQMODEL_RUN_FILE,
        GPTQMODEL_ARTIFACT_FILE,
        GPTQMODEL_LEDGER_MANIFEST_FILE,
    )
    return {
        "path": str(snapshot),
        "metadata_sha256": {name: hash_file(snapshot / name) for name in names},
    }


def validate_gptqmodel_retained_native_report(
    report: dict[str, Any],
    config: dict[str, Any],
    snapshot: Path,
) -> None:
    hidden_layers = config.get("num_hidden_layers")
    dspark_targets = config.get("dspark_target_layer_ids")
    expert_count = config.get("n_routed_experts")
    if (
        isinstance(hidden_layers, bool)
        or not isinstance(hidden_layers, int)
        or hidden_layers <= 0
        or not isinstance(dspark_targets, list)
        or isinstance(expert_count, bool)
        or not isinstance(expert_count, int)
        or expert_count <= 0
    ):
        raise ValueError("GPTQModel config has invalid retained-native geometry")
    blocks = hidden_layers + len(dspark_targets)
    if (
        report.get("schema") != "ds41rt-exl3-retained-native-integrity-v1"
        or report.get("recipe")
        != (
            GPTQMODEL_RECIPE
            if int(config["quantization_config"]["bits"]) == 2
            else GPTQMODEL_RECIPE_K3
        )
        or report.get("quantization_scope") != "routed_experts_only"
        or report.get("exl3_snapshot") != str(snapshot.resolve(strict=True))
        or report.get("gptqmodel_publication")
        != gptqmodel_publication_identity(snapshot)
    ):
        raise ValueError(
            "GPTQModel retained-native report is stale or has the wrong contract"
        )
    tensors = report.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise ValueError("GPTQModel retained-native report has no tensor records")
    names = [
        record.get("name") if isinstance(record, dict) else None for record in tensors
    ]
    bytes_by_tensor = [
        record.get("bytes") if isinstance(record, dict) else None for record in tensors
    ]
    digest_pattern = re.compile(r"[0-9a-f]{64}\Z")
    if (
        any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            for size in bytes_by_tensor
        )
        or any(
            not isinstance(record, dict)
            or digest_pattern.fullmatch(str(record.get("sha256", ""))) is None
            for record in tensors
        )
        or digest_pattern.fullmatch(str(report.get("aggregate_sha256", ""))) is None
        or report.get("retained_tensor_count") != len(tensors)
        or report.get("retained_bytes") != sum(bytes_by_tensor)
    ):
        raise ValueError("GPTQModel retained-native tensor accounting is invalid")

    expected_generated_count = blocks * expert_count * 12
    expected_marker_count = blocks * expert_count * 3
    generated_bytes = report.get("generated_exl3_bytes")
    if (
        report.get("generated_exl3_metadata_verified") is not True
        or report.get("generated_exl3_tensor_count") != expected_generated_count
        or report.get("generated_exl3_mcg_markers_verified") is not True
        or report.get("generated_exl3_mcg_tensor_count") != expected_marker_count
        or isinstance(generated_bytes, bool)
        or not isinstance(generated_bytes, int)
        or generated_bytes <= 0
        or report.get("artifact_tensor_count")
        != report["retained_tensor_count"] + expected_generated_count
        or report.get("artifact_bytes") != report["retained_bytes"] + generated_bytes
    ):
        raise ValueError("GPTQModel retained-native generated tensor proof is invalid")

    layout = report.get("strict_tp4_source_layout")
    per_expert = (
        layout.get("rank_source_bytes_per_expert") if isinstance(layout, dict) else None
    )
    per_block = (
        layout.get("rank_source_bytes_per_block") if isinstance(layout, dict) else None
    )
    total = layout.get("rank_source_bytes_total") if isinstance(layout, dict) else None
    common_layout_invalid = (
        not isinstance(layout, dict)
        or layout.get("world_size") != 4
        or layout.get("blocks") != blocks
        or layout.get("experts_per_block") != expert_count
        or layout.get("experts_checked") != blocks * expert_count
        or layout.get("equal_rank_source_bytes") is not True
    )
    if isinstance(layout, dict) and layout.get("mixed_projection_tiers") is True:
        by_block = layout.get("rank_source_bytes_by_block")
        tier_counts = layout.get("tier_counts")
        summary = layout.get("rank_source_bytes_per_expert_summary")
        mixed_invalid = (
            not isinstance(by_block, list)
            or len(by_block) != blocks
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in by_block
            )
            or not isinstance(total, list)
            or len(total) != 4
            or len(set(total)) != 1
            or total[0] != sum(by_block)
            or not isinstance(tier_counts, dict)
            or set(tier_counts) != {"2", "3"}
            or any(
                isinstance(count, bool) or not isinstance(count, int) or count <= 0
                for count in tier_counts.values()
            )
            or sum(tier_counts.values()) != blocks * expert_count * 3
            or not isinstance(summary, dict)
            or any(
                isinstance(summary.get(field), bool)
                or not isinstance(summary.get(field), (int, float))
                or not math.isfinite(summary[field])
                or summary[field] <= 0
                for field in ("min", "mean", "max")
            )
            or not summary["min"] <= summary["mean"] <= summary["max"]
            or digest_pattern.fullmatch(
                str(layout.get("rank_source_bytes_by_expert_sha256", ""))
            )
            is None
        )
        layout_invalid = common_layout_invalid or mixed_invalid
    else:
        uniform_invalid = (
            not isinstance(per_expert, list)
            or len(per_expert) != 4
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in per_expert
            )
            or len(set(per_expert)) != 1
            or per_block != [size * expert_count for size in per_expert]
            or total != [size * blocks for size in per_block]
        )
        layout_invalid = common_layout_invalid or uniform_invalid
    if layout_invalid:
        raise ValueError(
            "GPTQModel retained-native report does not prove strict TP4 residency"
        )


def read_external_qualification_report(
    path: Path, snapshot: Path, label: str
) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if resolved == snapshot or snapshot in resolved.parents:
        raise ValueError(
            f"{label} must remain outside the immutable GPTQModel artifact"
        )
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{label} must be a regular file: {resolved}")
    return read_json_object(resolved)


def qualification_report_record(
    path: Path,
    name: str,
    report: dict[str, Any],
) -> QualificationReport:
    resolved = path.expanduser().resolve(strict=True)
    payload = resolved.read_bytes()
    try:
        observed = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"qualification report changed while reading: {resolved}"
        ) from error
    if observed != report:
        raise ValueError(f"qualification report changed while reading: {resolved}")
    contract = report.get("validation_contract")
    contract_sha256 = contract.get("sha256") if isinstance(contract, dict) else None
    return QualificationReport(
        source=resolved,
        name=name,
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        schema=str(report.get("schema", "")),
        contract_sha256=(str(contract_sha256) if contract_sha256 is not None else None),
    )


def validate_multigpu_qualification(
    calibration: dict[str, Any], config: dict[str, Any], snapshot: Path
) -> None:
    embedded = calibration.get("multigpu_qualification")
    if embedded is None:
        return
    if not isinstance(embedded, dict):
        raise ValueError("EXL3 calibration has a malformed multi-GPU qualification")
    devices = embedded.get("devices")
    if not isinstance(devices, list) or any(
        isinstance(device, bool) or not isinstance(device, int) or device < 0
        for device in devices
    ):
        raise ValueError("EXL3 calibration has malformed multi-GPU device ordinals")
    if len(devices) <= 1:
        return
    if devices[0] != 0 or len(set(devices)) != len(devices):
        raise ValueError(
            "EXL3 multi-GPU qualification devices must be unique and start at zero"
        )

    report_path = snapshot / MULTIGPU_QUALIFICATION_FILE
    if not report_path.is_file() or report_path.is_symlink():
        raise ValueError(
            f"multi-GPU EXL3 artifact is missing full-shape qualification {report_path}"
        )
    report = read_json_object(report_path)
    hidden = config.get("hidden_size")
    intermediate = config.get("moe_intermediate_size")
    if (
        isinstance(hidden, bool)
        or not isinstance(hidden, int)
        or hidden <= 0
        or isinstance(intermediate, bool)
        or not isinstance(intermediate, int)
        or intermediate <= 0
    ):
        raise ValueError("EXL3 config lacks production projection dimensions")
    expected_shapes = [[hidden, intermediate], [intermediate, hidden]]
    expected_seed = calibration.get("seed")
    if (
        report.get("schema") != "ds41rt-exl3-multigpu-qualification-v1"
        or report.get("status") != "bit-exact"
        or report.get("devices") != devices
        or report.get("device_ratios") != embedded.get("device_ratios")
        or report.get("seed") != expected_seed
        or report.get("batches") != 4
        or report.get("tensors_per_batch") != 4
        or report.get("shape") != [128, 256]
        or report.get("production_projection_shapes") != expected_shapes
    ):
        raise ValueError(
            "multi-GPU EXL3 full-shape qualification does not match the artifact contract"
        )


def is_temporary_name(name: str) -> bool:
    return name.endswith(
        (".incomplete", ".tmp", ".temp", ".part", "~")
    ) or name.startswith((".tmp-", ".temp-", ".#"))


def discover_artifact_files(snapshot: Path) -> list[tuple[Path, PurePosixPath]]:
    files: list[tuple[Path, PurePosixPath]] = []
    for root, directories, names in os.walk(snapshot, followlinks=False):
        root_path = Path(root)
        kept_directories = []
        for name in sorted(directories):
            path = root_path / name
            relative = path.relative_to(snapshot)
            if relative.parts[0] == DEBUG_DIRECTORY:
                continue
            if path.is_symlink():
                raise ValueError(f"artifact directory must not be a symlink: {path}")
            if is_temporary_name(name):
                raise ValueError(f"artifact contains temporary directory: {path}")
            kept_directories.append(name)
        directories[:] = kept_directories

        for name in sorted(names):
            path = root_path / name
            relative_path = path.relative_to(snapshot)
            if relative_path.parts[0] == DEBUG_DIRECTORY:
                continue
            if path.is_symlink():
                raise ValueError(f"artifact file must not be a symlink: {path}")
            if name == ".ds41rt-exl3-state.json" or is_temporary_name(name):
                raise ValueError(
                    f"artifact contains incomplete or temporary file: {path}"
                )
            mode = path.lstat().st_mode
            if not stat.S_ISREG(mode):
                raise ValueError(f"artifact entry must be a regular file: {path}")
            files.append((path, PurePosixPath(*relative_path.parts)))
    return sorted(files, key=lambda item: item[1].as_posix())


def validate_indexed_artifact_files(
    snapshot: Path,
) -> list[tuple[Path, PurePosixPath]]:
    files = discover_artifact_files(snapshot)
    relative_files = {relative for _, relative in files}
    index = read_json_object(snapshot / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model.safetensors.index.json has no non-empty weight_map")
    referenced_shards = {validate_index_path(value) for value in weight_map.values()}
    missing = referenced_shards - relative_files
    if missing:
        rendered = ", ".join(sorted(path.as_posix() for path in missing))
        raise ValueError(f"model index references missing shards: {rendered}")
    artifact_shards = {
        relative
        for relative in relative_files
        if relative.name.endswith(".safetensors")
    }
    stale = artifact_shards - referenced_shards
    if stale:
        rendered = ", ".join(sorted(path.as_posix() for path in stale))
        raise ValueError(
            f"artifact contains unreferenced safetensors shards: {rendered}"
        )
    return files


def validate_complete_artifact(snapshot: Path) -> list[tuple[Path, PurePosixPath]]:
    config_path = snapshot / "config.json"
    if not config_path.is_file() or config_path.is_symlink():
        raise ValueError(
            f"complete EXL3 artifact is missing regular file {config_path}"
        )
    config = read_json_object(config_path)
    quant = config.get("quantization_config")
    if is_gptqmodel_native_exl3(quant):
        for name in GPTQMODEL_REQUIRED_FILES:
            path = snapshot / name
            if not path.is_file() or path.is_symlink():
                raise ValueError(
                    f"complete GPTQModel EXL3 artifact is missing regular file {path}"
                )
        recipe = validate_quantization_contract(
            quant,
            "config.json",
            model_config=config,
        )
        if recipe not in {EXL3_V4_RECIPE, EXL3_K3_V4_RECIPE}:
            raise ValueError("GPTQModel EXL3 artifact is not the production v4 recipe")
        qconfig = read_json_object(snapshot / "quantize_config.json")
        if qconfig != quant:
            raise ValueError(
                "config.json and quantize_config.json contain different EXL3 contracts"
            )
        validate_gptqmodel_publication(
            snapshot,
            config,
            verify_all_hashes=False,
            require_canonical=True,
        )
        family_join = validate_gptqmodel_native_exl3(
            quant,
            model_config=config,
        )
        ledger_meta = quant.get("meta", {}).get("ds41rt_error_ledger", {})
        projection_sources = (
            ledger_meta.get("projection_sources")
            if isinstance(ledger_meta, dict)
            else None
        )
        namespace_family_joins = (
            {
                namespace: source["family_join"]
                for namespace, source in projection_sources.items()
            }
            if isinstance(projection_sources, dict)
            and set(projection_sources) == {"base", "mtp"}
            and all(
                isinstance(source, dict) and isinstance(source.get("family_join"), dict)
                for source in projection_sources.values()
            )
            else None
        )
        validate_error_ledger(
            snapshot,
            config,
            expected_family_join=family_join,
            namespace_family_joins=namespace_family_joins,
        )
        return validate_indexed_artifact_files(snapshot)

    for name in REQUIRED_FILES:
        path = snapshot / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"complete EXL3 artifact is missing regular file {path}")

    recipe = validate_quantization_contract(
        config.get("quantization_config"), "config.json", model_config=config
    )
    qconfig = read_json_object(snapshot / "quantization_config.json")
    if validate_quantization_contract(qconfig, "quantization_config.json") != recipe:
        raise ValueError(
            "config and quantization_config declare different DS41RT recipes"
        )
    if recipe == EXL3_V4_RECIPE:
        validate_error_ledger(snapshot, config)
    calibration = read_json_object(snapshot / "ds41rt-exl3-calibration.json")
    if calibration.get("incomplete") is True:
        raise ValueError("calibration report still declares the artifact incomplete")
    if calibration.get("recipe") != recipe:
        raise ValueError("calibration report does not match the DS41RT recipe")
    validate_multigpu_qualification(calibration, config, snapshot)
    quality = read_json_object(snapshot / "ds41rt-exl3-quality.json")
    validate_quality_report(quality, calibration, config, snapshot)
    hidden_layers = config["num_hidden_layers"]
    expected_layer_ids = list(
        range(hidden_layers + len(config["dspark_target_layer_ids"]))
    )
    expert_count = config["n_routed_experts"]
    quality_layers = quality["layers"]
    expert_ids = quality["expert_ids"]
    integrity = read_json_object(snapshot / "ds41rt-exl3-retained-native.json")
    if (
        integrity.get("schema") != "ds41rt-exl3-retained-native-integrity-v1"
        or integrity.get("recipe") != recipe
        or integrity.get("quantization_scope") != "routed_experts_only"
    ):
        raise ValueError("retained-native integrity report has the wrong contract")
    tensors = integrity.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise ValueError("retained-native integrity report has no tensor hashes")
    if any(
        not isinstance(record, dict)
        or not isinstance(record.get("bytes"), int)
        or record["bytes"] < 0
        for record in tensors
    ):
        raise ValueError("retained-native integrity tensor records are invalid")
    if integrity.get("retained_tensor_count") != len(tensors):
        raise ValueError(
            "retained-native integrity tensor count does not match its records"
        )
    if integrity.get("retained_bytes") != sum(record["bytes"] for record in tensors):
        raise ValueError(
            "retained-native integrity byte count does not match its records"
        )
    generated_count = integrity.get("generated_exl3_tensor_count")
    generated_bytes = integrity.get("generated_exl3_bytes")
    expected_generated_count = len(expected_layer_ids) * expert_count * 12
    if (
        integrity.get("generated_exl3_metadata_verified") is not True
        or isinstance(generated_count, bool)
        or not isinstance(generated_count, int)
        or generated_count != expected_generated_count
        or isinstance(generated_bytes, bool)
        or not isinstance(generated_bytes, int)
        or generated_bytes <= 0
    ):
        raise ValueError(
            "generated EXL3 tensor metadata does not cover the full artifact"
        )
    expected_marker_count = len(expected_layer_ids) * expert_count * 3
    if (
        integrity.get("generated_exl3_mcg_markers_verified") is not True
        or integrity.get("generated_exl3_mcg_tensor_count") != expected_marker_count
    ):
        raise ValueError(
            "generated EXL3 MCG markers are not verified for every routed expert"
        )
    if (
        integrity.get("artifact_tensor_count")
        != integrity["retained_tensor_count"] + generated_count
        or integrity.get("artifact_bytes")
        != integrity["retained_bytes"] + generated_bytes
    ):
        raise ValueError(
            "hybrid EXL3 artifact accounting does not match its tensor families"
        )

    tp4_layout = integrity.get("strict_tp4_source_layout")
    if not isinstance(tp4_layout, dict):
        raise ValueError("hybrid EXL3 artifact has no full strict-TP4 layout proof")
    per_expert_rank_bytes = tp4_layout.get("rank_source_bytes_per_expert")
    per_block_rank_bytes = tp4_layout.get("rank_source_bytes_per_block")
    total_rank_bytes = tp4_layout.get("rank_source_bytes_total")
    if (
        tp4_layout.get("world_size") != 4
        or tp4_layout.get("blocks") != len(expected_layer_ids)
        or tp4_layout.get("experts_per_block") != expert_count
        or tp4_layout.get("experts_checked") != len(expected_layer_ids) * expert_count
        or tp4_layout.get("equal_rank_source_bytes") is not True
        or not isinstance(per_expert_rank_bytes, list)
        or len(per_expert_rank_bytes) != 4
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in per_expert_rank_bytes
        )
        or len(set(per_expert_rank_bytes)) != 1
        or not isinstance(per_block_rank_bytes, list)
        or per_block_rank_bytes
        != [size * expert_count for size in per_expert_rank_bytes]
        or not isinstance(total_rank_bytes, list)
        or total_rank_bytes
        != [size * len(expected_layer_ids) for size in per_block_rank_bytes]
    ):
        raise ValueError("hybrid EXL3 artifact does not prove full equal-residency TP4")
    for layer in quality_layers:
        sampled_rank_bytes = layer["tp4_rank_source_bytes"]
        if any(
            sampled * expert_count != complete * len(expert_ids)
            for sampled, complete in zip(sampled_rank_bytes, per_block_rank_bytes)
        ):
            raise ValueError(
                "sampled TP4 source bytes do not match the full artifact layout proof"
            )
    names = [record.get("name") for record in tensors]
    if any(not isinstance(name, str) or not name for name in names) or len(
        set(names)
    ) != len(names):
        raise ValueError(
            "retained-native integrity tensor names are invalid or duplicated"
        )
    digest_pattern = re.compile(r"[0-9a-f]{64}\Z")
    if digest_pattern.fullmatch(
        str(integrity.get("aggregate_sha256", ""))
    ) is None or any(
        digest_pattern.fullmatch(str(record.get("sha256", ""))) is None
        for record in tensors
    ):
        raise ValueError("retained-native integrity report contains an invalid SHA-256")

    return validate_indexed_artifact_files(snapshot)


def hash_artifact_file(source: Path, relative: PurePosixPath) -> ArtifactFile:
    before = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    after = source.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise ValueError(f"artifact file changed while it was hashed: {source}")
    return ArtifactFile(source, relative, after.st_size, digest.hexdigest())


def canonical_manifest(
    files: Iterable[ArtifactFile],
) -> tuple[bytes, list[dict[str, Any]]]:
    entries = [file.manifest_entry() for file in files]
    payload = {"schema": MANIFEST_SCHEMA, "files": entries}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return encoded, entries


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def install_blob(file: ArtifactFile, blob: Path, link_mode: str) -> None:
    if blob.exists():
        if not blob.is_file() or blob.is_symlink():
            raise ValueError(f"staged blob is not a regular file: {blob}")
        blob_stat = blob.stat()
        source_stat = file.source.stat()
        same_inode = (blob_stat.st_dev, blob_stat.st_ino) == (
            source_stat.st_dev,
            source_stat.st_ino,
        )
        if blob_stat.st_size != file.size or (
            not same_inode and hash_file(blob) != file.sha256
        ):
            raise ValueError(f"staged content-addressed blob is corrupt: {blob}")
        return

    temporary = blob.with_name(f".{blob.name}.{os.getpid()}.tmp")
    try:
        if link_mode == "hardlink":
            try:
                os.link(file.source, temporary)
            except OSError as error:
                raise ValueError(
                    f"cannot hardlink {file.source} into {blob.parent}; "
                    "use --link-mode copy for different filesystems"
                ) from error
        else:
            shutil.copyfile(file.source, temporary)
            if (
                temporary.stat().st_size != file.size
                or hash_file(temporary) != file.sha256
            ):
                raise ValueError(
                    f"copied blob failed content verification: {temporary}"
                )
        try:
            os.link(temporary, blob)
        except FileExistsError:
            if blob.stat().st_size != file.size or hash_file(blob) != file.sha256:
                raise ValueError(f"concurrently staged blob is corrupt: {blob}")
    finally:
        temporary.unlink(missing_ok=True)


def expected_link_target(
    snapshot_path: Path, relative: PurePosixPath, blob: Path
) -> str:
    link = snapshot_path.joinpath(*relative.parts)
    return os.path.relpath(blob, link.parent)


def validate_existing_snapshot(
    snapshot_path: Path, files: Iterable[ArtifactFile], blobs: Path
) -> None:
    expected = {file.relative: file for file in files}
    actual_files: set[PurePosixPath] = set()
    actual_directories: set[PurePosixPath] = set()
    for root, directories, names in os.walk(snapshot_path, followlinks=False):
        root_path = Path(root)
        for name in directories:
            path = root_path / name
            if path.is_symlink():
                raise ValueError(
                    f"staged snapshot contains a symlinked directory: {path}"
                )
            relative = path.relative_to(snapshot_path)
            actual_directories.add(PurePosixPath(*relative.parts))
        for name in names:
            path = root_path / name
            relative = path.relative_to(snapshot_path)
            actual_files.add(PurePosixPath(*relative.parts))

    if actual_files != set(expected):
        raise ValueError(
            f"existing staged snapshot has unexpected contents: {snapshot_path}"
        )
    expected_directories = {
        PurePosixPath(*relative.parts[:depth])
        for relative in expected
        for depth in range(1, len(relative.parts))
    }
    if actual_directories != expected_directories:
        raise ValueError(
            f"existing staged snapshot has unexpected directories: {snapshot_path}"
        )

    for relative, file in expected.items():
        link = snapshot_path.joinpath(*relative.parts)
        blob = blobs / file.sha256
        expected_target = expected_link_target(snapshot_path, relative, blob)
        if not link.is_symlink() or os.readlink(link) != expected_target:
            raise ValueError(f"existing staged snapshot has incorrect link: {link}")
        if not link.exists() or link.resolve() != blob.resolve():
            raise ValueError(f"existing staged snapshot has unresolved link: {link}")


def publish_snapshot(
    snapshot_path: Path, files: Iterable[ArtifactFile], blobs: Path, model_root: Path
) -> None:
    temporary = model_root / f".ds41rt-stage-{snapshot_path.name}-{os.getpid()}"
    if temporary.exists():
        raise ValueError(f"temporary staging path already exists: {temporary}")
    temporary.mkdir()
    try:
        for file in files:
            link = temporary.joinpath(*file.relative.parts)
            link.parent.mkdir(parents=True, exist_ok=True)
            blob = blobs / file.sha256
            # Compute the relative target from the published location. The
            # temporary tree sits one level higher until its atomic rename.
            link.symlink_to(expected_link_target(snapshot_path, file.relative, blob))
        try:
            temporary.rename(snapshot_path)
        except OSError:
            if not snapshot_path.exists():
                raise
            validate_existing_snapshot(snapshot_path, files, blobs)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def read_ref(ref_path: Path) -> str | None:
    if not ref_path.exists():
        return None
    if not ref_path.is_file() or ref_path.is_symlink():
        raise ValueError(f"refs/main is not a regular file: {ref_path}")
    revision = ref_path.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", revision):
        raise ValueError(f"refs/main contains an invalid staged revision: {revision!r}")
    return revision


def write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def install_qualification_report(
    report: QualificationReport, destination: Path
) -> None:
    payload = report.source.read_bytes()
    if (
        len(payload) != report.size
        or hashlib.sha256(payload).hexdigest() != report.sha256
    ):
        raise ValueError(f"qualification report changed while staging: {report.source}")
    if destination.exists():
        if (
            not destination.is_file()
            or destination.is_symlink()
            or destination.read_bytes() != payload
        ):
            raise ValueError(
                f"existing staged qualification report differs: {destination}"
            )
        return
    write_atomic(destination, payload)


def stage_snapshot(
    snapshot: Path,
    model_id: str,
    hf_home: Path,
    *,
    link_mode: str = "hardlink",
    update_ref: bool = False,
    retained_native_report: Path | None = None,
    quality_report: Path | None = None,
    development_unqualified: bool = False,
    standard_publication: bool = False,
) -> dict[str, Any]:
    if link_mode not in {"hardlink", "copy"}:
        raise ValueError(f"unsupported link mode {link_mode!r}")
    if snapshot.is_symlink():
        raise ValueError(f"artifact root must not be a symlink: {snapshot}")
    snapshot = snapshot.expanduser().resolve(strict=True)
    if not snapshot.is_dir():
        raise ValueError(f"artifact is not a directory: {snapshot}")
    hf_home = hf_home.expanduser().resolve()
    model_root = model_cache_dir(hf_home, model_id)
    if snapshot == model_root or model_root in snapshot.parents:
        raise ValueError(
            "source artifact must not be inside its destination model cache"
        )

    if standard_publication and (
        retained_native_report is not None
        or quality_report is not None
        or development_unqualified
    ):
        raise ValueError(
            "standard-only publications cannot carry private qualification options"
        )

    if standard_publication:
        discovered = [
            (source, PurePosixPath(name))
            for name, source in publication_sources(
                snapshot,
                snapshot / "README.md",
            )
        ]
        files = [
            hash_artifact_file(source, relative) for source, relative in discovered
        ]
        qualification_reports: tuple[QualificationReport, ...] = ()
        development_reports: tuple[QualificationReport, ...] = ()
        gptqmodel_native = False
    else:
        files = []

    source_config = read_json_object(snapshot / "config.json")
    if not standard_publication:
        gptqmodel_native = is_gptqmodel_native_exl3(
            source_config.get("quantization_config")
        )
        qualification_reports = ()
        development_reports = ()
    if not standard_publication and gptqmodel_native:
        if retained_native_report is None or quality_report is None:
            raise ValueError(
                "GPTQModel staging requires --retained-native-report and "
                "--quality-report"
            )
        retained = read_external_qualification_report(
            retained_native_report,
            snapshot,
            "retained-native report",
        )
        quality = read_external_qualification_report(
            quality_report,
            snapshot,
            "quality report",
        )
        validate_gptqmodel_retained_native_report(retained, source_config, snapshot)
        if development_unqualified:
            validate_development_quality_report(quality, source_config, snapshot)
            development_reports = (
                qualification_report_record(
                    retained_native_report,
                    "retained-native.json",
                    retained,
                ),
                qualification_report_record(
                    quality_report,
                    "diagnostic-quality.json",
                    quality,
                ),
            )
        else:
            validate_quality_report(quality, None, source_config, snapshot)
            qualification_reports = (
                qualification_report_record(
                    retained_native_report,
                    "retained-native.json",
                    retained,
                ),
                qualification_report_record(
                    quality_report,
                    "expert-quality.json",
                    quality,
                ),
            )
    elif not standard_publication and (
        retained_native_report is not None or quality_report is not None
    ):
        raise ValueError(
            "external qualification reports are only valid for immutable "
            "GPTQModel publications"
        )

    if not standard_publication:
        discovered = validate_complete_artifact(snapshot)
        files = [
            hash_artifact_file(source, relative) for source, relative in discovered
        ]
    if not standard_publication and gptqmodel_native:
        publication = read_json_object(snapshot / GPTQMODEL_ARTIFACT_FILE)
        published_files = publication.get("files")
        if not isinstance(published_files, dict):
            raise ValueError("GPTQModel artifact publication has no file records")
        actual = {
            file.relative.as_posix(): {
                "bytes": file.size,
                "sha256": file.sha256,
            }
            for file in files
            if file.relative.as_posix()
            not in {GPTQMODEL_ARTIFACT_FILE, GPTQMODEL_RUN_FILE}
        }
        if actual != published_files:
            raise ValueError(
                "GPTQModel artifact content hashes differ from its publication manifest"
            )
        validate_gptqmodel_retained_native_report(retained, source_config, snapshot)
        if development_unqualified:
            validate_development_quality_report(quality, source_config, snapshot)
        else:
            validate_quality_report(quality, None, source_config, snapshot)
    elif not standard_publication and development_unqualified:
        raise ValueError(
            "--development-unqualified is only valid for immutable GPTQModel publications"
        )
    manifest_bytes, entries = canonical_manifest(files)
    revision = hashlib.sha256(manifest_bytes).hexdigest()
    qualification_entries = [
        report.manifest_entry() for report in qualification_reports
    ]
    development_entries = [report.manifest_entry() for report in development_reports]

    refs = model_root / "refs"
    blobs = model_root / "blobs"
    snapshots = model_root / "snapshots"
    manifests = model_root / "ds41rt-manifests"
    for directory in (refs, blobs, snapshots, manifests):
        directory.mkdir(parents=True, exist_ok=True)

    ref_path = refs / "main"
    previous_revision = read_ref(ref_path)
    if previous_revision not in {None, revision} and not update_ref:
        raise ValueError(
            f"refs/main already selects {previous_revision}; pass --update-ref to select {revision}"
        )

    for file in files:
        install_blob(file, blobs / file.sha256, link_mode)

    snapshot_path = snapshots / revision
    if snapshot_path.exists():
        if not snapshot_path.is_dir() or snapshot_path.is_symlink():
            raise ValueError(
                f"staged snapshot path is not a directory: {snapshot_path}"
            )
        validate_existing_snapshot(snapshot_path, files, blobs)
    else:
        publish_snapshot(snapshot_path, files, blobs, model_root)

    if qualification_reports:
        qualification_root = model_root / "ds41rt-qualifications" / revision
        qualification_root.mkdir(parents=True, exist_ok=True)
        for report in qualification_reports:
            install_qualification_report(report, qualification_root / report.name)
    if development_reports:
        evidence_root = model_root / "ds41rt-development-evidence" / revision
        evidence_root.mkdir(parents=True, exist_ok=True)
        for report in development_reports:
            install_qualification_report(report, evidence_root / report.name)

    metadata = {
        "schema": MANIFEST_SCHEMA,
        "model_id": model_id,
        "revision": revision,
        "source_snapshot": str(snapshot),
        "link_mode": link_mode,
        "files": entries,
    }
    if qualification_entries:
        metadata["qualification"] = qualification_entries
    if development_entries:
        metadata["qualification_status"] = DEVELOPMENT_QUALIFICATION_STATUS
        metadata["development_blockers"] = list(DEVELOPMENT_BLOCKERS)
        metadata["development_evidence"] = development_entries
    metadata_path = manifests / f"{revision}.json"
    if metadata_path.exists():
        existing = read_json_object(metadata_path)
        if (
            existing.get("schema") != MANIFEST_SCHEMA
            or existing.get("model_id") != model_id
            or existing.get("revision") != revision
            or existing.get("files") != entries
            or existing.get("qualification")
            != (qualification_entries if qualification_entries else None)
            or existing.get("qualification_status")
            != metadata.get("qualification_status")
            or existing.get("development_blockers")
            != metadata.get("development_blockers")
            or existing.get("development_evidence")
            != metadata.get("development_evidence")
        ):
            raise ValueError(
                f"existing staging manifest does not match: {metadata_path}"
            )
    else:
        write_atomic(
            metadata_path,
            (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode(),
        )

    canonical_ref = revision.encode()
    if previous_revision != revision or ref_path.read_bytes() != canonical_ref:
        write_atomic(ref_path, canonical_ref)

    return {
        "schema": MANIFEST_SCHEMA,
        "model_id": model_id,
        "revision": revision,
        "files": len(files),
        "bytes": sum(file.size for file in files),
        "link_mode": link_mode,
        "snapshot": str(snapshot_path),
        "manifest": str(metadata_path),
        "qualification": qualification_entries,
        "qualification_status": metadata.get(
            "qualification_status", "production-qualified"
        ),
        "development_blockers": metadata.get("development_blockers", []),
        "development_evidence": development_entries,
        "previous_revision": previous_revision,
    }


def main() -> None:
    args = parse_args()
    try:
        result = stage_snapshot(
            args.snapshot,
            args.model_id,
            args.hf_home,
            link_mode=args.link_mode,
            update_ref=args.update_ref,
            retained_native_report=args.retained_native_report,
            quality_report=args.quality_report,
            development_unqualified=args.development_unqualified,
            standard_publication=args.standard_publication,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
