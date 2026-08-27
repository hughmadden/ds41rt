#!/usr/bin/env python3
"""Audit one completed DS4RT EXL3 block directly from live projection checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import time
from typing import Any

import torch

from gptqmodel.utils.exl3_error_ledger import (
    LEDGER_SCHEMA,
    LEDGER_SCHEMA_VERSION,
    ZERO_ROUTE_RECOVERY_AUTHORIZATION_SCHEMA,
    ZERO_ROUTE_RECOVERY_CAPTURE_METHOD,
    ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MAX,
    ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MIN,
    ZERO_ROUTE_RECOVERY_IDENTITY_POLICY,
    ZERO_ROUTE_RECOVERY_SAMPLE_SOURCE,
    ZERO_ROUTE_RECOVERY_SCHEMA,
    ZERO_ROUTE_RECOVERY_SELECTION_POLICY,
    ZERO_ROUTE_RECOVERY_TARGET_SAMPLE_COUNT,
    ZERO_ROUTE_RECOVERY_TRIGGER,
    derive_family_records,
    routed_expert_identity,
    validate_zero_route_recovery,
)
from gptqmodel.utils.exl3_projection_checkpoint import (
    CHECKPOINT_CONTRACT,
    EXL3ProjectionCheckpointStore,
)
from gptqmodel.utils.exl3_inline_mixed import (
    INLINE_MIXED_META_KEY,
    INLINE_MIXED_SCHEMA,
    INLINE_MIXED_SCHEMA_VERSION,
    PROJECTION_ORDER,
    inline_mixed_policy,
    projection_score,
)

import quantize_flash_gptqmodel as launcher


AUDIT_SCHEMA = "ds4rt-exl3-live-projection-block-audit-v1"
TP4_RESIDENCY_SCHEMA = "ds4rt-exl3-runtime-tp4-residency-v1"
TP4_WORLD_SIZE = 4
MCG_MARKER = 0xCBAC1FED
PROJECTION_NAMES = {
    "w1": "gate_proj",
    "w2": "down_proj",
    "w3": "up_proj",
}


class AuditError(RuntimeError):
    """The live projection store failed an artifact-facing invariant."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-state", type=Path, required=True)
    parser.add_argument(
        "--block-namespace",
        choices=("base", "mtp"),
        required=True,
    )
    parser.add_argument("--logical-layer", type=int, required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="report the current subset instead of requiring all expert families",
    )
    parser.add_argument(
        "--include-tp4-residency",
        action="store_true",
        help=(
            "derive the exact four rank-local runtime slabs; requires one "
            "complete block"
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _stable_bytes(path: Path, *, attempts: int = 8) -> bytes:
    """Read an append-only file without mistaking an active append for corruption."""

    for attempt in range(attempts):
        before = _file_identity(path)
        payload = path.read_bytes()
        after = _file_identity(path)
        if before == after:
            return payload
        if attempt + 1 < attempts:
            time.sleep(0.025)
    raise AuditError(f"file did not stabilize while being audited: {path}")


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise AuditError(f"{label} is not one regular file: {path}")
    try:
        value = json.loads(_stable_bytes(path))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise AuditError(f"{label} is not a JSON object: {path}")
    return value


def _validate_bound_record(record: dict[str, Any], *, label: str) -> str:
    digest = record.get("record_sha256")
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    if not isinstance(digest, str) or digest != _sha256(_canonical(body)):
        raise AuditError(f"{label} failed its content digest")
    return digest


def _load_journal(path: Path) -> dict[str, tuple[dict[str, Any], str]]:
    if not path.is_file() or path.is_symlink():
        raise AuditError(f"projection journal is not one regular file: {path}")
    payload = _stable_bytes(path)
    if payload and not payload.endswith(b"\n"):
        raise AuditError("projection journal ends with a partial record")
    records: dict[str, tuple[dict[str, Any], str]] = {}
    for line_number, line in enumerate(payload.splitlines(), 1):
        try:
            bound = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise AuditError(
                f"projection journal line {line_number} is invalid JSON"
            ) from error
        if not isinstance(bound, dict) or bound.get("record_kind") != "projection":
            raise AuditError(
                f"projection journal line {line_number} is not a projection"
            )
        digest = _validate_bound_record(
            bound,
            label=f"projection journal line {line_number}",
        )
        module = bound.get("module")
        if not isinstance(module, str) or not module or module in records:
            raise AuditError(
                "projection journal contains a missing or duplicate module"
            )
        records[module] = (
            {key: value for key, value in bound.items() if key != "record_sha256"},
            digest,
        )
    return records


def _expected_modules(
    *,
    namespace: str,
    logical_layer: int,
    expert_count: int,
) -> dict[str, tuple[int, str]]:
    block = (
        f"model.layers.{logical_layer}"
        if namespace == "base"
        else f"mtp.{logical_layer}"
    )
    return {
        f"{block}.mlp.experts.{expert}.{projection_name}": (expert, projection)
        for expert in range(expert_count)
        for projection, projection_name in PROJECTION_NAMES.items()
    }


def _portable_inline_policy(policy: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in policy.items() if key != "tier_plan_root"}


def _resolve_planned_path(
    path: Path,
    *,
    plan: dict[str, Any],
    run_state: Path,
) -> Path:
    """Resolve a run-state child across a host/container mount alias."""

    planned_run_state = Path(plan["run_state_dir"])
    try:
        relative = path.relative_to(planned_run_state)
    except ValueError:
        return path.resolve()
    return (run_state / relative).resolve()


def _inline_mixed_block_contract(
    *,
    plan: dict[str, Any],
    run_state: Path,
    namespace: str,
    logical_layer: int,
    expected: dict[str, tuple[int, str]] | dict[str, dict[str, Any]],
    family_join: dict[str, Any],
) -> dict[str, Any] | None:
    """Authenticate one immutable tier plan and return its exact module tiers."""

    policies = plan.get("inline_mixed", {})
    if not isinstance(policies, dict):
        raise AuditError("quantization plan inline-mixed policies are invalid")
    raw = policies.get(namespace)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise AuditError("quantization plan inline-mixed policy is invalid")
    portable = _portable_inline_policy(raw)
    joined = family_join.get("inline_mixed")
    if not isinstance(joined, dict) or joined.get(namespace) != portable:
        raise AuditError("inline-mixed policy differs from family provenance")
    try:
        policy = inline_mixed_policy({INLINE_MIXED_META_KEY: raw})
    except (TypeError, ValueError) as error:
        raise AuditError("inline-mixed policy contract is invalid") from error
    if policy is None or policy.namespace != namespace:
        raise AuditError("inline-mixed policy namespace is invalid")
    base_bits = plan.get("exl3", {}).get("bits")
    if policy.base_bits != base_bits or policy.upgrade_bits != base_bits + 1:
        raise AuditError("inline-mixed policy differs from the base EXL3 tier")

    geometry = plan["source"]["geometry"]
    layer_count = (
        geometry["num_hidden_layers"]
        if namespace == "base"
        else len(geometry["dspark_target_layer_ids"])
    )
    expert_count = geometry["n_routed_experts"]
    tier_root = _resolve_planned_path(
        policy.tier_plan_root,
        plan=plan,
        run_state=run_state,
    )
    tier_path = tier_root / namespace / f"layer-{logical_layer:06d}.json"
    tier_plan = _read_object(tier_path, "inline-mixed tier plan")
    body = {key: value for key, value in tier_plan.items() if key != "tier_plan_sha256"}
    digest = tier_plan.get("tier_plan_sha256")
    expected_tier_keys = {
        *policy.policy_body,
        "policy_sha256",
        "layer_index",
        "layer_count",
        "experts_per_layer",
        "quotas",
        "selected",
        "tier_plan_sha256",
    }
    try:
        quotas = policy.layer_quotas(
            layer_index=logical_layer,
            layer_count=layer_count,
            experts_per_layer=expert_count,
        )
    except ValueError as error:
        raise AuditError("inline-mixed tier quotas are invalid") from error
    if (
        set(tier_plan) != expected_tier_keys
        or tier_plan.get("schema") != INLINE_MIXED_SCHEMA
        or tier_plan.get("schema_version") != INLINE_MIXED_SCHEMA_VERSION
        or tier_plan.get("namespace") != namespace
        or tier_plan.get("layer_index") != logical_layer
        or tier_plan.get("layer_count") != layer_count
        or tier_plan.get("experts_per_layer") != expert_count
        or tier_plan.get("policy_sha256") != policy.policy_sha256
        or tier_plan.get("quotas") != quotas
        or not isinstance(digest, str)
        or digest != _sha256(_canonical(body))
        or any(tier_plan.get(key) != value for key, value in policy.policy_body.items())
    ):
        raise AuditError("inline-mixed tier plan failed its immutable contract")

    selected_raw = tier_plan.get("selected")
    if not isinstance(selected_raw, list) or len(selected_raw) != sum(quotas.values()):
        raise AuditError("inline-mixed tier plan has an invalid selection count")
    selected: dict[str, dict[str, Any]] = {}
    observed_quotas = {projection: 0 for projection in PROJECTION_ORDER}
    for entry in selected_raw:
        if not isinstance(entry, dict) or set(entry) != {
            "module",
            "expert",
            "projection",
            "score",
            "candidate_record_sha256",
        }:
            raise AuditError("inline-mixed tier plan selection is malformed")
        module = entry.get("module")
        identity = expected.get(module) if isinstance(module, str) else None
        if isinstance(identity, tuple):
            expert, projection = identity
        elif isinstance(identity, dict):
            expert, projection = identity.get("expert"), identity.get("projection")
        else:
            raise AuditError("inline-mixed tier plan selected an unexpected module")
        score = entry.get("score")
        candidate_digest = entry.get("candidate_record_sha256")
        if (
            entry.get("expert") != expert
            or entry.get("projection") != projection
            or projection not in observed_quotas
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or float(score) < 0
            or not isinstance(candidate_digest, str)
            or launcher.SHA256_RE.fullmatch(candidate_digest) is None
            or module in selected
        ):
            raise AuditError("inline-mixed tier plan selection identity is invalid")
        observed_quotas[projection] += 1
        selected[module] = entry
    if observed_quotas != quotas:
        raise AuditError("inline-mixed tier plan does not close projection quotas")

    bits_by_module = {
        module: policy.upgrade_bits if module in selected else policy.base_bits
        for module in expected
    }
    return {
        "policy": portable,
        "policy_sha256": policy.policy_sha256,
        "tier_plan_sha256": digest,
        "tier_plan": tier_plan,
        "quotas": quotas,
        "selected": selected,
        "bits_by_module": bits_by_module,
        "base_bits": policy.base_bits,
        "upgrade_bits": policy.upgrade_bits,
    }


def _checkpoint_manifest_selection(
    checkpoint_root: Path,
    *,
    plan: dict[str, Any],
    run_state: Path,
    expected: dict[str, tuple[int, str]] | dict[str, dict[str, Any]],
    family_join: dict[str, Any],
    ignore_unexpected: bool,
    allow_missing: bool = False,
) -> tuple[
    dict[str, tuple[dict[str, Any], dict[str, Any]]],
    dict[str, tuple[tuple[dict[str, Any], dict[str, Any]], ...]],
    dict[tuple[str, int], dict[str, Any] | None],
]:
    """Select authoritative manifests while retaining all authenticated tiers."""

    blocks: dict[tuple[str, int], dict[str, Any]] = {}
    for module, raw_identity in expected.items():
        if isinstance(raw_identity, tuple):
            identity = routed_expert_identity(module)
        else:
            identity = raw_identity
        if not isinstance(identity, dict):
            raise AuditError(f"projection identity differs for {module}")
        key = (identity["block_namespace"], identity["logical_layer"])
        blocks.setdefault(key, {})[module] = raw_identity
    tier_contracts = {
        key: _inline_mixed_block_contract(
            plan=plan,
            run_state=run_state,
            namespace=key[0],
            logical_layer=key[1],
            expected=block_expected,
            family_join=family_join,
        )
        for key, block_expected in blocks.items()
    }

    store = EXL3ProjectionCheckpointStore(checkpoint_root)
    manifests: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for manifest_path in sorted(checkpoint_root.rglob("*.json")):
        if manifest_path.name.startswith("."):
            continue
        manifest = _read_object(manifest_path, "projection checkpoint manifest")
        request = manifest.get("request")
        module = request.get("module") if isinstance(request, dict) else None
        if module not in expected:
            if ignore_unexpected:
                continue
            raise AuditError(
                f"projection checkpoint contains an unexpected module: {module!r}"
            )
        request_digest = request.get("request_sha256")
        if not isinstance(request_digest, str):
            raise AuditError(
                f"projection checkpoint has no request digest for {module}"
            )
        try:
            expected_manifest, _expected_tensor = store._paths(request_digest)
        except ValueError as error:
            raise AuditError(
                f"projection checkpoint request is invalid for {module}"
            ) from error
        if expected_manifest != manifest_path:
            raise AuditError("projection checkpoint manifest path differs from request")
        manifests.setdefault(module, []).append((request, manifest))

    selected: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    retained: dict[str, tuple[tuple[dict[str, Any], dict[str, Any]], ...]] = {}
    for module, raw_identity in expected.items():
        identity = (
            routed_expert_identity(module)
            if isinstance(raw_identity, tuple)
            else raw_identity
        )
        key = (identity["block_namespace"], identity["logical_layer"])
        tier = tier_contracts[key]
        entries = manifests.get(module, [])
        by_role: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for request, manifest in entries:
            contract = request.get("quantizer_contract")
            inline = (
                contract.get("inline_mixed") if isinstance(contract, dict) else None
            )
            try:
                _module, role = store._module_request_key(request)
            except ValueError as error:
                raise AuditError(
                    f"projection checkpoint tier role is invalid for {module}"
                ) from error
            if role in by_role:
                raise AuditError(f"duplicate projection checkpoint tier for {module}")
            if tier is None:
                if role != "uniform" or inline is not None:
                    raise AuditError(
                        f"uniform projection has an inline-mixed checkpoint for {module}"
                    )
            else:
                expected_static = {
                    "base_bits": tier["base_bits"],
                    "upgrade_bits": tier["upgrade_bits"],
                    "policy_sha256": tier["policy_sha256"],
                    "role": role,
                }
                if (
                    not isinstance(inline, dict)
                    or any(
                        inline.get(key) != value
                        for key, value in expected_static.items()
                    )
                    or (role == "candidate_k2" and inline != expected_static)
                ):
                    raise AuditError(
                        f"inline-mixed checkpoint policy differs for {module}"
                    )
            by_role[role] = (request, manifest)

        if tier is None:
            if not by_role and allow_missing:
                continue
            if set(by_role) != {"uniform"}:
                raise AuditError(f"projection checkpoint is missing for {module}")
            selected[module] = by_role["uniform"]
        else:
            upgraded = module in tier["selected"]
            expected_roles = {"candidate_k2"}
            if upgraded:
                expected_roles.add("selected_k3")
            if set(by_role) != expected_roles:
                if allow_missing and set(by_role) < expected_roles:
                    retained[module] = tuple(entries)
                    continue
                raise AuditError(
                    f"inline-mixed checkpoint tiers differ for {module}: "
                    f"actual={sorted(by_role)} expected={sorted(expected_roles)}"
                )
            candidate_request = by_role["candidate_k2"][0]
            if upgraded:
                selected_request = by_role["selected_k3"][0]
                inline = selected_request["quantizer_contract"]["inline_mixed"]
                expected_inline = {
                    "base_bits": tier["base_bits"],
                    "upgrade_bits": tier["upgrade_bits"],
                    "policy_sha256": tier["policy_sha256"],
                    "role": "selected_k3",
                    "candidate_request_sha256": candidate_request["request_sha256"],
                    "tier_plan_sha256": tier["tier_plan_sha256"],
                }
                if inline != expected_inline:
                    raise AuditError(
                        f"inline-mixed selected checkpoint binding differs for {module}"
                    )
                selected[module] = by_role["selected_k3"]
            else:
                selected[module] = by_role["candidate_k2"]
        retained[module] = tuple(entries)
    return selected, retained, tier_contracts


def _block_report_geometry(
    *,
    hidden_size: int,
    intermediate_size: int,
    expert_count: int,
    bits: int,
    tier: dict[str, Any] | None,
) -> dict[str, Any]:
    geometry: dict[str, Any] = {
        "hidden_size": hidden_size,
        "moe_intermediate_size": intermediate_size,
        "n_routed_experts": expert_count,
        "bits": bits,
        "codebook": "mcg",
    }
    if tier is not None:
        selected_count = len(tier["selected"])
        geometry["inline_mixed"] = {
            "policy": tier["policy"],
            "policy_sha256": tier["policy_sha256"],
            "tier_plan_sha256": tier["tier_plan_sha256"],
            "quotas": tier["quotas"],
            "tier_counts": {
                str(tier["base_bits"]): expert_count * 3 - selected_count,
                str(tier["upgrade_bits"]): selected_count,
            },
        }
    return geometry


def _expected_tensor_contract(
    *,
    projection: str,
    hidden_size: int,
    intermediate_size: int,
    bits: int = 2,
) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    if hidden_size % 16 or intermediate_size % 16:
        raise AuditError("EXL3 block geometry is not divisible by 16")
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in {2, 3, 4}:
        raise AuditError("EXL3 block bitrate must be integer K2, K3, or K4")
    if projection in {"w1", "w3"}:
        input_features, output_features = hidden_size, intermediate_size
    elif projection == "w2":
        input_features, output_features = intermediate_size, hidden_size
    else:
        raise AuditError(f"unknown EXL3 projection identity: {projection}")
    return {
        "trellis": (
            torch.int16,
            (input_features // 16, output_features // 16, bits * 16),
        ),
        "suh": (torch.float16, (input_features,)),
        "svh": (torch.float16, (output_features,)),
        "mcg": (torch.int32, ()),
    }


def _validate_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    projection: str,
    hidden_size: int,
    intermediate_size: int,
    bits: int = 2,
) -> int:
    expected = _expected_tensor_contract(
        projection=projection,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        bits=bits,
    )
    if set(tensors) != set(expected):
        raise AuditError(
            f"projection tensor names differ: actual={sorted(tensors)} "
            f"expected={sorted(expected)}"
        )
    encoded_bytes = 0
    for name, (dtype, shape) in expected.items():
        tensor = tensors[name]
        if tensor.dtype != dtype or tuple(tensor.shape) != shape:
            raise AuditError(
                f"projection tensor {name} has {tensor.dtype}/{tuple(tensor.shape)}, "
                f"expected {dtype}/{shape}"
            )
        encoded_bytes += tensor.numel() * tensor.element_size()
    if (
        not torch.isfinite(tensors["suh"]).all()
        or not torch.isfinite(tensors["svh"]).all()
    ):
        raise AuditError("projection rotations contain NaN or Inf")
    if int(tensors["mcg"].item()) & 0xFFFFFFFF != MCG_MARKER:
        raise AuditError("projection MCG marker does not match the DS4RT codebook")
    return encoded_bytes


def _assignment_for_module(
    root: Path,
    module: str,
    *,
    expected_scheduler: str,
) -> tuple[dict[str, Any], str]:
    key_sha256 = _sha256(module.encode("utf-8"))
    path = root / key_sha256[:2] / key_sha256[2:4] / f"{key_sha256}.json"
    record = _read_object(path, f"assignment for {module}")
    digest = _validate_bound_record(record, label=f"assignment for {module}")
    if (
        record.get("schema") != "ds4rt.exl3-dynamic-projection-assignments"
        or record.get("schema_version") != 1
        or record.get("scheduler") != expected_scheduler
        or record.get("assignment_key") != module
        or record.get("assignment_key_sha256") != key_sha256
        or not isinstance(record.get("slot_id"), str)
        or not isinstance(record.get("execution"), dict)
    ):
        raise AuditError(f"assignment for {module} failed its scheduler contract")
    return record, digest


def _execution_ownership_for_module(
    *,
    plan: dict[str, Any],
    run_state: Path,
    module: str,
    result: dict[str, Any],
    ledger: dict[str, Any],
) -> tuple[str, str]:
    """Validate distributed assignments or bind coordinator-only ownership."""

    remote = plan.get("remote_workers")
    if isinstance(remote, dict):
        scheduler = remote.get("scheduler")
        if not isinstance(scheduler, str) or not scheduler:
            raise AuditError("quantization plan has no dynamic scheduler identity")
        assignment, digest = _assignment_for_module(
            run_state / launcher.REMOTE_ASSIGNMENT_DIRNAME,
            module,
            expected_scheduler=scheduler,
        )
        execution_contract = result.get("execution_contract")
        execution_result = result.get("execution_result")
        if (
            assignment.get("execution") != execution_contract
            or ledger.get("provenance", {}).get("execution") != execution_contract
            or not isinstance(execution_result, dict)
            or execution_result.get("scheduler_assignment_key") != module
            or ledger.get("devices") != result.get("device_names")
        ):
            raise AuditError(f"projection execution ownership differs for {module}")
        return assignment["slot_id"], digest

    devices = result.get("device_names")
    execution_result = result.get("execution_result")
    allowed = {
        f"cuda:{gpu['index']}"
        for gpu in plan.get("preflight", {}).get("gpus", ())
        if isinstance(gpu, dict) and isinstance(gpu.get("index"), int)
    }
    if (
        result.get("execution_contract") is not None
        or ledger.get("provenance", {}).get("execution") is not None
        or not isinstance(execution_result, dict)
        or execution_result.get("kind") != "coordinator"
        or execution_result.get("scheduler_assignment_key") is not None
        or not isinstance(devices, list)
        or len(devices) != 1
        or devices[0] not in allowed
        or ledger.get("devices") != devices
    ):
        raise AuditError(f"coordinator-only ownership differs for {module}")
    record = {
        "schema": "ds4rt.exl3-coordinator-projection-ownership",
        "schema_version": 1,
        "plan_sha256": plan["plan_sha256"],
        "module": module,
        "slot_id": f"coordinator:{devices[0]}",
        "device": devices[0],
    }
    return record["slot_id"], _sha256(_canonical(record))


def _projection_error_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(values: list[float]) -> dict[str, float]:
        return {
            "min": min(values),
            "mean": sum(values) / len(values),
            "max": max(values),
        }

    weighted = [
        float(record["quantizer_metrics"]["hessian_weighted_relative_error"])
        for record in records
    ]
    corrections = [
        float(record["quantizer_metrics"]["hessian_symmetry_correction_max_abs"])
        for record in records
    ]
    by_projection = {}
    for projection in PROJECTION_NAMES:
        selected = [
            float(record["quantizer_metrics"]["hessian_weighted_relative_error"])
            for record in records
            if record["projection"] == projection
        ]
        if selected:
            by_projection[projection] = summarize(selected)
    return {
        "hessian_weighted_relative_error": summarize(weighted),
        "hessian_symmetry_correction_max_abs": summarize(corrections),
        "by_projection": by_projection,
    }


def _route_summary(families: list[dict[str, Any]]) -> dict[str, Any]:
    counts = [
        int(family["route_evidence"]["expert_route_count"]) for family in families
    ]
    summary = {
        "experts": len(counts),
        "zero_count": sum(count == 0 for count in counts),
        "min": min(counts),
        "mean": sum(counts) / len(counts),
        "max": max(counts),
    }
    recovered = [
        family for family in families if family.get("zero_route_recovery") is not None
    ]
    if recovered:
        router_augmented = [
            int(family["zero_route_recovery"]["router_augmented_sample_count"])
            for family in recovered
        ]
        identity = [
            int(family["zero_route_recovery"]["identity_calibration_count"])
            for family in recovered
        ]
        summary["augmented_expert_count"] = len(recovered)
        summary["augmented_experts"] = sorted(
            int(family["expert"]) for family in recovered
        )
        summary["router_augmented_samples"] = {
            "min": min(router_augmented),
            "mean": sum(router_augmented) / len(router_augmented),
            "max": max(router_augmented),
        }
        summary["identity_calibration"] = {
            "expert_count": sum(value > 0 for value in identity),
            "effective_count": sum(identity),
        }
    return summary


def _expected_recovery_authorization(
    family_join: dict[str, Any],
) -> dict[str, Any] | None:
    family_digest = _sha256(_canonical(family_join))
    if family_join.get("zero_route_recovery_contract") != ZERO_ROUTE_RECOVERY_SCHEMA:
        return None
    return {
        "schema": ZERO_ROUTE_RECOVERY_AUTHORIZATION_SCHEMA,
        "schema_version": 1,
        "kind": "immutable-family-join",
        "recovery_contract": ZERO_ROUTE_RECOVERY_SCHEMA,
        "trigger": ZERO_ROUTE_RECOVERY_TRIGGER,
        "sample_source": ZERO_ROUTE_RECOVERY_SAMPLE_SOURCE,
        "capture_method": ZERO_ROUTE_RECOVERY_CAPTURE_METHOD,
        "selection_policy": ZERO_ROUTE_RECOVERY_SELECTION_POLICY,
        "candidate_rank_min": ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MIN,
        "candidate_rank_max": ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MAX,
        "target_sample_count": ZERO_ROUTE_RECOVERY_TARGET_SAMPLE_COUNT,
        "identity_calibration_policy": ZERO_ROUTE_RECOVERY_IDENTITY_POLICY,
        "family_join_sha256": family_digest,
        "authorization_sha256": family_digest,
    }


def _tensor_payload(tensor: torch.Tensor) -> bytes:
    tensor = tensor.detach().contiguous().cpu()
    return tensor.view(torch.uint8).numpy().tobytes()


def _tp4_residency_summary(
    selected: dict[
        str,
        tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]],
    ],
    expected: dict[str, tuple[int, str]],
    *,
    hidden_size: int,
    intermediate_size: int,
    expert_count: int,
) -> dict[str, Any]:
    """Hash the exact plane order consumed by the Rust/SparkInfer K2 slabs."""

    if (
        expert_count <= 0
        or len(expected) != expert_count * len(PROJECTION_NAMES)
        or hidden_size % 16
        or intermediate_size % (16 * TP4_WORLD_SIZE)
        or len(selected) != expert_count * len(PROJECTION_NAMES)
    ):
        raise AuditError(
            "TP4 residency requires complete 16-tile-aligned expert geometry"
        )
    by_identity: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    for module, (expert, projection) in expected.items():
        item = selected.get(module)
        if item is None or (expert, projection) in by_identity:
            raise AuditError(
                "TP4 residency projection identity is incomplete or repeated"
            )
        by_identity[(expert, projection)] = item[0]

    local_intermediate = intermediate_size // TP4_WORLD_SIZE
    local_tiles = local_intermediate // 16
    component_order = (
        "w13_trellis",
        "w2_trellis",
        "gate_suh",
        "up_suh",
        "intermediate_rotations",
        "down_svh",
    )
    rank_reports = []
    for rank in range(TP4_WORLD_SIZE):
        tile_start = rank * local_tiles
        tile_stop = tile_start + local_tiles
        rotation_start = rank * local_intermediate
        rotation_stop = rotation_start + local_intermediate
        hashers = {name: hashlib.sha256() for name in component_order}
        component_bytes = {name: 0 for name in component_order}

        def consume(component: str, tensor: torch.Tensor) -> None:
            payload = _tensor_payload(tensor)
            hashers[component].update(payload)
            component_bytes[component] += len(payload)

        # Runtime W13 is projection-major: all gate experts, then all up
        # experts. W2 is all down experts. This is deliberately not the native
        # FP4 [up; gate] repack order.
        for projection in ("w1", "w3"):
            for expert in range(expert_count):
                trellis = by_identity[(expert, projection)]["trellis"]
                consume(
                    "w13_trellis",
                    trellis[:, tile_start:tile_stop, :],
                )
        for expert in range(expert_count):
            trellis = by_identity[(expert, "w2")]["trellis"]
            consume("w2_trellis", trellis[tile_start:tile_stop, :, :])
        for expert in range(expert_count):
            consume("gate_suh", by_identity[(expert, "w1")]["suh"])
        for expert in range(expert_count):
            consume("up_suh", by_identity[(expert, "w3")]["suh"])
        for expert in range(expert_count):
            consume(
                "intermediate_rotations",
                by_identity[(expert, "w1")]["svh"][rotation_start:rotation_stop],
            )
            consume(
                "intermediate_rotations",
                by_identity[(expert, "w3")]["svh"][rotation_start:rotation_stop],
            )
            consume(
                "intermediate_rotations",
                by_identity[(expert, "w2")]["suh"][rotation_start:rotation_stop],
            )
        for expert in range(expert_count):
            consume("down_svh", by_identity[(expert, "w2")]["svh"])

        components = {
            name: {
                "bytes": component_bytes[name],
                "sha256": hashers[name].hexdigest(),
            }
            for name in component_order
        }
        resident_bytes = sum(item["bytes"] for item in components.values())
        rank_reports.append(
            {
                "rank": rank,
                "intermediate_start": rotation_start,
                "intermediate_stop": rotation_stop,
                "checkpoint_payload_bytes": resident_bytes,
                "components": components,
                "component_manifest_sha256": _sha256(
                    b"".join(
                        bytes.fromhex(components[name]["sha256"])
                        for name in component_order
                    )
                ),
            }
        )

    # Prove the axes used above partition the complete source tensors. The
    # hidden rotations are intentionally replicated, while exactly the three
    # intermediate rotations are partitioned over TP4.
    for expert in range(expert_count):
        for projection in ("w1", "w3"):
            tensors = by_identity[(expert, projection)]
            if not torch.equal(
                torch.cat(
                    [
                        tensors["trellis"][
                            :,
                            rank * local_tiles : (rank + 1) * local_tiles,
                            :,
                        ]
                        for rank in range(TP4_WORLD_SIZE)
                    ],
                    dim=1,
                ),
                tensors["trellis"],
            ) or not torch.equal(
                torch.cat(
                    [
                        tensors["svh"][
                            rank * local_intermediate : (rank + 1) * local_intermediate
                        ]
                        for rank in range(TP4_WORLD_SIZE)
                    ]
                ),
                tensors["svh"],
            ):
                raise AuditError("TP4 FC1 trellis/rotation partition is not lossless")
        down = by_identity[(expert, "w2")]
        if not torch.equal(
            torch.cat(
                [
                    down["trellis"][
                        rank * local_tiles : (rank + 1) * local_tiles,
                        :,
                        :,
                    ]
                    for rank in range(TP4_WORLD_SIZE)
                ],
                dim=0,
            ),
            down["trellis"],
        ) or not torch.equal(
            torch.cat(
                [
                    down["suh"][
                        rank * local_intermediate : (rank + 1) * local_intermediate
                    ]
                    for rank in range(TP4_WORLD_SIZE)
                ]
            ),
            down["suh"],
        ):
            raise AuditError("TP4 FC2 trellis/rotation partition is not lossless")

    identity_expert_map = b"".join(
        value.to_bytes(4, "little", signed=True) for value in range(expert_count + 1)
    )
    dummy_scale = bytes(16)
    global_scale = struct.pack("<f", 1.0) * expert_count
    runtime_generated = {
        "bytes": len(identity_expert_map) + len(dummy_scale) + len(global_scale),
        "identity_expert_map_sha256": _sha256(identity_expert_map),
        "dummy_scale_sha256": _sha256(dummy_scale),
        "global_scale_sha256": _sha256(global_scale),
        "combined_sha256": _sha256(identity_expert_map + dummy_scale + global_scale),
    }
    checkpoint_sizes = {report["checkpoint_payload_bytes"] for report in rank_reports}
    if len(checkpoint_sizes) != 1:
        raise AuditError("TP4 ranks do not have equal resident payload bytes")
    for rank_report in rank_reports:
        components = rank_report["components"]
        rank_report["resident_weight_bytes"] = (
            components["w13_trellis"]["bytes"] + components["w2_trellis"]["bytes"]
        )
        rank_report["resident_metadata_bytes"] = (
            rank_report["checkpoint_payload_bytes"]
            - rank_report["resident_weight_bytes"]
            + runtime_generated["bytes"]
        )
        rank_report["resident_bytes"] = (
            rank_report["checkpoint_payload_bytes"] + runtime_generated["bytes"]
        )
    report = {
        "schema": TP4_RESIDENCY_SCHEMA,
        "world_size": TP4_WORLD_SIZE,
        "placement": "strict-tp4-replicated-experts",
        "expert_ownership": "reduction-metadata-only",
        "expert_count_per_rank": expert_count,
        "local_intermediate_size": local_intermediate,
        "rank_resident_bytes_equal": True,
        "rank_resident_bytes": rank_reports[0]["resident_bytes"],
        "ranks": rank_reports,
        "runtime_generated": runtime_generated,
    }
    report["report_sha256"] = _sha256(_canonical(report))
    return report


def audit_block(
    run_state: Path,
    *,
    block_namespace: str,
    logical_layer: int,
    require_complete: bool = True,
    include_tp4_residency: bool = False,
) -> dict[str, Any]:
    run_state = run_state.expanduser().resolve(strict=True)
    if not run_state.is_dir() or run_state.is_symlink():
        raise AuditError(f"run state is not one regular directory: {run_state}")
    if logical_layer < 0:
        raise AuditError("logical layer must be nonnegative")

    plan_path = run_state / launcher.PLAN_FILENAME
    plan = _read_object(plan_path, "quantization plan")
    plan_schema = plan.get("schema")
    if plan_schema in launcher.SUPPORTED_PLAN_SCHEMAS:
        try:
            launcher._validate_plan(plan)
        except launcher.LaunchError as error:
            raise AuditError(str(error)) from error
    elif plan_schema in {
        "ds4rt-deepseek-v4-dspark-overlay-plan-v1",
        "ds4rt-deepseek-v4-dspark-overlay-plan-v2",
    }:
        digest = plan.get("plan_sha256")
        body = {key: value for key, value in plan.items() if key != "plan_sha256"}
        if (
            not isinstance(digest, str)
            or digest != _sha256(_canonical(body))
            or plan.get("scope") != "mtp-routed-experts-only"
        ):
            raise AuditError("dSpark overlay plan digest or scope is invalid")
        if block_namespace != "mtp":
            raise AuditError("dSpark overlay plans contain only MTP blocks")
    else:
        raise AuditError("quantization plan schema is unsupported")
    if Path(plan.get("run_state_dir", "")).resolve() != run_state:
        raise AuditError("quantization plan names a different run-state directory")

    geometry = plan.get("source", {}).get("geometry", {})
    values = {
        "hidden_size": geometry.get("hidden_size"),
        "intermediate_size": geometry.get("moe_intermediate_size"),
        "expert_count": geometry.get("n_routed_experts"),
        "base_layer_count": geometry.get("num_hidden_layers"),
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values.values()
    ):
        raise AuditError("quantization plan has invalid DeepSeek V4 geometry")
    if block_namespace == "base" and logical_layer >= values["base_layer_count"]:
        raise AuditError("base logical layer is outside the plan")
    if block_namespace == "mtp" and logical_layer >= len(
        geometry.get("dspark_target_layer_ids", [])
    ):
        raise AuditError("MTP logical layer is outside the plan")
    bits = plan.get("exl3", {}).get("bits")
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in {2, 3, 4}:
        raise AuditError("quantization plan has an invalid integer EXL3 tier")

    family_join = plan.get("ledger_provenance", {}).get("family_join")
    if not isinstance(family_join, dict):
        raise AuditError("quantization plan has no family-join provenance")

    checkpoint_contract = plan.get("projection_checkpoint")
    checkpoint_root = Path(
        checkpoint_contract.get("root", "")
        if isinstance(checkpoint_contract, dict)
        else ""
    ).resolve()
    if (
        not isinstance(checkpoint_contract, dict)
        or checkpoint_contract.get("contract") != CHECKPOINT_CONTRACT
        or not checkpoint_root.is_dir()
        or checkpoint_root.is_symlink()
    ):
        raise AuditError("projection checkpoint root differs from the plan")

    expected = _expected_modules(
        namespace=block_namespace,
        logical_layer=logical_layer,
        expert_count=values["expert_count"],
    )
    store = EXL3ProjectionCheckpointStore(checkpoint_root)
    selected_manifests, retained_manifests, tier_contracts = (
        _checkpoint_manifest_selection(
            checkpoint_root,
            plan=plan,
            run_state=run_state,
            expected=expected,
            family_join=family_join,
            ignore_unexpected=True,
            allow_missing=True,
        )
    )
    journal = _load_journal(run_state / launcher.ERROR_JOURNAL_FILENAME)
    block_key = (block_namespace, logical_layer)
    block_tier = tier_contracts[block_key]
    candidate_journal = (
        _load_journal(run_state / launcher.INLINE_MIXED_CANDIDATE_JOURNAL_FILENAME)
        if block_tier is not None
        else None
    )
    loaded_by_request: dict[str, tuple[dict[str, torch.Tensor], dict[str, Any]]] = {}

    def load_entry(
        entry: tuple[dict[str, Any], dict[str, Any]],
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        request, _manifest = entry
        digest = request["request_sha256"]
        loaded = loaded_by_request.get(digest)
        if loaded is None:
            loaded = store.load(request)
            if loaded is None:
                raise AuditError(
                    f"projection checkpoint disappeared for {request.get('module')}"
                )
            loaded_by_request[digest] = loaded
        return loaded

    selected: dict[
        str,
        tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]],
    ] = {}
    for module, entry in selected_manifests.items():
        tensors, result = load_entry(entry)
        selected[module] = (tensors, result, entry[1])

    if block_tier is not None:
        assert candidate_journal is not None
        for module, entries in retained_manifests.items():
            candidate_entries = [
                entry
                for entry in entries
                if entry[0]
                .get("quantizer_contract", {})
                .get("inline_mixed", {})
                .get("role")
                == "candidate_k2"
            ]
            if len(candidate_entries) != 1:
                raise AuditError(f"inline-mixed K2 candidate differs for {module}")
            candidate_request, _candidate_manifest = candidate_entries[0]
            _candidate_tensors, candidate_result = load_entry(candidate_entries[0])
            candidate_ledger = candidate_result.get("ledger_record")
            candidate_bound = candidate_journal.get(module)
            if (
                not isinstance(candidate_ledger, dict)
                or candidate_ledger.get("bits") != block_tier["base_bits"]
                or candidate_ledger.get("module") != module
                or candidate_ledger.get("provenance", {}).get("family_join")
                != family_join
                or candidate_bound is None
                or candidate_bound[0] != candidate_ledger
            ):
                raise AuditError(
                    f"inline-mixed K2 candidate evidence differs for {module}"
                )
            selected_entry = block_tier["selected"].get(module)
            if selected_entry is not None and module in selected_manifests:
                try:
                    score = projection_score(candidate_ledger)
                except ValueError as error:
                    raise AuditError(
                        f"inline-mixed K2 candidate score differs for {module}"
                    ) from error
                candidate_digest = _sha256(_canonical(candidate_ledger))
                chosen_request = selected_manifests[module][0]
                candidate_body = {
                    key: value
                    for key, value in candidate_request.items()
                    if key not in {"request_sha256", "quantizer_contract"}
                }
                chosen_body = {
                    key: value
                    for key, value in chosen_request.items()
                    if key not in {"request_sha256", "quantizer_contract"}
                }
                candidate_contract = {
                    key: value
                    for key, value in candidate_request["quantizer_contract"].items()
                    if key not in {"bits", "inline_mixed"}
                }
                chosen_contract = {
                    key: value
                    for key, value in chosen_request["quantizer_contract"].items()
                    if key not in {"bits", "inline_mixed"}
                }
                if (
                    selected_entry["candidate_record_sha256"] != candidate_digest
                    or selected_entry["score"] != score
                    or candidate_body != chosen_body
                    or candidate_contract != chosen_contract
                ):
                    raise AuditError(
                        f"inline-mixed K3 selection differs from its K2 candidate for {module}"
                    )

    missing = sorted(set(expected) - set(selected))
    if require_complete and missing:
        raise AuditError(
            f"block has {len(selected)}/{len(expected)} projection checkpoints; "
            f"first missing module is {missing[0]}"
        )
    if not selected:
        raise AuditError("block has no committed projection checkpoints")
    if include_tp4_residency and missing:
        raise AuditError("TP4 residency evidence requires a complete block")

    projection_records: list[dict[str, Any]] = []
    assignment_digests: list[tuple[str, str]] = []
    checkpoint_digests: list[tuple[str, str, str]] = []
    journal_digests: list[tuple[str, str]] = []
    ownership: dict[str, int] = {}
    encoded_bytes = 0
    expected_recovery_authorization = _expected_recovery_authorization(
        family_join,
    )

    for module in sorted(selected):
        tensors, result, manifest = selected[module]
        expert, projection = expected[module]
        request = manifest["request"]
        expected_bits = (
            block_tier["bits_by_module"][module] if block_tier is not None else bits
        )
        identity = routed_expert_identity(module)
        expected_identity = {
            "block_namespace": block_namespace,
            "logical_layer": logical_layer,
            "expert": expert,
            "projection": projection,
        }
        if identity != expected_identity:
            raise AuditError(f"projection identity differs for {module}")
        if request.get("family_join") != family_join:
            raise AuditError(f"projection request provenance differs for {module}")
        quantizer_contract = request.get("quantizer_contract")
        if (
            not isinstance(quantizer_contract, dict)
            or quantizer_contract.get("bits") != expected_bits
            or quantizer_contract.get("codebook") != "mcg"
            or quantizer_contract.get("apply_out_scales") is not None
            or quantizer_contract.get("sigma_reg") != launcher.EXL3_SIGMA_REG
            or quantizer_contract.get("seed") != launcher.EXL3_SEED
            or quantizer_contract.get("hessian_capture")
            != launcher.EXL3_HESSIAN_CAPTURE_CONTRACT
            or quantizer_contract.get("hessian_numerical")
            != launcher.EXL3_HESSIAN_NUMERICAL_CONTRACT
            or quantizer_contract.get("hessian_symmetry")
            != launcher.EXL3_HESSIAN_SYMMETRY_CONTRACT
        ):
            raise AuditError(f"projection request numerics differ for {module}")

        projection_bytes = _validate_tensors(
            tensors,
            projection=projection,
            hidden_size=values["hidden_size"],
            intermediate_size=values["intermediate_size"],
            bits=expected_bits,
        )
        encoded_bytes += projection_bytes

        ledger = result.get("ledger_record")
        if not isinstance(ledger, dict):
            raise AuditError(f"projection checkpoint has no ledger for {module}")
        if any(
            not math.isfinite(float(value))
            for value in (
                result.get("duration_seconds", math.nan),
                result.get("proxy_error", math.nan),
            )
        ):
            raise AuditError(
                f"projection checkpoint has non-finite results for {module}"
            )
        metrics = ledger.get("quantizer_metrics")
        if (
            ledger.get("schema") != LEDGER_SCHEMA
            or ledger.get("schema_version") != LEDGER_SCHEMA_VERSION
            or ledger.get("record_kind") != "projection"
            or ledger.get("module") != module
            or any(ledger.get(key) != value for key, value in expected_identity.items())
            or ledger.get("bits") != expected_bits
            or ledger.get("codebook") != "mcg"
            or ledger.get("encoded_bytes") != projection_bytes
            or ledger.get("sample_count") != request.get("sample_count")
            or ledger.get("route_evidence") != request.get("route_evidence")
            or ledger.get("zero_route_recovery") != request.get("zero_route_recovery")
            or not isinstance(metrics, dict)
            or metrics.get("hessian_metric_status") != "ok"
            or metrics.get("hessian_regularization_sigma") != launcher.EXL3_SIGMA_REG
            or metrics.get("hessian_numerical_contract")
            != launcher.EXL3_HESSIAN_NUMERICAL_CONTRACT
            or metrics.get("hessian_transform_compute_dtype") != "torch.float64"
            or metrics.get("hessian_storage_dtype") != "torch.float32"
            or metrics.get("hessian_regularization_placement")
            != "before-fp64-congruence"
            or metrics.get("hessian_symmetry_restoration")
            != launcher.EXL3_HESSIAN_SYMMETRY_CONTRACT
            or not isinstance(
                metrics.get("hessian_regularization_diagonal_addend"),
                (int, float),
            )
            or not math.isfinite(
                float(metrics["hessian_regularization_diagonal_addend"])
            )
            or metrics["hessian_regularization_diagonal_addend"] <= 0
            or result.get("quantizer_metrics") != metrics
            or result.get("proxy_error") != metrics.get("reported_metric_value")
            or ledger.get("provenance", {}).get("family_join") != family_join
        ):
            raise AuditError(f"projection ledger differs for {module}")

        recovery = ledger.get("zero_route_recovery")
        if recovery is not None:
            if expected_recovery_authorization is None:
                raise AuditError(
                    f"augmented projection has no authorization for {module}"
                )
            try:
                validate_zero_route_recovery(
                    recovery,
                    identity=expected_identity,
                    sample_count=ledger["sample_count"],
                    family_join=family_join,
                    expected_authorization=expected_recovery_authorization,
                )
            except (TypeError, ValueError) as error:
                raise AuditError(
                    f"route-coverage augmentation evidence differs for {module}"
                ) from error

        journal_record = journal.get(module)
        if journal_record is None or journal_record[0] != ledger:
            raise AuditError(f"projection journal differs for {module}")
        journal_digests.append((module, journal_record[1]))

        slot_id, assignment_digest = _execution_ownership_for_module(
            plan=plan,
            run_state=run_state,
            module=module,
            result=result,
            ledger=ledger,
        )
        ownership[slot_id] = ownership.get(slot_id, 0) + 1
        assignment_digests.append((module, assignment_digest))
        checkpoint_digests.append(
            (module, manifest["manifest_sha256"], manifest["tensor_sha256"])
        )
        projection_records.append(ledger)

    families = derive_family_records(projection_records)
    complete_experts = len(families)
    if require_complete and complete_experts != values["expert_count"]:
        raise AuditError(
            f"block has {complete_experts}/{values['expert_count']} complete expert families"
        )
    if any("route_evidence" not in family for family in families):
        raise AuditError("complete expert family is missing natural-route evidence")

    report = {
        "schema": AUDIT_SCHEMA,
        "status": "complete" if not missing else "partial",
        "plan_sha256": plan["plan_sha256"],
        "run_state": os.fspath(run_state),
        "block_namespace": block_namespace,
        "logical_layer": logical_layer,
        "geometry": _block_report_geometry(
            hidden_size=values["hidden_size"],
            intermediate_size=values["intermediate_size"],
            expert_count=values["expert_count"],
            bits=bits,
            tier=block_tier,
        ),
        "projection_count": len(projection_records),
        "expected_projection_count": len(expected),
        "complete_expert_families": complete_experts,
        "encoded_bytes": encoded_bytes,
        "ownership": dict(sorted(ownership.items())),
        "route_coverage": _route_summary(families) if families else None,
        "errors": _projection_error_summary(projection_records),
        "content": {
            "checkpoint_records_sha256": _sha256(_canonical(checkpoint_digests)),
            "assignment_records_sha256": _sha256(_canonical(assignment_digests)),
            "journal_records_sha256": _sha256(_canonical(journal_digests)),
        },
        "missing_projection_count": len(missing),
        "first_missing_projection": missing[0] if missing else None,
    }
    if include_tp4_residency:
        report["tp4_residency"] = _tp4_residency_summary(
            selected,
            expected,
            hidden_size=values["hidden_size"],
            intermediate_size=values["intermediate_size"],
            expert_count=values["expert_count"],
        )
    report["report_sha256"] = _sha256(_canonical(report))
    return report


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as target:
            target.write(json.dumps(value, indent=2, sort_keys=True).encode("utf-8"))
            target.write(b"\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def main() -> None:
    args = parse_args()
    report = audit_block(
        args.run_state,
        block_namespace=args.block_namespace,
        logical_layer=args.logical_layer,
        require_complete=not args.allow_partial,
        include_tp4_residency=args.include_tp4_residency,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        write_json_atomic(args.output, report)


if __name__ == "__main__":
    main()
