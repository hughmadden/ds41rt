#!/usr/bin/env python3
"""Assemble a canonical DS41RT hybrid artifact from GPTQModel checkpoints.

GPTQModel's model writer serializes the transformed Transformers module tree.
DS41RT instead needs the original DeepSeek checkpoint representation for every
coordinator-owned tensor and GPTQModel's EXL3 representation only for routed
experts.  This tool performs that bounded, resumable merge without loading a
full model or trusting the writer's reconstructed dense state.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Iterable

from ds41rt_runtime.exl3_artifact_contract import (
    LEDGER_FILE,
    LEDGER_MANIFEST_FILE,
    validate_gptqmodel_publication,
)
from ds41rt_runtime.exl3_quantizer import (
    EXPERT_TENSOR_LAYOUT_GPTQMODEL,
    OutputTensor,
    SafetensorsArtifactWriter,
    build_artifact_plan,
    gptqmodel_tensor_storage_for_plan,
)
from gptqmodel.utils.exl3_error_ledger import (
    derive_family_records,
    routed_expert_identity,
    write_exl3_error_ledger,
)
from gptqmodel.utils.exl3_projection_checkpoint import EXL3ProjectionCheckpointStore

import quantize_flash_gptqmodel as launcher
import quantize_flash_dspark_overlay as dspark_overlay
import validate_projection_checkpoint_block as block_audit


PROJECTION_ASSEMBLY_SCHEMA = "ds41rt-exl3-canonical-hybrid-projection-assembly-v1"
COMPOSITE_PROJECTION_ASSEMBLY_SCHEMA = (
    "ds41rt-exl3-canonical-hybrid-composite-projection-assembly-v1"
)
ASSEMBLY_SCHEMA = "ds41rt-exl3-canonical-hybrid-assembly-v1"
ASSEMBLY_FILENAME = "ds41rt-exl3-canonical-assembly.json"
QUANT_CONFIG_ASSEMBLY_SCHEMA = "ds41rt-exl3-canonical-quant-config-assembly-v1"
COMPOSITE_ASSEMBLY_SCHEMA = "ds41rt-exl3-canonical-hybrid-assembly-v2"
COMPOSITE_QUANT_CONFIG_ASSEMBLY_SCHEMA = (
    "ds41rt-exl3-canonical-composite-quant-config-assembly-v1"
)
BLOCK_AUDIT_SET_SCHEMA = "ds41rt-exl3-independent-block-audit-set-v1"
COMPOSITE_BLOCK_AUDIT_SET_SCHEMA = "ds41rt-exl3-independent-composite-block-audit-set-v1"
PROJECTION_SUFFIXES = ("trellis", "suh", "svh", "mcg")
PROJECTION_NAMES = {
    "w1": "gate_proj",
    "w2": "down_proj",
    "w3": "up_proj",
}


class AssemblyError(RuntimeError):
    """The durable quantization evidence cannot produce a canonical artifact."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _bound_report(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "report_sha256": _sha256(_canonical(value))}


def _portable_source_identity(source: dict[str, Any]) -> dict[str, Any]:
    """Remove only the bind-mount spelling from a snapshot identity."""

    return {key: value for key, value in source.items() if key != "path"}


def _source_complete_raw_config(
    raw_config: dict[str, Any],
    source_config: dict[str, Any],
) -> dict[str, Any]:
    """Restore source-only config extensions for raw-envelope validation.

    Older GPTQModel terminal exports serialized the registered Transformers
    config class and could therefore omit extension fields unknown to that
    class. The raw artifact remains byte-for-byte authenticated; this merged
    in-memory view supplies only missing fields from its independently bound
    source snapshot. Canonical output is assembled from that native source
    config below rather than from the raw writer's model config.
    """

    if not isinstance(raw_config, dict) or not isinstance(source_config, dict):
        raise AssemblyError("model config must be a JSON object")
    completed = deepcopy(raw_config)
    for key, value in source_config.items():
        if key not in {"attn_implementation", "_attn_implementation"}:
            completed.setdefault(key, deepcopy(value))
    return completed


def _validate_planned_run_state_paths(plan: dict[str, Any]) -> None:
    """Validate immutable durable-store paths for legacy and split layouts."""

    raw_run_state = plan.get("run_state_dir")
    checkpoint = plan.get("projection_checkpoint")
    remote = plan.get("remote_workers")
    if (
        not isinstance(raw_run_state, str)
        or not raw_run_state
        or not Path(raw_run_state).is_absolute()
        or not isinstance(checkpoint, dict)
        or (remote is not None and not isinstance(remote, dict))
    ):
        raise AssemblyError("quantization plan has invalid durable-store paths")
    planned_run_state = Path(raw_run_state)
    checkpoint_root = Path(checkpoint.get("root", ""))
    has_projection_dir = "projection_checkpoint_dir" in plan
    has_active_source_dir = "active_layer_source_dir" in plan
    if not has_projection_dir and not has_active_source_dir:
        expected_checkpoint = planned_run_state / launcher.PROJECTION_CHECKPOINT_DIRNAME
    else:
        if not has_projection_dir or not has_active_source_dir:
            raise AssemblyError("quantization plan has incomplete split storage roots")
        expected_checkpoint = Path(plan.get("projection_checkpoint_dir", ""))
        active_source = Path(plan.get("active_layer_source_dir", ""))
        if not expected_checkpoint.is_absolute() or not active_source.is_absolute():
            raise AssemblyError("quantization plan has invalid split storage roots")
    if (
        checkpoint_root != expected_checkpoint
        or not checkpoint_root.is_absolute()
        or (
            isinstance(remote, dict)
            and Path(remote.get("assignment_store", ""))
            != planned_run_state / launcher.REMOTE_ASSIGNMENT_DIRNAME
        )
    ):
        raise AssemblyError(
            "quantization plan durable stores differ from their immutable roots"
        )


def _resolve_checkpoint_root(
    plan: dict[str, Any],
    actual_run_state: Path,
) -> Path:
    """Resolve a run-state-relative store across a host/container mount alias."""

    planned_run_state = Path(plan["run_state_dir"])
    planned_root = Path(plan["projection_checkpoint"]["root"])
    try:
        relative = planned_root.relative_to(planned_run_state)
    except ValueError:
        return planned_root
    return actual_run_state / relative


def _validate_source_plan(plan: dict[str, Any]) -> set[str]:
    schema = plan.get("schema")
    try:
        if schema in launcher.SUPPORTED_PLAN_SCHEMAS:
            launcher._validate_plan(plan)
            return {"base", "mtp"}
        if schema == dspark_overlay.PLAN_SCHEMA:
            digest = plan.get("plan_sha256")
            body = {key: value for key, value in plan.items() if key != "plan_sha256"}
            if (
                not isinstance(digest, str)
                or launcher.SHA256_RE.fullmatch(digest) is None
                or _sha256(_canonical(body)) != digest
                or plan.get("scope") != dspark_overlay.SCOPE
            ):
                raise AssemblyError("dSpark overlay plan digest or scope is invalid")
            return {"mtp"}
    except launcher.LaunchError as error:
        raise AssemblyError(str(error)) from error
    raise AssemblyError("quantization source plan schema is unsupported")


def _expected_modules(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    geometry = plan.get("source", {}).get("geometry")
    if not isinstance(geometry, dict):
        raise AssemblyError("quantization plan has no source geometry")
    layer_count = geometry.get("num_hidden_layers")
    expert_count = geometry.get("n_routed_experts")
    dspark_targets = geometry.get("dspark_target_layer_ids")
    if (
        isinstance(layer_count, bool)
        or not isinstance(layer_count, int)
        or layer_count <= 0
        or isinstance(expert_count, bool)
        or not isinstance(expert_count, int)
        or expert_count <= 0
        or not isinstance(dspark_targets, list)
    ):
        raise AssemblyError("quantization plan has invalid source geometry")
    expected = {}
    for namespace, layers in (
        ("base", range(layer_count)),
        ("mtp", range(len(dspark_targets))),
    ):
        for logical_layer in layers:
            block = (
                f"model.layers.{logical_layer}"
                if namespace == "base"
                else f"mtp.{logical_layer}"
            )
            for expert in range(expert_count):
                for projection, projection_name in PROJECTION_NAMES.items():
                    module = f"{block}.mlp.experts.{expert}.{projection_name}"
                    expected[module] = {
                        "block_namespace": namespace,
                        "logical_layer": logical_layer,
                        "expert": expert,
                        "projection": projection,
                    }
    return expected


def _projection_bits_from_tier_plans(
    plan: dict[str, Any],
    run_state: Path,
) -> tuple[dict[str, int], dict[tuple[str, int], dict[str, Any] | None]]:
    """Derive every canonical projection shape from signed per-layer plans."""

    expected = _expected_modules(plan)
    family_join = plan.get("ledger_provenance", {}).get("family_join")
    bits = plan.get("exl3", {}).get("bits")
    if (
        not isinstance(family_join, dict)
        or isinstance(bits, bool)
        or not isinstance(bits, int)
        or bits not in {2, 3}
    ):
        raise AssemblyError("quantization plan has invalid tier provenance")
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for module, identity in expected.items():
        key = (identity["block_namespace"], identity["logical_layer"])
        grouped.setdefault(key, {})[module] = identity
    tiers: dict[tuple[str, int], dict[str, Any] | None] = {}
    projection_bits: dict[str, int] = {}
    try:
        for key, block_expected in grouped.items():
            tier = block_audit._inline_mixed_block_contract(
                plan=plan,
                run_state=run_state,
                namespace=key[0],
                logical_layer=key[1],
                expected=block_expected,
                family_join=family_join,
            )
            tiers[key] = tier
            projection_bits.update(
                tier["bits_by_module"]
                if tier is not None
                else {module: bits for module in block_expected}
            )
    except block_audit.AuditError as error:
        raise AssemblyError(str(error)) from error
    return projection_bits, tiers


def _planned_generated_names(
    generated_tensors: Iterable[OutputTensor],
) -> set[str]:
    names = {tensor.name for tensor in generated_tensors}
    if len(names) == 0:
        raise AssemblyError("canonical artifact plan has no generated tensors")
    return names


def _load_checkpoint_manifest_index(
    checkpoint_root: Path,
    expected: dict[str, dict[str, Any]],
    *,
    allowed: set[str] | None = None,
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    if allowed is None:
        allowed = set(expected)
    if not set(expected) <= allowed:
        raise AssemblyError(
            "expected projection modules exceed the allowed source scope"
        )
    selected = {}
    for manifest_path in sorted(checkpoint_root.rglob("*.json")):
        if manifest_path.name.startswith("."):
            continue
        manifest = block_audit._read_object(
            manifest_path,
            "projection checkpoint manifest",
        )
        request = manifest.get("request")
        module = request.get("module") if isinstance(request, dict) else None
        if module not in allowed:
            raise AssemblyError(
                f"projection checkpoint contains an unexpected module: {module!r}"
            )
        if module not in expected:
            continue
        if module in selected:
            raise AssemblyError(f"duplicate projection checkpoint for {module}")
        selected[module] = (request, manifest)
    missing = sorted(set(expected) - set(selected))
    if missing:
        raise AssemblyError(
            f"projection checkpoint store has {len(selected)}/{len(expected)} results; "
            f"first missing module is {missing[0]}"
        )
    return selected


def _regular_tree_files(root: Path, label: str) -> set[str]:
    if not root.is_dir() or root.is_symlink():
        raise AssemblyError(f"{label} is not one regular directory")
    files: set[str] = set()
    for directory, names, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in names:
            path = directory_path / name
            if path.is_symlink() or not path.is_dir():
                raise AssemblyError(f"{label} contains a non-regular directory")
        for name in filenames:
            path = directory_path / name
            if path.is_symlink() or not path.is_file():
                raise AssemblyError(f"{label} contains a non-regular file")
            files.add(path.relative_to(root).as_posix())
    return files


def _validate_projection(
    *,
    module: str,
    identity: dict[str, Any],
    request: dict[str, Any],
    manifest: dict[str, Any],
    tensors: dict[str, Any],
    result: dict[str, Any],
    journal: dict[str, tuple[dict[str, Any], str]],
    plan: dict[str, Any],
    run_state: Path,
    family_join: dict[str, Any],
    hidden_size: int,
    intermediate_size: int,
    expected_bits: int,
) -> tuple[int, str, str, str, str]:
    if (
        isinstance(expected_bits, bool)
        or not isinstance(expected_bits, int)
        or expected_bits not in {2, 3}
    ):
        raise AssemblyError("projection source has an invalid EXL3 tier")
    if routed_expert_identity(module) != identity:
        raise AssemblyError(f"projection identity differs for {module}")
    quantizer_contract = request.get("quantizer_contract")
    if (
        request.get("family_join") != family_join
        or not isinstance(quantizer_contract, dict)
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
        raise AssemblyError(f"projection request numerics differ for {module}")

    encoded_bytes = block_audit._validate_tensors(
        tensors,
        projection=identity["projection"],
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        bits=expected_bits,
    )
    ledger = result.get("ledger_record")
    metrics = ledger.get("quantizer_metrics") if isinstance(ledger, dict) else None
    if (
        not isinstance(ledger, dict)
        or ledger.get("module") != module
        or any(ledger.get(key) != value for key, value in identity.items())
        or ledger.get("bits") != expected_bits
        or ledger.get("codebook") != "mcg"
        or ledger.get("encoded_bytes") != encoded_bytes
        or ledger.get("sample_count") != request.get("sample_count")
        or ledger.get("route_evidence") != request.get("route_evidence")
        or not isinstance(metrics, dict)
        or metrics.get("hessian_metric_status") != "ok"
        or metrics.get("hessian_regularization_sigma") != launcher.EXL3_SIGMA_REG
        or metrics.get("hessian_numerical_contract")
        != launcher.EXL3_HESSIAN_NUMERICAL_CONTRACT
        or metrics.get("hessian_transform_compute_dtype") != "torch.float64"
        or metrics.get("hessian_storage_dtype") != "torch.float32"
        or metrics.get("hessian_regularization_placement") != "before-fp64-congruence"
        or metrics.get("hessian_symmetry_restoration")
        != launcher.EXL3_HESSIAN_SYMMETRY_CONTRACT
        or result.get("quantizer_metrics") != metrics
        or result.get("proxy_error") != metrics.get("reported_metric_value")
        or ledger.get("provenance", {}).get("family_join") != family_join
    ):
        raise AssemblyError(f"projection ledger differs for {module}")
    for field in (
        "duration_seconds",
        "proxy_error",
    ):
        value = result.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise AssemblyError(f"projection result has invalid {field} for {module}")

    journal_record = journal.get(module)
    if journal_record is None or journal_record[0] != ledger:
        raise AssemblyError(f"projection journal differs for {module}")
    try:
        slot_id, assignment_digest = block_audit._execution_ownership_for_module(
            plan=plan,
            run_state=run_state,
            module=module,
            result=result,
            ledger=ledger,
        )
    except block_audit.AuditError as error:
        raise AssemblyError(str(error)) from error
    return (
        encoded_bytes,
        slot_id,
        assignment_digest,
        journal_record[1],
        manifest["manifest_sha256"],
    )


def assemble_projection_checkpoints(
    run_state: str | Path,
    *,
    generated_tensors: Iterable[OutputTensor],
    write_generated_tensor: Callable[[str, Any], None],
    expected_plan_sha256: str | None = None,
    checkpoint: Callable[[], None] | None = None,
    checkpoint_interval: int = 128,
) -> dict[str, Any]:
    """Validate and stream every durable projection into a hybrid writer."""

    if checkpoint_interval <= 0:
        raise ValueError("assembly checkpoint interval must be positive")
    run_state = Path(run_state).expanduser().resolve(strict=True)
    plan = block_audit._read_object(
        run_state / launcher.PLAN_FILENAME,
        "quantization plan",
    )
    try:
        launcher._validate_plan(plan)
    except launcher.LaunchError as error:
        raise AssemblyError(str(error)) from error
    _validate_planned_run_state_paths(plan)
    if expected_plan_sha256 is not None and plan["plan_sha256"] != expected_plan_sha256:
        raise AssemblyError("run-state plan differs from the raw artifact plan")

    expected = _expected_modules(plan)
    expected_generated = {
        f"{module}.{suffix}" for module in expected for suffix in PROJECTION_SUFFIXES
    }
    planned_generated = _planned_generated_names(generated_tensors)
    if planned_generated != expected_generated:
        missing = sorted(expected_generated - planned_generated)[:4]
        unexpected = sorted(planned_generated - expected_generated)[:4]
        raise AssemblyError(
            "canonical artifact generated tensor set differs from the run: "
            f"missing={missing} unexpected={unexpected}"
        )

    checkpoint_root = _resolve_checkpoint_root(plan, run_state)
    checkpoint_contract = plan.get("projection_checkpoint")
    if (
        not isinstance(checkpoint_contract, dict)
        or checkpoint_contract.get("contract")
        != launcher.PROJECTION_CHECKPOINT_CONTRACT
        or not checkpoint_root.is_dir()
        or checkpoint_root.is_symlink()
    ):
        raise AssemblyError("projection checkpoint root differs from the plan")
    family_join = plan.get("ledger_provenance", {}).get("family_join")
    if not isinstance(family_join, dict):
        raise AssemblyError("quantization plan has no family-join provenance")
    try:
        selected, retained, tier_contracts = block_audit._checkpoint_manifest_selection(
            checkpoint_root,
            plan=plan,
            run_state=run_state,
            expected=expected,
            family_join=family_join,
            ignore_unexpected=False,
        )
    except block_audit.AuditError as error:
        raise AssemblyError(str(error)) from error
    journal = block_audit._load_journal(run_state / launcher.ERROR_JOURNAL_FILENAME)
    if set(journal) != set(expected):
        raise AssemblyError("projection journal does not exactly cover the run")

    remote = plan.get("remote_workers")
    assignment_root = run_state / launcher.REMOTE_ASSIGNMENT_DIRNAME
    geometry = plan["source"]["geometry"]
    bits = plan.get("exl3", {}).get("bits")
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in {2, 3}:
        raise AssemblyError("quantization plan has an invalid EXL3 tier")
    hidden_size = geometry["hidden_size"]
    intermediate_size = geometry["moe_intermediate_size"]

    store = EXL3ProjectionCheckpointStore(checkpoint_root)
    expected_checkpoint_files = set()
    for entries in retained.values():
        for request, _manifest in entries:
            manifest_path, tensor_path = store._paths(request["request_sha256"])
            expected_checkpoint_files.update(
                {
                    manifest_path.relative_to(checkpoint_root).as_posix(),
                    tensor_path.relative_to(checkpoint_root).as_posix(),
                }
            )
    if (
        _regular_tree_files(
            checkpoint_root,
            "projection checkpoint store",
        )
        != expected_checkpoint_files
    ):
        raise AssemblyError(
            "projection checkpoint store contains orphaned or unexpected files"
        )
    expected_assignment_files = {
        (f"{digest[:2]}/{digest[2:4]}/{digest}.json")
        for module in expected
        for digest in (hashlib.sha256(module.encode("utf-8")).hexdigest(),)
    }
    actual_assignment_files = (
        _regular_tree_files(assignment_root, "dynamic projection assignment store")
        if assignment_root.exists()
        else set()
    )
    if actual_assignment_files != (
        expected_assignment_files if isinstance(remote, dict) else set()
    ):
        raise AssemblyError(
            "dynamic projection assignment store does not exactly cover the run"
        )
    checkpoint_digests = []
    assignment_digests = []
    journal_digests = []
    ownership: dict[str, int] = {}
    encoded_bytes = 0
    projection_count = 0
    family_count = 0
    current_family: tuple[str, int, int] | None = None
    current_family_records: list[dict[str, Any]] = []
    pending = 0
    for module in sorted(expected):
        identity = expected[module]
        family_identity = (
            identity["block_namespace"],
            identity["logical_layer"],
            identity["expert"],
        )
        if current_family is not None and family_identity != current_family:
            families = derive_family_records(current_family_records)
            if len(families) != 1:
                raise AssemblyError(
                    f"projection ledger does not close expert family {current_family}"
                )
            family_count += 1
            current_family_records = []
        current_family = family_identity
        request, manifest = selected[module]
        tier = tier_contracts[(identity["block_namespace"], identity["logical_layer"])]
        expected_bits = tier["bits_by_module"][module] if tier is not None else bits
        loaded = store.load(request)
        if loaded is None:
            raise AssemblyError(f"projection checkpoint disappeared for {module}")
        tensors, result = loaded
        (
            projection_bytes,
            slot_id,
            assignment_digest,
            journal_digest,
            checkpoint_digest,
        ) = _validate_projection(
            module=module,
            identity=identity,
            request=request,
            manifest=manifest,
            tensors=tensors,
            result=result,
            journal=journal,
            plan=plan,
            run_state=run_state,
            family_join=family_join,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            expected_bits=expected_bits,
        )
        for suffix in PROJECTION_SUFFIXES:
            write_generated_tensor(f"{module}.{suffix}", tensors[suffix])
        encoded_bytes += projection_bytes
        ownership[slot_id] = ownership.get(slot_id, 0) + 1
        current_family_records.append(result["ledger_record"])
        projection_count += 1
        checkpoint_digests.append(
            (module, checkpoint_digest, manifest["tensor_sha256"])
        )
        assignment_digests.append((module, assignment_digest))
        journal_digests.append((module, journal_digest))
        pending += 1
        if checkpoint is not None and pending >= checkpoint_interval:
            checkpoint()
            pending = 0
    if checkpoint is not None:
        checkpoint()

    if current_family is not None:
        families = derive_family_records(current_family_records)
        if len(families) != 1:
            raise AssemblyError(
                f"projection ledger does not close expert family {current_family}"
            )
        family_count += 1
    expected_family_count = len(expected) // 3
    if family_count != expected_family_count:
        raise AssemblyError(
            f"projection ledger has {family_count}/{expected_family_count} families"
        )
    report = {
        "schema": PROJECTION_ASSEMBLY_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "materialized_run_state": os.fspath(run_state),
        "planned_run_state": plan["run_state_dir"],
        "expert_tensor_layout": EXPERT_TENSOR_LAYOUT_GPTQMODEL,
        "retained_tensor_layout": "source_checkpoint_native_byte_exact",
        "projection_count": projection_count,
        "expert_family_count": family_count,
        "generated_tensor_count": len(expected_generated),
        "encoded_bytes": encoded_bytes,
        "ownership": dict(sorted(ownership.items())),
        "content": {
            "checkpoint_records_sha256": _sha256(_canonical(checkpoint_digests)),
            "assignment_records_sha256": _sha256(_canonical(assignment_digests)),
            "journal_records_sha256": _sha256(_canonical(journal_digests)),
        },
    }
    return _bound_report(report)


def _assemble_projection_source(
    run_state: Path,
    *,
    namespaces: set[str],
    canonical_expected: dict[str, dict[str, Any]],
    write_generated_tensor: Callable[[str, Any], None],
    checkpoint: Callable[[], None] | None,
    checkpoint_interval: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_state = run_state.expanduser().resolve(strict=True)
    plan = block_audit._read_object(
        run_state / launcher.PLAN_FILENAME,
        "quantization source plan",
    )
    available_namespaces = _validate_source_plan(plan)
    if not namespaces or not namespaces <= available_namespaces:
        raise AssemblyError("projection namespaces exceed the source plan scope")
    _validate_planned_run_state_paths(plan)
    source_expected_all = {
        module: identity
        for module, identity in _expected_modules(plan).items()
        if identity["block_namespace"] in available_namespaces
    }
    expected = {
        module: identity
        for module, identity in source_expected_all.items()
        if identity["block_namespace"] in namespaces
    }
    if not expected or any(
        canonical_expected.get(module) != identity
        for module, identity in expected.items()
    ):
        raise AssemblyError(
            "projection source geometry differs from the canonical model"
        )

    checkpoint_root = _resolve_checkpoint_root(plan, run_state)
    checkpoint_contract = plan.get("projection_checkpoint")
    if (
        not isinstance(checkpoint_contract, dict)
        or checkpoint_contract.get("contract")
        != launcher.PROJECTION_CHECKPOINT_CONTRACT
        or not checkpoint_root.is_dir()
        or checkpoint_root.is_symlink()
    ):
        raise AssemblyError("projection checkpoint root differs from the source plan")
    family_join = plan.get("ledger_provenance", {}).get("family_join")
    if not isinstance(family_join, dict):
        raise AssemblyError("quantization plan has no family-join provenance")
    try:
        selected_all, retained_all, tier_contracts = (
            block_audit._checkpoint_manifest_selection(
                checkpoint_root,
                plan=plan,
                run_state=run_state,
                expected=source_expected_all,
                family_join=family_join,
                ignore_unexpected=False,
            )
        )
    except block_audit.AuditError as error:
        raise AssemblyError(str(error)) from error
    selected = {module: selected_all[module] for module in expected}
    store = EXL3ProjectionCheckpointStore(checkpoint_root)
    all_checkpoint_files: set[str] = set()
    for entries in retained_all.values():
        for request, _manifest in entries:
            expected_manifest, expected_tensor = store._paths(request["request_sha256"])
            all_checkpoint_files.update(
                {
                    expected_manifest.relative_to(checkpoint_root).as_posix(),
                    expected_tensor.relative_to(checkpoint_root).as_posix(),
                }
            )
    if (
        _regular_tree_files(
            checkpoint_root,
            "projection checkpoint store",
        )
        != all_checkpoint_files
    ):
        raise AssemblyError(
            "projection checkpoint store contains orphaned or unexpected files"
        )

    journal = block_audit._load_journal(run_state / launcher.ERROR_JOURNAL_FILENAME)
    if not set(expected) <= set(journal) or not set(journal) <= set(
        source_expected_all
    ):
        raise AssemblyError("projection journal differs from the selected source scope")
    remote = plan.get("remote_workers")
    assignment_root = run_state / launcher.REMOTE_ASSIGNMENT_DIRNAME
    actual_assignments = (
        _regular_tree_files(assignment_root, "dynamic projection assignment store")
        if assignment_root.exists()
        else set()
    )

    def assignment_path(module: str) -> str:
        digest = hashlib.sha256(module.encode("utf-8")).hexdigest()
        return f"{digest[:2]}/{digest[2:4]}/{digest}.json"

    expected_assignments = {assignment_path(module) for module in expected}
    allowed_assignments = {assignment_path(module) for module in source_expected_all}
    if (
        isinstance(remote, dict)
        and (
            not expected_assignments <= actual_assignments
            or not actual_assignments <= allowed_assignments
        )
        or remote is None
        and actual_assignments
    ):
        raise AssemblyError("dynamic assignments differ from the selected source scope")
    geometry = plan["source"]["geometry"]
    bits = plan.get("exl3", {}).get("bits")
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in {2, 3}:
        raise AssemblyError("projection source has an invalid EXL3 tier")
    hidden_size = geometry["hidden_size"]
    intermediate_size = geometry["moe_intermediate_size"]

    checkpoint_digests = []
    assignment_digests = []
    journal_digests = []
    ownership: dict[str, int] = {}
    projection_records: list[dict[str, Any]] = []
    encoded_bytes = 0
    pending = 0
    family_count = 0
    current_family: tuple[str, int, int] | None = None
    current_records: list[dict[str, Any]] = []
    for module in sorted(expected):
        identity = expected[module]
        family = (
            identity["block_namespace"],
            identity["logical_layer"],
            identity["expert"],
        )
        if current_family is not None and family != current_family:
            if len(derive_family_records(current_records)) != 1:
                raise AssemblyError(
                    f"projection ledger does not close expert family {current_family}"
                )
            family_count += 1
            current_records = []
        current_family = family
        request, manifest = selected[module]
        tier = tier_contracts[(identity["block_namespace"], identity["logical_layer"])]
        expected_bits = tier["bits_by_module"][module] if tier is not None else bits
        loaded = store.load(request)
        if loaded is None:
            raise AssemblyError(f"projection checkpoint disappeared for {module}")
        tensors, result = loaded
        (
            projection_bytes,
            slot_id,
            assignment_digest,
            journal_digest,
            checkpoint_digest,
        ) = _validate_projection(
            module=module,
            identity=identity,
            request=request,
            manifest=manifest,
            tensors=tensors,
            result=result,
            journal=journal,
            plan=plan,
            run_state=run_state,
            family_join=family_join,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            expected_bits=expected_bits,
        )
        for suffix in PROJECTION_SUFFIXES:
            write_generated_tensor(f"{module}.{suffix}", tensors[suffix])
        record = result["ledger_record"]
        projection_records.append(record)
        current_records.append(record)
        encoded_bytes += projection_bytes
        ownership[slot_id] = ownership.get(slot_id, 0) + 1
        checkpoint_digests.append(
            (module, checkpoint_digest, manifest["tensor_sha256"])
        )
        assignment_digests.append((module, assignment_digest))
        journal_digests.append((module, journal_digest))
        pending += 1
        if checkpoint is not None and pending >= checkpoint_interval:
            checkpoint()
            pending = 0
    if current_family is not None:
        if len(derive_family_records(current_records)) != 1:
            raise AssemblyError(
                f"projection ledger does not close expert family {current_family}"
            )
        family_count += 1
    if checkpoint is not None:
        checkpoint()
    if family_count != len(expected) // 3:
        raise AssemblyError("projection source has incomplete expert families")
    report = {
        "schema": PROJECTION_ASSEMBLY_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "namespaces": sorted(namespaces),
        "materialized_run_state": os.fspath(run_state),
        "planned_run_state": plan["run_state_dir"],
        "expert_tensor_layout": EXPERT_TENSOR_LAYOUT_GPTQMODEL,
        "retained_tensor_layout": "source_checkpoint_native_byte_exact",
        "projection_count": len(expected),
        "expert_family_count": family_count,
        "generated_tensor_count": len(expected) * len(PROJECTION_SUFFIXES),
        "encoded_bytes": encoded_bytes,
        "ownership": dict(sorted(ownership.items())),
        "excluded_known_state": {
            "checkpoint_count": sum(
                len(entries)
                for module, entries in retained_all.items()
                if module not in expected
            ),
            "journal_count": len(set(journal) - set(expected)),
            "assignment_count": len(actual_assignments - expected_assignments),
        },
        "content": {
            "checkpoint_records_sha256": _sha256(_canonical(checkpoint_digests)),
            "assignment_records_sha256": _sha256(_canonical(assignment_digests)),
            "journal_records_sha256": _sha256(_canonical(journal_digests)),
        },
    }
    return _bound_report(report), projection_records


def assemble_composite_projection_checkpoints(
    *,
    base_run_state: str | Path,
    mtp_run_state: str | Path,
    generated_tensors: Iterable[OutputTensor],
    write_generated_tensor: Callable[[str, Any], None],
    checkpoint: Callable[[], None] | None = None,
    checkpoint_interval: int = 128,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Stream base and replacement MTP checkpoints into one tensor inventory."""

    if checkpoint_interval <= 0:
        raise ValueError("assembly checkpoint interval must be positive")
    base_path = Path(base_run_state).expanduser().resolve(strict=True)
    mtp_path = Path(mtp_run_state).expanduser().resolve(strict=True)
    if base_path == mtp_path:
        raise AssemblyError("base and MTP projection sources must be distinct")
    base_plan = block_audit._read_object(
        base_path / launcher.PLAN_FILENAME,
        "base quantization plan",
    )
    mtp_plan = block_audit._read_object(
        mtp_path / launcher.PLAN_FILENAME,
        "MTP quantization plan",
    )
    if _validate_source_plan(base_plan) != {"base", "mtp"}:
        raise AssemblyError("composite base source is not a full quantization plan")
    if _validate_source_plan(mtp_plan) != {"mtp"}:
        raise AssemblyError("composite MTP source is not an overlay plan")
    if _portable_source_identity(base_plan["source"]) != _portable_source_identity(
        mtp_plan["source"]
    ):
        raise AssemblyError("base and MTP projection sources describe different models")
    parent = mtp_plan.get("target_parent")
    if (
        not isinstance(parent, dict)
        or parent.get("plan_sha256") != base_plan["plan_sha256"]
    ):
        raise AssemblyError("MTP projection source is not bound to the base plan")
    canonical_expected = _expected_modules(base_plan)
    expected_generated = {
        f"{module}.{suffix}"
        for module in canonical_expected
        for suffix in PROJECTION_SUFFIXES
    }
    planned_generated = _planned_generated_names(generated_tensors)
    if planned_generated != expected_generated:
        raise AssemblyError(
            "canonical generated tensor set differs from composite sources"
        )

    base_report, base_records = _assemble_projection_source(
        base_path,
        namespaces={"base"},
        canonical_expected=canonical_expected,
        write_generated_tensor=write_generated_tensor,
        checkpoint=checkpoint,
        checkpoint_interval=checkpoint_interval,
    )
    mtp_report, mtp_records = _assemble_projection_source(
        mtp_path,
        namespaces={"mtp"},
        canonical_expected=canonical_expected,
        write_generated_tensor=write_generated_tensor,
        checkpoint=checkpoint,
        checkpoint_interval=checkpoint_interval,
    )
    sources = {"base": base_report, "mtp": mtp_report}
    ownership: dict[str, int] = {}
    for source in sources.values():
        for slot, count in source["ownership"].items():
            ownership[slot] = ownership.get(slot, 0) + count
    records = base_records + mtp_records
    report = {
        "schema": COMPOSITE_PROJECTION_ASSEMBLY_SCHEMA,
        "base_plan_sha256": base_plan["plan_sha256"],
        "mtp_plan_sha256": mtp_plan["plan_sha256"],
        "sources": sources,
        "expert_tensor_layout": EXPERT_TENSOR_LAYOUT_GPTQMODEL,
        "retained_tensor_layout": "source_checkpoint_native_byte_exact",
        "projection_count": sum(
            source["projection_count"] for source in sources.values()
        ),
        "expert_family_count": sum(
            source["expert_family_count"] for source in sources.values()
        ),
        "generated_tensor_count": sum(
            source["generated_tensor_count"] for source in sources.values()
        ),
        "encoded_bytes": sum(source["encoded_bytes"] for source in sources.values()),
        "ownership": dict(sorted(ownership.items())),
        "content": {
            "source_reports_sha256": _sha256(_canonical(sources)),
            "projection_records_sha256": _sha256(
                _canonical(
                    [
                        record.get("record_sha256") or _sha256(_canonical(record))
                        for record in records
                    ]
                )
            ),
        },
    }
    if report["projection_count"] != len(canonical_expected):
        raise AssemblyError("composite projection sources do not close the model")
    return _bound_report(report), records


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssemblyError(f"JSON file is not an object: {path}")
    return value


def validate_block_audit_set(
    report_dir: str | Path,
    plan: dict[str, Any],
    *,
    namespaces: set[str] | None = None,
    run_state: str | Path | None = None,
) -> dict[str, Any]:
    """Bind every independent complete-block report before canonical assembly."""

    report_dir = Path(report_dir).expanduser()
    if not report_dir.is_dir() or report_dir.is_symlink():
        raise AssemblyError("independent block-audit path is not one regular directory")
    report_dir = report_dir.resolve(strict=True)
    available_namespaces = _validate_source_plan(plan)
    if namespaces is None:
        namespaces = set(available_namespaces)
    if not namespaces or not namespaces <= available_namespaces:
        raise AssemblyError("block-audit namespaces exceed the quantization plan scope")
    geometry = plan["source"]["geometry"]
    bits = plan.get("exl3", {}).get("bits")
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in {2, 3}:
        raise AssemblyError("block-audit plan has an invalid EXL3 tier")
    base_blocks = geometry["num_hidden_layers"]
    mtp_blocks = len(geometry["dspark_target_layer_ids"])
    expert_count = geometry["n_routed_experts"]
    hidden_size = geometry["hidden_size"]
    intermediate_size = geometry["moe_intermediate_size"]
    projections_per_block = expert_count * 3
    materialized_run_state = (
        Path(plan["run_state_dir"] if run_state is None else run_state)
        .expanduser()
        .resolve(strict=True)
    )
    projection_bits, tier_contracts = _projection_bits_from_tier_plans(
        plan,
        materialized_run_state,
    )
    expected_modules = _expected_modules(plan)
    expected_blocks = [
        (namespace, logical_layer)
        for namespace, count in (("base", base_blocks), ("mtp", mtp_blocks))
        if namespace in namespaces
        for logical_layer in range(count)
    ]
    reports = []
    for ordinal, (namespace, logical_layer) in enumerate(expected_blocks, 1):
        filename = f"{namespace}-layer-{logical_layer}-projection-audit.json"
        path = report_dir / filename
        if not path.is_file() or path.is_symlink():
            raise AssemblyError(
                f"independent block audit is not one regular file: {filename}"
            )
        payload = block_audit._stable_bytes(path)
        try:
            report = json.loads(payload)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise AssemblyError(
                f"cannot read independent block audit: {filename}"
            ) from error
        if not isinstance(report, dict):
            raise AssemblyError(
                f"independent block audit is not a JSON object: {filename}"
            )
        report_body = {
            key: value for key, value in report.items() if key != "report_sha256"
        }
        digest = report.get("report_sha256")
        if (
            not isinstance(digest, str)
            or launcher.SHA256_RE.fullmatch(digest) is None
            or digest != _sha256(_canonical(report_body))
        ):
            raise AssemblyError(
                f"independent block audit failed its content digest: {filename}"
            )
        content = report.get("content")
        report_geometry = report.get("geometry")
        route_coverage = report.get("route_coverage")
        ownership = report.get("ownership")
        zero_count = (
            route_coverage.get("zero_count")
            if isinstance(route_coverage, dict)
            else None
        )
        augmented_count = (
            route_coverage.get("augmented_expert_count", 0)
            if isinstance(route_coverage, dict)
            else None
        )
        augmented_experts = (
            route_coverage.get("augmented_experts", [])
            if isinstance(route_coverage, dict)
            else None
        )
        router_augmented_samples = (
            route_coverage.get("router_augmented_samples")
            if isinstance(route_coverage, dict)
            else None
        )
        identity_calibration = (
            route_coverage.get("identity_calibration")
            if isinstance(route_coverage, dict)
            else None
        )
        identity_expert_count = (
            identity_calibration.get("expert_count")
            if isinstance(identity_calibration, dict)
            else None
        )
        identity_effective_count = (
            identity_calibration.get("effective_count")
            if isinstance(identity_calibration, dict)
            else None
        )
        router_sample_summary_valid = (
            isinstance(router_augmented_samples, dict)
            and all(
                isinstance(router_augmented_samples.get(field), (int, float))
                and not isinstance(router_augmented_samples.get(field), bool)
                and math.isfinite(router_augmented_samples[field])
                and router_augmented_samples[field] >= 0
                for field in ("min", "mean", "max")
            )
            and router_augmented_samples["min"]
            <= router_augmented_samples["mean"]
            <= router_augmented_samples["max"]
        )
        recovery_coverage_valid = (
            isinstance(zero_count, int)
            and not isinstance(zero_count, bool)
            and 0 <= zero_count <= expert_count
            and isinstance(augmented_count, int)
            and not isinstance(augmented_count, bool)
            and zero_count <= augmented_count <= expert_count
            and isinstance(augmented_experts, list)
            and len(augmented_experts) == augmented_count
            and len(set(augmented_experts)) == augmented_count
            and all(
                isinstance(expert, int)
                and not isinstance(expert, bool)
                and 0 <= expert < expert_count
                for expert in augmented_experts
            )
            and (
                (
                    augmented_count == 0
                    and zero_count == 0
                    and router_augmented_samples is None
                    and identity_calibration is None
                )
                or (
                    augmented_count > 0
                    and router_sample_summary_valid
                    and isinstance(identity_expert_count, int)
                    and not isinstance(identity_expert_count, bool)
                    and 0 <= identity_expert_count <= augmented_count
                    and isinstance(identity_effective_count, int)
                    and not isinstance(identity_effective_count, bool)
                    and (
                        (identity_expert_count == 0 and identity_effective_count == 0)
                        or (
                            identity_expert_count > 0
                            and identity_expert_count
                            <= identity_effective_count
                            <= identity_expert_count
                            * block_audit.ZERO_ROUTE_RECOVERY_TARGET_SAMPLE_COUNT
                        )
                    )
                    and (
                        router_augmented_samples["max"] != 0
                        or identity_expert_count == augmented_count
                    )
                )
            )
        )
        tier = tier_contracts[(namespace, logical_layer)]
        expected_geometry = block_audit._block_report_geometry(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            expert_count=expert_count,
            bits=bits,
            tier=tier,
        )
        block_modules = {
            module: identity
            for module, identity in expected_modules.items()
            if identity["block_namespace"] == namespace
            and identity["logical_layer"] == logical_layer
        }
        expected_encoded_bytes = sum(
            (
                (hidden_size // 16)
                * (intermediate_size // 16)
                * 32
                * projection_bits[module]
                + (hidden_size + intermediate_size) * 2
                + 4
            )
            for module in block_modules
        )
        if (
            report.get("schema") != block_audit.AUDIT_SCHEMA
            or report.get("status") != "complete"
            or report.get("plan_sha256") != plan["plan_sha256"]
            or report.get("block_namespace") != namespace
            or report.get("logical_layer") != logical_layer
            or report_geometry != expected_geometry
            or report.get("projection_count") != projections_per_block
            or report.get("expected_projection_count") != projections_per_block
            or report.get("missing_projection_count") != 0
            or report.get("first_missing_projection") is not None
            or report.get("complete_expert_families") != expert_count
            or report.get("encoded_bytes") != expected_encoded_bytes
            or not isinstance(content, dict)
            or any(
                launcher.SHA256_RE.fullmatch(content.get(field, "")) is None
                for field in (
                    "checkpoint_records_sha256",
                    "assignment_records_sha256",
                    "journal_records_sha256",
                )
            )
            or not isinstance(route_coverage, dict)
            or route_coverage.get("experts") != expert_count
            or not recovery_coverage_valid
            or isinstance(route_coverage.get("min"), bool)
            or not isinstance(route_coverage.get("min"), int)
            or route_coverage["min"] < 0
            or (route_coverage["min"] == 0) != (zero_count > 0)
            or isinstance(route_coverage.get("max"), bool)
            or not isinstance(route_coverage.get("max"), int)
            or route_coverage["max"] < route_coverage["min"]
            or isinstance(route_coverage.get("mean"), bool)
            or not isinstance(route_coverage.get("mean"), (int, float))
            or not math.isfinite(route_coverage["mean"])
            or not route_coverage["min"]
            <= route_coverage["mean"]
            <= route_coverage["max"]
            or not isinstance(ownership, dict)
            or not ownership
            or any(
                isinstance(count, bool) or not isinstance(count, int) or count <= 0
                for count in ownership.values()
            )
            or sum(ownership.values()) != projections_per_block
        ):
            raise AssemblyError(
                f"independent block audit failed its complete contract: {filename}"
            )
        reports.append(
            {
                "ordinal": ordinal,
                "block_namespace": namespace,
                "logical_layer": logical_layer,
                "filename": filename,
                "file_sha256": _sha256(payload),
                "report_sha256": digest,
            }
        )
    result = {
        "schema": BLOCK_AUDIT_SET_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "base_block_count": base_blocks if "base" in namespaces else 0,
        "mtp_block_count": mtp_blocks if "mtp" in namespaces else 0,
        "block_count": len(expected_blocks),
        "projections_per_block": projections_per_block,
        "projection_count": len(expected_blocks) * projections_per_block,
        "expert_family_count": len(expected_blocks) * expert_count,
        "reports": reports,
        "reports_sha256": _sha256(_canonical(reports)),
    }
    return _bound_report(result)


def validate_composite_block_audit_set(
    *,
    base_report_dir: str | Path,
    base_plan: dict[str, Any],
    mtp_report_dir: str | Path,
    mtp_plan: dict[str, Any],
) -> dict[str, Any]:
    """Bind the completed parent base and replacement MTP audit sets."""

    if _portable_source_identity(
        base_plan.get("source", {})
    ) != _portable_source_identity(mtp_plan.get("source", {})):
        raise AssemblyError("base and MTP audit plans describe different source models")
    parent = mtp_plan.get("target_parent")
    if not isinstance(parent, dict) or parent.get("plan_sha256") != base_plan.get(
        "plan_sha256"
    ):
        raise AssemblyError("MTP audit plan is not bound to the base parent plan")
    base = validate_block_audit_set(
        base_report_dir,
        base_plan,
        namespaces={"base"},
    )
    mtp = validate_block_audit_set(
        mtp_report_dir,
        mtp_plan,
        namespaces={"mtp"},
    )
    result = {
        "schema": COMPOSITE_BLOCK_AUDIT_SET_SCHEMA,
        "base": base,
        "mtp": mtp,
        "base_plan_sha256": base_plan["plan_sha256"],
        "mtp_plan_sha256": mtp_plan["plan_sha256"],
        "base_block_count": base["base_block_count"],
        "mtp_block_count": mtp["mtp_block_count"],
        "block_count": base["block_count"] + mtp["block_count"],
        "projections_per_block": base["projections_per_block"],
        "projection_count": base["projection_count"] + mtp["projection_count"],
        "expert_family_count": (
            base["expert_family_count"] + mtp["expert_family_count"]
        ),
    }
    if mtp["projections_per_block"] != base["projections_per_block"]:
        raise AssemblyError("base and MTP audit geometry differs")
    return _bound_report(result)


def canonical_quant_config(
    plan: Any,
    raw_quant_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate raw base metadata and close the disjoint MTP save overlay.

    GPTQModel computes ``tensor_storage`` before applying DeepSeek V4's
    out-of-tree MTP state overlay. Current raw exports therefore contain every
    packed MTP tensor but can describe only the base modules. The canonical
    artifact may add precisely the plan-derived MTP entries; partial base or
    partial MTP metadata remains a hard failure.
    """

    raw_storage = raw_quant_config.get("tensor_storage")
    if not isinstance(raw_storage, dict):
        raise AssemblyError("raw GPTQModel quantization config has no tensor_storage")
    base_bits = getattr(plan, "exl3_bits", 2)
    if (
        isinstance(base_bits, bool)
        or not isinstance(base_bits, int)
        or base_bits not in {2, 3}
        or raw_quant_config.get("bits") != base_bits
    ):
        raise AssemblyError(
            "raw GPTQModel top-level bits must identify the integer base tier"
        )
    expected = gptqmodel_tensor_storage_for_plan(plan)
    base_names = {name for name in expected if name.startswith("model.layers.")}
    mtp_names = set(expected) - base_names
    raw_names = set(raw_storage)
    if raw_names not in (base_names, set(expected)):
        missing = sorted(set(expected) - raw_names)[:4]
        unexpected = sorted(raw_names - set(expected))[:4]
        raise AssemblyError(
            "raw GPTQModel tensor_storage is neither exact base-only nor complete: "
            f"missing={missing} unexpected={unexpected}"
        )
    for name in raw_names:
        if raw_storage[name] != expected[name]:
            raise AssemblyError(
                f"raw GPTQModel tensor_storage differs from packed geometry for {name}"
            )
    added = sorted(set(expected) - raw_names)
    if added and set(added) != mtp_names:
        raise AssemblyError(
            "canonical tensor_storage may add only the complete MTP overlay"
        )
    canonical = deepcopy(raw_quant_config)
    canonical["bits"] = base_bits
    canonical["tensor_storage"] = expected
    report = _bound_report(
        {
            "schema": QUANT_CONFIG_ASSEMBLY_SCHEMA,
            "raw_module_count": len(raw_names),
            "canonical_module_count": len(expected),
            "added_mtp_module_count": len(added),
            "added_mtp_modules_sha256": _sha256(_canonical(added)),
            "raw_quant_config_sha256": _sha256(_canonical(raw_quant_config)),
            "canonical_quant_config_sha256": _sha256(_canonical(canonical)),
        }
    )
    return canonical, report


def composite_ledger_provenance(
    base_plan: dict[str, Any],
    mtp_plan: dict[str, Any],
) -> dict[str, Any]:
    if _portable_source_identity(
        base_plan.get("source", {})
    ) != _portable_source_identity(mtp_plan.get("source", {})):
        raise AssemblyError("composite ledger plans describe different source models")
    parent = mtp_plan.get("target_parent")
    if not isinstance(parent, dict) or parent.get("plan_sha256") != base_plan.get(
        "plan_sha256"
    ):
        raise AssemblyError("composite MTP ledger is not bound to the base plan")
    base_provenance = base_plan.get("ledger_provenance")
    mtp_provenance = mtp_plan.get("ledger_provenance")
    if (
        not isinstance(base_provenance, dict)
        or not isinstance(base_provenance.get("family_join"), dict)
        or not isinstance(base_provenance.get("run"), dict)
        or not isinstance(mtp_provenance, dict)
        or not isinstance(mtp_provenance.get("family_join"), dict)
        or not isinstance(mtp_provenance.get("run"), dict)
    ):
        raise AssemblyError("composite projection source has incomplete provenance")
    sources = {
        "base": {
            "plan_sha256": base_plan["plan_sha256"],
            "namespaces": ["base"],
            "family_join": deepcopy(base_provenance["family_join"]),
        },
        "mtp": {
            "plan_sha256": mtp_plan["plan_sha256"],
            "namespaces": ["mtp"],
            "family_join": deepcopy(mtp_provenance["family_join"]),
        },
    }
    return {
        "family_join": deepcopy(base_provenance["family_join"]),
        "run": deepcopy(base_provenance["run"]),
        "projection_sources": sources,
        "projection_sources_sha256": _sha256(_canonical(sources)),
    }


def canonical_composite_quant_config(
    artifact_plan: Any,
    *,
    base_plan: dict[str, Any],
    mtp_plan: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    bits = base_plan.get("exl3", {}).get("bits")
    if (
        isinstance(bits, bool)
        or not isinstance(bits, int)
        or bits not in {2, 3}
        or mtp_plan.get("exl3", {}).get("bits") != bits
    ):
        raise AssemblyError("composite base and MTP tiers differ")
    provenance = composite_ledger_provenance(base_plan, mtp_plan)
    storage = gptqmodel_tensor_storage_for_plan(artifact_plan)
    quant = {
        "quant_method": "exl3",
        "method": "exl3",
        "format": "exl3",
        "checkpoint_format": "exl3",
        "bits": bits,
        "codebook": "mcg",
        "out_scales": "auto",
        "group_size": -1,
        "desc_act": False,
        "lm_head": False,
        "pack_dtype": "int32",
        "module_include": [launcher.BASE_EXPERT_PATTERN],
        "tensor_storage": storage,
        "meta": {
            "fallback": None,
            "ds41rt_error_ledger": provenance,
        },
    }
    report = {
        "schema": COMPOSITE_QUANT_CONFIG_ASSEMBLY_SCHEMA,
        "base_plan_sha256": base_plan["plan_sha256"],
        "mtp_plan_sha256": mtp_plan["plan_sha256"],
        "canonical_module_count": len(storage),
        "tensor_storage_sha256": _sha256(_canonical(storage)),
        "ledger_provenance_sha256": _sha256(_canonical(provenance)),
        "canonical_quant_config_sha256": _sha256(_canonical(quant)),
    }
    return quant, _bound_report(report)


def _publish_stage(
    stage: Path,
    output: Path,
    plan: dict[str, Any],
    *,
    mtp_replay_batches: int,
    model_config: dict[str, Any],
) -> None:
    launcher.atomic_json(stage / launcher.PLAN_FILENAME, plan)
    manifest = launcher.write_artifact_manifest(stage, plan)
    run = launcher._bound_record(
        {
            "schema": launcher.RUN_SCHEMA,
            "status": "complete",
            "plan_sha256": plan["plan_sha256"],
            "artifact_manifest_sha256": manifest["manifest_sha256"],
            "mtp_replay_batches": mtp_replay_batches,
        },
        "run_sha256",
    )
    launcher.atomic_json(stage / launcher.RUN_FILENAME, run)
    launcher.validate_published_artifact(
        stage,
        plan,
        verify_file_hashes=False,
    )
    validate_gptqmodel_publication(
        stage,
        model_config,
        verify_all_hashes=False,
        require_canonical=True,
    )
    os.replace(stage, output)
    descriptor = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_publication_plan(
    raw_plan: dict[str, Any],
    *,
    output: Path,
    assembly: dict[str, Any],
) -> dict[str, Any]:
    body = {
        key: deepcopy(value) for key, value in raw_plan.items() if key != "plan_sha256"
    }
    body["output"] = os.fspath(output)
    body["canonical_assembly"] = {
        "schema": ASSEMBLY_SCHEMA,
        "filename": ASSEMBLY_FILENAME,
        "report_sha256": assembly["report_sha256"],
        "source_plan_sha256": raw_plan["plan_sha256"],
        "source_artifact_manifest_sha256": assembly["source_artifact_manifest_sha256"],
    }
    return launcher._bound_record(body, "plan_sha256")


def _validate_finished_assembly_for_resume(
    assembly: dict[str, Any],
    *,
    raw_plan: dict[str, Any],
    raw_run: dict[str, Any],
    raw_manifest: dict[str, Any],
    materialized_source: dict[str, Any],
    block_audit_set: dict[str, Any],
    recipe: str,
) -> None:
    """Bind a post-writer/pre-rename resume to the same immutable inputs."""

    body = {key: value for key, value in assembly.items() if key != "report_sha256"}
    projection = assembly.get("projection_assembly")
    if (
        assembly.get("report_sha256") != _sha256(_canonical(body))
        or assembly.get("schema") != ASSEMBLY_SCHEMA
        or assembly.get("recipe") != recipe
        or assembly.get("source_plan_sha256") != raw_plan["plan_sha256"]
        or assembly.get("source_run_sha256") != raw_run["run_sha256"]
        or assembly.get("source_artifact_manifest_sha256")
        != raw_manifest["manifest_sha256"]
        or assembly.get("source_snapshot")
        != _portable_source_identity(materialized_source)
        or assembly.get("block_audit_set") != block_audit_set
        or not isinstance(projection, dict)
        or projection.get("plan_sha256") != raw_plan["plan_sha256"]
    ):
        raise AssemblyError(
            "finished canonical work directory is bound to different inputs"
        )


def _composite_projection_sources(
    *,
    base_plan: dict[str, Any],
    mtp_plan: dict[str, Any],
    mtp_manifest: dict[str, Any],
    mtp_run: dict[str, Any],
) -> dict[str, Any]:
    base_family = base_plan["ledger_provenance"]["family_join"]
    mtp_family = mtp_plan["ledger_provenance"]["family_join"]
    return {
        "base": {
            "plan_sha256": base_plan["plan_sha256"],
            "family_join_sha256": _sha256(_canonical(base_family)),
        },
        "mtp": {
            "plan_sha256": mtp_plan["plan_sha256"],
            "family_join_sha256": _sha256(_canonical(mtp_family)),
            "overlay_sha256": mtp_manifest["overlay_sha256"],
            "overlay_run_sha256": mtp_run["run_sha256"],
        },
    }


def _canonical_composite_publication_plan(
    base_plan: dict[str, Any],
    *,
    output: Path,
    assembly: dict[str, Any],
    ledger_provenance: dict[str, Any],
) -> dict[str, Any]:
    body = {
        key: deepcopy(value) for key, value in base_plan.items() if key != "plan_sha256"
    }
    sources = assembly["projection_sources"]
    body["output"] = os.fspath(output)
    body["ledger_provenance"] = deepcopy(ledger_provenance)
    body["canonical_assembly"] = {
        "schema": COMPOSITE_ASSEMBLY_SCHEMA,
        "filename": ASSEMBLY_FILENAME,
        "report_sha256": assembly["report_sha256"],
        "base_plan_sha256": sources["base"]["plan_sha256"],
        "mtp_plan_sha256": sources["mtp"]["plan_sha256"],
        "mtp_overlay_sha256": sources["mtp"]["overlay_sha256"],
    }
    return launcher._bound_record(body, "plan_sha256")


def _validate_finished_composite_assembly_for_resume(
    assembly: dict[str, Any],
    *,
    materialized_source: dict[str, Any],
    projection_sources: dict[str, Any],
    block_audit_set: dict[str, Any],
    quant_config_assembly: dict[str, Any],
    recipe: str,
) -> None:
    body = {key: value for key, value in assembly.items() if key != "report_sha256"}
    projection = assembly.get("projection_assembly")
    if (
        assembly.get("report_sha256") != _sha256(_canonical(body))
        or assembly.get("schema") != COMPOSITE_ASSEMBLY_SCHEMA
        or assembly.get("recipe") != recipe
        or assembly.get("source_snapshot")
        != _portable_source_identity(materialized_source)
        or assembly.get("projection_sources") != projection_sources
        or assembly.get("block_audit_set") != block_audit_set
        or assembly.get("quant_config_assembly") != quant_config_assembly
        or not isinstance(projection, dict)
        or projection.get("base_plan_sha256")
        != projection_sources["base"]["plan_sha256"]
        or projection.get("mtp_plan_sha256") != projection_sources["mtp"]["plan_sha256"]
    ):
        raise AssemblyError(
            "finished composite work directory is bound to different inputs"
        )


def canonicalize_composite(
    *,
    base_run_state: Path,
    mtp_overlay: Path,
    source_snapshot: Path,
    base_block_audit_dir: Path,
    mtp_block_audit_dir: Path,
    output: Path,
    work_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    """Create one normal artifact from the completed base and MTP sources."""

    base_run_state = base_run_state.expanduser().resolve(strict=True)
    mtp_overlay = mtp_overlay.expanduser().resolve(strict=True)
    source_snapshot = source_snapshot.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    work_dir = work_dir.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise AssemblyError(f"canonical output already exists: {output}")
    if (
        work_dir == output
        or base_run_state in {output, work_dir}
        or mtp_overlay
        in {
            output,
            work_dir,
        }
    ):
        raise AssemblyError("composite inputs, work, and output paths must be distinct")
    for readonly in (base_run_state, mtp_overlay, source_snapshot):
        for writable in (output, work_dir):
            if writable.is_relative_to(readonly) or readonly.is_relative_to(writable):
                raise AssemblyError(
                    "canonical writable paths must not overlap a composite input"
                )
    if work_dir.parent != output.parent:
        raise AssemblyError(
            "canonical work and output directories must share one atomic parent"
        )

    base_plan = _read_json(base_run_state / launcher.PLAN_FILENAME)
    mtp_plan = _read_json(mtp_overlay / launcher.PLAN_FILENAME)
    if _validate_source_plan(base_plan) != {"base", "mtp"}:
        raise AssemblyError("composite base source is not a full plan")
    if _validate_source_plan(mtp_plan) != {"mtp"}:
        raise AssemblyError("composite MTP source is not an overlay plan")
    mtp_run = dspark_overlay.validate_overlay(
        mtp_overlay,
        mtp_plan,
        verify_hashes=True,
    )
    mtp_manifest = _read_json(mtp_overlay / dspark_overlay.OVERLAY_FILENAME)
    try:
        materialized_source = launcher.snapshot_identity(source_snapshot)
    except launcher.LaunchError as error:
        raise AssemblyError(str(error)) from error
    if _portable_source_identity(
        base_plan.get("source", {})
    ) != _portable_source_identity(materialized_source) or _portable_source_identity(
        mtp_plan.get("source", {})
    ) != _portable_source_identity(materialized_source):
        raise AssemblyError("composite plans are bound to another source snapshot")
    projection_sources = _composite_projection_sources(
        base_plan=base_plan,
        mtp_plan=mtp_plan,
        mtp_manifest=mtp_manifest,
        mtp_run=mtp_run,
    )
    block_audit_set = validate_composite_block_audit_set(
        base_report_dir=base_block_audit_dir,
        base_plan=base_plan,
        mtp_report_dir=mtp_block_audit_dir,
        mtp_plan=mtp_plan,
    )
    artifact_plan = build_artifact_plan(
        source_snapshot,
        expert_tensor_layout=EXPERT_TENSOR_LAYOUT_GPTQMODEL,
        exl3_bits=base_plan["exl3"]["bits"],
    )
    native_config = _read_json(source_snapshot / "config.json")
    quant_config, quant_config_assembly = canonical_composite_quant_config(
        artifact_plan,
        base_plan=base_plan,
        mtp_plan=mtp_plan,
    )
    native_config["quantization_config"] = quant_config
    artifact_plan = replace(artifact_plan, model_config=native_config)
    ledger_provenance = quant_config["meta"]["ds41rt_error_ledger"]
    recipe = base_plan["recipe"]

    state_path = work_dir / ".ds41rt-exl3-state.json"
    finished_assembly_path = work_dir / ASSEMBLY_FILENAME
    if resume and finished_assembly_path.is_file() and not state_path.exists():
        assembly = _read_json(finished_assembly_path)
        _validate_finished_composite_assembly_for_resume(
            assembly,
            materialized_source=materialized_source,
            projection_sources=projection_sources,
            block_audit_set=block_audit_set,
            quant_config_assembly=quant_config_assembly,
            recipe=recipe,
        )
        publication_plan = _canonical_composite_publication_plan(
            base_plan,
            output=output,
            assembly=assembly,
            ledger_provenance=ledger_provenance,
        )
        _publish_stage(
            work_dir,
            output,
            publication_plan,
            mtp_replay_batches=mtp_run["replay_batches"],
            model_config=native_config,
        )
        return assembly

    if work_dir.exists() and not resume:
        if work_dir.is_symlink() or not work_dir.is_dir() or any(work_dir.iterdir()):
            raise AssemblyError(f"canonical work directory is not empty: {work_dir}")
    writer = SafetensorsArtifactWriter(
        artifact_plan,
        work_dir,
        calibration_rows=base_plan["corpus"]["examples"],
        seed=launcher.EXL3_SEED,
        resume=resume,
        recipe=recipe,
        quant_config_override=quant_config,
        quant_config_filename="quantize_config.json",
    )
    writer.copy_native_tensors()
    projection_assembly, projection_records = assemble_composite_projection_checkpoints(
        base_run_state=base_run_state,
        mtp_run_state=mtp_overlay,
        generated_tensors=artifact_plan.generated_tensors,
        write_generated_tensor=writer.write_generated_tensor,
        checkpoint=writer.checkpoint,
    )
    assembly = _bound_report(
        {
            "schema": COMPOSITE_ASSEMBLY_SCHEMA,
            "recipe": recipe,
            "source_snapshot": _portable_source_identity(materialized_source),
            "projection_sources": projection_sources,
            "block_audit_set": block_audit_set,
            "quant_config_assembly": quant_config_assembly,
            "projection_assembly": projection_assembly,
        }
    )
    launcher.atomic_json(work_dir / ASSEMBLY_FILENAME, assembly)
    with tempfile.TemporaryDirectory(
        prefix=".ds41rt-composite-ledger-",
        dir=work_dir.parent,
    ) as ledger_directory:
        manifest = write_exl3_error_ledger(ledger_directory, projection_records)
        if (
            not isinstance(manifest, dict)
            or manifest.get("projection_records") != len(projection_records)
            or manifest.get("complete_family_records") != len(projection_records) // 3
        ):
            raise AssemblyError("composite EXL3 error ledger did not close")
        ledger_root = Path(ledger_directory)
        writer.finish(
            {
                "schema": COMPOSITE_ASSEMBLY_SCHEMA,
                "recipe": recipe,
                "canonical_assembly": assembly,
            },
            evidence_files={
                LEDGER_FILE: ledger_root / LEDGER_FILE,
                LEDGER_MANIFEST_FILE: ledger_root / LEDGER_MANIFEST_FILE,
            },
        )
    publication_plan = _canonical_composite_publication_plan(
        base_plan,
        output=output,
        assembly=assembly,
        ledger_provenance=ledger_provenance,
    )
    _publish_stage(
        work_dir,
        output,
        publication_plan,
        mtp_replay_batches=mtp_run["replay_batches"],
        model_config=native_config,
    )
    return assembly


def canonicalize(
    *,
    raw_artifact: Path,
    source_snapshot: Path,
    run_state: Path | None,
    block_audit_dir: Path,
    output: Path,
    work_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    raw_artifact = raw_artifact.expanduser().resolve(strict=True)
    source_snapshot = source_snapshot.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    work_dir = work_dir.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise AssemblyError(f"canonical output already exists: {output}")
    if work_dir == output or raw_artifact in {output, work_dir}:
        raise AssemblyError("raw, work, and canonical output paths must be distinct")
    for readonly in (raw_artifact, source_snapshot):
        for writable in (output, work_dir):
            if writable.is_relative_to(readonly) or readonly.is_relative_to(writable):
                raise AssemblyError(
                    "canonical writable paths must not overlap the raw artifact "
                    "or source snapshot"
                )
    if work_dir.parent != output.parent:
        raise AssemblyError(
            "canonical work and output directories must share one atomic parent"
        )

    raw_plan = _read_json(raw_artifact / launcher.PLAN_FILENAME)
    _validate_planned_run_state_paths(raw_plan)
    if run_state is None:
        run_state = Path(raw_plan["run_state_dir"])
    run_state = run_state.expanduser().resolve(strict=True)
    block_audit_set = validate_block_audit_set(
        block_audit_dir,
        raw_plan,
        run_state=run_state,
    )
    launcher.validate_published_artifact(
        raw_artifact,
        raw_plan,
        # The specialized validation immediately below hashes every manifest
        # entry and also validates the GPTQModel envelope. Avoid reading the
        # roughly 84 GB raw artifact twice.
        verify_file_hashes=False,
    )
    try:
        materialized_source = launcher.snapshot_identity(source_snapshot)
    except launcher.LaunchError as error:
        raise AssemblyError(str(error)) from error
    planned_source = raw_plan.get("source")
    if not isinstance(planned_source, dict) or _portable_source_identity(
        planned_source
    ) != _portable_source_identity(materialized_source):
        raise AssemblyError("raw artifact plan is bound to another source snapshot")
    raw_config = _read_json(raw_artifact / "config.json")
    native_config = _read_json(source_snapshot / "config.json")
    validate_gptqmodel_publication(
        raw_artifact,
        _source_complete_raw_config(raw_config, native_config),
        verify_all_hashes=True,
        require_canonical=False,
    )
    raw_run = _read_json(raw_artifact / launcher.RUN_FILENAME)
    raw_manifest = _read_json(raw_artifact / launcher.ARTIFACT_MANIFEST_FILENAME)
    projection_bits, _tier_contracts = _projection_bits_from_tier_plans(
        raw_plan,
        run_state,
    )
    plan = build_artifact_plan(
        source_snapshot,
        expert_tensor_layout=EXPERT_TENSOR_LAYOUT_GPTQMODEL,
        exl3_bits=raw_plan["exl3"]["bits"],
        exl3_projection_bits=projection_bits,
    )
    raw_quant_config = deepcopy(raw_config.get("quantization_config"))
    if not isinstance(raw_quant_config, dict):
        raise AssemblyError("raw artifact has no GPTQModel quantization config")
    quant_config, quant_config_assembly = canonical_quant_config(
        plan,
        raw_quant_config,
    )
    native_config["quantization_config"] = quant_config
    plan = replace(plan, model_config=native_config)
    recipe = raw_plan["recipe"]

    state_path = work_dir / ".ds41rt-exl3-state.json"
    finished_assembly_path = work_dir / ASSEMBLY_FILENAME
    if resume and finished_assembly_path.is_file() and not state_path.exists():
        assembly = _read_json(finished_assembly_path)
        _validate_finished_assembly_for_resume(
            assembly,
            raw_plan=raw_plan,
            raw_run=raw_run,
            raw_manifest=raw_manifest,
            materialized_source=materialized_source,
            block_audit_set=block_audit_set,
            recipe=recipe,
        )
        publication_plan = _canonical_publication_plan(
            raw_plan,
            output=output,
            assembly=assembly,
        )
        _publish_stage(
            work_dir,
            output,
            publication_plan,
            mtp_replay_batches=raw_run["mtp_replay_batches"],
            model_config=native_config,
        )
        return assembly

    if work_dir.exists() and not resume:
        if work_dir.is_symlink() or not work_dir.is_dir() or any(work_dir.iterdir()):
            raise AssemblyError(f"canonical work directory is not empty: {work_dir}")
    writer = SafetensorsArtifactWriter(
        plan,
        work_dir,
        calibration_rows=raw_plan["corpus"]["examples"],
        seed=launcher.EXL3_SEED,
        resume=resume,
        recipe=recipe,
        quant_config_override=quant_config,
        quant_config_filename="quantize_config.json",
    )
    writer.copy_native_tensors()
    projection_assembly = assemble_projection_checkpoints(
        run_state,
        generated_tensors=plan.generated_tensors,
        write_generated_tensor=writer.write_generated_tensor,
        expected_plan_sha256=raw_plan["plan_sha256"],
        checkpoint=writer.checkpoint,
    )
    assembly = _bound_report(
        {
            "schema": ASSEMBLY_SCHEMA,
            "recipe": recipe,
            "materialized_raw_artifact": os.fspath(raw_artifact),
            "source_plan_sha256": raw_plan["plan_sha256"],
            "source_run_sha256": raw_run["run_sha256"],
            "source_artifact_manifest_sha256": raw_manifest["manifest_sha256"],
            "source_snapshot": _portable_source_identity(materialized_source),
            "block_audit_set": block_audit_set,
            "quant_config_assembly": quant_config_assembly,
            "projection_assembly": projection_assembly,
        }
    )
    # Write the bound assembly before finish removes the resumable writer state.
    # A crash during finish can then either resume the writer or, after the state
    # is removed, re-enter only the validated publication/rename boundary.
    launcher.atomic_json(work_dir / ASSEMBLY_FILENAME, assembly)
    writer.finish(
        {
            "schema": ASSEMBLY_SCHEMA,
            "recipe": recipe,
            "canonical_assembly": assembly,
        },
        evidence_files={
            LEDGER_FILE: raw_artifact / LEDGER_FILE,
            LEDGER_MANIFEST_FILE: raw_artifact / LEDGER_MANIFEST_FILE,
        },
    )
    publication_plan = _canonical_publication_plan(
        raw_plan,
        output=output,
        assembly=assembly,
    )
    _publish_stage(
        work_dir,
        output,
        publication_plan,
        mtp_replay_batches=raw_run["mtp_replay_batches"],
        model_config=native_config,
    )
    return assembly


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-artifact", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument(
        "--run-state",
        type=Path,
        help=(
            "materialized run-state path; required when the plan's container "
            "path is not visible in this namespace"
        ),
    )
    parser.add_argument("--block-audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = canonicalize(
        raw_artifact=args.raw_artifact,
        source_snapshot=args.source_snapshot,
        run_state=args.run_state,
        block_audit_dir=args.block_audit_dir,
        output=args.output,
        work_dir=args.work_dir,
        resume=args.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        AssemblyError,
        ValueError,
        OSError,
        json.JSONDecodeError,
        launcher.LaunchError,
        block_audit.AuditError,
    ) as error:
        print(f"canonicalize-gptqmodel-artifact: {error}", file=sys.stderr)
        raise SystemExit(2) from error
