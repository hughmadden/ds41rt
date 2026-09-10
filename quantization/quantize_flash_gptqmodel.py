#!/usr/bin/env python3
"""Quantize DeepSeek-V4 routed experts and prepare its MTP experts."""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import math
import os
import re
import shutil
import sys
import urllib.parse
from fractions import Fraction
from pathlib import Path
from typing import Any

from deepseek_v4_layer_boundary_store import (
    BOUNDARY_CONTRACT,
    DeepSeekV4LayerBoundaryController,
    DeepSeekV4LayerBoundaryStore,
    LayerBoundaryStop,
)
from deepseek_v4_mtp_prefix_store import (
    ANCHOR_SELECTION_CONTRACT,
    SEQUENCE_REPLAY_BATCH_CONTRACT,
    DeepSeekV4MTPPrefixStore,
    sha256_file,
)
from preflight import report_identity_sha256

PLAN_SCHEMA = "ds41rt-deepseek-v4-gptqmodel-plan-v7"
LEGACY_PLAN_SCHEMA = "ds41rt-deepseek-v4-gptqmodel-plan-v5"
PREVIOUS_PLAN_SCHEMA = "ds41rt-deepseek-v4-gptqmodel-plan-v6"
SUPPORTED_PLAN_SCHEMAS = frozenset(
    (LEGACY_PLAN_SCHEMA, PREVIOUS_PLAN_SCHEMA, PLAN_SCHEMA)
)
STRICT_STORAGE_PLAN_SCHEMAS = frozenset((PREVIOUS_PLAN_SCHEMA, PLAN_SCHEMA))
RUN_SCHEMA = "ds41rt-deepseek-v4-gptqmodel-run-v5"
BASE_PREFIX_RUN_SCHEMA = "ds41rt-deepseek-v4-base-prefix-run-v1"
ARTIFACT_MANIFEST_SCHEMA = "ds41rt-deepseek-v4-gptqmodel-artifact-v1"
CALIBRATION_MANIFEST_SCHEMA = "ds41rt-flash-exl3-calibration-corpus-v3"
ROUTE_SCREEN_SCHEMA = "ds41rt-flash-natural-route-distribution-v1"
ROUTE_QUALIFICATION_SCREEN = "screen-report"
ROUTE_QUALIFICATION_INLINE = "inline-full-corpus"
ROUTE_QUALIFICATION_MODES = (
    ROUTE_QUALIFICATION_SCREEN,
    ROUTE_QUALIFICATION_INLINE,
)
NATURAL_ROUTE_RECIPE = "deepseek_v4_exl3_trellis_2bpw_v4_flash_natural_route"
ROUTE_EVIDENCE_CONTRACT = "ds41rt.exl3-natural-route"
ZERO_ROUTE_RECOVERY_CONTRACT = "ds41rt.exl3-zero-route-recovery"
ZERO_ROUTE_RECOVERY_TRIGGER = "natural-route-count-below-1024"
ZERO_ROUTE_RECOVERY_SAMPLE_SOURCE = "same-fixed-calibration-selection"
ZERO_ROUTE_RECOVERY_CAPTURE_METHOD = (
    "direct-expert-router-ranks-7-12-then-identity-residual"
)
ZERO_ROUTE_RECOVERY_SELECTION_POLICY = (
    "rank-ascending-then-fixed-replay-order-v1"
)
ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MIN = 7
ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MAX = 12
ZERO_ROUTE_RECOVERY_TARGET_SAMPLE_COUNT = 1024
ZERO_ROUTE_RECOVERY_IDENTITY_POLICY = (
    "normalized-2i-residual-to-effective-count-1024-v2"
)
PROJECTION_CHECKPOINT_CONTRACT = "ds41rt.exl3-projection-checkpoint-v1"
REMOTE_WORKER_CONTRACT = "ds41rt.exl3-remote-worker-v1"
REMOTE_WORKER_SCHEDULER = "dynamic-pipelined-slot-projection-v2"
REMOTE_PIPELINE_DEPTH = 2
REMOTE_POSTPROCESS_ALLOWANCE = 1
BASE_EXPERT_PATTERN = (
    r"^model\.layers\.\d+\.mlp\.experts\.\d+\."
    r"(?:gate_proj|up_proj|down_proj)$"
)
EXL3_SEED = 787
EXL3_SIGMA_REG = 0.025
EXL3_HESSIAN_CAPTURE_CONTRACT = "raw-xtx-sum-fp32-v1"
EXL3_HESSIAN_NUMERICAL_CONTRACT = "signed-block-hadamard-congruence-fp64-v1"
EXL3_HESSIAN_SYMMETRY_CONTRACT = "mean-with-transpose-fp64"
REVISION_RE = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
PREFLIGHT_ROLE_CONTRACTS = {
    "coordinator": ("linux/amd64", "120"),
    "expert": ("linux/arm64", "121"),
}
PLAN_FILENAME = "ds41rt-gptqmodel-plan.json"
RUN_FILENAME = "ds41rt-gptqmodel-run.json"
BASE_PREFIX_RUN_FILENAME = "ds41rt-base-prefix-run.json"
ARTIFACT_MANIFEST_FILENAME = "ds41rt-gptqmodel-artifact.json"
ERROR_JOURNAL_FILENAME = ".ds41rt-exl3-error-journal.jsonl"
INLINE_MIXED_CANDIDATE_JOURNAL_FILENAME = (
    f"{ERROR_JOURNAL_FILENAME}.k2-candidates"
)
DIRECT_STATE_PREFLIGHT_FILENAME = "lazy-direct-state-preflight.json"
PROJECTION_CHECKPOINT_DIRNAME = "projection-checkpoints"
ACTIVE_LAYER_SOURCE_DIRNAME = "active-layer-source"
REMOTE_ASSIGNMENT_DIRNAME = "dynamic-projection-assignments"
EXPORT_STAGE_DIRNAME = "export-stage"
LAYER_BOUNDARY_DIRNAME = "layer-boundary"
CAPTURE_FRONTIER_DIRNAME = "layer-capture-frontier"
CAPTURE_BATCH_SPOOL_DIRNAME = "capture-batch-journal"
POST_QUANT_REPLAY_DIRNAME = "post-quant-replay"
MTP_ACTIVATION_DIRNAME = "mtp-layer-activations"
INLINE_MIXED_TIER_PLAN_DIRNAME = "inline-mixed-tier-plans"
INLINE_MIXED_SCHEMA = "gptqmodel.exl3-inline-mixed"
INLINE_MIXED_SCORE = (
    "k2-hessian-weighted-relative-error-times-natural-gate-squared-mass-v1"
)
CAPTURE_FRONTIER_CONTRACT = "ds41rt.exl3-capture-frontier-v1"
HOST_RSS_LIMIT_BYTES = 150 * 1024**3
CUDA_ALLOCATION_LIMIT_BYTES = 82 * 1024**3
MEMORY_TELEMETRY_INTERVAL_BATCHES = 64
DEFAULT_CAPTURE_BATCH_CHECKPOINT_INTERVAL = 64
EXECUTION_UPGRADE_FILENAME = "ds41rt-execution-upgrade.json"
EXECUTION_UPGRADE_HISTORY_DIRNAME = "execution-upgrade-history"
EXECUTION_UPGRADE_SCHEMA = "ds41rt-deepseek-v4-execution-upgrade-v1"
BOUNDARY_DIRECTORY_RE = re.compile(
    r"layer-(?P<layer>[0-9]{6})-(?P<digest>[0-9a-f]{16})\Z"
)
MTP_EXECUTION_INTEGRATED = "integrated"
MTP_EXECUTION_EXTERNAL_OVERLAY = "external-overlay"
MTP_EXECUTION_MODES = (
    MTP_EXECUTION_INTEGRATED,
    MTP_EXECUTION_EXTERNAL_OVERLAY,
)
DEFAULT_MTP_ANCHOR_SAMPLE_COUNT = 327_680
DEFAULT_MTP_ANCHOR_SAMPLE_SEED = 20_260_809


class LaunchError(RuntimeError):
    """The requested production quantization run is not reproducible."""


def natural_route_recipe(bits: int) -> str:
    if bits == 2:
        return NATURAL_ROUTE_RECIPE
    if bits == 3:
        return "deepseek_v4_exl3_trellis_3bpw_v4_flash_natural_route"
    raise LaunchError("EXL3 bitrate must be integer K2 or K3")


def _mixed_policy(
    *,
    namespace: str,
    base_bits: int,
    target_bpw: str | None,
    projection_ratio: tuple[int, int, int],
    tier_plan_root: Path,
) -> dict[str, Any] | None:
    """Build exact-rational private metadata while keeping standard bits integer."""

    if target_bpw is None:
        return None
    try:
        target = Fraction(str(target_bpw))
    except (ValueError, ZeroDivisionError) as error:
        raise LaunchError("mixed target BPW must be an exact decimal or fraction") from error
    extra = target - base_bits
    if not 0 < extra < 1:
        raise LaunchError("mixed target BPW must lie strictly between K and K+1")
    if base_bits != 2:
        raise LaunchError("inline mixed quantization currently supports K2/K3")
    if (
        len(projection_ratio) != 3
        or any(isinstance(value, bool) or int(value) <= 0 for value in projection_ratio)
    ):
        raise LaunchError("mixed gate:up:down ratio must contain three positive integers")
    return {
        "schema": INLINE_MIXED_SCHEMA,
        "schema_version": 1,
        "namespace": namespace,
        "base_bits": base_bits,
        "upgrade_bits": base_bits + 1,
        "extra_bits": {
            "numerator": extra.numerator,
            "denominator": extra.denominator,
        },
        "target_bpw": f"{target.numerator}/{target.denominator}",
        "projection_ratio": dict(
            zip(("w1", "w3", "w2"), map(int, projection_ratio))
        ),
        "score_kind": INLINE_MIXED_SCORE,
        "tier_plan_root": os.fspath(tier_plan_root),
    }


@contextmanager
def capture_frontier_scope(root: Path):
    """Keep one exact recovery frontier active through target and dSpark."""

    variable = "GPTQMODEL_EXL3_CAPTURE_FRONTIER"
    expected = os.fspath(root)
    previous = os.environ.get(variable)
    if previous not in {None, expected}:
        raise LaunchError(f"{variable} conflicts with the immutable run state")
    os.environ[variable] = expected
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous


@contextmanager
def capture_batch_spool_scope(root: Path, *, checkpoint_interval: int):
    """Bind additive Hessian recovery records to the immutable run state."""

    variable = "GPTQMODEL_EXL3_CAPTURE_BATCH_SPOOL"
    interval_variable = "GPTQMODEL_EXL3_CAPTURE_BATCH_CHECKPOINT_INTERVAL"
    expected = os.fspath(root)
    previous = os.environ.get(variable)
    previous_interval = os.environ.get(interval_variable)
    if previous not in {None, expected}:
        raise LaunchError(f"{variable} conflicts with the immutable run state")
    if (
        isinstance(checkpoint_interval, bool)
        or not isinstance(checkpoint_interval, int)
        or checkpoint_interval <= 0
    ):
        raise LaunchError("capture batch checkpoint interval is invalid")
    expected_interval = str(checkpoint_interval)
    if previous_interval not in {None, expected_interval}:
        raise LaunchError(
            f"{interval_variable} conflicts with the authorized execution"
        )
    os.environ[variable] = expected
    os.environ[interval_variable] = expected_interval
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous
        if previous_interval is None:
            os.environ.pop(interval_variable, None)
        else:
            os.environ[interval_variable] = previous_interval


@contextmanager
def memory_safety_scope(contract: dict[str, Any]):
    """Expose immutable fail-closed capture limits to GPTQModel."""

    values = {
        "GPTQMODEL_EXL3_HOST_RSS_LIMIT_BYTES": contract["host_rss_limit_bytes"],
        "GPTQMODEL_EXL3_CUDA_ALLOCATION_LIMIT_BYTES": contract[
            "cuda_allocation_limit_bytes"
        ],
        "GPTQMODEL_EXL3_MEMORY_TELEMETRY_INTERVAL_BATCHES": contract[
            "telemetry_interval_batches"
        ],
    }
    previous = {name: os.environ.get(name) for name in values}
    for name, value in values.items():
        expected = str(value)
        if previous[name] not in {None, expected}:
            raise LaunchError(f"{name} conflicts with the immutable run state")
        os.environ[name] = expected
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LaunchError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LaunchError(f"expected a JSON object in {path}")
    return value


def shard_identity(snapshot: Path, name: str) -> dict[str, Any]:
    path = snapshot / name
    if path.is_symlink():
        link = Path(os.readlink(path))
        if link.is_absolute() or link.parts[:3] != ("..", "..", "blobs"):
            raise LaunchError(
                f"source snapshot shard is not a canonical Hugging Face blob: {path}"
            )
        if len(link.parts) != 4 or SHA256_RE.fullmatch(link.parts[3]) is None:
            raise LaunchError(
                f"source snapshot shard has no SHA-256 blob identity: {path}"
            )
        blob_root = (snapshot.parent.parent / "blobs").resolve(strict=True)
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise LaunchError(
                f"source snapshot has a broken shard link: {path}"
            ) from exc
        expected_blob = blob_root / link.parts[3]
        if (
            resolved != expected_blob
            or expected_blob.is_symlink()
            or not expected_blob.is_file()
        ):
            raise LaunchError(
                f"source snapshot shard escapes its Hugging Face blob store: {path}"
            )
        return {
            "name": name,
            "bytes": resolved.stat().st_size,
            "hf_blob_sha256": link.parts[3],
        }
    if not path.is_file():
        raise LaunchError(f"source snapshot is missing regular shard {path}")
    return {"name": name, "bytes": path.stat().st_size}


def snapshot_identity(snapshot: Path) -> dict[str, Any]:
    snapshot = snapshot.expanduser().resolve(strict=True)
    if not snapshot.is_dir() or not REVISION_RE.fullmatch(snapshot.name):
        raise LaunchError(
            "--snapshot must be an immutable 40- or 64-hex Hugging Face snapshot"
        )
    config_path = snapshot / "config.json"
    index_path = snapshot / "model.safetensors.index.json"
    config = read_json_object(config_path)
    index = read_json_object(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise LaunchError("source model index has no weight map")
    shards = sorted(set(weight_map.values()))
    if any(not isinstance(name, str) or Path(name).name != name for name in shards):
        raise LaunchError("source model index has unsafe shard names")
    shard_records = [shard_identity(snapshot, name) for name in shards]
    geometry = {
        "num_hidden_layers": config.get("num_hidden_layers"),
        "n_routed_experts": config.get("n_routed_experts"),
        "hidden_size": config.get("hidden_size"),
        "moe_intermediate_size": config.get("moe_intermediate_size"),
        "dspark_target_layer_ids": config.get("dspark_target_layer_ids"),
        "mtp_block_count": 3,
        "num_hash_layers": config.get("num_hash_layers"),
        "num_experts_per_tok": config.get("num_experts_per_tok"),
        "hc_mult": config.get("hc_mult"),
        "num_nextn_predict_layers": config.get("num_nextn_predict_layers"),
    }
    if any(
        isinstance(geometry[key], bool)
        or not isinstance(geometry[key], int)
        or geometry[key] <= 0
        for key in (
            "num_hidden_layers",
            "n_routed_experts",
            "hidden_size",
            "moe_intermediate_size",
        )
    ) or geometry["dspark_target_layer_ids"] != list(
        range(geometry["num_hidden_layers"] - 3, geometry["num_hidden_layers"])
    ):
        raise LaunchError(
            f"source snapshot is not supported DeepSeek-V4 geometry: {geometry}"
        )
    namespace_audit = deepseek_v4_namespace_audit(config, weight_map)
    return {
        "path": os.fspath(snapshot),
        "revision": snapshot.name,
        "config_sha256": sha256_file(config_path),
        "index_sha256": sha256_file(index_path),
        "shards": shard_records,
        "total_shard_bytes": sum(record["bytes"] for record in shard_records),
        "geometry": geometry,
        "namespace_audit": namespace_audit,
    }


def deepseek_v4_namespace_audit(
    config: dict[str, Any],
    weight_map: dict[str, Any],
) -> dict[str, Any]:
    """Prove the native routed/MTP tensor inventory without loading weights."""

    layer_count = config.get("num_hidden_layers")
    expert_count = config.get("n_routed_experts")
    hash_layer_count = config.get("num_hash_layers")
    target_ids = config.get("dspark_target_layer_ids")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (layer_count, expert_count)
    ) or not isinstance(hash_layer_count, int) or not 0 <= hash_layer_count <= layer_count:
        raise LaunchError("source configuration has invalid layer/router geometry")
    if target_ids != list(range(layer_count - 3, layer_count)):
        raise LaunchError("source configuration does not identify three final MTP taps")

    names = set(weight_map)
    expected_base_experts = {
        f"layers.{layer}.ffn.experts.{expert}.{projection}.{suffix}"
        for layer in range(layer_count)
        for expert in range(expert_count)
        for projection in ("w1", "w2", "w3")
        for suffix in ("scale", "weight")
    }
    expected_mtp_experts = {
        f"mtp.{block}.ffn.experts.{expert}.{projection}.{suffix}"
        for block in range(3)
        for expert in range(expert_count)
        for projection in ("w1", "w2", "w3")
        for suffix in ("scale", "weight")
    }
    actual_base_experts = {
        name
        for name in names
        if re.fullmatch(
            r"layers\.\d+\.ffn\.experts\.\d+\.w[123]\.(?:scale|weight)",
            name,
        )
    }
    actual_mtp_experts = {
        name
        for name in names
        if re.fullmatch(
            r"mtp\.\d+\.ffn\.experts\.\d+\.w[123]\.(?:scale|weight)",
            name,
        )
    }
    for label, actual, expected in (
        ("base routed experts", actual_base_experts, expected_base_experts),
        ("MTP routed experts", actual_mtp_experts, expected_mtp_experts),
    ):
        if actual != expected:
            missing = sorted(expected - actual)[:4]
            unexpected = sorted(actual - expected)[:4]
            raise LaunchError(
                f"source {label} inventory differs: missing={missing} "
                f"unexpected={unexpected}"
            )

    expected_shared = {
        f"{prefix}.ffn.shared_experts.{projection}.{suffix}"
        for prefix in (
            *(f"layers.{layer}" for layer in range(layer_count)),
            *(f"mtp.{block}" for block in range(3)),
        )
        for projection in ("w1", "w2", "w3")
        for suffix in ("scale", "weight")
    }
    if not expected_shared <= names:
        raise LaunchError(
            "source shared-expert inventory is incomplete: "
            f"{sorted(expected_shared - names)[:4]}"
        )

    hash_router_layers: list[int] = []
    learned_router_layers: list[int] = []
    for layer in range(layer_count):
        prefix = f"layers.{layer}.ffn.gate"
        if f"{prefix}.weight" not in names:
            raise LaunchError(f"source layer {layer} has no router weight")
        if layer < hash_layer_count:
            if f"{prefix}.tid2eid" not in names or f"{prefix}.bias" in names:
                raise LaunchError(f"source layer {layer} hash-router state differs")
            hash_router_layers.append(layer)
        else:
            if f"{prefix}.bias" not in names or f"{prefix}.tid2eid" in names:
                raise LaunchError(f"source layer {layer} learned-router state differs")
            learned_router_layers.append(layer)
    for block in range(3):
        prefix = f"mtp.{block}.ffn.gate"
        if not {f"{prefix}.weight", f"{prefix}.bias"} <= names:
            raise LaunchError(f"source MTP block {block} learned router is incomplete")

    return {
        "contract": "ds41rt.deepseek-v4-native-namespace-audit-v1",
        "base_layers": layer_count,
        "mtp_blocks": 3,
        "routed_experts_per_block": expert_count,
        "base_routed_projection_tensors": len(actual_base_experts),
        "mtp_routed_projection_tensors": len(actual_mtp_experts),
        "shared_expert_tensors": len(expected_shared),
        "hash_router_layers": hash_router_layers,
        "learned_router_layers": learned_router_layers,
        "learned_mtp_router_blocks": [0, 1, 2],
        "configured_next_token_layers": config.get("num_nextn_predict_layers"),
        "resolved_mtp_blocks_from_tensors": 3,
        "quantized_scope": "routed-expert-w1-w2-w3-only",
    }


def calibration_stream(path: Path) -> tuple[list[str], dict[str, Any]]:
    path = path.expanduser().resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise LaunchError("--calibration-jsonl must be one regular file")
    texts: list[str] = []
    identifiers: list[str] = []
    text_field: str | None = None
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise LaunchError(
                        f"calibration JSONL line {line_number} is not an object"
                    )
                present_text_fields = [
                    field for field in ("text", "prompt") if field in value
                ]
                if len(present_text_fields) != 1:
                    raise LaunchError(
                        f"calibration JSONL line {line_number} must contain exactly "
                        "one of `text` or `prompt`"
                    )
                row_text_field = present_text_fields[0]
                if text_field is None:
                    text_field = row_text_field
                elif row_text_field != text_field:
                    raise LaunchError(
                        "calibration JSONL mixes `text` and `prompt` row schemas"
                    )
                text = value[row_text_field]
                identifier = value.get("id", f"line-{line_number:08d}")
                if not isinstance(text, str) or not text.strip():
                    raise LaunchError(
                        f"calibration JSONL line {line_number} has no non-empty "
                        f"`{row_text_field}`"
                    )
                if not isinstance(identifier, str) or not identifier:
                    raise LaunchError(
                        f"calibration JSONL line {line_number} has an invalid id"
                    )
                texts.append(text)
                identifiers.append(identifier)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LaunchError(f"cannot read calibration JSONL {path}: {exc}") from exc
    if not texts or len(set(identifiers)) != len(identifiers):
        raise LaunchError("calibration JSONL is empty or contains duplicate ids")
    normalized = canonical_json(
        [
            {"id": identifier, "text": text}
            for identifier, text in zip(identifiers, texts, strict=True)
        ]
    )
    return texts, {
        "path": os.fspath(path),
        "file_sha256": sha256_file(path),
        "normalized_stream_sha256": hashlib.sha256(normalized).hexdigest(),
        "text_field": text_field,
        "examples": len(texts),
        "utf8_bytes": sum(len(text.encode()) for text in texts),
    }


def calibration_evidence(
    manifest_path: Path,
    route_screen_path: Path | None,
    *,
    corpus: dict[str, Any],
    source: dict[str, Any],
    mtp_execution_mode: str = MTP_EXECUTION_INTEGRATED,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve(strict=True)
    manifest = read_json_object(manifest_path)
    splits = manifest.get("splits")
    calibration = splits.get("calibration") if isinstance(splits, dict) else None
    screening = splits.get("screening") if isinstance(splits, dict) else None
    builder = manifest.get("builder")
    training_data = manifest.get("training_data_snapshot")
    base_layer_count = source["geometry"]["num_hidden_layers"]
    full_layer_count = base_layer_count + source["geometry"]["mtp_block_count"]
    valid_layer_counts = (
        {full_layer_count}
        if mtp_execution_mode == MTP_EXECUTION_INTEGRATED
        else {base_layer_count, full_layer_count}
    )
    if (
        manifest.get("schema") != CALIBRATION_MANIFEST_SCHEMA
        or not isinstance(calibration, dict)
        or not isinstance(screening, dict)
        or not isinstance(builder, dict)
        or not isinstance(training_data, dict)
        or manifest.get("screening_calibration_identity_subset") is not True
        or manifest.get("source_group_overlap") != []
        or calibration.get("sha256") != corpus["file_sha256"]
        or calibration.get("summary", {}).get("records") != corpus["examples"]
        or not isinstance(calibration.get("records"), list)
        or len(calibration["records"]) != corpus["examples"]
        or screening.get("derived_from") != "calibration"
        or not isinstance(screening.get("records"), list)
        or screening.get("summary", {}).get("records") != len(screening["records"])
        or REVISION_RE.fullmatch(str(builder.get("revision", ""))) is None
        or REVISION_RE.fullmatch(str(training_data.get("revision", ""))) is None
    ):
        raise LaunchError("calibration manifest does not bind the production corpus")
    manifest_corpus_path = (manifest_path.parent / str(calibration.get("file", ""))).resolve()
    if manifest_corpus_path != Path(corpus["path"]):
        raise LaunchError("calibration manifest selects a different JSONL stream")
    calibration_ids = [record.get("id") for record in calibration["records"]]
    calibration_prompt_hashes = [
        record.get("prompt_sha256") for record in calibration["records"]
    ]
    token_hashes = [
        record.get("token_ids_sha256") for record in calibration["records"]
    ]
    if (
        len(set(calibration_ids)) != len(calibration_ids)
        or any(not isinstance(value, str) or not value for value in calibration_ids)
        or any(SHA256_RE.fullmatch(str(value)) is None for value in calibration_prompt_hashes)
        or any(SHA256_RE.fullmatch(str(value)) is None for value in token_hashes)
    ):
        raise LaunchError("calibration manifest has invalid prompt/token identities")

    manifest_evidence = {
        "path": os.fspath(manifest_path),
        "sha256": sha256_file(manifest_path),
        "schema": manifest["schema"],
        "builder_revision": builder["revision"],
        "builder_sha256": builder.get("sha256"),
        "training_data_revision": training_data["revision"],
        "tokenizer_sha256": manifest.get("tokenizer_sha256"),
        "calibration_prompt_tokens": calibration["summary"].get("prompt_tokens"),
        "calibration_examples": corpus["examples"],
        "calibration_token_identity_sha256": hashlib.sha256(
            canonical_json(token_hashes)
        ).hexdigest(),
    }
    if route_screen_path is None:
        return {
            "manifest": manifest_evidence,
            "route_qualification": {
                "mode": ROUTE_QUALIFICATION_INLINE,
                "status": "deferred-to-full-corpus-capture",
                "scope": (
                    "base-and-integrated-mtp"
                    if mtp_execution_mode == MTP_EXECUTION_INTEGRATED
                    else "base-with-external-mtp-deferred"
                ),
                "natural_route_contract": ROUTE_EVIDENCE_CONTRACT,
                "recovery_contract": ZERO_ROUTE_RECOVERY_CONTRACT,
                "recovery_trigger": ZERO_ROUTE_RECOVERY_TRIGGER,
                "target_effective_rows": ZERO_ROUTE_RECOVERY_TARGET_SAMPLE_COUNT,
                "failure_policy": "fail-only-on-router-or-evidence-invariant",
            },
        }

    route_screen_path = route_screen_path.expanduser().resolve(strict=True)
    route_screen = read_json_object(route_screen_path)
    corpora = route_screen.get("corpora")
    screen_corpus = corpora.get("screening") if isinstance(corpora, dict) else None
    prompts = route_screen.get("prompts")
    distributions = route_screen.get("distributions")
    distribution = (
        distributions.get("screening") if isinstance(distributions, dict) else None
    )
    layers = distribution.get("layers") if isinstance(distribution, dict) else None
    if (
        route_screen.get("schema") != ROUTE_SCREEN_SCHEMA
        or route_screen.get("checkpoint") != source["path"]
        or not isinstance(screen_corpus, dict)
        or screen_corpus.get("sha256") != screening.get("sha256")
        or screen_corpus.get("prompts") != len(screening["records"])
        or not isinstance(prompts, list)
        or len(prompts) != len(screening["records"])
        or [record.get("id") for record in prompts]
        != [record.get("id") for record in screening["records"]]
        or not isinstance(layers, list)
        or len(layers) not in valid_layer_counts
    ):
        raise LaunchError("router screen does not bind the candidate corpus/checkpoint")
    base_layers = layers[:base_layer_count]
    if any(
        layer.get("layer_id") != layer_id
        or layer.get("routes") != layer.get("rows", 0) * 6
        or layer.get("zero_hit_experts") != 0
        for layer_id, layer in enumerate(base_layers)
    ):
        raise LaunchError("router screen did not cover every natural base expert")
    mtp_verify_cycles = sum(int(record.get("mtp_verify_cycles", 0)) for record in prompts)
    if mtp_execution_mode == MTP_EXECUTION_INTEGRATED:
        if mtp_verify_cycles < 1:
            raise LaunchError("router screen did not execute dSpark verification")
    elif len(layers) == base_layer_count:
        if (
            route_screen.get("scope") != "base-only-mtp-deferred"
            or mtp_verify_cycles != 0
        ):
            raise LaunchError(
                "external-overlay base router screen must defer MTP explicitly"
            )
    elif mtp_verify_cycles < 1:
        raise LaunchError("full router screen did not execute dSpark verification")
    status = (
        "base-qualified-mtp-deferred"
        if len(layers) == base_layer_count
        else "qualified"
    )
    return {
        "manifest": manifest_evidence,
        "route_qualification": {
            "mode": ROUTE_QUALIFICATION_SCREEN,
            "status": status,
        },
        "router_screen": {
            "path": os.fspath(route_screen_path),
            "sha256": sha256_file(route_screen_path),
            "schema": route_screen["schema"],
            "corpus_sha256": screening["sha256"],
            "prompts": len(prompts),
            "prompt_tokens": screening["summary"].get("prompt_tokens"),
            "base_rows": sum(int(layer["rows"]) for layer in base_layers),
            "base_routes": sum(int(layer["routes"]) for layer in base_layers),
            "base_zero_hit_experts": 0,
            "mtp_verify_cycles": mtp_verify_cycles,
            "status": status,
        },
    }


def quantization_toolchain_identity(*, include_route_screen: bool = True) -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    names = [
        "preflight.py",
        "quantize_flash_gptqmodel.py",
        "deepseek_v4_layer_boundary_store.py",
        "deepseek_v4_mtp_prefix_store.py",
    ]
    if include_route_screen:
        names.insert(0, "collect_deepseek_v4_route_screen.py")
    return {
        "files": {
            name: sha256_file(root / name)
            for name in names
        }
    }


def preflight_identity(
    path: Path,
    expected_revision: str,
    *,
    role: str = "coordinator",
    expected_gpu_count: int | None = None,
) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    report = read_json_object(path)
    gptqmodel = report.get("gptqmodel")
    image_digest = report.get("image_digest")
    gpus = report.get("gpus")
    expected_platform_arch = PREFLIGHT_ROLE_CONTRACTS.get(role)
    if expected_gpu_count is None:
        expected_gpu_count = 2 if role == "coordinator" else 1
    if (
        isinstance(expected_gpu_count, bool)
        or not isinstance(expected_gpu_count, int)
        or expected_gpu_count not in ({1, 2} if role == "coordinator" else {1})
    ):
        raise LaunchError(f"{role} preflight GPU count is invalid")
    if (
        expected_platform_arch is None
        or report.get("status") != "qualified"
        or report.get("role") != role
        or (report.get("target_platform"), report.get("cuda_arch"))
        != expected_platform_arch
        or not isinstance(gptqmodel, dict)
        or gptqmodel.get("revision") != expected_revision
        or not isinstance(image_digest, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", image_digest) is None
        or report.get("python", {}).get("gil_enabled") is not False
        or not isinstance(gpus, list)
        or len(gpus) != expected_gpu_count
        or any(not isinstance(gpu, dict) for gpu in gpus)
        or [gpu.get("index") for gpu in gpus] != list(range(expected_gpu_count))
        or any(
            not isinstance(gpu.get("uuid"), str) or not gpu["uuid"]
            for gpu in gpus
        )
        or len({gpu["uuid"] for gpu in gpus}) != expected_gpu_count
    ):
        raise LaunchError(f"{role} preflight does not match the production run")
    return {
        "path": os.fspath(path),
        "sha256": report_identity_sha256(report),
        "image_digest": image_digest,
        "gptqmodel": gptqmodel,
        "python": report["python"],
        "torch": report.get("torch"),
        "gpus": gpus,
    }


def remote_worker_configuration(
    args: argparse.Namespace,
    *,
    lock: dict[str, Any],
    coordinator_preflight: dict[str, Any],
) -> dict[str, Any] | None:
    """Bind each declared Spark endpoint to its immutable expert preflight."""

    declarations = getattr(args, "remote_worker", None)
    if not declarations:
        return None
    names: set[str] = set()
    urls: set[str] = set()
    endpoints: list[dict[str, Any]] = []
    preflights: dict[str, dict[str, Any]] = {}
    for declaration in declarations:
        if not isinstance(declaration, (list, tuple)) or len(declaration) != 3:
            raise LaunchError("each --remote-worker requires NAME URL PREFLIGHT")
        name, raw_url, raw_preflight = declaration
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name) is None
            or name in names
        ):
            raise LaunchError("remote worker names must be unique stable identifiers")
        if not isinstance(raw_url, str):
            raise LaunchError("remote worker URL is invalid")
        try:
            parsed = urllib.parse.urlsplit(raw_url)
            port = parsed.port
        except ValueError as error:
            raise LaunchError("remote worker URL is invalid") from error
        if (
            parsed.scheme != "http"
            or not parsed.hostname
            or port is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise LaunchError(
                "remote worker URL must be an explicit internal http://host:port"
            )
        normalized_url = raw_url.rstrip("/")
        if normalized_url in urls:
            raise LaunchError("remote worker URLs must be unique")
        worker_preflight = preflight_identity(
            Path(raw_preflight),
            str(lock.get("revision")),
            role="expert",
        )
        if (
            worker_preflight["gptqmodel"].get("revision") != lock.get("revision")
            or worker_preflight["gptqmodel"].get("source_tree_sha256")
            != lock.get("source_tree_sha256")
        ):
            raise LaunchError(
                f"remote worker `{name}` GPTQModel identity differs from the source lock"
            )
        names.add(name)
        urls.add(normalized_url)
        endpoints.append(
            {
                "name": name,
                "url": normalized_url,
                "preflight_sha256": worker_preflight["sha256"],
                "image_digest": worker_preflight["image_digest"],
            }
        )
        preflights[name] = worker_preflight
    endpoints.sort(key=lambda endpoint: endpoint["name"])
    if len(endpoints) != 4:
        raise LaunchError("production quantization requires exactly four Spark workers")
    if len({endpoint["image_digest"] for endpoint in endpoints}) != 1:
        raise LaunchError("all remote workers must use one identical image digest")
    preflights = {name: preflights[name] for name in sorted(preflights)}
    coordinator_slots = [
        {
            "device": f"cuda:{gpu['index']}",
            "gpu_uuid": gpu["uuid"],
            "preflight_sha256": coordinator_preflight["sha256"],
            "image_digest": coordinator_preflight["image_digest"],
        }
        for gpu in coordinator_preflight["gpus"]
    ]
    # Each Spark admits one executing request and one CPU/network-staged request;
    # its worker-side quantize lock remains the one-kernel-per-GPU barrier. Keep
    # one further orchestration thread per Spark for coordinator postprocessing
    # after the durable packed result releases a pipeline position.
    orchestration_workers = len(coordinator_slots) + (
        REMOTE_PIPELINE_DEPTH + REMOTE_POSTPROCESS_ALLOWANCE
    ) * len(endpoints)
    cuda_workers_per_device = math.ceil(
        orchestration_workers / len(coordinator_slots)
    )
    token_env = getattr(args, "remote_token_env", "DS41RT_EXL3_WORKER_TOKEN")
    timeout_seconds = getattr(args, "remote_timeout_seconds", 7200.0)
    max_attempts = getattr(args, "remote_max_attempts", 2)
    if (
        not isinstance(token_env, str)
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token_env) is None
        or isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or timeout_seconds <= 0
        or isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or not 1 <= max_attempts <= 10
    ):
        raise LaunchError("remote worker resource/authentication limits are invalid")
    return {
        "contract": REMOTE_WORKER_CONTRACT,
        "scheduler": REMOTE_WORKER_SCHEDULER,
        "token_env": token_env,
        "coordinator_slots": coordinator_slots,
        "timeout_seconds": timeout_seconds,
        "max_attempts": max_attempts,
        "orchestration_workers": orchestration_workers,
        "cuda_workers_per_device": cuda_workers_per_device,
        "endpoints": endpoints,
        "preflights": preflights,
    }


def build_plan(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    lock_path = args.gptqmodel_lock.expanduser().resolve(strict=True)
    lock = read_json_object(lock_path)
    source = snapshot_identity(args.snapshot)
    texts, corpus = calibration_stream(args.calibration_jsonl)
    manifest_path = getattr(args, "calibration_manifest", None)
    route_screen_path = getattr(args, "route_screen_report", None)
    route_qualification = getattr(
        args,
        "route_qualification",
        ROUTE_QUALIFICATION_SCREEN,
    )
    mtp_execution_mode = getattr(
        args,
        "mtp_execution_mode",
        MTP_EXECUTION_INTEGRATED,
    )
    if mtp_execution_mode not in MTP_EXECUTION_MODES:
        raise LaunchError("MTP execution mode is invalid")
    if route_qualification not in ROUTE_QUALIFICATION_MODES:
        raise LaunchError("route qualification mode is invalid")
    if route_screen_path is not None and manifest_path is None:
        raise LaunchError(
            "router-screen report requires a calibration manifest"
        )
    if manifest_path is not None and (
        (route_qualification == ROUTE_QUALIFICATION_SCREEN)
        != (route_screen_path is not None)
    ):
        raise LaunchError(
            "route qualification mode and router-screen report disagree"
        )
    evidence = (
        calibration_evidence(
            manifest_path,
            route_screen_path,
            corpus=corpus,
            source=source,
            mtp_execution_mode=mtp_execution_mode,
        )
        if manifest_path is not None
        else None
    )
    toolchain = (
        quantization_toolchain_identity(
            include_route_screen=route_screen_path is not None,
        )
        if evidence is not None
        else None
    )
    bits = getattr(args, "bits", 2)
    recipe = natural_route_recipe(bits)
    coordinator_gpu_count = getattr(args, "coordinator_gpu_count", 2)
    preflight = preflight_identity(
        args.preflight_report,
        str(lock.get("revision")),
        expected_gpu_count=coordinator_gpu_count,
    )
    if preflight["gptqmodel"].get("revision") != lock.get("revision") or preflight[
        "gptqmodel"
    ].get("source_tree_sha256") != lock.get("source_tree_sha256"):
        raise LaunchError("preflight GPTQModel identity differs from the source lock")
    remote_workers = remote_worker_configuration(
        args,
        lock=lock,
        coordinator_preflight=preflight,
    )
    output = args.output.expanduser().resolve()
    offload = args.offload_dir.expanduser().resolve()
    prefix_store = args.mtp_prefix_store.expanduser().resolve()
    raw_run_state = getattr(args, "run_state_dir", None)
    run_state = (
        raw_run_state.expanduser().resolve()
        if raw_run_state is not None
        else output.with_name(f".{output.name}.ds41rt-run")
    )
    raw_projection_root = getattr(args, "projection_checkpoint_dir", None)
    projection_root = (
        raw_projection_root.expanduser().resolve()
        if raw_projection_root is not None
        else run_state / PROJECTION_CHECKPOINT_DIRNAME
    )
    raw_active_source = getattr(args, "active_layer_source_dir", None)
    active_source = (
        raw_active_source.expanduser().resolve()
        if raw_active_source is not None
        else run_state / ACTIVE_LAYER_SOURCE_DIRNAME
    )
    projection_ratio = tuple(
        getattr(args, "mixed_projection_ratio", (3, 5, 8))
    )
    tier_plan_root = run_state / INLINE_MIXED_TIER_PLAN_DIRNAME
    base_mixed_policy = _mixed_policy(
        namespace="base",
        base_bits=bits,
        target_bpw=getattr(args, "base_target_bpw", None),
        projection_ratio=projection_ratio,
        tier_plan_root=tier_plan_root,
    )
    mtp_mixed_policy = _mixed_policy(
        namespace="mtp",
        base_bits=bits,
        target_bpw=getattr(args, "mtp_target_bpw", None),
        projection_ratio=projection_ratio,
        tier_plan_root=tier_plan_root,
    )
    if mtp_mixed_policy is not None and mtp_execution_mode != MTP_EXECUTION_INTEGRATED:
        raise LaunchError("mixed dSpark quantization requires integrated MTP execution")
    inline_mixed = {
        namespace: policy
        for namespace, policy in (
            ("base", base_mixed_policy),
            ("mtp", mtp_mixed_policy),
        )
        if policy is not None
    }
    mtp_anchor_selection = {
        "contract": ANCHOR_SELECTION_CONTRACT,
        "count": int(
            getattr(
                args,
                "mtp_anchor_sample_count",
                DEFAULT_MTP_ANCHOR_SAMPLE_COUNT,
            )
        ),
        "seed": int(
            getattr(
                args,
                "mtp_anchor_sample_seed",
                DEFAULT_MTP_ANCHOR_SAMPLE_SEED,
            )
        ),
    }
    mtp_sequence_anchor_cap = int(
        getattr(args, "mtp_sequence_anchor_cap", 0)
    )
    if mtp_anchor_selection["count"] <= 0 or mtp_sequence_anchor_cap < 0:
        raise LaunchError("integrated MTP anchor selection is invalid")
    mtp_replay_batching = {
        "contract": SEQUENCE_REPLAY_BATCH_CONTRACT,
        "source_sequence_anchor_cap": mtp_sequence_anchor_cap or None,
        "proposal_rows_per_anchor": 5,
    }
    if remote_workers is not None:
        remote_workers["assignment_store"] = os.fspath(
            run_state / REMOTE_ASSIGNMENT_DIRNAME
        )
    independent_paths = (output, run_state, offload, prefix_store)
    if len(set(independent_paths)) != len(independent_paths) or any(
        left.is_relative_to(right) or right.is_relative_to(left)
        for index, left in enumerate(independent_paths)
        for right in independent_paths[index + 1 :]
    ):
        raise LaunchError(
            "output, run-state, offload, and MTP prefix-store paths "
            "must be distinct and non-nested"
        )
    subordinate_paths = (
        (projection_root, run_state / PROJECTION_CHECKPOINT_DIRNAME),
        (active_source, run_state / ACTIVE_LAYER_SOURCE_DIRNAME),
    )
    for path, canonical_child in subordinate_paths:
        if path in independent_paths or any(
            path.is_relative_to(other) or other.is_relative_to(path)
            for other in (output, offload, prefix_store)
        ):
            raise LaunchError("active and checkpoint stores overlap another run path")
        if (path.is_relative_to(run_state) or run_state.is_relative_to(path)) and (
            path != canonical_child
        ):
            raise LaunchError(
                "a run-state child store must use its canonical directory name"
            )
    if projection_root == active_source or projection_root.is_relative_to(
        active_source
    ) or active_source.is_relative_to(projection_root):
        raise LaunchError("projection checkpoints and active source staging overlap")
    family_join = {
        "recipe": recipe,
        "source": source,
        "corpus": corpus,
        "gptqmodel": lock,
        "preflight_sha256": preflight["sha256"],
        "image_digest": preflight["image_digest"],
        "quantizer_seed": EXL3_SEED,
        "quantizer_numerics": {
            "sigma_reg": EXL3_SIGMA_REG,
            "hessian_capture": EXL3_HESSIAN_CAPTURE_CONTRACT,
            "hessian_numerical": EXL3_HESSIAN_NUMERICAL_CONTRACT,
            "hessian_symmetry": EXL3_HESSIAN_SYMMETRY_CONTRACT,
        },
        "bits": bits,
        "codebook": "mcg",
        "module_include": BASE_EXPERT_PATTERN,
        "operator_contract": "ds41rt-deepseek-v4-target-plus-joint-mtp-v1",
        "route_evidence_contract": ROUTE_EVIDENCE_CONTRACT,
        "zero_route_recovery_contract": ZERO_ROUTE_RECOVERY_CONTRACT,
        "mtp_anchor_selection": mtp_anchor_selection,
        "mtp_replay_batching": mtp_replay_batching,
    }
    if inline_mixed:
        family_join["inline_mixed"] = {
            namespace: {
                key: value
                for key, value in policy.items()
                if key != "tier_plan_root"
            }
            for namespace, policy in inline_mixed.items()
        }
    if evidence is not None:
        family_join["calibration_evidence"] = evidence
        family_join["quantization_toolchain"] = toolchain
    if remote_workers is not None:
        family_join["execution_topology"] = {
            "contract": remote_workers["contract"],
            "scheduler": remote_workers["scheduler"],
            "assignment_store": remote_workers["assignment_store"],
            "coordinator": {
                "preflight_sha256": preflight["sha256"],
                "image_digest": preflight["image_digest"],
            },
            "coordinator_slots": remote_workers["coordinator_slots"],
            "workers": [
                {
                    "name": endpoint["name"],
                    "preflight_sha256": endpoint["preflight_sha256"],
                    "image_digest": endpoint["image_digest"],
                }
                for endpoint in remote_workers["endpoints"]
            ],
        }
    provenance = {
        "family_join": family_join,
        "run": {
            "coordinator": preflight,
            "output": os.fspath(output),
            "run_state": os.fspath(run_state),
            "offload": os.fspath(offload),
            "mtp_prefix_store": os.fspath(prefix_store),
            "active_layer_source": os.fspath(active_source),
            "target_batch_size": args.batch_size,
            "capture_batch_checkpoint_interval": (
                getattr(
                    args,
                    "capture_batch_checkpoint_interval",
                    DEFAULT_CAPTURE_BATCH_CHECKPOINT_INTERVAL,
                )
            ),
            "mtp_replay_batch_size": args.mtp_replay_batch_size,
            "mtp_execution_mode": mtp_execution_mode,
            "mtp_anchor_selection": mtp_anchor_selection,
            "mtp_replay_batching": mtp_replay_batching,
            "projection_checkpoint": {
                "contract": PROJECTION_CHECKPOINT_CONTRACT,
                "root": os.fspath(projection_root),
            },
        },
    }
    if remote_workers is not None:
        provenance["run"]["remote_workers"] = remote_workers
    layer_boundary = {
        "contract": BOUNDARY_CONTRACT,
        "root": os.fspath(run_state / LAYER_BOUNDARY_DIRNAME),
        "retention": "latest-complete-layer",
        "dtype": "bfloat16",
    }
    provenance["run"]["layer_boundary"] = layer_boundary
    plan = {
        "schema": PLAN_SCHEMA,
        "recipe": recipe,
        "source": source,
        "corpus": corpus,
        "preflight": preflight,
        "output": os.fspath(output),
        "run_state_dir": os.fspath(run_state),
        "projection_checkpoint_dir": os.fspath(projection_root),
        "active_layer_source_dir": os.fspath(active_source),
        "offload_dir": os.fspath(offload),
        "mtp_prefix_store": os.fspath(prefix_store),
        "target_batch_size": args.batch_size,
        "capture_batch_checkpoint_interval": (
            getattr(
                args,
                "capture_batch_checkpoint_interval",
                DEFAULT_CAPTURE_BATCH_CHECKPOINT_INTERVAL,
            )
        ),
        "mtp_replay_batch_size": args.mtp_replay_batch_size,
        "mtp_execution_mode": mtp_execution_mode,
        "mtp_anchor_selection": mtp_anchor_selection,
        "mtp_replay_batching": mtp_replay_batching,
        "projection_checkpoint": provenance["run"]["projection_checkpoint"],
        "layer_boundary": layer_boundary,
        "memory_safety": {
            "host_rss_limit_bytes": HOST_RSS_LIMIT_BYTES,
            "cuda_allocation_limit_bytes": CUDA_ALLOCATION_LIMIT_BYTES,
            "telemetry_interval_batches": MEMORY_TELEMETRY_INTERVAL_BATCHES,
            "spill_policy": "fail-closed-no-cpu-or-managed-memory",
        },
        "inline_mixed": inline_mixed,
        "remote_workers": remote_workers,
        "exl3": {
            "bits": bits,
            "codebook": "mcg",
            "seed": EXL3_SEED,
            "module_include": [BASE_EXPERT_PATTERN],
            "fallback": None,
            "out_scales": "auto",
            "sigma_reg": EXL3_SIGMA_REG,
            "hessian_capture": EXL3_HESSIAN_CAPTURE_CONTRACT,
            "hessian_numerical": EXL3_HESSIAN_NUMERICAL_CONTRACT,
            "hessian_symmetry": EXL3_HESSIAN_SYMMETRY_CONTRACT,
            "zero_route_recovery": {
                "contract": ZERO_ROUTE_RECOVERY_CONTRACT,
                "trigger": ZERO_ROUTE_RECOVERY_TRIGGER,
                "sample_source": ZERO_ROUTE_RECOVERY_SAMPLE_SOURCE,
                "capture_method": ZERO_ROUTE_RECOVERY_CAPTURE_METHOD,
                "selection_policy": ZERO_ROUTE_RECOVERY_SELECTION_POLICY,
                "candidate_rank_min": ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MIN,
                "candidate_rank_max": ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MAX,
                "target_sample_count": ZERO_ROUTE_RECOVERY_TARGET_SAMPLE_COUNT,
                "identity_calibration_policy": ZERO_ROUTE_RECOVERY_IDENTITY_POLICY,
                "scope": "all-learned-top-k-routers",
            },
        },
        "ledger_provenance": provenance,
    }
    if evidence is not None:
        plan["calibration_evidence"] = evidence
        plan["quantization_toolchain"] = toolchain
    plan["plan_sha256"] = hashlib.sha256(canonical_json(plan)).hexdigest()
    return plan, texts


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as target:
        target.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _bound_record(value: dict[str, Any], digest_field: str) -> dict[str, Any]:
    if digest_field in value:
        raise LaunchError(f"record already contains reserved field {digest_field}")
    clean = dict(value)
    clean[digest_field] = hashlib.sha256(canonical_json(value)).hexdigest()
    return clean


def _validate_bound_record(
    value: dict[str, Any],
    *,
    digest_field: str,
    label: str,
) -> None:
    digest = value.get(digest_field)
    body = {key: item for key, item in value.items() if key != digest_field}
    if (
        not isinstance(digest, str)
        or SHA256_RE.fullmatch(digest) is None
        or hashlib.sha256(canonical_json(body)).hexdigest() != digest
    ):
        raise LaunchError(f"{label} digest is invalid")


def _read_execution_upgrade(
    run_state: Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    path = run_state / EXECUTION_UPGRADE_FILENAME
    if not path.is_file() or path.is_symlink():
        raise LaunchError("parent plan requires a regular execution-upgrade record")
    upgrade = read_json_object(path)
    _validate_bound_record(
        upgrade,
        digest_field="upgrade_sha256",
        label="execution upgrade",
    )
    if (
        upgrade.get("schema") != EXECUTION_UPGRADE_SCHEMA
        or upgrade.get("parent_plan_sha256") != plan.get("plan_sha256")
    ):
        raise LaunchError("execution upgrade does not bind the parent plan")
    history = run_state / EXECUTION_UPGRADE_HISTORY_DIRNAME
    history_records: dict[str, dict[str, Any]] = {}
    if history.exists():
        _regular_directory(history, "execution-upgrade history")
        for archived in history.iterdir():
            match = re.fullmatch(r"([0-9a-f]{64})\.json", archived.name)
            if match is None or not archived.is_file() or archived.is_symlink():
                raise LaunchError("execution-upgrade history contains an unsafe entry")
            record = read_json_object(archived)
            _validate_bound_record(
                record,
                digest_field="upgrade_sha256",
                label="archived execution upgrade",
            )
            digest = match.group(1)
            if (
                record.get("upgrade_sha256") != digest
                or record.get("schema") != EXECUTION_UPGRADE_SCHEMA
                or record.get("parent_plan_sha256") != plan.get("plan_sha256")
            ):
                raise LaunchError("archived execution upgrade is inconsistent")
            history_records[digest] = record
    links = [
        upgrade.get(field)
        for field in (
            "previous_upgrade_sha256",
            "previous_failed_upgrade_sha256",
        )
        if upgrade.get(field) is not None
    ]
    if len(links) > 1:
        raise LaunchError("execution upgrade has ambiguous history links")
    cursor = links[0] if links else None
    visited: set[str] = set()
    while cursor is not None:
        if cursor in visited or cursor not in history_records:
            raise LaunchError("execution-upgrade history chain is incomplete")
        visited.add(cursor)
        record = history_records[cursor]
        links = [
            record.get(field)
            for field in (
                "previous_upgrade_sha256",
                "previous_failed_upgrade_sha256",
            )
            if record.get(field) is not None
        ]
        if len(links) > 1:
            raise LaunchError("archived execution upgrade has ambiguous history links")
        cursor = links[0] if links else None
    return upgrade


def _journal_resume_identity(path: Path) -> dict[str, Any]:
    """Bind the exact journal frontier present when execution code changes."""

    if not path.is_file() or path.is_symlink():
        raise LaunchError("execution upgrade requires a regular error journal")
    digest = hashlib.sha256()
    records = 0
    total_bytes = 0
    last_byte = b""
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
            records += block.count(b"\n")
            total_bytes += len(block)
            last_byte = block[-1:]
    if total_bytes and last_byte != b"\n":
        raise LaunchError("execution-upgrade journal ends with a partial record")
    return {
        "bytes": total_bytes,
        "records": records,
        "sha256": digest.hexdigest(),
    }


def _latest_boundary_resume_identity(
    run_state: Path,
    plan: dict[str, Any],
) -> dict[str, Any] | None:
    """Authenticate the rolling boundary named by a chained code upgrade."""

    root = run_state / LAYER_BOUNDARY_DIRNAME
    if not root.exists():
        return None
    _regular_directory(root, "layer-boundary directory")
    entries = list(root.iterdir())
    if not entries:
        return None
    if len(entries) != 1:
        raise LaunchError("execution upgrade requires one retained layer boundary")
    directory = entries[0]
    match = BOUNDARY_DIRECTORY_RE.fullmatch(directory.name)
    if match is None or not directory.is_dir() or directory.is_symlink():
        raise LaunchError("execution upgrade found an unsafe layer boundary")
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise LaunchError("execution upgrade boundary manifest is unavailable")
    manifest = read_json_object(manifest_path)
    digest = manifest.get("manifest_sha256")
    body = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if (
        manifest.get("schema") != "ds41rt.deepseek-v4-layer-boundary"
        or manifest.get("schema_version") != 2
        or manifest.get("payload_hash_algorithm") != "xxh3-128"
        or manifest.get("plan_sha256") != plan.get("plan_sha256")
        or not isinstance(digest, str)
        or SHA256_RE.fullmatch(digest) is None
        or hashlib.sha256(canonical_json(body)).hexdigest() != digest
        or match.group("digest") != digest[:16]
        or int(match.group("layer")) != manifest.get("layer_index")
    ):
        raise LaunchError("execution-upgrade layer boundary failed validation")
    return {
        "directory": directory.name,
        "layer_index": manifest["layer_index"],
        "layer_name": manifest.get("layer_name"),
        "manifest_sha256": digest,
        "activation_batches": manifest.get("activation_batches"),
        "activation_bytes": manifest.get("activation_bytes"),
        "completed_projection_entries": len(
            manifest.get("completed_projection_entries", ())
        ),
    }


def _validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema") not in SUPPORTED_PLAN_SCHEMAS:
        raise LaunchError(
            "plan does not use a supported DeepSeek V4 GPTQModel schema"
        )
    digest = plan.get("plan_sha256")
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if (
        not isinstance(digest, str)
        or SHA256_RE.fullmatch(digest) is None
        or hashlib.sha256(canonical_json(body)).hexdigest() != digest
    ):
        raise LaunchError("plan digest is invalid")
    if plan.get("mtp_execution_mode", MTP_EXECUTION_INTEGRATED) not in (
        MTP_EXECUTION_MODES
    ):
        raise LaunchError("plan MTP execution mode is invalid")
    exl3 = plan.get("exl3")
    if not isinstance(exl3, dict) or exl3.get("bits") not in {2, 3}:
        raise LaunchError("plan EXL3 bitrate is invalid")
    capture_checkpoint_interval = plan.get(
        "capture_batch_checkpoint_interval", 1
    )
    if (
        isinstance(capture_checkpoint_interval, bool)
        or not isinstance(capture_checkpoint_interval, int)
        or capture_checkpoint_interval <= 0
    ):
        raise LaunchError("plan capture checkpoint interval is invalid")
    if plan.get("schema") in STRICT_STORAGE_PLAN_SCHEMAS:
        checkpoint = plan.get("projection_checkpoint")
        path_fields = (
            "output",
            "run_state_dir",
            "projection_checkpoint_dir",
            "active_layer_source_dir",
            "offload_dir",
            "mtp_prefix_store",
        )
        if any(
            not isinstance(plan.get(field), str)
            or not plan[field]
            or not Path(plan[field]).is_absolute()
            for field in path_fields
        ) or (
            not isinstance(checkpoint, dict)
            or checkpoint.get("root") != plan["projection_checkpoint_dir"]
        ):
            raise LaunchError("plan storage roots are invalid or inconsistent")
        memory_safety = plan.get("memory_safety")
        if memory_safety != {
            "host_rss_limit_bytes": HOST_RSS_LIMIT_BYTES,
            "cuda_allocation_limit_bytes": CUDA_ALLOCATION_LIMIT_BYTES,
            "telemetry_interval_batches": MEMORY_TELEMETRY_INTERVAL_BATCHES,
            "spill_policy": "fail-closed-no-cpu-or-managed-memory",
        }:
            raise LaunchError("plan memory-safety contract is invalid")
    inline_mixed = plan.get("inline_mixed", {})
    if not isinstance(inline_mixed, dict) or not set(inline_mixed).issubset(
        {"base", "mtp"}
    ):
        raise LaunchError("plan inline-mixed policies are invalid")
    expected_tier_root = os.fspath(
        Path(plan["run_state_dir"]) / INLINE_MIXED_TIER_PLAN_DIRNAME
    )
    for namespace, policy in inline_mixed.items():
        extra = policy.get("extra_bits") if isinstance(policy, dict) else None
        ratio = policy.get("projection_ratio") if isinstance(policy, dict) else None
        if (
            policy.get("schema") != INLINE_MIXED_SCHEMA
            or policy.get("schema_version") != 1
            or policy.get("namespace") != namespace
            or policy.get("base_bits") != exl3["bits"]
            or policy.get("upgrade_bits") != exl3["bits"] + 1
            or policy.get("tier_plan_root") != expected_tier_root
            or policy.get("score_kind") != INLINE_MIXED_SCORE
            or not isinstance(extra, dict)
            or set(extra) != {"numerator", "denominator"}
            or any(
                isinstance(extra.get(key), bool)
                or not isinstance(extra.get(key), int)
                or extra[key] <= 0
                for key in ("numerator", "denominator")
            )
            or not isinstance(ratio, dict)
            or set(ratio) != {"w1", "w3", "w2"}
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in ratio.values()
            )
        ):
            raise LaunchError("plan inline-mixed policy contract is invalid")
        fraction = Fraction(extra["numerator"], extra["denominator"])
        if not 0 < fraction < 1:
            raise LaunchError("plan inline-mixed target is outside K2/K3")
    if "mtp" in inline_mixed and plan.get("mtp_execution_mode") != MTP_EXECUTION_INTEGRATED:
        raise LaunchError("plan mixed dSpark policy is not integrated")
    selection = plan.get("mtp_anchor_selection")
    batching = plan.get("mtp_replay_batching")
    family_join = plan.get("ledger_provenance", {}).get("family_join", {})
    run = plan.get("ledger_provenance", {}).get("run", {})
    if (
        not isinstance(selection, dict)
        or selection.get("contract") != ANCHOR_SELECTION_CONTRACT
        or isinstance(selection.get("count"), bool)
        or not isinstance(selection.get("count"), int)
        or selection["count"] <= 0
        or isinstance(selection.get("seed"), bool)
        or not isinstance(selection.get("seed"), int)
        or not isinstance(batching, dict)
        or batching.get("contract") != SEQUENCE_REPLAY_BATCH_CONTRACT
        or batching.get("proposal_rows_per_anchor") != 5
        or (
            batching.get("source_sequence_anchor_cap") is not None
            and (
                isinstance(batching["source_sequence_anchor_cap"], bool)
                or not isinstance(batching["source_sequence_anchor_cap"], int)
                or batching["source_sequence_anchor_cap"] <= 0
            )
        )
        or family_join.get("mtp_anchor_selection") != selection
        or family_join.get("mtp_replay_batching") != batching
        or run.get("mtp_anchor_selection") != selection
        or run.get("mtp_replay_batching") != batching
    ):
        raise LaunchError("plan integrated MTP replay contract is invalid")


def _regular_directory(path: Path, label: str) -> None:
    if not path.is_dir() or path.is_symlink():
        raise LaunchError(f"{label} is not a regular directory: {path}")


def _empty_or_missing_directory(path: Path, label: str) -> None:
    if path.is_symlink():
        raise LaunchError(f"{label} is a symbolic link: {path}")
    if not path.exists():
        return
    _regular_directory(path, label)
    if any(path.iterdir()):
        raise LaunchError(f"{label} is not empty: {path}")


def _export_stage_path(plan: dict[str, Any]) -> Path:
    """Return a staging path that can be atomically renamed to the output.

    The run-state and final output are commonly separate container bind mounts.
    Even when both host paths reside on the same filesystem, Linux exposes the
    mounts as distinct devices inside the container and ``os.replace`` fails
    with ``EXDEV``.  A hidden sibling of the output necessarily shares the
    output mount and preserves atomic directory publication.
    """

    output = Path(plan["output"])
    return output.parent / f".{output.name}.{EXPORT_STAGE_DIRNAME}"


def _initialize_empty_error_journal(run_state: Path) -> None:
    """Durably create the ledger before recovery discovery can inspect it."""

    path = run_state / ERROR_JOURNAL_FILENAME
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o644,
        )
    except FileExistsError as error:
        raise LaunchError(f"run-state error journal already exists: {path}") from error
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(run_state, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _validate_run_state_entries(
    run_state: Path,
    *,
    inline_mixed: bool,
) -> None:
    allowed = {
        PLAN_FILENAME,
        EXECUTION_UPGRADE_FILENAME,
        ERROR_JOURNAL_FILENAME,
        DIRECT_STATE_PREFLIGHT_FILENAME,
        REMOTE_ASSIGNMENT_DIRNAME,
        PROJECTION_CHECKPOINT_DIRNAME,
        ACTIVE_LAYER_SOURCE_DIRNAME,
        LAYER_BOUNDARY_DIRNAME,
        CAPTURE_FRONTIER_DIRNAME,
        CAPTURE_BATCH_SPOOL_DIRNAME,
        POST_QUANT_REPLAY_DIRNAME,
        MTP_ACTIVATION_DIRNAME,
        EXECUTION_UPGRADE_HISTORY_DIRNAME,
        EXPORT_STAGE_DIRNAME,
        BASE_PREFIX_RUN_FILENAME,
    }
    if inline_mixed:
        allowed.update(
            {
                INLINE_MIXED_CANDIDATE_JOURNAL_FILENAME,
                INLINE_MIXED_TIER_PLAN_DIRNAME,
            }
        )
    unexpected = sorted(
        path.name for path in run_state.iterdir() if path.name not in allowed
    )
    if unexpected:
        raise LaunchError(
            "run-state directory contains unexpected entries: " + ", ".join(unexpected)
        )
    for name in (
        PLAN_FILENAME,
        EXECUTION_UPGRADE_FILENAME,
        ERROR_JOURNAL_FILENAME,
        DIRECT_STATE_PREFLIGHT_FILENAME,
        BASE_PREFIX_RUN_FILENAME,
    ):
        path = run_state / name
        if path.exists() and (not path.is_file() or path.is_symlink()):
            raise LaunchError(f"run-state entry is not a regular file: {path}")
    if inline_mixed:
        path = run_state / INLINE_MIXED_CANDIDATE_JOURNAL_FILENAME
        if path.exists() and (not path.is_file() or path.is_symlink()):
            raise LaunchError(f"run-state entry is not a regular file: {path}")
    for name in (
        PROJECTION_CHECKPOINT_DIRNAME,
        ACTIVE_LAYER_SOURCE_DIRNAME,
        REMOTE_ASSIGNMENT_DIRNAME,
        LAYER_BOUNDARY_DIRNAME,
        CAPTURE_FRONTIER_DIRNAME,
        CAPTURE_BATCH_SPOOL_DIRNAME,
        POST_QUANT_REPLAY_DIRNAME,
        MTP_ACTIVATION_DIRNAME,
        EXECUTION_UPGRADE_HISTORY_DIRNAME,
        EXPORT_STAGE_DIRNAME,
    ):
        path = run_state / name
        if path.exists():
            _regular_directory(path, "run-state entry")
    if inline_mixed:
        path = run_state / INLINE_MIXED_TIER_PLAN_DIRNAME
        if path.exists():
            _regular_directory(path, "run-state entry")


def _artifact_file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise LaunchError(f"artifact entry is not a regular file: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while block := source.read(8 * 1024 * 1024):
                digest.update(block)
            os.fsync(source.fileno())
    except OSError as exc:
        raise LaunchError(f"cannot hash artifact file {path}: {exc}") from exc
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise LaunchError(f"artifact file changed while hashing: {path}")
    return {"bytes": after.st_size, "sha256": digest.hexdigest()}


def _artifact_paths(root: Path) -> list[Path]:
    _regular_directory(root, "artifact root")
    paths: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise LaunchError(f"artifact contains a symbolic link: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise LaunchError(f"artifact contains an unsupported entry: {path}")
        paths.append(path)
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def write_artifact_manifest(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    excluded = {ARTIFACT_MANIFEST_FILENAME, RUN_FILENAME}
    records = {}
    for path in _artifact_paths(root):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        records[relative] = _artifact_file_identity(path)
    if PLAN_FILENAME not in records or not any(
        name.endswith(".safetensors") for name in records
    ):
        raise LaunchError("export does not contain its plan and safetensor payload")
    body = {
        "schema": ARTIFACT_MANIFEST_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "files": records,
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records.values()),
    }
    manifest = _bound_record(body, "manifest_sha256")
    atomic_json(root / ARTIFACT_MANIFEST_FILENAME, manifest)
    return manifest


def validate_published_artifact(
    root: Path,
    plan: dict[str, Any],
    *,
    verify_file_hashes: bool,
) -> dict[str, Any]:
    _validate_plan(plan)
    _regular_directory(root, "published artifact")
    saved_plan = read_json_object(root / PLAN_FILENAME)
    if saved_plan != plan:
        raise LaunchError("published artifact plan differs from the requested run")
    manifest = read_json_object(root / ARTIFACT_MANIFEST_FILENAME)
    run = read_json_object(root / RUN_FILENAME)
    _validate_bound_record(
        manifest,
        digest_field="manifest_sha256",
        label="artifact manifest",
    )
    _validate_bound_record(run, digest_field="run_sha256", label="run manifest")
    records = manifest.get("files")
    if (
        manifest.get("schema") != ARTIFACT_MANIFEST_SCHEMA
        or manifest.get("plan_sha256") != plan["plan_sha256"]
        or not isinstance(records, dict)
        or manifest.get("file_count") != len(records)
        or manifest.get("total_bytes")
        != sum(
            record.get("bytes", -1)
            for record in records.values()
            if isinstance(record, dict)
        )
        or run.get("schema") != RUN_SCHEMA
        or run.get("status") != "complete"
        or run.get("plan_sha256") != plan["plan_sha256"]
        or run.get("artifact_manifest_sha256") != manifest.get("manifest_sha256")
        or isinstance(run.get("mtp_replay_batches"), bool)
        or not isinstance(run.get("mtp_replay_batches"), int)
        or run["mtp_replay_batches"] <= 0
    ):
        raise LaunchError("published artifact manifests are inconsistent")
    expected_paths = set(records) | {ARTIFACT_MANIFEST_FILENAME, RUN_FILENAME}
    actual_paths = {
        path.relative_to(root).as_posix() for path in _artifact_paths(root)
    }
    if actual_paths != expected_paths:
        raise LaunchError("published artifact file set differs from its manifest")
    for relative, record in records.items():
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(record, dict)
            or isinstance(record.get("bytes"), bool)
            or not isinstance(record.get("bytes"), int)
            or record["bytes"] < 0
            or not isinstance(record.get("sha256"), str)
            or SHA256_RE.fullmatch(record["sha256"]) is None
        ):
            raise LaunchError("published artifact contains an invalid file record")
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != record["bytes"]
        ):
            raise LaunchError(f"published artifact file differs in size: {relative}")
        if verify_file_hashes and _artifact_file_identity(path) != record:
            raise LaunchError(f"published artifact file failed hashing: {relative}")
    return run


def prepare_run(plan: dict[str, Any], *, resume: bool) -> bool:
    """Prepare exact unpublished state; return True for an existing complete output."""

    _validate_plan(plan)
    output = Path(plan["output"])
    run_state = Path(plan["run_state_dir"])
    projection_root = Path(plan["projection_checkpoint"]["root"])
    active_source = Path(
        plan.get(
            "active_layer_source_dir",
            run_state / ACTIVE_LAYER_SOURCE_DIRNAME,
        )
    )
    offload = Path(plan["offload_dir"])
    prefix_root = Path(plan["mtp_prefix_store"])
    output.parent.mkdir(parents=True, exist_ok=True)

    if resume and (output.exists() or output.is_symlink()):
        validate_published_artifact(output, plan, verify_file_hashes=True)
        return True
    if resume:
        _regular_directory(run_state, "run-state directory")
        saved_plan = read_json_object(run_state / PLAN_FILENAME)
        if saved_plan != plan:
            raise LaunchError("run-state plan differs from the requested run")
        _validate_run_state_entries(
            run_state,
            inline_mixed=bool(plan.get("inline_mixed")),
        )
        journal = run_state / ERROR_JOURNAL_FILENAME
        if not journal.is_file() or journal.is_symlink():
            raise LaunchError(f"run-state error journal is unavailable: {journal}")
        for path, label in (
            (projection_root, "projection-checkpoint directory"),
            (active_source, "active-layer source directory"),
            (offload, "offload directory"),
            (prefix_root, "MTP prefix-store directory"),
        ):
            _regular_directory(path, label)
        export_stage = _export_stage_path(plan)
        if export_stage.exists() or export_stage.is_symlink():
            _regular_directory(export_stage, "partial export directory")
            # Keep unpublished, regular files so GPTQModel can authenticate and
            # reuse complete shards. Partial, corrupt, or differently planned
            # shards are rewritten by the streaming writer before publication.
            _artifact_paths(export_stage)
        return False

    _empty_or_missing_directory(output, "output directory")
    _empty_or_missing_directory(
        _export_stage_path(plan), "partial export directory"
    )
    _empty_or_missing_directory(run_state, "run-state directory")
    _empty_or_missing_directory(
        projection_root, "projection-checkpoint directory"
    )
    _empty_or_missing_directory(active_source, "active-layer source directory")
    _empty_or_missing_directory(offload, "offload directory")
    _empty_or_missing_directory(prefix_root, "MTP prefix-store directory")
    if output.exists():
        output.rmdir()
    if run_state.exists():
        run_state.rmdir()
    run_state.mkdir(parents=True)
    atomic_json(run_state / PLAN_FILENAME, plan)
    _initialize_empty_error_journal(run_state)
    projection_root.mkdir(parents=True, exist_ok=True)
    active_source.mkdir(parents=True, exist_ok=True)
    offload.mkdir(parents=True, exist_ok=True)
    prefix_root.mkdir(parents=True, exist_ok=True)
    return False


def _validate_upgrade_remote_workers(
    args: argparse.Namespace,
    plan: dict[str, Any],
) -> None:
    """Validate unchanged Spark declarations against the parent plan."""

    expected = plan.get("remote_workers")
    declarations = getattr(args, "remote_worker", None)
    if expected is None:
        if declarations:
            raise LaunchError("execution upgrade adds remote workers")
        return
    if not isinstance(expected, dict) or not declarations:
        raise LaunchError("execution upgrade omits the parent remote workers")
    ledger = plan.get("ledger_provenance")
    family_join = ledger.get("family_join") if isinstance(ledger, dict) else None
    parent_lock = (
        family_join.get("gptqmodel") if isinstance(family_join, dict) else None
    )
    if not isinstance(parent_lock, dict):
        raise LaunchError("parent plan has no GPTQModel source identity")
    expected_endpoints = expected.get("endpoints")
    expected_preflights = expected.get("preflights")
    if not isinstance(expected_endpoints, list) or not isinstance(
        expected_preflights, dict
    ):
        raise LaunchError("parent remote-worker contract is malformed")
    observed_endpoints: list[dict[str, Any]] = []
    observed_preflights: dict[str, dict[str, Any]] = {}
    for declaration in declarations:
        if not isinstance(declaration, (list, tuple)) or len(declaration) != 3:
            raise LaunchError("each --remote-worker requires NAME URL PREFLIGHT")
        name, raw_url, raw_preflight = declaration
        try:
            parsed = urllib.parse.urlsplit(raw_url)
            port = parsed.port
        except (TypeError, ValueError) as error:
            raise LaunchError("remote worker URL is invalid") from error
        if (
            not isinstance(name, str)
            or parsed.scheme != "http"
            or not parsed.hostname
            or port is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise LaunchError("remote worker declaration is invalid")
        worker_preflight = preflight_identity(
            Path(raw_preflight),
            str(parent_lock.get("revision")),
            role="expert",
        )
        if (
            worker_preflight["gptqmodel"].get("source_tree_sha256")
            != parent_lock.get("source_tree_sha256")
        ):
            raise LaunchError(
                f"remote worker `{name}` GPTQModel identity differs from the parent"
            )
        observed_endpoints.append(
            {
                "name": name,
                "url": raw_url.rstrip("/"),
                "preflight_sha256": worker_preflight["sha256"],
                "image_digest": worker_preflight["image_digest"],
            }
        )
        if name in observed_preflights:
            raise LaunchError("execution upgrade repeats a remote worker")
        observed_preflights[name] = worker_preflight
    observed_endpoints.sort(key=lambda endpoint: endpoint["name"])
    observed_preflights = {
        name: observed_preflights[name] for name in sorted(observed_preflights)
    }
    if (
        observed_endpoints != expected_endpoints
        or observed_preflights != expected_preflights
        or getattr(args, "remote_token_env", "DS41RT_EXL3_WORKER_TOKEN")
        != expected.get("token_env")
        or getattr(args, "remote_timeout_seconds", 7200.0)
        != expected.get("timeout_seconds")
        or getattr(args, "remote_max_attempts", 2) != expected.get("max_attempts")
    ):
        raise LaunchError("execution upgrade changes the parent remote-worker contract")


def build_execution_upgrade(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Bind new checkpoint-only execution code to an immutable parent plan."""

    output = args.output.expanduser().resolve()
    raw_run_state = getattr(args, "run_state_dir", None)
    run_state = (
        raw_run_state.expanduser().resolve()
        if raw_run_state is not None
        else output.with_name(f".{output.name}.ds41rt-run")
    )
    _regular_directory(run_state, "run-state directory")
    plan = read_json_object(run_state / PLAN_FILENAME)
    _validate_plan(plan)
    _validate_run_state_entries(
        run_state,
        inline_mixed=bool(plan.get("inline_mixed")),
    )

    source = snapshot_identity(args.snapshot)
    texts, corpus = calibration_stream(args.calibration_jsonl)
    expected_paths = {
        "output": os.fspath(output),
        "run_state_dir": os.fspath(run_state),
        "offload_dir": os.fspath(args.offload_dir.expanduser().resolve()),
        "mtp_prefix_store": os.fspath(
            args.mtp_prefix_store.expanduser().resolve()
        ),
    }
    upgrade_bits = getattr(args, "bits", 2)
    upgrade_ratio = tuple(
        getattr(args, "mixed_projection_ratio", (3, 5, 8))
    )
    expected_inline_mixed = {
        namespace: policy
        for namespace, policy in (
            (
                "base",
                _mixed_policy(
                    namespace="base",
                    base_bits=upgrade_bits,
                    target_bpw=getattr(args, "base_target_bpw", None),
                    projection_ratio=upgrade_ratio,
                    tier_plan_root=run_state / INLINE_MIXED_TIER_PLAN_DIRNAME,
                ),
            ),
            (
                "mtp",
                _mixed_policy(
                    namespace="mtp",
                    base_bits=upgrade_bits,
                    target_bpw=getattr(args, "mtp_target_bpw", None),
                    projection_ratio=upgrade_ratio,
                    tier_plan_root=run_state / INLINE_MIXED_TIER_PLAN_DIRNAME,
                ),
            ),
        )
        if policy is not None
    }
    if plan.get("schema") == PLAN_SCHEMA:
        raw_checkpoint = getattr(args, "projection_checkpoint_dir", None)
        raw_active_source = getattr(args, "active_layer_source_dir", None)
        expected_paths.update(
            {
                "projection_checkpoint_dir": os.fspath(
                    raw_checkpoint.expanduser().resolve()
                    if raw_checkpoint is not None
                    else run_state / PROJECTION_CHECKPOINT_DIRNAME
                ),
                "active_layer_source_dir": os.fspath(
                    raw_active_source.expanduser().resolve()
                    if raw_active_source is not None
                    else run_state / ACTIVE_LAYER_SOURCE_DIRNAME
                ),
            }
        )
    if (
        any(plan.get(key) != value for key, value in expected_paths.items())
        or plan.get("source") != source
        or plan.get("corpus") != corpus
        or plan.get("exl3", {}).get("bits") != getattr(args, "bits", 2)
        or plan.get("inline_mixed", {}) != expected_inline_mixed
        or plan.get("target_batch_size") != args.batch_size
        or (
            "capture_batch_checkpoint_interval" in plan
            and plan["capture_batch_checkpoint_interval"]
            != getattr(
                args,
                "capture_batch_checkpoint_interval",
                DEFAULT_CAPTURE_BATCH_CHECKPOINT_INTERVAL,
            )
        )
        or plan.get("mtp_replay_batch_size") != args.mtp_replay_batch_size
        or plan.get("mtp_execution_mode", MTP_EXECUTION_INTEGRATED)
        != getattr(args, "mtp_execution_mode", MTP_EXECUTION_INTEGRATED)
    ):
        raise LaunchError("execution upgrade inputs differ from the parent plan")
    _validate_upgrade_remote_workers(args, plan)

    lock_path = args.gptqmodel_lock.expanduser().resolve(strict=True)
    current_lock = read_json_object(lock_path)
    current_preflight = preflight_identity(
        args.preflight_report,
        str(current_lock.get("revision")),
        expected_gpu_count=getattr(args, "coordinator_gpu_count", 2),
    )
    if (
        current_preflight["gptqmodel"].get("source_tree_sha256")
        != current_lock.get("source_tree_sha256")
    ):
        raise LaunchError("upgrade preflight GPTQModel identity differs from its lock")
    parent_preflight = plan.get("preflight")
    ledger = plan.get("ledger_provenance")
    family_join = ledger.get("family_join") if isinstance(ledger, dict) else None
    parent_lock = (
        family_join.get("gptqmodel") if isinstance(family_join, dict) else None
    )
    if not isinstance(parent_preflight, dict) or not isinstance(parent_lock, dict):
        raise LaunchError("parent plan lacks execution provenance")
    if (
        current_preflight["image_digest"] == parent_preflight.get("image_digest")
        and current_lock == parent_lock
    ):
        raise LaunchError("execution upgrade does not change the execution image")

    stable_preflight = {
        key: current_preflight[key]
        for key in ("image_digest", "gptqmodel", "python", "torch", "gpus")
    }
    upgrade_body = {
        "schema": EXECUTION_UPGRADE_SCHEMA,
        "parent_plan_sha256": plan["plan_sha256"],
        "parent_execution": {
            "image_digest": parent_preflight.get("image_digest"),
            "gptqmodel": parent_lock,
        },
        "upgraded_execution": stable_preflight,
        "change_contract": {
            "purpose": "rolling-layer-boundary-resume",
            "layer_boundary": BOUNDARY_CONTRACT,
            "capture_frontier": CAPTURE_FRONTIER_CONTRACT,
            "projection_restore": "packed-checkpoint-direct-v1",
            "quantization_algorithm": "declared-seeded-true-sequential-v1",
            "concurrent_rng": "projection-local-generator-v1",
            "capture_batch_checkpoint_interval": getattr(
                args,
                "capture_batch_checkpoint_interval",
                DEFAULT_CAPTURE_BATCH_CHECKPOINT_INTERVAL,
            ),
            "remote_workers": "unchanged-parent-plan",
            "mtp_execution": (
                "sampled-external-overlay-handoff"
                if getattr(args, "mtp_overlay_handoff", False)
                else "unchanged-parent-plan"
            ),
        },
    }
    boundary_identity = _latest_boundary_resume_identity(run_state, plan)
    if boundary_identity is not None:
        upgrade_body["resume_state"] = {
            "contract": "latest-boundary-plus-journal-v1",
            "layer_boundary": boundary_identity,
            "error_journal": _journal_resume_identity(
                run_state / ERROR_JOURNAL_FILENAME
            ),
        }
        upgrade_body["change_contract"]["resume_state"] = (
            "content-bound-immutable-frontier"
        )
    upgrade_path = run_state / EXECUTION_UPGRADE_FILENAME
    if upgrade_path.exists():
        saved_upgrade = _read_execution_upgrade(run_state, plan)
        expected_change_contract = upgrade_body["change_contract"]
        if (
            saved_upgrade.get("parent_execution")
            == upgrade_body["parent_execution"]
            and saved_upgrade.get("upgraded_execution")
            == upgrade_body["upgraded_execution"]
            and isinstance(saved_upgrade.get("change_contract"), dict)
            and saved_upgrade.get("resume_state")
            == upgrade_body.get("resume_state")
            and all(
                saved_upgrade["change_contract"].get(key) == value
                for key, value in expected_change_contract.items()
            )
        ):
            return plan, texts, saved_upgrade

        if boundary_identity is not None:
            upgrade_body["previous_upgrade_sha256"] = saved_upgrade[
                "upgrade_sha256"
            ]
        else:
            upgrade_body["previous_failed_upgrade_sha256"] = saved_upgrade[
                "upgrade_sha256"
            ]
        upgrade = _bound_record(upgrade_body, "upgrade_sha256")
        if saved_upgrade != upgrade:
            history = run_state / EXECUTION_UPGRADE_HISTORY_DIRNAME
            history.mkdir(exist_ok=True)
            _regular_directory(history, "execution-upgrade history")
            archived = history / f"{saved_upgrade['upgrade_sha256']}.json"
            if archived.exists():
                if read_json_object(archived) != saved_upgrade:
                    raise LaunchError("execution-upgrade history collision")
            else:
                atomic_json(archived, saved_upgrade)
            atomic_json(upgrade_path, upgrade)
    else:
        upgrade = _bound_record(upgrade_body, "upgrade_sha256")
        atomic_json(upgrade_path, upgrade)
    return plan, texts, upgrade


def preflight_lazy_nonpersistent_buffers(
    model: Any,
    *,
    device: Any,
    plan_sha256: str,
) -> dict[str, Any]:
    """Exercise all constructor-only buffer owners before expensive trellis work."""

    import torch

    target_device = torch.device(device)
    if target_device.type == "meta":
        raise LaunchError("lazy direct-state preflight cannot target the META device")

    candidates: list[tuple[str, Any, list[str]]] = []
    for module_name, module in model.model.named_modules():
        direct_buffers = dict(module.named_buffers(recurse=False))
        nonpersistent = set(
            getattr(module, "_non_persistent_buffers_set", set())
        )
        meta_names = sorted(
            name
            for name, buffer in direct_buffers.items()
            if name in nonpersistent and getattr(buffer, "is_meta", False)
        )
        if meta_names:
            candidates.append((module_name, module, meta_names))

    config = getattr(model.model, "config", None)
    layer_types = list(getattr(config, "layer_types", []) or [])
    compressed_layers = [
        index
        for index, layer_type in enumerate(layer_types)
        if layer_type != "sliding_attention"
    ]
    if not compressed_layers:
        raise LaunchError(
            "lazy direct-state preflight found no compressed-attention layer"
        )
    first_compressed = compressed_layers[0]
    required_suffix = (
        f"model.layers.{first_compressed}.self_attn.compressor.rotary_emb"
    )
    if not any(name.endswith(required_suffix) for name, _, _ in candidates):
        raise LaunchError(
            "lazy direct-state preflight did not find the first compressed-layer "
            f"rotary owner `{required_suffix}`"
        )

    records: list[dict[str, Any]] = []
    for module_name, module, expected_names in candidates:
        model.shell_direct_meta_materialize(
            target_submodule=module,
            device=target_device,
        )
        direct_buffers = dict(module.named_buffers(recurse=False))
        remaining = sorted(
            name
            for name in expected_names
            if name not in direct_buffers
            or getattr(direct_buffers[name], "is_meta", False)
        )
        if remaining:
            raise LaunchError(
                "lazy direct-state preflight left META constructor buffers under "
                f"`{module_name}`: {remaining}"
            )
        nonpersistent = set(
            getattr(module, "_non_persistent_buffers_set", set())
        )
        for buffer_name in expected_names:
            if buffer_name not in nonpersistent:
                raise LaunchError(
                    "lazy direct-state preflight changed persistence for "
                    f"`{module_name}.{buffer_name}`"
                )
            buffer = direct_buffers[buffer_name]
            if buffer.device != target_device:
                raise LaunchError(
                    "lazy direct-state preflight restored a buffer on the wrong "
                    f"device: `{module_name}.{buffer_name}` is on {buffer.device}, "
                    f"expected {target_device}"
                )
            payload = (
                buffer.detach()
                .to(device="cpu")
                .contiguous()
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
            records.append(
                {
                    "module": module_name,
                    "buffer": buffer_name,
                    "shape": list(buffer.shape),
                    "dtype": str(buffer.dtype),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )

    report = {
        "schema": "ds41rt-deepseek-v4-lazy-direct-state-preflight-v1",
        "plan_sha256": plan_sha256,
        "device": str(target_device),
        "first_compressed_layer": first_compressed,
        "owner_count": len(candidates),
        "buffer_count": len(records),
        "owners": [
            {"module": name, "buffers": names}
            for name, _, names in candidates
        ],
        "buffers_sha256": hashlib.sha256(canonical_json(records)).hexdigest(),
    }
    print(
        "Lazy direct-state preflight restored "
        f"{report['buffer_count']} constructor-only buffers across "
        f"{report['owner_count']} owners on {target_device}.",
        flush=True,
    )
    return report


def publish_export(
    plan: dict[str, Any],
    *,
    mtp_replay_batches: int,
) -> None:
    _validate_plan(plan)
    if (
        isinstance(mtp_replay_batches, bool)
        or not isinstance(mtp_replay_batches, int)
        or mtp_replay_batches <= 0
    ):
        raise LaunchError("MTP replay batch count must be positive")
    output = Path(plan["output"])
    run_state = Path(plan["run_state_dir"])
    export_stage = _export_stage_path(plan)
    if output.exists() or output.is_symlink():
        raise LaunchError(f"output appeared before artifact publication: {output}")
    atomic_json(export_stage / PLAN_FILENAME, plan)
    execution_upgrade = run_state / EXECUTION_UPGRADE_FILENAME
    if execution_upgrade.exists():
        upgrade = _read_execution_upgrade(run_state, plan)
        atomic_json(export_stage / EXECUTION_UPGRADE_FILENAME, upgrade)
        history = run_state / EXECUTION_UPGRADE_HISTORY_DIRNAME
        if history.exists():
            export_history = export_stage / EXECUTION_UPGRADE_HISTORY_DIRNAME
            export_history.mkdir()
            for archived in sorted(history.iterdir()):
                atomic_json(export_history / archived.name, read_json_object(archived))
    manifest = write_artifact_manifest(export_stage, plan)
    run = _bound_record(
        {
            "schema": RUN_SCHEMA,
            "status": "complete",
            "plan_sha256": plan["plan_sha256"],
            "artifact_manifest_sha256": manifest["manifest_sha256"],
            "mtp_replay_batches": mtp_replay_batches,
        },
        "run_sha256",
    )
    atomic_json(export_stage / RUN_FILENAME, run)
    validate_published_artifact(
        export_stage,
        plan,
        verify_file_hashes=False,
    )
    os.replace(export_stage, output)
    descriptor = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _make_quantize_config_metadata_portable(quantize_config: Any) -> None:
    """Remove coordinator-local state paths before serializing an artifact."""

    meta = copy.deepcopy(getattr(quantize_config, "meta", None) or {})
    mixed = meta.get("ds41rt_inline_mixed")
    if isinstance(mixed, dict):
        mixed.pop("tier_plan_root", None)
        meta["ds41rt_inline_mixed"] = mixed
    quantize_config.meta = meta


def publish_base_prefix_completion(
    plan: dict[str, Any],
    *,
    prefix_manifest_path: Path,
) -> dict[str, Any]:
    """Durably mark the base/prefix boundary for a separate MTP overlay."""

    _validate_plan(plan)
    handoff = _base_prefix_handoff_identity(plan)
    manifest = read_json_object(prefix_manifest_path)
    target_layer_ids = plan.get("source", {}).get("geometry", {}).get(
        "dspark_target_layer_ids"
    )
    source_batch_count = manifest.get("batch_count")
    if (
        manifest.get("status") != "complete"
        or manifest.get("completed_layers") != target_layer_ids
        or manifest.get("target_layer_ids") != target_layer_ids
        or isinstance(source_batch_count, bool)
        or not isinstance(source_batch_count, int)
        or source_batch_count <= 0
    ):
        raise LaunchError("MTP prefix manifest is not complete")
    record = _bound_record(
        {
            "schema": BASE_PREFIX_RUN_SCHEMA,
            "status": "complete",
            "plan_sha256": plan["plan_sha256"],
            "prefix_manifest_sha256": sha256_file(prefix_manifest_path),
            "target_layer_ids": target_layer_ids,
            "source_batch_count": source_batch_count,
            "handoff": handoff,
        },
        "run_sha256",
    )
    atomic_json(Path(plan["run_state_dir"]) / BASE_PREFIX_RUN_FILENAME, record)
    return record


def validate_base_prefix_completion(plan: dict[str, Any]) -> dict[str, Any]:
    """Validate a clean external-overlay handoff without trusting process timing."""

    _validate_plan(plan)
    record = read_json_object(
        Path(plan["run_state_dir"]) / BASE_PREFIX_RUN_FILENAME
    )
    _validate_bound_record(
        record,
        digest_field="run_sha256",
        label="base-prefix run manifest",
    )
    prefix_manifest_path = Path(plan["mtp_prefix_store"]) / "manifest.json"
    prefix_manifest = read_json_object(prefix_manifest_path)
    handoff = _base_prefix_handoff_identity(plan)
    if (
        record.get("schema") != BASE_PREFIX_RUN_SCHEMA
        or record.get("status") != "complete"
        or record.get("plan_sha256") != plan["plan_sha256"]
        or record.get("prefix_manifest_sha256")
        != sha256_file(prefix_manifest_path)
        or prefix_manifest.get("status") != "complete"
        or record.get("target_layer_ids") != prefix_manifest.get("target_layer_ids")
        or record.get("target_layer_ids")
        != plan.get("source", {}).get("geometry", {}).get(
            "dspark_target_layer_ids"
        )
        or isinstance(record.get("source_batch_count"), bool)
        or not isinstance(record.get("source_batch_count"), int)
        or record["source_batch_count"] <= 0
        or record["source_batch_count"] != prefix_manifest.get("batch_count")
        or record.get("handoff") != handoff
    ):
        raise LaunchError("base-prefix run manifest is inconsistent")
    return record


def _base_prefix_handoff_identity(plan: dict[str, Any]) -> dict[str, Any]:
    """Bind either the planned overlay handoff or a safe legacy-plan upgrade."""

    mode = plan.get("mtp_execution_mode", MTP_EXECUTION_EXTERNAL_OVERLAY)
    if mode == MTP_EXECUTION_EXTERNAL_OVERLAY:
        return {"mode": MTP_EXECUTION_EXTERNAL_OVERLAY, "authorization": "plan"}
    if mode != MTP_EXECUTION_INTEGRATED:
        raise LaunchError("base-prefix completion has an invalid MTP execution mode")
    upgrade = _read_execution_upgrade(Path(plan["run_state_dir"]), plan)
    if upgrade.get("change_contract", {}).get("mtp_execution") != (
        "sampled-external-overlay-handoff"
    ):
        raise LaunchError(
            "integrated parent plan has no sampled-overlay execution upgrade"
        )
    return {
        "mode": MTP_EXECUTION_EXTERNAL_OVERLAY,
        "authorization": "execution-upgrade",
        "upgrade_sha256": upgrade["upgrade_sha256"],
    }


def execute(
    plan: dict[str, Any],
    texts: list[str],
    *,
    resume: bool = False,
    mtp_overlay_handoff: bool = False,
    stop_after_layer: int | None = None,
) -> None:
    output = Path(plan["output"])
    if resume and (output.exists() or output.is_symlink()):
        if prepare_run(plan, resume=True):
            return
    if (
        resume
        and (
            plan.get("mtp_execution_mode") == MTP_EXECUTION_EXTERNAL_OVERLAY
            or mtp_overlay_handoff
        )
        and (Path(plan["run_state_dir"]) / BASE_PREFIX_RUN_FILENAME).exists()
    ):
        # Validate the saved plan, journal, directories, and prefix store before
        # accepting the completion marker as a successful idempotent resume.
        prepare_run(plan, resume=True)
        validate_base_prefix_completion(plan)
        return
    remote_workers = plan.get("remote_workers")
    if remote_workers is not None:
        token_env = remote_workers["token_env"]
        if not os.environ.get(token_env):
            raise LaunchError(f"remote worker token env `{token_env}` is unset")
        required_workers = str(remote_workers["cuda_workers_per_device"])
        configured_workers = os.environ.get("GPTQMODEL_CUDA_WORKERS")
        if configured_workers not in {None, required_workers}:
            raise LaunchError(
                "GPTQMODEL_CUDA_WORKERS conflicts with the immutable remote plan"
            )
        os.environ["GPTQMODEL_CUDA_WORKERS"] = required_workers
    if prepare_run(plan, resume=resume):
        return
    import torch
    from gptqmodel import GPTQModel
    from gptqmodel.looper.exllamav3_processor import EXL3Processor
    from gptqmodel.models.definitions.deepseek_v4 import DeepSeekV4QModel
    from gptqmodel.quantization import AutoModuleDecoderConfig, EXL3Config

    run_state = Path(plan["run_state_dir"])
    export_stage = _export_stage_path(plan)
    offload = Path(plan["offload_dir"])
    prefix_root = Path(plan["mtp_prefix_store"])
    os.environ["GPTQMODEL_EXL3_ERROR_JOURNAL"] = os.fspath(
        run_state / ERROR_JOURNAL_FILENAME
    )
    coordinator_devices = [
        f"cuda:{gpu['index']}" for gpu in plan["preflight"]["gpus"]
    ]
    if not coordinator_devices:
        raise LaunchError("quantization plan has no coordinator GPU")
    primary_device = coordinator_devices[0]
    qcfg_meta = {"ds41rt_error_ledger": plan["ledger_provenance"]}
    base_mixed_policy = plan.get("inline_mixed", {}).get("base")
    if base_mixed_policy is not None:
        qcfg_meta["ds41rt_inline_mixed"] = base_mixed_policy
    qcfg = EXL3Config(
        bits=int(plan["exl3"]["bits"]),
        codebook="mcg",
        out_scales="auto",
        module_include=[BASE_EXPERT_PATTERN],
        preprocessors=[AutoModuleDecoderConfig(target_dtype=torch.bfloat16)],
        fallback=None,
        offload_to_disk=True,
        offload_to_disk_path=os.fspath(offload),
        device=primary_device,
        calibration_data_device="cpu",
        dense_vram_strategy_devices=[primary_device],
        moe_vram_strategy="balanced",
        moe_vram_strategy_devices=coordinator_devices,
        meta=qcfg_meta,
    )
    model = GPTQModel.load(
        plan["source"]["path"],
        quantize_config=qcfg,
        trust_remote_code=False,
    )
    if not isinstance(model, DeepSeekV4QModel):
        raise LaunchError(f"unexpected GPTQModel definition: {type(model).__name__}")
    turtle = getattr(model, "turtle_model", None)
    configure_active_source = getattr(
        turtle, "configure_active_source_staging", None
    )
    if not callable(configure_active_source):
        raise LaunchError("DeepSeek V4 lazy source cannot stage active layers")
    configure_active_source(
        plan["active_layer_source_dir"],
        provenance={
            "plan_sha256": plan["plan_sha256"],
            "source_revision": plan["source"]["revision"],
            "source_index_sha256": plan["source"]["index_sha256"],
        },
    )
    direct_state_report = preflight_lazy_nonpersistent_buffers(
        model,
        device=primary_device,
        plan_sha256=plan["plan_sha256"],
    )
    atomic_json(
        run_state / DIRECT_STATE_PREFLIGHT_FILENAME,
        direct_state_report,
    )
    runtime = model.build_mtp_prefix_runtime(device=primary_device)
    geometry = plan["source"]["geometry"]
    prefix_store = DeepSeekV4MTPPrefixStore(
        prefix_root,
        target_layer_ids=tuple(geometry["dspark_target_layer_ids"]),
        hidden_size=geometry["hidden_size"],
        hc_mult=int(model.model.config.hc_mult),
        projector=runtime.project_target_taps,
        anchor_resolver=runtime.anchor_resolver,
        projection_device=runtime.device,
        projection_dtype=runtime.dtype,
        provenance={
            "plan_sha256": plan["plan_sha256"],
            "family_join": plan["ledger_provenance"]["family_join"],
        },
    )
    boundary_contract = plan.get("layer_boundary")
    if boundary_contract is None:
        # Explicit compatibility for a content-validated parent run created
        # before rolling boundaries existed. The old plan digest remains the
        # immutable quantization identity; an execution-upgrade record is
        # required by the resume launcher before such a run can use new code.
        _read_execution_upgrade(run_state, plan)
        boundary_root = run_state / LAYER_BOUNDARY_DIRNAME
    elif (
        not isinstance(boundary_contract, dict)
        or boundary_contract.get("contract") != BOUNDARY_CONTRACT
        or boundary_contract.get("root")
        != os.fspath(run_state / LAYER_BOUNDARY_DIRNAME)
        or boundary_contract.get("retention") != "latest-complete-layer"
        or boundary_contract.get("dtype") != "bfloat16"
    ):
        raise LaunchError("invalid rolling layer-boundary plan contract")
    else:
        boundary_root = Path(boundary_contract["root"])
    boundary_store = DeepSeekV4LayerBoundaryStore(
        boundary_root,
        plan_sha256=plan["plan_sha256"],
        family_join=plan["ledger_provenance"]["family_join"],
        projection_checkpoint_root=plan["projection_checkpoint"]["root"],
        error_journal_path=run_state / ERROR_JOURNAL_FILENAME,
        hidden_size=geometry["hidden_size"],
        hc_mult=int(model.model.config.hc_mult),
        routed_experts=geometry["n_routed_experts"],
    )
    boundary_controller = DeepSeekV4LayerBoundaryController(
        boundary_store,
        # The packed base tree is needed only by publication.  Keeping it out
        # of the model while the full MTP corpus is replayed avoids combining
        # 67 GB of restored/offloaded payload with the MTP activation frontier.
        defer_publication_materialization=True,
        stop_after_layer=stop_after_layer,
    )
    model.quantization_layer_boundary_checkpoint = boundary_controller
    configure_base_replay = getattr(model, "configure_base_replay_store", None)
    if not callable(configure_base_replay):
        raise LaunchError("DeepSeek V4 model cannot checkpoint base replay batches")
    configure_base_replay(
        run_state / POST_QUANT_REPLAY_DIRNAME,
        provenance={
            "plan_sha256": plan["plan_sha256"],
            "family_join": plan["ledger_provenance"]["family_join"],
            "output_dtype": "torch.bfloat16",
        },
    )
    capture_checkpoint_interval = int(
        plan.get("capture_batch_checkpoint_interval", 1)
    )
    execution_upgrade_path = run_state / EXECUTION_UPGRADE_FILENAME
    if execution_upgrade_path.exists():
        execution_upgrade = _read_execution_upgrade(run_state, plan)
        capture_checkpoint_interval = int(
            execution_upgrade.get("change_contract", {}).get(
                "capture_batch_checkpoint_interval",
                capture_checkpoint_interval,
            )
        )
    with capture_frontier_scope(
        run_state / CAPTURE_FRONTIER_DIRNAME
    ), capture_batch_spool_scope(
        run_state / CAPTURE_BATCH_SPOOL_DIRNAME,
        checkpoint_interval=capture_checkpoint_interval,
    ), memory_safety_scope(plan["memory_safety"]):
        model.set_mtp_target_tap_sink(prefix_store)
        try:
            try:
                model.quantize(
                    texts,
                    batch_size=plan["target_batch_size"],
                    calibration_sort=None,
                )
            except LayerBoundaryStop as stop:
                if stop_after_layer is None or stop.layer_index != stop_after_layer:
                    raise LaunchError(
                        "quantization stopped at an unexpected layer boundary"
                    ) from stop
                print(
                    json.dumps(
                        {
                            "event": "quantization-stopped-after-durable-layer",
                            "layer": stop.layer_index,
                            "plan_sha256": plan["plan_sha256"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                return
            finally:
                model.set_mtp_target_tap_sink(None)
            if (
                plan.get("mtp_execution_mode", MTP_EXECUTION_EXTERNAL_OVERLAY)
                == MTP_EXECUTION_EXTERNAL_OVERLAY
                or mtp_overlay_handoff
            ):
                # The overlay owns anchor selection and will scan/validate the
                # prefix exactly once. Avoid constructing the all-anchor replay
                # view here merely to throw it away at the process boundary.
                publish_base_prefix_completion(
                    plan,
                    prefix_manifest_path=prefix_store.manifest_path,
                )
                return
            selection = plan["mtp_anchor_selection"]
            batching = plan["mtp_replay_batching"]
            replay_batches = prefix_store.replay_dataset(
                replay_batch_size=plan["mtp_replay_batch_size"],
                device="cpu",
                anchor_sample_count=selection["count"],
                anchor_sample_seed=selection["seed"],
                batch_by_source_sequence=True,
                source_sequence_anchor_cap=batching[
                    "source_sequence_anchor_cap"
                ],
            )
            if not replay_batches:
                raise LaunchError(
                    "target calibration produced no dSpark replay positions"
                )
            mtp_model = model.build_mtp_quantization_model(
                runtime,
                calibration_embedding_device="cpu",
            )
            mtp_mixed_policy = plan.get("inline_mixed", {}).get("mtp")
            mtp_meta = copy.deepcopy(
                getattr(mtp_model.quantize_config, "meta", None) or {}
            )
            if mtp_mixed_policy is None:
                mtp_meta.pop("ds41rt_inline_mixed", None)
            else:
                mtp_meta["ds41rt_inline_mixed"] = mtp_mixed_policy
            mtp_model.quantize_config.meta = mtp_meta
            mtp_model.configure_mtp_activation_store(
                os.fspath(run_state / MTP_ACTIVATION_DIRNAME),
                provenance={
                    "plan_sha256": plan["plan_sha256"],
                    "family_join": plan["ledger_provenance"]["family_join"],
                    "prefix_manifest_sha256": sha256_file(
                        prefix_store.manifest_path
                    ),
                    "replay_batch_size": plan["mtp_replay_batch_size"],
                    "replay_batches": len(replay_batches),
                    "replay_positions": replay_batches.position_count,
                    "anchor_selection": replay_batches.anchor_selection_identity,
                    "replay_batching": replay_batches.replay_batching_identity,
                },
            )
            restored_mtp = (
                EXL3Processor.restore_completed_checkpoint_tree_if_complete(
                    model=mtp_model,
                    block_namespace="mtp",
                    layer_count=len(geometry["dspark_target_layer_ids"]),
                    experts_per_layer=geometry["n_routed_experts"],
                    error_journal_path=run_state / ERROR_JOURNAL_FILENAME,
                )
            )
            if restored_mtp:
                print(
                    json.dumps(
                        {
                            "event": "mtp-packed-checkpoint-tree-restored",
                            "layers": len(geometry["dspark_target_layer_ids"]),
                            "projections": (
                                len(geometry["dspark_target_layer_ids"])
                                * geometry["n_routed_experts"]
                                * 3
                            ),
                            "plan_sha256": plan["plan_sha256"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            else:
                mtp_model.quantize(
                    replay_batches,
                    batch_size=1,
                    calibration_sort=None,
                )
        finally:
            model.set_mtp_target_tap_sink(None)
    boundary_controller.materialize_deferred_prefix(
        model=model,
        force=True,
    )
    model.attach_mtp_quantization_model(mtp_model)
    _make_quantize_config_metadata_portable(model.quantize_config)
    _make_quantize_config_metadata_portable(mtp_model.quantize_config)
    export_stage.mkdir(exist_ok=True)
    model.save(os.fspath(export_stage), max_shard_size="8GB")
    publish_export(
        plan,
        mtp_replay_batches=len(replay_batches),
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--calibration-jsonl", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--route-screen-report", type=Path)
    parser.add_argument(
        "--route-qualification",
        choices=ROUTE_QUALIFICATION_MODES,
        default=ROUTE_QUALIFICATION_SCREEN,
        help=(
            "bind a completed screen report, or record routing during the exact "
            "full-corpus quantization capture"
        ),
    )
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--run-state-dir",
        type=Path,
        help="durable NVMe run-state root; defaults to the legacy output sibling",
    )
    parser.add_argument(
        "--projection-checkpoint-dir",
        type=Path,
        help="durable packed-projection root, which may reside on SATA",
    )
    parser.add_argument(
        "--active-layer-source-dir",
        type=Path,
        help="rolling NVMe staging root for the currently active source layer",
    )
    parser.add_argument("--offload-dir", type=Path, required=True)
    parser.add_argument("--mtp-prefix-store", type=Path, required=True)
    parser.add_argument(
        "--gptqmodel-lock",
        type=Path,
        default=root / "third_party" / "gptqmodel.lock.json",
    )
    parser.add_argument(
        "--bits",
        type=int,
        choices=(2, 3),
        default=2,
        help="integer EXL3 trellis bitrate for every routed expert family",
    )
    parser.add_argument(
        "--base-target-bpw",
        help=(
            "exact mixed target such as 2.1; standard EXL3 bits remain integer "
            "K2 while selected projections are encoded at K3"
        ),
    )
    parser.add_argument(
        "--mtp-target-bpw",
        help="exact integrated dSpark mixed target such as 2.2",
    )
    parser.add_argument(
        "--mixed-projection-ratio",
        nargs=3,
        type=int,
        default=(3, 5, 8),
        metavar=("GATE", "UP", "DOWN"),
        help="allocation ratio for K3 gate/up/down projections",
    )
    parser.add_argument(
        "--coordinator-gpu-count",
        type=int,
        choices=(1, 2),
        default=2,
        help="exact visible coordinator GPU count bound by the preflight report",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--capture-batch-checkpoint-interval",
        type=int,
        default=DEFAULT_CAPTURE_BATCH_CHECKPOINT_INTERVAL,
        help=(
            "number of calibration batches durably grouped per recovery "
            "checkpoint; the final short group is always forced durable"
        ),
    )
    parser.add_argument("--mtp-replay-batch-size", type=int, default=4)
    parser.add_argument(
        "--mtp-anchor-sample-count",
        type=int,
        default=DEFAULT_MTP_ANCHOR_SAMPLE_COUNT,
        help="deterministic eligible-anchor sample used by integrated dSpark",
    )
    parser.add_argument(
        "--mtp-anchor-sample-seed",
        type=int,
        default=DEFAULT_MTP_ANCHOR_SAMPLE_SEED,
    )
    parser.add_argument(
        "--mtp-sequence-anchor-cap",
        type=int,
        default=0,
        help="optional cap per source sequence; zero keeps each sequence joint",
    )
    parser.add_argument(
        "--mtp-execution-mode",
        choices=MTP_EXECUTION_MODES,
        default=MTP_EXECUTION_INTEGRATED,
        help=(
            "quantize MTP inline in the same resumable process (default), or "
            "explicitly exit after the audited target-prefix boundary for the "
            "legacy quantize_flash_dspark_overlay.py recovery path"
        ),
    )
    parser.add_argument(
        "--remote-worker",
        action="append",
        nargs=3,
        metavar=("NAME", "URL", "PREFLIGHT"),
        help="repeat once per Spark worker",
    )
    parser.add_argument(
        "--remote-token-env",
        default="DS41RT_EXL3_WORKER_TOKEN",
    )
    parser.add_argument("--remote-timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--remote-max-attempts", type=int, default=2)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--stop-after-layer",
        type=int,
        help=(
            "execution-only qualification stop after this decoder layer has "
            "committed its rolling boundary"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only an exact content-bound unfinished run",
    )
    parser.add_argument(
        "--execution-upgrade",
        action="store_true",
        help=(
            "resume a parent plan under a separately content-bound checkpoint-only "
            "execution upgrade"
        ),
    )
    parser.add_argument(
        "--mtp-overlay-handoff",
        action="store_true",
        help=(
            "resume a legacy integrated plan only through its base-prefix boundary, "
            "then hand it to the sampled external overlay"
        ),
    )
    args = parser.parse_args()
    if (
        args.batch_size <= 0
        or args.capture_batch_checkpoint_interval <= 0
        or args.mtp_replay_batch_size <= 0
        or args.remote_timeout_seconds <= 0
        or not 1 <= args.remote_max_attempts <= 10
        or not args.remote_token_env
        or any(value <= 0 for value in args.mixed_projection_ratio)
        or args.mtp_anchor_sample_count <= 0
        or args.mtp_sequence_anchor_cap < 0
        or (args.stop_after_layer is not None and args.stop_after_layer < 0)
    ):
        parser.error("batch sizes and remote-worker limits must be positive")
    if args.execution_upgrade and not args.resume:
        parser.error("--execution-upgrade requires --resume")
    if args.mtp_overlay_handoff and not args.execution_upgrade:
        parser.error("--mtp-overlay-handoff requires --execution-upgrade")
    return args


def main() -> int:
    args = parse_args()
    if args.execution_upgrade:
        plan, texts, upgrade = build_execution_upgrade(args)
        print(json.dumps(upgrade, indent=2, sort_keys=True), flush=True)
    else:
        plan, texts = build_plan(args)
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if not args.plan_only:
        execute(
            plan,
            texts,
            resume=args.resume,
            mtp_overlay_handoff=args.mtp_overlay_handoff,
            stop_after_layer=args.stop_after_layer,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LaunchError as exc:
        print(f"quantize-flash-gptqmodel: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
