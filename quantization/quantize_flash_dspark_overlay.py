#!/usr/bin/env python3
"""Quantize only DeepSeek-V4 Flash dSpark experts from a completed target prefix."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any

from deepseek_v4_mtp_prefix_store import (
    ANCHOR_SELECTION_CONTRACT,
    SEQUENCE_REPLAY_BATCH_CONTRACT,
    DeepSeekV4MTPPrefixStore,
    PrefixStoreError,
    sha256_file,
)
import quantize_flash_gptqmodel as base


PLAN_SCHEMA = "ds4rt-deepseek-v4-dspark-overlay-plan-v2"
RUN_SCHEMA = "ds4rt-deepseek-v4-dspark-overlay-run-v2"
OVERLAY_SCHEMA = "ds4rt-exl3-dspark-overlay-v2"
INDEX_SCHEMA = "ds4rt-exl3-dspark-overlay-checkpoint-index-v1"
RECIPE = "deepseek_v4_exl3_dspark_overlay_v2"
SCOPE = "mtp-routed-experts-only"
MTP_EXPERT_PATTERN = (
    r"^mtp\.\d+\.mlp\.experts\.\d+\."
    r"(?:gate_proj|up_proj|down_proj)$"
)
ACTIVATION_DIRNAME = "mtp-layer-activations"
MTP_CAPTURE_FRONTIER_DIRNAME = "mtp-capture-frontier"
REPORT_DIRNAME = "reports"
EXPORT_STAGE_DIRNAME = "overlay-export-stage"
INDEX_FILENAME = "ds4rt-dspark-overlay-index.json"
OVERLAY_FILENAME = "ds4rt-dspark-overlay.json"
RUN_FILENAME = "ds4rt-dspark-overlay-run.json"
SELECTION_FILENAME = "ds4rt-mtp-anchor-selection.json"
EXECUTION_UPGRADE_FILENAME = "ds4rt-dspark-overlay-execution-upgrade.json"
EXECUTION_UPGRADE_SCHEMA = "ds4rt-dspark-overlay-execution-upgrade-v1"
EXECUTION_UPGRADE_HISTORY_DIRNAME = base.EXECUTION_UPGRADE_HISTORY_DIRNAME
MTP_RESUME_CUDA_ALLOCATION_LIMIT_BYTES = 84 * 1024**3
ZERO_ROUTE_RECOVERY_SCHEMA = "ds4rt.exl3-zero-route-recovery"
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
ZERO_ROUTE_RECOVERY_AUTHORIZATION_SCHEMA = (
    "ds4rt.exl3-zero-route-recovery-authorization"
)
HESSIAN_OWNER_POLICY_CONTRACT = (
    "ds4rt.exl3-hessian-owner-weighted-round-robin-v1"
)
HESSIAN_OWNER_POLICY_META = "ds4rt_hessian_owner_policy"
MTP_REPLAY_CONTRACT = "deepseek-v4-mtp-joint-five-row-v1"
_COMPLETE_MTP_BOUNDARY = re.compile(r"layer-([0-9]{6})\Z")
_PARTIAL_MTP_BOUNDARY = re.compile(r"\.layer-([0-9]{6})\.partial\Z")


class OverlayError(RuntimeError):
    """The requested dSpark-only quantization is not reproducible."""


class MTPActivationBoundaryController:
    """Resume MTP at the newest durable post-block activation boundary.

    Packed expert tensors remain authoritative in the projection checkpoint
    store, so a completed MTP block does not need to be reconstructed in the
    live module tree.  Its post-quantized outputs are exactly the next block's
    inputs.  This controller verifies that rolling frontier, installs it in the
    EXL3 processor cache, and tells GPTQModel to begin with the next block.
    """

    def __init__(
        self,
        root: Path,
        *,
        provenance: dict[str, Any],
        block_count: int,
        hidden_size: int,
        hc_mult: int,
        proposal_rows: int,
        expert_count: int,
    ) -> None:
        if (
            not isinstance(provenance, dict)
            or not provenance
            or isinstance(block_count, bool)
            or not isinstance(block_count, int)
            or block_count <= 0
            or min(hidden_size, hc_mult, proposal_rows, expert_count) <= 0
        ):
            raise OverlayError("MTP activation-boundary geometry is invalid")
        self.root = root.expanduser().resolve()
        self.provenance = json.loads(json.dumps(provenance, sort_keys=True))
        self.block_count = block_count
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.proposal_rows = proposal_rows
        self.expert_count = expert_count
        self._committed_through = -1

    @staticmethod
    def _processor(processors: list[Any]) -> Any:
        candidates = [
            processor
            for processor in processors
            if callable(
                getattr(processor, "completed_layer_checkpoint_entries", None)
            )
            and callable(getattr(processor, "receive_layer_inputs", None))
        ]
        if len(candidates) != 1:
            raise OverlayError(
                "MTP activation resume requires exactly one EXL3 processor"
            )
        return candidates[0]

    def _entries(self) -> tuple[list[tuple[int, Path]], list[tuple[int, Path]]]:
        if not self.root.exists():
            return [], []
        _regular_directory(self.root, "MTP activation-boundary root")
        complete: list[tuple[int, Path]] = []
        partial: list[tuple[int, Path]] = []
        for path in self.root.iterdir():
            match = _COMPLETE_MTP_BOUNDARY.fullmatch(path.name)
            partial_match = _PARTIAL_MTP_BOUNDARY.fullmatch(path.name)
            if match is not None:
                _regular_directory(path, "completed MTP activation boundary")
                complete.append((int(match.group(1)), path))
            elif partial_match is not None:
                _regular_directory(path, "partial MTP activation boundary")
                partial.append((int(partial_match.group(1)), path))
            else:
                raise OverlayError(
                    f"MTP activation-boundary root has unexpected entry: {path.name}"
                )
        return sorted(complete), sorted(partial)

    def _open(self, path: Path, *, verify_hashes: bool):
        import torch
        from gptqmodel.looper.input_cache import DiskBackedLayerOutputSequence

        try:
            sequence = DiskBackedLayerOutputSequence.open(
                path,
                verify_hashes=verify_hashes,
            )
        except (OSError, ValueError) as error:
            raise OverlayError(
                f"cannot open durable MTP activation boundary {path.name}: {error}"
            ) from error
        manifest = sequence.manifest
        shards = manifest.get("shards")
        match = _COMPLETE_MTP_BOUNDARY.fullmatch(path.name)
        expected_provenance = {
            **self.provenance,
            "block_index": int(match.group(1)) if match is not None else None,
            "replay_contract": MTP_REPLAY_CONTRACT,
        }
        if (
            match is None
            or manifest.get("provenance") != expected_provenance
            or manifest.get("shard_batches") != 1
            or not isinstance(shards, list)
            or any(
                not isinstance(shard, dict)
                or shard.get("dtype") != str(torch.bfloat16)
                or not isinstance(shard.get("shapes"), list)
                or any(
                    not isinstance(shape, list)
                    or len(shape) != 4
                    or tuple(shape[1:])
                    != (self.proposal_rows, self.hc_mult, self.hidden_size)
                    for shape in shard.get("shapes", [])
                )
                for shard in shards
            )
        ):
            raise OverlayError(
                f"MTP activation boundary {path.name} has invalid provenance or geometry"
            )
        return sequence

    def _prune_older_boundaries(self, keep: Path) -> None:
        changed = False
        for _index, path in self._entries()[0]:
            if path != keep:
                shutil.rmtree(path)
                changed = True
        if changed:
            descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def restore(self, *, model: Any, processors: list[Any]) -> int:
        processor = self._processor(processors)
        complete, partial = self._entries()
        if not complete:
            if any(index != 0 for index, _path in partial):
                raise OverlayError("first partial MTP activation boundary is not block 0")
            return 0
        latest_index, latest_path = complete[-1]
        if not 0 <= latest_index < self.block_count:
            raise OverlayError("completed MTP activation boundary is outside the model")
        if any(index <= latest_index or index > latest_index + 1 for index, _ in partial):
            raise OverlayError("partial MTP activation boundary is not the next block")
        sequence = self._open(latest_path, verify_hashes=True)
        _audit_mtp_checkpoint_block(
            self.root.parent,
            block_index=latest_index,
            plan_sha256=self.provenance["plan_sha256"],
            expected_projection_count=self.expert_count * 3,
        )
        cache = processor.inputs_cache
        if len(sequence) != len(cache.layer_inputs):
            raise OverlayError(
                "MTP activation-boundary batch count differs from replay selection"
            )
        processor.receive_layer_inputs(sequence)
        self._committed_through = latest_index
        self._prune_older_boundaries(latest_path)

        capture_root = os.getenv("GPTQMODEL_EXL3_CAPTURE_FRONTIER")
        if capture_root:
            from gptqmodel.utils.exl3_capture_frontier import EXL3CaptureFrontierStore

            family_join = self.provenance.get("family_join")
            if not isinstance(family_join, dict):
                raise OverlayError("MTP activation provenance has no family join")
            EXL3CaptureFrontierStore(
                capture_root,
                family_join=family_join,
            ).discard_through(latest_index, block_namespace="mtp")
        prune_source = getattr(
            getattr(model, "turtle_model", None),
            "prune_active_source_scope_through",
            None,
        )
        if callable(prune_source):
            prune_source("mtp", latest_index)
        return latest_index + 1

    def commit_layer(
        self,
        *,
        model: Any,
        processor: Any,
        layer_index: int,
        layer_name: str,
    ) -> dict[str, Any]:
        del model
        if layer_index != self._committed_through + 1:
            raise OverlayError("MTP activation boundaries are not committed in order")
        expected_name = f"mtp.{layer_index}"
        if layer_name != expected_name:
            raise OverlayError(
                f"MTP activation boundary has noncanonical layer name {layer_name!r}"
            )
        entries = processor.completed_layer_checkpoint_entries(layer_index)
        expected_count = self.expert_count * 3
        names = {
            entry.get("module") for entry in entries if isinstance(entry, dict)
        }
        expected_names = {
            f"mtp.{layer_index}.mlp.experts.{expert}.{projection}"
            for expert in range(self.expert_count)
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        if (
            len(entries) != expected_count
            or names != expected_names
        ):
            raise OverlayError(
                f"MTP block {layer_index} boundary lacks all projection checkpoints"
            )
        sequence = self._open(
            self.root / f"layer-{layer_index:06d}",
            # Do not discard the previous rolling boundary until the new one
            # has survived an independent read and full payload rehash.
            verify_hashes=True,
        )
        if sequence is not processor.inputs_cache.layer_inputs:
            # ForwardExecutor may reopen the finalized sequence; identity is
            # established by its complete manifest rather than object identity.
            current = getattr(processor.inputs_cache.layer_inputs, "manifest", None)
            if current != sequence.manifest:
                raise OverlayError(
                    "processor handoff differs from the durable MTP activation boundary"
                )
        self._committed_through = layer_index
        self._prune_older_boundaries(sequence.root)
        return sequence.manifest

    def materialize_deferred_prefix(self, **_kwargs) -> None:
        """Overlay publication consumes checkpoints directly; no model rebuild."""


def _audit_mtp_checkpoint_block(
    run_state: Path,
    *,
    block_index: int,
    plan_sha256: str,
    expected_projection_count: int,
) -> dict[str, Any]:
    """Reopen every packed checkpoint before skipping its MTP block."""

    from validate_projection_checkpoint_block import audit_block

    report = audit_block(
        run_state,
        block_namespace="mtp",
        logical_layer=block_index,
    )
    if (
        report.get("status") != "complete"
        or report.get("plan_sha256") != plan_sha256
        or report.get("logical_layer") != block_index
        or report.get("projection_count") != expected_projection_count
        or report.get("expected_projection_count") != expected_projection_count
        or report.get("missing_projection_count") != 0
    ):
        raise OverlayError("completed MTP checkpoint block failed resume audit")
    reports = run_state / REPORT_DIRNAME
    reports.mkdir(exist_ok=True)
    base.atomic_json(
        reports / f"mtp-layer-{block_index}-resume-audit.json",
        report,
    )
    return report


def _bound(value: dict[str, Any], field: str) -> dict[str, Any]:
    if field in value:
        raise OverlayError(f"record already contains reserved field {field}")
    return {
        **value,
        field: hashlib.sha256(base.canonical_json(value)).hexdigest(),
    }


def _validate_bound(value: dict[str, Any], field: str, label: str) -> None:
    digest = value.get(field)
    body = {key: item for key, item in value.items() if key != field}
    if (
        not isinstance(digest, str)
        or base.SHA256_RE.fullmatch(digest) is None
        or hashlib.sha256(base.canonical_json(body)).hexdigest() != digest
    ):
        raise OverlayError(f"{label} digest is invalid")


def _parent_plan(
    path: Path,
    source: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise OverlayError("--parent-plan must be one regular immutable plan")
    parent = base.read_json_object(resolved)
    try:
        base._validate_plan(parent)
    except base.LaunchError as error:
        raise OverlayError(str(error)) from error
    if (
        parent.get("source") != source
        or parent.get("exl3", {}).get("bits") not in {2, 3}
        or parent.get("exl3", {}).get("codebook") != "mcg"
    ):
        raise OverlayError("parent plan is not an exact integer-tier target source")
    family_join = parent.get("ledger_provenance", {}).get("family_join")
    if not isinstance(family_join, dict):
        raise OverlayError("parent plan has no family-join provenance")
    try:
        base.validate_base_prefix_completion(parent)
    except base.LaunchError as error:
        raise OverlayError(str(error)) from error
    return parent, {
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "plan_sha256": parent["plan_sha256"],
    }


def _prefix_identity(
    root: Path,
    *,
    parent: dict[str, Any],
) -> tuple[DeepSeekV4MTPPrefixStore, dict[str, Any]]:
    expected_provenance = {
        "plan_sha256": parent["plan_sha256"],
        "family_join": parent["ledger_provenance"]["family_join"],
    }
    try:
        store = DeepSeekV4MTPPrefixStore.open_complete(
            root,
            expected_provenance=expected_provenance,
        )
    except Exception as error:
        raise OverlayError(f"cannot open complete parent dSpark prefix: {error}") from error
    projected_bytes = sum(
        int(record["projected_main"]["bytes"])
        for record in store.manifest["batches"].values()
    )
    target_tap_bytes = sum(
        int(tap["bytes"])
        for record in store.manifest["batches"].values()
        for tap in record["target_taps"].values()
    )
    return store, {
        "schema": store.manifest["schema"],
        "manifest_bytes": store.manifest_path.stat().st_size,
        "manifest_sha256": store.manifest_sha256,
        "batch_count": store.manifest["batch_count"],
        "target_layer_ids": list(store.target_layer_ids),
        "hidden_size": store.hidden_size,
        "hc_mult": store.hc_mult,
        "projected_main_bytes": projected_bytes,
        "target_tap_bytes": target_tap_bytes,
    }


def _execution_topology(
    remote: dict[str, Any] | None, preflight: dict[str, Any]
) -> dict[str, Any]:
    if remote is None:
        return {
            "contract": "ds4rt.exl3-coordinator-only-v1",
            "scheduler": "coordinator-only-v1",
            "coordinator": {
                "preflight_sha256": preflight["sha256"],
                "image_digest": preflight["image_digest"],
            },
            "coordinator_slots": [
                {
                    "slot_id": f"coordinator:cuda:{gpu['index']}",
                    "device": f"cuda:{gpu['index']}",
                }
                for gpu in preflight["gpus"]
            ],
            "workers": [],
        }
    return {
        "contract": remote["contract"],
        "scheduler": remote["scheduler"],
        "assignment_store": remote["assignment_store"],
        "coordinator": {
            "preflight_sha256": preflight["sha256"],
            "image_digest": preflight["image_digest"],
        },
        "coordinator_slots": remote["coordinator_slots"],
        "workers": [
            {
                "name": endpoint["name"],
                "preflight_sha256": endpoint["preflight_sha256"],
                "image_digest": endpoint["image_digest"],
            }
            for endpoint in remote["endpoints"]
        ],
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    if isinstance(args.bits, bool) or args.bits not in {2, 3}:
        raise OverlayError("--bits must be integer K2 or K3")
    if (
        isinstance(args.mtp_anchor_sample_count, bool)
        or not isinstance(args.mtp_anchor_sample_count, int)
        or args.mtp_anchor_sample_count <= 0
    ):
        raise OverlayError("--mtp-anchor-sample-count must be positive")
    if isinstance(args.mtp_anchor_sample_seed, bool) or not isinstance(
        args.mtp_anchor_sample_seed, int
    ):
        raise OverlayError("--mtp-anchor-sample-seed must be an integer")
    if (
        isinstance(args.mtp_sequence_anchor_cap, bool)
        or not isinstance(args.mtp_sequence_anchor_cap, int)
        or args.mtp_sequence_anchor_cap < 0
    ):
        raise OverlayError("--mtp-sequence-anchor-cap must be non-negative")
    lock_path = args.gptqmodel_lock.expanduser().resolve(strict=True)
    lock = base.read_json_object(lock_path)
    source = base.snapshot_identity(args.snapshot)
    parent, parent_identity = _parent_plan(args.parent_plan, source)
    prefix_store, prefix_identity = _prefix_identity(
        args.mtp_prefix_store,
        parent=parent,
    )
    source_config = base.read_json_object(
        Path(source["path"]) / "config.json"
    )
    if (
        prefix_identity["target_layer_ids"]
        != source["geometry"]["dspark_target_layer_ids"]
        or prefix_identity["hidden_size"] != source["geometry"]["hidden_size"]
        or prefix_identity["hc_mult"] != source_config.get("hc_mult")
    ):
        raise OverlayError("parent prefix geometry differs from the source model")
    preflight = base.preflight_identity(
        args.preflight_report,
        str(lock.get("revision")),
        expected_gpu_count=args.coordinator_gpu_count,
    )
    if (
        preflight["gptqmodel"].get("revision") != lock.get("revision")
        or preflight["gptqmodel"].get("source_tree_sha256")
        != lock.get("source_tree_sha256")
    ):
        raise OverlayError("preflight GPTQModel identity differs from the source lock")
    raw_owner_weights = getattr(args, "mtp_hessian_owner_device_weights", None)
    owner_weights = (
        list(raw_owner_weights)
        if raw_owner_weights is not None
        else [1] * len(preflight["gpus"])
    )
    if (
        len(owner_weights) != len(preflight["gpus"])
        or any(
            isinstance(weight, bool)
            or not isinstance(weight, int)
            or weight <= 0
            for weight in owner_weights
        )
    ):
        raise OverlayError(
            "--mtp-hessian-owner-device-weights must provide one positive "
            "integer per coordinator GPU"
        )
    hessian_owner_policy = {
        "contract": HESSIAN_OWNER_POLICY_CONTRACT,
        "device_weights": owner_weights,
    }
    try:
        remote = base.remote_worker_configuration(
            args,
            lock=lock,
            coordinator_preflight=preflight,
        )
    except base.LaunchError as error:
        raise OverlayError(str(error)) from error
    output = args.output.expanduser().resolve()
    raw_run_state = getattr(args, "run_state_dir", None)
    run_state = (
        raw_run_state.expanduser().resolve()
        if raw_run_state is not None
        else output.with_name(f".{output.name}.ds4rt-run")
    )
    raw_checkpoint = getattr(args, "projection_checkpoint_dir", None)
    checkpoint_root = (
        raw_checkpoint.expanduser().resolve()
        if raw_checkpoint is not None
        else run_state / base.PROJECTION_CHECKPOINT_DIRNAME
    )
    raw_active_source = getattr(args, "active_layer_source_dir", None)
    active_source = (
        raw_active_source.expanduser().resolve()
        if raw_active_source is not None
        else run_state / base.ACTIVE_LAYER_SOURCE_DIRNAME
    )
    offload = args.offload_dir.expanduser().resolve()
    prefix_root = prefix_store.root
    if remote is not None:
        remote["assignment_store"] = os.fspath(
            run_state / base.REMOTE_ASSIGNMENT_DIRNAME
        )
    writable = (output, run_state, offload)
    if len(set(writable)) != len(writable) or any(
        left.is_relative_to(right) or right.is_relative_to(left)
        for index, left in enumerate(writable)
        for right in writable[index + 1 :]
    ):
        raise OverlayError("output, run-state, and offload paths must be non-nested")
    if any(
        prefix_root.is_relative_to(path) or path.is_relative_to(prefix_root)
        for path in (*writable, checkpoint_root, active_source)
    ):
        raise OverlayError("read-only parent prefix must not overlap overlay paths")
    for path, canonical_child in (
        (checkpoint_root, run_state / base.PROJECTION_CHECKPOINT_DIRNAME),
        (active_source, run_state / base.ACTIVE_LAYER_SOURCE_DIRNAME),
    ):
        if path in writable or any(
            path.is_relative_to(other) or other.is_relative_to(path)
            for other in (output, offload)
        ):
            raise OverlayError("overlay active/checkpoint store overlaps another path")
        if (path.is_relative_to(run_state) or run_state.is_relative_to(path)) and (
            path != canonical_child
        ):
            raise OverlayError("overlay run-state child store has the wrong name")
    if checkpoint_root == active_source or checkpoint_root.is_relative_to(
        active_source
    ) or active_source.is_relative_to(checkpoint_root):
        raise OverlayError("overlay active source and checkpoint stores overlap")

    checkpoint = {
        "contract": base.PROJECTION_CHECKPOINT_CONTRACT,
        "root": os.fspath(checkpoint_root),
    }
    anchor_selection = {
        "contract": ANCHOR_SELECTION_CONTRACT,
        "count": args.mtp_anchor_sample_count,
        "seed": args.mtp_anchor_sample_seed,
    }
    replay_batching = {
        "contract": SEQUENCE_REPLAY_BATCH_CONTRACT,
        "source_sequence_anchor_cap": args.mtp_sequence_anchor_cap or None,
        "proposal_rows_per_anchor": 5,
    }
    family_join = {
        "recipe": RECIPE,
        "scope": SCOPE,
        "source": source,
        "corpus": parent["corpus"],
        "target_parent": parent_identity,
        "prefix": prefix_identity,
        "gptqmodel": lock,
        "preflight_sha256": preflight["sha256"],
        "image_digest": preflight["image_digest"],
        "quantizer_seed": base.EXL3_SEED,
        "quantizer_numerics": {
            "sigma_reg": base.EXL3_SIGMA_REG,
            "hessian_capture": base.EXL3_HESSIAN_CAPTURE_CONTRACT,
            "hessian_numerical": base.EXL3_HESSIAN_NUMERICAL_CONTRACT,
            "hessian_symmetry": base.EXL3_HESSIAN_SYMMETRY_CONTRACT,
        },
        "bits": args.bits,
        "codebook": "mcg",
        "module_include": MTP_EXPERT_PATTERN,
        "operator_contract": "ds4rt-deepseek-v4-joint-mtp-from-quantized-prefix-v3",
        "anchor_selection": anchor_selection,
        "replay_batching": replay_batching,
        "hessian_owner_policy": hessian_owner_policy,
        "route_evidence_contract": base.ROUTE_EVIDENCE_CONTRACT,
        "zero_route_recovery_contract": ZERO_ROUTE_RECOVERY_SCHEMA,
        "execution_topology": _execution_topology(remote, preflight),
    }
    provenance = {
        "family_join": family_join,
        "run": {
            "coordinator": preflight,
            "output": os.fspath(output),
            "run_state": os.fspath(run_state),
            "offload": os.fspath(offload),
            "active_layer_source": os.fspath(active_source),
            "mtp_prefix_store": os.fspath(prefix_root),
            "anchor_selection": anchor_selection,
            "replay_batching": replay_batching,
            "projection_checkpoint": checkpoint,
        },
    }
    if remote is not None:
        provenance["run"]["remote_workers"] = remote
    plan = {
        "schema": PLAN_SCHEMA,
        "recipe": RECIPE,
        "scope": SCOPE,
        "source": source,
        "corpus": parent["corpus"],
        "target_parent": parent_identity,
        "prefix": prefix_identity,
        "preflight": preflight,
        "output": os.fspath(output),
        "run_state_dir": os.fspath(run_state),
        "projection_checkpoint_dir": os.fspath(checkpoint_root),
        "active_layer_source_dir": os.fspath(active_source),
        "offload_dir": os.fspath(offload),
        "mtp_prefix_store": os.fspath(prefix_root),
        "anchor_selection": anchor_selection,
        "replay_batching": replay_batching,
        "projection_checkpoint": checkpoint,
        "remote_workers": remote,
        "memory_safety": {
            "host_rss_limit_bytes": base.HOST_RSS_LIMIT_BYTES,
            "cuda_allocation_limit_bytes": base.CUDA_ALLOCATION_LIMIT_BYTES,
            "telemetry_interval_batches": base.MEMORY_TELEMETRY_INTERVAL_BATCHES,
            "spill_policy": "fail-closed-no-cpu-or-managed-memory",
        },
        "exl3": {
            "bits": args.bits,
            "codebook": "mcg",
            "seed": base.EXL3_SEED,
            "module_include": [MTP_EXPERT_PATTERN],
            "fallback": None,
            "out_scales": "auto",
            "sigma_reg": base.EXL3_SIGMA_REG,
            "hessian_capture": base.EXL3_HESSIAN_CAPTURE_CONTRACT,
            "hessian_numerical": base.EXL3_HESSIAN_NUMERICAL_CONTRACT,
            "hessian_symmetry": base.EXL3_HESSIAN_SYMMETRY_CONTRACT,
            "hessian_owner_policy": hessian_owner_policy,
            "zero_route_recovery": {
                "contract": ZERO_ROUTE_RECOVERY_SCHEMA,
                "trigger": ZERO_ROUTE_RECOVERY_TRIGGER,
                "sample_source": ZERO_ROUTE_RECOVERY_SAMPLE_SOURCE,
                "capture_method": ZERO_ROUTE_RECOVERY_CAPTURE_METHOD,
                "selection_policy": ZERO_ROUTE_RECOVERY_SELECTION_POLICY,
                "candidate_rank_min": ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MIN,
                "candidate_rank_max": ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MAX,
                "target_sample_count": ZERO_ROUTE_RECOVERY_TARGET_SAMPLE_COUNT,
                "identity_calibration_policy": ZERO_ROUTE_RECOVERY_IDENTITY_POLICY,
            },
        },
        "ledger_provenance": provenance,
    }
    plan["plan_sha256"] = hashlib.sha256(base.canonical_json(plan)).hexdigest()
    return plan


def _validate_plan(plan: dict[str, Any]) -> None:
    digest = plan.get("plan_sha256")
    bits = plan.get("exl3", {}).get("bits")
    selection = plan.get("anchor_selection")
    batching = plan.get("replay_batching")
    provenance = plan.get("ledger_provenance")
    family_join = (
        provenance.get("family_join") if isinstance(provenance, dict) else None
    )
    run = provenance.get("run") if isinstance(provenance, dict) else None
    checkpoint = plan.get("projection_checkpoint")
    hessian_owner_policy = plan.get("exl3", {}).get("hessian_owner_policy")
    owner_weights = (
        hessian_owner_policy.get("device_weights")
        if isinstance(hessian_owner_policy, dict)
        else None
    )
    preflight_gpus = plan.get("preflight", {}).get("gpus")
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("recipe") != RECIPE
        or plan.get("scope") != SCOPE
        or isinstance(bits, bool)
        or not isinstance(bits, int)
        or bits not in {2, 3}
        or not isinstance(selection, dict)
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
        or not isinstance(family_join, dict)
        or family_join.get("anchor_selection") != selection
        or family_join.get("replay_batching") != batching
        or not isinstance(run, dict)
        or run.get("anchor_selection") != selection
        or run.get("replay_batching") != batching
        or not isinstance(checkpoint, dict)
        or plan.get("projection_checkpoint_dir") != checkpoint.get("root")
        or not isinstance(preflight_gpus, list)
        or not isinstance(hessian_owner_policy, dict)
        or hessian_owner_policy.get("contract")
        != HESSIAN_OWNER_POLICY_CONTRACT
        or not isinstance(owner_weights, list)
        or len(owner_weights) != len(preflight_gpus)
        or any(
            isinstance(weight, bool)
            or not isinstance(weight, int)
            or weight <= 0
            for weight in owner_weights
        )
        or family_join.get("hessian_owner_policy") != hessian_owner_policy
        or plan.get("memory_safety")
        != {
            "host_rss_limit_bytes": base.HOST_RSS_LIMIT_BYTES,
            "cuda_allocation_limit_bytes": base.CUDA_ALLOCATION_LIMIT_BYTES,
            "telemetry_interval_batches": base.MEMORY_TELEMETRY_INTERVAL_BATCHES,
            "spill_policy": "fail-closed-no-cpu-or-managed-memory",
        }
        or not isinstance(plan.get("active_layer_source_dir"), str)
        or not isinstance(digest, str)
        or hashlib.sha256(base.canonical_json(body)).hexdigest() != digest
    ):
        raise OverlayError("dSpark overlay plan is invalid")


def _execution_neutral_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Remove only coordinator observation fields allowed to change on upgrade."""

    neutral = copy.deepcopy(plan)
    neutral.pop("plan_sha256", None)
    neutral.pop("preflight", None)
    provenance = neutral.get("ledger_provenance")
    family_join = (
        provenance.get("family_join") if isinstance(provenance, dict) else None
    )
    run = provenance.get("run") if isinstance(provenance, dict) else None
    if not isinstance(family_join, dict) or not isinstance(run, dict):
        raise OverlayError("overlay plan lacks execution provenance")
    for key in ("preflight_sha256", "image_digest"):
        family_join.pop(key, None)
    topology = family_join.get("execution_topology")
    coordinator = (
        topology.get("coordinator") if isinstance(topology, dict) else None
    )
    if not isinstance(topology, dict) or not isinstance(coordinator, dict):
        raise OverlayError("overlay plan lacks coordinator execution topology")
    coordinator.pop("preflight_sha256", None)
    coordinator.pop("image_digest", None)
    run.pop("coordinator", None)
    return neutral


def _activation_provenance(
    plan: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    _validate_selection_record(plan, selection)
    return {
        "plan_sha256": plan["plan_sha256"],
        "family_join": plan["ledger_provenance"]["family_join"],
        "prefix_manifest_sha256": plan["prefix"]["manifest_sha256"],
        "anchor_selection": selection["anchor_selection"],
        "replay_batching": selection["replay_batching"],
        "replay_batches": selection["replay_batching"]["batch_count"],
        "replay_positions": selection["anchor_selection"][
            "selected_position_count"
        ],
        "bits": plan["exl3"]["bits"],
    }


def _activation_boundary_controller(
    plan: dict[str, Any],
    selection: dict[str, Any],
) -> MTPActivationBoundaryController:
    geometry = plan["source"]["geometry"]
    return MTPActivationBoundaryController(
        Path(plan["run_state_dir"]) / ACTIVATION_DIRNAME,
        provenance=_activation_provenance(plan, selection),
        block_count=len(geometry["dspark_target_layer_ids"]),
        hidden_size=int(geometry["hidden_size"]),
        hc_mult=int(geometry["hc_mult"]),
        proposal_rows=int(plan["replay_batching"]["proposal_rows_per_anchor"]),
        expert_count=int(geometry["n_routed_experts"]),
    )


def _activation_boundary_upgrade_identity(plan: dict[str, Any]) -> dict[str, Any]:
    run_state = Path(plan["run_state_dir"])
    selection = base.read_json_object(run_state / SELECTION_FILENAME)
    controller = _activation_boundary_controller(plan, selection)
    complete, _partial = controller._entries()
    if not complete:
        raise OverlayError("execution upgrade requires a completed MTP boundary")
    layer_index, directory = sorted(complete)[-1]
    manifest_path = directory / "manifest.json"
    sequence = controller._open(directory, verify_hashes=False)
    manifest = sequence.manifest
    if manifest.get("layer_index") != layer_index:
        raise OverlayError("execution-upgrade MTP boundary manifest is invalid")
    return {
        "layer_index": layer_index,
        "directory": directory.name,
        "manifest_bytes": manifest_path.stat().st_size,
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "batch_count": manifest.get("batch_count"),
        "payload_bytes": sum(
            int(shard.get("bytes", 0))
            for shard in manifest.get("shards", [])
            if isinstance(shard, dict)
        ),
    }


def _read_execution_upgrade(
    run_state: Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    path = run_state / EXECUTION_UPGRADE_FILENAME
    if not path.is_file() or path.is_symlink():
        raise OverlayError("overlay execution-upgrade record is unavailable")
    upgrade = base.read_json_object(path)
    _validate_bound(upgrade, "upgrade_sha256", "overlay execution upgrade")
    if (
        upgrade.get("schema") != EXECUTION_UPGRADE_SCHEMA
        or upgrade.get("parent_plan_sha256") != plan.get("plan_sha256")
    ):
        raise OverlayError("overlay execution upgrade does not bind the parent plan")
    history = run_state / EXECUTION_UPGRADE_HISTORY_DIRNAME
    history_records: dict[str, dict[str, Any]] = {}
    if history.exists():
        _regular_directory(history, "overlay execution-upgrade history")
        for archived in history.iterdir():
            match = re.fullmatch(r"([0-9a-f]{64})\.json", archived.name)
            if match is None or not archived.is_file() or archived.is_symlink():
                raise OverlayError(
                    "overlay execution-upgrade history contains an unsafe entry"
                )
            record = base.read_json_object(archived)
            _validate_bound(record, "upgrade_sha256", "archived execution upgrade")
            digest = match.group(1)
            if (
                record.get("upgrade_sha256") != digest
                or record.get("schema") != EXECUTION_UPGRADE_SCHEMA
                or record.get("parent_plan_sha256") != plan.get("plan_sha256")
            ):
                raise OverlayError(
                    "archived overlay execution upgrade is inconsistent"
                )
            history_records[digest] = record
    cursor = upgrade.get("previous_upgrade_sha256")
    visited: set[str] = set()
    while cursor is not None:
        if (
            not isinstance(cursor, str)
            or cursor in visited
            or cursor not in history_records
        ):
            raise OverlayError(
                "overlay execution-upgrade history chain is incomplete"
            )
        visited.add(cursor)
        record = history_records[cursor]
        cursor = record.get("previous_upgrade_sha256")
    _effective_memory_safety(plan, upgrade)
    return upgrade


def _effective_memory_safety(
    plan: dict[str, Any], upgrade: dict[str, Any] | None
) -> dict[str, Any]:
    parent = plan["memory_safety"]
    if upgrade is None or "effective_memory_safety" not in upgrade:
        return parent
    effective = upgrade.get("effective_memory_safety")
    raised = {
        **parent,
        "cuda_allocation_limit_bytes": MTP_RESUME_CUDA_ALLOCATION_LIMIT_BYTES,
    }
    contract = upgrade.get("change_contract")
    if effective == parent:
        return effective
    if (
        effective != raised
        or not isinstance(contract, dict)
        or contract.get("memory_safety") != "cuda-allocation-ceiling-84-gib"
        or upgrade.get("previous_upgrade_sha256") is None
    ):
        raise OverlayError("overlay execution-upgrade memory safety is invalid")
    return effective


def build_execution_upgrade(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authorize boundary-only resume code without rewriting quantization identity."""

    if not args.resume:
        raise OverlayError("--execution-upgrade requires --resume")
    output = args.output.expanduser().resolve()
    run_state = (
        args.run_state_dir.expanduser().resolve()
        if args.run_state_dir is not None
        else output.with_name(f".{output.name}.ds4rt-run")
    )
    _regular_directory(run_state, "overlay run-state")
    saved = base.read_json_object(run_state / base.PLAN_FILENAME)
    _validate_plan(saved)
    _validate_run_entries(run_state)
    requested = build_plan(args)

    saved_preflight = saved.get("preflight")
    requested_preflight = requested.get("preflight")
    if not isinstance(saved_preflight, dict) or not isinstance(
        requested_preflight, dict
    ):
        raise OverlayError("overlay execution upgrade lacks coordinator preflight")
    existing = None
    existing_path = run_state / EXECUTION_UPGRADE_FILENAME
    if existing_path.exists() or existing_path.is_symlink():
        existing = _read_execution_upgrade(run_state, saved)
    stable_keys = ("python", "torch", "gpus")
    if any(
        requested_preflight.get(key) != saved_preflight.get(key)
        for key in stable_keys
    ):
        raise OverlayError(
            "overlay execution upgrade changes GPTQModel, Python, Torch, or GPUs"
        )
    gptq_changed = (
        requested_preflight.get("gptqmodel")
        != saved_preflight.get("gptqmodel")
    )
    if existing is None and gptq_changed:
        raise OverlayError(
            "first overlay execution upgrade changes GPTQModel"
        )
    if saved.get("remote_workers") is not None or requested.get("remote_workers") is not None:
        raise OverlayError(
            "overlay execution upgrade currently requires coordinator-only execution"
        )
    requested_neutral = _execution_neutral_plan(requested)
    saved_neutral = _execution_neutral_plan(saved)
    if gptq_changed:
        requested_neutral["ledger_provenance"]["family_join"].pop(
            "gptqmodel", None
        )
        saved_neutral["ledger_provenance"]["family_join"].pop(
            "gptqmodel", None
        )
    if requested_neutral != saved_neutral:
        raise OverlayError("overlay execution-upgrade inputs differ from parent plan")
    if requested_preflight.get("image_digest") == saved_preflight.get("image_digest"):
        raise OverlayError("overlay execution upgrade does not change the image")

    script = Path(__file__).resolve()
    replacement = existing is not None and (
        existing.get("upgraded_execution", {}).get("image_digest")
        != requested_preflight.get("image_digest")
        or existing.get("upgraded_execution", {}).get("gptqmodel")
        != requested_preflight.get("gptqmodel")
    )
    replacement_execution = replacement or bool(
        existing is not None
        and existing.get("previous_upgrade_sha256") is not None
        and existing.get("upgraded_execution", {}).get("image_digest")
        == requested_preflight.get("image_digest")
        and existing.get("upgraded_execution", {}).get("gptqmodel")
        == requested_preflight.get("gptqmodel")
    )
    effective_memory_safety = dict(saved["memory_safety"])
    if replacement_execution:
        effective_memory_safety["cuda_allocation_limit_bytes"] = (
            MTP_RESUME_CUDA_ALLOCATION_LIMIT_BYTES
        )
    body = {
        "schema": EXECUTION_UPGRADE_SCHEMA,
        "parent_plan_sha256": saved["plan_sha256"],
        "parent_execution": {
            "image_digest": saved_preflight.get("image_digest"),
            "preflight_sha256": saved_preflight.get("sha256"),
        },
        "upgraded_execution": {
            "image_digest": requested_preflight.get("image_digest"),
            "preflight_sha256": requested_preflight.get("sha256"),
            "gptqmodel": requested_preflight.get("gptqmodel"),
            "python": requested_preflight.get("python"),
            "torch": requested_preflight.get("torch"),
            "gpus": requested_preflight.get("gpus"),
            "overlay_script_sha256": sha256_file(script),
        },
        "resume_state": _activation_boundary_upgrade_identity(saved),
        "change_contract": {
            "purpose": "mtp-post-block-activation-boundary-resume",
            "quantization_plan": "unchanged-parent-plan",
            "projection_family_join": "unchanged-parent-plan",
            "gptqmodel_source": (
                "preferred-forward-device-materialization-only"
                if replacement_execution and gptq_changed
                else "unchanged-parent-plan"
            ),
            "quantizer_algorithm": "unchanged-seeded-exl3-trellis",
            "new_behavior": "skip-hash-verified-completed-mtp-blocks",
            "memory_safety": (
                "cuda-allocation-ceiling-84-gib"
                if replacement_execution
                else "unchanged-parent-plan"
            ),
        },
        "effective_memory_safety": effective_memory_safety,
    }
    if replacement_execution:
        body["previous_upgrade_sha256"] = (
            existing["upgrade_sha256"]
            if replacement
            else existing["previous_upgrade_sha256"]
        )
    upgrade = _bound(body, "upgrade_sha256")
    path = run_state / EXECUTION_UPGRADE_FILENAME
    if existing is not None:
        # The record authorizes this execution image from the boundary that
        # existed when the code changed.  Later boundaries are outputs of that
        # already-authorized image, so advancing the rolling frontier must not
        # rewrite the authorization or require the older frontier to survive.
        stable_fields = (
            "schema",
            "parent_plan_sha256",
            "parent_execution",
            "upgraded_execution",
            "change_contract",
            "effective_memory_safety",
            "previous_upgrade_sha256",
        )
        if any(existing.get(field) != upgrade.get(field) for field in stable_fields):
            if not replacement:
                raise OverlayError("overlay execution-upgrade record differs")
            history = run_state / EXECUTION_UPGRADE_HISTORY_DIRNAME
            history.mkdir(exist_ok=True)
            _regular_directory(history, "overlay execution-upgrade history")
            archived = history / f"{existing['upgrade_sha256']}.json"
            if archived.exists():
                if base.read_json_object(archived) != existing:
                    raise OverlayError("overlay execution-upgrade history collision")
            else:
                base.atomic_json(archived, existing)
            base.atomic_json(path, upgrade)
            return saved, upgrade
        existing_layer = existing.get("resume_state", {}).get("layer_index")
        current_layer = body["resume_state"]["layer_index"]
        if (
            isinstance(existing_layer, bool)
            or not isinstance(existing_layer, int)
            or existing_layer > current_layer
        ):
            raise OverlayError("overlay execution-upgrade frontier regressed")
        return saved, existing
    else:
        base.atomic_json(path, upgrade)
    return saved, upgrade


def _regular_directory(path: Path, label: str) -> None:
    if not path.is_dir() or path.is_symlink():
        raise OverlayError(f"{label} is not one regular directory: {path}")


def _empty_or_missing(path: Path, label: str) -> None:
    if path.is_symlink():
        raise OverlayError(f"{label} is a symbolic link")
    if path.exists():
        _regular_directory(path, label)
        if any(path.iterdir()):
            raise OverlayError(f"{label} is not empty")


def _validate_run_entries(run_state: Path) -> None:
    allowed = {
        base.PLAN_FILENAME,
        EXECUTION_UPGRADE_FILENAME,
        EXECUTION_UPGRADE_HISTORY_DIRNAME,
        base.ERROR_JOURNAL_FILENAME,
        base.PROJECTION_CHECKPOINT_DIRNAME,
        base.ACTIVE_LAYER_SOURCE_DIRNAME,
        base.REMOTE_ASSIGNMENT_DIRNAME,
        ACTIVATION_DIRNAME,
        MTP_CAPTURE_FRONTIER_DIRNAME,
        base.CAPTURE_BATCH_SPOOL_DIRNAME,
        REPORT_DIRNAME,
        EXPORT_STAGE_DIRNAME,
        SELECTION_FILENAME,
    }
    unexpected = sorted(path.name for path in run_state.iterdir() if path.name not in allowed)
    if unexpected:
        raise OverlayError("overlay run state has unexpected entries: " + ", ".join(unexpected))


def _export_stage_path(plan: dict[str, Any]) -> Path:
    """Return an atomic publication stage on the output filesystem."""

    output = Path(plan["output"])
    return output.with_name(f".{output.name}.{EXPORT_STAGE_DIRNAME}")


def prepare_run(plan: dict[str, Any], *, resume: bool) -> bool:
    _validate_plan(plan)
    output = Path(plan["output"])
    run_state = Path(plan["run_state_dir"])
    offload = Path(plan["offload_dir"])
    checkpoint_root = Path(plan["projection_checkpoint"]["root"])
    active_source = Path(plan["active_layer_source_dir"])
    prefix = Path(plan["mtp_prefix_store"])
    output.parent.mkdir(parents=True, exist_ok=True)

    if resume and (output.exists() or output.is_symlink()):
        validate_overlay(output, plan, verify_hashes=True)
        return True
    if resume:
        _regular_directory(run_state, "overlay run-state")
        if base.read_json_object(run_state / base.PLAN_FILENAME) != plan:
            raise OverlayError("overlay run-state plan differs")
        _validate_run_entries(run_state)
        _regular_directory(checkpoint_root, "overlay projection checkpoints")
        _regular_directory(active_source, "overlay active-layer source")
        _regular_directory(offload, "overlay offload")
        _regular_directory(prefix, "parent prefix")
        export_stages = (
            _export_stage_path(plan),
            # Compatibility with runs interrupted before the publication
            # stage moved beside the output.
            run_state / EXPORT_STAGE_DIRNAME,
        )
        for export_stage in export_stages:
            if export_stage.exists() or export_stage.is_symlink():
                _regular_directory(export_stage, "overlay export stage")
                shutil.rmtree(export_stage)
        return False

    _empty_or_missing(output, "overlay output")
    export_stage = _export_stage_path(plan)
    _empty_or_missing(export_stage, "overlay export stage")
    _empty_or_missing(run_state, "overlay run-state")
    _empty_or_missing(checkpoint_root, "overlay projection checkpoints")
    _empty_or_missing(active_source, "overlay active-layer source")
    _empty_or_missing(offload, "overlay offload")
    _regular_directory(prefix, "parent prefix")
    if output.exists():
        output.rmdir()
    if export_stage.exists():
        export_stage.rmdir()
    if run_state.exists():
        run_state.rmdir()
    run_state.mkdir(parents=True)
    base.atomic_json(run_state / base.PLAN_FILENAME, plan)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    active_source.mkdir(parents=True, exist_ok=True)
    # An explicitly qualified empty offload directory may be a container bind
    # mount and therefore cannot be removed/recreated at the mount point.
    offload.mkdir(parents=True, exist_ok=True)
    return False


def _load_target_embedding(snapshot: Path, *, vocab_size: int, hidden_size: int):
    import torch
    from safetensors import safe_open

    index = base.read_json_object(snapshot / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    # The official V4 checkpoint uses its native ``embed.weight`` spelling.
    # A canonical Transformers artifact may already use the mapped name, so
    # accept exactly one of the two representations and preserve its BF16
    # values byte-for-byte.
    names = (
        "embed.weight",
        "model.embed_tokens.weight",
    )
    present = [
        name for name in names if isinstance(weight_map, dict) and name in weight_map
    ]
    if len(present) != 1:
        raise OverlayError(
            "source checkpoint must contain exactly one target embedding tensor"
        )
    name = present[0]
    shard = weight_map.get(name)
    if not isinstance(shard, str) or Path(shard).name != shard:
        raise OverlayError("source checkpoint has no target embedding tensor")
    with safe_open(snapshot / shard, framework="pt", device="cpu") as source:
        if name not in source.keys():
            raise OverlayError("target embedding index points to the wrong shard")
        embedding = source.get_tensor(name)
    if (
        tuple(embedding.shape) != (vocab_size, hidden_size)
        or embedding.dtype is not torch.bfloat16
    ):
        raise OverlayError("target embedding geometry or dtype is invalid")
    return embedding


def _link_tree(source: Path, target: Path) -> None:
    _regular_directory(source, "overlay evidence tree")
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        destination = target / relative
        if path.is_symlink():
            raise OverlayError(f"overlay evidence contains a symlink: {path}")
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, destination)
            # Projection checkpoints and dynamic assignments may be created
            # under a restrictive container umask.  A published artifact is
            # consumed by an unprivileged host process, so its evidence must
            # remain readable after the container exits.  chmod applies to the
            # shared hard-link inode and therefore also makes resume evidence
            # consistently readable.
            destination.chmod(0o644)
        else:
            raise OverlayError(f"overlay evidence contains an unsupported entry: {path}")


def _copy_tree(source: Path, target: Path) -> None:
    """Copy small evidence trees that may cross the run/output filesystems."""

    _regular_directory(source, "overlay evidence tree")
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        destination = target / relative
        if path.is_symlink():
            raise OverlayError(f"overlay evidence contains a symlink: {path}")
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            destination.chmod(0o644)
        else:
            raise OverlayError(f"overlay evidence contains an unsupported entry: {path}")


def _checkpoint_index(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    modules: dict[str, Any] = {}
    for manifest_path in sorted(root.rglob("*.json")):
        manifest = base.read_json_object(manifest_path)
        request = manifest.get("request")
        module = request.get("module") if isinstance(request, dict) else None
        request_sha256 = request.get("request_sha256") if isinstance(request, dict) else None
        tensor_path = manifest_path.with_suffix(".safetensors")
        if (
            not isinstance(module, str)
            or not module.startswith("mtp.")
            or module in modules
            or not isinstance(request_sha256, str)
            or manifest_path.stem != request_sha256
            or not tensor_path.is_file()
            or tensor_path.is_symlink()
        ):
            raise OverlayError("overlay checkpoint index contains an invalid projection")
        modules[module] = {
            "request_sha256": request_sha256,
            "manifest": manifest_path.relative_to(root).as_posix(),
            "manifest_sha256": manifest.get("manifest_sha256"),
            "tensor_file": tensor_path.relative_to(root).as_posix(),
            "tensor_sha256": manifest.get("tensor_sha256"),
            "tensors": manifest.get("tensors"),
        }
    geometry = plan["source"]["geometry"]
    expected = len(geometry["dspark_target_layer_ids"]) * geometry["n_routed_experts"] * 3
    if len(modules) != expected:
        raise OverlayError(f"overlay checkpoint index has {len(modules)}/{expected} projections")
    body = {
        "schema": INDEX_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "bits": plan["exl3"]["bits"],
        "projection_count": len(modules),
        "modules": modules,
    }
    return _bound(body, "index_sha256")


def _selection_record(plan: dict[str, Any], replay) -> dict[str, Any]:
    selection = replay.anchor_selection_identity
    batching = replay.replay_batching_identity
    if (
        selection.get("contract") != plan["anchor_selection"]["contract"]
        or selection.get("seed") != plan["anchor_selection"]["seed"]
        or selection.get("selected_position_count")
        != plan["anchor_selection"]["count"]
        or batching.get("contract") != plan["replay_batching"]["contract"]
        or batching.get("source_sequence_anchor_cap")
        != plan["replay_batching"]["source_sequence_anchor_cap"]
    ):
        raise OverlayError("materialized MTP anchor selection differs from the plan")
    record = _bound(
        {
            "schema": "ds4rt-mtp-anchor-selection-v1",
            "plan_sha256": plan["plan_sha256"],
            "prefix_manifest_sha256": plan["prefix"]["manifest_sha256"],
            "anchor_selection": selection,
            "replay_batching": batching,
        },
        "selection_sha256",
    )
    _validate_selection_record(plan, record)
    return record


def _validate_selection_record(
    plan: dict[str, Any], record: dict[str, Any]
) -> None:
    _validate_bound(record, "selection_sha256", "MTP anchor selection")
    selection = record.get("anchor_selection")
    batching = record.get("replay_batching")
    selected = (
        selection.get("selected_position_count")
        if isinstance(selection, dict)
        else None
    )
    source = (
        selection.get("source_position_count")
        if isinstance(selection, dict)
        else None
    )
    batch_count = (
        batching.get("batch_count") if isinstance(batching, dict) else None
    )
    minimum = (
        batching.get("minimum_anchors") if isinstance(batching, dict) else None
    )
    maximum = (
        batching.get("maximum_anchors") if isinstance(batching, dict) else None
    )
    if (
        record.get("schema") != "ds4rt-mtp-anchor-selection-v1"
        or record.get("plan_sha256") != plan["plan_sha256"]
        or record.get("prefix_manifest_sha256")
        != plan["prefix"]["manifest_sha256"]
        or not isinstance(selection, dict)
        or selection.get("contract") != plan["anchor_selection"]["contract"]
        or selection.get("seed") != plan["anchor_selection"]["seed"]
        or isinstance(selected, bool)
        or not isinstance(selected, int)
        or selected != plan["anchor_selection"]["count"]
        or isinstance(source, bool)
        or not isinstance(source, int)
        or source < selected
        or not isinstance(selection.get("selected_coordinates_sha256"), str)
        or base.SHA256_RE.fullmatch(selection["selected_coordinates_sha256"])
        is None
        or not isinstance(batching, dict)
        or batching.get("contract") != plan["replay_batching"]["contract"]
        or batching.get("source_sequence_anchor_cap")
        != plan["replay_batching"]["source_sequence_anchor_cap"]
        or isinstance(batch_count, bool)
        or not isinstance(batch_count, int)
        or not 0 < batch_count <= selected
        or isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or not 0 < minimum <= maximum <= selected
        or not minimum * batch_count <= selected <= maximum * batch_count
        or not isinstance(batching.get("row_counts_sha256"), str)
        or base.SHA256_RE.fullmatch(batching["row_counts_sha256"]) is None
    ):
        raise OverlayError("MTP anchor selection record is invalid")


def _write_or_verify_selection(plan: dict[str, Any], replay) -> dict[str, Any]:
    record = _selection_record(plan, replay)
    path = Path(plan["run_state_dir"]) / SELECTION_FILENAME
    if path.exists() or path.is_symlink():
        if not path.is_file() or path.is_symlink():
            raise OverlayError("MTP anchor selection record is not a regular file")
        if base.read_json_object(path) != record:
            raise OverlayError("MTP anchor selection changed across resume")
    else:
        base.atomic_json(path, record)
    return record


def _recovery_authorization(plan: dict[str, Any]) -> dict[str, Any] | None:
    family_join = plan["ledger_provenance"]["family_join"]
    family_digest = hashlib.sha256(base.canonical_json(family_join)).hexdigest()
    if family_join.get("zero_route_recovery_contract") == ZERO_ROUTE_RECOVERY_SCHEMA:
        authorization_sha256 = family_digest
    else:
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
        "authorization_sha256": authorization_sha256,
    }


def publish_overlay(
    plan: dict[str, Any], replay_batches: int, selection: dict[str, Any]
) -> None:
    from validate_projection_checkpoint_block import audit_block

    _validate_plan(plan)
    run_state = Path(plan["run_state_dir"])
    output = Path(plan["output"])
    export_stage = _export_stage_path(plan)
    if output.exists() or export_stage.exists():
        raise OverlayError("overlay publication target already exists")
    reports = run_state / REPORT_DIRNAME
    reports.mkdir(exist_ok=True)
    audits = []
    for block_index in range(len(plan["source"]["geometry"]["dspark_target_layer_ids"])):
        report = audit_block(
            run_state,
            block_namespace="mtp",
            logical_layer=block_index,
        )
        report_path = reports / f"mtp-layer-{block_index}-projection-audit.json"
        base.atomic_json(report_path, report)
        audits.append(
            {
                "block_index": block_index,
                "report": report_path.name,
                "report_sha256": report["report_sha256"],
            }
        )

    export_stage.mkdir()
    checkpoint_source = Path(plan["projection_checkpoint"]["root"])
    if checkpoint_source.stat().st_dev != export_stage.parent.stat().st_dev:
        raise OverlayError(
            "overlay output must share a filesystem with projection checkpoints"
        )
    _link_tree(checkpoint_source, export_stage / base.PROJECTION_CHECKPOINT_DIRNAME)
    assignment_source = run_state / base.REMOTE_ASSIGNMENT_DIRNAME
    if assignment_source.exists():
        _copy_tree(assignment_source, export_stage / base.REMOTE_ASSIGNMENT_DIRNAME)
    _copy_tree(reports, export_stage / REPORT_DIRNAME)
    journal = run_state / base.ERROR_JOURNAL_FILENAME
    if not journal.is_file() or journal.is_symlink():
        raise OverlayError("overlay projection journal is unavailable")
    shutil.copy2(journal, export_stage / base.ERROR_JOURNAL_FILENAME)
    (export_stage / base.ERROR_JOURNAL_FILENAME).chmod(0o644)
    selection_path = run_state / SELECTION_FILENAME
    if base.read_json_object(selection_path) != selection:
        raise OverlayError("overlay anchor selection evidence changed before publication")
    shutil.copy2(selection_path, export_stage / SELECTION_FILENAME)
    (export_stage / SELECTION_FILENAME).chmod(0o644)
    execution_upgrade_path = run_state / EXECUTION_UPGRADE_FILENAME
    execution_upgrade = None
    if execution_upgrade_path.exists() or execution_upgrade_path.is_symlink():
        execution_upgrade = _read_execution_upgrade(run_state, plan)
        base.atomic_json(
            export_stage / EXECUTION_UPGRADE_FILENAME,
            execution_upgrade,
        )
        history = run_state / EXECUTION_UPGRADE_HISTORY_DIRNAME
        if history.exists():
            _copy_tree(
                history,
                export_stage / EXECUTION_UPGRADE_HISTORY_DIRNAME,
            )
    base.atomic_json(export_stage / base.PLAN_FILENAME, plan)
    index = _checkpoint_index(checkpoint_source, plan)
    base.atomic_json(export_stage / INDEX_FILENAME, index)

    records = {}
    for path in base._artifact_paths(export_stage):
        relative = path.relative_to(export_stage).as_posix()
        records[relative] = base._artifact_file_identity(path)
    manifest = _bound(
        {
            "schema": OVERLAY_SCHEMA,
            "status": "complete",
            "plan_sha256": plan["plan_sha256"],
            "bits": plan["exl3"]["bits"],
            "codebook": "mcg",
            "target_parent": plan["target_parent"],
            "prefix": plan["prefix"],
            "selection_sha256": selection["selection_sha256"],
            "execution_upgrade_sha256": (
                execution_upgrade["upgrade_sha256"]
                if execution_upgrade is not None
                else None
            ),
            "checkpoint_index_sha256": index["index_sha256"],
            "block_audits": audits,
            "file_count": len(records),
            "total_bytes": sum(record["bytes"] for record in records.values()),
            "files": records,
        },
        "overlay_sha256",
    )
    base.atomic_json(export_stage / OVERLAY_FILENAME, manifest)
    run = _bound(
        {
            "schema": RUN_SCHEMA,
            "status": "complete",
            "plan_sha256": plan["plan_sha256"],
            "overlay_sha256": manifest["overlay_sha256"],
            "replay_batches": replay_batches,
            "replay_positions": selection["anchor_selection"][
                "selected_position_count"
            ],
            "selection_sha256": selection["selection_sha256"],
            "execution_upgrade_sha256": (
                execution_upgrade["upgrade_sha256"]
                if execution_upgrade is not None
                else None
            ),
        },
        "run_sha256",
    )
    base.atomic_json(export_stage / RUN_FILENAME, run)
    validate_overlay(export_stage, plan, verify_hashes=False)
    os.replace(export_stage, output)
    descriptor = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    validate_overlay(output, plan, verify_hashes=False)
    _prune_completed_activation_frontier(plan)


def validate_overlay(root: Path, plan: dict[str, Any], *, verify_hashes: bool) -> dict[str, Any]:
    _validate_plan(plan)
    _regular_directory(root, "published dSpark overlay")
    if base.read_json_object(root / base.PLAN_FILENAME) != plan:
        raise OverlayError("published overlay plan differs")
    manifest = base.read_json_object(root / OVERLAY_FILENAME)
    run = base.read_json_object(root / RUN_FILENAME)
    selection = base.read_json_object(root / SELECTION_FILENAME)
    upgrade_path = root / EXECUTION_UPGRADE_FILENAME
    upgrade = (
        _read_execution_upgrade(root, plan)
        if upgrade_path.exists() or upgrade_path.is_symlink()
        else None
    )
    _validate_bound(manifest, "overlay_sha256", "overlay manifest")
    _validate_bound(run, "run_sha256", "overlay run")
    _validate_selection_record(plan, selection)
    records = manifest.get("files")
    if (
        manifest.get("schema") != OVERLAY_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("plan_sha256") != plan["plan_sha256"]
        or manifest.get("bits") != plan["exl3"]["bits"]
        or not isinstance(records, dict)
        or manifest.get("file_count") != len(records)
        or run.get("schema") != RUN_SCHEMA
        or run.get("status") != "complete"
        or run.get("plan_sha256") != plan["plan_sha256"]
        or run.get("overlay_sha256") != manifest.get("overlay_sha256")
        or selection.get("schema") != "ds4rt-mtp-anchor-selection-v1"
        or selection.get("plan_sha256") != plan["plan_sha256"]
        or selection.get("prefix_manifest_sha256")
        != plan["prefix"]["manifest_sha256"]
        or selection.get("anchor_selection", {}).get("contract")
        != plan["anchor_selection"]["contract"]
        or selection.get("anchor_selection", {}).get("seed")
        != plan["anchor_selection"]["seed"]
        or selection.get("anchor_selection", {}).get("selected_position_count")
        != plan["anchor_selection"]["count"]
        or selection.get("replay_batching", {}).get("contract")
        != plan["replay_batching"]["contract"]
        or selection.get("replay_batching", {}).get("source_sequence_anchor_cap")
        != plan["replay_batching"]["source_sequence_anchor_cap"]
        or manifest.get("selection_sha256") != selection.get("selection_sha256")
        or run.get("selection_sha256") != selection.get("selection_sha256")
        or manifest.get("execution_upgrade_sha256")
        != (upgrade.get("upgrade_sha256") if upgrade is not None else None)
        or run.get("execution_upgrade_sha256")
        != (upgrade.get("upgrade_sha256") if upgrade is not None else None)
        or run.get("replay_batches")
        != selection.get("replay_batching", {}).get("batch_count")
        or run.get("replay_positions")
        != selection.get("anchor_selection", {}).get("selected_position_count")
    ):
        raise OverlayError("published overlay metadata is inconsistent")
    expected = set(records) | {OVERLAY_FILENAME, RUN_FILENAME}
    actual = {
        path.relative_to(root).as_posix() for path in base._artifact_paths(root)
    }
    if actual != expected:
        raise OverlayError("published overlay file inventory differs")
    for relative, identity in records.items():
        path = root / relative
        if (
            not isinstance(identity, dict)
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != identity.get("bytes")
            or verify_hashes and base._artifact_file_identity(path) != identity
        ):
            raise OverlayError(f"published overlay file failed validation: {relative}")
    return run


def _prune_completed_activation_frontier(plan: dict[str, Any]) -> None:
    run_state = Path(plan["run_state_dir"])
    for dirname, label in (
        (ACTIVATION_DIRNAME, "completed overlay activation frontier"),
        (MTP_CAPTURE_FRONTIER_DIRNAME, "completed overlay Hessian frontier"),
        (base.CAPTURE_BATCH_SPOOL_DIRNAME, "completed overlay capture batches"),
    ):
        activation_root = run_state / dirname
        if not (activation_root.exists() or activation_root.is_symlink()):
            continue
        _regular_directory(
            activation_root,
            label,
        )
        shutil.rmtree(activation_root)


def execute(
    plan: dict[str, Any],
    *,
    resume: bool,
    execution_upgrade: dict[str, Any] | None = None,
) -> None:
    run_state = Path(plan["run_state_dir"])
    upgrade_path = run_state / EXECUTION_UPGRADE_FILENAME
    if execution_upgrade is None:
        if upgrade_path.exists() or upgrade_path.is_symlink():
            raise OverlayError(
                "overlay run state requires explicit --execution-upgrade authorization"
            )
    elif _read_execution_upgrade(run_state, plan) != execution_upgrade:
        raise OverlayError("overlay execution-upgrade authorization changed")
    if prepare_run(plan, resume=resume):
        _prune_completed_activation_frontier(plan)
        return
    remote = plan["remote_workers"]
    if remote is not None:
        token_env = remote["token_env"]
        if not os.environ.get(token_env):
            raise OverlayError(f"remote worker token env `{token_env}` is unset")
        required_workers = str(remote["cuda_workers_per_device"])
        configured = os.environ.get("GPTQMODEL_CUDA_WORKERS")
        if configured not in {None, required_workers}:
            raise OverlayError("GPTQMODEL_CUDA_WORKERS conflicts with the overlay plan")
        os.environ["GPTQMODEL_CUDA_WORKERS"] = required_workers

    import torch
    from transformers import AutoConfig
    from gptqmodel.models.definitions.deepseek_v4 import (
        DeepSeekV4MTPAuxiliary,
        DeepSeekV4MTPQuantizationModel,
    )
    from gptqmodel.quantization import AutoModuleDecoderConfig, EXL3Config

    os.environ["GPTQMODEL_EXL3_ERROR_JOURNAL"] = os.fspath(
        run_state / base.ERROR_JOURNAL_FILENAME
    )
    prefix = DeepSeekV4MTPPrefixStore.open_complete(
        Path(plan["mtp_prefix_store"]),
        expected_manifest_sha256=plan["prefix"]["manifest_sha256"],
    )
    replay = prefix.replay_dataset(
        # The fixed-width value is unused in source-sequence mode but remains
        # explicit in the generic prefix-store API.
        replay_batch_size=1,
        device="cpu",
        anchor_sample_count=plan["anchor_selection"]["count"],
        anchor_sample_seed=plan["anchor_selection"]["seed"],
        batch_by_source_sequence=True,
        source_sequence_anchor_cap=plan["replay_batching"][
            "source_sequence_anchor_cap"
        ],
    )
    # Persist this before loading or executing any MTP block. A resume can
    # therefore never reuse Hessians or projections from a different subset.
    selection = _write_or_verify_selection(plan, replay)
    qcfg_meta = {
        "ds4rt_error_ledger": plan["ledger_provenance"],
        HESSIAN_OWNER_POLICY_META: plan["exl3"]["hessian_owner_policy"],
    }
    recovery_authorization = _recovery_authorization(plan)
    if recovery_authorization is not None:
        qcfg_meta["ds4rt_zero_route_recovery"] = recovery_authorization
    coordinator_devices = [
        f"cuda:{gpu['index']}" for gpu in plan["preflight"]["gpus"]
    ]
    qcfg = EXL3Config(
        bits=float(plan["exl3"]["bits"]),
        codebook="mcg",
        out_scales="auto",
        module_include=[MTP_EXPERT_PATTERN],
        preprocessors=[AutoModuleDecoderConfig(target_dtype=torch.bfloat16)],
        fallback=None,
        offload_to_disk=True,
        offload_to_disk_path=plan["offload_dir"],
        device=coordinator_devices[0],
        calibration_data_device="cpu",
        dense_vram_strategy_devices=coordinator_devices,
        moe_vram_strategy="balanced",
        moe_vram_strategy_devices=coordinator_devices,
        meta=qcfg_meta,
    )
    snapshot = Path(plan["source"]["path"])
    config = AutoConfig.from_pretrained(snapshot, trust_remote_code=False)
    auxiliary = DeepSeekV4MTPAuxiliary.from_checkpoint(
        config=config,
        model_local_path=os.fspath(snapshot),
    )
    auxiliary.turtle_model.configure_active_source_staging(
        plan["active_layer_source_dir"],
        provenance={
            "plan_sha256": plan["plan_sha256"],
            "source_revision": plan["source"]["revision"],
            "source_index_sha256": plan["source"]["index_sha256"],
        },
    )
    embedding = _load_target_embedding(
        snapshot,
        vocab_size=int(config.vocab_size),
        hidden_size=int(config.hidden_size),
    )
    model = DeepSeekV4MTPQuantizationModel.from_auxiliary(
        auxiliary=auxiliary,
        embedding_weight=embedding,
        quantize_config=qcfg,
        model_local_path=os.fspath(snapshot),
    )
    activation_provenance = _activation_provenance(plan, selection)
    if (
        activation_provenance["replay_batches"] != len(replay)
        or activation_provenance["replay_positions"] != replay.position_count
    ):
        raise OverlayError("materialized MTP replay differs from its selection")
    activation_root = run_state / ACTIVATION_DIRNAME
    model.configure_mtp_activation_store(
        os.fspath(activation_root),
        provenance=activation_provenance,
    )
    model.quantization_layer_boundary_checkpoint = _activation_boundary_controller(
        plan,
        selection,
    )
    memory_safety = _effective_memory_safety(plan, execution_upgrade)
    with base.capture_frontier_scope(
        run_state / MTP_CAPTURE_FRONTIER_DIRNAME
    ), base.capture_batch_spool_scope(
        run_state / base.CAPTURE_BATCH_SPOOL_DIRNAME
    ), base.memory_safety_scope(memory_safety):
        model.quantize(replay, batch_size=1, calibration_sort=None)
    publish_overlay(plan, len(replay), selection)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--mtp-prefix-store", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-state-dir", type=Path)
    parser.add_argument("--projection-checkpoint-dir", type=Path)
    parser.add_argument("--active-layer-source-dir", type=Path)
    parser.add_argument("--offload-dir", type=Path, required=True)
    parser.add_argument("--bits", type=int, choices=(2, 3), required=True)
    parser.add_argument(
        "--coordinator-gpu-count",
        type=int,
        choices=(1, 2),
        default=2,
    )
    parser.add_argument(
        "--gptqmodel-lock",
        type=Path,
        default=root / "third_party" / "gptqmodel.lock.json",
    )
    parser.add_argument("--mtp-anchor-sample-count", type=int, default=327680)
    parser.add_argument("--mtp-anchor-sample-seed", type=int, default=20260809)
    parser.add_argument(
        "--mtp-hessian-owner-device-weights",
        type=int,
        nargs="+",
        help=(
            "positive deterministic Hessian-capture ownership weights, one "
            "per coordinator GPU; defaults to equal ownership"
        ),
    )
    parser.add_argument(
        "--mtp-sequence-anchor-cap",
        type=int,
        default=0,
        help="maximum selected anchors per source-sequence replay item; 0 keeps each sequence whole",
    )
    parser.add_argument(
        "--remote-worker",
        action="append",
        nargs=3,
        metavar=("NAME", "URL", "PREFLIGHT"),
        help=(
            "repeat once per Spark worker; omit for coordinator-only overlay "
            "quantization"
        ),
    )
    parser.add_argument(
        "--remote-token-env",
        default="DS4RT_EXL3_WORKER_TOKEN",
    )
    parser.add_argument("--remote-timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--remote-max-attempts", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--execution-upgrade",
        action="store_true",
        help=(
            "resume an immutable parent plan from a fully verified completed "
            "MTP activation boundary under this repaired execution image"
        ),
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.execution_upgrade and not args.resume:
        parser.error("--execution-upgrade requires --resume")
    return args


def main() -> int:
    args = parse_args()
    try:
        execution_upgrade = None
        if args.execution_upgrade:
            plan, execution_upgrade = build_execution_upgrade(args)
            print(json.dumps(execution_upgrade, indent=2, sort_keys=True), flush=True)
        else:
            plan = build_plan(args)
        if args.plan_only:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        execute(
            plan,
            resume=args.resume,
            execution_upgrade=execution_upgrade,
        )
        return 0
    except (
        base.LaunchError,
        OverlayError,
        PrefixStoreError,
        OSError,
        ValueError,
    ) as error:
        print(f"quantize-flash-dspark-overlay: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
