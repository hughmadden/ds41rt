#!/usr/bin/env python3
"""Measure real DeepSeek V4 native-to-EXL3 error and strict-TP4 parity."""

from __future__ import annotations

import argparse
from copy import copy
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterator


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _pinned_sparkinfer  # noqa: E402,F401

from ds4rt_runtime.exl3_artifact_contract import (  # noqa: E402
    ARTIFACT_FILE as GPTQMODEL_ARTIFACT_FILE,
    LEDGER_FILE as GPTQMODEL_LEDGER_FILE,
    LEDGER_MANIFEST_FILE as GPTQMODEL_LEDGER_MANIFEST_FILE,
    PLAN_FILE as GPTQMODEL_PLAN_FILE,
    RUN_FILE as GPTQMODEL_RUN_FILE,
    is_gptqmodel_native_exl3,
    validate_gptqmodel_native_exl3,
)
from ds4rt_runtime.exl3_experts import (  # noqa: E402
    ValidatedExl3ExpertSnapshot,
    load_exl3_expert_reference_layer,
    load_exl3_expert_tp_layer,
    validate_exl3_expert_snapshot,
)
from ds4rt_runtime.exl3_quantizer import (  # noqa: E402
    deterministic_expert_row_indices,
    load_activation_corpus,
    load_activation_layer_samples,
    load_routed_activation_layer,
    read_native_model_config,
)
from ds4rt_runtime.native_experts import (  # noqa: E402
    load_native_expert_reference_layer,
    read_native_expert_config,
)


MIN_QUANT_COSINE = 0.88
MAX_QUANT_RELATIVE_L2 = 0.50
MIN_DSPARK_QUANT_COSINE = 0.84
MAX_DSPARK_QUANT_RELATIVE_L2 = 0.56
PROGRESS_SCHEMA = "ds4rt-exl3-checkpoint-quality-progress-v1"
GPTQMODEL_CALIBRATION_FORMAT = "gptqmodel-native-exl3-v4"
QUALITY_GPU_SCHEMA = "ds4rt-quality-physical-gpu-v1"
NATURAL_TARGET_SYNTHETIC_MTP = "natural-target-stratified-mtp-rms-isotropic"
GPU_UUID_RE = re.compile(
    r"GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\Z"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-snapshot", type=Path, required=True)
    parser.add_argument("--exl3-snapshot", type=Path, required=True)
    layers = parser.add_mutually_exclusive_group(required=True)
    layers.add_argument("--layer-id", type=int)
    layers.add_argument(
        "--all-layers",
        action="store_true",
        help="validate every target and dSpark expert block",
    )
    layers.add_argument(
        "--target-layers",
        action="store_true",
        help="validate every target block, excluding dSpark/MTP blocks",
    )
    layers.add_argument(
        "--learned-target-layers",
        action="store_true",
        help="validate target blocks that use a learned correction-bias router",
    )
    parser.add_argument(
        "--expert-ids",
        default="auto",
        help=(
            "six comma-separated experts, or auto for six experts stratified "
            "across the checkpoint (default: %(default)s)"
        ),
    )
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument(
        "--sampling-mode",
        choices=(
            "synthetic-stratified",
            "natural-stratified",
            "natural-route-events",
            NATURAL_TARGET_SYNTHETIC_MTP,
        ),
        default="synthetic-stratified",
        help=(
            "synthetic routes through six experts, rows naturally selected by "
            "each of those experts, independent events from exact natural "
            "top-k routes, or natural target rows plus deterministic RMS-"
            "isotropic dSpark rows (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--activation-corpus",
        type=Path,
        help="checkpoint-bound held-out activations to use instead of simulated inputs",
    )
    parser.add_argument(
        "--calibration-jsonl",
        type=Path,
        help=(
            "production calibration JSONL used to prove the held-out activation "
            "prompts are source-disjoint"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--expected-gpu-uuid",
        help=(
            "physical GPU UUID; for a GPTQModel publication this must equal "
            "coordinator GPU0 in the immutable quantization preflight"
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="validate a completed layer in a resumable, unpublished conversion",
    )
    parser.add_argument(
        "--min-quant-cosine",
        type=float,
        default=MIN_QUANT_COSINE,
        help="minimum native-to-EXL3 output cosine (default: %(default)s)",
    )
    parser.add_argument(
        "--max-quant-relative-l2",
        type=float,
        default=MAX_QUANT_RELATIVE_L2,
        help="maximum native-to-EXL3 output relative L2 (default: %(default)s)",
    )
    parser.add_argument(
        "--min-dspark-quant-cosine",
        type=float,
        default=MIN_DSPARK_QUANT_COSINE,
        help=(
            "minimum native-to-EXL3 dSpark/MTP synthetic-proxy cosine "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--max-dspark-quant-relative-l2",
        type=float,
        default=MAX_DSPARK_QUANT_RELATIVE_L2,
        help=(
            "maximum native-to-EXL3 dSpark/MTP synthetic-proxy relative L2 "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument("--min-tp-cosine", type=float, default=0.999)
    parser.add_argument("--max-tp-relative-l2", type=float, default=0.03)
    parser.add_argument(
        "--output",
        type=Path,
        help="also write the complete JSON quality report to this path",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an all-layer output from its contract-bound .incomplete journal",
    )
    parser.add_argument(
        "--development-diagnostic",
        action="store_true",
        help=(
            "allow an explicitly non-production all-layer synthetic diagnostic "
            "without held-out activation evidence"
        ),
    )
    return parser.parse_args()


def stratified_expert_ids(expert_count: int) -> tuple[int, ...]:
    if expert_count < 6:
        raise ValueError("quality validation requires at least six routed experts")
    return tuple(index * (expert_count - 1) // 5 for index in range(6))


def parse_expert_ids(raw: str, expert_count: int | None = None) -> tuple[int, ...]:
    if raw.strip().lower() == "auto":
        if expert_count is None:
            raise ValueError("automatic expert selection requires checkpoint geometry")
        return stratified_expert_ids(expert_count)
    try:
        values = tuple(sorted(int(value.strip()) for value in raw.split(",")))
    except ValueError as error:
        raise ValueError("--expert-ids must be a comma-separated integer list") from error
    if len(values) != 6 or len(set(values)) != len(values) or values[0] < 0:
        raise ValueError("quality validation requires exactly six unique nonnegative experts")
    return values


def layer_sampling_mode(
    requested: str, *, layer_id: int, captured_layer_count: int
) -> str:
    """Resolve the explicit target/dSpark hybrid qualification policy."""

    if requested != NATURAL_TARGET_SYNTHETIC_MTP:
        return requested
    return (
        "natural-stratified"
        if layer_id < captured_layer_count
        else "synthetic-stratified"
    )


def expert_geometry(config: Any) -> tuple[Any, ...]:
    """Return serving geometry without conflating native and EXL3 layouts."""

    return tuple(
        getattr(config, field)
        for field in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "dspark_blocks",
            "global_experts",
            "top_k",
            "swiglu_limit",
        )
    )


def _bound_quality_gpu(value: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return {**value, "sha256": hashlib.sha256(encoded).hexdigest()}


def quality_gpu_identity(
    snapshot: Path,
    *,
    gptqmodel_native: bool,
    requested_uuid: str | None,
) -> dict[str, Any] | None:
    """Resolve the one validation GPU to immutable physical coordinator GPU0."""

    planned_gpu = None
    if gptqmodel_native:
        plan_path = snapshot.expanduser().resolve(strict=True) / GPTQMODEL_PLAN_FILE
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        preflight = plan.get("preflight")
        gpus = preflight.get("gpus") if isinstance(preflight, dict) else None
        candidates = (
            [gpu for gpu in gpus if isinstance(gpu, dict) and gpu.get("index") == 0]
            if isinstance(gpus, list)
            else []
        )
        if len(candidates) != 1:
            raise ValueError(
                "GPTQModel quantization preflight does not identify physical GPU0"
            )
        planned_gpu = candidates[0]
        planned_uuid = planned_gpu.get("uuid")
        if (
            not isinstance(planned_uuid, str)
            or GPU_UUID_RE.fullmatch(planned_uuid) is None
        ):
            raise ValueError(
                "GPTQModel quantization preflight has an invalid GPU0 UUID"
            )
        if requested_uuid is not None and requested_uuid != planned_uuid:
            raise ValueError(
                "--expected-gpu-uuid differs from quantization preflight GPU0"
            )
        expected_uuid = planned_uuid
    else:
        expected_uuid = requested_uuid

    if expected_uuid is None:
        return None
    if GPU_UUID_RE.fullmatch(expected_uuid) is None:
        raise ValueError("--expected-gpu-uuid is not a valid NVIDIA GPU UUID")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != expected_uuid:
        raise RuntimeError(
            "quality validation requires CUDA_VISIBLE_DEVICES to be exactly "
            f"physical GPU UUID {expected_uuid}, got {visible!r}"
        )
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={expected_uuid}",
                "--query-gpu=uuid,pci.bus_id,name,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"cannot resolve physical quality GPU {expected_uuid}: {error}"
        ) from error
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(
            f"nvidia-smi returned {len(rows)} rows for physical GPU {expected_uuid}"
        )
    fields = [field.strip() for field in rows[0].split(",", maxsplit=4)]
    if len(fields) != 5 or fields[0] != expected_uuid:
        raise RuntimeError("nvidia-smi returned a malformed physical GPU identity")
    observed_uuid, pci_bus_id, name, driver_version, compute_capability = fields
    validation_inventory = {
        "uuid": observed_uuid,
        "pci_bus_id": pci_bus_id,
        "name": name,
        "driver_version": driver_version,
        "compute_capability": compute_capability,
    }
    quantization_preflight = None
    if planned_gpu is not None:
        planned_compute = planned_gpu.get("compute_capability")
        planned_name = planned_gpu.get("name")
        planned_driver = planned_gpu.get("driver_version")
        if (
            not isinstance(planned_name, str)
            or not planned_name
            or planned_name != name
            or not isinstance(planned_driver, str)
            or not planned_driver
            or not isinstance(planned_compute, list)
            or len(planned_compute) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in planned_compute
            )
            or f"{planned_compute[0]}.{planned_compute[1]}" != compute_capability
        ):
            raise RuntimeError(
                "physical quality GPU differs from quantization preflight GPU0"
            )
        quantization_preflight = {
            "index": 0,
            "uuid": expected_uuid,
            "name": planned_name,
            "driver_version": planned_driver,
            "compute_capability": planned_compute,
        }
    return _bound_quality_gpu(
        {
            "schema": QUALITY_GPU_SCHEMA,
            "physical_role": (
                "coordinator-gpu0" if gptqmodel_native else "explicit-gpu"
            ),
            "cuda_visible_devices": visible,
            "quantization_preflight": quantization_preflight,
            "validation_inventory": validation_inventory,
        }
    )


def incomplete_report(snapshot: Path) -> dict[str, Any] | None:
    path = snapshot / "ds4rt-exl3-calibration.json.incomplete"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def gptqmodel_calibration_report(
    validated_snapshot: ValidatedExl3ExpertSnapshot,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    """Expose validated GPTQModel ledger evidence to the quality sampler."""

    snapshot = validated_snapshot.path
    family = validate_gptqmodel_native_exl3(
        model_config.get("quantization_config"),
        model_config=model_config,
    )
    projection_records = []
    ledger_path = snapshot / GPTQMODEL_LEDGER_FILE
    with ledger_path.open(encoding="utf-8") as ledger:
        for line_number, line in enumerate(ledger, start=1):
            if not line.strip():
                raise ValueError(
                    f"GPTQModel EXL3 ledger has an empty record at line {line_number}"
                )
            record = json.loads(line)
            if record.get("record_kind") == "projection":
                projection_records.append(
                    {
                        "block_namespace": record["block_namespace"],
                        "logical_layer": record["logical_layer"],
                        "expert": record["expert"],
                        "projection": record["projection"],
                        "sample_count": record["sample_count"],
                        "hessian_weighted_relative_error": record[
                            "quantizer_metrics"
                        ]["hessian_weighted_relative_error"],
                    }
                )
    expected = (
        int(model_config["num_hidden_layers"])
        + len(model_config["dspark_target_layer_ids"])
    ) * int(model_config["n_routed_experts"]) * 3
    if len(projection_records) != expected:
        raise ValueError(
            "GPTQModel EXL3 quality evidence does not cover every routed projection: "
            f"got {len(projection_records)}, expected {expected}"
        )
    return {
        "format": GPTQMODEL_CALIBRATION_FORMAT,
        "family_join": family,
        "projection_records": projection_records,
    }


@lru_cache(maxsize=2)
def cached_validated_exl3_snapshot(snapshot: Path) -> ValidatedExl3ExpertSnapshot:
    """Retain one fail-closed proof for an immutable quality-run snapshot."""

    return validate_exl3_expert_snapshot(snapshot.resolve())


@lru_cache(maxsize=2)
def cached_gptqmodel_calibration_report(snapshot: Path) -> dict[str, Any]:
    """Parse one already-validated publication once per quality process."""

    validated = cached_validated_exl3_snapshot(snapshot)
    model_config = json.loads(
        (validated.path / "config.json").read_text(encoding="utf-8")
    )
    return gptqmodel_calibration_report(validated, model_config)


@contextmanager
def readable_exl3_snapshot(
    snapshot: Path,
    *,
    layer_id: int,
    allow_incomplete: bool,
) -> Iterator[tuple[Path, dict[str, Any]]]:
    snapshot = snapshot.resolve()
    final_report_path = snapshot / "ds4rt-exl3-calibration.json"
    config_path = snapshot / "config.json"
    if config_path.is_file():
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
        if is_gptqmodel_native_exl3(model_config.get("quantization_config")):
            yield snapshot, cached_gptqmodel_calibration_report(snapshot)
            return
        if final_report_path.is_file():
            yield snapshot, json.loads(final_report_path.read_text(encoding="utf-8"))
            return
    if not allow_incomplete:
        raise ValueError(
            "EXL3 snapshot is unpublished; pass --allow-incomplete for a completed resumable layer"
        )
    report = incomplete_report(snapshot)
    if report is None:
        raise ValueError("incomplete EXL3 snapshot has no calibration checkpoint report")
    completed_layers = {int(layer["layer_id"]) for layer in report.get("layers", [])}
    if layer_id not in completed_layers:
        raise ValueError(
            f"incomplete EXL3 snapshot has not checkpointed requested layer {layer_id}"
        )
    config = snapshot / "config.json.incomplete"
    index = snapshot / "model.safetensors.index.json.incomplete"
    if not config.is_file() or not index.is_file():
        raise ValueError("incomplete EXL3 snapshot is missing staged Hugging Face metadata")
    with tempfile.TemporaryDirectory(prefix="ds4rt-exl3-validation-") as temporary:
        view = Path(temporary)
        (view / "config.json").symlink_to(config)
        (view / "model.safetensors.index.json").symlink_to(index)
        for shard in snapshot.glob("*.safetensors"):
            (view / shard.name).symlink_to(shard)
        yield view, report


def output_metrics(actual: Any, expected: Any) -> dict[str, float]:
    import torch

    actual = actual.float()
    expected = expected.float()
    difference = actual - expected
    expected_norm = torch.linalg.vector_norm(expected).clamp_min(1.0e-12)
    row_cosines = torch.nn.functional.cosine_similarity(actual, expected, dim=1)
    return {
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                actual.reshape(1, -1), expected.reshape(1, -1)
            ).item()
        ),
        "min_row_cosine": float(row_cosines.min().item()),
        "relative_l2": float(torch.linalg.vector_norm(difference) / expected_norm),
        "rmse": float(difference.square().mean().sqrt().item()),
        "max_abs": float(difference.abs().max().item()),
        "reference_rms": float(expected.square().mean().sqrt().item()),
    }


def load_native_router_tensors(
    snapshot: Path,
    layer_id: int,
    *,
    device: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load one retained-native learned router without materializing the model."""

    import torch
    from safetensors import safe_open

    snapshot = snapshot.resolve()
    raw = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    target_layers = int(raw["num_hidden_layers"])
    dspark_blocks = len(raw["dspark_target_layer_ids"])
    if not 0 <= layer_id < target_layers + dspark_blocks:
        raise ValueError(f"router layer {layer_id} is outside the checkpoint")
    if (
        raw.get("topk_method") != "noaux_tc"
        or raw.get("scoring_func") != "sqrtsoftplus"
        or raw.get("norm_topk_prob") is not True
    ):
        raise ValueError("natural quality sampling requires the learned V4 router contract")
    prefix = (
        f"layers.{layer_id}.ffn.gate"
        if layer_id < target_layers
        else f"mtp.{layer_id - target_layers}.ffn.gate"
    )
    weight_name = f"{prefix}.weight"
    bias_name = f"{prefix}.bias"
    index = json.loads(
        (snapshot / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]

    def read(name: str) -> Any:
        try:
            shard_path = snapshot / index[name]
        except KeyError as error:
            raise ValueError(f"checkpoint index is missing router tensor {name}") from error
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            return shard.get_tensor(name)

    weight = read(weight_name)
    bias = read(bias_name)
    experts = int(raw["n_routed_experts"])
    hidden = int(raw["hidden_size"])
    if tuple(weight.shape) != (experts, hidden) or weight.dtype != torch.bfloat16:
        raise ValueError(
            f"router weight {weight_name} has {tuple(weight.shape)}/{weight.dtype}, "
            f"expected {(experts, hidden)}/torch.bfloat16"
        )
    if tuple(bias.shape) != (experts,) or bias.dtype != torch.float32:
        raise ValueError(
            f"router bias {bias_name} has {tuple(bias.shape)}/{bias.dtype}, "
            f"expected {(experts,)}/torch.float32"
        )
    return weight.to(device), bias.to(device), {
        "weight": weight_name,
        "bias": bias_name,
        "math": "fp32-linear-sqrtsoftplus-noaux-tc-v1",
    }


def natural_stratified_expert_inputs(
    captured: Any,
    *,
    router_weight: Any,
    router_bias: Any,
    expert_ids: tuple[int, ...],
    rows_per_expert: int,
    layer_id: int,
    seed: int,
    top_k: int,
    routed_scaling_factor: float,
    device: Any,
) -> tuple[Any, Any, Any, dict[str, Any], tuple[tuple[int, int, int], ...]]:
    """Select held-out rows naturally routed to each sampled expert.

    Each execution row retains only the naturally selected expert's actual gate
    weight. The other sampled experts have zero weight. This isolates local
    quantization error without sending an expert arbitrary out-of-distribution
    residuals or allowing errors from six experts to cancel one another.
    """

    import torch
    import torch.nn.functional as functional

    if captured.ndim != 2 or captured.shape[0] <= 0:
        raise ValueError("natural quality sampling requires captured activation rows")
    if len(expert_ids) != top_k:
        raise ValueError("natural stratified sampling requires one sampled expert per top-k slot")
    if rows_per_expert <= 0:
        raise ValueError("natural stratified rows per expert must be positive")
    if router_weight.device != device or router_bias.device != device:
        raise ValueError("natural router tensors must already reside on the quality GPU")

    weight_f32 = router_weight.float()
    bias_f32 = router_bias.float()
    route_id_chunks = []
    route_weight_chunks = []
    with torch.inference_mode():
        for start in range(0, int(captured.shape[0]), 2048):
            hidden_chunk = captured[start : start + 2048].to(device=device)
            logits = functional.linear(hidden_chunk.float(), weight_f32)
            scores = functional.softplus(logits).sqrt()
            route_ids = torch.topk(
                scores + bias_f32,
                top_k,
                dim=-1,
                sorted=False,
            ).indices
            route_weights = scores.gather(1, route_ids)
            route_weights /= route_weights.sum(dim=1, keepdim=True) + 1.0e-20
            route_weights *= routed_scaling_factor
            route_id_chunks.append(route_ids.cpu())
            route_weight_chunks.append(route_weights.cpu())
            del hidden_chunk, logits, scores, route_ids, route_weights
    natural_ids = torch.cat(route_id_chunks, dim=0)
    natural_weights = torch.cat(route_weight_chunks, dim=0)

    selected_rows = []
    selected_weights = []
    candidate_counts: dict[str, int] = {}
    slices = []
    offset = 0
    for slot, expert_id in enumerate(expert_ids):
        matches = natural_ids == expert_id
        candidates = torch.nonzero(matches.any(dim=1), as_tuple=False).flatten()
        candidate_count = int(candidates.numel())
        candidate_counts[str(expert_id)] = candidate_count
        if candidate_count < rows_per_expert:
            raise ValueError(
                f"held-out layer {layer_id} naturally routed only {candidate_count} "
                f"rows to expert {expert_id}; need {rows_per_expert}"
            )
        candidate_offsets = deterministic_expert_row_indices(
            total_rows=candidate_count,
            rows=rows_per_expert,
            layer_id=layer_id,
            expert_id=expert_id,
            seed=seed,
        )
        rows = candidates.index_select(0, candidate_offsets)
        selected_rows.append(rows)
        selected_route_columns = torch.argmax(
            matches.index_select(0, rows).to(torch.int64), dim=1
        )
        selected_weights.append(
            natural_weights.index_select(0, rows).gather(
                1, selected_route_columns.unsqueeze(1)
            ).squeeze(1)
        )
        slices.append((expert_id, offset, offset + rows_per_expert))
        offset += rows_per_expert

    row_indices = torch.cat(selected_rows)
    gate_weights = torch.cat(selected_weights)
    hidden = captured.index_select(0, row_indices).to(device=device)
    topk_ids = torch.tensor(
        expert_ids,
        dtype=torch.int32,
        device=device,
    ).repeat(hidden.shape[0], 1)
    topk_weights = torch.zeros(
        (hidden.shape[0], top_k), dtype=torch.float32, device=device
    )
    for slot, (_, start, stop) in enumerate(slices):
        topk_weights[start:stop, slot] = gate_weights[start:stop].to(device=device)

    evidence = {
        "input_mode": "checkpoint_bound_natural_selected_expert_isolated",
        "captured_rows": int(captured.shape[0]),
        "rows_per_expert": rows_per_expert,
        "execution_rows": int(hidden.shape[0]),
        "natural_candidate_rows": candidate_counts,
        "selected_row_indices_sha256": hashlib.sha256(
            row_indices.numpy().tobytes()
        ).hexdigest(),
        "selected_gate_weights_sha256": hashlib.sha256(
            gate_weights.numpy().tobytes()
        ).hexdigest(),
    }
    return hidden, topk_ids, topk_weights, evidence, tuple(slices)


def captured_stratified_expert_inputs(
    routed: Any,
    *,
    expert_ids: tuple[int, ...],
    rows_per_expert: int,
    layer_id: int,
    seed: int,
    top_k: int,
    device: Any,
) -> tuple[Any, Any, Any, dict[str, Any], tuple[tuple[int, int, int], ...]]:
    """Select isolated expert rows from exact native capture sidecars.

    DeepSeek V4's first ``num_hash_layers`` blocks route by token ID and do
    not have a learned correction bias.  Exact route sidecars are therefore
    the common ground truth for both those hash routers and later learned
    routers, without replaying either router during expert quality testing.
    """

    import torch

    captured = routed.samples
    natural_ids = routed.expert_ids
    natural_weights = routed.gate_weights
    if (
        captured.ndim != 2
        or natural_ids.ndim != 2
        or natural_ids.shape != natural_weights.shape
        or int(natural_ids.shape[0]) != int(captured.shape[0])
        or int(natural_ids.shape[1]) != top_k
    ):
        raise ValueError("captured natural routes do not match activation geometry")
    if len(expert_ids) != top_k:
        raise ValueError("natural stratified sampling requires one sampled expert per top-k slot")
    if rows_per_expert <= 0:
        raise ValueError("natural stratified rows per expert must be positive")

    selected_rows = []
    selected_weights = []
    candidate_counts: dict[str, int] = {}
    slices = []
    offset = 0
    for expert_id in expert_ids:
        matches = natural_ids == expert_id
        candidates = torch.nonzero(matches.any(dim=1), as_tuple=False).flatten()
        candidate_count = int(candidates.numel())
        candidate_counts[str(expert_id)] = candidate_count
        if candidate_count < rows_per_expert:
            raise ValueError(
                f"held-out layer {layer_id} naturally routed only {candidate_count} "
                f"rows to expert {expert_id}; need {rows_per_expert}"
            )
        candidate_offsets = deterministic_expert_row_indices(
            total_rows=candidate_count,
            rows=rows_per_expert,
            layer_id=layer_id,
            expert_id=expert_id,
            seed=seed,
        )
        rows = candidates.index_select(0, candidate_offsets)
        selected_rows.append(rows)
        selected_route_columns = torch.argmax(
            matches.index_select(0, rows).to(torch.int64), dim=1
        )
        selected_weights.append(
            natural_weights.index_select(0, rows).gather(
                1, selected_route_columns.unsqueeze(1)
            ).squeeze(1)
        )
        slices.append((expert_id, offset, offset + rows_per_expert))
        offset += rows_per_expert

    row_indices = torch.cat(selected_rows)
    gate_weights = torch.cat(selected_weights)
    hidden = captured.index_select(0, row_indices).to(device=device)
    topk_ids = torch.tensor(expert_ids, dtype=torch.int32, device=device).repeat(
        hidden.shape[0], 1
    )
    topk_weights = torch.zeros(
        (hidden.shape[0], top_k), dtype=torch.float32, device=device
    )
    for slot, (_, start, stop) in enumerate(slices):
        topk_weights[start:stop, slot] = gate_weights[start:stop].to(device=device)

    evidence = {
        "input_mode": "checkpoint_bound_captured_natural_selected_expert_isolated",
        "route_source": "native_capture_sidecar",
        "captured_rows": int(captured.shape[0]),
        "rows_per_expert": rows_per_expert,
        "execution_rows": int(hidden.shape[0]),
        "natural_candidate_rows": candidate_counts,
        "selected_row_indices_sha256": hashlib.sha256(
            row_indices.numpy().tobytes()
        ).hexdigest(),
        "selected_gate_weights_sha256": hashlib.sha256(
            gate_weights.numpy().tobytes()
        ).hexdigest(),
    }
    return hidden, topk_ids, topk_weights, evidence, tuple(slices)


def natural_route_event_inputs(
    captured: Any,
    *,
    router_weight: Any,
    router_bias: Any,
    rows: int,
    layer_id: int,
    seed: int,
    top_k: int,
    routed_scaling_factor: float,
    device: Any,
) -> tuple[Any, Any, Any, tuple[int, ...], dict[str, Any]]:
    """Expand exact natural top-k rows into independently measured route events."""

    import torch
    import torch.nn.functional as functional

    if captured.ndim != 2 or not 0 < rows <= int(captured.shape[0]):
        raise ValueError("natural route-event rows must fit the held-out capture")
    if top_k <= 0 or top_k > int(router_weight.shape[0]):
        raise ValueError("natural route-event top-k is outside the router geometry")
    if router_weight.device != device or router_bias.device != device:
        raise ValueError("natural router tensors must already reside on the quality GPU")
    if tuple(router_weight.shape) != (int(router_bias.numel()), int(captured.shape[1])):
        raise ValueError("natural router tensors do not match the activation geometry")
    indices = deterministic_expert_row_indices(
        total_rows=int(captured.shape[0]),
        rows=rows,
        layer_id=layer_id,
        expert_id=0,
        seed=seed,
    )
    selected = captured.index_select(0, indices).to(device=device)
    with torch.inference_mode():
        logits = functional.linear(selected.float(), router_weight.float())
        scores = functional.softplus(logits).sqrt()
        natural_ids = torch.topk(
            scores + router_bias.float(),
            top_k,
            dim=-1,
            sorted=False,
        ).indices
        natural_weights = scores.gather(1, natural_ids)
        natural_weights /= natural_weights.sum(dim=1, keepdim=True) + 1.0e-20
        natural_weights *= routed_scaling_factor

    hidden = selected.repeat_interleave(top_k, dim=0)
    topk_ids = natural_ids.to(torch.int32).repeat_interleave(top_k, dim=0)
    topk_weights = torch.zeros(
        (rows * top_k, top_k), dtype=torch.float32, device=device
    )
    event_slots = torch.arange(top_k, device=device).repeat(rows)
    event_rows = torch.arange(rows, device=device).repeat_interleave(top_k)
    topk_weights[event_rows * top_k + event_slots, event_slots] = natural_weights[
        event_rows, event_slots
    ]
    execution_experts = tuple(
        int(value) for value in torch.unique(natural_ids.cpu(), sorted=True).tolist()
    )
    route_counts = torch.bincount(
        natural_ids.reshape(-1).cpu(), minlength=int(router_weight.shape[0])
    )
    evidence = {
        "input_mode": "checkpoint_bound_exact_natural_route_events",
        "captured_rows": int(captured.shape[0]),
        "selected_rows": rows,
        "execution_rows": rows * top_k,
        "unique_routed_experts": len(execution_experts),
        "selected_row_indices_sha256": hashlib.sha256(
            indices.numpy().tobytes()
        ).hexdigest(),
        "natural_route_ids_sha256": hashlib.sha256(
            natural_ids.cpu().numpy().tobytes()
        ).hexdigest(),
        "natural_route_weights_sha256": hashlib.sha256(
            natural_weights.cpu().numpy().tobytes()
        ).hexdigest(),
        "natural_route_counts": {
            str(expert_id): int(route_counts[expert_id].item())
            for expert_id in execution_experts
        },
    }
    return hidden, topk_ids, topk_weights, execution_experts, evidence


def captured_route_event_inputs(
    routed: Any,
    *,
    rows: int,
    layer_id: int,
    seed: int,
    top_k: int,
    routed_experts: int,
    device: Any,
) -> tuple[Any, Any, Any, tuple[int, ...], dict[str, Any]]:
    """Expand captured native top-k rows into isolated route events."""

    import torch

    captured = routed.samples
    captured_ids = routed.expert_ids
    captured_weights = routed.gate_weights
    if (
        captured.ndim != 2
        or captured_ids.ndim != 2
        or captured_ids.shape != captured_weights.shape
        or int(captured_ids.shape[0]) != int(captured.shape[0])
        or int(captured_ids.shape[1]) != top_k
        or not 0 < rows <= int(captured.shape[0])
    ):
        raise ValueError("captured route-event rows do not match activation geometry")
    indices = deterministic_expert_row_indices(
        total_rows=int(captured.shape[0]),
        rows=rows,
        layer_id=layer_id,
        expert_id=0,
        seed=seed,
    )
    selected = captured.index_select(0, indices).to(device=device)
    natural_ids = captured_ids.index_select(0, indices).to(device=device)
    natural_weights = captured_weights.index_select(0, indices).to(device=device)
    hidden = selected.repeat_interleave(top_k, dim=0)
    topk_ids = natural_ids.to(torch.int32).repeat_interleave(top_k, dim=0)
    topk_weights = torch.zeros(
        (rows * top_k, top_k), dtype=torch.float32, device=device
    )
    event_slots = torch.arange(top_k, device=device).repeat(rows)
    event_rows = torch.arange(rows, device=device).repeat_interleave(top_k)
    topk_weights[event_rows * top_k + event_slots, event_slots] = natural_weights[
        event_rows, event_slots
    ]
    execution_experts = tuple(
        int(value) for value in torch.unique(natural_ids.cpu(), sorted=True).tolist()
    )
    route_counts = torch.bincount(
        natural_ids.reshape(-1).cpu(), minlength=routed_experts
    )
    evidence = {
        "input_mode": "checkpoint_bound_captured_exact_natural_route_events",
        "route_source": "native_capture_sidecar",
        "captured_rows": int(captured.shape[0]),
        "selected_rows": rows,
        "execution_rows": rows * top_k,
        "unique_routed_experts": len(execution_experts),
        "selected_row_indices_sha256": hashlib.sha256(
            indices.numpy().tobytes()
        ).hexdigest(),
        "natural_route_ids_sha256": hashlib.sha256(
            natural_ids.cpu().numpy().tobytes()
        ).hexdigest(),
        "natural_route_weights_sha256": hashlib.sha256(
            natural_weights.cpu().numpy().tobytes()
        ).hexdigest(),
        "natural_route_counts": {
            str(expert): int(route_counts[expert].item())
            for expert in execution_experts
        },
    }
    return hidden, topk_ids, topk_weights, execution_experts, evidence


def proxy_summary(
    report: dict[str, Any],
    layer_id: int,
    num_hidden_layers: int,
    expert_ids: tuple[int, ...],
) -> dict[str, float | int]:
    if report.get("format") == GPTQMODEL_CALIBRATION_FORMAT:
        if layer_id < num_hidden_layers:
            namespace = "base"
            logical_layer = layer_id
        else:
            namespace = "mtp"
            logical_layer = layer_id - num_hidden_layers
        selected = [
            record
            for record in report.get("projection_records", [])
            if record.get("block_namespace") == namespace
            and record.get("logical_layer") == logical_layer
            and record.get("expert") in expert_ids
        ]
        values = sorted(
            float(record["hessian_weighted_relative_error"])
            for record in selected
        )
    else:
        layer = next(
            (layer for layer in report.get("layers", []) if int(layer["layer_id"]) == layer_id),
            None,
        )
        if layer is None:
            raise ValueError(f"calibration report has no layer {layer_id}")
        block = (
            f"layers.{layer_id}"
            if layer_id < num_hidden_layers
            else f"mtp.{layer_id - num_hidden_layers}"
        )
        prefixes = tuple(f"{block}.ffn.experts.{expert_id}." for expert_id in expert_ids)
        values = sorted(
            float(projection["proxy_error"])
            for projection in layer.get("projections", [])
            if str(projection.get("name", "")).startswith(prefixes)
        )
    expected = len(expert_ids) * 3
    if len(values) != expected:
        raise ValueError(
            f"layer {layer_id} report has {len(values)} selected proxy errors, expected {expected}"
        )

    def percentile(fraction: float) -> float:
        index = min(len(values) - 1, math.ceil(fraction * len(values)) - 1)
        return values[index]

    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": values[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": values[-1],
    }


def calibration_summary(
    report: dict[str, Any],
    layer_id: int,
    num_hidden_layers: int,
    expert_ids: tuple[int, ...],
) -> dict[str, Any]:
    selected_proxy_error = proxy_summary(
        report,
        layer_id,
        num_hidden_layers,
        expert_ids,
    )
    if report.get("format") != GPTQMODEL_CALIBRATION_FORMAT:
        return {
            "rows": report["calibration_rows"],
            "seed": report["seed"],
            "activation": report["activation"],
            "swiglu_limit": report["swiglu_limit"],
            "selected_proxy_error": selected_proxy_error,
        }

    if layer_id < num_hidden_layers:
        namespace = "base"
        logical_layer = layer_id
    else:
        namespace = "mtp"
        logical_layer = layer_id - num_hidden_layers
    selected = [
        record
        for record in report["projection_records"]
        if record.get("block_namespace") == namespace
        and record.get("logical_layer") == logical_layer
        and record.get("expert") in expert_ids
    ]
    route_rows = [int(record["sample_count"]) for record in selected]
    family = report["family_join"]
    return {
        "format": GPTQMODEL_CALIBRATION_FORMAT,
        "corpus_examples": int(family["corpus"]["examples"]),
        "corpus_normalized_stream_sha256": family["corpus"][
            "normalized_stream_sha256"
        ],
        "quantizer_seed": int(family["quantizer_seed"]),
        "hessian_numerical": family["quantizer_numerics"]["hessian_numerical"],
        "selected_route_rows": {
            "min": min(route_rows),
            "max": max(route_rows),
            "mean": sum(route_rows) / len(route_rows),
        },
        "selected_proxy_error": selected_proxy_error,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if args.rows <= 0:
        raise ValueError("--rows must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "validation requires exactly one CUDA_VISIBLE_DEVICES GPU"
        )
    native_snapshot = args.native_snapshot.resolve()
    native_config = read_native_expert_config(native_snapshot)
    requested_expert_ids = parse_expert_ids(
        args.expert_ids, native_config.global_experts
    )
    expert_ids = requested_expert_ids
    with readable_exl3_snapshot(
        args.exl3_snapshot,
        layer_id=args.layer_id,
        allow_incomplete=args.allow_incomplete,
    ) as (exl3_snapshot, report):
        model_config = json.loads(
            (exl3_snapshot / "config.json").read_text(encoding="utf-8")
        )
        execution_gpu = quality_gpu_identity(
            args.exl3_snapshot,
            gptqmodel_native=is_gptqmodel_native_exl3(
                model_config.get("quantization_config")
            ),
            requested_uuid=getattr(args, "expected_gpu_uuid", None),
        )
        # The canonical publication includes a multi-gigabyte error ledger.
        # Audit that immutable snapshot once, then pass the resulting proof to
        # all five loaders rather than rehashing and reparsing it for every
        # unsharded/TP view of this layer.
        validated_exl3 = cached_validated_exl3_snapshot(exl3_snapshot)
        exl3_config = validated_exl3.config
        if expert_geometry(native_config) != expert_geometry(exl3_config):
            raise ValueError("native and EXL3 expert geometry/configuration differ")
        if requested_expert_ids[-1] >= native_config.global_experts:
            raise ValueError("selected expert is outside the checkpoint")

        device = torch.device("cuda:0")
        generator = torch.Generator(device=device).manual_seed(args.seed + args.layer_id)
        requested_sampling_mode = getattr(
            args, "sampling_mode", "synthetic-stratified"
        )
        routed_scale = float(
            json.loads((native_snapshot / "config.json").read_text(encoding="utf-8"))[
                "routed_scaling_factor"
            ]
        )
        activation_evidence = None
        expert_row_slices: tuple[tuple[int, int, int], ...] = ()
        natural_route_groups = 0
        natural_event_expert_ids = None
        topk_ids = None
        topk_weights = None
        corpus = None
        if args.activation_corpus is not None:
            _, model_shape = read_native_model_config(native_snapshot)
            corpus = load_activation_corpus(
                args.activation_corpus,
                snapshot=native_snapshot,
                shape=model_shape,
                # A held-out quality split is not a calibration Hessian. Its
                # declared global floor may be zero; the natural-stratified
                # selector below independently requires --rows for every one
                # of the six validation experts in each target layer.
                required_natural_routes_per_expert=0,
                # This validator recomputes the learned router over held-out
                # hidden rows; it does not use captured route sidecars as
                # calibration Hessian evidence.
                require_exact_route_replay=False,
            )
        if requested_sampling_mode == NATURAL_TARGET_SYNTHETIC_MTP:
            if corpus is None:
                raise ValueError(
                    f"{NATURAL_TARGET_SYNTHETIC_MTP} requires an activation corpus"
                )
            if corpus.layer_count != native_config.num_hidden_layers:
                raise ValueError(
                    "hybrid quality sampling requires exactly one captured input "
                    "frontier for every target layer"
                )
        sampling_mode = layer_sampling_mode(
            requested_sampling_mode,
            layer_id=args.layer_id,
            captured_layer_count=corpus.layer_count if corpus is not None else 0,
        )
        if corpus is not None and args.layer_id < corpus.layer_count:
            if sampling_mode in ("natural-stratified", "natural-route-events"):
                routed = load_routed_activation_layer(corpus, args.layer_id)
                captured = routed.samples
                if sampling_mode == "natural-stratified":
                    (
                        hidden,
                        topk_ids,
                        topk_weights,
                        natural_evidence,
                        expert_row_slices,
                    ) = captured_stratified_expert_inputs(
                        routed,
                        expert_ids=expert_ids,
                        rows_per_expert=args.rows,
                        layer_id=args.layer_id,
                        seed=args.seed,
                        top_k=native_config.top_k,
                        device=device,
                    )
                else:
                    (
                        hidden,
                        topk_ids,
                        topk_weights,
                        expert_ids,
                        natural_evidence,
                    ) = captured_route_event_inputs(
                        routed,
                        rows=args.rows,
                        layer_id=args.layer_id,
                        seed=args.seed,
                        top_k=native_config.top_k,
                        routed_experts=native_config.global_experts,
                        device=device,
                    )
                    natural_route_groups = args.rows
                    event_slots = torch.arange(
                        native_config.top_k, device=device
                    ).repeat(args.rows)
                    natural_event_expert_ids = topk_ids[
                        torch.arange(args.rows * native_config.top_k, device=device),
                        event_slots,
                    ]
                activation_evidence = {
                    "manifest": str(corpus.manifest_path),
                    "corpus_sha256": corpus.corpus_sha256,
                    **natural_evidence,
                }
                del routed
            else:
                captured = load_activation_layer_samples(corpus, args.layer_id)
                if args.rows > captured.shape[0]:
                    raise ValueError(
                        f"--rows={args.rows} exceeds {captured.shape[0]} held-out rows"
                    )
                indices = deterministic_expert_row_indices(
                    total_rows=int(captured.shape[0]),
                    rows=args.rows,
                    layer_id=args.layer_id,
                    expert_id=0,
                    seed=args.seed,
                )
                hidden = captured.index_select(0, indices).to(device=device)
                activation_evidence = {
                    "manifest": str(corpus.manifest_path),
                    "corpus_sha256": corpus.corpus_sha256,
                    "captured_rows": int(captured.shape[0]),
                    "selected_row_indices_sha256": hashlib.sha256(
                        indices.numpy().tobytes()
                    ).hexdigest(),
                    "input_mode": "checkpoint_bound_heldout_activation",
                }
                del indices
            del captured
        else:
            if sampling_mode in ("natural-stratified", "natural-route-events"):
                raise ValueError(
                    f"{sampling_mode} quality sampling has no captured inputs "
                    f"for layer {args.layer_id}"
                )
            hidden_f32 = torch.randn(
                (args.rows, native_config.hidden_size),
                generator=generator,
                dtype=torch.float32,
                device=device,
            )
            hidden_f32 *= torch.rsqrt(
                hidden_f32.square().mean(dim=1, keepdim=True) + 1.0e-6
            )
            hidden = hidden_f32.to(torch.bfloat16)
            if corpus is not None:
                activation_evidence = {
                    "manifest": str(corpus.manifest_path),
                    "corpus_sha256": corpus.corpus_sha256,
                    "input_mode": "uncaptured_mtp_rms_isotropic",
                }
        if topk_ids is None or topk_weights is None:
            topk_ids = torch.stack(
                [
                    torch.tensor(expert_ids, dtype=torch.int32, device=device).roll(
                        row % len(expert_ids)
                    )
                    for row in range(args.rows)
                ]
            )
            topk_weights = torch.rand(
                (args.rows, native_config.top_k),
                generator=generator,
                dtype=torch.float32,
                device=device,
            )
            topk_weights *= routed_scale / topk_weights.sum(dim=1, keepdim=True)

        native_layer = load_native_expert_reference_layer(
            native_snapshot, args.layer_id, expert_ids
        )
        native_output = native_layer.run_partial(hidden, topk_ids, topk_weights).float().clone()
        torch.cuda.synchronize(device)
        del native_layer
        torch.cuda.empty_cache()

        exl3_layer = load_exl3_expert_reference_layer(
            validated_exl3, args.layer_id, expert_ids
        )
        exl3_output = exl3_layer.run_partial(hidden, topk_ids, topk_weights).float().clone()
        torch.cuda.synchronize(device)
        del exl3_layer
        torch.cuda.empty_cache()

        tp_sum = torch.zeros_like(exl3_output)
        rank_source_bytes = []
        for rank in range(4):
            rank_layer = load_exl3_expert_tp_layer(
                validated_exl3,
                args.layer_id,
                tp_rank=rank,
                expert_ids=expert_ids,
            )
            partial = rank_layer.run_partial(hidden, topk_ids, topk_weights)
            torch.cuda.synchronize(device)
            tp_sum.add_(partial.float())
            rank_source_bytes.append(rank_layer.source_bytes)
            del rank_layer, partial
            torch.cuda.empty_cache()

        natural_routed_sum = None
        natural_route_event_experts = None
        if natural_route_groups:
            grouped_shape = (
                natural_route_groups,
                native_config.top_k,
                native_config.hidden_size,
            )
            native_routed = native_output.reshape(grouped_shape).sum(dim=1)
            exl3_routed = exl3_output.reshape(grouped_shape).sum(dim=1)
            tp_routed = tp_sum.reshape(grouped_shape).sum(dim=1)
            natural_routed_sum = {
                "rows": natural_route_groups,
                "native_to_exl3": output_metrics(exl3_routed, native_routed),
                "exl3_tp4_to_unsharded": output_metrics(tp_routed, exl3_routed),
            }
            assert natural_event_expert_ids is not None
            natural_route_event_experts = []
            for expert_id in expert_ids:
                event_indices = torch.nonzero(
                    natural_event_expert_ids == expert_id, as_tuple=False
                ).flatten()
                natural_route_event_experts.append(
                    {
                        "expert_id": expert_id,
                        "events": int(event_indices.numel()),
                        "native_to_exl3": output_metrics(
                            exl3_output.index_select(0, event_indices),
                            native_output.index_select(0, event_indices),
                        ),
                        "exl3_tp4_to_unsharded": output_metrics(
                            tp_sum.index_select(0, event_indices),
                            exl3_output.index_select(0, event_indices),
                        ),
                    }
                )

        result = {
            "native_snapshot": str(native_snapshot),
            "exl3_snapshot": str(args.exl3_snapshot.resolve()),
            "layer_id": args.layer_id,
            "expert_ids": list(expert_ids),
            "rows": args.rows,
            "execution_rows": int(hidden.shape[0]),
            "sampling_mode": sampling_mode,
            **(
                {"sampling_policy": requested_sampling_mode}
                if requested_sampling_mode != sampling_mode
                else {}
            ),
            "expert_scope": (
                "exact_natural_route_union"
                if sampling_mode == "natural-route-events"
                else "requested_fixed_experts"
            ),
            "seed": args.seed,
            "routed_scaling_factor": routed_scale,
            **(
                {"heldout_activation_evidence": activation_evidence}
                if activation_evidence is not None
                else {}
            ),
            "calibration": calibration_summary(
                report,
                args.layer_id,
                native_config.num_hidden_layers,
                expert_ids,
            ),
            "native_to_exl3": output_metrics(exl3_output, native_output),
            "exl3_tp4_to_unsharded": output_metrics(tp_sum, exl3_output),
            **(
                {"natural_routed_sum": natural_routed_sum}
                if natural_routed_sum is not None
                else {}
            ),
            **(
                {"natural_route_event_experts": natural_route_event_experts}
                if natural_route_event_experts is not None
                else {}
            ),
            **(
                {
                    "natural_selected_experts": [
                        {
                            "expert_id": expert_id,
                            "rows": stop - start,
                            "native_to_exl3": output_metrics(
                                exl3_output[start:stop], native_output[start:stop]
                            ),
                            "exl3_tp4_to_unsharded": output_metrics(
                                tp_sum[start:stop], exl3_output[start:stop]
                            ),
                        }
                        for expert_id, start, stop in expert_row_slices
                    ]
                }
                if expert_row_slices
                else {}
            ),
            "tp4_rank_source_bytes": rank_source_bytes,
            "tp4_equal_rank_source_bytes": len(set(rank_source_bytes)) == 1,
            **(
                {"execution_gpu_sha256": execution_gpu["sha256"]}
                if execution_gpu is not None
                else {}
            ),
        }
        return result


def enforce_thresholds(result: dict[str, Any], args: argparse.Namespace) -> None:
    quant = result["native_to_exl3"]
    tp = result["exl3_tp4_to_unsharded"]
    if quant["cosine"] < args.min_quant_cosine:
        raise RuntimeError(
            f"native-to-EXL3 cosine {quant['cosine']} is below {args.min_quant_cosine}"
        )
    if quant["relative_l2"] > args.max_quant_relative_l2:
        raise RuntimeError(
            f"native-to-EXL3 relative L2 {quant['relative_l2']} exceeds "
            f"{args.max_quant_relative_l2}"
        )
    if tp["cosine"] < args.min_tp_cosine or tp["relative_l2"] > args.max_tp_relative_l2:
        raise RuntimeError(
            f"EXL3 TP4 parity failed: cosine={tp['cosine']} relative_l2={tp['relative_l2']}"
        )
    if not result["tp4_equal_rank_source_bytes"]:
        raise RuntimeError("EXL3 TP4 ranks do not have equal resident source bytes")


def threshold_args_for_layer(
    args: argparse.Namespace, layer_id: int, target_layer_count: int
) -> argparse.Namespace:
    """Return target-natural or dSpark-synthetic quality gates for one block."""

    effective = copy(args)
    if layer_id >= target_layer_count:
        effective.min_quant_cosine = args.min_dspark_quant_cosine
        effective.max_quant_relative_l2 = args.max_dspark_quant_relative_l2
    return effective


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("quality report requires at least one layer result")
    quant = [result["native_to_exl3"] for result in results]
    tp = [result["exl3_tp4_to_unsharded"] for result in results]
    aggregate = {
        "layers": len(results),
        "native_to_exl3": {
            "min_cosine": min(metric["cosine"] for metric in quant),
            "min_row_cosine": min(metric["min_row_cosine"] for metric in quant),
            "max_relative_l2": max(metric["relative_l2"] for metric in quant),
            "max_rmse": max(metric["rmse"] for metric in quant),
            "max_abs": max(metric["max_abs"] for metric in quant),
        },
        "exl3_tp4_to_unsharded": {
            "min_cosine": min(metric["cosine"] for metric in tp),
            "min_row_cosine": min(metric["min_row_cosine"] for metric in tp),
            "max_relative_l2": max(metric["relative_l2"] for metric in tp),
            "max_rmse": max(metric["rmse"] for metric in tp),
            "max_abs": max(metric["max_abs"] for metric in tp),
        },
        "all_tp4_ranks_equal_source_bytes": all(
            result["tp4_equal_rank_source_bytes"] for result in results
        ),
    }
    if all("natural_routed_sum" in result for result in results):
        routed_quant = [
            result["natural_routed_sum"]["native_to_exl3"] for result in results
        ]
        routed_tp = [
            result["natural_routed_sum"]["exl3_tp4_to_unsharded"]
            for result in results
        ]
        aggregate["natural_routed_sum"] = {
            "native_to_exl3": {
                "min_cosine": min(metric["cosine"] for metric in routed_quant),
                "min_row_cosine": min(
                    metric["min_row_cosine"] for metric in routed_quant
                ),
                "max_relative_l2": max(
                    metric["relative_l2"] for metric in routed_quant
                ),
                "max_rmse": max(metric["rmse"] for metric in routed_quant),
                "max_abs": max(metric["max_abs"] for metric in routed_quant),
            },
            "exl3_tp4_to_unsharded": {
                "min_cosine": min(metric["cosine"] for metric in routed_tp),
                "min_row_cosine": min(
                    metric["min_row_cosine"] for metric in routed_tp
                ),
                "max_relative_l2": max(
                    metric["relative_l2"] for metric in routed_tp
                ),
                "max_rmse": max(metric["rmse"] for metric in routed_tp),
                "max_abs": max(metric["max_abs"] for metric in routed_tp),
            },
        }
    return aggregate


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_identity(snapshot: Path) -> dict[str, Any]:
    snapshot = snapshot.resolve(strict=True)
    metadata = {}
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "quantize_config.json",
        "ds4rt-exl3-calibration.json",
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
        stat = path.stat()
        shards.append(
            {
                "name": path.name,
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    if not shards:
        raise ValueError(f"checkpoint has no safetensor shards: {snapshot}")
    return {"path": str(snapshot), "metadata_sha256": metadata, "shards": shards}


def activation_manifest_identity(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve(strict=True)
    manifest = resolved / "manifest.json" if resolved.is_dir() else resolved
    return {"path": str(manifest), "sha256": hash_file(manifest)}


def calibration_disjointness(args: argparse.Namespace) -> dict[str, Any] | None:
    snapshot = args.exl3_snapshot.expanduser().resolve(strict=True)
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    quant = config.get("quantization_config")
    if not is_gptqmodel_native_exl3(quant):
        return None
    if args.calibration_jsonl is None or args.activation_corpus is None:
        if args.development_diagnostic:
            if args.sampling_mode != "synthetic-stratified":
                raise ValueError(
                    "--development-diagnostic without held-out activations "
                    "requires --sampling-mode synthetic-stratified"
                )
            return None
        raise ValueError(
            "all-layer GPTQModel qualification requires --calibration-jsonl and "
            "--activation-corpus to prove held-out prompt disjointness"
        )
    family = validate_gptqmodel_native_exl3(quant, model_config=config)
    calibration_path = args.calibration_jsonl.expanduser().resolve(strict=True)
    calibration_sha256 = hash_file(calibration_path)
    corpus = family["corpus"]
    if calibration_sha256 != corpus["file_sha256"]:
        raise ValueError(
            "quality calibration JSONL does not match the GPTQModel artifact provenance"
        )
    calibration_hashes = set()
    calibration_prompt_count = 0
    with calibration_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(
                    f"calibration JSONL has an empty record at line {line_number}"
                )
            record = json.loads(line)
            prompt = record.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(
                    f"calibration JSONL has no prompt at line {line_number}"
                )
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            calibration_hashes.add(digest)
            calibration_prompt_count += 1
    if calibration_prompt_count != corpus["examples"]:
        raise ValueError(
            "quality calibration JSONL example count differs from artifact provenance"
        )

    activation_root = args.activation_corpus.expanduser().resolve(strict=True)
    activation_manifest = (
        activation_root / "manifest.json" if activation_root.is_dir() else activation_root
    )
    activation = json.loads(activation_manifest.read_text(encoding="utf-8"))
    prompts = activation.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("held-out activation manifest has no prompt records")
    heldout_hashes = []
    for record in prompts:
        digest = record.get("prompt_sha256") if isinstance(record, dict) else None
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("held-out activation manifest has an invalid prompt hash")
        heldout_hashes.append(digest)
    if len(set(heldout_hashes)) != len(heldout_hashes):
        raise ValueError("held-out activation manifest contains a duplicate prompt")
    overlap = calibration_hashes.intersection(heldout_hashes)
    if overlap:
        raise ValueError(
            f"held-out activation prompts overlap {len(overlap)} calibration prompts"
        )
    return {
        "calibration_jsonl": {
            "path": str(calibration_path),
            "sha256": calibration_sha256,
            "prompts": calibration_prompt_count,
            "unique_prompts": len(calibration_hashes),
        },
        "heldout_activation_manifest": {
            "path": str(activation_manifest),
            "sha256": hash_file(activation_manifest),
            "prompts": len(heldout_hashes),
        },
        "prompt_sha256_overlap": 0,
    }


def validation_contract(
    args: argparse.Namespace, layer_ids: tuple[int, ...]
) -> dict[str, Any]:
    expert_count = read_native_expert_config(args.native_snapshot.resolve()).global_experts
    exl3_snapshot = args.exl3_snapshot.expanduser().resolve(strict=True)
    model_config = json.loads(
        (exl3_snapshot / "config.json").read_text(encoding="utf-8")
    )
    gptqmodel_native = is_gptqmodel_native_exl3(
        model_config.get("quantization_config")
    )
    execution_gpu = quality_gpu_identity(
        exl3_snapshot,
        gptqmodel_native=gptqmodel_native,
        requested_uuid=getattr(args, "expected_gpu_uuid", None),
    )
    if gptqmodel_native and execution_gpu is None:
        raise ValueError(
            "GPTQModel all-layer quality validation has no physical GPU0 identity"
        )
    sampling_mode = getattr(args, "sampling_mode", "synthetic-stratified")
    requested_expert_ids = list(
        parse_expert_ids(args.expert_ids, expert_count)
    )
    payload = {
        "native": snapshot_identity(args.native_snapshot),
        "exl3": snapshot_identity(args.exl3_snapshot),
        "layer_ids": list(layer_ids),
        "expert_ids": (
            None if sampling_mode == "natural-route-events" else requested_expert_ids
        ),
        "expert_scope": (
            "exact_natural_route_union_per_layer"
            if sampling_mode == "natural-route-events"
            else "requested_fixed_experts"
        ),
        "rows": args.rows,
        "sampling_mode": sampling_mode,
        "seed": args.seed,
        "allow_incomplete": bool(args.allow_incomplete),
        "development_diagnostic": bool(args.development_diagnostic),
        "execution_gpu": execution_gpu,
        "heldout_activation_manifest": (
            activation_manifest_identity(args.activation_corpus)
            if args.activation_corpus is not None
            else None
        ),
        "calibration_disjointness": calibration_disjointness(args),
        "thresholds": {
            "min_quant_cosine": args.min_quant_cosine,
            "max_quant_relative_l2": args.max_quant_relative_l2,
            "min_dspark_quant_cosine": args.min_dspark_quant_cosine,
            "max_dspark_quant_relative_l2": args.max_dspark_quant_relative_l2,
            "min_tp_cosine": args.min_tp_cosine,
            "max_tp_relative_l2": args.max_tp_relative_l2,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}


def progress_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.incomplete")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_progress(
    path: Path, contract: dict[str, Any], results: list[dict[str, Any]]
) -> None:
    write_json_atomic(
        path,
        {"schema": PROGRESS_SCHEMA, "contract": contract, "layers": results},
    )


def load_progress(
    path: Path, contract: dict[str, Any], args: argparse.Namespace
) -> list[dict[str, Any]]:
    progress = json.loads(path.read_text(encoding="utf-8"))
    if progress.get("schema") != PROGRESS_SCHEMA:
        raise ValueError(f"quality resume journal does not use {PROGRESS_SCHEMA}")
    progress_contract = progress.get("contract")
    if progress_contract != contract:
        # Journals written before target and dSpark proxy gates were separated
        # contain only completed target layers and the target threshold pair.
        # Accept exactly that one contract upgrade; the next checkpoint write
        # replaces it with the current digest before any dSpark result lands.
        if not isinstance(progress_contract, dict):
            raise ValueError(
                "quality resume journal does not match the active validation contract"
            )
        legacy_contract = copy(contract)
        current_thresholds = legacy_contract.get("thresholds")
        if (
            not isinstance(current_thresholds, dict)
            or "min_dspark_quant_cosine" not in current_thresholds
            or "max_dspark_quant_relative_l2" not in current_thresholds
        ):
            raise ValueError(
                "quality resume journal does not match the active validation contract"
            )
        legacy_thresholds = dict(current_thresholds)
        legacy_thresholds.pop("min_dspark_quant_cosine")
        legacy_thresholds.pop("max_dspark_quant_relative_l2")
        legacy_contract["thresholds"] = legacy_thresholds
        legacy_payload = {
            key: value for key, value in legacy_contract.items() if key != "sha256"
        }
        legacy_contract["sha256"] = hashlib.sha256(
            json.dumps(
                legacy_payload, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        if progress_contract != legacy_contract:
            raise ValueError(
                "quality resume journal does not match the active validation contract"
            )
    results = progress.get("layers")
    if not isinstance(results, list):
        raise ValueError("quality resume journal has no layer result list")
    seen = set()
    allowed = set(contract["layer_ids"])
    native_snapshot = getattr(args, "native_snapshot", None)
    target_layer_count = (
        read_native_expert_config(native_snapshot.resolve()).num_hidden_layers
        if native_snapshot is not None
        else max(allowed) + 1
    )
    for result in results:
        if not isinstance(result, dict) or not isinstance(result.get("layer_id"), int):
            raise ValueError("quality resume journal contains a malformed layer result")
        layer_id = result["layer_id"]
        if layer_id in seen or layer_id not in allowed:
            raise ValueError(f"quality resume journal has invalid layer {layer_id}")
        if layer_id >= target_layer_count and progress_contract != contract:
            raise ValueError(
                "legacy quality journal contains dSpark results without dSpark thresholds"
            )
        enforce_thresholds(
            result, threshold_args_for_layer(args, layer_id, target_layer_count)
        )
        seen.add(layer_id)
    return sorted(results, key=lambda result: result["layer_id"])


def layer_ids_for_args(args: argparse.Namespace) -> tuple[int, ...]:
    if not args.all_layers and not args.target_layers and not args.learned_target_layers:
        assert args.layer_id is not None
        return (int(args.layer_id),)
    config = read_native_expert_config(args.native_snapshot.resolve())
    if args.all_layers:
        return tuple(range(config.total_blocks))
    if args.target_layers:
        return tuple(range(config.num_hidden_layers))
    index = json.loads(
        (args.native_snapshot.resolve() / "model.safetensors.index.json").read_text(
            encoding="utf-8"
        )
    )["weight_map"]
    return tuple(
        layer_id
        for layer_id in range(config.num_hidden_layers)
        if f"layers.{layer_id}.ffn.gate.bias" in index
    )


def main() -> None:
    args = parse_args()
    multi_layer = bool(
        args.all_layers or args.target_layers or args.learned_target_layers
    )
    if args.resume and (not multi_layer or args.output is None):
        raise SystemExit(
            "--resume requires a multi-layer selection and --output"
        )
    layer_ids = layer_ids_for_args(args)
    target_layer_count = read_native_expert_config(
        args.native_snapshot.resolve()
    ).num_hidden_layers
    contract = None
    journal = None
    results: list[dict[str, Any]] = []
    if multi_layer:
        contract = validation_contract(args, layer_ids)
        if args.output is not None:
            journal = progress_path(args.output)
            if journal.exists():
                if not args.resume:
                    raise SystemExit(
                        f"quality resume journal already exists: {journal}; pass --resume"
                    )
                results = load_progress(journal, contract, args)
    completed = {result["layer_id"] for result in results}
    for layer_id in layer_ids:
        if layer_id in completed:
            print(f"resumed validated layer {layer_id}", file=sys.stderr, flush=True)
            continue
        layer_args = copy(args)
        layer_args.layer_id = layer_id
        result = run(layer_args)
        enforce_thresholds(
            result,
            threshold_args_for_layer(args, layer_id, target_layer_count),
        )
        results.append(result)
        results.sort(key=lambda item: item["layer_id"])
        if journal is not None and contract is not None:
            write_progress(journal, contract, results)
        if multi_layer:
            quant = result["native_to_exl3"]
            tp = result["exl3_tp4_to_unsharded"]
            print(
                f"validated layer {layer_id}: quant cosine={quant['cosine']:.6f} "
                f"rel_l2={quant['relative_l2']:.6f}; "
                f"tp4 cosine={tp['cosine']:.8f} rel_l2={tp['relative_l2']:.8f}",
                file=sys.stderr,
                flush=True,
            )
    report: dict[str, Any]
    if len(results) == 1 and not multi_layer:
        report = results[0]
    else:
        assert contract is not None
        if validation_contract(args, layer_ids) != contract:
            raise RuntimeError("EXL3 checkpoint changed during all-layer validation")
        report = {
            "schema": "ds4rt-exl3-checkpoint-quality-v1",
            "native_snapshot": str(args.native_snapshot.resolve()),
            "exl3_snapshot": str(args.exl3_snapshot.resolve()),
            "expert_ids": (
                None
                if args.sampling_mode == "natural-route-events"
                else results[0]["expert_ids"]
            ),
            "expert_scope": (
                "exact_natural_route_union_per_layer"
                if args.sampling_mode == "natural-route-events"
                else "requested_fixed_experts"
            ),
            "rows": args.rows,
            "sampling_mode": args.sampling_mode,
            "seed": args.seed,
            "thresholds": {
                "min_quant_cosine": args.min_quant_cosine,
                "max_quant_relative_l2": args.max_quant_relative_l2,
                "min_dspark_quant_cosine": args.min_dspark_quant_cosine,
                "max_dspark_quant_relative_l2": args.max_dspark_quant_relative_l2,
                "min_tp_cosine": args.min_tp_cosine,
                "max_tp_relative_l2": args.max_tp_relative_l2,
            },
            "validation_contract": contract,
            "summary": aggregate_results(results),
            "layers": results,
        }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        write_json_atomic(args.output, report)
    if journal is not None:
        journal.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
