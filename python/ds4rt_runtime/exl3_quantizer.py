"""Layerwise calibrated EXL3 conversion for DeepSeek V4 routed experts.

The persistent output is a normal Hugging Face safetensors snapshot. Only
routed experts change representation; coordinator and dSpark-envelope tensors
are copied byte-for-byte. Quantization works on bounded expert groups with one
primary CUDA device and optional multi-GPU trellis tile search, and never
materializes a full model or a full output shard in host memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import time
from typing import Any, Callable, Iterable

EXL3_RECIPE = "deepseek_v4_exl3_trellis_2bpw_v2"
EXL3_FORCED_ACTIVATION_RECIPE = (
    "deepseek_v4_exl3_trellis_2bpw_v3_flash_activation_pilot"
)
EXL3_ACTIVATION_RECIPE = (
    "deepseek_v4_exl3_trellis_2bpw_v4_flash_natural_route"
)
EXL3_SCHEMA = "ds4rt.exl3.expert-trellis"
EXL3_SCHEMA_VERSION = 1
EXL3_BITS = 2
EXL3_CODEBOOK = "mcg"
EXL3_TENSOR_FORMAT = "exllamav3_trellis_mcg"
NATIVE_SOURCE_FORMAT = "fp4_e8m0_k32"
EXPERT_TP_WORLD_SIZE = 4
EXPERT_TENSOR_LAYOUT_CHECKPOINT_NATIVE = "checkpoint_native"
EXPERT_TENSOR_LAYOUT_GPTQMODEL = "gptqmodel"
EXPERT_TENSOR_LAYOUTS = frozenset(
    {
        EXPERT_TENSOR_LAYOUT_CHECKPOINT_NATIVE,
        EXPERT_TENSOR_LAYOUT_GPTQMODEL,
    }
)
ACTIVATION_CORPUS_SCHEMA = "ds4rt-flash-exl3-activation-corpus-v1"
ROUTED_ACTIVATION_CORPUS_SCHEMA = "ds4rt-flash-exl3-activation-corpus-v2"
ROUTED_ACTIVATION_ROUTE_RECORD = struct.Struct("<Hf")
ROUTED_ACTIVATION_ROUTE_RECORD_FORMAT = "u16le_expert_id_f32le_gate_weight"
ROUTE_REPLAY_REPORT_SCHEMA = "ds4rt-flash-route-replay-v1"
ROUTE_REPLAY_REPORT_FILENAME = "route-replay.json"
EXL3_ROUTE_REPLAY_ARTIFACT_FILENAME = "ds4rt-exl3-route-replay.json"
MIN_NATURAL_ROUTES_PER_EXPERT = 1024
EXLLAMAV3_REPOSITORY = "https://github.com/turboderp-org/exllamav3.git"
EXLLAMAV3_REVISION = "0b9745c526a13d5b30f1b58a864efc1932d3d9eb"
EXLLAMAV3_SOURCE_TREE_SHA256 = (
    "8c2f94e3a7335e304c47dad85a5254428950c324acf74263fb7bff59fe057dec"
)
EXLLAMAV3_VERSION = "1.3.0"
MCG_MARKER = 0xCBAC1FED
COPY_CHUNK_BYTES = 64 * 1024 * 1024
EXL3_G_SCALE_SEARCH_MIN = 0.01
EXL3_G_SCALE_SEARCH_MAX = 2.05
EXL3_PROXY_ERROR_FAIL_FAST_MAX = 0.5
ROUTED_EXPERT_RE = re.compile(r"(?:^|\.)(?:ffn|mlp)\.experts\.\d+\.")
DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
}


@dataclass(frozen=True)
class ModelShape:
    hidden_size: int
    intermediate_size: int
    hidden_layers: int
    dspark_layers: int
    experts: int
    swiglu_limit: float

    @property
    def total_layers(self) -> int:
        return self.hidden_layers + self.dspark_layers


@dataclass(frozen=True)
class SourceTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    source_file: Path
    source_offset: int
    nbytes: int


@dataclass(frozen=True)
class OutputTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    source: SourceTensor | None


@dataclass(frozen=True)
class OutputLocation:
    file_name: str
    data_start: int
    data_offset: int
    nbytes: int

    @property
    def absolute_offset(self) -> int:
        return self.data_start + self.data_offset


@dataclass(frozen=True)
class ArtifactPlan:
    snapshot: Path
    model_config: dict[str, Any]
    shape: ModelShape
    tensors: tuple[OutputTensor, ...]
    shard_tensors: tuple[tuple[OutputTensor, ...], ...]
    total_size: int
    source_bytes: int
    trellis_bytes: int
    expert_tensor_layout: str = EXPERT_TENSOR_LAYOUT_CHECKPOINT_NATIVE
    exl3_bits: int = EXL3_BITS

    @property
    def generated_tensors(self) -> tuple[OutputTensor, ...]:
        return tuple(tensor for tensor in self.tensors if tensor.source is None)


@dataclass(frozen=True)
class ActivationCapture:
    path: Path
    layer_id: int
    rows: int
    hidden_size: int
    nbytes: int
    sha256: str
    route_path: Path | None = None
    route_nbytes: int = 0
    route_sha256: str | None = None
    routes_per_row: int = 0


@dataclass(frozen=True)
class ActivationCorpus:
    root: Path
    manifest_path: Path
    checkpoint: Path
    corpus_path: Path
    corpus_sha256: str
    hidden_size: int
    layer_count: int
    dspark_layer_count: int
    routed_experts: int
    top_k: int
    routed_scaling_factor: float
    route_aware: bool
    minimum_natural_routes_per_expert: int
    route_replay_report_path: Path | None
    route_replay_report_sha256: str | None
    prompt_count: int
    captures_by_layer: tuple[tuple[ActivationCapture, ...], ...]

    def rows_for_layer(self, layer_id: int) -> int:
        if not 0 <= layer_id < self.layer_count:
            raise ValueError(
                f"activation layer {layer_id} is outside 0..{self.layer_count}"
            )
        return sum(capture.rows for capture in self.captures_by_layer[layer_id])


@dataclass(frozen=True)
class RoutedActivationLayer:
    samples: Any
    expert_ids: Any
    gate_weights: Any


def is_routed_expert_tensor(name: str) -> bool:
    return ROUTED_EXPERT_RE.search(name) is not None


def block_prefix(shape: ModelShape, layer_id: int) -> str:
    if not 0 <= layer_id < shape.total_layers:
        raise ValueError(f"layer {layer_id} is outside 0..{shape.total_layers}")
    if layer_id < shape.hidden_layers:
        return f"layers.{layer_id}"
    return f"mtp.{layer_id - shape.hidden_layers}"


def expert_projection_base(
    shape: ModelShape,
    layer_id: int,
    expert_id: int,
    stem: str,
) -> str:
    if not 0 <= expert_id < shape.experts:
        raise ValueError(f"expert {expert_id} is outside 0..{shape.experts}")
    if stem not in {"w1", "w2", "w3"}:
        raise ValueError(f"unsupported routed projection {stem!r}")
    return f"{block_prefix(shape, layer_id)}.ffn.experts.{expert_id}.{stem}"


def gptqmodel_expert_projection_base(
    shape: ModelShape,
    layer_id: int,
    expert_id: int,
    stem: str,
) -> str:
    """Return the module namespace emitted by GPTQModel's EXL3 writer.

    DeepSeek's source checkpoint keeps coordinator tensors in its native flat
    namespace (``layers.*`` and ``mtp.*``), while GPTQModel identifies packed
    routed projections by their runtime module paths.  A canonical hybrid
    artifact intentionally retains that distinction instead of transforming
    the coordinator tensors or duplicating the source routed weights.
    """

    if not 0 <= layer_id < shape.total_layers:
        raise ValueError(f"layer {layer_id} is outside 0..{shape.total_layers}")
    if not 0 <= expert_id < shape.experts:
        raise ValueError(f"expert {expert_id} is outside 0..{shape.experts}")
    try:
        projection = {
            "w1": "gate_proj",
            "w2": "down_proj",
            "w3": "up_proj",
        }[stem]
    except KeyError as exc:
        raise ValueError(f"unsupported routed projection {stem!r}") from exc
    if layer_id < shape.hidden_layers:
        block = f"model.layers.{layer_id}"
    else:
        block = f"mtp.{layer_id - shape.hidden_layers}"
    return f"{block}.mlp.experts.{expert_id}.{projection}"


def read_native_model_config(snapshot: str | Path) -> tuple[dict[str, Any], ModelShape]:
    snapshot = Path(snapshot).resolve()
    raw = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    if raw.get("model_type") != "deepseek_v4":
        raise ValueError(f"expected model_type=deepseek_v4, got {raw.get('model_type')!r}")
    quant = raw.get("quantization_config") or {}
    if str(quant.get("quant_method", "")).lower() != "fp8":
        raise ValueError("EXL3 conversion requires the native FP4/FP8 source checkpoint")
    if raw.get("expert_dtype") != "fp4":
        raise ValueError(f"expected expert_dtype=fp4, got {raw.get('expert_dtype')!r}")
    hidden = int(raw["hidden_size"])
    intermediate = int(raw["moe_intermediate_size"])
    if hidden % 128:
        raise ValueError(f"hidden_size={hidden} must be divisible by EXL3 H128")
    alignment = EXPERT_TP_WORLD_SIZE * 128
    if intermediate % alignment:
        raise ValueError(
            f"intermediate_size={intermediate} must be divisible by TP4 H128 ({alignment})"
        )
    target_layers = tuple(int(value) for value in raw["dspark_target_layer_ids"])
    shape = ModelShape(
        hidden_size=hidden,
        intermediate_size=intermediate,
        hidden_layers=int(raw["num_hidden_layers"]),
        dspark_layers=len(target_layers),
        experts=int(raw["n_routed_experts"]),
        swiglu_limit=float(raw.get("swiglu_limit", 0.0)),
    )
    if shape.experts <= 0 or shape.total_layers <= 0:
        raise ValueError("DeepSeek V4 source has no routed expert layers")
    if not math.isfinite(shape.swiglu_limit) or shape.swiglu_limit <= 0.0:
        raise ValueError(
            f"DeepSeek V4 source requires a positive finite swiglu_limit, "
            f"got {shape.swiglu_limit}"
        )
    return raw, shape


def validate_route_distribution_manifest(
    value: object,
    *,
    label: str,
    layer_count: int,
    routed_experts: int,
    top_k: int,
    minimum_routes: int,
) -> None:
    if not isinstance(value, list) or len(value) != layer_count:
        raise ValueError(f"routed activation manifest has invalid {label}")
    for layer_id, layer in enumerate(value):
        if not isinstance(layer, dict) or layer.get("layer_id") != layer_id:
            raise ValueError(f"routed activation {label} has invalid layer {layer_id}")
        counts = layer.get("expert_route_counts")
        if (
            not isinstance(counts, list)
            or len(counts) != routed_experts
            or any(
                isinstance(count, bool) or not isinstance(count, int) or count < 0
                for count in counts
            )
        ):
            raise ValueError(
                f"routed activation {label} has invalid expert counts at layer {layer_id}"
            )
        rows = layer.get("rows")
        routes = layer.get("routes")
        ordered = sorted(counts)
        if (
            isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows <= 0
            or isinstance(routes, bool)
            or not isinstance(routes, int)
            or routes != rows * top_k
            or routes != sum(counts)
            or layer.get("zero_hit_experts") != sum(count == 0 for count in counts)
            or layer.get("min_hits") != ordered[0]
            or layer.get("p50_hits") != ordered[len(ordered) // 2]
            or layer.get("max_hits") != ordered[-1]
            or ordered[0] < minimum_routes
        ):
            raise ValueError(
                f"routed activation {label} does not satisfy its route floor at "
                f"layer {layer_id}"
            )


def validate_route_replay_report(
    path: Path,
    *,
    manifest_sha256: str,
    checkpoint: Path,
    layer_count: int,
    top_k: int,
) -> None:
    if not path.is_file():
        raise ValueError(f"routed activation replay report is missing: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    required_layers = list(range(3, layer_count))
    reported_snapshot = Path(str(report.get("snapshot", ""))).expanduser()
    try:
        reported_snapshot = reported_snapshot.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(
            f"routed activation replay snapshot no longer exists: {reported_snapshot}"
        ) from error
    zero_fields = (
        "ranked_mismatch_rows",
        "set_mismatch_rows",
        "route_entry_mismatches",
        "weight_bit_mismatches",
    )
    layer_summaries = report.get("layer_summaries")
    if (
        report.get("schema") != ROUTE_REPLAY_REPORT_SCHEMA
        or report.get("status") != "exact"
        or report.get("metadata_sha256") != manifest_sha256
        or reported_snapshot != checkpoint
        or report.get("layers") != required_layers
        or report.get("max_capture_rows_per_layer") != -1
        or isinstance(report.get("repeats"), bool)
        or not isinstance(report.get("repeats"), int)
        or report["repeats"] < 2
        or any(report.get(field) != 0 for field in zero_fields)
        or not isinstance(report.get("native_library_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", report["native_library_sha256"])
        or not isinstance(layer_summaries, list)
        or len(layer_summaries) != len(required_layers)
    ):
        raise ValueError("routed activation replay report is incomplete or mismatched")
    total_rows = 0
    total_routes = 0
    for layer_id, summary in zip(required_layers, layer_summaries, strict=True):
        rows = summary.get("rows") if isinstance(summary, dict) else None
        routes = summary.get("routes") if isinstance(summary, dict) else None
        if (
            not isinstance(summary, dict)
            or summary.get("layer") != layer_id
            or isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows <= 0
            or routes != rows * top_k
            or any(summary.get(field) != 0 for field in zero_fields)
            or summary.get("weight_max_abs") != 0.0
        ):
            raise ValueError(
                f"routed activation replay report failed at layer {layer_id}"
            )
        total_rows += rows
        total_routes += routes
    if report.get("rows") != total_rows or report.get("routes") != total_routes:
        raise ValueError("routed activation replay report totals are inconsistent")


def load_activation_corpus(
    manifest_or_root: str | Path,
    *,
    snapshot: str | Path,
    shape: ModelShape,
    required_natural_routes_per_expert: int = MIN_NATURAL_ROUTES_PER_EXPERT,
    require_exact_route_replay: bool = True,
) -> ActivationCorpus:
    """Load a capture manifest and bind it to one exact native checkpoint.

    Activation statistics are checkpoint-specific. In particular, a Flash
    capture must never be accepted while converting Pro merely because a few
    dimensions happen to be compatible. The resolved checkpoint identity and
    all model geometry are therefore part of the fail-closed contract. The
    default route floor is the production calibration floor; callers loading
    a distinct held-out evaluation split must opt into its lower declared
    floor and enforce their sampled-expert row requirement separately.
    """

    if (
        isinstance(required_natural_routes_per_expert, bool)
        or not isinstance(required_natural_routes_per_expert, int)
        or required_natural_routes_per_expert < 0
    ):
        raise ValueError("required natural-route floor must be a nonnegative integer")
    if not isinstance(require_exact_route_replay, bool):
        raise ValueError("exact route-replay requirement must be boolean")
    supplied = Path(manifest_or_root).expanduser().resolve(strict=True)
    manifest_path = supplied / "manifest.json" if supplied.is_dir() else supplied
    if not manifest_path.is_file():
        raise ValueError(f"activation manifest is not a file: {manifest_path}")
    root = manifest_path.parent.resolve()
    manifest_payload = manifest_path.read_bytes()
    manifest = json.loads(manifest_payload)
    schema = manifest.get("schema") if isinstance(manifest, dict) else None
    if schema not in {ACTIVATION_CORPUS_SCHEMA, ROUTED_ACTIVATION_CORPUS_SCHEMA}:
        raise ValueError(
            "activation manifest must use schema "
            f"{ACTIVATION_CORPUS_SCHEMA!r} or {ROUTED_ACTIVATION_CORPUS_SCHEMA!r}"
        )
    route_aware = schema == ROUTED_ACTIVATION_CORPUS_SCHEMA

    captured_checkpoint = Path(str(manifest.get("checkpoint", ""))).expanduser()
    try:
        captured_checkpoint = captured_checkpoint.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(
            f"activation manifest checkpoint no longer exists: {captured_checkpoint}"
        ) from error
    expected_checkpoint = Path(snapshot).expanduser().resolve(strict=True)
    if captured_checkpoint != expected_checkpoint:
        raise ValueError(
            "activation corpus is bound to a different checkpoint: "
            f"captured={captured_checkpoint} quantized={expected_checkpoint}"
        )

    hidden_size = int(manifest.get("hidden_size", 0))
    layer_count = int(manifest.get("layer_count", 0))
    dspark_layer_count = int(manifest.get("dspark_layer_count", -1))
    expected_config = json.loads(
        (expected_checkpoint / "config.json").read_text(encoding="utf-8")
    )
    expected_top_k = int(expected_config["num_experts_per_tok"])
    expected_routed_scaling_factor = float(
        expected_config.get("routed_scaling_factor", 1.0)
    )
    routed_experts = int(manifest.get("routed_experts", shape.experts))
    top_k = int(manifest.get("top_k", expected_top_k))
    routed_scaling_factor = float(
        manifest.get("routed_scaling_factor", expected_routed_scaling_factor)
    )
    if (
        hidden_size != shape.hidden_size
        or layer_count != shape.hidden_layers
        or dspark_layer_count != shape.dspark_layers
        or routed_experts != shape.experts
        or top_k != expected_top_k
        or not math.isfinite(routed_scaling_factor)
        or abs(routed_scaling_factor - expected_routed_scaling_factor) > 1.0e-6
    ):
        raise ValueError(
            "activation corpus geometry does not match the checkpoint: "
            f"capture=({hidden_size},{layer_count},{dspark_layer_count},"
            f"{routed_experts},{top_k}) "
            f"model=({shape.hidden_size},{shape.hidden_layers},{shape.dspark_layers},"
            f"{shape.experts},{expected_top_k})"
        )
    if route_aware and manifest.get("route_record_format") != (
        ROUTED_ACTIVATION_ROUTE_RECORD_FORMAT
    ):
        raise ValueError("routed activation manifest has an unsupported route format")
    minimum_natural_routes_per_expert = 0
    route_replay_report_path = None
    route_replay_report_sha256 = None
    if route_aware:
        retention = manifest.get("retention")
        if not isinstance(retention, dict):
            raise ValueError("routed activation manifest has no retention contract")
        target = retention.get("routes_per_expert_target")
        minimum = retention.get("minimum_natural_routes_per_expert")
        if (
            retention.get("policy")
            != "first_joint_rows_until_per_layer_expert_route_quota"
            or isinstance(target, bool)
            or not isinstance(target, int)
            or isinstance(minimum, bool)
            or not isinstance(minimum, int)
            or minimum < required_natural_routes_per_expert
            or target < minimum
        ):
            raise ValueError(
                "routed activation manifest does not satisfy the required "
                f"{required_natural_routes_per_expert}-route floor"
            )
        minimum_natural_routes_per_expert = minimum
        validate_route_distribution_manifest(
            manifest.get("route_distribution"),
            label="observed route distribution",
            layer_count=layer_count,
            routed_experts=routed_experts,
            top_k=top_k,
            minimum_routes=minimum,
        )
        validate_route_distribution_manifest(
            manifest.get("retained_route_distribution"),
            label="retained route distribution",
            layer_count=layer_count,
            routed_experts=routed_experts,
            top_k=top_k,
            minimum_routes=minimum,
        )
        summary = manifest.get("summary")
        weakest_summary = (
            summary.get("weakest_natural_route_coverage")
            if isinstance(summary, dict)
            else None
        )
        if (
            not isinstance(summary, dict)
            or isinstance(weakest_summary, bool)
            or not isinstance(weakest_summary, int)
            or weakest_summary < minimum
            or summary.get("covered_layers") != list(range(layer_count))
        ):
            raise ValueError("routed activation manifest summary fails its route floor")
        if require_exact_route_replay:
            route_replay_report_path = root / ROUTE_REPLAY_REPORT_FILENAME
            validate_route_replay_report(
                route_replay_report_path,
                manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
                checkpoint=expected_checkpoint,
                layer_count=layer_count,
                top_k=top_k,
            )
            route_replay_report_sha256 = hashlib.sha256(
                route_replay_report_path.read_bytes()
            ).hexdigest()
    corpus_sha256 = manifest.get("corpus_sha256")
    if not isinstance(corpus_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", corpus_sha256
    ):
        raise ValueError("activation manifest has an invalid corpus SHA-256")
    corpus_path_raw = manifest.get("corpus_path")
    if not isinstance(corpus_path_raw, str) or not corpus_path_raw:
        raise ValueError("activation manifest has no source corpus path")
    # Collection may run inside the coordinator container, so this provenance
    # path is descriptive on the host. The raw corpus digest is the portable
    # identity used by the converter.
    corpus_path = Path(corpus_path_raw).expanduser()

    prompt_records = manifest.get("prompts")
    if not isinstance(prompt_records, list) or not prompt_records:
        raise ValueError("activation manifest has no prompt captures")
    captures: list[list[ActivationCapture]] = [[] for _ in range(layer_count)]
    seen_paths: set[Path] = set()
    for prompt_index, prompt in enumerate(prompt_records):
        if not isinstance(prompt, dict) or not isinstance(prompt.get("capture_files"), list):
            raise ValueError(f"activation prompt {prompt_index} has invalid captures")
        for raw_capture in prompt["capture_files"]:
            if not isinstance(raw_capture, dict):
                raise ValueError(
                    f"activation prompt {prompt_index} contains a malformed capture"
                )
            raw_path = raw_capture.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError("activation capture has no path")
            relative = Path(raw_path)
            if relative.is_absolute():
                raise ValueError("activation capture paths must be relative to the manifest")
            path = (root / relative).resolve(strict=True)
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ValueError(f"activation capture escapes its manifest root: {path}") from error
            if path in seen_paths:
                raise ValueError(f"duplicate activation capture path: {relative}")
            seen_paths.add(path)
            layer_id = int(raw_capture.get("layer_id", -1))
            rows = int(raw_capture.get("rows", 0))
            capture_hidden = int(raw_capture.get("hidden_size", 0))
            nbytes = int(raw_capture.get("bytes", 0))
            digest = raw_capture.get("sha256")
            if not 0 <= layer_id < layer_count:
                raise ValueError(f"activation capture has invalid layer {layer_id}")
            if rows <= 0 or capture_hidden != hidden_size:
                raise ValueError(f"activation capture has invalid geometry: {relative}")
            expected_bytes = rows * hidden_size * 2
            if nbytes != expected_bytes or path.stat().st_size != expected_bytes:
                raise ValueError(
                    f"activation capture byte count changed for {relative}: "
                    f"manifest={nbytes} file={path.stat().st_size} expected={expected_bytes}"
                )
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"activation capture has invalid SHA-256: {relative}")
            route_path = None
            route_nbytes = 0
            route_digest = None
            routes_per_row = 0
            if route_aware:
                raw_route_path = raw_capture.get("route_path")
                if not isinstance(raw_route_path, str) or not raw_route_path:
                    raise ValueError(f"routed activation capture has no route path: {relative}")
                route_relative = Path(raw_route_path)
                if route_relative.is_absolute():
                    raise ValueError("activation route paths must be relative to the manifest")
                route_path = (root / route_relative).resolve(strict=True)
                try:
                    route_path.relative_to(root)
                except ValueError as error:
                    raise ValueError(
                        f"activation route capture escapes its manifest root: {route_path}"
                    ) from error
                if route_path in seen_paths:
                    raise ValueError(f"duplicate activation route path: {route_relative}")
                seen_paths.add(route_path)
                route_nbytes = int(raw_capture.get("route_bytes", 0))
                route_digest = raw_capture.get("route_sha256")
                routes_per_row = int(raw_capture.get("routes_per_row", 0))
                expected_route_bytes = (
                    rows * top_k * ROUTED_ACTIVATION_ROUTE_RECORD.size
                )
                if (
                    raw_capture.get("route_record_format")
                    != ROUTED_ACTIVATION_ROUTE_RECORD_FORMAT
                    or routes_per_row != top_k
                    or route_nbytes != expected_route_bytes
                    or route_path.stat().st_size != expected_route_bytes
                ):
                    raise ValueError(
                        f"activation route capture has invalid geometry: {route_relative}"
                    )
                if not isinstance(route_digest, str) or not re.fullmatch(
                    r"[0-9a-f]{64}", route_digest
                ):
                    raise ValueError(
                        f"activation route capture has invalid SHA-256: {route_relative}"
                    )
            captures[layer_id].append(
                ActivationCapture(
                    path=path,
                    layer_id=layer_id,
                    rows=rows,
                    hidden_size=hidden_size,
                    nbytes=nbytes,
                    sha256=digest,
                    route_path=route_path,
                    route_nbytes=route_nbytes,
                    route_sha256=route_digest,
                    routes_per_row=routes_per_row,
                )
            )
    missing_layers = [layer for layer, layer_captures in enumerate(captures) if not layer_captures]
    if missing_layers:
        raise ValueError(f"activation corpus has no captures for layers {missing_layers}")
    return ActivationCorpus(
        root=root,
        manifest_path=manifest_path,
        checkpoint=captured_checkpoint,
        corpus_path=corpus_path,
        corpus_sha256=corpus_sha256,
        hidden_size=hidden_size,
        layer_count=layer_count,
        dspark_layer_count=dspark_layer_count,
        routed_experts=routed_experts,
        top_k=top_k,
        routed_scaling_factor=routed_scaling_factor,
        route_aware=route_aware,
        minimum_natural_routes_per_expert=minimum_natural_routes_per_expert,
        route_replay_report_path=route_replay_report_path,
        route_replay_report_sha256=route_replay_report_sha256,
        prompt_count=len(prompt_records),
        captures_by_layer=tuple(tuple(layer_captures) for layer_captures in captures),
    )


def load_activation_layer_samples(
    corpus: ActivationCorpus,
    layer_id: int,
    *,
    verify_sha256: bool = True,
) -> Any:
    """Load one layer's BF16 capture rows and verify every persistent byte."""

    torch = _torch()
    captures = corpus.captures_by_layer[layer_id]
    total_rows = corpus.rows_for_layer(layer_id)
    samples = torch.empty((total_rows, corpus.hidden_size), dtype=torch.bfloat16)
    offset = 0
    for capture in captures:
        payload = capture.path.read_bytes()
        if len(payload) != capture.nbytes:
            raise ValueError(f"activation capture was truncated: {capture.path}")
        if verify_sha256 and hashlib.sha256(payload).hexdigest() != capture.sha256:
            raise ValueError(f"activation capture SHA-256 changed: {capture.path}")
        values = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).reshape(
            capture.rows, corpus.hidden_size
        )
        samples[offset : offset + capture.rows].copy_(values)
        offset += capture.rows
    if offset != total_rows:
        raise AssertionError(f"activation row accounting changed: {offset} != {total_rows}")
    return samples


def load_routed_activation_layer(
    corpus: ActivationCorpus,
    layer_id: int,
    *,
    verify_sha256: bool = True,
) -> RoutedActivationLayer:
    """Load exact BF16 rows together with their jointly captured router plane."""

    if not corpus.route_aware:
        raise ValueError("activation corpus has no exact per-row routes")
    torch = _torch()
    import numpy as np

    samples = load_activation_layer_samples(
        corpus, layer_id, verify_sha256=verify_sha256
    )
    total_rows = corpus.rows_for_layer(layer_id)
    expert_ids = torch.empty((total_rows, corpus.top_k), dtype=torch.int64)
    gate_weights = torch.empty((total_rows, corpus.top_k), dtype=torch.float32)
    route_dtype = np.dtype([("expert_id", "<u2"), ("gate_weight", "<f4")])
    offset = 0
    for capture in corpus.captures_by_layer[layer_id]:
        if capture.route_path is None or capture.route_sha256 is None:
            raise ValueError(f"routed activation capture is incomplete: {capture.path}")
        payload = capture.route_path.read_bytes()
        if len(payload) != capture.route_nbytes:
            raise ValueError(f"activation route capture was truncated: {capture.route_path}")
        if verify_sha256 and hashlib.sha256(payload).hexdigest() != capture.route_sha256:
            raise ValueError(
                f"activation route capture SHA-256 changed: {capture.route_path}"
            )
        decoded = np.frombuffer(payload, dtype=route_dtype).reshape(
            capture.rows, corpus.top_k
        )
        capture_experts = torch.from_numpy(decoded["expert_id"].copy()).to(torch.int64)
        capture_weights = torch.from_numpy(decoded["gate_weight"].copy())
        if bool((capture_experts >= corpus.routed_experts).any().item()):
            raise ValueError(
                f"activation route capture contains an out-of-range expert: {capture.route_path}"
            )
        if not bool(torch.isfinite(capture_weights).all().item()) or bool(
            (capture_weights < 0).any().item()
        ):
            raise ValueError(
                f"activation route capture contains an invalid gate weight: {capture.route_path}"
            )
        if not bool(
            torch.allclose(
                capture_weights.sum(dim=1),
                torch.full((capture.rows,), corpus.routed_scaling_factor),
                rtol=0.0,
                atol=1.0e-3,
            )
        ):
            raise ValueError(
                "activation route capture gate weights do not sum to the checkpoint "
                f"routing scale {corpus.routed_scaling_factor}: {capture.route_path}"
            )
        ordered_experts = capture_experts.sort(dim=1).values
        if bool((ordered_experts[:, 1:] == ordered_experts[:, :-1]).any().item()):
            raise ValueError(
                f"activation route capture repeats an expert within a row: {capture.route_path}"
            )
        end = offset + capture.rows
        expert_ids[offset:end].copy_(capture_experts)
        gate_weights[offset:end].copy_(capture_weights)
        offset = end
    if offset != total_rows:
        raise AssertionError(f"activation route row accounting changed: {offset} != {total_rows}")
    route_counts = torch.bincount(
        expert_ids.reshape(-1), minlength=corpus.routed_experts
    )
    weakest_route_count = int(route_counts.min().item())
    if weakest_route_count < corpus.minimum_natural_routes_per_expert:
        raise ValueError(
            f"activation route sidecars for layer {layer_id} have only "
            f"{weakest_route_count} hits for the weakest expert; expected at least "
            f"{corpus.minimum_natural_routes_per_expert}"
        )
    return RoutedActivationLayer(
        samples=samples,
        expert_ids=expert_ids,
        gate_weights=gate_weights,
    )


def routed_expert_activation_rows(
    layer: RoutedActivationLayer,
    *,
    expert_id: int,
) -> tuple[Any, Any]:
    """Select exact naturally routed rows and their matching router gates."""

    torch = _torch()
    if expert_id < 0:
        raise ValueError("routed activation expert ID must be non-negative")
    if (
        layer.samples.ndim != 2
        or layer.expert_ids.ndim != 2
        or layer.gate_weights.shape != layer.expert_ids.shape
        or layer.samples.shape[0] != layer.expert_ids.shape[0]
    ):
        raise ValueError("routed activation layer has incompatible geometry")
    matches = (layer.expert_ids == expert_id).nonzero(as_tuple=False)
    if matches.numel() == 0:
        raise ValueError(f"routed activation layer has no rows for expert {expert_id}")
    row_indices = matches[:, 0]
    route_slots = matches[:, 1]
    if int(torch.unique(row_indices).numel()) != int(row_indices.numel()):
        raise ValueError(f"routed activation layer repeats expert {expert_id} in a row")
    samples = layer.samples.index_select(0, row_indices)
    gates = layer.gate_weights[row_indices, route_slots]
    if not bool(torch.isfinite(gates).all().item()) or bool((gates <= 0).any().item()):
        raise ValueError(f"routed activation expert {expert_id} has invalid gates")
    return samples, gates


def exl3_quantization_config(
    *,
    calibration_rows: int,
    seed: int,
    swiglu_limit: float | None = None,
    recipe: str = EXL3_RECIPE,
    calibration_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if calibration_rows <= 0:
        raise ValueError("calibration_rows must be positive")
    if swiglu_limit is not None and (
        not math.isfinite(swiglu_limit) or swiglu_limit <= 0.0
    ):
        raise ValueError("swiglu_limit must be positive and finite")
    if not recipe:
        raise ValueError("EXL3 recipe must be non-empty")
    calibration = (
        dict(calibration_override)
        if calibration_override is not None
        else {
            "method": "layerwise_simulated_isotropic",
            "device": "cuda:0",
            "rows": int(calibration_rows),
            "seed": int(seed),
            "hessian": "analytic_identity",
            "distribution": "rms_normalized_isotropic",
        }
    )
    calibration.setdefault("device", "cuda:0")
    calibration.setdefault("rows", int(calibration_rows))
    calibration.setdefault("seed", int(seed))
    if swiglu_limit is not None:
        calibration.update(
            {
                "activation": "silu",
                "swiglu_limit": float(swiglu_limit),
                "gate_clamp": [None, float(swiglu_limit)],
                "up_clamp": [-float(swiglu_limit), float(swiglu_limit)],
            }
        )
    return {
        "quant_method": "exl3",
        "version": EXLLAMAV3_VERSION,
        "bits": float(EXL3_BITS),
        "codebook": EXL3_CODEBOOK,
        "calibration": calibration,
        "ds4rt": {
            "schema": EXL3_SCHEMA,
            "schema_version": EXL3_SCHEMA_VERSION,
            "recipe": recipe,
            "calibrated": True,
            "scope": "routed_experts",
            "source_format": NATIVE_SOURCE_FORMAT,
            "tensor_format": EXL3_TENSOR_FORMAT,
            "expert_tp_world_size": EXPERT_TP_WORLD_SIZE,
            "quantizer_source": {
                "repository": EXLLAMAV3_REPOSITORY,
                "revision": EXLLAMAV3_REVISION,
                "source_tree_sha256": EXLLAMAV3_SOURCE_TREE_SHA256,
            },
        },
    }


def _read_safetensors_header(path: Path) -> dict[str, SourceTensor]:
    with path.open("rb") as stream:
        raw_len = stream.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"truncated safetensors length: {path}")
        header_len = struct.unpack("<Q", raw_len)[0]
        header = json.loads(stream.read(header_len))
    data_start = 8 + header_len
    result: dict[str, SourceTensor] = {}
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        dtype = str(metadata["dtype"])
        shape = tuple(int(value) for value in metadata["shape"])
        start, end = (int(value) for value in metadata["data_offsets"])
        expected = math.prod(shape) * DTYPE_BYTES.get(dtype, 0)
        if expected <= 0 or end - start != expected:
            raise ValueError(
                f"unsupported or inconsistent tensor {name} in {path}: "
                f"dtype={dtype} shape={shape} bytes={end-start}"
            )
        result[name] = SourceTensor(
            name=name,
            dtype=dtype,
            shape=shape,
            source_file=path,
            source_offset=data_start + start,
            nbytes=end - start,
        )
    return result


def read_source_index(snapshot: str | Path) -> dict[str, SourceTensor]:
    snapshot = Path(snapshot).resolve()
    return _read_indexed_tensors(snapshot, snapshot / "model.safetensors.index.json")


def _read_indexed_tensors(snapshot: Path, index_path: Path) -> dict[str, SourceTensor]:
    index = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    by_file: dict[str, dict[str, SourceTensor]] = {}
    result: dict[str, SourceTensor] = {}
    for name, file_name in index.items():
        file_name = str(file_name)
        if file_name not in by_file:
            by_file[file_name] = _read_safetensors_header(snapshot / file_name)
        try:
            tensor = by_file[file_name][name]
        except KeyError as exc:
            raise ValueError(f"index points to missing tensor {name} in {file_name}") from exc
        result[name] = tensor
    return result


def read_artifact_index(
    snapshot: str | Path, *, allow_incomplete: bool = False
) -> dict[str, SourceTensor]:
    snapshot = Path(snapshot).resolve()
    published = snapshot / "model.safetensors.index.json"
    incomplete = snapshot / "model.safetensors.index.json.incomplete"
    if published.is_file():
        index_path = published
    elif allow_incomplete and incomplete.is_file():
        index_path = incomplete
    else:
        raise ValueError(
            f"EXL3 artifact has no {'published or incomplete' if allow_incomplete else 'published'} "
            f"safetensors index: {snapshot}"
        )
    return _read_indexed_tensors(snapshot, index_path)


def _generated_projection_tensors(
    base: str,
    input_size: int,
    output_size: int,
    *,
    bits: int,
) -> list[OutputTensor]:
    specs = [
        ("trellis", "I16", (input_size // 16, output_size // 16, 16 * bits)),
        ("suh", "F16", (input_size,)),
        ("svh", "F16", (output_size,)),
        ("mcg", "I32", ()),
    ]
    return [
        OutputTensor(
            name=f"{base}.{suffix}",
            dtype=dtype,
            shape=shape,
            nbytes=math.prod(shape) * DTYPE_BYTES[dtype] if shape else DTYPE_BYTES[dtype],
            source=None,
        )
        for suffix, dtype, shape in specs
    ]


def build_artifact_plan(
    snapshot: str | Path,
    *,
    max_shard_bytes: int = 8 * 1024**3,
    expert_tensor_layout: str = EXPERT_TENSOR_LAYOUT_CHECKPOINT_NATIVE,
    exl3_bits: int = EXL3_BITS,
) -> ArtifactPlan:
    if max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive")
    if expert_tensor_layout not in EXPERT_TENSOR_LAYOUTS:
        raise ValueError(
            f"unsupported EXL3 expert tensor layout {expert_tensor_layout!r}"
        )
    if isinstance(exl3_bits, bool) or exl3_bits not in {2, 3}:
        raise ValueError("EXL3 artifact tier must be integer K2 or K3")
    raw_config, shape = read_native_model_config(snapshot)
    source_index = read_source_index(snapshot)
    tensors = [
        OutputTensor(
            name=tensor.name,
            dtype=tensor.dtype,
            shape=tensor.shape,
            nbytes=tensor.nbytes,
            source=tensor,
        )
        for tensor in source_index.values()
        if not is_routed_expert_tensor(tensor.name)
    ]
    native_expert_names = {
        name for name in source_index if is_routed_expert_tensor(name)
    }
    expected_native_names = set()
    for layer_id in range(shape.total_layers):
        for expert_id in range(shape.experts):
            for stem in ("w1", "w2", "w3"):
                source_base = expert_projection_base(shape, layer_id, expert_id, stem)
                expected_native_names.update(
                    {f"{source_base}.weight", f"{source_base}.scale"}
                )
                output_base = (
                    source_base
                    if expert_tensor_layout == EXPERT_TENSOR_LAYOUT_CHECKPOINT_NATIVE
                    else gptqmodel_expert_projection_base(
                        shape,
                        layer_id,
                        expert_id,
                        stem,
                    )
                )
                if stem in {"w1", "w3"}:
                    input_size, output_size = shape.hidden_size, shape.intermediate_size
                else:
                    input_size, output_size = shape.intermediate_size, shape.hidden_size
                tensors.extend(
                    _generated_projection_tensors(
                        output_base,
                        input_size,
                        output_size,
                        bits=exl3_bits,
                    )
                )
    if native_expert_names != expected_native_names:
        missing = sorted(expected_native_names - native_expert_names)[:8]
        unexpected = sorted(native_expert_names - expected_native_names)[:8]
        raise ValueError(
            "native routed expert tensor set mismatch: "
            f"missing={missing} unexpected={unexpected}"
        )
    tensors.sort(
        key=lambda tensor: (
            tensor.source is None,
            str(tensor.source.source_file) if tensor.source else tensor.name,
            tensor.source.source_offset if tensor.source else 0,
            tensor.name,
        )
    )
    shards: list[list[OutputTensor]] = [[]]
    shard_size = 0
    for tensor in tensors:
        if shard_size and shard_size + tensor.nbytes > max_shard_bytes:
            shards.append([])
            shard_size = 0
        shards[-1].append(tensor)
        shard_size += tensor.nbytes
    source_bytes = sum(tensor.nbytes for tensor in tensors if tensor.source is not None)
    trellis_bytes = sum(tensor.nbytes for tensor in tensors if tensor.source is None)
    return ArtifactPlan(
        snapshot=Path(snapshot).resolve(),
        model_config=raw_config,
        shape=shape,
        tensors=tuple(tensors),
        shard_tensors=tuple(tuple(shard) for shard in shards),
        total_size=source_bytes + trellis_bytes,
        source_bytes=source_bytes,
        trellis_bytes=trellis_bytes,
        expert_tensor_layout=expert_tensor_layout,
        exl3_bits=exl3_bits,
    )


def strict_tp4_source_layout(plan: ArtifactPlan) -> dict[str, Any]:
    """Derive the exact rank-local source bytes consumed by the Python oracle.

    Trellis tensors and intermediate-axis rotations are quarter-sliced.  The
    three hidden-axis rotations and three scalar MCG markers per expert remain
    replicated.  Metadata validation against ``plan`` makes this derivation a
    full-artifact structural proof rather than an estimate from model config.
    """

    shape = plan.shape
    hidden = shape.hidden_size
    intermediate = shape.intermediate_size
    if hidden <= 0 or hidden % 128:
        raise ValueError(
            f"EXL3 hidden size {hidden} must be a positive multiple of H128"
        )
    alignment = EXPERT_TP_WORLD_SIZE * 128
    if intermediate <= 0 or intermediate % alignment:
        raise ValueError(
            f"EXL3 intermediate size {intermediate} must be divisible by {alignment}"
        )
    local_intermediate = intermediate // EXPERT_TP_WORLD_SIZE
    projection_trellis_bytes = hidden * local_intermediate * plan.exl3_bits // 8
    per_expert = (
        3 * projection_trellis_bytes
        + (3 * hidden + 3 * local_intermediate) * DTYPE_BYTES["F16"]
        + 3 * DTYPE_BYTES["I32"]
    )
    per_block = per_expert * shape.experts
    total = per_block * shape.total_layers
    return {
        "world_size": EXPERT_TP_WORLD_SIZE,
        "blocks": shape.total_layers,
        "experts_per_block": shape.experts,
        "experts_checked": shape.total_layers * shape.experts,
        "local_intermediate_size": local_intermediate,
        "rank_source_bytes_per_expert": [per_expert] * EXPERT_TP_WORLD_SIZE,
        "rank_source_bytes_per_block": [per_block] * EXPERT_TP_WORLD_SIZE,
        "rank_source_bytes_total": [total] * EXPERT_TP_WORLD_SIZE,
        "equal_rank_source_bytes": True,
    }


def verify_retained_native_tensors(
    plan: ArtifactPlan,
    artifact: str | Path,
    *,
    allow_incomplete: bool = False,
    recipe: str = EXL3_RECIPE,
) -> dict[str, Any]:
    """Verify the hybrid artifact layout and byte-compare retained tensors."""

    artifact = Path(artifact).resolve()
    actual_index = read_artifact_index(artifact, allow_incomplete=allow_incomplete)
    planned_names = {tensor.name for tensor in plan.tensors}
    if set(actual_index) != planned_names:
        missing = sorted(planned_names - set(actual_index))[:8]
        unexpected = sorted(set(actual_index) - planned_names)[:8]
        raise ValueError(
            f"EXL3 artifact tensor set changed: missing={missing} unexpected={unexpected}"
        )

    planned_by_name = {tensor.name: tensor for tensor in plan.tensors}
    for name in sorted(planned_by_name):
        tensor = planned_by_name[name]
        actual = actual_index[name]
        if (
            actual.dtype != tensor.dtype
            or actual.shape != tensor.shape
            or actual.nbytes != tensor.nbytes
        ):
            raise ValueError(
                f"EXL3 artifact tensor metadata changed for {name}: "
                f"expected {tensor.dtype}{tensor.shape}/{tensor.nbytes}, got "
                f"{actual.dtype}{actual.shape}/{actual.nbytes}"
            )

    retained = sorted(
        (tensor for tensor in plan.tensors if tensor.source is not None),
        key=lambda tensor: tensor.name,
    )
    generated = sorted(plan.generated_tensors, key=lambda tensor: tensor.name)
    published = (artifact / "model.safetensors.index.json").is_file()
    generated_complete = published
    if not generated_complete and allow_incomplete:
        state_path = artifact / ".ds4rt-exl3-state.json"
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            completed = {str(name) for name in state.get("completed", [])}
            generated_complete = all(tensor.name in completed for tensor in generated)
    source_descriptors: dict[Path, int] = {}
    artifact_descriptors: dict[Path, int] = {}
    tensor_records = []
    aggregate = hashlib.sha256()
    marker_tensors = [tensor for tensor in generated if tensor.name.endswith(".mcg")]
    markers_verified = False
    try:
        if generated_complete:
            for tensor in marker_tensors:
                actual = actual_index[tensor.name]
                if actual.source_file not in artifact_descriptors:
                    artifact_descriptors[actual.source_file] = os.open(
                        actual.source_file, os.O_RDONLY
                    )
                payload = os.pread(
                    artifact_descriptors[actual.source_file],
                    DTYPE_BYTES["I32"],
                    actual.source_offset,
                )
                if len(payload) != DTYPE_BYTES["I32"]:
                    raise IOError(f"short read while checking EXL3 marker {tensor.name}")
                if struct.unpack("<I", payload)[0] != MCG_MARKER:
                    raise ValueError(f"EXL3 tensor {tensor.name} has a non-MCG marker")
            markers_verified = True
        for tensor in retained:
            source = tensor.source
            assert source is not None
            actual = actual_index[tensor.name]
            if source.source_file not in source_descriptors:
                source_descriptors[source.source_file] = os.open(
                    source.source_file, os.O_RDONLY
                )
            if actual.source_file not in artifact_descriptors:
                artifact_descriptors[actual.source_file] = os.open(
                    actual.source_file, os.O_RDONLY
                )
            source_fd = source_descriptors[source.source_file]
            artifact_fd = artifact_descriptors[actual.source_file]
            digest = hashlib.sha256()
            remaining = tensor.nbytes
            source_offset = source.source_offset
            artifact_offset = actual.source_offset
            while remaining:
                count = min(remaining, COPY_CHUNK_BYTES)
                expected = os.pread(source_fd, count, source_offset)
                observed = os.pread(artifact_fd, count, artifact_offset)
                if len(expected) != count or len(observed) != count:
                    raise IOError(f"short read while checking retained tensor {tensor.name}")
                if observed != expected:
                    raise ValueError(f"retained native tensor bytes changed for {tensor.name}")
                digest.update(observed)
                remaining -= count
                source_offset += count
                artifact_offset += count
            record = {
                "name": tensor.name,
                "dtype": tensor.dtype,
                "shape": list(tensor.shape),
                "bytes": tensor.nbytes,
                "sha256": digest.hexdigest(),
            }
            tensor_records.append(record)
            aggregate.update(
                json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
            )
            aggregate.update(b"\n")
    finally:
        for descriptor in source_descriptors.values():
            os.close(descriptor)
        for descriptor in artifact_descriptors.values():
            os.close(descriptor)

    retained_bytes = sum(tensor.nbytes for tensor in retained)
    if retained_bytes != plan.source_bytes:
        raise AssertionError(
            f"retained byte accounting changed: {retained_bytes} != {plan.source_bytes}"
        )
    generated_bytes = sum(tensor.nbytes for tensor in generated)
    if generated_bytes != plan.trellis_bytes:
        raise AssertionError(
            f"generated byte accounting changed: {generated_bytes} != {plan.trellis_bytes}"
        )
    return {
        "schema": "ds4rt-exl3-retained-native-integrity-v1",
        "recipe": recipe,
        "quantization_scope": "routed_experts_only",
        "native_snapshot": str(plan.snapshot),
        "exl3_snapshot": str(artifact),
        "retained_tensor_count": len(retained),
        "retained_bytes": retained_bytes,
        "generated_exl3_tensor_count": len(generated),
        "generated_exl3_bytes": generated_bytes,
        "generated_exl3_metadata_verified": True,
        "generated_exl3_mcg_tensor_count": len(marker_tensors),
        "generated_exl3_mcg_markers_verified": markers_verified,
        "artifact_tensor_count": len(plan.tensors),
        "artifact_bytes": retained_bytes + generated_bytes,
        "strict_tp4_source_layout": strict_tp4_source_layout(plan),
        "aggregate_sha256": aggregate.hexdigest(),
        "tensors": tensor_records,
    }


def gptqmodel_tensor_storage_for_plan(
    plan: ArtifactPlan,
) -> dict[str, dict[str, Any]]:
    """Build exact GPTQModel EXL3 storage metadata for a canonical plan."""

    if plan.expert_tensor_layout != EXPERT_TENSOR_LAYOUT_GPTQMODEL:
        raise ValueError(
            "GPTQModel tensor_storage requires the GPTQModel expert layout"
        )
    bits = getattr(plan, "exl3_bits", EXL3_BITS)
    if isinstance(bits, bool) or bits not in {2, 3}:
        raise ValueError("GPTQModel tensor_storage has an invalid EXL3 tier")
    expected_by_base: dict[str, dict[str, OutputTensor]] = {}
    for tensor in plan.generated_tensors:
        base, separator, suffix = tensor.name.rpartition(".")
        if not separator or suffix not in {"trellis", "suh", "svh", "mcg"}:
            raise ValueError(f"invalid planned EXL3 tensor name {tensor.name!r}")
        expected_by_base.setdefault(base, {})[suffix] = tensor
    dtype_names = {"I16": "int16", "F16": "float16", "I32": "int32"}
    storage: dict[str, dict[str, Any]] = {}
    for base, expected in expected_by_base.items():
        if set(expected) != {"trellis", "suh", "svh", "mcg"}:
            raise ValueError(f"planned EXL3 projection is incomplete for {base}")
        storage[base] = {
            "stored_tensors": {
                f"{base}.{suffix}": {
                    "shape": list(tensor.shape),
                    "torch_dtype": dtype_names[tensor.dtype],
                }
                for suffix, tensor in expected.items()
            },
            "quant_format": "exl3",
            "bits_per_weight": bits,
            "mcg_multiplier": MCG_MARKER,
        }
    if not storage:
        raise ValueError("GPTQModel tensor_storage plan is empty")
    return storage


class SafetensorsArtifactWriter:
    """Preplanned, resumable streaming writer for the hybrid HF snapshot."""

    def __init__(
        self,
        plan: ArtifactPlan,
        output: str | Path,
        *,
        calibration_rows: int,
        seed: int,
        resume: bool = False,
        recipe: str = EXL3_RECIPE,
        calibration_override: dict[str, Any] | None = None,
        quant_config_override: dict[str, Any] | None = None,
        quant_config_filename: str = "quantization_config.json",
    ) -> None:
        self.plan = plan
        self.output = Path(output).resolve()
        self.recipe = recipe
        if Path(quant_config_filename).name != quant_config_filename:
            raise ValueError("EXL3 quantization-config filename must be one basename")
        self.quant_config_filename = quant_config_filename
        self.preserve_quant_config = quant_config_override is not None
        self.quant_config = (
            deepcopy(quant_config_override)
            if quant_config_override is not None
            else exl3_quantization_config(
                calibration_rows=calibration_rows,
                seed=seed,
                swiglu_limit=plan.shape.swiglu_limit,
                recipe=recipe,
                calibration_override=calibration_override,
            )
        )
        if not isinstance(self.quant_config, dict) or not self.quant_config:
            raise ValueError("EXL3 quantization config must be a non-empty object")
        if self.preserve_quant_config:
            self._validate_preserved_quant_config()
        self.state_path = self.output / ".ds4rt-exl3-state.json"
        self.partial_report_path = self.output / "ds4rt-exl3-calibration.json.incomplete"
        self.locations: dict[str, OutputLocation] = {}
        self.dirty_shards: set[str] = set()
        self.generated_by_name = {
            tensor.name: tensor for tensor in self.plan.generated_tensors
        }
        self.completed: set[str] = set()
        if self.output == plan.snapshot:
            raise ValueError("EXL3 output must differ from the source snapshot")
        if resume:
            self._load_state()
            self._rebuild_locations(create=False)
        else:
            if self.output.exists() and any(self.output.iterdir()):
                raise ValueError(f"output directory is not empty: {self.output}")
            self.output.mkdir(parents=True, exist_ok=True)
            free_bytes = shutil.disk_usage(self.output).free
            if free_bytes < self.plan.total_size + self.plan.total_size // 100:
                raise ValueError(
                    f"output filesystem has {free_bytes} free bytes but the planned "
                    f"artifact plus 1% headroom needs {self.plan.total_size + self.plan.total_size // 100}"
                )
            self._copy_assets()
            self._rebuild_locations(create=True)
            self._write_incomplete_metadata()
            self._save_state()

    def _copy_assets(self) -> None:
        for source in self.plan.snapshot.iterdir():
            if not source.is_file():
                continue
            if source.name == "config.json" or source.name == "model.safetensors.index.json":
                continue
            if source.suffix == ".safetensors":
                continue
            shutil.copy2(source, self.output / source.name)

    def _shard_name(self, index: int) -> str:
        count = len(self.plan.shard_tensors)
        return (
            "model.safetensors"
            if count == 1
            else f"model-{index + 1:05}-of-{count:05}.safetensors"
        )

    def _rebuild_locations(self, *, create: bool) -> None:
        for shard_index, tensors in enumerate(self.plan.shard_tensors):
            file_name = self._shard_name(shard_index)
            header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
            offset = 0
            for tensor in tensors:
                header[tensor.name] = {
                    "dtype": tensor.dtype,
                    "shape": list(tensor.shape),
                    "data_offsets": [offset, offset + tensor.nbytes],
                }
                offset += tensor.nbytes
            header_bytes = json.dumps(
                header, separators=(",", ":"), sort_keys=True
            ).encode()
            while (8 + len(header_bytes)) % 8:
                header_bytes += b" "
            data_start = 8 + len(header_bytes)
            path = self.output / file_name
            if create:
                with path.open("wb") as stream:
                    stream.write(struct.pack("<Q", len(header_bytes)))
                    stream.write(header_bytes)
                    stream.truncate(data_start + offset)
            elif not path.is_file() or path.stat().st_size != data_start + offset:
                raise ValueError(f"resume shard does not match its plan: {path}")
            current = 0
            for tensor in tensors:
                self.locations[tensor.name] = OutputLocation(
                    file_name=file_name,
                    data_start=data_start,
                    data_offset=current,
                    nbytes=tensor.nbytes,
                )
                current += tensor.nbytes

    def _state_identity(self) -> dict[str, Any]:
        tensor_layout_sha256 = hashlib.sha256(
            b"".join(
                json.dumps(
                    {
                        "name": tensor.name,
                        "dtype": tensor.dtype,
                        "shape": tensor.shape,
                        "bytes": tensor.nbytes,
                        "retained": tensor.source is not None,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                + b"\n"
                for tensor in self.plan.tensors
            )
        ).hexdigest()
        return {
            "schema": 2,
            "recipe": self.recipe,
            "source_snapshot": str(self.plan.snapshot),
            "tensor_count": len(self.plan.tensors),
            "total_size": self.plan.total_size,
            "expert_tensor_layout": self.plan.expert_tensor_layout,
            "tensor_layout_sha256": tensor_layout_sha256,
            "quant_config_filename": self.quant_config_filename,
            "quant_config_sha256": hashlib.sha256(
                json.dumps(
                    self.quant_config,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        }

    def _validate_preserved_quant_config(self) -> None:
        """Bind GPTQModel tensor-storage metadata to the planned packed tensors."""

        storage = self.quant_config.get("tensor_storage")
        expected_storage = gptqmodel_tensor_storage_for_plan(self.plan)
        if not isinstance(storage, dict) or storage != expected_storage:
            raise ValueError(
                "preserved EXL3 tensor_storage differs from the artifact plan"
            )

    def _save_state(self) -> None:
        state = self._state_identity()
        state["completed"] = sorted(self.completed)
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        temporary.replace(self.state_path)

    def _sync_dirty_shards(self) -> None:
        for file_name in sorted(self.dirty_shards):
            descriptor = os.open(self.output / file_name, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        self.dirty_shards.clear()

    def _load_state(self) -> None:
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        expected = self._state_identity()
        actual = {key: state.get(key) for key in expected}
        if actual != expected:
            raise ValueError(f"resume state does not match conversion plan: {actual}")
        self.completed = set(str(name) for name in state.get("completed", []))

    def _write_incomplete_metadata(self) -> None:
        config = dict(self.plan.model_config)
        config["quantization_config"] = self.quant_config
        (self.output / "config.json.incomplete").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        weight_map = {
            name: location.file_name for name, location in self.locations.items()
        }
        index = {"metadata": {"total_size": self.plan.total_size}, "weight_map": weight_map}
        (self.output / "model.safetensors.index.json.incomplete").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _copy_range(self, source: SourceTensor, location: OutputLocation) -> None:
        source_fd = os.open(source.source_file, os.O_RDONLY)
        destination_fd = os.open(self.output / location.file_name, os.O_WRONLY)
        try:
            remaining = source.nbytes
            source_offset = source.source_offset
            destination_offset = location.absolute_offset
            while remaining:
                count = min(remaining, COPY_CHUNK_BYTES)
                payload = os.pread(source_fd, count, source_offset)
                if len(payload) != count:
                    raise IOError(f"short read while copying {source.name}")
                written = os.pwrite(destination_fd, payload, destination_offset)
                if written != count:
                    raise IOError(f"short write while copying {source.name}")
                remaining -= count
                source_offset += count
                destination_offset += count
        finally:
            os.close(destination_fd)
            os.close(source_fd)

    def copy_native_tensors(self, *, checkpoint_interval: int = 128) -> None:
        pending = 0
        for tensor in self.plan.tensors:
            if tensor.source is None or tensor.name in self.completed:
                continue
            self._copy_range(tensor.source, self.locations[tensor.name])
            self.dirty_shards.add(self.locations[tensor.name].file_name)
            self.completed.add(tensor.name)
            pending += 1
            if pending >= checkpoint_interval:
                self.checkpoint()
                pending = 0
        self.checkpoint()

    def write_generated_tensor(self, name: str, tensor: Any) -> None:
        if name in self.completed:
            return
        try:
            expected = self.generated_by_name[name]
        except KeyError as exc:
            raise ValueError(f"{name} is not a planned EXL3 tensor") from exc
        torch = _torch()
        actual = tensor.detach().contiguous().cpu()
        dtype_map = {
            "I16": torch.int16,
            "F16": torch.float16,
            "I32": torch.int32,
        }
        if actual.dtype != dtype_map[expected.dtype] or tuple(actual.shape) != expected.shape:
            raise ValueError(
                f"generated tensor {name} is {actual.dtype} {tuple(actual.shape)}, "
                f"expected {expected.dtype} {expected.shape}"
            )
        payload = actual.numpy().tobytes(order="C")
        location = self.locations[name]
        if len(payload) != location.nbytes:
            raise ValueError(f"generated tensor {name} byte count changed")
        destination_fd = os.open(self.output / location.file_name, os.O_WRONLY)
        try:
            if os.pwrite(destination_fd, payload, location.absolute_offset) != len(payload):
                raise IOError(f"short write for generated tensor {name}")
        finally:
            os.close(destination_fd)
        self.completed.add(name)
        self.dirty_shards.add(location.file_name)

    def checkpoint(self) -> None:
        self._sync_dirty_shards()
        self._save_state()

    def generated_group_complete(self, bases: Iterable[str]) -> bool:
        return all(
            f"{base}.{suffix}" in self.completed
            for base in bases
            for suffix in ("trellis", "suh", "svh", "mcg")
        )

    def load_partial_report(self, template: dict[str, Any]) -> dict[str, Any]:
        if not self.partial_report_path.is_file():
            return template
        report = json.loads(self.partial_report_path.read_text(encoding="utf-8"))
        for key in (
            "schema",
            "recipe",
            "source_snapshot",
            "calibration_rows",
            "seed",
            "activation",
            "swiglu_limit",
            "hessian",
            "distribution",
            "calibration_contract_sha256",
        ):
            if report.get(key) != template.get(key):
                raise ValueError(f"partial calibration report changed at {key}")
        return report

    def checkpoint_report(self, report: dict[str, Any]) -> None:
        self._sync_dirty_shards()
        temporary = self.partial_report_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.partial_report_path)
        self._save_state()

    def finish(
        self,
        calibration_report: dict[str, Any],
        *,
        evidence_files: dict[str, Path] | None = None,
    ) -> None:
        missing = {tensor.name for tensor in self.plan.tensors} - self.completed
        if missing:
            raise ValueError(f"cannot publish incomplete EXL3 artifact; missing {len(missing)} tensors")
        self.checkpoint()
        retained_integrity = verify_retained_native_tensors(
            self.plan,
            self.output,
            allow_incomplete=True,
            recipe=self.recipe,
        )
        retained_path = self.output / "ds4rt-exl3-retained-native.json"
        retained_temporary = retained_path.with_suffix(".json.tmp")
        retained_temporary.write_text(
            json.dumps(retained_integrity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        retained_temporary.replace(retained_path)
        calibration_report = dict(calibration_report)
        calibration_report["retained_native_integrity"] = {
            key: value for key, value in retained_integrity.items() if key != "tensors"
        }
        qconfig = deepcopy(self.quant_config)
        if not self.preserve_quant_config:
            qconfig["tensor_storage"] = {
                tensor.name.removesuffix(".trellis"): {
                    "quant_format": "exl3",
                    "bits_per_weight": self.plan.exl3_bits,
                }
                for tensor in self.plan.generated_tensors
                if tensor.name.endswith(".trellis")
            }
        (self.output / self.quant_config_filename).write_text(
            json.dumps(qconfig, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (self.output / "ds4rt-exl3-calibration.json").write_text(
            json.dumps(calibration_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for name, source in (evidence_files or {}).items():
            if Path(name).name != name or not source.is_file():
                raise ValueError(f"invalid EXL3 evidence file {name!r}: {source}")
            destination = self.output / name
            temporary = destination.with_name(f".{destination.name}.tmp")
            shutil.copy2(source, temporary)
            # Source journals are private run-state evidence and may be 0600.
            # The canonical artifact is a portable HF snapshot, so its bound
            # evidence must be readable by the serving/staging account.
            os.chmod(temporary, 0o644)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            temporary.replace(destination)
        self.partial_report_path.unlink(missing_ok=True)
        for incomplete_name, final_name in (
            ("config.json.incomplete", "config.json"),
            (
                "model.safetensors.index.json.incomplete",
                "model.safetensors.index.json",
            ),
        ):
            incomplete = self.output / incomplete_name
            final = self.output / final_name
            if incomplete.is_symlink() or final.is_symlink():
                raise ValueError("EXL3 publication metadata cannot be a symlink")
            if incomplete.is_file():
                if final.exists():
                    raise ValueError(
                        f"EXL3 publication has both {incomplete_name} and {final_name}"
                    )
                incomplete.replace(final)
            elif not final.is_file():
                raise ValueError(
                    f"EXL3 publication metadata is missing {incomplete_name}"
                )
        self.state_path.unlink()


def dequantize_native_fp4_projection(packed: Any, scales: Any) -> Any:
    """Decode checkpoint `[N,K/2]` FP4 + `[N,K/32]` E8M0 to `[K,N]` f32."""

    torch = _torch()
    if packed.ndim != 2 or scales.ndim != 2:
        raise ValueError("native expert weight and scale must both be rank 2")
    packed_u8 = packed.view(torch.uint8)
    scale_u8 = scales.view(torch.uint8)
    rows, packed_columns = (int(value) for value in packed_u8.shape)
    columns = packed_columns * 2
    if tuple(scale_u8.shape) != (rows, columns // 32):
        raise ValueError(
            f"native E8M0 shape {tuple(scale_u8.shape)} does not match FP4 {(rows, columns)}"
        )
    codes = torch.empty((rows, columns), dtype=torch.uint8, device=packed.device)
    codes[:, 0::2] = packed_u8 & 0x0F
    codes[:, 1::2] = (packed_u8 >> 4) & 0x0F
    codebook = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=packed.device,
    )
    block_scales = scale_u8.view(torch.float8_e8m0fnu).float()
    if not bool(torch.isfinite(block_scales).all().item()):
        raise ValueError("native expert E8M0 scale contains a non-finite value")
    output_input = codebook[codes.long()] * block_scales.repeat_interleave(32, dim=1)
    return output_input.T.contiguous()


def simulated_hidden_states(
    *,
    rows: int,
    hidden_size: int,
    layer_id: int,
    seed: int,
    device: Any,
    rms_norm_eps: float,
) -> Any:
    torch = _torch()
    if rows <= 0 or hidden_size <= 0:
        raise ValueError("simulated activation dimensions must be positive")
    generator = torch.Generator(device=device).manual_seed(seed + 1_000_003 * layer_id)
    hidden = torch.randn(
        (rows, hidden_size),
        generator=generator,
        dtype=torch.float32,
        device=device,
    )
    inverse_rms = torch.rsqrt(hidden.square().mean(dim=1, keepdim=True) + rms_norm_eps)
    return hidden * inverse_rms


def hessian_data(samples: Any, *, key: str) -> dict[str, Any]:
    torch = _torch()
    samples = samples.float()
    return {
        "H": samples.T @ samples,
        "first_key": key,
        "count": int(samples.shape[0]),
        "finalized": False,
        "num_total": int(samples.numel()),
        "inf_nan": torch.zeros(2, dtype=torch.long, device=samples.device),
        "device": samples.device,
    }


def hessian_matrix_data(
    matrix: Any,
    *,
    count: int,
    key: str,
) -> dict[str, Any]:
    """Wrap a summed sample covariance in ExLlamaV3's capture contract."""

    torch = _torch()
    if count <= 0 or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Hessian matrix data requires a positive square covariance")
    if matrix.dtype != torch.float32:
        raise ValueError("Hessian matrix data must be float32")
    if not bool(torch.isfinite(matrix).all().item()):
        raise ValueError(f"Hessian matrix contains non-finite values for {key}")
    return {
        "H": matrix,
        "first_key": key,
        "count": int(count),
        "finalized": False,
        "num_total": int(count * matrix.shape[0]),
        "inf_nan": torch.zeros(2, dtype=torch.long, device=matrix.device),
        "device": matrix.device,
    }


def accumulate_activation_hessian(
    samples: Any,
    *,
    key: str,
    device: Any,
    chunk_rows: int = 2048,
) -> dict[str, Any]:
    """Accumulate X^T X without materializing the complete capture on GPU."""

    torch = _torch()
    if samples.ndim != 2 or samples.shape[0] <= 0 or samples.shape[1] <= 0:
        raise ValueError("activation Hessian samples must be a non-empty matrix")
    if chunk_rows <= 0:
        raise ValueError("activation Hessian chunk size must be positive")
    target = torch.device(device)
    features = int(samples.shape[1])
    matrix = torch.zeros((features, features), dtype=torch.float32, device=target)
    for start in range(0, int(samples.shape[0]), chunk_rows):
        chunk = samples[start : start + chunk_rows].to(
            device=target, dtype=torch.float32, non_blocking=False
        )
        matrix.addmm_(chunk.T, chunk)
        del chunk
    return hessian_matrix_data(
        matrix,
        count=int(samples.shape[0]),
        key=key,
    )


def normalized_route_gate_weights(gates: Any) -> Any:
    """Normalize route gates to unit RMS while preserving relative error weight."""

    torch = _torch()
    gates = gates.to(dtype=torch.float32, device="cpu")
    if gates.ndim != 1 or gates.numel() <= 0:
        raise ValueError("route gate weights must be a non-empty vector")
    if not bool(torch.isfinite(gates).all().item()) or bool((gates <= 0).any().item()):
        raise ValueError("route gate weights must be positive and finite")
    rms = gates.square().mean().sqrt()
    if not bool(torch.isfinite(rms).item()) or float(rms.item()) <= 0.0:
        raise ValueError("route gate RMS must be positive and finite")
    return gates / rms


def accumulate_routed_activation_hessian(
    samples: Any,
    gates: Any,
    *,
    key: str,
    device: Any,
    chunk_rows: int = 2048,
) -> dict[str, Any]:
    """Accumulate X^T diag(gate^2) X with route gates normalized to unit RMS."""

    torch = _torch()
    if (
        samples.ndim != 2
        or samples.shape[0] <= 0
        or samples.shape[1] <= 0
        or gates.ndim != 1
        or samples.shape[0] != gates.shape[0]
    ):
        raise ValueError("routed Hessian samples and gates have incompatible geometry")
    if chunk_rows <= 0:
        raise ValueError("routed Hessian chunk size must be positive")
    target = torch.device(device)
    normalized_gates = normalized_route_gate_weights(gates)
    features = int(samples.shape[1])
    matrix = torch.zeros((features, features), dtype=torch.float32, device=target)
    for start in range(0, int(samples.shape[0]), chunk_rows):
        end = min(int(samples.shape[0]), start + chunk_rows)
        chunk = samples[start:end].to(
            device=target, dtype=torch.float32, non_blocking=False
        )
        chunk.mul_(
            normalized_gates[start:end].to(
                device=target, dtype=torch.float32, non_blocking=False
            )[:, None]
        )
        matrix.addmm_(chunk.T, chunk)
        del chunk
    return hessian_matrix_data(
        matrix,
        count=int(samples.shape[0]),
        key=key,
    )


def routed_expert_down_hessian(
    samples: Any,
    gates: Any,
    gate_weight: Any,
    up_weight: Any,
    *,
    swiglu_limit: float,
    key: str,
    device: Any,
) -> dict[str, Any]:
    """Build the exact native SwiGLU down-input covariance for one routed expert."""

    torch = _torch()
    hidden = samples.to(device=device, dtype=torch.float32, non_blocking=False)
    route_weights = normalized_route_gate_weights(gates).to(
        device=device, dtype=torch.float32, non_blocking=False
    )
    gate = hidden @ gate_weight
    up = hidden @ up_weight
    del hidden
    activations = deepseek_v4_swiglu_activation(
        gate,
        up,
        limit=swiglu_limit,
    )
    del gate, up
    activations.mul_(route_weights[:, None])
    return hessian_data(activations, key=key)


def deterministic_expert_row_indices(
    *,
    total_rows: int,
    rows: int,
    layer_id: int,
    expert_id: int,
    seed: int,
    device: Any = "cpu",
) -> Any:
    """Select a deterministic, non-repeating bounded row set for one expert."""

    torch = _torch()
    if total_rows <= 0 or rows <= 0 or rows > total_rows:
        raise ValueError("forced expert rows must be in 1..total_rows")
    if layer_id < 0 or expert_id < 0:
        raise ValueError("forced expert layer and expert IDs must be non-negative")
    digest = hashlib.sha256(
        f"{seed}:{layer_id}:{expert_id}:{total_rows}".encode()
    ).digest()
    offset = int.from_bytes(digest[:8], "little") % total_rows
    step = int.from_bytes(digest[8:16], "little") % total_rows
    step = max(step, 1)
    while math.gcd(step, total_rows) != 1:
        step = step + 1 if step + 1 < total_rows else 1
    return (
        torch.arange(rows, dtype=torch.int64, device=device) * step + offset
    ).remainder(total_rows)


def blended_forced_down_hessian(
    activations: Any,
    *,
    pooled_covariance: Any,
    expert_weight: float,
    key: str,
) -> dict[str, Any]:
    """Shrink one forced expert covariance toward the equal-expert pool."""

    torch = _torch()
    if activations.ndim != 2 or activations.shape[0] <= 0:
        raise ValueError("forced down activations must be a non-empty matrix")
    if (
        pooled_covariance.ndim != 2
        or pooled_covariance.shape[0] != pooled_covariance.shape[1]
        or pooled_covariance.shape[0] != activations.shape[1]
    ):
        raise ValueError("forced down pooled covariance has incompatible geometry")
    if not 0.0 <= expert_weight <= 1.0 or not math.isfinite(expert_weight):
        raise ValueError("forced down expert weight must be finite and in 0..1")
    rows = int(activations.shape[0])
    expert_covariance = activations.float().T @ activations.float()
    expert_covariance.div_(rows)
    blended = pooled_covariance.clone()
    blended.lerp_(expert_covariance, expert_weight)
    blended.mul_(rows)
    del expert_covariance
    return hessian_matrix_data(blended, count=rows, key=key)


def analytic_isotropic_hessian(
    *, features: int, equivalent_rows: int, key: str, device: Any
) -> dict[str, Any]:
    """Return the exact covariance of the simulated RMS-isotropic distribution.

    A 512-row sample covariance is rank deficient for Flash's 4096-wide hidden
    state and overfits an arbitrary random subspace. The simulated source
    distribution is isotropic by construction, so use its analytic covariance
    instead. Scaling by ``equivalent_rows`` preserves the quantizer's Hessian
    normalization without changing the optimum.
    """

    torch = _torch()
    if features <= 0 or equivalent_rows <= 0:
        raise ValueError("analytic isotropic Hessian dimensions must be positive")
    return {
        "H": torch.eye(features, dtype=torch.float32, device=device) * equivalent_rows,
        "first_key": key,
        "count": int(equivalent_rows),
        "finalized": False,
        "num_total": int(equivalent_rows * features),
        "inf_nan": torch.zeros(2, dtype=torch.long, device=device),
        "device": device,
    }


def deepseek_v4_swiglu_activation(gate: Any, up: Any, *, limit: float) -> Any:
    """Match the serving kernel's bounded DeepSeek V4 SwiGLU formula."""

    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("DeepSeek V4 SwiGLU limit must be positive and finite")
    torch = _torch()
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return torch.nn.functional.silu(gate) * up


def calibration_report_for_run(
    writer: SafetensorsArtifactWriter, template: dict[str, Any]
) -> dict[str, Any]:
    """Resume a checkpoint without carrying its previous stop marker forward."""

    report = writer.load_partial_report(template)
    report.pop("incomplete", None)
    return report


def quantization_progress(
    layer_reports: list[dict[str, Any]], *, total_layers: int
) -> dict[str, int | float | None]:
    """Summarize durable layer timings for operator-visible ETA reporting."""

    if total_layers <= 0:
        raise ValueError("quantization progress requires a positive layer count")
    completed_layers = len(layer_reports)
    if completed_layers > total_layers:
        raise ValueError("completed quantization layers exceed the planned layer count")
    durations = [
        float(layer["elapsed_seconds"])
        for layer in layer_reports
        if isinstance(layer.get("elapsed_seconds"), (int, float))
        and math.isfinite(float(layer["elapsed_seconds"]))
        and float(layer["elapsed_seconds"]) >= 0.0
    ]
    quantized_layer_seconds = math.fsum(durations)
    mean_layer_seconds = quantized_layer_seconds / len(durations) if durations else None
    estimated_remaining_seconds = (
        mean_layer_seconds * (total_layers - completed_layers)
        if mean_layer_seconds is not None
        else None
    )
    return {
        "completed_layers": completed_layers,
        "total_layers": total_layers,
        "completed_fraction": completed_layers / total_layers,
        "timed_layers": len(durations),
        "quantized_layer_seconds": quantized_layer_seconds,
        "mean_layer_seconds": mean_layer_seconds,
        "estimated_remaining_seconds": estimated_remaining_seconds,
    }


def validate_projection_quantization_diagnostic(
    *, name: str, proxy_error: float, g_scale: float
) -> None:
    """Reject numerical corruption before an entire artifact is committed.

    These bounds are deliberately looser than the production operator-quality
    gate.  A finite K2 search for the frozen DeepSeek recipe must not hit the
    upstream global-scale bracket itself, and its Hessian-weighted proxy error
    must remain below a gross-corruption ceiling.  End-to-end promotion still
    depends on the stricter native-output validator.
    """

    if not math.isfinite(g_scale):
        raise RuntimeError(f"EXL3 global scale is non-finite for {name}: {g_scale}")
    if not EXL3_G_SCALE_SEARCH_MIN < g_scale < EXL3_G_SCALE_SEARCH_MAX:
        raise RuntimeError(
            f"EXL3 global scale hit the search boundary for {name}: {g_scale} "
            f"not in ({EXL3_G_SCALE_SEARCH_MIN}, {EXL3_G_SCALE_SEARCH_MAX})"
        )
    if (
        not math.isfinite(proxy_error)
        or not 0.0 <= proxy_error < EXL3_PROXY_ERROR_FAIL_FAST_MAX
    ):
        raise RuntimeError(
            f"EXL3 proxy error indicates numerical corruption for {name}: "
            f"{proxy_error} not in [0, {EXL3_PROXY_ERROR_FAIL_FAST_MAX})"
        )


def qualify_multigpu_batch_equivalence(
    *,
    quantize_exl3_batch: Callable[..., Any],
    quantization_devices: tuple[int, ...],
    device_ratios: tuple[int, ...] | None,
    seed: int,
    debug_dir: str | Path,
    production_projection_shapes: tuple[tuple[int, int], ...] = (),
) -> dict[str, Any] | None:
    """Prove repeated and production-shape one/two-GPU equivalence.

    The production converter reuses one finalized Hessian across expert
    groups.  A single-projection tile test cannot cover the caller/worker CUDA
    stream handoff used by the batched scale search, so exercise four successive
    four-tensor batches.  Small repeated batches expose stream reuse cheaply;
    model-sized FC1/FC2 projections additionally exercise the exact trellis
    tile extents used by the selected checkpoint.  Every diagnostic and emitted
    tensor must be bit exact.
    """

    if len(quantization_devices) <= 1:
        return None
    torch = _torch()
    if quantization_devices[0] != 0 or len(set(quantization_devices)) != len(
        quantization_devices
    ):
        raise ValueError(
            "multi-GPU qualification devices must be unique visible ordinals beginning with 0"
        )
    visible_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if visible_devices <= max(quantization_devices):
        raise RuntimeError(
            "multi-GPU qualification requested visible CUDA ordinals "
            f"{quantization_devices}, but only {visible_devices} devices are visible"
        )
    if device_ratios is not None and len(device_ratios) != len(quantization_devices):
        raise ValueError("multi-GPU qualification requires one ratio per device")
    if any(
        len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0
        for shape in production_projection_shapes
    ):
        raise ValueError("production projection qualification shapes must be positive rank-2")

    primary = torch.device("cuda:0")
    generator = torch.Generator(device=primary).manual_seed(seed)
    source_batches = [
        [
            torch.randn(
                (128, 256),
                generator=generator,
                dtype=torch.float32,
                device=primary,
            )
            for _ in range(4)
        ]
        for _ in range(4)
    ]
    debug_dir = Path(debug_dir) / "multigpu-qualification"

    def run_repeated_batches(devices: list[int], ratios: list[int] | None):
        shared_hessian = analytic_isotropic_hessian(
            features=128,
            equivalent_rows=512,
            key="ds4rt.multigpu_repeated_batch_qualification.w1",
            device=primary,
        )
        batches = []
        for batch_index, source_weights in enumerate(source_batches):
            args = [
                {
                    "seed": seed + 10 * batch_index + tensor_index,
                    "K": EXL3_BITS,
                    "devices": devices,
                    "device_ratios": ratios,
                    "apply_out_scales": True,
                    "debug_dir": str(debug_dir),
                    "mcg": True,
                }
                for tensor_index in range(len(source_weights))
            ]
            results = quantize_exl3_batch(
                [weight.clone() for weight in source_weights],
                [shared_hessian] * len(source_weights),
                args,
            )
            for device_index in devices:
                torch.cuda.synchronize(device_index)
            batches.append(
                (
                    [float(arg["g_scale"]) for arg in args],
                    [float(proxy_error) for proxy_error, _ in results],
                    [output for _, output in results],
                )
            )
        return batches

    def require_equal(single_batch, dual_batch, *, label: str) -> None:
        if single_batch[0] != dual_batch[0]:
            raise RuntimeError(
                f"multi-GPU EXL3 scale mismatch in {label}: "
                f"single={single_batch[0]} multi={dual_batch[0]}"
            )
        if single_batch[1] != dual_batch[1]:
            raise RuntimeError(
                f"multi-GPU EXL3 proxy mismatch in {label}: "
                f"single={single_batch[1]} multi={dual_batch[1]}"
            )
        for tensor_index, (single_output, dual_output) in enumerate(
            zip(single_batch[2], dual_batch[2])
        ):
            if single_output.keys() != dual_output.keys():
                raise RuntimeError(
                    f"multi-GPU EXL3 output fields differ in {label} tensor {tensor_index}"
                )
            for field in single_output:
                if not torch.equal(single_output[field], dual_output[field]):
                    raise RuntimeError(
                        f"multi-GPU EXL3 tensor mismatch in {label} "
                        f"tensor {tensor_index} field {field}"
                    )

    single = run_repeated_batches([0], None)
    dual = run_repeated_batches(
        list(quantization_devices),
        list(device_ratios) if device_ratios else None,
    )
    for batch_index, (single_batch, dual_batch) in enumerate(zip(single, dual)):
        require_equal(
            single_batch,
            dual_batch,
            label=f"repeated qualification batch {batch_index}",
        )

    def run_production_projection(
        shape: tuple[int, int],
        *,
        shape_index: int,
        devices: list[int],
        ratios: list[int] | None,
    ):
        generator = torch.Generator(device=primary).manual_seed(
            seed + 100_000 + shape_index
        )
        source_weight = torch.randn(
            shape,
            generator=generator,
            dtype=torch.float32,
            device=primary,
        )
        hessian = analytic_isotropic_hessian(
            features=shape[0],
            equivalent_rows=512,
            key=f"ds4rt.multigpu_production_shape_qualification.{shape_index}",
            device=primary,
        )
        args = {
            "seed": seed + 100_000 + shape_index,
            "K": EXL3_BITS,
            "devices": devices,
            "device_ratios": ratios,
            "apply_out_scales": True,
            "debug_dir": str(debug_dir),
            "mcg": True,
        }
        results = quantize_exl3_batch(
            [source_weight],
            [hessian],
            [args],
        )
        for device_index in devices:
            torch.cuda.synchronize(device_index)
        return (
            [float(args["g_scale"])],
            [float(results[0][0])],
            [results[0][1]],
        )

    dual_devices = list(quantization_devices)
    dual_ratios = list(device_ratios) if device_ratios else None
    for shape_index, shape in enumerate(production_projection_shapes):
        single_projection = run_production_projection(
            shape,
            shape_index=shape_index,
            devices=[0],
            ratios=None,
        )
        dual_projection = run_production_projection(
            shape,
            shape_index=shape_index,
            devices=dual_devices,
            ratios=dual_ratios,
        )
        require_equal(
            single_projection,
            dual_projection,
            label=f"production-shape qualification {shape_index} shape={shape}",
        )
    for device_index in quantization_devices:
        with torch.cuda.device(device_index):
            torch.cuda.empty_cache()
    return {
        "schema": "ds4rt-exl3-multigpu-qualification-v1",
        "status": "bit-exact",
        "devices": list(quantization_devices),
        "device_ratios": list(device_ratios) if device_ratios else None,
        "batches": len(source_batches),
        "tensors_per_batch": len(source_batches[0]),
        "shape": [128, 256],
        "production_projection_shapes": [
            list(shape) for shape in production_projection_shapes
        ],
        "seed": seed,
    }


def quantization_seed(seed: int, tensor_name: str) -> int:
    digest = hashlib.sha256(tensor_name.encode()).digest()
    return (int(seed) + int.from_bytes(digest[:8], "little")) % (2**63 - 1)


def load_native_projection(
    snapshot: Path,
    source_index: dict[str, SourceTensor],
    base: str,
    *,
    device: str = "cuda:0",
) -> Any:
    from safetensors import safe_open

    weight_name = f"{base}.weight"
    scale_name = f"{base}.scale"
    weight_source = source_index[weight_name]
    scale_source = source_index[scale_name]
    with safe_open(weight_source.source_file, framework="pt", device=device) as shard:
        packed = shard.get_tensor(weight_name)
    with safe_open(scale_source.source_file, framework="pt", device=device) as shard:
        scales = shard.get_tensor(scale_name)
    return dequantize_native_fp4_projection(packed, scales)


def forced_expert_hidden_rows(
    samples: Any,
    *,
    rows: int,
    layer_id: int,
    expert_id: int,
    seed: int,
    device: Any,
) -> Any:
    """Gather the bounded natural layer inputs assigned to one forced expert."""

    indices = deterministic_expert_row_indices(
        total_rows=int(samples.shape[0]),
        rows=rows,
        layer_id=layer_id,
        expert_id=expert_id,
        seed=seed,
        device="cpu",
    )
    selected = samples.index_select(0, indices)
    return selected.to(device=device, dtype=_torch().float32, non_blocking=False)


def forced_expert_down_activations(
    samples: Any,
    gate_weight: Any,
    up_weight: Any,
    *,
    rows: int,
    layer_id: int,
    expert_id: int,
    seed: int,
    swiglu_limit: float,
    device: Any,
) -> Any:
    hidden = forced_expert_hidden_rows(
        samples,
        rows=rows,
        layer_id=layer_id,
        expert_id=expert_id,
        seed=seed,
        device=device,
    )
    gate = hidden @ gate_weight
    up = hidden @ up_weight
    del hidden
    return deepseek_v4_swiglu_activation(gate, up, limit=swiglu_limit)


def build_forced_down_pooled_covariance(
    plan: ArtifactPlan,
    source_index: dict[str, SourceTensor],
    samples: Any,
    *,
    layer_id: int,
    rows_per_expert: int,
    seed: int,
    batch_experts: int,
    device: Any,
) -> Any:
    """Build an equal-expert down-input covariance from native expert outputs."""

    torch = _torch()
    if rows_per_expert > int(samples.shape[0]):
        raise ValueError(
            f"forced down rows {rows_per_expert} exceed layer {layer_id} capture "
            f"rows {samples.shape[0]}"
        )
    pooled_sum = torch.zeros(
        (plan.shape.intermediate_size, plan.shape.intermediate_size),
        dtype=torch.float32,
        device=device,
    )
    for expert_start in range(0, plan.shape.experts, batch_experts):
        expert_ids = range(
            expert_start,
            min(plan.shape.experts, expert_start + batch_experts),
        )
        for expert_id in expert_ids:
            gate = load_native_projection(
                plan.snapshot,
                source_index,
                expert_projection_base(plan.shape, layer_id, expert_id, "w1"),
                device=str(device),
            )
            up = load_native_projection(
                plan.snapshot,
                source_index,
                expert_projection_base(plan.shape, layer_id, expert_id, "w3"),
                device=str(device),
            )
            activations = forced_expert_down_activations(
                samples,
                gate,
                up,
                rows=rows_per_expert,
                layer_id=layer_id,
                expert_id=expert_id,
                seed=seed,
                swiglu_limit=plan.shape.swiglu_limit,
                device=device,
            )
            pooled_sum.addmm_(activations.T, activations)
            del gate, up, activations
        torch.cuda.empty_cache()
    pooled_sum.div_(plan.shape.experts * rows_per_expert)
    if not bool(torch.isfinite(pooled_sum).all().item()):
        raise ValueError(f"forced down pooled covariance is non-finite at layer {layer_id}")
    return pooled_sum


def run_layerwise_quantization(
    plan: ArtifactPlan,
    writer: SafetensorsArtifactWriter,
    *,
    quantize_exl3_batch: Callable[..., Any],
    calibration_rows: int,
    seed: int,
    batch_experts: int,
    debug_dir: str | Path,
    stop_after_layer: int | None = None,
    quantization_devices: tuple[int, ...] = (0,),
    device_ratios: tuple[int, ...] | None = None,
    multigpu_qualification: dict[str, Any] | None = None,
    activation_corpus: ActivationCorpus | None = None,
    forced_down_rows_per_expert: int = 512,
    forced_down_expert_weight: float = 0.25,
    only_layer: int | None = None,
) -> dict[str, Any]:
    torch = _torch()
    if batch_experts <= 0:
        raise ValueError("batch_experts must be positive")
    if activation_corpus is not None:
        if activation_corpus.checkpoint != plan.snapshot.resolve():
            raise ValueError("activation corpus is not bound to the quantization plan")
        if forced_down_rows_per_expert <= 0:
            raise ValueError("forced down rows per expert must be positive")
        if not 0.0 <= forced_down_expert_weight <= 1.0 or not math.isfinite(
            forced_down_expert_weight
        ):
            raise ValueError("forced down expert weight must be finite and in 0..1")
    if only_layer is not None and not 0 <= only_layer < plan.shape.total_layers:
        raise ValueError(
            f"only layer {only_layer} is outside 0..{plan.shape.total_layers}"
        )
    if (
        not quantization_devices
        or quantization_devices[0] != 0
        or len(set(quantization_devices)) != len(quantization_devices)
        or any(index < 0 for index in quantization_devices)
    ):
        raise ValueError(
            "quantization_devices must be unique visible CUDA ordinals beginning with 0"
        )
    if device_ratios is not None and (
        len(device_ratios) != len(quantization_devices)
        or any(ratio <= 0 for ratio in device_ratios)
    ):
        raise ValueError("device_ratios must provide one positive value per device")
    visible_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if visible_devices <= max(quantization_devices):
        raise RuntimeError(
            "EXL3 conversion requested visible CUDA ordinals "
            f"{quantization_devices}, but only {visible_devices} devices are visible"
        )
    device = torch.device("cuda:0")
    quantizer_devices = list(quantization_devices)
    quantizer_device_ratios = list(device_ratios) if device_ratios is not None else None
    source_index = read_source_index(plan.snapshot)
    if activation_corpus is None:
        calibration_contract = {
            "hessian": "analytic_identity",
            "distribution": "rms_normalized_isotropic",
        }
    else:
        calibration_contract = {
            "hessian": (
                "per_expert_natural_route_gate_squared_covariance"
                if activation_corpus.route_aware
                else "native_sample_covariance_with_forced_down_shrinkage"
            ),
            "distribution": "checkpoint_bound_native_expert_inputs",
            "activation_manifest": str(activation_corpus.manifest_path),
            "activation_corpus_sha256": activation_corpus.corpus_sha256,
            "activation_checkpoint": str(activation_corpus.checkpoint),
            "activation_base_layers": activation_corpus.layer_count,
            "activation_prompts": activation_corpus.prompt_count,
            **(
                {
                    "natural_routing": True,
                    "forced_expert_activation": False,
                    "route_gate_weighting": "squared_unit_rms",
                    "minimum_natural_routes_per_expert": (
                        activation_corpus.minimum_natural_routes_per_expert
                    ),
                    "route_replay_report": str(
                        activation_corpus.route_replay_report_path
                    ),
                    "route_replay_report_sha256": (
                        activation_corpus.route_replay_report_sha256
                    ),
                }
                if activation_corpus.route_aware
                else {
                    "forced_down_rows_per_expert": forced_down_rows_per_expert,
                    "forced_down_expert_weight": forced_down_expert_weight,
                    "forced_down_pool_weight": 1.0 - forced_down_expert_weight,
                }
            ),
            "mtp_calibration": "analytic_identity_pilot_only",
        }
    calibration_contract_sha256 = hashlib.sha256(
        json.dumps(calibration_contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report: dict[str, Any] = calibration_report_for_run(
        writer,
        {
            "schema": "ds4rt-exl3-calibration-report-v1",
            "recipe": writer.recipe,
            "source_snapshot": str(plan.snapshot),
            "calibration_rows": calibration_rows,
            "seed": seed,
            **calibration_contract,
            "calibration_contract_sha256": calibration_contract_sha256,
            "activation": "silu",
            "swiglu_limit": plan.shape.swiglu_limit,
            **(
                {"multigpu_qualification": multigpu_qualification}
                if multigpu_qualification is not None
                else {}
            ),
            "layers": [],
        },
    )
    completed_layers = {int(layer["layer_id"]) for layer in report["layers"]}
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    target_layers = (
        range(plan.shape.total_layers) if only_layer is None else (only_layer,)
    )
    for layer_id in target_layers:
        if layer_id in completed_layers:
            continue
        layer_started = time.monotonic()
        layer_samples = None
        routed_layer = None
        pooled_down_covariance = None
        activation_calibrated = (
            activation_corpus is not None and layer_id < plan.shape.hidden_layers
        )
        if activation_calibrated:
            assert activation_corpus is not None
            if activation_corpus.route_aware:
                routed_layer = load_routed_activation_layer(
                    activation_corpus,
                    layer_id,
                )
                layer_samples = routed_layer.samples
                shared_h = None
            else:
                layer_samples = load_activation_layer_samples(
                    activation_corpus, layer_id
                )
                shared_h = accumulate_activation_hessian(
                    layer_samples,
                    key=f"{block_prefix(plan.shape, layer_id)}.ffn.experts.*.w1",
                    device=device,
                )
                pooled_down_covariance = build_forced_down_pooled_covariance(
                    plan,
                    source_index,
                    layer_samples,
                    layer_id=layer_id,
                    rows_per_expert=forced_down_rows_per_expert,
                    seed=seed,
                    batch_experts=batch_experts,
                    device=device,
                )
            shared_down_h = None
        else:
            shared_h = analytic_isotropic_hessian(
                features=plan.shape.hidden_size,
                equivalent_rows=calibration_rows,
                key=f"{block_prefix(plan.shape, layer_id)}.ffn.experts.*.w1",
                device=device,
            )
            shared_down_h = analytic_isotropic_hessian(
                features=plan.shape.intermediate_size,
                equivalent_rows=calibration_rows,
                key=f"{block_prefix(plan.shape, layer_id)}.ffn.experts.*.w2",
                device=device,
            )
        layer_report = {
            "layer_id": layer_id,
            "quantization_devices": quantizer_devices,
            "device_ratios": quantizer_device_ratios,
            "calibration_mode": (
                "natural_route_per_expert_exact_swiglu"
                if routed_layer is not None
                else "native_input_forced_balanced_down"
                if activation_calibrated
                else "analytic_identity_mtp_pilot"
                if activation_corpus is not None
                else "analytic_identity"
            ),
            **(
                {
                    "input_activation_rows": int(layer_samples.shape[0]),
                    **(
                        {
                            "natural_routing": True,
                            "route_gate_weighting": "squared_unit_rms",
                        }
                        if routed_layer is not None
                        else {
                            "forced_down_rows_per_expert": (
                                forced_down_rows_per_expert
                            ),
                            "forced_down_total_rows": (
                                forced_down_rows_per_expert * plan.shape.experts
                            ),
                            "forced_down_expert_weight": (
                                forced_down_expert_weight
                            ),
                        }
                    ),
                }
                if activation_calibrated
                else {}
            ),
            "projections": [],
        }
        if routed_layer is not None:
            layer_report["expert_natural_route_rows"] = []
        for expert_start in range(0, plan.shape.experts, batch_experts):
            expert_ids = tuple(
                range(expert_start, min(plan.shape.experts, expert_start + batch_experts))
            )
            group_bases = [
                expert_projection_base(plan.shape, layer_id, expert_id, stem)
                for expert_id in expert_ids
                for stem in ("w1", "w2", "w3")
            ]
            if writer.generated_group_complete(group_bases):
                continue
            gate_weights = []
            up_weights = []
            down_weights = []
            for expert_id in expert_ids:
                gate = load_native_projection(
                    plan.snapshot,
                    source_index,
                    expert_projection_base(plan.shape, layer_id, expert_id, "w1"),
                )
                up = load_native_projection(
                    plan.snapshot,
                    source_index,
                    expert_projection_base(plan.shape, layer_id, expert_id, "w3"),
                )
                down = load_native_projection(
                    plan.snapshot,
                    source_index,
                    expert_projection_base(plan.shape, layer_id, expert_id, "w2"),
                )
                gate_weights.append(gate)
                up_weights.append(up)
                down_weights.append(down)
            fc1_weights = [value for pair in zip(gate_weights, up_weights) for value in pair]
            fc1_bases = [
                expert_projection_base(plan.shape, layer_id, expert_id, stem)
                for expert_id in expert_ids
                for stem in ("w1", "w3")
            ]
            routed_inputs: list[tuple[Any, Any]] | None = None
            if routed_layer is not None:
                routed_inputs = [
                    routed_expert_activation_rows(
                        routed_layer,
                        expert_id=expert_id,
                    )
                    for expert_id in expert_ids
                ]
                expert_input_hessians = [
                    accumulate_routed_activation_hessian(
                        samples,
                        gates,
                        key=expert_projection_base(
                            plan.shape, layer_id, expert_id, "w1"
                        ),
                        device=device,
                    )
                    for expert_id, (samples, gates) in zip(
                        expert_ids, routed_inputs
                    )
                ]
                fc1_hessians = [
                    hessian
                    for hessian in expert_input_hessians
                    for _ in ("w1", "w3")
                ]
                layer_report["expert_natural_route_rows"].extend(
                    {
                        "expert_id": expert_id,
                        "rows": int(samples.shape[0]),
                        "gate_min": float(gates.min().item()),
                        "gate_mean": float(gates.mean().item()),
                        "gate_max": float(gates.max().item()),
                    }
                    for expert_id, (samples, gates) in zip(
                        expert_ids, routed_inputs
                    )
                )
            else:
                assert shared_h is not None
                fc1_hessians = [shared_h] * len(fc1_weights)
            if routed_inputs is not None:
                down_hessians = [
                    routed_expert_down_hessian(
                        samples,
                        route_gates,
                        gate,
                        up,
                        swiglu_limit=plan.shape.swiglu_limit,
                        key=expert_projection_base(
                            plan.shape, layer_id, expert_id, "w2"
                        ),
                        device=device,
                    )
                    for expert_id, (samples, route_gates), gate, up in zip(
                        expert_ids,
                        routed_inputs,
                        gate_weights,
                        up_weights,
                    )
                ]
            elif activation_calibrated:
                assert layer_samples is not None
                assert pooled_down_covariance is not None
                down_hessians = []
                for expert_id, gate, up in zip(expert_ids, gate_weights, up_weights):
                    down_activations = forced_expert_down_activations(
                        layer_samples,
                        gate,
                        up,
                        rows=forced_down_rows_per_expert,
                        layer_id=layer_id,
                        expert_id=expert_id,
                        seed=seed,
                        swiglu_limit=plan.shape.swiglu_limit,
                        device=device,
                    )
                    down_hessians.append(
                        blended_forced_down_hessian(
                            down_activations,
                            pooled_covariance=pooled_down_covariance,
                            expert_weight=forced_down_expert_weight,
                            key=expert_projection_base(
                                plan.shape, layer_id, expert_id, "w2"
                            ),
                        )
                    )
                    del down_activations
            else:
                assert shared_down_h is not None
                down_hessians = [shared_down_h] * len(down_weights)
            all_weights = [fc1_weights, down_weights]
            all_hessians = [fc1_hessians, down_hessians]
            all_bases = [
                fc1_bases,
                [
                    expert_projection_base(plan.shape, layer_id, expert_id, "w2")
                    for expert_id in expert_ids
                ],
            ]
            for weights, hessians, bases in zip(all_weights, all_hessians, all_bases):
                quant_args = [
                    {
                        "seed": quantization_seed(seed, base),
                        "K": plan.exl3_bits,
                        "devices": quantizer_devices,
                        "device_ratios": quantizer_device_ratios,
                        "apply_out_scales": True,
                        "debug_dir": str(debug_dir),
                        "mcg": True,
                    }
                    for base in bases
                ]
                results = quantize_exl3_batch(weights, hessians, quant_args)
                diagnostics = []
                for base, quant_arg, (proxy_error, _) in zip(
                    bases, quant_args, results
                ):
                    if quant_arg.get("q_fallback"):
                        raise RuntimeError(f"calibration Hessian fell back for {base}")
                    proxy_error = float(proxy_error)
                    g_scale = float(quant_arg["g_scale"])
                    validate_projection_quantization_diagnostic(
                        name=base,
                        proxy_error=proxy_error,
                        g_scale=g_scale,
                    )
                    diagnostics.append((proxy_error, g_scale))
                for base, quant_arg, (_, output_tensors), diagnostic in zip(
                    bases, quant_args, results, diagnostics
                ):
                    proxy_error, g_scale = diagnostic
                    for suffix in ("trellis", "suh", "svh", "mcg"):
                        writer.write_generated_tensor(
                            f"{base}.{suffix}", output_tensors[suffix]
                        )
                    layer_report["projections"].append(
                        {
                            "name": base,
                            "proxy_error": proxy_error,
                            "g_scale": g_scale,
                            "trellis_bits": int(quant_arg["K"]),
                        }
                    )
            del (
                gate_weights,
                up_weights,
                down_weights,
                down_hessians,
                fc1_hessians,
                all_hessians,
                all_weights,
            )
            if routed_inputs is not None:
                del expert_input_hessians, routed_inputs
            for device_index in quantization_devices:
                with torch.cuda.device(device_index):
                    torch.cuda.empty_cache()
        report["layers"].append(layer_report)
        layer_report["elapsed_seconds"] = time.monotonic() - layer_started
        report["progress"] = quantization_progress(
            report["layers"], total_layers=plan.shape.total_layers
        )
        writer.checkpoint_report(report)
        print(
            json.dumps(
                {
                    "event": "exl3-layer-complete",
                    "layer_id": layer_id,
                    "elapsed_seconds": layer_report["elapsed_seconds"],
                    "quantization_devices": quantizer_devices,
                    **report["progress"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del (
            shared_h,
            shared_down_h,
            pooled_down_covariance,
            layer_samples,
            routed_layer,
        )
        for device_index in quantization_devices:
            with torch.cuda.device(device_index):
                torch.cuda.empty_cache()
        if stop_after_layer is not None and layer_id >= stop_after_layer:
            report["incomplete"] = True
            return report
    if only_layer is not None:
        report["incomplete"] = True
    return report


def plan_summary(plan: ArtifactPlan) -> dict[str, Any]:
    return {
        "source_snapshot": str(plan.snapshot),
        "hidden_size": plan.shape.hidden_size,
        "intermediate_size": plan.shape.intermediate_size,
        "layers": plan.shape.total_layers,
        "experts": plan.shape.experts,
        "strict_expert_tp": EXPERT_TP_WORLD_SIZE,
        "local_intermediate_size": plan.shape.intermediate_size // EXPERT_TP_WORLD_SIZE,
        "recipe": EXL3_RECIPE,
        "trellis_bits": plan.exl3_bits,
        "output_tensors": len(plan.tensors),
        "generated_exl3_tensors": len(plan.generated_tensors),
        "shards": len(plan.shard_tensors),
        "quantization_scope": "routed_experts_only",
        "expert_tensor_layout": plan.expert_tensor_layout,
        "retained_native_tensor_bytes": plan.source_bytes,
        "routed_expert_exl3_bytes": plan.trellis_bytes,
        "source_bytes": plan.source_bytes,
        "trellis_bytes": plan.trellis_bytes,
        "total_size": plan.total_size,
    }


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment preflight
        raise RuntimeError("EXL3 conversion requires the torch runtime extra") from exc
    return torch
