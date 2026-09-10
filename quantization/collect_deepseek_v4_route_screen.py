#!/usr/bin/env python3
"""Collect a Pro-native base-router screen through GPTQModel's lazy shell."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import threading
from typing import Any, Callable

import torch

from gptqmodel import GPTQModel
from gptqmodel.looper.module_looper import StopMainLoop
from gptqmodel.models.definitions.deepseek_v4 import DeepSeekV4QModel
from gptqmodel.quantization import AutoModuleDecoderConfig, EXL3Config


SCHEMA = "ds41rt-flash-natural-route-distribution-v1"
PROGRESS_SCHEMA = "ds41rt-deepseek-v4-base-route-screen-progress-v1"


class RouteScreenError(RuntimeError):
    """The source, corpus, or lazy route replay violated its contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(value, target, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _load_corpus(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    texts: list[str] = []
    prompts: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RouteScreenError(
                    f"invalid corpus JSON at line {line_number}"
                ) from error
            identifier = record.get("id") if isinstance(record, dict) else None
            prompt = record.get("prompt") if isinstance(record, dict) else None
            if (
                not isinstance(identifier, str)
                or not identifier
                or identifier in seen
                or not isinstance(prompt, str)
                or not prompt
            ):
                raise RouteScreenError(
                    f"invalid or duplicate corpus record at line {line_number}"
                )
            seen.add(identifier)
            texts.append(prompt)
            prompts.append(
                {
                    "schedule_index": len(prompts),
                    "split": "screening",
                    "corpus_index": len(prompts),
                    "id": identifier,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "mtp_verify_cycles": 0,
                }
            )
    if not texts:
        raise RouteScreenError("screening corpus is empty")
    return texts, prompts


def _layer_summary(
    layer_id: int, counts: list[int], rows: int, *, top_k: int
) -> dict[str, Any]:
    total = sum(counts)
    if rows <= 0 or total != rows * top_k:
        raise RouteScreenError(f"layer {layer_id} route totals are invalid")
    ordered = sorted(counts)
    mean = total / len(counts)
    probabilities = [count / total for count in counts if count]
    entropy = -sum(value * math.log(value) for value in probabilities)
    return {
        "layer_id": layer_id,
        "rows": rows,
        "routes": total,
        "counts": counts,
        "zero_hit_experts": sum(count == 0 for count in counts),
        "under_32_hit_experts": sum(count < 32 for count in counts),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[math.ceil(0.95 * len(counts)) - 1],
        "max": ordered[-1],
        "mean": mean,
        "max_to_mean": ordered[-1] / mean,
        "normalized_entropy": entropy / math.log(len(counts)),
    }


class _RouteCollector:
    def __init__(
        self,
        *,
        layers: int,
        experts: int,
        top_k: int,
        progress_path: Path,
        identity: dict[str, Any],
        stop_after_layer: int | None,
        prune_completed_layer: Callable[[int], None],
    ) -> None:
        self.counts = [[0] * experts for _ in range(layers)]
        self.rows = [0] * layers
        self.top_k = top_k
        self.progress_path = progress_path
        self.identity = identity
        self.stop_after_layer = stop_after_layer
        self.prune_completed_layer = prune_completed_layer
        self._locks = [threading.Lock() for _ in range(layers)]

    def hook(self, layer_index: int):
        def collect(_module, _inputs, output) -> None:
            if not isinstance(output, (tuple, list)) or len(output) != 3:
                raise RouteScreenError("router did not return logits, weights, indices")
            indices = output[2]
            if (
                not isinstance(indices, torch.Tensor)
                or indices.ndim != 2
                or indices.shape[1] != self.top_k
            ):
                raise RouteScreenError("router returned invalid top-k geometry")
            flat = indices.detach().reshape(-1).to(device="cpu", dtype=torch.int64)
            counts = torch.bincount(flat, minlength=len(self.counts[layer_index]))
            if counts.numel() != len(self.counts[layer_index]):
                raise RouteScreenError("router emitted an invalid expert id")
            with self._locks[layer_index]:
                self.rows[layer_index] += int(indices.shape[0])
                target = self.counts[layer_index]
                for expert, value in enumerate(counts.tolist()):
                    target[expert] += int(value)

        return collect

    def layer_complete(self, *, layer_idx: int, submodule_finalized: bool):
        if not submodule_finalized:
            return None
        if self.rows[layer_idx] <= 0:
            raise RouteScreenError(f"layer {layer_idx} produced no router rows")
        # The source layer is no longer needed once its native replay is
        # finalized. Keep this screen's NVMe use rolling just like the
        # production layer-boundary controller instead of retaining a second
        # copy of the complete checkpoint.
        self.prune_completed_layer(layer_idx)
        _atomic_json(
            self.progress_path,
            {
                "schema": PROGRESS_SCHEMA,
                "identity": self.identity,
                "completed_through": layer_idx,
                "rows": self.rows[: layer_idx + 1],
                "counts": self.counts[: layer_idx + 1],
            },
        )
        print(
            json.dumps(
                {
                    "event": "route-screen-layer-complete",
                    "layer": layer_idx,
                    "rows": self.rows[layer_idx],
                    "min_routes": min(self.counts[layer_idx]),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if self.stop_after_layer == layer_idx:
            return StopMainLoop
        return None


def collect(args: argparse.Namespace) -> dict[str, Any]:
    snapshot = args.snapshot.expanduser().resolve(strict=True)
    corpus = args.corpus.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    progress = output.with_suffix(".progress.json")
    csv_path = output.with_suffix(".csv")
    if any(path.exists() or path.is_symlink() for path in (output, csv_path, progress)):
        raise RouteScreenError("route-screen output already exists")
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    layers = int(config["num_hidden_layers"])
    experts = int(config["n_routed_experts"])
    top_k = int(config["num_experts_per_tok"])
    if layers <= 0 or experts <= 0 or top_k != 6:
        raise RouteScreenError("unsupported DeepSeek V4 routing geometry")
    texts, prompts = _load_corpus(corpus)
    identity = {
        "checkpoint": os.fspath(snapshot),
        "checkpoint_revision": snapshot.name,
        "config_sha256": _sha256_file(snapshot / "config.json"),
        "index_sha256": _sha256_file(snapshot / "model.safetensors.index.json"),
        "corpus": os.fspath(corpus),
        "corpus_sha256": _sha256_file(corpus),
        "prompts": len(prompts),
        "layers": layers,
        "experts": experts,
        "top_k": top_k,
    }
    devices = [f"cuda:{index}" for index in range(args.gpus)]
    qcfg = EXL3Config(
        bits=2.0,
        codebook="mcg",
        out_scales="auto",
        module_include=[r"(?!)"],
        preprocessors=[AutoModuleDecoderConfig(target_dtype=torch.bfloat16)],
        fallback=None,
        offload_to_disk=True,
        offload_to_disk_path=os.fspath(args.offload_dir.expanduser().resolve()),
        device=devices[0],
        calibration_data_device="cpu",
        dense_vram_strategy_devices=[devices[0]],
        moe_vram_strategy="balanced",
        moe_vram_strategy_devices=devices,
    )
    model = GPTQModel.load(
        os.fspath(snapshot), quantize_config=qcfg, trust_remote_code=False
    )
    if not isinstance(model, DeepSeekV4QModel):
        raise RouteScreenError(
            f"unexpected GPTQModel definition: {type(model).__name__}"
        )
    turtle = getattr(model, "turtle_model", None)
    configure_staging = getattr(turtle, "configure_active_source_staging", None)
    if not callable(configure_staging):
        raise RouteScreenError("lazy source cannot stage active layers")
    configure_staging(
        os.fspath(args.active_layer_source_dir.expanduser().resolve()),
        provenance=identity,
    )
    prune_source = getattr(turtle, "prune_active_source_scope_through", None)
    if not callable(prune_source):
        raise RouteScreenError("lazy source cannot prune completed base layers")
    collector = _RouteCollector(
        layers=layers,
        experts=experts,
        top_k=top_k,
        progress_path=progress,
        identity=identity,
        stop_after_layer=args.stop_after_layer,
        prune_completed_layer=lambda layer_index: prune_source("base", layer_index),
    )
    target_layers = tuple(model.model.model.layers)
    if len(target_layers) != layers:
        raise RouteScreenError("lazy model layer count differs from source config")
    handles = [
        layer.mlp.gate.register_forward_hook(collector.hook(layer_index))
        for layer_index, layer in enumerate(target_layers)
    ]
    model.layer_callback = collector
    try:
        model.quantize(texts, batch_size=1, calibration_sort=None)
    finally:
        for handle in handles:
            handle.remove()
    if args.stop_after_layer is not None:
        return {
            "schema": PROGRESS_SCHEMA,
            "status": "probe-complete",
            "completed_through": args.stop_after_layer,
            "identity": identity,
        }
    layer_reports = [
        _layer_summary(layer, collector.counts[layer], collector.rows[layer], top_k=top_k)
        for layer in range(layers)
    ]
    global_counts = [
        sum(collector.counts[layer][expert] for layer in range(layers))
        for expert in range(experts)
    ]
    distribution = {
        "rows": sum(collector.rows),
        "routes": sum(sum(counts) for counts in collector.counts),
        "layers": layer_reports,
        "global_expert_counts": global_counts,
    }
    report = {
        "schema": SCHEMA,
        "scope": "base-only-mtp-deferred",
        "checkpoint": os.fspath(snapshot),
        "model": config.get("_name_or_path", "deepseek-ai/DeepSeek-V4-Pro-0813"),
        "top_k": top_k,
        "experts": experts,
        "layers": layers,
        "corpora": {
            "screening": {
                "path": os.fspath(corpus),
                "sha256": identity["corpus_sha256"],
                "prompts": len(prompts),
            }
        },
        "prompts": prompts,
        "cohort_rows_by_split": {"screening": {"base": sum(collector.rows)}},
        "distributions": {"screening": distribution, "combined": distribution},
    }
    _atomic_json(output, report)
    with csv_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("split", "layer", "expert", "count", "fraction"))
        for layer in layer_reports:
            for expert, count in enumerate(layer["counts"]):
                writer.writerow(
                    (
                        "screening",
                        layer["layer_id"],
                        expert,
                        count,
                        count / layer["routes"],
                    )
                )
    progress.unlink()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--active-layer-source-dir", type=Path, required=True)
    parser.add_argument("--offload-dir", type=Path, required=True)
    parser.add_argument("--gpus", type=int, choices=(1, 2), default=2)
    parser.add_argument("--stop-after-layer", type=int)
    args = parser.parse_args()
    if args.stop_after_layer is not None and args.stop_after_layer < 0:
        parser.error("--stop-after-layer must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    report = collect(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RouteScreenError as error:
        print(f"collect-deepseek-v4-route-screen: {error}", file=__import__("sys").stderr)
        raise SystemExit(2) from error
