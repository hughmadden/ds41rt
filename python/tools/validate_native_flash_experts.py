#!/usr/bin/env python3
"""Validate native Flash experts with the production TP=4 Spark contract."""

from __future__ import annotations

import _pinned_sparkinfer

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ds4rt_runtime.native_experts import (
    EXPERT_TP_WORLD_SIZE,
    NATIVE_SOURCE_FORMAT,
    load_native_expert_reference_layer,
    load_native_expert_tp_layer,
)


DEFAULT_SNAPSHOT = Path(
    "/home/tj/.cache/huggingface/hub/"
    "models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/"
    "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
)


def _run_layer(layer, hidden, topk_ids, topk_weights, *, warmup, iterations):
    plan, scratch = layer.plan_tp(max_tokens=int(hidden.shape[0]))
    output = torch.empty_like(hidden)

    def launch():
        return layer.run_partial(
            hidden,
            topk_ids,
            topk_weights,
            plan=plan,
            scratch=scratch,
            output=output,
        )

    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        launch()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("native Flash expert output contains non-finite values")
    return output.float().clone(), elapsed * 1_000.0 / iterations


def _release_cuda_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--experts", default="0,1,2,3,4,5")
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise SystemExit("set CUDA_VISIBLE_DEVICES=0; exactly one GPU must be visible")
    if args.iterations <= 0 or args.warmup < 0 or args.tokens <= 0:
        raise SystemExit("tokens and iterations must be positive; warmup cannot be negative")

    config_json = json.loads(
        (args.snapshot / "config.json").read_text(encoding="utf-8")
    )
    expert_ids = (
        tuple(range(int(config_json["n_routed_experts"])))
        if args.experts == "all"
        else tuple(int(value) for value in args.experts.split(",") if value)
    )
    if len(expert_ids) < 6:
        raise SystemExit("validation needs at least six experts for DeepSeek V4 top-k 6")

    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(20260804)
    hidden_size = config_json["hidden_size"]
    hidden = (
        torch.randn(
            args.tokens,
            int(hidden_size),
            generator=generator,
            device=device,
        )
        * 0.25
    ).to(torch.bfloat16)
    route_base = torch.tensor(expert_ids, dtype=torch.int32, device=device)
    topk_ids = torch.stack(
        [
            route_base[
                (torch.arange(args.tokens, device=device) + offset)
                % len(expert_ids)
            ]
            for offset in range(6)
        ],
        dim=1,
    ).to(torch.int32)
    topk_weights = torch.softmax(
        torch.randn(
            args.tokens,
            6,
            generator=generator,
            device=device,
        ),
        dim=-1,
    ).to(torch.float32)

    tp_sum = torch.zeros_like(hidden, dtype=torch.float32)
    rank_reports = []
    for tp_rank in range(EXPERT_TP_WORLD_SIZE):
        layer = load_native_expert_tp_layer(
            args.snapshot,
            args.layer,
            tp_rank=tp_rank,
            expert_ids=expert_ids,
        )
        partial, mean_ms = _run_layer(
            layer,
            hidden,
            topk_ids,
            topk_weights,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        tp_sum.add_(partial)
        rank_reports.append(
            {
                "rank": tp_rank,
                "intermediate_start": layer.tp_slice.start,
                "intermediate_stop": layer.tp_slice.stop,
                "source_bytes": layer.source_bytes,
                "load_seconds": layer.load_seconds,
                "mean_kernel_ms": mean_ms,
                "partial_l2": float(partial.norm().item()),
            }
        )
        del partial
        del layer
        _release_cuda_cache()

    if len({report["source_bytes"] for report in rank_reports}) != 1:
        raise RuntimeError("TP ranks do not have stable equal source-memory demand")

    reference = load_native_expert_reference_layer(
        args.snapshot,
        args.layer,
        expert_ids,
    )
    reference_output, reference_ms = _run_layer(
        reference,
        hidden,
        topk_ids,
        topk_weights,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    difference = tp_sum - reference_output
    cosine = torch.nn.functional.cosine_similarity(
        tp_sum.flatten(), reference_output.flatten(), dim=0
    )
    relative_l2 = difference.norm() / reference_output.norm().clamp_min(1e-12)
    if float(cosine.item()) < 0.999 or float(relative_l2.item()) > 0.02:
        raise RuntimeError(
            "TP=4 partial sum differs from the unsharded native oracle: "
            f"cosine={float(cosine.item()):.8f} "
            f"relative_l2={float(relative_l2.item()):.8f} "
            f"max_abs={float(difference.abs().max().item()):.8f}"
        )

    report = {
        "snapshot": str(args.snapshot.resolve()),
        "layer": args.layer,
        "architecture": "replicated routes, intermediate TP=4, hidden partial sum",
        "global_expert_ids_on_every_rank": expert_ids,
        "expert_tp_world_size": EXPERT_TP_WORLD_SIZE,
        "hidden_size": reference.config.hidden_size,
        "global_intermediate_size": reference.config.intermediate_size,
        "local_intermediate_size": rank_reports[0]["intermediate_stop"]
        - rank_reports[0]["intermediate_start"],
        "top_k": reference.config.top_k,
        "swiglu_limit": reference.config.swiglu_limit,
        "source_format": NATIVE_SOURCE_FORMAT,
        "quant_mode": "w4a16",
        "sparkinfer_revision": _pinned_sparkinfer.REVISION,
        "gpu": torch.cuda.get_device_name(device),
        "visible_gpu_count": torch.cuda.device_count(),
        "tokens": args.tokens,
        "ranks": rank_reports,
        "predicted_parallel_kernel_ms": max(
            report["mean_kernel_ms"] for report in rank_reports
        ),
        "reference_kernel_ms": reference_ms,
        "tp_sum_l2": float(tp_sum.norm().item()),
        "reference_l2": float(reference_output.norm().item()),
        "tp4_oracle_cosine": float(cosine.item()),
        "tp4_oracle_relative_l2_error": float(relative_l2.item()),
        "tp4_oracle_max_abs_error": float(difference.abs().max().item()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
