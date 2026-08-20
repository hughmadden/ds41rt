#!/usr/bin/env python3
"""Reproduce EXL3 global-scale search on one captured DeepSeek expert."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import threading

import torch
from safetensors import safe_open

from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import (
    g_scale_gss,
    regularize,
)
from gptqmodel.quantization.dtype import dequantize_f4_e2m1


def load_tensor(path: Path, name: str) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--hessian", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expert", type=int, required=True)
    parser.add_argument("--projection", choices=("w1", "w2", "w3"), required=True)
    parser.add_argument(
        "--device",
        action="append",
        dest="devices",
        help="CUDA device; repeat to exercise concurrent devices",
    )
    parser.add_argument(
        "--thread-device",
        help="Force every worker thread's current CUDA device independently of its tensor device",
    )
    parser.add_argument(
        "--free-gib",
        type=float,
        help="After warming the scale-search buffers, reserve VRAM until this much remains free",
    )
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    prefix = f"layers.{args.layer}.ffn.experts.{args.expert}.{args.projection}"
    packed = load_tensor(args.source, f"{prefix}.weight")
    scales = load_tensor(args.source, f"{prefix}.scale")
    source_weight = dequantize_f4_e2m1(
        packed,
        scale=scales,
        axis=None,
        target_dtype=torch.bfloat16,
    ).T.contiguous()
    hessian_diag_cpu = torch.diag(load_tensor(args.hessian, "H"))
    # The captured Hessian is an X^T X sum. Its diagonal is needed only for
    # the same output-scale policy used by the production regularizer; the
    # common normalization factor does not affect that policy.
    devices = args.devices or ["cuda:0"]
    barrier = threading.Barrier(len(devices))

    def run_device(device_name: str) -> dict[str, object]:
        device = torch.device(device_name)
        torch.cuda.set_device(args.thread_device or device)
        weight = source_weight.to(device=device, dtype=torch.float32)
        hessian_diag = hessian_diag_cpu.to(device=device)
        generator = torch.Generator(device=device).manual_seed(787)
        su = (
            torch.randn(weight.shape[0], device=device, generator=generator).sign()
            + 1e-5
        ).sign()
        su = su.float().unsqueeze(1)
        sv = (
            torch.randn(weight.shape[1], device=device, generator=generator).sign()
            + 1e-5
        ).sign()
        sv = sv.float().unsqueeze(0)
        quant_args = {
            "K": 2,
            "devices": [device],
            "apply_out_scales": None,
            "sigma_reg": 0.025,
            "seed": 787,
            "mcg": True,
        }
        _, regularized, _, _, _, _ = regularize(
            weight,
            su,
            sv,
            quant_args,
            False,
            hessian_diag,
            None,
            skip_g_scale=True,
        )
        # Allocate the 2+ GiB K2 scratch cache before optional pressure so the
        # pressure probe models a long-running production worker.
        g_scale_gss(regularized, False, quant_args)
        torch.cuda.synchronize(device)
        filler = None
        if args.free_gib is not None:
            free_bytes, _ = torch.cuda.mem_get_info(device)
            reserve = max(0, int(free_bytes - args.free_gib * (1024**3)))
            filler = torch.empty(reserve, dtype=torch.uint8, device=device)
            filler.fill_(0xA5)
            torch.cuda.synchronize(device)
        barrier.wait()
        bad = 0
        values: list[float] = []
        scales_found: list[float] = []
        for index in range(args.iterations):
            scale, mse = g_scale_gss(regularized, False, quant_args)
            value = float(mse.item())
            values.append(value)
            scales_found.append(float(scale))
            if not math.isfinite(value) or value < 0.0:
                bad += 1
                print(
                    f"BAD device={device_name} iteration={index} "
                    f"scale={scale:.17g} mse={value!r}",
                    flush=True,
                )
        return {
            "device": device_name,
            "iterations": args.iterations,
            "bad": bad,
            "mse_min": min(values),
            "mse_max": max(values),
            "unique_mse": len(set(values)),
            "scale_min": min(scales_found),
            "scale_max": max(scales_found),
            "shape": list(regularized.shape),
            "filler_bytes": 0 if filler is None else filler.numel(),
        }

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        reports = list(executor.map(run_device, devices))
    for report in reports:
        print(report, flush=True)
    return 1 if any(report["bad"] for report in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
