#!/usr/bin/env python3
"""Replay one captured EXL3 projection and report deterministic packed output."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import threading

import torch
from safetensors import safe_open

from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
from gptqmodel.quantization.dtype import dequantize_f4_e2m1


def load_tensor(path: Path, name: str) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def tensor_digest(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--hessian", action="append", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expert", action="append", type=int, required=True)
    parser.add_argument("--projection", choices=("w1", "w2", "w3"), required=True)
    parser.add_argument("--sample-count", action="append", type=int, required=True)
    parser.add_argument("--device", action="append", dest="devices")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--workers-per-device", type=int, default=1)
    parser.add_argument("--serialize-per-device", action="store_true")
    parser.add_argument("--compare-checkpoint", type=Path)
    args = parser.parse_args()

    if not (len(args.expert) == len(args.hessian) == len(args.sample_count)):
        parser.error("--expert, --hessian, and --sample-count counts must match")
    cases = []
    for expert, hessian_path, sample_count in zip(
        args.expert,
        args.hessian,
        args.sample_count,
        strict=True,
    ):
        prefix = f"layers.{args.layer}.ffn.experts.{expert}.{args.projection}"
        packed = load_tensor(args.source, f"{prefix}.weight")
        scales = load_tensor(args.source, f"{prefix}.scale")
        source_weight = dequantize_f4_e2m1(
            packed,
            scale=scales,
            axis=None,
            target_dtype=torch.bfloat16,
        ).T.contiguous()
        cases.append(
            {
                "expert": expert,
                "sample_count": sample_count,
                "weight": source_weight,
                "hessian": load_tensor(hessian_path, "H"),
            }
        )
    devices = args.devices or ["cuda:0"]

    # Match the production ownership contract: every worker receives distinct,
    # already-materialized device storage. Concurrent host-to-device copies from
    # the same safetensors-backed CPU storage would add an unrelated shared-input
    # race to the replay.
    device_inputs: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
    for case_index, case in enumerate(cases):
        for device_name in devices:
            device = torch.device(device_name)
            with torch.cuda.device(device):
                device_inputs[(case_index, device_name)] = (
                    case["weight"].to(device=device, dtype=torch.float32).contiguous(),
                    case["hessian"].to(device=device, dtype=torch.float32).contiguous(),
                )

    if args.iterations < 1 or args.workers_per_device < 1:
        parser.error("--iterations and --workers-per-device must be positive")

    if len(cases) not in {1, args.workers_per_device}:
        parser.error(
            "provide either one replay case or one case per --workers-per-device"
        )
    device_locks = {device_name: threading.Lock() for device_name in devices}

    def run_device(job: tuple[str, int, int]) -> list[dict[str, object]]:
        device_name, worker, case_index = job
        device = torch.device(device_name)
        torch.cuda.set_device(device)
        case = cases[case_index]
        device_weight, device_hessian = device_inputs[(case_index, device_name)]
        reports = []
        for iteration in range(args.iterations):
            quant_args = {
                "K": 2,
                "devices": [device],
                "apply_out_scales": None,
                "sigma_reg": 0.025,
                "seed": 787,
                "mcg": True,
            }
            quant_context = (
                device_locks[device_name]
                if args.serialize_per_device
                else nullcontext()
            )
            with quant_context:
                _, proxy_error, tensors = quantize_exl3(
                    device_weight.clone(),
                    {
                        "H": device_hessian.clone(),
                        "count": case["sample_count"],
                        "finalized": False,
                    },
                    quant_args,
                    return_weight_q=False,
                )
            metrics = quant_args["error_metrics"]
            scale_mse = float(metrics["scale_search_mse"])
            reports.append(
                {
                    "device": device_name,
                    "worker": worker,
                    "expert": case["expert"],
                    "iteration": iteration,
                    "scale": float(metrics["selected_global_scale"]),
                    "scale_search_mse": scale_mse,
                    "scale_search_valid": math.isfinite(scale_mse) and scale_mse >= 0.0,
                    "hessian_error": float(metrics["hessian_weighted_relative_error"]),
                    "reconstruction_nmse": float(metrics["reconstruction"]["nmse"]),
                    "proxy_error": float(proxy_error),
                    "packed_sha256": tensor_digest(tensors),
                }
            )
            del tensors
            torch.cuda.empty_cache()
        return reports

    jobs = [
        (device_name, worker, 0 if len(cases) == 1 else worker)
        for device_name in devices
        for worker in range(args.workers_per_device)
    ]
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        reports = [item for group in executor.map(run_device, jobs) for item in group]
    for report in reports:
        print(json.dumps(report, sort_keys=True), flush=True)
    digests_by_expert = {
        expert: {
            report["packed_sha256"]
            for report in reports
            if report["expert"] == expert
        }
        for expert in args.expert
    }
    valid = all(report["scale_search_valid"] for report in reports)
    checkpoint_digest = None
    if args.compare_checkpoint is not None:
        if len(cases) != 1:
            parser.error("--compare-checkpoint accepts exactly one replay case")
        manifest = json.loads(args.compare_checkpoint.read_text())
        tensor_path = args.compare_checkpoint.parent / manifest["tensor_file"]
        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            checkpoint_digest = tensor_digest(
                {name: handle.get_tensor(name) for name in handle.keys()}
            )
    summary = {
        "all_valid": valid,
        "unique_packed_sha256_by_expert": {
            str(expert): sorted(digests)
            for expert, digests in digests_by_expert.items()
        },
        "checkpoint_packed_sha256": checkpoint_digest,
        "matches_checkpoint": (
            checkpoint_digest is None
            or next(iter(digests_by_expert.values())) == {checkpoint_digest}
        ),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if (
        valid
        and all(len(digests) == 1 for digests in digests_by_expert.values())
        and summary["matches_checkpoint"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
