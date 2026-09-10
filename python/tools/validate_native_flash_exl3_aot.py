#!/usr/bin/env python3
"""Qualify production-shape Flash/Pro EXL3 K2 native AOT on GPU 0."""

from __future__ import annotations

import _pinned_sparkinfer  # noqa: F401

import argparse
import ctypes
from dataclasses import dataclass
import json
from pathlib import Path
import statistics


TOP_K = 6
PREFILL_MAX_ROWS = 2048
PREFILL_SCRATCH_ELEMENTS = 3_145_728
TRELLIS_LUT_BYTES = 1 << 12


@dataclass(frozen=True)
class Profile:
    variant: str
    hidden: int
    local_intermediate: int
    experts: int
    prefill_route_slots: int
    prefill_route_blocks: int
    decode_fc1_scratch: int
    decode_fc2_scratch: int
    tile_config: tuple[int, int, int, int]


PROFILES = {
    "flash": Profile(
        "flash", 4096, 512, 256, 20_224, 632, 98_304, 393_216,
        (64, 128, 64, 128),
    ),
    "pro": Profile(
        "pro", 7168, 768, 384, 24_192, 756, 147_456, 688_128,
        (64, 256, 64, 256),
    ),
}


class DeviceBuffer(ctypes.Structure):
    _fields_ = (
        ("ptr", ctypes.c_void_p),
        ("bytes", ctypes.c_size_t),
        ("device_id", ctypes.c_int),
        ("flags", ctypes.c_uint64),
    )


class Exl3Buffers(ctypes.Structure):
    _fields_ = tuple(
        (name, DeviceBuffer)
        for name in (
            "input",
            "w13_trellis",
            "w2_trellis",
            "gate_suh",
            "up_suh",
            "intermediate_rotations",
            "down_svh",
            "expert_map",
            "dummy_scale",
            "trellis_lut",
            "global_scale",
            "topk_ids",
            "topk_weights",
            "rotation_gate",
            "rotation_up",
            "fc1_output",
            "activated",
            "routed_output",
            "output_f32",
            "output_bf16",
            "packed_route_indices",
            "block_expert_ids",
            "packed_route_count",
            "expert_counts",
            "expert_offsets",
            "fc1_scratch",
            "fc2_scratch",
            "workspace",
        )
    )


def tensor_buffer(tensor) -> DeviceBuffer:
    return DeviceBuffer(
        ctypes.c_void_p(tensor.data_ptr()),
        tensor.numel() * tensor.element_size(),
        tensor.device.index or 0,
        0,
    )


def require_ok(status: int, action: str) -> None:
    if status != 0:
        raise RuntimeError(f"{action} failed with ds41rt status {status}")


def configure_native(path: Path, variant: str):
    library = ctypes.CDLL(str(path.resolve()))
    initialize = getattr(library, f"ds41rt_cuda_ds4_{variant}_spark_aot_init")
    decode = getattr(
        library, f"ds41rt_cuda_ds4_{variant}_spark_exl3_k2_decode_m1_async"
    )
    prefill = getattr(
        library,
        f"ds41rt_cuda_ds4_{variant}_spark_exl3_k2_prefill_topk6_async",
    )
    initialize.restype = ctypes.c_int
    decode.argtypes = (
        ctypes.POINTER(Exl3Buffers),
        ctypes.c_void_p,
    )
    decode.restype = ctypes.c_int
    prefill.argtypes = (
        ctypes.POINTER(Exl3Buffers),
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    prefill.restype = ctypes.c_int
    return initialize, decode, prefill


def prefill_capacity(rows: int) -> int:
    capacity = 1 if rows == 1 else 2
    while capacity < rows and capacity < PREFILL_MAX_ROWS:
        capacity *= 2
    if rows < 1 or rows > capacity:
        raise ValueError(f"rows must be in 1..={PREFILL_MAX_ROWS}, got {rows}")
    return capacity


def run(
    native_path: Path,
    seed: int,
    rows: int,
    model_variant: str,
    benchmark_iterations: int = 0,
    benchmark_rounds: int = 5,
) -> dict[str, float | int | str]:
    import torch
    from b12x.moe._shared.kernels.w4a16.host import (
        make_w4a16_packed_buffers,
    )
    from b12x.moe._shared.kernels.w4a16.kernel import run_w4a16_moe
    from b12x.moe._shared.kernels.w4a16.prepare import (
        prepare_trellis256_moe_weights,
    )

    profile = PROFILES[model_variant]
    hidden_size = profile.hidden
    local_intermediate = profile.local_intermediate
    global_experts = profile.experts

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "validation requires CUDA_VISIBLE_DEVICES=0 and one visible GPU"
        )
    torch.cuda.set_device(0)
    torch.manual_seed(seed)
    device = torch.device("cuda", 0)
    initialize, decode, prefill = configure_native(native_path, model_variant)
    require_ok(
        initialize(),
        f"initialize {model_variant} AOT modules",
    )

    # These are the final resident checkpoint layouts.  Preparation below only
    # flattens int32 views over them; it does not repack or widen either slab.
    w13 = torch.empty(
        (
            2,
            global_experts,
            hidden_size // 16,
            local_intermediate // 16,
            32,
        ),
        dtype=torch.int16,
        device=device,
    )
    w13[0].zero_()
    w13[1].fill_(1)
    w2 = torch.full(
        (
            global_experts,
            local_intermediate // 16,
            hidden_size // 16,
            32,
        ),
        2,
        dtype=torch.int16,
        device=device,
    )
    gate_suh = torch.full(
        (global_experts, hidden_size), 0.75, dtype=torch.float16, device=device
    )
    up_suh = torch.full_like(gate_suh, 1.25)
    intermediate_rotations = torch.empty(
        (global_experts, 3 * local_intermediate),
        dtype=torch.float16,
        device=device,
    )
    intermediate_rotations[:, :local_intermediate].fill_(0.875)
    intermediate_rotations[
        :, local_intermediate : 2 * local_intermediate
    ].fill_(1.125)
    intermediate_rotations[:, 2 * local_intermediate :].fill_(0.9375)
    down_svh = torch.full_like(gate_suh, 1.0625)
    prepared = prepare_trellis256_moe_weights(
        w13,
        w2,
        hidden_size=hidden_size,
        intermediate_size=local_intermediate,
        num_experts=global_experts,
        activation="silu",
        fc1_tile_n=profile.tile_config[1],
        fc2_tile_n=profile.tile_config[3],
        params_dtype=torch.float16,
        w13_layout="trellis3_t256_proj",
        trellis_bits=2,
        codebook="mcg",
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
        tile_config=profile.tile_config,
    )
    assert prepared.w13.data_ptr() == w13.data_ptr()
    assert prepared.w2.data_ptr() == w2.data_ptr()

    capacity_rows = prefill_capacity(rows)
    hidden = torch.empty(
        (capacity_rows, hidden_size), dtype=torch.bfloat16, device=device
    )
    hidden[:rows] = (
        torch.randn((rows, hidden_size), dtype=torch.bfloat16, device=device) * 0.01
    )
    topk_ids = torch.randint(
        0,
        global_experts,
        (capacity_rows, TOP_K),
        dtype=torch.int32,
        device=device,
    )
    topk_weights = torch.rand(
        (capacity_rows, TOP_K), dtype=torch.float32, device=device
    )
    topk_weights[:rows] /= topk_weights[:rows].sum(dim=-1, keepdim=True)
    expert_map = torch.arange(global_experts, dtype=torch.int32, device=device)
    block_size = 8 if rows == 1 else 32
    buffers = make_w4a16_packed_buffers(
        prepared,
        m=capacity_rows,
        topk=TOP_K,
        dtype=torch.float16,
        device=device,
        route_num_experts=global_experts,
        full_rotation=True,
        block_size_m=block_size,
    )
    oracle = run_w4a16_moe(
        hidden[:rows],
        prepared,
        topk_weights[:rows],
        topk_ids[:rows],
        activation="silu",
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output[:rows],
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
        expert_counts=buffers.expert_counts,
        expert_map=expert_map,
        output_expert_map=expert_map,
        swiglu_limit=10.0,
        route_block_size_m=block_size,
        intermediate_rotation_scales=intermediate_rotations,
        full_rotation=True,
        suh_gate_table=gate_suh,
        suh_up_table=up_suh,
        svh_table=down_svh,
        rotation_a_gate=buffers.rotation_a_gate,
        rotation_a_up=buffers.rotation_a_up,
    ).clone()

    # Native serving uses one maximum-capacity arena reused by every layer.
    # FC1 and routed FC2 intentionally alias intermediate_cache13 across the
    # fused kernel's grid barriers, matching SparkInfer's planner.
    packed_routes = torch.empty(
        profile.prefill_route_slots, dtype=torch.int32, device=device
    )
    block_experts = torch.empty(
        profile.prefill_route_blocks, dtype=torch.int32, device=device
    )
    packed_count = torch.empty(1, dtype=torch.int32, device=device)
    expert_counts = torch.empty(
        global_experts, dtype=torch.int32, device=device
    )
    expert_offsets = torch.empty(
        global_experts + 1, dtype=torch.int32, device=device
    )
    fc1_scratch = torch.empty(
        profile.decode_fc1_scratch if rows == 1 else PREFILL_SCRATCH_ELEMENTS,
        dtype=torch.float32,
        device=device,
    )
    fc2_scratch = torch.empty(
        profile.decode_fc2_scratch if rows == 1 else PREFILL_SCRATCH_ELEMENTS,
        dtype=torch.float32,
        device=device,
    )
    output_bf16 = torch.empty(
        (capacity_rows, hidden_size), dtype=torch.bfloat16, device=device
    )
    # b12x exposes one generic trellis ABI for both MCG and SQG.  Production
    # MCG kernels must ignore this SQG-only table, so an opaque zero allocation
    # is sufficient here and makes an accidental SQG compile fail parity.
    trellis_lut = torch.zeros(TRELLIS_LUT_BYTES, dtype=torch.uint8, device=device)
    native_buffers = Exl3Buffers(
        tensor_buffer(hidden),
        tensor_buffer(w13),
        tensor_buffer(w2),
        tensor_buffer(gate_suh),
        tensor_buffer(up_suh),
        tensor_buffer(intermediate_rotations),
        tensor_buffer(down_svh),
        tensor_buffer(expert_map),
        tensor_buffer(prepared.w13_scale),
        tensor_buffer(trellis_lut),
        tensor_buffer(prepared.w13_global_scale),
        tensor_buffer(topk_ids),
        tensor_buffer(topk_weights),
        tensor_buffer(buffers.rotation_a_gate),
        tensor_buffer(buffers.rotation_a_up),
        tensor_buffer(buffers.intermediate_cache13),
        tensor_buffer(buffers.intermediate_cache2),
        tensor_buffer(buffers.intermediate_cache13),
        tensor_buffer(buffers.output),
        tensor_buffer(output_bf16),
        tensor_buffer(packed_routes),
        tensor_buffer(block_experts),
        tensor_buffer(packed_count),
        tensor_buffer(expert_counts),
        tensor_buffer(expert_offsets),
        tensor_buffer(fc1_scratch),
        tensor_buffer(fc2_scratch),
        tensor_buffer(prepared.workspace),
    )
    if rows == 1:
        launch = lambda: decode(
            ctypes.byref(native_buffers),
            ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream),
        )
    else:
        launch = (
            lambda: prefill(
                ctypes.byref(native_buffers),
                rows,
                ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream),
            )
        )
    require_ok(launch(), f"launch {model_variant} EXL3 K2 rows={rows}")
    torch.cuda.synchronize(device)
    actual = buffers.output[:rows].clone()
    eager_bf16 = output_bf16[:rows].clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        require_ok(launch(), f"capture {model_variant} EXL3 K2 rows={rows}")
    graph.replay()
    torch.cuda.synchronize(device)
    if not torch.equal(buffers.output[:rows], actual):
        raise AssertionError("native EXL3 graph replay changed the FP32 result")
    if not torch.equal(output_bf16[:rows], eager_bf16):
        raise AssertionError("native EXL3 graph replay changed the BF16 result")

    actual_f32 = actual.float()
    oracle_f32 = oracle.float()
    cosine = float(
        torch.nn.functional.cosine_similarity(
            actual_f32.reshape(1, -1), oracle_f32.reshape(1, -1)
        ).item()
    )
    relative_l2 = float(
        torch.linalg.vector_norm(actual_f32 - oracle_f32)
        / torch.linalg.vector_norm(oracle_f32).clamp_min(1e-12)
    )
    max_abs = float((actual_f32 - oracle_f32).abs().max().item())
    bf16_max_abs = float((eager_bf16.float() - actual_f32).abs().max().item())
    if not torch.isfinite(actual_f32).all() or cosine < 0.99999 or relative_l2 > 1e-5:
        raise AssertionError(
            f"{model_variant} native EXL3 mismatch for rows={rows}: "
            f"cosine={cosine} relative_l2={relative_l2} max_abs={max_abs}"
        )
    result: dict[str, float | int | str] = {
        "model_variant": model_variant,
        "rows": rows,
        "capacity_rows": capacity_rows,
        "cosine": cosine,
        "relative_l2": relative_l2,
        "max_abs": max_abs,
        "bf16_max_abs": bf16_max_abs,
        "resident_trellis_bytes": int(
            w13.numel() * w13.element_size() + w2.numel() * w2.element_size()
        ),
        "prepared_w13_alias": int(prepared.w13.data_ptr() == w13.data_ptr()),
        "prepared_w2_alias": int(prepared.w2.data_ptr() == w2.data_ptr()),
        "graph_replay_equal": 1,
    }
    if benchmark_iterations > 0:
        if benchmark_rounds <= 0:
            raise ValueError("benchmark_rounds must be positive")
        for _ in range(20):
            graph.replay()
        torch.cuda.synchronize(device)
        round_ms = []
        for _ in range(benchmark_rounds):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(benchmark_iterations):
                graph.replay()
            end.record()
            end.synchronize()
            round_ms.append(start.elapsed_time(end) / benchmark_iterations)
        result.update(
            {
                "benchmark_iterations": benchmark_iterations,
                "benchmark_rounds": benchmark_rounds,
                "graph_replay_mean_ms": statistics.fmean(round_ms),
                "graph_replay_median_ms": statistics.median(round_ms),
                "graph_replay_min_ms": min(round_ms),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-lib", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument(
        "--model-variant", choices=tuple(PROFILES), default="flash"
    )
    parser.add_argument("--benchmark-iterations", type=int, default=0)
    parser.add_argument("--benchmark-rounds", type=int, default=5)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.native_lib,
                args.seed,
                args.rows,
                args.model_variant,
                args.benchmark_iterations,
                args.benchmark_rounds,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
