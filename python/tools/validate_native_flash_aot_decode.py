#!/usr/bin/env python3
"""Run Flash TP4 native decode/prefill AOT against SparkInfer's source oracle."""

from __future__ import annotations

import _pinned_sparkinfer  # noqa: F401

import argparse
import ctypes
import json
from pathlib import Path


HIDDEN = 4096
LOCAL_INTERMEDIATE = 512
GLOBAL_EXPERTS = 256
TOP_K = 6
LOCK_ELEMENTS = 48 * 4 + 2
FC1_SCRATCH_ELEMENTS = 98_304
FC2_SCRATCH_ELEMENTS = 393_216
PREFILL_MAX_ROWS = 2048
PREFILL_ROUTE_BLOCK = 32
PREFILL_MAX_PACKED_ROUTE_SLOTS = 20_224
PREFILL_MAX_ROUTE_BLOCKS = 632
PREFILL_SCRATCH_ELEMENTS = 3_145_728


class DeviceBuffer(ctypes.Structure):
    _fields_ = (
        ("ptr", ctypes.c_void_p),
        ("bytes", ctypes.c_size_t),
        ("device_id", ctypes.c_int),
        ("flags", ctypes.c_uint64),
    )


class FlashBuffers(ctypes.Structure):
    _fields_ = tuple(
        (name, DeviceBuffer)
        for name in (
            "input",
            "w13_weight",
            "w2_weight",
            "fc1_output",
            "activated",
            "routed_output",
            "output",
            "w13_scale",
            "w2_scale",
            "w13_global_scale",
            "w2_global_scale",
            "packed_route_indices",
            "block_expert_ids",
            "packed_route_count",
            "topk_weights",
            "fc1_scratch",
            "fc2_scratch",
            "locks",
        )
    )


def tensor_buffer(tensor) -> DeviceBuffer:
    return DeviceBuffer(
        ctypes.c_void_p(tensor.data_ptr()),
        tensor.numel() * tensor.element_size(),
        tensor.device.index or 0,
        0,
    )


def configure_native(path: Path):
    library = ctypes.CDLL(str(path.resolve()))
    pack_args = (
        DeviceBuffer,
        DeviceBuffer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    for name in (
        "ds41rt_cuda_ds4_flash_w4a16_pack_weight_async",
        "ds41rt_cuda_ds4_flash_w4a16_pack_e8m0_scale_async",
    ):
        function = getattr(library, name)
        function.argtypes = pack_args
        function.restype = ctypes.c_int
    library.ds41rt_cuda_ds4_flash_spark_aot_available.argtypes = (
        ctypes.POINTER(ctypes.c_int),
    )
    library.ds41rt_cuda_ds4_flash_spark_aot_available.restype = ctypes.c_int
    library.ds41rt_cuda_ds4_flash_spark_w4a16_decode_m1_bf16_async.argtypes = (
        ctypes.POINTER(FlashBuffers),
        ctypes.c_void_p,
    )
    library.ds41rt_cuda_ds4_flash_spark_w4a16_decode_m1_bf16_async.restype = (
        ctypes.c_int
    )
    library.ds41rt_cuda_ds4_flash_spark_w4a16_prefill_topk6_bf16_async.argtypes = (
        ctypes.POINTER(FlashBuffers),
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    library.ds41rt_cuda_ds4_flash_spark_w4a16_prefill_topk6_bf16_async.restype = (
        ctypes.c_int
    )
    return library


def require_ok(status: int, action: str) -> None:
    if status != 0:
        raise RuntimeError(f"{action} failed with ds41rt status {status}")


def prefill_capacity(rows: int) -> int:
    capacity = 1 if rows == 1 else 2
    while capacity < rows and capacity < PREFILL_MAX_ROWS:
        capacity *= 2
    if rows < 1 or rows > capacity:
        raise ValueError(f"rows must be in 1..={PREFILL_MAX_ROWS}, got {rows}")
    return capacity


def pack_prefill_routes(topk_ids) -> tuple[list[int], list[int]]:
    flat_ids = topk_ids.reshape(-1).tolist()
    route_count = len(flat_ids)
    packed = []
    block_experts = []
    for expert_id in range(GLOBAL_EXPERTS):
        indices = [
            route_index
            for route_index, routed_expert in enumerate(flat_ids)
            if routed_expert == expert_id
        ]
        if not indices:
            continue
        blocks = (len(indices) + PREFILL_ROUTE_BLOCK - 1) // PREFILL_ROUTE_BLOCK
        padded = blocks * PREFILL_ROUTE_BLOCK
        packed.extend(indices)
        packed.extend([route_count] * (padded - len(indices)))
        block_experts.extend([expert_id] * blocks)
    if len(packed) > PREFILL_MAX_PACKED_ROUTE_SLOTS:
        raise AssertionError("packed route plan exceeds native AOT capacity")
    if len(block_experts) > PREFILL_MAX_ROUTE_BLOCKS:
        raise AssertionError("route block plan exceeds native AOT capacity")
    return packed, block_experts


def run(native_path: Path, seed: int, rows: int) -> dict[str, float | int]:
    import torch
    from b12x.moe._shared.kernels.w4a16.kernel import run_w4a16_moe
    from b12x.moe._shared.kernels.w4a16.prepare import (
        make_w4a16_packed_buffers,
        prepare_w4a16_e8m0_native_weights,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "validation requires CUDA_VISIBLE_DEVICES=0 and one visible GPU"
        )
    torch.cuda.set_device(0)
    torch.manual_seed(seed)
    device = torch.device("cuda", 0)
    stream = ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream)
    native = configure_native(native_path)

    available = ctypes.c_int()
    require_ok(
        native.ds41rt_cuda_ds4_flash_spark_aot_available(ctypes.byref(available)),
        "query Flash AOT availability",
    )
    if available.value != 1:
        raise RuntimeError("native library was built without Flash AOT")

    # Six local source experts cover every route while the native resident
    # slabs retain the production all-256-expert geometry.
    local_experts = TOP_K
    w13_source = torch.randint(
        0,
        256,
        (local_experts, 2 * LOCAL_INTERMEDIATE, HIDDEN // 2),
        dtype=torch.uint8,
        device=device,
    )
    w2_source = torch.randint(
        0,
        256,
        (local_experts, HIDDEN, LOCAL_INTERMEDIATE // 2),
        dtype=torch.uint8,
        device=device,
    )
    # Keep synthetic E8M0 exponents in a finite, checkpoint-like neighborhood.
    w13_scale_source = torch.randint(
        118,
        132,
        (local_experts, 2 * LOCAL_INTERMEDIATE, HIDDEN // 32),
        dtype=torch.uint8,
        device=device,
    )
    w2_scale_source = torch.randint(
        118,
        132,
        (local_experts, HIDDEN, LOCAL_INTERMEDIATE // 32),
        dtype=torch.uint8,
        device=device,
    )
    source_global = torch.ones(local_experts, dtype=torch.float32, device=device)
    oracle_weights = prepare_w4a16_e8m0_native_weights(
        w13_source,
        w13_scale_source,
        source_global,
        w2_source,
        w2_scale_source,
        source_global,
        activation="silu",
        params_dtype=torch.bfloat16,
        w13_layout="w13",
    )

    w13_weight = torch.empty(
        (
            GLOBAL_EXPERTS,
            HIDDEN // 16,
            ((2 * LOCAL_INTERMEDIATE) // 64) * 128,
        ),
        dtype=torch.int32,
        device=device,
    )
    w2_weight = torch.empty(
        (
            GLOBAL_EXPERTS,
            LOCAL_INTERMEDIATE // 16,
            (HIDDEN // 64) * 128,
        ),
        dtype=torch.int32,
        device=device,
    )
    w13_scale = torch.empty(
        (GLOBAL_EXPERTS, HIDDEN // 32, 2 * LOCAL_INTERMEDIATE),
        dtype=torch.uint8,
        device=device,
    )
    w2_scale = torch.empty(
        (GLOBAL_EXPERTS, LOCAL_INTERMEDIATE // 32, HIDDEN),
        dtype=torch.uint8,
        device=device,
    )
    for expert in range(local_experts):
        require_ok(
            native.ds41rt_cuda_ds4_flash_w4a16_pack_weight_async(
                tensor_buffer(w13_source[expert]),
                tensor_buffer(w13_weight[expert]),
                HIDDEN,
                2 * LOCAL_INTERMEDIATE,
                LOCAL_INTERMEDIATE,
                stream,
            ),
            f"pack expert {expert} FC1",
        )
        require_ok(
            native.ds41rt_cuda_ds4_flash_w4a16_pack_e8m0_scale_async(
                tensor_buffer(w13_scale_source[expert]),
                tensor_buffer(w13_scale[expert]),
                HIDDEN,
                2 * LOCAL_INTERMEDIATE,
                LOCAL_INTERMEDIATE,
                stream,
            ),
            f"pack expert {expert} FC1 scale",
        )
        require_ok(
            native.ds41rt_cuda_ds4_flash_w4a16_pack_weight_async(
                tensor_buffer(w2_source[expert]),
                tensor_buffer(w2_weight[expert]),
                LOCAL_INTERMEDIATE,
                HIDDEN,
                0,
                stream,
            ),
            f"pack expert {expert} FC2",
        )
        require_ok(
            native.ds41rt_cuda_ds4_flash_w4a16_pack_e8m0_scale_async(
                tensor_buffer(w2_scale_source[expert]),
                tensor_buffer(w2_scale[expert]),
                LOCAL_INTERMEDIATE,
                HIDDEN,
                0,
                stream,
            ),
            f"pack expert {expert} FC2 scale",
        )

    capacity_rows = prefill_capacity(rows)
    hidden = torch.empty(
        (capacity_rows, HIDDEN), dtype=torch.bfloat16, device=device
    )
    hidden[:rows] = (
        torch.randn((rows, HIDDEN), dtype=torch.bfloat16, device=device) * 0.125
    )
    topk_ids = torch.stack(
        [torch.arange(TOP_K, device=device).roll(row % TOP_K) for row in range(rows)]
    ).to(torch.int32)
    topk_weights = torch.rand((rows, TOP_K), dtype=torch.float32, device=device)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    oracle_buffers = make_w4a16_packed_buffers(
        oracle_weights,
        m=rows,
        topk=TOP_K,
        dtype=torch.bfloat16,
        device=device,
    )
    oracle = run_w4a16_moe(
        hidden[:rows],
        oracle_weights,
        topk_weights,
        topk_ids,
        activation="silu",
        intermediate_cache13=oracle_buffers.intermediate_cache13,
        intermediate_cache2=oracle_buffers.intermediate_cache2,
        output=oracle_buffers.output,
        fc1_c_tmp=oracle_buffers.fc1_c_tmp,
        fc2_c_tmp=oracle_buffers.fc2_c_tmp,
        packed_route_indices=oracle_buffers.packed_route_indices,
        block_expert_ids=oracle_buffers.block_expert_ids,
        packed_route_count=oracle_buffers.packed_route_count,
        swiglu_limit=10.0,
    ).clone()

    output = torch.empty(
        (capacity_rows, HIDDEN), dtype=torch.bfloat16, device=device
    )
    global_scales = torch.ones(GLOBAL_EXPERTS, dtype=torch.float32, device=device)
    fc1_output = torch.empty(
        capacity_rows * TOP_K * 2 * LOCAL_INTERMEDIATE,
        dtype=torch.bfloat16,
        device=device,
    )
    activated = torch.empty(
        capacity_rows * TOP_K * LOCAL_INTERMEDIATE,
        dtype=torch.bfloat16,
        device=device,
    )
    routed_output = torch.empty(
        (capacity_rows * TOP_K, HIDDEN), dtype=torch.bfloat16, device=device
    )
    if rows == 1:
        packed_route_indices = topk_ids
        block_expert_ids = torch.zeros(1, dtype=torch.int32, device=device)
        packed_route_count = torch.zeros(1, dtype=torch.int32, device=device)
    else:
        packed, block_experts = pack_prefill_routes(topk_ids.cpu())
        packed_route_indices = torch.full(
            (PREFILL_MAX_PACKED_ROUTE_SLOTS,),
            rows * TOP_K,
            dtype=torch.int32,
            device=device,
        )
        packed_route_indices[: len(packed)] = torch.tensor(
            packed, dtype=torch.int32, device=device
        )
        block_expert_ids = torch.zeros(
            PREFILL_MAX_ROUTE_BLOCKS, dtype=torch.int32, device=device
        )
        block_expert_ids[: len(block_experts)] = torch.tensor(
            block_experts, dtype=torch.int32, device=device
        )
        packed_route_count = torch.tensor(
            [len(packed)], dtype=torch.int32, device=device
        )
    native_topk_weights = torch.empty(
        capacity_rows * TOP_K, dtype=torch.float32, device=device
    )
    native_topk_weights[: rows * TOP_K] = topk_weights.reshape(-1)
    scratch_elements = (
        FC1_SCRATCH_ELEMENTS if rows == 1 else PREFILL_SCRATCH_ELEMENTS
    )
    fc1_scratch = torch.empty(
        scratch_elements, dtype=torch.float32, device=device
    )
    fc2_scratch = torch.empty(
        FC2_SCRATCH_ELEMENTS if rows == 1 else PREFILL_SCRATCH_ELEMENTS,
        dtype=torch.float32,
        device=device,
    )
    locks = torch.empty(LOCK_ELEMENTS, dtype=torch.int32, device=device)
    buffers = FlashBuffers(
        tensor_buffer(hidden),
        tensor_buffer(w13_weight),
        tensor_buffer(w2_weight),
        tensor_buffer(fc1_output),
        tensor_buffer(activated),
        tensor_buffer(routed_output),
        tensor_buffer(output),
        tensor_buffer(w13_scale),
        tensor_buffer(w2_scale),
        tensor_buffer(global_scales),
        tensor_buffer(global_scales),
        tensor_buffer(packed_route_indices),
        tensor_buffer(block_expert_ids),
        tensor_buffer(packed_route_count),
        tensor_buffer(native_topk_weights),
        tensor_buffer(fc1_scratch),
        tensor_buffer(fc2_scratch),
        tensor_buffer(locks),
    )
    if rows == 1:
        require_ok(
            native.ds41rt_cuda_ds4_flash_spark_w4a16_decode_m1_bf16_async(
                ctypes.byref(buffers), stream
            ),
            "launch Flash AOT decode",
        )
    else:
        require_ok(
            native.ds41rt_cuda_ds4_flash_spark_w4a16_prefill_topk6_bf16_async(
                ctypes.byref(buffers), rows, stream
            ),
            "launch Flash AOT prefill",
        )
    torch.cuda.synchronize(device)

    actual_f32 = output[:rows].float()
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
    if cosine < 0.999 or relative_l2 > 0.02:
        raise AssertionError(
            f"Flash native AOT mismatch for rows={rows}: "
            f"cosine={cosine} relative_l2={relative_l2}"
        )
    return {
        "cosine": cosine,
        "relative_l2": relative_l2,
        "max_abs": max_abs,
        "rows": rows,
        "capacity_rows": capacity_rows,
        "resident_weight_scale_bytes": int(
            w13_weight.numel() * w13_weight.element_size()
            + w2_weight.numel() * w2_weight.element_size()
            + w13_scale.numel()
            + w2_scale.numel()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-lib", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--rows", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(run(args.native_lib, args.seed, args.rows), indent=2))


if __name__ == "__main__":
    main()
