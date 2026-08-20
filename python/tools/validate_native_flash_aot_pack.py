#!/usr/bin/env python3
"""Validate DS4 Flash native packers against SparkInfer's format oracle."""

from __future__ import annotations

import _pinned_sparkinfer  # noqa: F401

import argparse
import ctypes
import json
from pathlib import Path


class DeviceBuffer(ctypes.Structure):
    _fields_ = (
        ("ptr", ctypes.c_void_p),
        ("bytes", ctypes.c_size_t),
        ("device_id", ctypes.c_int),
        ("flags", ctypes.c_uint64),
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
    contiguous_args = (
        DeviceBuffer,
        DeviceBuffer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    strided_args = (
        DeviceBuffer,
        DeviceBuffer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    for name in (
        "ds4rt_cuda_ds4_flash_w4a16_pack_weight_async",
        "ds4rt_cuda_ds4_flash_w4a16_pack_e8m0_scale_async",
    ):
        function = getattr(library, name)
        function.argtypes = contiguous_args
        function.restype = ctypes.c_int
    for name in (
        "ds4rt_cuda_ds4_flash_w4a16_pack_weight_strided_async",
        "ds4rt_cuda_ds4_flash_w4a16_pack_e8m0_scale_strided_async",
    ):
        function = getattr(library, name)
        function.argtypes = strided_args
        function.restype = ctypes.c_int
    return library


def require_ok(status: int, action: str) -> None:
    if status != 0:
        raise RuntimeError(f"{action} failed with ds4rt status {status}")


def compare_bytes(actual, expected, label: str) -> dict[str, int | str]:
    import torch

    actual_bytes = actual.view(torch.uint8).flatten()
    expected_bytes = expected.view(torch.uint8).flatten()
    mismatches = int((actual_bytes != expected_bytes).sum().item())
    if mismatches:
        indices = (actual_bytes != expected_bytes).nonzero().flatten()[:8].tolist()
        raise AssertionError(
            f"{label} differs at {mismatches} bytes; first indices={indices}"
        )
    return {"case": label, "bytes": int(actual_bytes.numel()), "mismatches": 0}


def run(native_path: Path, seed: int) -> list[dict[str, int | str]]:
    import torch
    from b12x.moe._shared.kernels.w4a16.prepare import (
        _pack_e8m0_k32_scales,
        _repack_weight,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("validation requires CUDA")
    torch.cuda.set_device(0)
    torch.manual_seed(seed)
    device = torch.device("cuda", 0)
    stream = ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream)
    native = configure_native(native_path)
    results: list[dict[str, int | str]] = []

    # FC1 is a contiguous TP row shard after w3/w1 concatenation.  The row
    # rotation changes source [w3,w1] to the kernel's [gate,up] order.
    fc1_k, fc1_n, fc1_rotation = 4096, 1024, 512
    fc1_source = torch.randint(
        0, 256, (1, fc1_n, fc1_k // 2), dtype=torch.uint8, device=device
    )
    fc1_expected = _repack_weight(
        fc1_source,
        size_k=fc1_k,
        size_n=fc1_n,
        row_rotation=fc1_rotation,
    )
    fc1_actual = torch.empty_like(fc1_expected)
    require_ok(
        native.ds4rt_cuda_ds4_flash_w4a16_pack_weight_async(
            tensor_buffer(fc1_source),
            tensor_buffer(fc1_actual),
            fc1_k,
            fc1_n,
            fc1_rotation,
            stream,
        ),
        "pack FC1 weight",
    )

    fc1_scale_source = torch.randint(
        0, 256, (1, fc1_n, fc1_k // 32), dtype=torch.uint8, device=device
    )
    fc1_scale_expected = _pack_e8m0_k32_scales(
        fc1_scale_source,
        size_k=fc1_k,
        size_n=fc1_n,
        row_rotation=fc1_rotation,
    )
    fc1_scale_actual = torch.empty_like(fc1_scale_expected)
    require_ok(
        native.ds4rt_cuda_ds4_flash_w4a16_pack_e8m0_scale_async(
            tensor_buffer(fc1_scale_source),
            tensor_buffer(fc1_scale_actual),
            fc1_k,
            fc1_n,
            fc1_rotation,
            stream,
        ),
        "pack FC1 E8M0 scale",
    )

    # FC2 is column-sharded.  Validate rank 3 so a mistaken full-width or
    # zero-offset implementation cannot pass this check.
    fc2_source_k, fc2_k, fc2_n, fc2_start_k = 2048, 512, 4096, 1536
    fc2_source = torch.randint(
        0,
        256,
        (1, fc2_n, fc2_source_k // 2),
        dtype=torch.uint8,
        device=device,
    )
    fc2_slice = fc2_source[:, :, fc2_start_k // 2 : (fc2_start_k + fc2_k) // 2]
    fc2_expected = _repack_weight(
        fc2_slice.contiguous(),
        size_k=fc2_k,
        size_n=fc2_n,
    )
    fc2_actual = torch.empty_like(fc2_expected)
    require_ok(
        native.ds4rt_cuda_ds4_flash_w4a16_pack_weight_strided_async(
            tensor_buffer(fc2_source),
            tensor_buffer(fc2_actual),
            fc2_k,
            fc2_source_k,
            fc2_start_k,
            fc2_n,
            0,
            stream,
        ),
        "pack rank-3 FC2 weight",
    )

    fc2_scale_source = torch.randint(
        0,
        256,
        (1, fc2_n, fc2_source_k // 32),
        dtype=torch.uint8,
        device=device,
    )
    fc2_scale_slice = fc2_scale_source[
        :, :, fc2_start_k // 32 : (fc2_start_k + fc2_k) // 32
    ]
    fc2_scale_expected = _pack_e8m0_k32_scales(
        fc2_scale_slice.contiguous(),
        size_k=fc2_k,
        size_n=fc2_n,
    )
    fc2_scale_actual = torch.empty_like(fc2_scale_expected)
    require_ok(
        native.ds4rt_cuda_ds4_flash_w4a16_pack_e8m0_scale_strided_async(
            tensor_buffer(fc2_scale_source),
            tensor_buffer(fc2_scale_actual),
            fc2_k,
            fc2_source_k,
            fc2_start_k,
            fc2_n,
            0,
            stream,
        ),
        "pack rank-3 FC2 E8M0 scale",
    )

    torch.cuda.synchronize(device)
    results.extend(
        [
            compare_bytes(fc1_actual, fc1_expected, "fc1_weight"),
            compare_bytes(fc1_scale_actual, fc1_scale_expected, "fc1_e8m0_scale"),
            compare_bytes(fc2_actual, fc2_expected, "fc2_rank3_weight"),
            compare_bytes(
                fc2_scale_actual,
                fc2_scale_expected,
                "fc2_rank3_e8m0_scale",
            ),
        ]
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-lib", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    print(json.dumps({"results": run(args.native_lib, args.seed)}, indent=2))


if __name__ == "__main__":
    main()
