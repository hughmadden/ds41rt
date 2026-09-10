#!/usr/bin/env python3
"""Validate the native Flash shared expert against its checkpoint contract."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path


HIDDEN = 4096
INTERMEDIATE = 2048
SCALE_MMA_BYTES = (HIDDEN // 128) * (INTERMEDIATE // 128) * 512
CHUNK_ROWS = 8


class DeviceBuffer(ctypes.Structure):
    _fields_ = (
        ("ptr", ctypes.c_void_p),
        ("bytes", ctypes.c_size_t),
        ("device_id", ctypes.c_int),
        ("flags", ctypes.c_uint64),
    )


class SharedBuffers(ctypes.Structure):
    _fields_ = tuple(
        (name, DeviceBuffer)
        for name in (
            "input",
            "w1_weight",
            "w1_scale_mma",
            "w3_weight",
            "w3_scale_mma",
            "w2_weight",
            "w2_scale_mma",
            "gate",
            "up",
            "activated",
            "output",
            "alpha",
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
    library.ds41rt_cuda_ds4_flash_fp8_pack_block_scale_mma_async.argtypes = (
        DeviceBuffer,
        DeviceBuffer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    library.ds41rt_cuda_ds4_flash_fp8_pack_block_scale_mma_async.restype = (
        ctypes.c_int
    )
    library.ds41rt_cuda_ds4_flash_shared_expert_fp8_bf16_async.argtypes = (
        ctypes.POINTER(SharedBuffers),
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    library.ds41rt_cuda_ds4_flash_shared_expert_fp8_bf16_async.restype = ctypes.c_int
    library.ds41rt_last_error.argtypes = (ctypes.c_char_p, ctypes.c_size_t)
    library.ds41rt_last_error.restype = ctypes.c_int
    return library


def native_error(library) -> str:
    message = ctypes.create_string_buffer(1024)
    library.ds41rt_last_error(message, len(message))
    return message.value.decode("utf-8", errors="replace")


def require_ok(library, status: int, action: str) -> None:
    if status != 0:
        raise RuntimeError(
            f"{action} failed with ds41rt status {status}: {native_error(library)}"
        )


def load_layer(snapshot: Path, layer: int):
    from safetensors import safe_open

    index = json.loads((snapshot / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = f"layers.{layer}.ffn.shared_experts"
    tensors = {}
    for projection in ("w1", "w3", "w2"):
        for suffix in ("weight", "scale"):
            name = f"{prefix}.{projection}.{suffix}"
            with safe_open(snapshot / index[name], framework="pt", device="cpu") as file:
                tensors[name] = file.get_tensor(name)
    return tuple(
        tensors[f"{prefix}.{projection}.{suffix}"]
        for projection in ("w1", "w3", "w2")
        for suffix in ("weight", "scale")
    )


def quantize_k128(source):
    import torch

    rows, width = source.shape
    blocks = source.float().reshape(rows, width // 128, 128)
    maximum = blocks.abs().amax(dim=-1).clamp_min(1.0e-4)
    exponent = torch.ceil(torch.log2(maximum / 448.0)).clamp(-127, 127)
    scale = torch.exp2(exponent)
    values = (
        (blocks / scale.unsqueeze(-1))
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
    )
    return values.float() * scale.unsqueeze(-1)


def dequantize_weight(weight, scale):
    return weight.float() * scale.float().repeat_interleave(128, 0).repeat_interleave(
        128, 1
    )


def linear_reference(source, weight, scale):
    quantized = quantize_k128(source).reshape_as(source)
    return (quantized @ dequantize_weight(weight, scale).T).to(source.dtype)


def run(args) -> dict[str, float | int]:
    import torch
    import torch.nn.functional as functional

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "validation requires CUDA_VISIBLE_DEVICES=0 and exactly one visible GPU"
        )
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    stream = ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream)
    native = configure_native(args.native_library)
    torch.manual_seed(args.seed)

    w1, s1, w3, s3, w2, s2 = load_layer(args.snapshot, args.layer)
    expected_shapes = (
        ((INTERMEDIATE, HIDDEN), (INTERMEDIATE // 128, HIDDEN // 128)),
        ((INTERMEDIATE, HIDDEN), (INTERMEDIATE // 128, HIDDEN // 128)),
        ((HIDDEN, INTERMEDIATE), (HIDDEN // 128, INTERMEDIATE // 128)),
    )
    for (weight, scale), shapes in zip(
        ((w1, s1), (w3, s3), (w2, s2)), expected_shapes, strict=True
    ):
        if tuple(weight.shape) != shapes[0] or tuple(scale.shape) != shapes[1]:
            raise ValueError(
                f"unexpected shared tensor shapes: {tuple(weight.shape)}, "
                f"{tuple(scale.shape)}"
            )

    hidden = (
        torch.randn(args.rows, HIDDEN, device=device, dtype=torch.float32) / 8
    ).to(torch.bfloat16)
    weights = [tensor.to(device) for tensor in (w1, w3, w2)]
    compact_scales = [tensor.to(device) for tensor in (s1, s3, s2)]
    packed_scales = [
        torch.empty(SCALE_MMA_BYTES, dtype=torch.uint8, device=device)
        for _ in range(3)
    ]
    for projection, weight, compact, packed in zip(
        ("w1", "w3", "w2"), weights, compact_scales, packed_scales, strict=True
    ):
        require_ok(
            native,
            native.ds41rt_cuda_ds4_flash_fp8_pack_block_scale_mma_async(
                tensor_buffer(compact),
                tensor_buffer(packed),
                weight.shape[0],
                weight.shape[1],
                stream,
            ),
            f"pack {projection} scale",
        )

    workspace_rows = CHUNK_ROWS
    gate = torch.empty(
        workspace_rows, INTERMEDIATE, dtype=torch.bfloat16, device=device
    )
    up = torch.empty_like(gate)
    activated = torch.empty_like(gate)
    output = torch.empty(args.rows, HIDDEN, dtype=torch.bfloat16, device=device)
    alpha = torch.ones(1, dtype=torch.float32, device=device)
    buffers = SharedBuffers(
        tensor_buffer(hidden),
        tensor_buffer(weights[0]),
        tensor_buffer(packed_scales[0]),
        tensor_buffer(weights[1]),
        tensor_buffer(packed_scales[1]),
        tensor_buffer(weights[2]),
        tensor_buffer(packed_scales[2]),
        tensor_buffer(gate),
        tensor_buffer(up),
        tensor_buffer(activated),
        tensor_buffer(output),
        tensor_buffer(alpha),
    )
    require_ok(
        native,
        native.ds41rt_cuda_ds4_flash_shared_expert_fp8_bf16_async(
            ctypes.byref(buffers), args.rows, stream
        ),
        "run shared expert",
    )
    torch.cuda.synchronize(device)

    reference_gate = linear_reference(hidden, weights[0], compact_scales[0]).float()
    reference_up = linear_reference(hidden, weights[1], compact_scales[1]).float()
    reference_up = reference_up.clamp(-10.0, 10.0)
    reference_gate = reference_gate.clamp(max=10.0)
    reference_activated = (functional.silu(reference_gate) * reference_up).to(
        torch.bfloat16
    )
    reference = linear_reference(reference_activated, weights[2], compact_scales[2])
    error = (output.float() - reference.float()).abs()
    max_abs = error.max().item()
    mean_abs = error.mean().item()
    if max_abs > args.atol:
        raise AssertionError(
            f"shared expert mismatch: max_abs={max_abs}, mean_abs={mean_abs}, "
            f"atol={args.atol}"
        )
    return {
        "layer": args.layer,
        "rows": args.rows,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--atol", type=float, default=0.0625)
    args = parser.parse_args()
    if not 1 <= args.rows <= 2048:
        parser.error("--rows must be in 1..2048")
    if not 0 <= args.layer < 43:
        parser.error("--layer must be in 0..42")
    print(json.dumps(run(args), sort_keys=True))


if __name__ == "__main__":
    main()
