#!/usr/bin/env python3
"""Export native DeepSeek-V4 Flash/Pro expert-TP4 SparkInfer kernels.

This is deliberately separate from the GLM/B12X exporter.  The two model
families have different hidden sizes, route counts, scale encodings, and
activation contracts; sharing an export script would make it too easy to
silently compile one model with the other's ABI.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path


TP_WORLD_SIZE = 4
PREFILL_REGIMES = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
PREFILL_ROUTE_BLOCK_SIZE = 32
FLASH_DIRECT_M4_CAPACITY = 4
FLASH_DIRECT_M4_MIN_ACTIVE = 2
FLASH_DIRECT_M4_BLOCK_SIZE = 16
FLASH_DIRECT_M6_CAPACITY = 8
FLASH_DIRECT_M6_BLOCK_SIZE = 16
EXL3_TIERS = (2, 3)
DEFAULT_EXL3_BITS = 2
EXL3_CODEBOOK = "mcg"
EXL3_TRELLIS_LUT_BYTES = 1 << 12
EXL3_TILE_CONFIGS = {
    # Flash's H4096/I512 TP4 K2 kernel is measurably faster with the narrower
    # 128-thread tile for both M1 decode and M2048-capacity prefill.  Keep Pro
    # on its independently established geometry until its real weights exist
    # and the same benchmark can qualify that different H7168/I768 shape.
    "flash": (64, 128, 64, 128),
    "pro": (64, 256, 64, 256),
}
DEFAULT_TRANSFORMER_BLOCKS = {"flash": 46, "pro": 64}


@dataclass(frozen=True)
class KernelProfile:
    variant: str
    model: str
    hidden_size: int
    global_intermediate_size: int
    num_experts: int
    top_k: int = 6
    swiglu_limit: float = 10.0
    native_fp4: bool = False
    decode_grid_x: int = 32

    def __post_init__(self) -> None:
        if self.variant not in {"flash", "pro"}:
            raise ValueError(f"unsupported DeepSeek V4 variant {self.variant!r}")
        if self.hidden_size <= 0 or self.hidden_size % 128:
            raise ValueError("hidden_size must be a positive H128 multiple")
        if self.global_intermediate_size % (TP_WORLD_SIZE * 128):
            raise ValueError("intermediate size must split into four H128 slices")
        if not 0 < self.top_k <= self.num_experts:
            raise ValueError("top_k must be in 1..num_experts")

    @property
    def symbol_prefix(self) -> str:
        return f"ds4_{self.variant}"

    @property
    def macro_prefix(self) -> str:
        return f"DS4RT_DS4_{self.variant.upper()}"

    @property
    def local_intermediate_size(self) -> int:
        return self.global_intermediate_size // TP_WORLD_SIZE

    @property
    def projection_trellis_bytes(self) -> int:
        return (
            self.hidden_size
            * self.local_intermediate_size
            * DEFAULT_EXL3_BITS
            // 8
        )

    @property
    def resident_weight_bytes_per_block(self) -> int:
        return 3 * self.num_experts * self.projection_trellis_bytes

    @property
    def resident_rotation_bytes_per_block(self) -> int:
        return (
            3 * self.num_experts * self.hidden_size * 2
            + self.num_experts * 3 * self.local_intermediate_size * 2
        )

    @property
    def resident_scalar_metadata_bytes_per_block(self) -> int:
        return (
            (self.num_experts + 1) * 4
            + 16
            + self.num_experts * 4
            + EXL3_TRELLIS_LUT_BYTES
        )


KERNEL_PROFILES = {
    "flash": KernelProfile(
        variant="flash",
        model="deepseek_v4_flash_0731",
        hidden_size=4096,
        global_intermediate_size=2048,
        num_experts=256,
        native_fp4=True,
    ),
    "pro": KernelProfile(
        variant="pro",
        model="deepseek_v4_pro_preview",
        hidden_size=7168,
        global_intermediate_size=3072,
        num_experts=384,
        native_fp4=False,
    ),
}


def profile_manifest(profile: KernelProfile, *, transformer_blocks: int) -> dict[str, object]:
    if transformer_blocks <= 0:
        raise ValueError("transformer_blocks must be positive")
    per_block = (
        profile.resident_weight_bytes_per_block
        + profile.resident_rotation_bytes_per_block
        + profile.resident_scalar_metadata_bytes_per_block
    )
    return {
        **asdict(profile),
        "tp_world_size": TP_WORLD_SIZE,
        "local_intermediate_size": profile.local_intermediate_size,
        "trellis_bits": DEFAULT_EXL3_BITS,
        "trellis_codebook": EXL3_CODEBOOK,
        "trellis_tile_config": list(EXL3_TILE_CONFIGS[profile.variant]),
        "trellis_lut_bytes": EXL3_TRELLIS_LUT_BYTES,
        "projection_trellis_bytes": profile.projection_trellis_bytes,
        "resident_weight_bytes_per_block": profile.resident_weight_bytes_per_block,
        "resident_rotation_bytes_per_block": profile.resident_rotation_bytes_per_block,
        "resident_scalar_metadata_bytes_per_block": (
            profile.resident_scalar_metadata_bytes_per_block
        ),
        "resident_total_bytes_per_block": per_block,
        "transformer_blocks": transformer_blocks,
        "resident_total_bytes_per_rank": per_block * transformer_blocks,
    }


def prepare_export(output_dir: Path):
    # Metadata-only profile inspection must remain CUDA- and SparkInfer-free.
    # Verify and expose the pinned source immediately before a real export.
    import _pinned_sparkinfer  # noqa: F401

    # A disk-cache hit is executable-only and has no IR for export_to_c().
    os.environ["SPARKINFER_COMPILE_DISK_CACHE"] = "0"
    os.environ["SPARKINFER_COMPILE_MEMORY_CACHE"] = "0"

    import cuda.bindings.driver as cuda
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    torch.empty(1, dtype=torch.uint8, device=device)
    return cuda, torch, device


def export_kernels(output_dir: Path, target_sms: int, profile: KernelProfile) -> None:
    cuda, torch, device = prepare_export(output_dir)
    from b12x.moe._shared.kernels.w4a16.host import (
        max_packed_route_slots,
        select_route_block_size_m,
    )
    from b12x.moe._shared.kernels.w4a16.kernel import (
        W4A16FusedMoeKernel,
        _w4a16_fused_persistent_grid_x,
        compile_w4a16_fused_moe,
        compile_w4a16_topk_sum,
    )
    from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
        compile_mixed_trellis,
    )
    from b12x._lib.dense_gemm import (
        compile_dense_gemm_mxfp8_aot,
        compile_dense_gemm_fused_quant_a_aot,
    )
    from b12x._lib.quant.mxfp8_rows import compile_mxfp8_rows_quant_aot

    launch = W4A16FusedMoeKernel.__call__
    launch.__annotations__["stream"] = cuda.CUstream
    wrapped = getattr(launch, "__wrapped__", None)
    if wrapped is not None:
        wrapped.__annotations__["stream"] = cuda.CUstream

    properties = torch.cuda.get_device_properties(device)
    physical_sms = int(properties.multi_processor_count)
    if target_sms <= 0 or target_sms > physical_sms:
        raise ValueError(
            f"target_sms must be in 1..{physical_sms} on this export host, got {target_sms}"
        )
    max_shared_mem = int(properties.shared_memory_per_block_optin)

    # Kernel code derives cooperative-barrier storage from device properties,
    # independent of the planner's explicit `sms` argument.  Present the GB10
    # Spark target while exporting on the larger coordinator GPU.
    physical_get_device_properties = torch.cuda.get_device_properties

    class TargetDeviceProperties:
        def __init__(self, base: object) -> None:
            self._base = base

        @property
        def multi_processor_count(self) -> int:
            return target_sms

        def __getattr__(self, name: str) -> object:
            return getattr(self._base, name)

    def target_get_device_properties(device_arg: object = None) -> object:
        return TargetDeviceProperties(physical_get_device_properties(device_arg))

    torch.cuda.get_device_properties = target_get_device_properties

    prefix = profile.symbol_prefix
    macro_prefix = profile.macro_prefix
    config_lines = [
        "#pragma once",
        f"#define {macro_prefix}_HIDDEN_SIZE {profile.hidden_size}",
        f"#define {macro_prefix}_TP_WORLD_SIZE {TP_WORLD_SIZE}",
        (
            f"#define {macro_prefix}_TP_INTERMEDIATE_SIZE "
            f"{profile.local_intermediate_size}"
        ),
        f"#define {macro_prefix}_NUM_EXPERTS {profile.num_experts}",
        f"#define {macro_prefix}_TOP_K {profile.top_k}",
        f"#define {macro_prefix}_SWIGLU_LIMIT {profile.swiglu_limit}f",
        (
            f"#define {macro_prefix}_SHARED_INTERMEDIATE_SIZE "
            f"{profile.global_intermediate_size}"
        ),
        f"#define {macro_prefix}_PREFILL_MAX_ROWS {PREFILL_REGIMES[-1]}",
    ]
    if profile.variant == "flash":
        config_lines.append(
            f"#define {macro_prefix}_EXL3_K2_DIRECT_M4_MIN_ACTIVE "
            f"{FLASH_DIRECT_M4_MIN_ACTIVE}"
        )
    metadata_lines = [
        f"w4a16_target_sms={target_sms}",
        f"model={profile.model}",
        f"variant={profile.variant}",
        "parallelism=expert_tp4",
        f"hidden_size={profile.hidden_size}",
        f"global_intermediate_size={profile.global_intermediate_size}",
        f"intermediate_size={profile.local_intermediate_size}",
        f"num_experts={profile.num_experts}",
        f"top_k={profile.top_k}",
        "activation=silu",
        f"swiglu_limit={profile.swiglu_limit}",
        f"native_fp4={int(profile.native_fp4)}",
        "exl3_weight_layout=trellis3_t256",
        "exl3_trellis_bits=" + ",".join(str(bits) for bits in EXL3_TIERS),
        "exl3_codebook=mcg",
        "exl3_compute=fp16_full_rotation_bf16_input_f32_partial",
        f"resident_weight_bytes_per_block={profile.resident_weight_bytes_per_block}",
        f"resident_rotation_bytes_per_block={profile.resident_rotation_bytes_per_block}",
    ]
    if profile.native_fp4:
        metadata_lines.extend(("weight_layout=packed", "scale_format=e8m0_k32"))

    def export_w4a16(
        *,
        rows: int,
        label: str,
        direct_topk: bool,
        fused_sum: bool,
        block_size: int,
        grid_x: int | None = None,
    ) -> None:
        packed_route_slots = max_packed_route_slots(
            rows * profile.top_k,
            block_size,
            profile.num_experts,
        )
        max_m_blocks = (
            rows * profile.top_k
            if direct_topk
            else (packed_route_slots + block_size - 1) // block_size
        )
        fused = compile_w4a16_fused_moe(
            size_m=rows,
            hidden_size=profile.hidden_size,
            intermediate_size=profile.local_intermediate_size,
            num_experts=profile.num_experts,
            top_k=profile.top_k,
            activation="silu",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            moe_block_size=block_size,
            max_m_blocks=max_m_blocks,
            element_dtype="bf16",
            fast_math=True,
            sms=target_sms,
            max_shared_mem=max_shared_mem,
            swiglu_limit=profile.swiglu_limit,
            weight_layout="packed",
            scale_format="e8m0_k32",
            direct_topk_routes=direct_topk,
            tc_decode_fused_sum=fused_sum,
        )
        export_name = f"{prefix}_tp4_w4a16_{label}"
        fused.compiled.export_to_c(
            str(output_dir),
            export_name,
            f"ds4rt_{export_name}",
        )
        if grid_x is None:
            grid_x = _w4a16_fused_persistent_grid_x(
                fused=fused,
                m=rows,
                topk=profile.top_k,
                intermediate_size=profile.local_intermediate_size,
                activation="silu",
                direct_topk_routes=direct_topk,
                sms=target_sms,
            )
        macro = label.upper()
        config_lines.extend(
            [
                f"#define {macro_prefix}_W4A16_{macro}_GRID_X {grid_x}",
                f"#define {macro_prefix}_W4A16_{macro}_BLOCK_SIZE {block_size}",
                (
                    f"#define {macro_prefix}_W4A16_{macro}_PACKED_ROUTE_SLOTS "
                    f"{packed_route_slots}"
                ),
                (
                    f"#define {macro_prefix}_W4A16_{macro}_MAX_M_BLOCKS "
                    f"{max_m_blocks}"
                ),
            ]
        )
        metadata_lines.append(
            f"{label}=grid:{grid_x},block:{block_size},"
            f"route_slots:{packed_route_slots},max_m_blocks:{max_m_blocks},"
            f"direct_topk:{int(direct_topk)},tc_decode_fused_sum:{int(fused_sum)}"
        )

    if profile.native_fp4:
        export_w4a16(
            rows=1,
            label="decode_m1_fused_sum",
            direct_topk=True,
            fused_sum=True,
            block_size=select_route_block_size_m(
                1, profile.top_k, profile.num_experts
            ),
            grid_x=profile.decode_grid_x,
        )
        for rows in PREFILL_REGIMES:
            export_w4a16(
                rows=rows,
                label=f"prefill_m{rows}_topk{profile.top_k}",
                direct_topk=False,
                fused_sum=False,
                block_size=PREFILL_ROUTE_BLOCK_SIZE,
            )

    def export_exl3(
        *,
        bits: int,
        rows: int,
        label: str,
        direct_topk: bool,
        block_size: int,
    ) -> None:
        packed_route_slots = max_packed_route_slots(
            rows * profile.top_k,
            block_size,
            profile.num_experts,
        )
        max_m_blocks = (
            rows * profile.top_k
            if direct_topk
            else (packed_route_slots + block_size - 1) // block_size
        )
        fused = compile_w4a16_fused_moe(
            size_m=rows,
            hidden_size=profile.hidden_size,
            intermediate_size=profile.local_intermediate_size,
            num_experts=profile.num_experts,
            top_k=profile.top_k,
            activation="silu",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            moe_block_size=block_size,
            max_m_blocks=max_m_blocks,
            element_dtype="fp16",
            fast_math=True,
            sms=target_sms,
            max_shared_mem=max_shared_mem,
            swiglu_limit=profile.swiglu_limit,
            weight_layout="trellis3_t256",
            scale_format="e4m3_k32",
            w13_layout="trellis3_t256_proj",
            trellis_bits=bits,
            # Checkpoints are exllamav3_trellis_mcg.  Never inherit the
            # compiler default: b12x 1.1 changed it to SQG-XOR-Cheb T12.
            trellis_codebook=EXL3_CODEBOOK,
            direct_topk_routes=direct_topk,
            use_expert_map=direct_topk,
            force_tile_config=EXL3_TILE_CONFIGS[profile.variant],
            intermediate_rotation=True,
            full_rotation=True,
            rotation_input_dtype="bf16",
            broadcast_suh=False,
        )
        tier = f"k{bits}"
        export_name = f"{prefix}_tp4_exl3_{tier}_{label}"
        fused.compiled.export_to_c(
            str(output_dir),
            export_name,
            f"ds4rt_{export_name}",
        )
        grid_x = _w4a16_fused_persistent_grid_x(
            fused=fused,
            m=rows,
            topk=profile.top_k,
            intermediate_size=profile.local_intermediate_size,
            activation="silu",
            direct_topk_routes=direct_topk,
            sms=target_sms,
        )
        macro = label.upper()
        config_lines.extend(
            [
                f"#define {macro_prefix}_EXL3_{tier.upper()}_{macro}_GRID_X {grid_x}",
                f"#define {macro_prefix}_EXL3_{tier.upper()}_{macro}_BLOCK_SIZE {block_size}",
                (
                    f"#define {macro_prefix}_EXL3_{tier.upper()}_{macro}_PACKED_ROUTE_SLOTS "
                    f"{packed_route_slots}"
                ),
                (
                    f"#define {macro_prefix}_EXL3_{tier.upper()}_{macro}_MAX_M_BLOCKS "
                    f"{max_m_blocks}"
                ),
            ]
        )
        metadata_lines.append(
            f"exl3_{tier}_{label}=grid:{grid_x},block:{block_size},"
            f"route_slots:{packed_route_slots},max_m_blocks:{max_m_blocks},"
            f"direct_topk:{int(direct_topk)},full_rotation:1,bits:{bits},"
            f"codebook:{EXL3_CODEBOOK}"
        )

    exl3_tiers = EXL3_TIERS if profile.variant == "flash" else (DEFAULT_EXL3_BITS,)
    for bits in exl3_tiers:
        export_exl3(
            bits=bits,
            rows=1,
            label="decode_m1",
            direct_topk=True,
            block_size=select_route_block_size_m(
                1, profile.top_k, profile.num_experts
            ),
        )
        if profile.variant == "flash":
            # The exact-capacity-four direct kernel wins for every active M2--M4
            # shape. Keep M5 on the packed capacity-eight path and M6 on its
            # independently qualified direct-capacity-eight kernel.
            export_exl3(
                bits=bits,
                rows=FLASH_DIRECT_M4_CAPACITY,
                label="decode_m4_direct_m4",
                direct_topk=True,
                block_size=FLASH_DIRECT_M4_BLOCK_SIZE,
            )
            # Preserve the independently qualified six-row joint issue path used
            # when the adaptive policy selects its longest proposal.
            export_exl3(
                bits=bits,
                rows=FLASH_DIRECT_M6_CAPACITY,
                label="decode_m8_direct_m6",
                direct_topk=True,
                block_size=FLASH_DIRECT_M6_BLOCK_SIZE,
            )
        for rows in PREFILL_REGIMES:
            export_exl3(
                bits=bits,
                rows=rows,
                label=f"prefill_m{rows}_topk{profile.top_k}",
                direct_topk=False,
                block_size=PREFILL_ROUTE_BLOCK_SIZE,
            )

    if profile.variant == "flash":
        # Mixed K2/K3 artifacts keep a single cooperative FC1/activation/FC2
        # grid. Expert counts are runtime scalars, so the same binary covers
        # the balanced 230/26 and 231/25 layer partitions.
        for rows in (1, *PREFILL_REGIMES):
            # Decode widths through the maximum dSpark joint issue (M6 in an
            # M8 capacity kernel) use direct combined route ids. This preserves
            # the one-grid tier dispatcher while avoiding generic block-32
            # route packing and restoring the qualified 128-thread Flash tile.
            direct_topk = rows <= FLASH_DIRECT_M6_CAPACITY
            block_size = (
                select_route_block_size_m(
                    rows, profile.top_k, profile.num_experts
                )
                if rows == 1
                else FLASH_DIRECT_M6_BLOCK_SIZE
                if direct_topk
                else PREFILL_ROUTE_BLOCK_SIZE
            )
            packed_route_slots = (
                rows * profile.top_k
                if direct_topk
                else max_packed_route_slots(
                    rows * profile.top_k,
                    block_size,
                    profile.num_experts,
                )
            )
            max_m_blocks = (
                rows * profile.top_k
                if direct_topk
                else (packed_route_slots + block_size - 1) // block_size
            )
            mixed = compile_mixed_trellis(
                size_m=rows,
                hidden_size=profile.hidden_size,
                intermediate_size=profile.local_intermediate_size,
                tier0_num_experts=230,
                tier1_num_experts=26,
                top_k=profile.top_k,
                max_m_blocks=max_m_blocks,
                sms=target_sms,
                max_shared_mem=max_shared_mem,
                force_tile_config=(
                    EXL3_TILE_CONFIGS[profile.variant]
                    if direct_topk
                    else (128, 128, 128, 128)
                ),
                tier0_bits=2,
                tier1_bits=3,
                moe_block_size=block_size,
                rotation_input_dtype="bf16",
                direct_topk_routes=direct_topk,
            )
            label = f"mixed_k2_k3_m{rows}"
            export_name = f"{prefix}_tp4_exl3_{label}"
            mixed.compiled.export_to_c(
                str(output_dir),
                export_name,
                f"ds4rt_{export_name}",
            )
            macro = label.upper()
            grid_x = mixed.blocks_per_sm * target_sms
            config_lines.extend(
                [
                    f"#define {macro_prefix}_EXL3_{macro}_GRID_X {grid_x}",
                    (
                        f"#define {macro_prefix}_EXL3_{macro}_BLOCK_SIZE "
                        f"{block_size}"
                    ),
                    (
                        f"#define {macro_prefix}_EXL3_{macro}_PACKED_ROUTE_SLOTS "
                        f"{packed_route_slots}"
                    ),
                    (
                        f"#define {macro_prefix}_EXL3_{macro}_DIRECT_TOPK "
                        f"{int(direct_topk)}"
                    ),
                    (
                        f"#define {macro_prefix}_EXL3_{macro}_MAX_M_BLOCKS "
                        f"{max_m_blocks}"
                    ),
                ]
            )
            metadata_lines.append(
                f"exl3_{label}=grid:{grid_x},block:{block_size},"
                f"route_slots:{packed_route_slots},max_m_blocks:{max_m_blocks},"
                f"direct_topk:{int(direct_topk)},"
                "tier0:K2,tier1:K3,full_rotation:1,one_grid:1"
            )

    topk_sum = compile_w4a16_topk_sum(
        m=PREFILL_REGIMES[-1],
        topk=profile.top_k,
        hidden_size=profile.hidden_size,
        element_dtype="fp16",
        full_rotation=True,
        num_experts=profile.num_experts,
        route_num_experts=profile.num_experts,
        route_ids_dtype=torch.int32,
        use_expert_map=True,
        broadcast_svh=False,
    )
    topk_sum.compiled.export_to_c(
        str(output_dir),
        f"{prefix}_tp4_exl3_k2_topk{profile.top_k}_sum",
        f"ds4rt_{prefix}_tp4_exl3_k2_topk{profile.top_k}_sum",
    )
    metadata_lines.append(
        f"exl3_topk{profile.top_k}_sum="
        "route:fp16,output:f32,down_svh:per_expert,expert_map:1"
    )

    for label, rows, size_n, size_k in (
        ("shared_up_m1", 1, profile.global_intermediate_size, profile.hidden_size),
        ("shared_up_m8", 8, profile.global_intermediate_size, profile.hidden_size),
        ("shared_down_m1", 1, profile.hidden_size, profile.global_intermediate_size),
        ("shared_down_m8", 8, profile.hidden_size, profile.global_intermediate_size),
    ):
        kernel = compile_dense_gemm_fused_quant_a_aot(
            size_m=rows,
            size_n=size_n,
            size_k=size_k,
            activation_scale_block_size=128,
            expected_m=rows,
            device=device,
        )
        export_name = f"{prefix}_{label}"
        kernel.export_to_c(
            str(output_dir),
            export_name,
            f"ds4rt_{export_name}",
        )
        metadata_lines.append(
            f"{label}=m:{rows},n:{size_n},k:{size_k},activation_scale:k128"
        )

    if profile.variant == "pro":
        # Pro's shared expert is large enough that replaying the decode-tuned
        # M8 kernel over a prefill chunk rereads each weight hundreds of times.
        # Quantize each activation matrix once, then run ordinary full-width
        # MXFP8 GEMMs.  These kernels execute on the RTX coordinator, so retain
        # its physical SM geometry rather than the Spark expert target above.
        for label, size_k in (
            ("shared_input_quant_m2048", profile.hidden_size),
            ("shared_activated_quant_m2048", profile.global_intermediate_size),
        ):
            kernel = compile_mxfp8_rows_quant_aot(
                size_k=size_k,
                source_dtype=torch.bfloat16,
                scale_block_size=128,
                expected_m=PREFILL_REGIMES[-1],
            )
            export_name = f"{prefix}_{label}"
            kernel.export_to_c(
                str(output_dir),
                export_name,
                f"ds4rt_{export_name}",
            )
            metadata_lines.append(
                f"{label}=m:runtime<=2048,k:{size_k},activation_scale:k128"
            )

        for label, size_n, size_k in (
            (
                "shared_up_m2048",
                profile.global_intermediate_size,
                profile.hidden_size,
            ),
            (
                "shared_down_m2048",
                profile.hidden_size,
                profile.global_intermediate_size,
            ),
        ):
            kernel = compile_dense_gemm_mxfp8_aot(
                size_m=PREFILL_REGIMES[-1],
                size_n=size_n,
                size_k=size_k,
                expected_m=PREFILL_REGIMES[-1],
                sfb_k_replicated=True,
                sm_count=physical_sms,
                device=device,
            )
            export_name = f"{prefix}_{label}"
            kernel.export_to_c(
                str(output_dir),
                export_name,
                f"ds4rt_{export_name}",
            )
            metadata_lines.append(
                f"{label}=m:runtime<=2048,n:{size_n},k:{size_k},"
                f"coordinator_sms:{physical_sms},weight_scale:k128_replicated"
            )

    (output_dir / f"{prefix}_spark_moe_aot_config.h").write_text(
        "\n".join(config_lines) + "\n",
        encoding="ascii",
    )
    (output_dir / f"{prefix}_spark_moe_aot.meta").write_text(
        "\n".join(metadata_lines) + "\n",
        encoding="ascii",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export DeepSeek-V4 Flash/Pro expert-TP4 SparkInfer MoE kernels."
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-sms", type=int, default=48)
    parser.add_argument(
        "--model-variant", choices=tuple(KERNEL_PROFILES), default="flash"
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="print the exact strict-TP4 resident geometry without initializing CUDA",
    )
    parser.add_argument(
        "--transformer-blocks",
        type=int,
        help="target blocks including integrated dSpark blocks for --describe",
    )
    args = parser.parse_args()
    profile = KERNEL_PROFILES[args.model_variant]
    if args.describe:
        blocks = args.transformer_blocks
        if blocks is None:
            blocks = DEFAULT_TRANSFORMER_BLOCKS[profile.variant]
        print(json.dumps(profile_manifest(profile, transformer_blocks=blocks), sort_keys=True))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --describe is used")
    export_kernels(args.output_dir.resolve(), args.target_sms, profile)


if __name__ == "__main__":
    main()
