#!/usr/bin/env python3
"""Export official V4.1 expert kernels and planner-owned native scratch layouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

import _pinned_sparkinfer


POINTER_SLOTS = (
    "a_ptr",
    "topk_ids_ptr",
    "topk_weights_ptr",
    "packed_a_ptr",
    "sfa_ptr",
    "packed_a_storage_ptr",
    "scale_storage_ptr",
    "intermediate_ptr",
    "barrier_count",
    "barrier_epoch",
    "pair_head",
    "producers_done_count",
    "all_work_published",
    "task_head",
    "task_tail",
    "task_ready_ptr",
    "task_expert_ptr",
    "task_m_tile_ptr",
    "task_slice_begin_ptr",
    "task_slice_count_ptr",
    "task_valid_rows_ptr",
    "tile_write_count_ptr",
    "b_w13",
    "sfb_w13_ptr",
    "b_down",
    "sfb_down_ptr",
    "sfb_w13_mx_ptr",
    "sfb_down_mx_ptr",
    "w13_residual_ptr",
    "down_residual_ptr",
    "w13_rp_ptr",
    "w13_sfb_rp_ptr",
    "down_rp_ptr",
    "down_sfb_rp_ptr",
    "row_counts",
    "expert_write_rows",
    "expert_tile_base",
    "input_global_scale",
    "alpha",
    "down_alpha",
    "global_scale",
    "scatter_ptr",
    "token_map_ptr",
    "token_weights_ptr",
)
SCALAR_SLOTS = (
    "num_tokens",
    "max_rows",
    "scatter_rows",
    "rows_padded",
    "max_tasks",
    "max_phys_tiles",
    "max_active_clusters",
    "stream",
)


def write_native_bridge(output_dir: Path, manifest: dict) -> None:
    """Reject exporter ABI drift before generating a native argument bridge."""
    entries, includes = [], []
    geometry = manifest["geometry"]
    for variant in manifest["variants"]:
        name = variant["name"]
        header = (output_dir / (name + ".h")).read_text()
        wrapper = re.search(
            r"static inline int32_t cute_dsl_\w+_wrapper\((.*?)\) \{", header, re.S
        )
        if wrapper is None:
            raise ValueError(f"missing C wrapper for {name}")
        parameters = tuple(
            re.search(r"(\w+)$", arg.strip())[1] for arg in wrapper[1].split(",")[1:]
        )
        if parameters != POINTER_SLOTS + SCALAR_SLOTS:
            raise ValueError(
                f"unsupported V4.1 native launch ABI for {name}: {parameters}"
            )
        tensor_slots = (
            set(POINTER_SLOTS[8:15]) | {"b_w13", "b_down"} | set(POINTER_SLOTS[34:41])
        )
        prefix = "ds41rt_" + name
        expected_declarations = [f"{prefix}_Kernel_Module_t*module"]
        expected_declarations += [
            f"{prefix}_Tensor_{slot}_t*{slot}"
            if slot in tensor_slots
            else f"void*{slot}"
            for slot in POINTER_SLOTS
        ]
        expected_declarations += [f"int32_t{slot}" for slot in SCALAR_SLOTS[:-1]]
        expected_declarations.append("cudaStream_tstream")
        if [
            re.sub(r"\s+", "", arg) for arg in wrapper[1].split(",")
        ] != expected_declarations:
            raise ValueError(f"unsupported exported parameter types for {name}")
        tensors = re.findall(
            r"typedef struct\s*\{([^}]+)\}\s*\w+_Tensor_\w+_t;", header
        )
        if len(tensors) != 16 or any(
            re.sub(r"\s+", "", body) != "void*data;" for body in tensors
        ):
            raise ValueError(f"unsupported exported tensor ABI for {name}")
        entry = re.search(
            r"void (_mlir_\w+)\(void \*\*args, int32_t num_args\)", header
        )
        if entry is None:
            raise ValueError(f"missing native launch entry for {name}")
        variant["native_entry"] = entry[1]
        packed = next(
            t for t in variant["scratch_tensors"] if t["name"] == "packed_input"
        )
        variant["rows_padded"] = packed["shape"][1]
        info = [
            1,
            int(manifest["role"] == "spark"),
            geometry["experts"],
            geometry["hidden"],
            geometry["intermediate"],
            geometry["kernel_intermediate"],
            geometry["topk"],
            variant["capacity_rows"],
            variant["core_scratch_nbytes"],
            variant["max_rows"],
            variant["rows_padded"],
            variant["task_capacity"],
            variant["physical_tiles"],
            variant["max_active_clusters"],
        ]
        entries.append(
            "{{"
            + ", ".join(map(str, info))
            + "}, "
            + f"_mlir_ds41rt_{name}_cuda_init, _mlir_ds41rt_{name}_cuda_load_to_device, "
            + entry[1]
            + "}"
        )
        includes.append(f'#include "{name}.h"')
    manifest["native_abi_version"] = 1
    manifest["pointer_slots"] = list(POINTER_SLOTS)
    lines = [
        "#pragma once",
        *includes,
        f"#define DS41RT_V41_CC_MINOR {manifest['capability'][1]}",
        f"#define DS41RT_V41_SMS {manifest['physical_sms']}",
        "#define DS41RT_V41_VARIANTS " + ", ".join(entries),
    ]
    (output_dir / "v41_expert_variants.h").write_text("\n".join(lines) + "\n")


def export(output_dir: Path, role: str, rows: tuple[int, ...]) -> None:
    # Export requires compiler IR, which executable-only cache entries omit.
    os.environ["SPARKINFER_COMPILE_DISK_CACHE"] = "0"
    os.environ["SPARKINFER_COMPILE_MEMORY_CACHE"] = "0"
    import torch
    from b12x.moe.fused_moe import _impl as moe

    torch.empty(1, dtype=torch.uint8, device="cuda")
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    capability = (properties.major, properties.minor)
    expected = (12, 1) if role == "spark" else (12, 0)
    if capability != expected:
        raise ValueError(
            f"{role} exports require native SM{expected[0]}{expected[1]}, got {capability}"
        )
    experts, intermediate, topk = (384, 576, 6) if role == "spark" else (128, 2304, 3)
    weight_plan = moe.plan_b12x_fp4_moe_weights(
        quant_modes="w4a8_mx",
        source_format="fp4_e8m0_k32",
        activation="silu_v41",
        params_dtype=torch.bfloat16,
        num_experts=experts,
        hidden_size=5120,
        intermediate_size=intermediate,
        w13_layout="w13",
    )
    kernel_intermediate = moe._dynamic_kernel_intermediate_size(intermediate, "w4a8_mx")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": 1,
        "role": role,
        "sparkinfer_revision": _pinned_sparkinfer.REVISION,
        "device": properties.name,
        "capability": list(capability),
        "physical_sms": properties.multi_processor_count,
        "geometry": {
            "experts": experts,
            "hidden": 5120,
            "intermediate": intermediate,
            "kernel_intermediate": kernel_intermediate,
            "topk": topk,
        },
        "output": "FP32 token-major route planes; reduction is a separate launch",
        "variants": [],
    }
    for requested_rows in rows:
        scratch_plan = moe.plan_tp_moe_scratch(
            moe.TPMoEScratchCaps(
                max_tokens=requested_rows,
                core_token_counts=(requested_rows,),
                num_topk=topk,
                device=device,
                weight_plan=weight_plan,
                quant_mode="w4a8_mx",
                deterministic_output=True,
                swiglu_limit=10,
            ),
            prewarm_launches=False,
        )
        plan = scratch_plan.launch_plan
        capacity = plan.routed_rows // topk
        core = scratch_plan._core_workspace_plan
        compiled, clusters = moe._get_dynamic_kernel(
            experts,
            capacity,
            5120,
            kernel_intermediate,
            topk,
            plan.max_rows,
            topk_ids_dtype=torch.int32,
            fast_math=True,
            activation="silu_v41",
            quant_mode="w4a8_mx",
            w4a8_repacked=True,
            deterministic_output=True,
            swiglu_limit=10,
        )
        name = f"v41_{role}_m{requested_rows}"
        compiled.export_to_c(str(output_dir), name, f"ds41rt_{name}")
        tensors = []
        offset = 0
        for spec in core.tensor_specs:
            alignment = max(16, spec.dtype.itemsize)
            offset = (offset + alignment - 1) // alignment * alignment
            nbytes = math.prod(spec.shape) * spec.dtype.itemsize
            tensors.append(
                {
                    "name": spec.name,
                    "shape": list(spec.shape),
                    "dtype": str(spec.dtype).removeprefix("torch."),
                    "offset": offset,
                    "nbytes": nbytes,
                    "init": spec.init,
                }
            )
            offset += nbytes
        assert offset == moe._core_workspace_nbytes(core)
        manifest["variants"].append(
            {
                "name": name,
                "requested_rows": requested_rows,
                "capacity_rows": capacity,
                "max_rows": plan.max_rows,
                "max_active_clusters": clusters,
                "physical_tiles": core.dynamic_physical_tiles,
                "task_capacity": core.dynamic_task_capacity,
                "core_scratch_nbytes": offset,
                "scratch_tensors": tensors,
                "files": {
                    ext: {
                        "name": name + ext,
                        "sha256": hashlib.sha256(
                            (output_dir / (name + ext)).read_bytes()
                        ).hexdigest(),
                    }
                    for ext in (".h", ".o")
                },
            }
        )
        print(f"exported {name}: capacity={capacity}, scratch={offset}", flush=True)
    write_native_bridge(output_dir, manifest)
    (output_dir / "v41_experts.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--role", choices=("spark", "coordinator"), required=True)
    parser.add_argument("--rows", default="1,16,80,256,1024,4096")
    args = parser.parse_args()
    rows = tuple(int(value) for value in args.rows.split(","))
    if (
        not rows
        or len(set(rows)) != len(rows)
        or any(value < 1 or value > 4096 for value in rows)
    ):
        parser.error("--rows must contain distinct positive capacities up to 4096")
    export(args.output_dir, args.role, rows)


if __name__ == "__main__":
    main()
