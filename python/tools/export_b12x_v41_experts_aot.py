#!/usr/bin/env python3
"""Export official V4.1 expert kernels and planner-owned native scratch layouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import _pinned_sparkinfer


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
