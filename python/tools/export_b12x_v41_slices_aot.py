#!/usr/bin/env python3
"""Export the experimental Spark slice pipeline through the existing expert ABI.

Explicit width only: serving policy selection awaits native qualification.
The ordinary expert exporter and production defaults are unchanged.
"""

import argparse
import hashlib
import os
import json
import re
from pathlib import Path

os.environ["B12X_COMPILE_DISK_CACHE"] = "0"
os.environ["B12X_COMPILE_MEMORY_CACHE"] = "0"
import _pinned_sparkinfer


def export(output, capacities, width, atomic_min_capacity=None):
    import torch
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_tensor
    from b12x._lib.utils import current_cuda_stream
    from b12x.moe._shared.kernels.v41_slice_pipeline import V41SlicePipeline
    from export_b12x_v41_experts_aot import export_input_quantizer

    torch.cuda.init()
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) not in ((12, 0), (12, 1)):
        raise ValueError("native Blackwell device required")
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        schema=1,
        experimental=True,
        role="spark",
        input_format="fp8_k32",
        sparkinfer_revision=_pinned_sparkinfer.REVISION,
        capability=[props.major, props.minor],
        physical_sms=props.multi_processor_count,
        width=width,
        variants=[],
    )
    includes, entries = [], []
    for capacity in capacities:
        atomic = atomic_min_capacity is not None and capacity >= atomic_min_capacity
        selected_width = width[capacity] if isinstance(width, dict) else width
        routes = capacity * 6
        planes = (576 + selected_width - 1) // selected_width
        specs = [
            (cutlass.Uint32, (capacity, 1280), (1320, 1)),
            (cutlass.Uint8, (capacity, 160), (5280, 1)),
            *[
                (cutlass.Uint32, (384 * n,), (1,))
                for n in (819200, 51200, 409600, 25600)
            ],
            (cutlass.Int32, (routes,), (1,)),
            (cutlass.Float32, (routes,), (1,)),
            (cutlass.Int32, (1,), (1,)),
            (cutlass.Int32, (384 * routes,), (1,)),
            (cutlass.Int32, (384,), (1,)),
            (cutlass.Int32, (384, 2), (2, 1)),
            (cutlass.Int32, (routes, 19), (19, 1)),
            (cutlass.Float32, (routes,), (1,)),
            (cutlass.Int32, (routes,), (1,)),
            (cutlass.Float32, (1,), (1,)) if atomic else
                (cutlass.Float32, (planes, routes, 5120), (routes * 5120, 5120, 1)),
            (cutlass.Float32, (capacity * 5120,), (1,)) if atomic else
                (cutlass.Float32, (routes, 5120), (5120, 1)),
        ]
        args = [
            make_fake_tensor(dtype, shape, stride, assumed_align=16)
            for dtype, shape, stride in specs
        ]
        label = f"v41_slices_m{capacity}_w{selected_width}"
        compiled = cute.compile(
            V41SlicePipeline(capacity, selected_width, atomic_tokens=atomic),
            *args,
            cutlass.Int32(capacity),
            current_cuda_stream(),
        )
        compiled.export_to_c(str(output), label, "ds41rt_" + label)
        header = (output / f"{label}.h").read_text()
        symbol = re.findall(
            r"void (_mlir_\w+)\(void \*\*args, int32_t num_args\);", header
        )
        if len(symbol) != 1:
            raise ValueError("unexpected slice export entry")
        names = (
            "x",
            "xs",
            "w13",
            "s13",
            "w2",
            "s2",
            "ids",
            "routing",
            "live",
            "packed",
            "counts",
            "prefixes",
            "metadata",
            "grouped",
            "inverse",
            "partial",
            "output",
        )
        signature = re.search(
            r"static inline int32_t cute_dsl_\w+_wrapper\(([^)]*)\)", header
        )
        prefix = "ds41rt_" + label
        declarations = (
            [f"{prefix}_Kernel_Module_t*module"]
            + [f"{prefix}_Tensor_{name}_t*{name}" for name in names]
            + ["int32_trows", "cudaStream_tstream"]
        )
        if (
            signature is None
            or [re.sub(r"\s+", "", x) for x in signature[1].split(",")] != declarations
        ):
            raise ValueError("unexpected slice parameter ABI")
        arguments = re.search(r"void \*args\[20\] = \{([^}]+)\}", header)
        if arguments is None or re.sub(r"\s+", "", arguments[1]) != ",".join(
            name for name in (*names, "&rows", "&stream", "&ret")
        ):
            raise ValueError("unexpected slice argument order")
        # Static tensor layouts must expose exactly one data pointer each.
        bodies = re.findall(r"typedef struct\s*\{([^}]+)\}\s*\w+_Tensor_\w+_t;", header)
        if len(bodies) != 17 or any(
            re.sub(r"\s+", "", b) != "void*data;" for b in bodies
        ):
            raise ValueError("unexpected slice tensor ABI")
        offset, scratch, offsets = 0, [], [None] * 44
        # Existing weight binder owns slots 22..33, 38/39. Other unused slots
        # point at valid dummy scratch to preserve public nonnull validation.
        scratch_slots = [3, 4, 5, 6, 7, 8, 9, 10, 41]
        for index, slot in zip(range(8, 17), scratch_slots):
            dtype, shape, _ = specs[index]
            size = 4
            for dim in shape:
                size *= dim
            offset = (offset + 15) // 16 * 16
            offsets[slot] = offset
            scratch.append(dict(slot=slot, offset=offset, nbytes=size, shape=shape))
            offset += size
        for slot in (37, 40):
            offset = (offset + 15) // 16 * 16
            offsets[slot] = offset
            offset += 384 * 4
        for slot in list(range(3, 22)) + [34, 35, 36, 42, 43]:
            if offsets[slot] is None:
                offsets[slot] = 0
        includes.append(f'#include "{label}.h"')
        # The old entry gets 44 pointer addresses, seven scalar addresses,
        # stream and status. Repackage them into the generated static ABI.
        mapping = [0, None, 30, 31, 32, 33, 1, 2, *scratch_slots]
        mapped = [
            f"args[{slot}]" if slot is not None else "&scales" for slot in mapping
        ]
        includes.extend(
            [
                f"static void {label}_bridge(void** args, int32_t count) {{",
                "  if (count != 53) { *static_cast<int32_t*>(args[52]) = 1; return; }",
                "  void* scales = static_cast<char*>(*static_cast<void**>(args[0])) + 5120;",
                f"  void* mapped[] = {{{', '.join(mapped)}, args[44], args[51], args[52]}};",
                f"  {symbol[0]}(mapped, 20);",
                "}",
            ]
        )
        info = [
            3 if atomic else 2,
            1,
            384,
            5120,
            576,
            640,
            6,
            capacity,
            offset,
            capacity,
            capacity,
            routes,
            routes,
            props.multi_processor_count,
            7,
        ]
        prefix = "_mlir_ds41rt_" + label
        entries.append(
            "{{"
            + ",".join(map(str, info))
            + "},"
            + prefix
            + "_cuda_init,"
            + prefix
            + "_cuda_load_to_device,"
            + label
            + "_bridge,{"
            + ",".join("UINT64_MAX" if x is None else str(x) for x in offsets)
            + "}}"
        )
        manifest["variants"].append(
            dict(
                name=label,
                width=selected_width,
                output_kind="fp32_tokens" if atomic else "fp32_routes",
                native_abi_version=3 if atomic else 2,
                capacity_rows=capacity,
                core_scratch_nbytes=offset,
                scratch=scratch,
                scratch_pointer_offsets=offsets,
                native_entry=symbol[0],
            )
        )
    (output / "v41_expert_variants.h").write_text(
        "\n".join(
            [
                "#pragma once",
                *includes,
                f"#define DS41RT_V41_CC_MINOR {props.minor}",
                f"#define DS41RT_V41_SMS {props.multi_processor_count}",
                "#define DS41RT_V41_VARIANTS " + ",".join(entries),
                "#define DS41RT_V41_OUTPUT_KIND(capacity) (" +
                (" || ".join(f"((capacity)=={v['capacity_rows']})" for v in manifest['variants']
                             if v['output_kind'] == 'fp32_tokens') or "0") + " ? 1u : 0u)",
                "",
            ]
        )
    )
    export_input_quantizer(output, manifest)
    manifest["artifact_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output.iterdir()
        if path.suffix in (".h", ".o")
    }
    (output / "v41_experts.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", default="1,16,80")
    parser.add_argument(
        "--width",
        required=True,
        help="64/128/192 or explicit capacity:width pairs, e.g. 1:64,16:192,80:192",
    )
    parser.add_argument("--atomic-min-capacity", type=int, choices=[256, 1024, 4096],
                        help="Use ABI 3 direct FP32 token accumulation at these larger capacities")
    args = parser.parse_args()
    capacities = tuple(int(x) for x in args.rows.split(","))
    if (
        not capacities
        or len(set(capacities)) != len(capacities)
        or any(x < 1 or x > 4096 for x in capacities)
    ):
        parser.error("rows must be unique capacities in 1..4096")
    try:
        if ":" in args.width:
            pairs = [tuple(map(int, x.split(":"))) for x in args.width.split(",")]
            width = dict(pairs)
            if len(width) != len(pairs) or set(width) != set(capacities):
                raise ValueError("width map must cover every capacity exactly once")
            widths = width.values()
        else:
            width = int(args.width)
            widths = [width]
        if any(w not in (64, 128, 192) for w in widths):
            raise ValueError("width must be 64, 128 or 192")
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    export(args.output_dir, capacities, width, args.atomic_min_capacity)
