#!/usr/bin/env python3
"""Extract embedded CUDA ELF images from CuTe AOT objects and inspect resources.

CuTe embeds cubins as host data, which cuobjdump does not discover directly.
Keep the input object unchanged and record hashes of both object and images.
Reported shared memory is static; dynamic launch storage must be accounted for
separately. This is an offline diagnostic, not a serving or loading path.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess


def cuda_images(data):
    offset = 0
    while (offset := data.find(b"\x7fELF", offset)) >= 0:
        start = offset
        offset += 4
        if len(data) - start < 64:
            continue
        blob = memoryview(data)[start:]
        if bytes(blob[:6]) != b"\x7fELF\x02\x01":
            continue
        if struct.unpack_from("<H", blob, 18)[0] != 190:
            continue
        phoff, shoff = struct.unpack_from("<QQ", blob, 32)
        phsize, phcount, shsize, shcount = struct.unpack_from("<HHHH", blob, 54)
        if phsize != 56 or shsize != 64 or not shcount:
            raise ValueError(f"unsupported CUDA ELF tables at {start}")
        end = max(64, phoff + phsize * phcount, shoff + shsize * shcount)
        if end > len(blob):
            raise ValueError(f"truncated CUDA ELF tables at {start}")
        for i in range(shcount):
            entry = shoff + i * shsize
            kind = struct.unpack_from("<I", blob, entry + 4)[0]
            pos, size = struct.unpack_from("<QQ", blob, entry + 24)
            if kind != 8:  # SHT_NOBITS has no file payload.
                end = max(end, pos + size)
        for i in range(phcount):
            entry = phoff + i * phsize
            pos = struct.unpack_from("<Q", blob, entry + 8)[0]
            size = struct.unpack_from("<Q", blob, entry + 32)[0]
            end = max(end, pos + size)
        if end > len(blob):
            raise ValueError(f"truncated CUDA ELF payload at {start}")
        yield start, bytes(blob[:end])
        offset = start + end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("object", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuobjdump", default="/usr/local/cuda/bin/cuobjdump")
    args = parser.parse_args()
    data = args.object.read_bytes()
    images = list(cuda_images(data))
    if not images:
        raise ValueError("no embedded ELF64 little-endian CUDA image found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "object": str(args.object.resolve()),
        "object_sha256": hashlib.sha256(data).hexdigest(),
        "cuobjdump_version": subprocess.check_output(
            [args.cuobjdump, "--version"], text=True
        ),
        "images": [],
    }
    for index, (offset, blob) in enumerate(images):
        path = args.output_dir / f"{args.object.stem}.{index}.cubin"
        if path.resolve() == args.object.resolve():
            raise ValueError("output would overwrite input")
        path.write_bytes(blob)
        resources = subprocess.check_output(
            [args.cuobjdump, "--dump-resource-usage", str(path)], text=True
        )
        if "Function " not in resources:
            raise ValueError("cuobjdump returned no kernel resource records")
        manifest["images"].append({
            "offset": offset, "bytes": len(blob), "path": str(path.resolve()),
            "sha256": hashlib.sha256(blob).hexdigest(), "resources": resources,
        })
    if args.object.read_bytes() != data:
        raise ValueError("input object changed during inspection")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
