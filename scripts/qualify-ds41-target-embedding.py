#!/usr/bin/env python3
"""Compare native target/draft embedding payloads to the official checkpoint on CPU."""
import argparse
import ast
import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "reference-dir", "vectors-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "docs/ds41-reference-lock.json").read_text())
    source = args.reference_dir / "inference/model.py"
    reference_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert reference_hash == lock["files"]["inference/model.py"]
    tree = ast.parse(source.read_text())
    embedding = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                     and n.name == "ParallelEmbedding")
    forward = next(n for n in embedding.body if isinstance(n, ast.FunctionDef)
                   and n.name == "forward")
    pre_mix = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == "make_identity_pre_mix")
    namespace = {"torch": torch, "F": torch.nn.functional, "world_size": 1}
    exec(compile(ast.Module(body=[forward, pre_mix], type_ignores=[]),
                 str(source), "exec"), namespace)
    index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())
    with (args.snapshot / index["weight_map"]["embed.weight"]).open("rb") as f:
        header_bytes = struct.unpack("<Q", f.read(8))[0]
        entry = json.loads(f.read(header_bytes))["embed.weight"]
        assert entry["dtype"] == "BF16" and entry["shape"] == [129280, 5120]
        begin, end = entry["data_offsets"]
        f.seek(8 + header_bytes + begin)
        payload = bytearray(f.read(end - begin))
    weight_hash = hashlib.sha256(payload).hexdigest()
    weight = torch.frombuffer(payload, dtype=torch.bfloat16).reshape(129280, 5120)
    cases = json.loads((args.vectors_dir / "cases.json").read_text())
    assert cases, "empty qualification case list"
    results = []
    for case in cases:
        assert case["kind"] in ("target", "draft")
        ids = torch.tensor(case["ids"], dtype=torch.int64)
        assert ids.numel() > 0 and ((ids >= 0) & (ids < 129280)).all()
        if case["kind"] == "target":
            assert len(case["positions"]) == ids.numel()
            ids = ids.unsqueeze(0)
        else:
            seeds = ids
            ids = torch.full((seeds.numel(), 5), 128799, dtype=torch.int64)
            ids[:, 0] = seeds
        expected = namespace["forward"](SimpleNamespace(weight=weight), ids)
        expected = expected.unsqueeze(2).repeat(1, 1, 4, 1)
        pre = namespace["make_identity_pre_mix"](expected, 4)
        hashes = {}
        for name, value in (("residual", expected), ("pre", pre)):
            actual = (args.vectors_dir / f'{case["prefix"]}-{name}.bin').read_bytes()
            assert actual == value.view(torch.uint8).numpy().tobytes(), (case["prefix"], name)
            hashes[name] = hashlib.sha256(actual).hexdigest()
        results.append({"prefix": case["prefix"], "kind": case["kind"],
                        "rows": ids.numel(), "bit_exact": True, "payloads_sha256": hashes})
        print(f'PASS {case["prefix"]} rows={ids.numel()} bit_exact=true', flush=True)
    args.output.write_text(json.dumps({
        "scope": "Real checkpoint embedding/initial pre-mix only; no block or model logits claim",
        "reference_sha256": reference_hash, "embedding_sha256": weight_hash,
        "qualifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
