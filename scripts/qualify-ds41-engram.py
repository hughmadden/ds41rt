#!/usr/bin/env python3
"""Compare the native token map and 16 request histories against pinned upstream code."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    args = parser.parse_args()
    reference = args.reference_dir
    lock = json.loads((ROOT / "docs/ds41-reference-lock.json").read_text())
    source = reference / "inference/engram.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == lock["files"]["inference/engram.py"]
    spec = importlib.util.spec_from_file_location("official_engram", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    tokenizer_path = reference / "tokenizer.json"
    assert hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() == lock["files"]["tokenizer.json"]
    backend = Tokenizer.from_file(str(tokenizer_path))

    class Wrapped:
        backend_tokenizer = backend
        def __len__(self):
            return backend.get_vocab_size(with_added_tokens=True)

    def native(example, *extra):
        return json.loads(subprocess.check_output([
            "cargo", "run", "--quiet", "--manifest-path", str(ROOT / "rust/Cargo.toml"),
            "-p", "ds41rt-loader", "--example", example, "--",
            str(reference / "tokenizer.json"), *map(str, extra),
        ], cwd=ROOT))

    expected_map, compressed_size = module.build_compressed_token_map(Wrapped())
    assert native("engram_token_map") == expected_map
    assert compressed_size == 99092
    config_path = reference / "inference/config.json"
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == lock["files"]["inference/config.json"]
    config = SimpleNamespace(**json.loads(config_path.read_text()), max_batch_size=1, max_seq_len=1024)
    layout = module.EngramLayout.from_args(config)
    states = [module.NgramHashState(config, layout, Wrapped()) for _ in range(16)]
    positions = [0] * 16
    events, expected = [], []
    # Include zero acceptance, partial verification, images, and repeated wraparound of the 3-token tail.
    for round_id in range(8):
        for request in range(16):
            length = 1 + (round_id + request) % 9
            tokens = [(129279 - request * 7919 - round_id * 193 - i * 71) % 129280 for i in range(length)]
            images = [(i + request + round_id) % 5 == 0 for i in range(length)]
            accepted = (request + round_id) % (length + 1)
            hashes = states[request](torch.tensor([tokens]), positions[request], torch.tensor([[not image for image in images]]))
            positions[request] += accepted
            events.append(dict(request=request, tokens=tokens, images=images, accept=accepted))
            expected.append(dict(hashes=hashes[0].tolist(), position=positions[request]))
    with tempfile.TemporaryDirectory(prefix="ds41-engram-qualification-") as directory:
        path = Path(directory) / "events.json"
        path.write_text(json.dumps(events))
        actual = native("engram_addresses", path)
    assert actual == expected, "native hash transactions differ from official reference"
    print(json.dumps(dict(token_ids=len(expected_map), compressed_ids=compressed_size,
                          requests=16, batches=len(events), hash_match=True,
                          reference_revision=lock["revision"])))


if __name__ == "__main__":
    main()
