"""Pinned external helpers and native library (path + sha256, all overridable).

No hidden absolute imports: every pinned artifact is verified against its
sha256 before use, and every path can be overridden from the CLI.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

# Frozen helper: CPU-only capture reader (worker flash-attention-input-extractor-fix-01).
EXTRACTOR_PATH = Path(
    "/home/turq/.cache/afd-dsh-workers-20260919/flash-attention-input-extractor-fix-01/"
    "extract_attention_inputs.py"
)
EXTRACTOR_SHA256 = (
    "63b4146b95740ee79f8606226be7ccbf27e69b351fb461996689a72cc2e2f10d"
)

# Reviewed integer address/mask oracle (worker flash-attention-address-oracle-01).
ORACLE_PATH = Path(
    "/home/turq/.cache/afd-dsh-workers-20260919/flash-attention-address-oracle-01/"
    "attention_address_oracle.py"
)
ORACLE_SHA256 = (
    "40d769cd8682a3e0757939791b30665884822b28c719cb39ea48dbe4579cd7dd"
)

# Existing corrected native library (worker kimi-fullnative-build-02 artifacts).
NATIVE_LIBRARY_PATH = Path(
    "/home/turq/.cache/afd-dsh-workers-20260919/kimi-fullnative-build-02/"
    "artifacts/native/libds41rt_native.so"
)
NATIVE_LIBRARY_SHA256 = (
    "e0921fed65a37ee6679a3cfe84a96ee91365ae6cf7b6d529f4476cba3b333533"
)

# Interface map (typo sentinel note: partial[..., 513] = -1, not 512).
INTERFACE_MAP_PATH = Path(
    "/home/turq/.cache/afd-dsh-workers-20260919/flash-sparse-replay-interface-map-01/MAP.md"
)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_pinned_module(name: str, path: Path, expected_sha256: str):
    """Import ``path`` as ``name`` after verifying its sha256.

    Bytecode generation is disabled so read-only helper directories are never
    written to.  Raises ValueError on a missing file or sha mismatch.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"pinned module not found: {path}")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(
            f"pinned module sha256 mismatch for {path}: "
            f"expected {expected_sha256}, got {digest}"
        )
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_extractor(
    path: Path = EXTRACTOR_PATH,
    sha256: str = EXTRACTOR_SHA256,
    oracle_path: Path = ORACLE_PATH,
    oracle_sha256: str = ORACLE_SHA256,
):
    """Load the pinned extractor, with the oracle redirected to our pin.

    The extractor carries its own absolute oracle pin; overriding the module
    attributes before first use routes it to the CLI-selected (sha-verified)
    oracle instead, so no import path is hidden from the caller.
    """
    module = load_pinned_module("sparse_replay_extractor", path, sha256)
    module.ORACLE_PATH = Path(oracle_path)
    module.ORACLE_SHA256 = oracle_sha256
    return module


def load_oracle(path: Path = ORACLE_PATH, sha256: str = ORACLE_SHA256):
    return load_pinned_module("sparse_replay_oracle", path, sha256)
