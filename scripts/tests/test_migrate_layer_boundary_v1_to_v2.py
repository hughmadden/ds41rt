from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest
import torch
from safetensors.torch import save_file as save_safetensors_file


SCRIPT = (
    Path(__file__).parents[2] / "scripts" / "migrate-layer-boundary-v1-to-v2.py"
)
SPEC = importlib.util.spec_from_file_location("migrate_layer_boundary_v1_to_v2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _legacy_boundary(tmp_path: Path, *, batches: int = 2) -> tuple[Path, Path]:
    root = tmp_path / ".run" / "layer-boundary"
    staging = root / "staging"
    activation_root = staging / "activations"
    activation_root.mkdir(parents=True)
    shards = []
    activation_bytes = 0
    for index in range(batches):
        hidden = (
            torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4) + index
        ).to(torch.bfloat16)
        relative = Path("activations") / f"batch-{index:06d}.safetensors"
        path = staging / relative
        save_safetensors_file({"hidden": hidden}, path)
        payload = path.read_bytes()
        activation_bytes += hidden.numel() * hidden.element_size()
        shards.append(
            {
                "file": relative.as_posix(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "tensor": {
                    "shape": list(hidden.shape),
                    "dtype": str(hidden.dtype),
                    "bytes": hidden.numel() * hidden.element_size(),
                },
            }
        )
    body = {
        "schema": MODULE.SCHEMA,
        "schema_version": MODULE.LEGACY_VERSION,
        "plan_sha256": "a" * 64,
        "layer_index": 25,
        "layer_name": "model.layers.25",
        "hidden_size": 4,
        "hc_mult": 2,
        "activation_batches": batches,
        "activation_bytes": activation_bytes,
        "activation_shards": shards,
        "replay_metadata": {"sha256": "b" * 64},
        "projection_entries": [],
        "completed_projection_entries": [],
    }
    digest = hashlib.sha256(MODULE.canonical_json(body)).hexdigest()
    manifest = {**body, "manifest_sha256": digest}
    (staging / MODULE.MANIFEST_FILENAME).write_bytes(
        MODULE.canonical_json(manifest) + b"\n"
    )
    committed = root / f"layer-000025-{digest[:16]}"
    os.replace(staging, committed)
    return root, committed


def test_migration_hardlinks_v2_and_archives_v1(tmp_path: Path) -> None:
    root, legacy = _legacy_boundary(tmp_path)
    source_inode = (legacy / "activations/batch-000000.safetensors").stat().st_ino

    dry_run = MODULE.migrate(root)
    assert dry_run["status"] == "ready"
    assert legacy.exists()
    assert len(list(root.iterdir())) == 1

    report = MODULE.migrate(root, apply=True)
    active = Path(report["active"])
    archive = Path(report["legacy_archive"])
    assert report["status"] == "migrated"
    assert active.is_dir()
    assert archive.is_dir()
    assert archive.parent.parent == root.parent.parent
    assert not legacy.exists()
    manifest = json.loads((active / MODULE.MANIFEST_FILENAME).read_text())
    assert manifest["schema_version"] == 2
    assert manifest["payload_hash_algorithm"] == "xxh3-128"
    assert "sha256" not in manifest["activation_shards"][0]
    assert len(manifest["activation_shards"][0]["xxh3_128"]) == 32
    assert (active / "activations/batch-000000.safetensors").stat().st_ino == source_inode
    assert (archive / "activations/batch-000000.safetensors").stat().st_ino == source_inode

    repeated = MODULE.migrate(root, apply=True)
    assert repeated["status"] == "already-v2"


def test_migration_rejects_legacy_payload_tampering(tmp_path: Path) -> None:
    root, legacy = _legacy_boundary(tmp_path)
    payload = legacy / "activations/batch-000000.safetensors"
    damaged = bytearray(payload.read_bytes())
    damaged[-1] ^= 1
    payload.write_bytes(damaged)
    with pytest.raises(MODULE.MigrationError, match="failed SHA-256"):
        MODULE.migrate(root)


def test_migration_rejects_unexpected_boundary_entries(tmp_path: Path) -> None:
    root, _legacy = _legacy_boundary(tmp_path)
    (root / "unexpected").write_text("unsafe", encoding="utf-8")
    with pytest.raises(MODULE.MigrationError, match="unexpected"):
        MODULE.migrate(root)
