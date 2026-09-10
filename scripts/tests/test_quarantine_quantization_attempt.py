from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


SCRIPT = (
    Path(__file__).parents[2] / "scripts" / "quarantine-quantization-attempt.py"
)
SPEC = importlib.util.spec_from_file_location("quarantine_quantization_attempt", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()


def bound(value: dict) -> dict:
    return {**value, "record_sha256": hashlib.sha256(canonical(value)).hexdigest()}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")


def fixture(tmp_path: Path) -> dict:
    run_state = tmp_path / ".flash.ds41rt-run"
    checkpoint_root = tmp_path / "slow-disk" / MODULE.CHECKPOINT_DIRNAME
    checkpoint_root.mkdir(parents=True)
    plan = {
        "schema": "test",
        "run_state_dir": os.fspath(run_state),
        "projection_checkpoint": {
            "root": os.fspath(checkpoint_root)
        },
    }
    plan_sha256 = hashlib.sha256(canonical(plan)).hexdigest()
    plan["plan_sha256"] = plan_sha256
    write_json(run_state / MODULE.PLAN_FILENAME, plan)

    modules = ["layer.0.a", "layer.0.b", "layer.0.c", "layer.0.a", "layer.0.d"]
    records = []
    request_ids = []
    for index, module in enumerate(modules):
        ledger = {
            "record_kind": "projection",
            "module": module,
            "processor_layer_index": 0 if index < 3 else 1,
            "sequence": index,
        }
        record = bound(ledger)
        records.append(record)
        request = {"module": module, "sequence": index}
        request_sha256 = hashlib.sha256(canonical(request)).hexdigest()
        request["request_sha256"] = request_sha256
        request_ids.append(request_sha256)
        tensor_payload = f"tensor-{index}".encode()
        tensor_name = f"{request_sha256}.safetensors"
        manifest = {
            "schema": MODULE.CHECKPOINT_SCHEMA,
            "schema_version": MODULE.CHECKPOINT_SCHEMA_VERSION,
            "request": request,
            "request_sha256": request_sha256,
            "tensor_file": tensor_name,
            "tensor_sha256": hashlib.sha256(tensor_payload).hexdigest(),
            "tensors": {"trellis": {"test": True}},
            "result": {"ledger_record": ledger},
        }
        manifest["manifest_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
        prefix = checkpoint_root / request_sha256[:2] / request_sha256[2:4]
        prefix.mkdir(parents=True, exist_ok=True)
        (prefix / tensor_name).write_bytes(tensor_payload)
        write_json(prefix / f"{request_sha256}.json", manifest)

    lines = [canonical(record) + b"\n" for record in records]
    prefix_payload = b"".join(lines[:3])
    full_payload = b"".join(lines)
    (run_state / MODULE.JOURNAL_FILENAME).write_bytes(full_payload)
    return {
        "run_state": run_state,
        "checkpoint_root": checkpoint_root,
        "plan_sha256": plan_sha256,
        "prefix_payload": prefix_payload,
        "full_payload": full_payload,
        "prefix_sha256": hashlib.sha256(prefix_payload).hexdigest(),
        "full_sha256": hashlib.sha256(full_payload).hexdigest(),
        "request_ids": request_ids,
    }


def recover_args(state: dict, *, apply: bool = False) -> dict:
    return {
        "run_state": state["run_state"],
        "expected_plan_sha256": state["plan_sha256"],
        "accepted_prefix_bytes": len(state["prefix_payload"]),
        "accepted_prefix_sha256": state["prefix_sha256"],
        "expected_full_sha256": state["full_sha256"],
        "expected_prefix_records": 3,
        "expected_suffix_records": 2,
        "expected_suffix_layer": 1,
        "apply": apply,
    }


def test_recovery_dry_run_is_read_only(tmp_path: Path) -> None:
    state = fixture(tmp_path)
    before = {
        path.relative_to(state["run_state"]): path.read_bytes()
        for path in state["run_state"].rglob("*")
        if path.is_file()
    }
    report = MODULE.recover(**recover_args(state))
    after = {
        path.relative_to(state["run_state"]): path.read_bytes()
        for path in state["run_state"].rglob("*")
        if path.is_file()
    }
    assert report["mode"] == "dry-run"
    assert report["retained_checkpoint_count"] == 3
    assert len(report["rejected_checkpoints"]) == 2
    assert after == before


def test_recovery_swaps_hardlinked_prefix_and_preserves_full_attempt(
    tmp_path: Path,
) -> None:
    state = fixture(tmp_path)
    report = MODULE.recover(**recover_args(state, apply=True))
    run_state = state["run_state"]
    quarantine = Path(report["quarantine"])

    assert (run_state / MODULE.JOURNAL_FILENAME).read_bytes() == state["prefix_payload"]
    assert (quarantine / "full-journal.jsonl").read_bytes() == state["full_payload"]
    assert len(list(state["checkpoint_root"].rglob("*.json"))) == 3
    assert len(list((quarantine / MODULE.CHECKPOINT_DIRNAME).rglob("*.json"))) == 5
    assert report["completion"]["status"] == "complete"

    retained_request = state["request_ids"][0]
    relative = (
        Path(retained_request[:2])
        / retained_request[2:4]
        / f"{retained_request}.safetensors"
    )
    active = state["checkpoint_root"] / relative
    frozen = quarantine / MODULE.CHECKPOINT_DIRNAME / relative
    assert os.stat(active).st_ino == os.stat(frozen).st_ino


def test_recovery_rejects_wrong_prefix_before_writing(tmp_path: Path) -> None:
    state = fixture(tmp_path)
    arguments = recover_args(state, apply=True)
    arguments["accepted_prefix_sha256"] = "0" * 64
    with pytest.raises(MODULE.RecoveryError, match="prefix hash"):
        MODULE.recover(**arguments)
    assert not MODULE.quarantine_parent_for(
        state["run_state"], state["checkpoint_root"]
    ).exists()


def test_recovery_authenticates_rejected_tensor_before_writing(tmp_path: Path) -> None:
    state = fixture(tmp_path)
    rejected_request = state["request_ids"][-1]
    tensor = (
        state["checkpoint_root"]
        / rejected_request[:2]
        / rejected_request[2:4]
        / f"{rejected_request}.safetensors"
    )
    tensor.write_bytes(b"corrupt")
    with pytest.raises(MODULE.RecoveryError, match="tensor failed"):
        MODULE.recover(**recover_args(state, apply=True))
    assert not MODULE.quarantine_parent_for(
        state["run_state"], state["checkpoint_root"]
    ).exists()
