#!/usr/bin/env python3
"""Recover an accepted EXL3 prefix while retaining a rejected attempt intact.

The converter's projection checkpoints are immutable.  If an execution-only
recovery bug appends a numerically different attempt, this tool does not delete
or rewrite that evidence.  It validates the complete journal/checkpoint join,
builds an accepted checkpoint tree from hardlinks, and atomically swaps the old
tree and journal into a content-addressed quarantine.

All writers using the run state must be stopped before ``--apply``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable


PLAN_FILENAME = "ds4rt-gptqmodel-plan.json"
JOURNAL_FILENAME = ".ds4rt-exl3-error-journal.jsonl"
CHECKPOINT_DIRNAME = "projection-checkpoints"
QUARANTINE_DIRNAME = "rejected-attempts"
CHECKPOINT_SCHEMA = "ds4rt.exl3-projection-checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class RecoveryError(RuntimeError):
    """The rejected attempt cannot be separated without losing evidence."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RecoveryError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise RecoveryError(f"{label} is not a JSON object: {path}")
    return value


def _validate_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise RecoveryError(f"{label} is not a SHA-256 digest")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _validate_plan(run_state: Path, expected_plan_sha256: str) -> dict[str, Any]:
    plan = _read_json(run_state / PLAN_FILENAME, "quantization plan")
    digest = _validate_digest(plan.get("plan_sha256"), "plan digest")
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if hashlib.sha256(canonical_json(body)).hexdigest() != digest:
        raise RecoveryError("quantization plan failed its canonical digest")
    if digest != expected_plan_sha256:
        raise RecoveryError("quantization plan differs from the accepted plan")
    checkpoint = plan.get("projection_checkpoint")
    if not isinstance(checkpoint, dict):
        raise RecoveryError("quantization plan has no projection-checkpoint contract")
    planned_root = checkpoint.get("root")
    if (
        not isinstance(planned_root, str)
        or not Path(planned_root).is_absolute()
        or Path(planned_root).name != CHECKPOINT_DIRNAME
    ):
        raise RecoveryError("quantization plan has an unexpected checkpoint root")
    planned_state = plan.get("run_state_dir")
    if (
        not isinstance(planned_state, str)
        or Path(planned_state).expanduser().resolve() != run_state
    ):
        raise RecoveryError("quantization plan identifies a different run state")
    return plan


@dataclass(frozen=True)
class JournalScan:
    prefix_records: dict[str, str]
    suffix_records: dict[str, str]
    prefix_bytes: int
    prefix_sha256: str
    suffix_bytes: int
    suffix_sha256: str
    full_bytes: int
    full_sha256: str


def _bound_record_digest(record: dict[str, Any]) -> str:
    digest = _validate_digest(record.get("record_sha256"), "journal record digest")
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    if hashlib.sha256(canonical_json(body)).hexdigest() != digest:
        raise RecoveryError("projection journal record failed its canonical digest")
    return digest


def scan_journal(
    path: Path,
    *,
    accepted_prefix_bytes: int,
    accepted_prefix_sha256: str,
    expected_full_sha256: str,
    expected_prefix_records: int,
    expected_suffix_records: int,
    expected_suffix_layer: int,
) -> JournalScan:
    if not path.is_file() or path.is_symlink():
        raise RecoveryError("projection journal is missing or nonregular")
    prefix_hash = hashlib.sha256()
    suffix_hash = hashlib.sha256()
    full_hash = hashlib.sha256()
    prefix_records: dict[str, str] = {}
    suffix_records: dict[str, str] = {}
    prefix_modules: set[str] = set()
    suffix_modules: set[str] = set()
    offset = 0

    with path.open("rb") as source:
        for line_number, line in enumerate(source, 1):
            start = offset
            offset += len(line)
            if not line.endswith(b"\n"):
                raise RecoveryError("projection journal ends with a partial record")
            if start < accepted_prefix_bytes < offset:
                raise RecoveryError("accepted prefix does not end at a record boundary")
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RecoveryError(
                    f"projection journal record {line_number} is invalid"
                ) from error
            if not isinstance(record, dict) or record.get("record_kind") != "projection":
                raise RecoveryError(
                    f"projection journal record {line_number} is not a projection"
                )
            digest = _bound_record_digest(record)
            module = record.get("module")
            if not isinstance(module, str) or not module:
                raise RecoveryError("projection journal record has no module")
            full_hash.update(line)
            target = prefix_records if offset <= accepted_prefix_bytes else suffix_records
            modules = prefix_modules if offset <= accepted_prefix_bytes else suffix_modules
            segment_hash = prefix_hash if offset <= accepted_prefix_bytes else suffix_hash
            segment_hash.update(line)
            if digest in prefix_records or digest in suffix_records:
                raise RecoveryError("projection journal contains a duplicate record digest")
            if module in modules:
                raise RecoveryError(
                    f"projection journal segment contains duplicate module `{module}`"
                )
            target[digest] = module
            modules.add(module)
            if offset > accepted_prefix_bytes and record.get(
                "processor_layer_index"
            ) != expected_suffix_layer:
                raise RecoveryError("rejected journal suffix crosses a layer boundary")

    if offset < accepted_prefix_bytes:
        raise RecoveryError("accepted prefix extends beyond the projection journal")
    if len(prefix_records) != expected_prefix_records:
        raise RecoveryError("accepted projection record count differs")
    if len(suffix_records) != expected_suffix_records:
        raise RecoveryError("rejected projection record count differs")
    if prefix_hash.hexdigest() != accepted_prefix_sha256:
        raise RecoveryError("accepted journal prefix hash differs")
    if full_hash.hexdigest() != expected_full_sha256:
        raise RecoveryError("complete projection journal hash differs")
    return JournalScan(
        prefix_records=prefix_records,
        suffix_records=suffix_records,
        prefix_bytes=accepted_prefix_bytes,
        prefix_sha256=accepted_prefix_sha256,
        suffix_bytes=offset - accepted_prefix_bytes,
        suffix_sha256=suffix_hash.hexdigest(),
        full_bytes=offset,
        full_sha256=expected_full_sha256,
    )


@dataclass(frozen=True)
class CheckpointEntry:
    request_sha256: str
    record_sha256: str
    module: str
    manifest_path: Path
    tensor_path: Path
    tensor_sha256: str


def _checkpoint_files(root: Path) -> Iterable[tuple[str, Path, Path]]:
    if not root.is_dir() or root.is_symlink():
        raise RecoveryError("projection-checkpoint root is missing or unsafe")
    manifests: dict[str, Path] = {}
    tensors: dict[str, Path] = {}
    for first in root.iterdir():
        if (
            not first.is_dir()
            or first.is_symlink()
            or re.fullmatch(r"[0-9a-f]{2}", first.name) is None
        ):
            raise RecoveryError("projection-checkpoint root contains an unsafe entry")
        for second in first.iterdir():
            if (
                not second.is_dir()
                or second.is_symlink()
                or re.fullmatch(r"[0-9a-f]{2}", second.name) is None
            ):
                raise RecoveryError("projection-checkpoint prefix is unsafe")
            for path in second.iterdir():
                if not path.is_file() or path.is_symlink():
                    raise RecoveryError("projection checkpoint contains a nonregular file")
                if path.suffix not in {".json", ".safetensors"}:
                    raise RecoveryError("projection checkpoint contains an unknown file")
                request_sha256 = path.name.removesuffix(path.suffix)
                _validate_digest(request_sha256, "checkpoint filename")
                if (
                    request_sha256[:2] != first.name
                    or request_sha256[2:4] != second.name
                ):
                    raise RecoveryError("projection checkpoint is under the wrong prefix")
                destination = manifests if path.suffix == ".json" else tensors
                if request_sha256 in destination:
                    raise RecoveryError("projection checkpoint contains a duplicate file")
                destination[request_sha256] = path
    if set(manifests) != set(tensors):
        raise RecoveryError("projection checkpoint contains an incomplete pair")
    for request_sha256 in sorted(manifests):
        yield request_sha256, manifests[request_sha256], tensors[request_sha256]


def scan_checkpoints(
    root: Path,
    journal: JournalScan,
) -> tuple[list[CheckpointEntry], list[CheckpointEntry]]:
    retained: list[CheckpointEntry] = []
    rejected: list[CheckpointEntry] = []
    seen_record_hashes: set[str] = set()
    retained_modules: dict[str, str] = {}

    for request_sha256, manifest_path, tensor_path in _checkpoint_files(root):
        manifest = _read_json(manifest_path, "projection checkpoint manifest")
        manifest_digest = _validate_digest(
            manifest.get("manifest_sha256"), "checkpoint manifest digest"
        )
        manifest_body = {
            key: value for key, value in manifest.items() if key != "manifest_sha256"
        }
        request = manifest.get("request")
        request_body = (
            {key: value for key, value in request.items() if key != "request_sha256"}
            if isinstance(request, dict)
            else None
        )
        tensor_sha256 = _validate_digest(
            manifest.get("tensor_sha256"), "checkpoint tensor digest"
        )
        result = manifest.get("result")
        ledger_record = result.get("ledger_record") if isinstance(result, dict) else None
        if (
            manifest.get("schema") != CHECKPOINT_SCHEMA
            or manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or manifest.get("request_sha256") != request_sha256
            or not isinstance(request, dict)
            or request.get("request_sha256") != request_sha256
            or hashlib.sha256(canonical_json(request_body)).hexdigest()
            != request_sha256
            or manifest.get("tensor_file") != tensor_path.name
            or not isinstance(ledger_record, dict)
            or hashlib.sha256(canonical_json(manifest_body)).hexdigest()
            != manifest_digest
        ):
            raise RecoveryError("projection checkpoint failed manifest validation")
        record_sha256 = hashlib.sha256(canonical_json(ledger_record)).hexdigest()
        module = request.get("module")
        if (
            not isinstance(module, str)
            or not module
            or ledger_record.get("module") != module
            or record_sha256 in seen_record_hashes
        ):
            raise RecoveryError("projection checkpoint has an ambiguous journal identity")
        seen_record_hashes.add(record_sha256)
        entry = CheckpointEntry(
            request_sha256=request_sha256,
            record_sha256=record_sha256,
            module=module,
            manifest_path=manifest_path,
            tensor_path=tensor_path,
            tensor_sha256=tensor_sha256,
        )
        if record_sha256 in journal.prefix_records:
            if journal.prefix_records[record_sha256] != module:
                raise RecoveryError("accepted checkpoint module differs from its journal")
            previous = retained_modules.setdefault(module, request_sha256)
            if previous != request_sha256:
                raise RecoveryError(
                    f"accepted checkpoint prefix contains module drift for `{module}`"
                )
            retained.append(entry)
        elif record_sha256 in journal.suffix_records:
            if journal.suffix_records[record_sha256] != module:
                raise RecoveryError("rejected checkpoint module differs from its journal")
            if sha256_file(tensor_path) != tensor_sha256:
                raise RecoveryError("rejected checkpoint tensor failed its digest")
            rejected.append(entry)
        else:
            raise RecoveryError("projection checkpoint is absent from the journal")

    if seen_record_hashes != set(journal.prefix_records) | set(journal.suffix_records):
        raise RecoveryError("projection journal and checkpoint store do not close exactly")
    if len(retained) != len(journal.prefix_records):
        raise RecoveryError("accepted checkpoint count differs from its journal")
    if len(rejected) != len(journal.suffix_records):
        raise RecoveryError("rejected checkpoint count differs from its journal")
    return retained, rejected


def _copy_segment(
    source_path: Path,
    target_path: Path,
    *,
    offset: int,
    length: int,
) -> None:
    descriptor = os.open(target_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with source_path.open("rb") as source, os.fdopen(descriptor, "wb") as target:
            source.seek(offset)
            remaining = length
            while remaining:
                block = source.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise RecoveryError("projection journal changed while copying")
                target.write(block)
                remaining -= len(block)
            target.flush()
            os.fsync(target.fileno())
    except BaseException:
        try:
            target_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _link_checkpoint_tree(
    source_root: Path,
    target_root: Path,
    entries: Iterable[CheckpointEntry],
) -> None:
    target_root.mkdir(mode=0o700)
    created_directories: set[Path] = {target_root}
    for entry in entries:
        for source in (entry.manifest_path, entry.tensor_path):
            relative = source.relative_to(source_root)
            target = target_root / relative
            if target.parent not in created_directories:
                target.parent.mkdir(parents=True, exist_ok=True)
                created_directories.add(target.parent)
            os.link(source, target, follow_symlinks=False)
    for directory in sorted(created_directories, key=lambda value: len(value.parts), reverse=True):
        _fsync_directory(directory)


def _entry_manifest(entry: CheckpointEntry) -> dict[str, str]:
    return {
        "module": entry.module,
        "record_sha256": entry.record_sha256,
        "request_sha256": entry.request_sha256,
        "tensor_sha256": entry.tensor_sha256,
    }


def quarantine_parent_for(
    run_state: Path,
    checkpoint_root: Path | None = None,
) -> Path:
    """Keep forensic evidence outside the launcher's closed run namespace."""

    if checkpoint_root is None:
        checkpoint_root = run_state / CHECKPOINT_DIRNAME
    return checkpoint_root.parent / f"{checkpoint_root.name}.{QUARANTINE_DIRNAME}"


def recover(
    *,
    run_state: Path,
    expected_plan_sha256: str,
    accepted_prefix_bytes: int,
    accepted_prefix_sha256: str,
    expected_full_sha256: str,
    expected_prefix_records: int,
    expected_suffix_records: int,
    expected_suffix_layer: int,
    apply: bool,
) -> dict[str, Any]:
    run_state = run_state.expanduser().resolve()
    if not run_state.is_dir() or run_state.is_symlink():
        raise RecoveryError("run state is missing or unsafe")
    expected_plan_sha256 = _validate_digest(
        expected_plan_sha256, "expected plan digest"
    )
    accepted_prefix_sha256 = _validate_digest(
        accepted_prefix_sha256, "accepted prefix digest"
    )
    expected_full_sha256 = _validate_digest(
        expected_full_sha256, "expected full journal digest"
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (
            accepted_prefix_bytes,
            expected_prefix_records,
            expected_suffix_records,
            expected_suffix_layer,
        )
    ):
        raise RecoveryError("recovery counts and offsets must be nonnegative integers")

    plan = _validate_plan(run_state, expected_plan_sha256)
    journal_path = run_state / JOURNAL_FILENAME
    checkpoint_root = (
        Path(plan["projection_checkpoint"]["root"]).expanduser().resolve()
    )
    journal = scan_journal(
        journal_path,
        accepted_prefix_bytes=accepted_prefix_bytes,
        accepted_prefix_sha256=accepted_prefix_sha256,
        expected_full_sha256=expected_full_sha256,
        expected_prefix_records=expected_prefix_records,
        expected_suffix_records=expected_suffix_records,
        expected_suffix_layer=expected_suffix_layer,
    )
    retained, rejected = scan_checkpoints(checkpoint_root, journal)
    attempt_contract = {
        "schema": "ds4rt.quantization-rejected-attempt",
        "schema_version": 1,
        "plan_sha256": expected_plan_sha256,
        "accepted_prefix": {
            "bytes": journal.prefix_bytes,
            "records": len(journal.prefix_records),
            "sha256": journal.prefix_sha256,
        },
        "rejected_suffix": {
            "bytes": journal.suffix_bytes,
            "records": len(journal.suffix_records),
            "sha256": journal.suffix_sha256,
            "processor_layer_index": expected_suffix_layer,
        },
        "full_journal": {
            "bytes": journal.full_bytes,
            "records": len(journal.prefix_records) + len(journal.suffix_records),
            "sha256": journal.full_sha256,
        },
        "retained_checkpoint_count": len(retained),
        "rejected_checkpoints": [_entry_manifest(entry) for entry in rejected],
    }
    attempt_sha256 = hashlib.sha256(canonical_json(attempt_contract)).hexdigest()
    attempt_contract["attempt_sha256"] = attempt_sha256
    summary = {
        **attempt_contract,
        "mode": "apply" if apply else "dry-run",
        "quarantine": os.fspath(
            quarantine_parent_for(run_state, checkpoint_root)
            / f"attempt-{attempt_sha256[:16]}"
        ),
    }
    if not apply:
        return summary

    quarantine_parent = quarantine_parent_for(run_state, checkpoint_root)
    quarantine_parent.mkdir(mode=0o700, exist_ok=True)
    quarantine = quarantine_parent / f"attempt-{attempt_sha256[:16]}"
    checkpoint_stage = checkpoint_root.parent / (
        f".{CHECKPOINT_DIRNAME}.accepted-{attempt_sha256[:16]}"
    )
    journal_stage = run_state / f".{JOURNAL_FILENAME}.accepted-{attempt_sha256[:16]}"
    journal_backup = run_state / f".{JOURNAL_FILENAME}.full-{attempt_sha256[:16]}"
    quarantine_stage = quarantine_parent / f".attempt-{attempt_sha256[:16]}.staging"
    for path in (
        quarantine,
        checkpoint_stage,
        journal_stage,
        journal_backup,
        quarantine_stage,
    ):
        if path.exists() or path.is_symlink():
            raise RecoveryError(f"recovery target already exists: {path}")

    quarantine_stage.mkdir(mode=0o700)
    try:
        _link_checkpoint_tree(checkpoint_root, checkpoint_stage, retained)
        _copy_segment(
            journal_path,
            journal_stage,
            offset=0,
            length=journal.prefix_bytes,
        )
        _copy_segment(
            journal_path,
            quarantine_stage / "rejected-journal-suffix.jsonl",
            offset=journal.prefix_bytes,
            length=journal.suffix_bytes,
        )
        _copy_segment(
            journal_path,
            quarantine_stage / "full-journal.jsonl",
            offset=0,
            length=journal.full_bytes,
        )
        if (
            sha256_file(journal_stage) != journal.prefix_sha256
            or sha256_file(quarantine_stage / "rejected-journal-suffix.jsonl")
            != journal.suffix_sha256
            or sha256_file(quarantine_stage / "full-journal.jsonl")
            != journal.full_sha256
        ):
            raise RecoveryError("staged journal evidence failed its digest")
        _atomic_write(
            quarantine_stage / "recovery-manifest.json",
            canonical_json(attempt_contract) + b"\n",
        )

        old_checkpoints = quarantine_stage / CHECKPOINT_DIRNAME
        checkpoints_swapped = False
        journal_backed_up = False
        journal_swapped = False
        try:
            os.replace(checkpoint_root, old_checkpoints)
            os.replace(checkpoint_stage, checkpoint_root)
            checkpoints_swapped = True
            os.replace(journal_path, journal_backup)
            journal_backed_up = True
            os.replace(journal_stage, journal_path)
            journal_swapped = True
            _fsync_directory(run_state)
        except BaseException as error:
            rollback_errors: list[str] = []
            if journal_swapped:
                try:
                    os.replace(journal_path, journal_stage)
                except BaseException as rollback_error:
                    rollback_errors.append(f"journal: {rollback_error}")
            if journal_backed_up:
                try:
                    os.replace(journal_backup, journal_path)
                except BaseException as rollback_error:
                    rollback_errors.append(f"journal backup: {rollback_error}")
            if checkpoints_swapped:
                try:
                    os.replace(checkpoint_root, checkpoint_stage)
                    os.replace(old_checkpoints, checkpoint_root)
                except BaseException as rollback_error:
                    rollback_errors.append(f"checkpoints: {rollback_error}")
            if rollback_errors:
                raise RecoveryError(
                    "recovery swap and rollback failed; preserve all paths: "
                    + "; ".join(rollback_errors)
                ) from error
            raise

        completion = {
            "attempt_sha256": attempt_sha256,
            "active_journal_sha256": sha256_file(journal_path),
            "active_checkpoint_count": sum(
                1 for _path in checkpoint_root.rglob("*.json")
            ),
            "quarantined_full_journal_sha256": sha256_file(
                quarantine_stage / "full-journal.jsonl"
            ),
            "status": "complete",
        }
        if (
            completion["active_journal_sha256"] != journal.prefix_sha256
            or completion["active_checkpoint_count"] != len(retained)
            or completion["quarantined_full_journal_sha256"] != journal.full_sha256
        ):
            raise RecoveryError("post-swap recovery evidence differs")
        if sha256_file(journal_backup) != journal.full_sha256:
            raise RecoveryError("journal rollback copy differs from quarantine evidence")
        _atomic_write(
            quarantine_stage / "recovery-complete.json",
            canonical_json(completion) + b"\n",
        )
        journal_backup.unlink()
        _fsync_directory(run_state)
        os.replace(quarantine_stage, quarantine)
        _fsync_directory(quarantine_parent)
        return {**summary, "completion": completion}
    except BaseException:
        # Only pre-swap staging paths are removed automatically.  If a swap
        # could not be rolled back, the exception above identifies every path
        # and this cleanup deliberately leaves the evidence in place.
        for path in (journal_stage, checkpoint_stage):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
        if (
            quarantine_stage.is_dir()
            and not (quarantine_stage / CHECKPOINT_DIRNAME).exists()
            and not (quarantine_stage / "full-journal.jsonl").exists()
        ):
            shutil.rmtree(quarantine_stage)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-state", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--accepted-prefix-bytes", type=int, required=True)
    parser.add_argument("--accepted-prefix-sha256", required=True)
    parser.add_argument("--expected-full-sha256", required=True)
    parser.add_argument("--expected-prefix-records", type=int, required=True)
    parser.add_argument("--expected-suffix-records", type=int, required=True)
    parser.add_argument("--expected-suffix-layer", type=int, required=True)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = recover(
            run_state=args.run_state,
            expected_plan_sha256=args.expected_plan_sha256,
            accepted_prefix_bytes=args.accepted_prefix_bytes,
            accepted_prefix_sha256=args.accepted_prefix_sha256,
            expected_full_sha256=args.expected_full_sha256,
            expected_prefix_records=args.expected_prefix_records,
            expected_suffix_records=args.expected_suffix_records,
            expected_suffix_layer=args.expected_suffix_layer,
            apply=args.apply,
        )
    except (OSError, RecoveryError, ValueError) as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
