#!/usr/bin/env python3
"""Validate exact model-snapshot identity across all five DS41RT daemons."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from typing import Any, Sequence


RUNTIME_SCHEMA = "ds41rt-runtime-model-snapshot-v1"
REPORT_SCHEMA = "ds41rt-runtime-model-snapshot-qualification-v1"
MARKER = "runtime_model_snapshot "
REVISION_RE = re.compile(r"[0-9a-f]{64}\Z")
MODEL_ID_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z"
)
LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
REQUIRED_LABELS = ("coordinator", "ostrich", "dodo", "emu", "kiwi")


def parse_log_arguments(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if (
            not separator
            or LABEL_RE.fullmatch(label) is None
            or not raw_path
            or label in result
        ):
            raise ValueError(
                "--log must use each safe LABEL=PATH exactly once"
            )
        result[label] = Path(raw_path).expanduser()
    observed = set(result)
    expected = set(REQUIRED_LABELS)
    if observed != expected:
        raise ValueError(
            "--log labels must be exactly "
            + ",".join(REQUIRED_LABELS)
            + f"; got {','.join(sorted(observed))}"
        )
    return result


def hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def runtime_records(payload: bytes, label: str) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} log is not UTF-8: {error}") from error
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if MARKER not in line:
            continue
        raw = line.split(MARKER, 1)[1].strip()
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{label} has malformed runtime snapshot JSON on line {line_number}"
            ) from error
        if not isinstance(record, dict) or record.get("schema") != RUNTIME_SCHEMA:
            raise ValueError(
                f"{label} has an invalid runtime snapshot record on line {line_number}"
            )
        records.append({"line": line_number, "record": record})
    if not records:
        raise ValueError(f"{label} has no {RUNTIME_SCHEMA} record")
    return records


def validate_record(
    record: dict[str, Any],
    *,
    label: str,
    model_id: str,
    revision: str,
) -> None:
    if record.get("model_id") != model_id:
        raise ValueError(f"{label} selected a different model ID")
    if record.get("selection") != "explicit":
        raise ValueError(f"{label} did not use explicit snapshot selection")
    if record.get("requested_revision") != revision:
        raise ValueError(f"{label} requested a different model revision")
    if record.get("selected_revision") != revision:
        raise ValueError(f"{label} selected a different model revision")
    raw_path = record.get("snapshot_path")
    if not isinstance(raw_path, str) or "\\" in raw_path:
        raise ValueError(f"{label} reported an invalid snapshot path")
    path = PurePosixPath(raw_path)
    if not path.is_absolute() or path.name != revision:
        raise ValueError(f"{label} snapshot path does not end in the revision")


def qualification_report(
    logs: dict[str, Path], model_id: str, revision: str
) -> dict[str, Any]:
    if MODEL_ID_RE.fullmatch(model_id) is None:
        raise ValueError("--model-id must be a safe two-component Hugging Face ID")
    if REVISION_RE.fullmatch(revision) is None:
        raise ValueError("--revision must be exactly 64 lowercase hex characters")

    daemons: list[dict[str, Any]] = []
    for label in REQUIRED_LABELS:
        configured_path = logs[label]
        if configured_path.is_symlink():
            raise ValueError(f"{label} log must not be a symlink: {configured_path}")
        path = configured_path.resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"{label} log is not a regular file: {path}")
        payload = path.read_bytes()
        records = runtime_records(payload, label)
        selected = records[-1]
        record = selected["record"]
        validate_record(
            record,
            label=label,
            model_id=model_id,
            revision=revision,
        )
        daemons.append(
            {
                "label": label,
                "log_path": str(path),
                "log_bytes": len(payload),
                "log_sha256": hash_bytes(payload),
                "runtime_record_count": len(records),
                "selected_record_line": selected["line"],
                "runtime_snapshot": record,
            }
        )

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "model_id": model_id,
        "revision": revision,
        "required_daemons": list(REQUIRED_LABELS),
        "daemons": daemons,
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hash_bytes(canonical)
    return report


def write_json_atomic(path: Path, report: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--log",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="captured daemon log; required labels are coordinator and four Spark hosts",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = qualification_report(
            parse_log_arguments(args.log),
            args.model_id,
            args.revision,
        )
        if args.output is not None:
            write_json_atomic(args.output, report)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
