#!/usr/bin/env python3
"""Verify and publish one immutable DS4RT HF snapshot on the Spark ranks."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any, Sequence


MANIFEST_SCHEMA = "ds4rt-hf-staged-snapshot-v1"
MODEL_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
HOST_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
REVISION_RE = re.compile(r"[0-9a-f]{64}\Z")
REMOTE_PATH_RE = re.compile(r"/[A-Za-z0-9_./-]+\Z")
DEFAULT_HOSTS = ("ostrich", "dodo", "emu", "kiwi")
QUALIFICATION_SCHEMAS = {
    "retained-native.json": "ds4rt-exl3-retained-native-integrity-v1",
    "expert-quality.json": "ds4rt-exl3-checkpoint-quality-v1",
}
DEVELOPMENT_QUALIFICATION_STATUS = "development-unqualified"
DEVELOPMENT_BLOCKERS = (
    "complete token-aware all-layer quality evidence is absent",
    "natural MTP held-out activation quality evidence is absent",
)
DEVELOPMENT_EVIDENCE_SCHEMAS = {
    "retained-native.json": "ds4rt-exl3-retained-native-integrity-v1",
    "diagnostic-quality.json": "ds4rt-exl3-checkpoint-quality-v1",
}


REMOTE_HF_HOME_SOURCE = (
    "import os,pathlib; "
    "print(pathlib.Path(os.environ.get('HF_HOME', "
    "pathlib.Path.home()/'.cache'/'huggingface')).expanduser().resolve())"
)


REMOTE_VERIFY_SOURCE = r'''
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys

SCHEMA = "ds4rt-hf-staged-snapshot-v1"
root = Path(sys.argv[1]).resolve()
revision = sys.argv[2]
model_id = sys.argv[3]
allow_development_unqualified = len(sys.argv) == 5 and sys.argv[4] == "1"
if re.fullmatch(r"[0-9a-f]{64}", revision) is None:
    raise SystemExit("invalid expected staged revision")
manifest_path = root / "ds4rt-manifests" / f"{revision}.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if (
    manifest.get("schema") != SCHEMA
    or manifest.get("model_id") != model_id
    or manifest.get("revision") != revision
    or not isinstance(manifest.get("files"), list)
):
    raise SystemExit("remote staged manifest contract mismatch")
entries = manifest["files"]
canonical = json.dumps(
    {"schema": SCHEMA, "files": entries},
    sort_keys=True,
    separators=(",", ":"),
).encode()
if hashlib.sha256(canonical).hexdigest() != revision:
    raise SystemExit("remote staged manifest revision mismatch")
snapshot = root / "snapshots" / revision
if not snapshot.is_dir() or snapshot.is_symlink():
    raise SystemExit("remote staged snapshot is missing or unsafe")
expected_paths = set()
verified_blobs = set()
total_bytes = 0
for entry in entries:
    if not isinstance(entry, dict):
        raise SystemExit("remote staged manifest entry is malformed")
    raw_path = entry.get("path")
    digest = entry.get("sha256")
    size = entry.get("size")
    if not isinstance(raw_path, str) or "\\" in raw_path:
        raise SystemExit("remote staged manifest path is invalid")
    relative = PurePosixPath(raw_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise SystemExit("remote staged manifest path is unsafe")
    if re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None:
        raise SystemExit("remote staged manifest digest is invalid")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise SystemExit("remote staged manifest size is invalid")
    if raw_path in expected_paths:
        raise SystemExit("remote staged manifest path is duplicated")
    expected_paths.add(raw_path)
    total_bytes += size
    link = snapshot.joinpath(*relative.parts)
    blob = root / "blobs" / digest
    if not link.is_symlink() or link.resolve(strict=True) != blob.resolve(strict=True):
        raise SystemExit(f"remote snapshot link mismatch: {raw_path}")
    if not blob.is_file() or blob.is_symlink() or blob.stat().st_size != size:
        raise SystemExit(f"remote staged blob metadata mismatch: {digest}")
    if digest in verified_blobs:
        continue
    hasher = hashlib.sha256()
    with blob.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            hasher.update(chunk)
    if hasher.hexdigest() != digest:
        raise SystemExit(f"remote staged blob digest mismatch: {digest}")
    verified_blobs.add(digest)
actual_paths = {
    path.relative_to(snapshot).as_posix()
    for path in snapshot.rglob("*")
    if path.is_file() or path.is_symlink()
}
if actual_paths != expected_paths:
    raise SystemExit("remote staged snapshot file set differs from its manifest")
config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
quant = config.get("quantization_config")
gptqmodel_native = (
    isinstance(quant, dict)
    and isinstance(quant.get("meta"), dict)
    and "ds4rt_error_ledger" in quant["meta"]
)
qualification = manifest.get("qualification")
qualification_count = 0
development_evidence_count = 0
qualification_status = manifest.get("qualification_status")
if gptqmodel_native:
    if qualification_status is None:
        expected_qualification = {
            "retained-native.json": "ds4rt-exl3-retained-native-integrity-v1",
            "expert-quality.json": "ds4rt-exl3-checkpoint-quality-v1",
        }
        evidence = qualification
        evidence_root_name = "ds4rt-qualifications"
        quality_name = "expert-quality.json"
        qualification_status = "production-qualified"
        if manifest.get("development_evidence") not in (None, []):
            raise SystemExit("remote production cache has unexpected development evidence")
    elif qualification_status == "development-unqualified":
        if not allow_development_unqualified:
            raise SystemExit("remote GPTQModel cache is development-unqualified")
        if (
            qualification not in (None, [])
            or manifest.get("development_blockers") != [
                "complete token-aware all-layer quality evidence is absent",
                "natural MTP held-out activation quality evidence is absent",
            ]
        ):
            raise SystemExit("remote development qualification state is invalid")
        expected_qualification = {
            "retained-native.json": "ds4rt-exl3-retained-native-integrity-v1",
            "diagnostic-quality.json": "ds4rt-exl3-checkpoint-quality-v1",
        }
        evidence = manifest.get("development_evidence")
        evidence_root_name = "ds4rt-development-evidence"
        quality_name = "diagnostic-quality.json"
    else:
        raise SystemExit("remote GPTQModel qualification status is invalid")
    if not isinstance(evidence, list) or len(evidence) != 2:
        raise SystemExit("remote GPTQModel cache has no complete qualification evidence")
    qualification_root = root / evidence_root_name / revision
    if (
        not qualification_root.is_dir()
        or qualification_root.is_symlink()
        or {path.name for path in qualification_root.iterdir()}
        != set(expected_qualification)
    ):
        raise SystemExit("remote GPTQModel qualification directory is invalid")
    observed_names = set()
    for entry in evidence:
        if not isinstance(entry, dict):
            raise SystemExit("remote qualification manifest entry is malformed")
        name = entry.get("path")
        digest = entry.get("sha256")
        size = entry.get("size")
        schema = entry.get("schema")
        if (
            name not in expected_qualification
            or name in observed_names
            or schema != expected_qualification[name]
            or re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise SystemExit("remote qualification manifest entry is invalid")
        report_path = qualification_root / name
        if not os.path.isfile(report_path) or os.path.islink(report_path):
            raise SystemExit("remote qualification evidence is not a regular file")
        report_metadata = os.lstat(report_path)
        if report_metadata.st_size != size:
            raise SystemExit("remote qualification evidence size mismatch")
        report_payload = report_path.read_bytes()
        if hashlib.sha256(report_payload).hexdigest() != digest:
            raise SystemExit("remote qualification evidence digest mismatch")
        report = json.loads(report_payload)
        if not isinstance(report, dict) or report.get("schema") != schema:
            raise SystemExit("remote qualification evidence schema mismatch")
        if name == quality_name:
            contract = report.get("validation_contract")
            contract_sha = contract.get("sha256") if isinstance(contract, dict) else None
            if (
                re.fullmatch(r"[0-9a-f]{64}", str(contract_sha)) is None
                or entry.get("contract_sha256") != contract_sha
            ):
                raise SystemExit("remote quality contract digest mismatch")
        observed_names.add(name)
    if observed_names != set(expected_qualification):
        raise SystemExit("remote GPTQModel qualification evidence is incomplete")
    if qualification_status == "production-qualified":
        qualification_count = len(observed_names)
    else:
        development_evidence_count = len(observed_names)
elif (
    qualification not in (None, [])
    or qualification_status is not None
    or manifest.get("development_evidence") not in (None, [])
):
    raise SystemExit("remote non-GPTQModel cache has unexpected qualification evidence")
refs = root / "refs"
refs.mkdir(parents=True, exist_ok=True)
temporary = refs / f".main.{os.getpid()}.tmp"
try:
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(revision + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, refs / "main")
finally:
    temporary.unlink(missing_ok=True)
print(json.dumps({
    "revision": revision,
    "files": len(entries),
    "bytes": total_bytes,
    "verified_blobs": len(verified_blobs),
    "qualification_reports": qualification_count,
    "development_evidence_reports": development_evidence_count,
    "qualification_status": qualification_status,
}, sort_keys=True))
'''


@dataclass(frozen=True)
class StagedCacheContract:
    model_id: str
    revision: str
    root: Path
    snapshot: Path
    manifest: Path
    files: int
    bytes: int
    qualification_reports: int
    development_evidence_reports: int
    qualification_status: str | None


def default_hf_home() -> Path:
    configured = os.environ.get("HF_HOME")
    return Path(configured).expanduser() if configured else Path.home() / ".cache/huggingface"


def validate_model_id(model_id: str) -> tuple[str, str]:
    components = model_id.split("/")
    if len(components) != 2 or any(
        MODEL_COMPONENT_RE.fullmatch(component) is None for component in components
    ):
        raise ValueError("--model-id must be a safe two-component Hugging Face ID")
    return components[0], components[1]


def validate_hosts(raw_hosts: str | Sequence[str]) -> tuple[str, ...]:
    values = raw_hosts.split(",") if isinstance(raw_hosts, str) else list(raw_hosts)
    hosts = tuple(value.strip() for value in values if value.strip())
    if not hosts or len(set(hosts)) != len(hosts):
        raise ValueError("--hosts must contain unique host names")
    if any(HOST_RE.fullmatch(host) is None for host in hosts):
        raise ValueError("--hosts contains an unsafe host name")
    return hosts


def model_cache_root(hf_home: Path, model_id: str) -> Path:
    organization, repository = validate_model_id(model_id)
    return hf_home / "hub" / f"models--{organization}--{repository}"


def hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def is_gptqmodel_native_config(config: dict[str, Any]) -> bool:
    quant = config.get("quantization_config")
    return (
        isinstance(quant, dict)
        and isinstance(quant.get("meta"), dict)
        and "ds4rt_error_ledger" in quant["meta"]
    )


def validate_staged_qualification(
    root: Path,
    revision: str,
    manifest: dict[str, Any],
    snapshot: Path,
    *,
    allow_development_unqualified: bool = False,
) -> tuple[str | None, int, int]:
    config = read_json_object(snapshot / "config.json")
    entries = manifest.get("qualification")
    if not is_gptqmodel_native_config(config):
        if (
            entries not in (None, [])
            or manifest.get("qualification_status") is not None
            or manifest.get("development_evidence") not in (None, [])
        ):
            raise ValueError(
                "non-GPTQModel staged cache has unexpected qualification evidence"
            )
        return None, 0, 0
    qualification_status = manifest.get("qualification_status")
    if qualification_status is None:
        expected_schemas = QUALIFICATION_SCHEMAS
        evidence_root_name = "ds4rt-qualifications"
        quality_name = "expert-quality.json"
        qualification_status = "production-qualified"
        if manifest.get("development_evidence") not in (None, []):
            raise ValueError(
                "production GPTQModel staged cache has unexpected development evidence"
            )
    elif qualification_status == DEVELOPMENT_QUALIFICATION_STATUS:
        if not allow_development_unqualified:
            raise ValueError(
                "GPTQModel staged cache is development-unqualified; explicit override required"
            )
        if (
            entries not in (None, [])
            or manifest.get("development_blockers") != list(DEVELOPMENT_BLOCKERS)
        ):
            raise ValueError("GPTQModel staged development qualification state is invalid")
        entries = manifest.get("development_evidence")
        expected_schemas = DEVELOPMENT_EVIDENCE_SCHEMAS
        evidence_root_name = "ds4rt-development-evidence"
        quality_name = "diagnostic-quality.json"
    else:
        raise ValueError("GPTQModel staged qualification status is invalid")
    if not isinstance(entries, list) or len(entries) != len(expected_schemas):
        raise ValueError("GPTQModel staged cache has no complete qualification evidence")
    qualification_root = root / evidence_root_name / revision
    if (
        not qualification_root.is_dir()
        or qualification_root.is_symlink()
        or {path.name for path in qualification_root.iterdir()}
        != set(expected_schemas)
    ):
        raise ValueError("GPTQModel staged qualification directory is invalid")
    observed: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("staged qualification manifest entry is malformed")
        name = entry.get("path")
        digest = entry.get("sha256")
        size = entry.get("size")
        schema = entry.get("schema")
        if (
            name not in expected_schemas
            or name in observed
            or schema != expected_schemas[name]
            or REVISION_RE.fullmatch(str(digest)) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise ValueError("staged qualification manifest entry is invalid")
        report_path = qualification_root / str(name)
        if (
            not report_path.is_file()
            or report_path.is_symlink()
            or report_path.stat().st_size != size
            or hash_file(report_path) != digest
        ):
            raise ValueError(f"staged qualification evidence differs: {report_path}")
        report = read_json_object(report_path)
        if report.get("schema") != schema:
            raise ValueError("staged qualification evidence schema is invalid")
        if name == quality_name:
            contract = report.get("validation_contract")
            contract_sha256 = contract.get("sha256") if isinstance(contract, dict) else None
            if (
                REVISION_RE.fullmatch(str(contract_sha256)) is None
                or entry.get("contract_sha256") != contract_sha256
            ):
                raise ValueError("staged quality contract digest is invalid")
        observed.add(str(name))
    if observed != set(expected_schemas):
        raise ValueError("GPTQModel staged qualification evidence is incomplete")
    if qualification_status == "production-qualified":
        return qualification_status, len(observed), 0
    return qualification_status, 0, len(observed)


def load_staged_cache(
    hf_home: Path,
    model_id: str,
    *,
    verify_hashes: bool = False,
    allow_development_unqualified: bool = False,
) -> StagedCacheContract:
    root = model_cache_root(hf_home.expanduser().resolve(), model_id)
    ref = root / "refs" / "main"
    if not ref.is_file() or ref.is_symlink():
        raise ValueError(f"staged cache is missing regular refs/main: {ref}")
    revision = ref.read_text(encoding="utf-8").strip()
    if REVISION_RE.fullmatch(revision) is None:
        raise ValueError(f"staged cache has invalid revision {revision!r}")
    manifest_path = root / "ds4rt-manifests" / f"{revision}.json"
    manifest = read_json_object(manifest_path)
    entries = manifest.get("files")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("model_id") != model_id
        or manifest.get("revision") != revision
        or not isinstance(entries, list)
        or not entries
    ):
        raise ValueError("staged cache manifest contract is invalid")
    canonical = json.dumps(
        {"schema": MANIFEST_SCHEMA, "files": entries},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if hashlib.sha256(canonical).hexdigest() != revision:
        raise ValueError("staged cache manifest does not derive refs/main")
    snapshot = root / "snapshots" / revision
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise ValueError(f"staged cache snapshot is missing or unsafe: {snapshot}")
    paths: set[str] = set()
    total_bytes = 0
    verified: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("staged cache manifest contains a malformed entry")
        raw_path = entry.get("path")
        digest = entry.get("sha256")
        size = entry.get("size")
        if not isinstance(raw_path, str) or "\\" in raw_path:
            raise ValueError("staged cache manifest contains an invalid path")
        relative = PurePosixPath(raw_path)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError(f"staged cache manifest contains unsafe path {raw_path!r}")
        if REVISION_RE.fullmatch(str(digest)) is None:
            raise ValueError("staged cache manifest contains an invalid digest")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("staged cache manifest contains an invalid size")
        if raw_path in paths:
            raise ValueError(f"staged cache manifest duplicates path {raw_path}")
        paths.add(raw_path)
        total_bytes += size
        link = snapshot.joinpath(*relative.parts)
        blob = root / "blobs" / str(digest)
        if not link.is_symlink() or link.resolve(strict=True) != blob.resolve(strict=True):
            raise ValueError(f"staged snapshot link does not select its blob: {raw_path}")
        if not blob.is_file() or blob.is_symlink() or blob.stat().st_size != size:
            raise ValueError(f"staged cache blob metadata mismatch: {digest}")
        if verify_hashes and digest not in verified:
            if hash_file(blob) != digest:
                raise ValueError(f"staged cache blob digest mismatch: {digest}")
            verified.add(str(digest))
    actual = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual != paths:
        raise ValueError("staged cache snapshot file set differs from its manifest")
    (
        qualification_status,
        qualification_reports,
        development_evidence_reports,
    ) = validate_staged_qualification(
        root,
        revision,
        manifest,
        snapshot,
        allow_development_unqualified=allow_development_unqualified,
    )
    return StagedCacheContract(
        model_id=model_id,
        revision=revision,
        root=root,
        snapshot=snapshot,
        manifest=manifest_path,
        files=len(entries),
        bytes=total_bytes,
        qualification_reports=qualification_reports,
        development_evidence_reports=development_evidence_reports,
        qualification_status=qualification_status,
    )


def run_checked(
    command: Sequence[str],
    *,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        input=input_text,
        text=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def remote_hf_home(host: str) -> Path:
    result = run_checked(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            host,
            "python3",
            "-",
        ],
        input_text=REMOTE_HF_HOME_SOURCE + "\n",
    )
    raw = result.stdout.strip()
    if REMOTE_PATH_RE.fullmatch(raw) is None:
        raise ValueError(f"{host} returned unsafe HF_HOME {raw!r}")
    return Path(raw)


def sync_host(contract: StagedCacheContract, host: str) -> dict[str, Any]:
    remote_home = remote_hf_home(host)
    remote_root = model_cache_root(remote_home, contract.model_id)
    remote_raw = str(remote_root)
    if REMOTE_PATH_RE.fullmatch(remote_raw) is None:
        raise ValueError(f"unsafe remote model cache path for {host}: {remote_raw!r}")
    run_checked(["ssh", "-o", "BatchMode=yes", host, "mkdir", "-p", "--", remote_raw])
    run_checked(
        [
            "rdmasync",
            "--rdma=required",
            "--rdma-rails=auto",
            "--archive",
            "--partial",
            "--protect-args",
            "--exclude=/refs/main",
            "--rsync-path=/home/tj/.local/bin/rdmasync",
            f"{contract.root}/",
            f"{host}:{remote_raw}/",
        ]
    )
    verified = run_checked(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            host,
            "python3",
            "-",
            remote_raw,
            contract.revision,
            contract.model_id,
            "1" if contract.qualification_status == DEVELOPMENT_QUALIFICATION_STATUS else "0",
        ],
        input_text=REMOTE_VERIFY_SOURCE,
    )
    remote = json.loads(verified.stdout)
    if (
        remote.get("revision") != contract.revision
        or remote.get("files") != contract.files
        or remote.get("bytes") != contract.bytes
        or remote.get("qualification_reports") != contract.qualification_reports
        or remote.get("development_evidence_reports")
        != contract.development_evidence_reports
        or remote.get("qualification_status") != contract.qualification_status
    ):
        raise RuntimeError(f"{host} returned mismatched staged-cache verification")
    return {"host": host, **remote, "cache_root": remote_raw}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--hosts", default=",".join(DEFAULT_HOSTS))
    parser.add_argument("--hf-home", type=Path, default=default_hf_home())
    parser.add_argument(
        "--verify-local-hashes",
        action="store_true",
        help="rehash local blobs before transfer (remote blobs are always rehashed)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the local immutable cache and print the transfer plan",
    )
    parser.add_argument(
        "--allow-development-unqualified",
        action="store_true",
        help="sync an explicitly marked development-only snapshot for WIP serving",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        hosts = validate_hosts(args.hosts)
        contract = load_staged_cache(
            args.hf_home,
            args.model_id,
            verify_hashes=args.verify_local_hashes,
            allow_development_unqualified=args.allow_development_unqualified,
        )
        if args.dry_run:
            results: list[dict[str, Any]] = []
        else:
            results = []
            with ThreadPoolExecutor(max_workers=len(hosts)) as executor:
                futures = {
                    executor.submit(sync_host, contract, host): host for host in hosts
                }
                for future in as_completed(futures):
                    results.append(future.result())
            results.sort(key=lambda result: hosts.index(str(result["host"])))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema": "ds4rt-hf-spark-sync-v1",
                "model_id": contract.model_id,
                "revision": contract.revision,
                "files": contract.files,
                "bytes": contract.bytes,
                "qualification_reports": contract.qualification_reports,
                "development_evidence_reports": contract.development_evidence_reports,
                "qualification_status": contract.qualification_status,
                "hosts": list(hosts),
                "dry_run": args.dry_run,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
