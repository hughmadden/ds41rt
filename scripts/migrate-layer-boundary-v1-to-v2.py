#!/usr/bin/env python3
"""One-time migration of a SHA-256 layer boundary to XXH3-128.

The quantizer must be stopped while this tool runs.  Legacy activation bytes
are authenticated once with their recorded SHA-256 digests, then hard-linked
into a v2 generation whose large payloads use XXH3-128.  The v1 generation is
moved to a sibling forensic directory only after v2 is durable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from safetensors.torch import load_file as load_safetensors_file
import xxhash


SCHEMA = "ds4rt.deepseek-v4-layer-boundary"
LEGACY_VERSION = 1
CURRENT_VERSION = 2
PAYLOAD_HASH_ALGORITHM = "xxh3-128"
MANIFEST_FILENAME = "manifest.json"
COMMITTED_DIRECTORY = re.compile(
    r"layer-(?P<layer>[0-9]{6})-(?P<digest>[0-9a-f]{16})\Z"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
XXH3_128_RE = re.compile(r"[0-9a-f]{32}\Z")


class MigrationError(RuntimeError):
    """The legacy generation cannot be migrated without ambiguity."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MigrationError(f"cannot read manifest: {path}") from error
    if not isinstance(value, dict):
        raise MigrationError(f"manifest is not an object: {path}")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(canonical_json(manifest) + b"\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _manifest_body(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }


def _validate_manifest_binding(
    directory: Path, manifest: dict[str, Any], expected_version: int
) -> None:
    match = COMMITTED_DIRECTORY.fullmatch(directory.name)
    digest = manifest.get("manifest_sha256")
    body = _manifest_body(manifest)
    if (
        match is None
        or not directory.is_dir()
        or directory.is_symlink()
        or manifest.get("schema") != SCHEMA
        or manifest.get("schema_version") != expected_version
        or not isinstance(digest, str)
        or SHA256_RE.fullmatch(digest) is None
        or hashlib.sha256(canonical_json(body)).hexdigest() != digest
        or match.group("digest") != digest[:16]
        or int(match.group("layer")) != manifest.get("layer_index")
    ):
        raise MigrationError(f"generation failed manifest validation: {directory}")


def _safe_payload_path(directory: Path, record: dict[str, Any]) -> Path:
    relative = record.get("file")
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or Path(relative).parts[:1] != ("activations",)
    ):
        raise MigrationError("manifest contains an unsafe activation path")
    return directory / relative


def _actual_files(directory: Path) -> set[str]:
    files: set[str] = set()
    for path in directory.rglob("*"):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink():
            raise MigrationError(f"generation contains a symlink: {relative}")
        if path.is_dir():
            if relative != "activations":
                raise MigrationError(f"generation has an unexpected directory: {relative}")
        elif path.is_file():
            files.add(relative)
        else:
            raise MigrationError(f"generation has an unsupported entry: {relative}")
    return files


def _hash_legacy_payload(path: Path) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    xxh3 = xxhash.xxh3_128()
    with path.open("rb") as source:
        while block := source.read(32 * 1024 * 1024):
            sha256.update(block)
            xxh3.update(block)
    return sha256.hexdigest(), xxh3.hexdigest()


def build_v2_manifest(
    source: Path, legacy: dict[str, Any]
) -> tuple[dict[str, Any], list[tuple[Path, str]]]:
    _validate_manifest_binding(source, legacy, LEGACY_VERSION)
    if "payload_hash_algorithm" in legacy:
        raise MigrationError("legacy manifest unexpectedly declares a payload hash")
    shards = legacy.get("activation_shards")
    if not isinstance(shards, list) or not shards:
        raise MigrationError("legacy manifest contains no activation shards")
    expected_files = {MANIFEST_FILENAME}
    links: list[tuple[Path, str]] = []
    migrated_shards: list[dict[str, Any]] = []
    for index, record in enumerate(shards):
        if not isinstance(record, dict):
            raise MigrationError(f"legacy activation shard {index} is not an object")
        path = _safe_payload_path(source, record)
        relative = path.relative_to(source).as_posix()
        recorded_sha256 = record.get("sha256")
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != record.get("bytes")
            or not isinstance(recorded_sha256, str)
            or SHA256_RE.fullmatch(recorded_sha256) is None
        ):
            raise MigrationError(f"legacy activation shard {index} is invalid")
        payload_sha256, payload_xxh3 = _hash_legacy_payload(path)
        if payload_sha256 != recorded_sha256:
            raise MigrationError(
                f"legacy activation shard {index} failed SHA-256 validation"
            )
        tensors = load_safetensors_file(path, device="cpu")
        hidden = tensors.get("hidden")
        tensor = record.get("tensor")
        if (
            set(tensors) != {"hidden"}
            or hidden is None
            or not isinstance(tensor, dict)
            or tensor.get("shape") != list(hidden.shape)
            or tensor.get("dtype") != str(hidden.dtype)
            or tensor.get("bytes") != hidden.numel() * hidden.element_size()
        ):
            raise MigrationError(
                f"legacy activation shard {index} failed tensor validation"
            )
        migrated_record = {
            key: value for key, value in record.items() if key != "sha256"
        }
        migrated_record["xxh3_128"] = payload_xxh3
        migrated_shards.append(migrated_record)
        links.append((path, relative))
        expected_files.add(relative)
    if _actual_files(source) != expected_files:
        raise MigrationError("legacy generation file set is inconsistent")

    body = _manifest_body(legacy)
    body["schema_version"] = CURRENT_VERSION
    body["payload_hash_algorithm"] = PAYLOAD_HASH_ALGORITHM
    body["activation_shards"] = migrated_shards
    return {
        **body,
        "manifest_sha256": hashlib.sha256(canonical_json(body)).hexdigest(),
    }, links


def _validate_existing_v2(directory: Path, expected: dict[str, Any]) -> None:
    manifest = _read_json(directory / MANIFEST_FILENAME)
    _validate_manifest_binding(directory, manifest, CURRENT_VERSION)
    if (
        manifest != expected
        or manifest.get("payload_hash_algorithm") != PAYLOAD_HASH_ALGORITHM
    ):
        raise MigrationError("existing v2 generation differs from the migration")
    shards = manifest.get("activation_shards", [])
    expected_files = {MANIFEST_FILENAME}
    for index, record in enumerate(shards):
        path = _safe_payload_path(directory, record)
        digest = xxhash.xxh3_128()
        if not path.is_file() or path.is_symlink() or path.stat().st_size != record.get(
            "bytes"
        ):
            raise MigrationError(f"v2 activation shard {index} is invalid")
        with path.open("rb") as source:
            while block := source.read(32 * 1024 * 1024):
                digest.update(block)
        if (
            not isinstance(record.get("xxh3_128"), str)
            or XXH3_128_RE.fullmatch(record["xxh3_128"]) is None
            or digest.hexdigest() != record["xxh3_128"]
        ):
            raise MigrationError(f"v2 activation shard {index} failed XXH3 validation")
        expected_files.add(path.relative_to(directory).as_posix())
    if _actual_files(directory) != expected_files:
        raise MigrationError("v2 generation file set is inconsistent")


def migrate(
    boundary_root: Path,
    *,
    legacy_root: Path | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    boundary_root = boundary_root.expanduser().resolve()
    if not boundary_root.is_dir() or boundary_root.is_symlink():
        raise MigrationError("boundary root is missing or unsafe")
    legacy_root = (
        legacy_root.expanduser().resolve()
        if legacy_root is not None
        else boundary_root.parent.parent
        / f"{boundary_root.parent.name}.{boundary_root.name}.legacy-v1"
    )
    if legacy_root == boundary_root or boundary_root in legacy_root.parents:
        raise MigrationError("legacy forensic root must be outside active boundary state")

    generations: list[tuple[Path, dict[str, Any]]] = []
    for path in boundary_root.iterdir():
        if path.name.startswith(".layer-") and path.name.endswith(".tmp"):
            raise MigrationError("boundary root contains an incomplete generation")
        if COMMITTED_DIRECTORY.fullmatch(path.name) is None:
            raise MigrationError(f"unexpected boundary-root entry: {path.name}")
        manifest = _read_json(path / MANIFEST_FILENAME)
        version = manifest.get("schema_version")
        if version not in (LEGACY_VERSION, CURRENT_VERSION):
            raise MigrationError(f"unsupported boundary schema version: {version}")
        _validate_manifest_binding(path, manifest, version)
        generations.append((path, manifest))
    legacy_generations = [item for item in generations if item[1]["schema_version"] == 1]
    current_generations = [item for item in generations if item[1]["schema_version"] == 2]

    if not legacy_generations:
        if len(current_generations) != 1:
            raise MigrationError("active boundary does not contain exactly one v2 generation")
        current_path, current = current_generations[0]
        if current.get("payload_hash_algorithm") != PAYLOAD_HASH_ALGORITHM:
            raise MigrationError("active v2 generation has an unexpected hash contract")
        return {
            "status": "already-v2",
            "active": os.fspath(current_path),
            "manifest_sha256": current["manifest_sha256"],
        }
    if len(legacy_generations) != 1 or len(current_generations) > 1:
        raise MigrationError("boundary root has ambiguous generations")

    source, legacy = legacy_generations[0]
    migrated, links = build_v2_manifest(source, legacy)
    destination = boundary_root / (
        f"layer-{legacy['layer_index']:06d}-{migrated['manifest_sha256'][:16]}"
    )
    report = {
        "status": "ready" if not apply else "migrated",
        "source": os.fspath(source),
        "active": os.fspath(destination),
        "legacy_archive": os.fspath(legacy_root / source.name),
        "activation_shards": len(links),
        "activation_file_bytes": sum(path.stat().st_size for path, _ in links),
        "manifest_sha256": migrated["manifest_sha256"],
        "payload_hash_algorithm": PAYLOAD_HASH_ALGORITHM,
    }
    if not apply:
        return report

    if destination.exists():
        _validate_existing_v2(destination, migrated)
    else:
        temporary = Path(
            tempfile.mkdtemp(prefix=".layer-migrate-v2-", suffix=".tmp", dir=boundary_root)
        )
        try:
            activation_root = temporary / "activations"
            activation_root.mkdir()
            for source_path, relative in links:
                target = temporary / relative
                os.link(source_path, target)
                if source_path.stat().st_ino != target.stat().st_ino:
                    raise MigrationError("activation migration did not create a hard link")
            _fsync_directory(activation_root)
            _write_manifest(temporary / MANIFEST_FILENAME, migrated)
            _fsync_directory(temporary)
            os.replace(temporary, destination)
            _fsync_directory(boundary_root)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        _validate_existing_v2(destination, migrated)

    legacy_root.mkdir(parents=True, exist_ok=True)
    if legacy_root.is_symlink():
        raise MigrationError("legacy forensic root cannot be a symlink")
    archive = legacy_root / source.name
    if archive.exists():
        raise MigrationError(f"legacy forensic destination already exists: {archive}")
    os.replace(source, archive)
    _fsync_directory(boundary_root)
    _fsync_directory(legacy_root)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary-root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the migration; the default validates and reports only",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            migrate(
                args.boundary_root,
                legacy_root=args.legacy_root,
                apply=args.apply,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
