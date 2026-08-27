#!/usr/bin/env python3
"""Fail-closed exact-plan resume for a stopped DS4RT quantization container.

The source container is immutable evidence of the original launch.  This tool
clones its content-addressed image, environment, mounts, and command, appending
only ``--resume`` after checking the saved plan and the physical GPU identities.
It never renders the source environment or the reconstructed Docker command.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Sequence


SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
IMAGE_ID_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
USER_RE = re.compile(r"[1-9][0-9]*:[1-9][0-9]*\Z")
CONTAINER_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
PLAN_FILENAME = "ds4rt-gptqmodel-plan.json"
EXECUTION_UPGRADE_FILENAME = "ds4rt-execution-upgrade.json"
EXECUTION_UPGRADE_HISTORY_DIRNAME = "execution-upgrade-history"
EXECUTION_UPGRADE_SCHEMA = "ds4rt-deepseek-v4-execution-upgrade-v1"
IMAGE_OWNED_UPGRADE_ENV = frozenset(
    {
        "DS4RT_QUANT_BASE_IMAGE",
        "DS4RT_QUANT_BUILD_REQUIREMENTS_SHA256",
        "DS4RT_QUANT_CUDA_ARCH",
        "DS4RT_QUANT_MIN_GPUS",
        "DS4RT_QUANT_PYTHON_VERSION",
        "DS4RT_QUANT_REQUIREMENTS_LOCK",
        "DS4RT_QUANT_REQUIREMENTS_SHA256",
        "DS4RT_QUANT_ROLE",
        "DS4RT_QUANT_TARGET_PLATFORM",
    }
)
PLAN_SCHEMA = "ds4rt-deepseek-v4-gptqmodel-plan-v5"
PREVIOUS_PLAN_SCHEMA = "ds4rt-deepseek-v4-gptqmodel-plan-v6"
CURRENT_PLAN_SCHEMA = "ds4rt-deepseek-v4-gptqmodel-plan-v7"
SUPPORTED_PLAN_SCHEMAS = frozenset(
    (PLAN_SCHEMA, PREVIOUS_PLAN_SCHEMA, CURRENT_PLAN_SCHEMA)
)
STRICT_STORAGE_PLAN_SCHEMAS = frozenset(
    (PREVIOUS_PLAN_SCHEMA, CURRENT_PLAN_SCHEMA)
)
QUANTIZER = "/opt/ds4rt/quantization/quantize_flash_gptqmodel.py"
LEGACY_QUANTIZER_SUFFIX = PurePosixPath(
    "quantization/quantize_flash_gptqmodel.py"
)
ENTRYPOINT = [
    "/usr/bin/tini",
    "--",
    "/usr/local/bin/ds4rt-quantization-entrypoint",
]
DEFAULT_PREFLIGHT_REPORT = "/tmp/ds4rt-quantization-preflight.json"
ALL_GPU_REQUEST = [
    {
        "Driver": "",
        "Count": -1,
        "DeviceIDs": None,
        "Capabilities": [["gpu"]],
        "Options": {},
    }
]


class ResumeError(RuntimeError):
    """The stopped container cannot be resumed without contract drift."""


Run = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class Bind:
    source: Path
    destination: PurePosixPath
    options: str | None
    raw: str


@dataclass(frozen=True)
class ResumeSpec:
    source_container: str
    target_container: str
    image_id: str
    command: tuple[str, ...]
    resume_arguments: tuple[str, ...]
    environment: tuple[str, ...] = field(repr=False)
    binds: tuple[Bind, ...]
    network_mode: str
    runtime: str
    working_dir: str
    user: str
    ipc_mode: str
    shm_size: int
    memory: int
    memory_swap: int
    ulimits: tuple[tuple[str, int, int], ...]
    security_options: tuple[str, ...]
    plan_sha256: str
    expected_gpus: tuple[tuple[int, str], ...]
    gpu_device_ids: tuple[str, ...] | None
    preflight_report: str


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResumeError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ResumeError(f"{label} is not a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ResumeError(f"cannot hash required file: {path}") from exc
    return digest.hexdigest()


def _run(
    command: Sequence[str],
    *,
    runner: Run = subprocess.run,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        result = runner(
            list(command),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ResumeError(f"required executable is unavailable: {command[0]}") from exc
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        suffix = f": {detail}" if detail else ""
        raise ResumeError(f"{command[0]} command failed{suffix}")
    return result


def inspect_container(name: str, *, runner: Run = subprocess.run) -> dict[str, Any]:
    result = _run(
        ["docker", "container", "inspect", name], runner=runner, check=False
    )
    if result.returncode != 0:
        raise ResumeError(f"source container does not exist: {name}")
    try:
        values = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ResumeError("Docker returned invalid source-container metadata") from exc
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
        raise ResumeError("Docker returned ambiguous source-container metadata")
    return values[0]


def container_exists(name: str, *, runner: Run = subprocess.run) -> bool:
    result = _run(
        [
            "docker",
            "container",
            "ls",
            "--all",
            "--filter",
            f"name=^/{name}$",
            "--format",
            "{{.Names}}",
        ],
        runner=runner,
        check=True,
    )
    return name in {line.strip() for line in result.stdout.splitlines() if line.strip()}


def parse_bind(raw: str) -> Bind:
    if not isinstance(raw, str) or not raw or "\n" in raw or "\r" in raw:
        raise ResumeError("source container contains an invalid bind mount")
    fields = raw.split(":")
    if len(fields) == 2:
        source_text, destination_text = fields
        options = None
    elif len(fields) == 3:
        source_text, destination_text, options = fields
    else:
        raise ResumeError("source container contains a noncanonical bind mount")
    source = Path(source_text)
    destination = PurePosixPath(destination_text)
    if (
        not source.is_absolute()
        or not destination.is_absolute()
        or ".." in destination.parts
        or options not in {None, "ro", "rw"}
    ):
        raise ResumeError("source container contains an unsupported bind mount")
    if not source.exists() or source.is_symlink():
        raise ResumeError(f"bind source is missing or symbolic: {source}")
    return Bind(source=source, destination=destination, options=options, raw=raw)


def parse_ulimits(value: Any) -> tuple[tuple[str, int, int], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ResumeError("source container has invalid ulimits")
    parsed: list[tuple[str, int, int]] = []
    for record in value:
        if not isinstance(record, dict):
            raise ResumeError("source container has invalid ulimits")
        name = record.get("Name")
        soft = record.get("Soft")
        hard = record.get("Hard")
        if (
            not isinstance(name, str)
            or ENV_NAME_RE.fullmatch(name) is None
            or isinstance(soft, bool)
            or not isinstance(soft, int)
            or isinstance(hard, bool)
            or not isinstance(hard, int)
        ):
            raise ResumeError("source container has invalid ulimits")
        parsed.append((name, soft, hard))
    if len({name for name, _soft, _hard in parsed}) != len(parsed):
        raise ResumeError("source container repeats a ulimit")
    return tuple(sorted(parsed))


def parse_security_options(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if (
        not isinstance(value, list)
        or any(
            not isinstance(option, str)
            or not option
            or "\n" in option
            or "\r" in option
            for option in value
        )
    ):
        raise ResumeError("source container has invalid security options")
    return tuple(sorted(value))


def map_container_path(raw_path: str, binds: Sequence[Bind]) -> Path:
    path = PurePosixPath(raw_path)
    if not path.is_absolute() or ".." in path.parts:
        raise ResumeError(f"container path is not canonical: {raw_path!r}")
    candidates = [
        bind
        for bind in binds
        if path == bind.destination or path.is_relative_to(bind.destination)
    ]
    if not candidates:
        raise ResumeError(f"container path is not backed by a host bind: {raw_path}")
    bind = max(candidates, key=lambda item: len(item.destination.parts))
    relative = path.relative_to(bind.destination)
    return bind.source.joinpath(*relative.parts)


def backing_bind(raw_path: str, binds: Sequence[Bind]) -> Bind:
    path = PurePosixPath(raw_path)
    if not path.is_absolute() or ".." in path.parts:
        raise ResumeError(f"container path is not canonical: {raw_path!r}")
    candidates = [
        bind
        for bind in binds
        if path == bind.destination or path.is_relative_to(bind.destination)
    ]
    if not candidates:
        raise ResumeError(f"container path is not backed by a host bind: {raw_path}")
    return max(candidates, key=lambda item: len(item.destination.parts))


def is_legacy_quantizer_path(raw_path: str) -> bool:
    path = PurePosixPath(raw_path)
    return (
        path.is_absolute()
        and ".." not in path.parts
        and path.parts[-len(LEGACY_QUANTIZER_SUFFIX.parts) :]
        == LEGACY_QUANTIZER_SUFFIX.parts
    )


def option_value(command: Sequence[str], option: str) -> str:
    indices = [index for index, value in enumerate(command) if value == option]
    if len(indices) != 1 or indices[0] + 1 >= len(command):
        raise ResumeError(f"source command must contain exactly one {option}")
    value = command[indices[0] + 1]
    if value.startswith("--"):
        raise ResumeError(f"source command has no value for {option}")
    return value


def optional_option_value(command: Sequence[str], option: str) -> str | None:
    indices = [index for index, value in enumerate(command) if value == option]
    if not indices:
        return None
    if len(indices) != 1 or indices[0] + 1 >= len(command):
        raise ResumeError(f"source command must contain at most one {option}")
    value = command[indices[0] + 1]
    if value.startswith("--"):
        raise ResumeError(f"source command has no value for {option}")
    return value


def parse_environment(values: Any) -> tuple[tuple[str, ...], dict[str, str]]:
    if not isinstance(values, list) or not values:
        raise ResumeError("source container has no environment contract")
    records: list[str] = []
    parsed: dict[str, str] = {}
    for record in values:
        if not isinstance(record, str) or "=" not in record:
            raise ResumeError("source container contains an invalid environment record")
        key, value = record.split("=", 1)
        if (
            ENV_NAME_RE.fullmatch(key) is None
            or key in parsed
            or any(character in value for character in ("\x00", "\n", "\r"))
        ):
            raise ResumeError("source container contains an unsafe environment record")
        parsed[key] = value
        records.append(record)
    return tuple(records), parsed


def validate_plan(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ResumeError(f"saved plan is missing or nonregular: {path}")
    plan = read_json_object(path, "saved quantization plan")
    digest = plan.get("plan_sha256")
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if (
        plan.get("schema") not in SUPPORTED_PLAN_SCHEMAS
        or not isinstance(digest, str)
        or SHA256_RE.fullmatch(digest) is None
        or hashlib.sha256(canonical_json(body)).hexdigest() != digest
    ):
        raise ResumeError("saved quantization plan has an invalid schema or digest")
    if digest != expected_sha256:
        raise ResumeError(
            "saved quantization plan does not match --expected-plan-sha256"
        )
    return plan


def validate_execution_upgrade_chain(
    run_state: Path,
    plan_sha256: str,
) -> tuple[dict[str, Any], ...]:
    latest_path = run_state / EXECUTION_UPGRADE_FILENAME
    if not latest_path.is_file() or latest_path.is_symlink():
        raise ResumeError("chained resume lacks a regular execution-upgrade record")
    latest = read_json_object(latest_path, "execution upgrade")
    history_root = run_state / EXECUTION_UPGRADE_HISTORY_DIRNAME
    history: dict[str, dict[str, Any]] = {}
    if history_root.exists():
        if not history_root.is_dir() or history_root.is_symlink():
            raise ResumeError("execution-upgrade history is not a regular directory")
        for path in history_root.iterdir():
            match = re.fullmatch(r"([0-9a-f]{64})\.json", path.name)
            if match is None or not path.is_file() or path.is_symlink():
                raise ResumeError("execution-upgrade history contains an unsafe entry")
            history[match.group(1)] = read_json_object(
                path, "archived execution upgrade"
            )

    def validate(record: dict[str, Any], expected_digest: str | None) -> str:
        digest = record.get("upgrade_sha256")
        body = {
            key: value for key, value in record.items() if key != "upgrade_sha256"
        }
        if (
            record.get("schema") != EXECUTION_UPGRADE_SCHEMA
            or record.get("parent_plan_sha256") != plan_sha256
            or not isinstance(digest, str)
            or SHA256_RE.fullmatch(digest) is None
            or hashlib.sha256(canonical_json(body)).hexdigest() != digest
            or (expected_digest is not None and digest != expected_digest)
        ):
            raise ResumeError("execution-upgrade history is invalid")
        return digest

    validate(latest, None)
    chain = [latest]
    links = [
        latest.get(field)
        for field in ("previous_upgrade_sha256", "previous_failed_upgrade_sha256")
        if latest.get(field) is not None
    ]
    visited: set[str] = set()
    while links:
        if len(links) != 1 or links[0] in visited or links[0] not in history:
            raise ResumeError("execution-upgrade history chain is incomplete")
        digest = links[0]
        visited.add(digest)
        record = history[digest]
        validate(record, digest)
        chain.append(record)
        links = [
            record.get(field)
            for field in (
                "previous_upgrade_sha256",
                "previous_failed_upgrade_sha256",
            )
            if record.get(field) is not None
        ]
    return tuple(chain)


def expected_gpus_from_plan(plan: dict[str, Any]) -> tuple[tuple[int, str], ...]:
    preflight = plan.get("preflight")
    gpus = preflight.get("gpus") if isinstance(preflight, dict) else None
    if not isinstance(gpus, list) or len(gpus) not in {1, 2}:
        raise ResumeError("saved plan does not bind one or two coordinator GPUs")
    result: list[tuple[int, str]] = []
    for position, gpu in enumerate(gpus):
        if (
            not isinstance(gpu, dict)
            or gpu.get("index") != position
            or not isinstance(gpu.get("uuid"), str)
            or not gpu["uuid"].startswith("GPU-")
        ):
            raise ResumeError("saved plan contains invalid coordinator GPU identities")
        result.append((position, gpu["uuid"]))
    if len({uuid for _, uuid in result}) != len(result):
        raise ResumeError("saved plan repeats a coordinator GPU identity")
    return tuple(result)


def parse_gpu_request(value: Any) -> tuple[str, ...] | None:
    """Return explicit UUIDs, or ``None`` for Docker's canonical all-GPU request."""

    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise ResumeError("source container has an unsupported GPU request")
    request = value[0]
    if (
        set(request) != {"Driver", "Count", "DeviceIDs", "Capabilities", "Options"}
        or request.get("Driver") != ""
        or request.get("Capabilities") != [["gpu"]]
        or request.get("Options") != {}
    ):
        raise ResumeError("source container has an unsupported GPU request")
    if request.get("Count") == -1 and request.get("DeviceIDs") is None:
        return None
    device_ids = request.get("DeviceIDs")
    if (
        request.get("Count") != 0
        or not isinstance(device_ids, list)
        or len(device_ids) not in {1, 2}
        or any(
            not isinstance(uuid, str) or not uuid.startswith("GPU-")
            for uuid in device_ids
        )
        or len(set(device_ids)) != len(device_ids)
    ):
        raise ResumeError("source container has an unsupported exact-GPU request")
    return tuple(device_ids)


def docker_gpu_request(device_ids: tuple[str, ...] | None) -> list[dict[str, Any]]:
    if device_ids is None:
        return ALL_GPU_REQUEST
    return [
        {
            "Driver": "",
            "Count": 0,
            "DeviceIDs": list(device_ids),
            "Capabilities": [["gpu"]],
            "Options": {},
        }
    ]


def validate_live_gpus(
    live_gpus: tuple[tuple[int, str], ...],
    expected_gpus: tuple[tuple[int, str], ...],
    gpu_device_ids: tuple[str, ...] | None,
) -> None:
    if gpu_device_ids is None:
        if live_gpus != expected_gpus:
            raise ResumeError(
                "live coordinator GPUs do not exactly match the saved index/UUID identities"
            )
        return

    expected_uuids = tuple(uuid for _, uuid in expected_gpus)
    if gpu_device_ids != expected_uuids:
        raise ResumeError(
            "source exact-GPU request differs from the saved coordinator GPU identities"
        )
    live_uuids = [uuid for _, uuid in live_gpus]
    if len(live_uuids) != len(set(live_uuids)) or any(
        uuid not in live_uuids for uuid in expected_uuids
    ):
        raise ResumeError(
            "live host does not contain every exact GPU UUID bound by the saved plan"
        )


def parse_host_gpus(output: str) -> tuple[tuple[int, str], ...]:
    records: list[tuple[int, str]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise ResumeError("nvidia-smi returned a malformed GPU identity row")
        try:
            index = int(fields[0])
        except ValueError as exc:
            raise ResumeError("nvidia-smi returned a malformed GPU index") from exc
        if not fields[1].startswith("GPU-"):
            raise ResumeError("nvidia-smi returned a malformed GPU UUID")
        records.append((index, fields[1]))
    return tuple(sorted(records))


def live_host_gpus(*, runner: Run = subprocess.run) -> tuple[tuple[int, str], ...]:
    result = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        runner=runner,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        suffix = f": {detail}" if detail else ""
        raise ResumeError(f"coordinator GPU preflight failed; do not resume{suffix}")
    return parse_host_gpus(result.stdout)


def _require_container_contract(metadata: dict[str, Any]) -> tuple[str, ...] | None:
    config = metadata.get("Config")
    host = metadata.get("HostConfig")
    state = metadata.get("State")
    if not all(isinstance(value, dict) for value in (config, host, state)):
        raise ResumeError("source container metadata is incomplete")
    if state.get("Running") or state.get("Paused") or state.get("Restarting"):
        raise ResumeError("source container must be completely stopped")
    if config.get("Entrypoint") != ENTRYPOINT:
        raise ResumeError("source container does not use the qualified entrypoint")
    user = config.get("User")
    if not isinstance(user, str) or (user and USER_RE.fullmatch(user) is None):
        raise ResumeError("source container uses an unsafe user identity")
    if host.get("NetworkMode") not in {"host", "bridge"}:
        raise ResumeError("source container uses unsupported networking")
    gpu_device_ids = parse_gpu_request(host.get("DeviceRequests"))
    if host.get("Privileged") or host.get("ReadonlyRootfs"):
        raise ResumeError("source container has unsupported privilege/rootfs settings")
    if host.get("RestartPolicy") != {"Name": "no", "MaximumRetryCount": 0}:
        raise ResumeError("source container has an automatic restart policy")
    if host.get("AutoRemove"):
        raise ResumeError("source container uses automatic removal")
    return gpu_device_ids


def build_resume_spec(
    metadata: dict[str, Any],
    *,
    source_container: str,
    target_container: str,
    expected_plan_sha256: str,
    live_gpus: tuple[tuple[int, str], ...],
    upgrade_image_id: str | None = None,
) -> ResumeSpec:
    gpu_device_ids = _require_container_contract(metadata)
    config = metadata["Config"]
    host = metadata["HostConfig"]
    parent_image_id = metadata.get("Image")
    if (
        not isinstance(parent_image_id, str)
        or IMAGE_ID_RE.fullmatch(parent_image_id) is None
    ):
        raise ResumeError("source container does not bind a content-addressed image")
    environment, environment_map = parse_environment(config.get("Env"))
    require_image_digest = environment_map.get(
        "DS4RT_QUANT_REQUIRE_IMAGE_DIGEST"
    )
    if (
        environment_map.get("DS4RT_QUANT_IMAGE_DIGEST") != parent_image_id
        or require_image_digest not in {None, "1"}
        or (require_image_digest is None and upgrade_image_id is None)
        or environment_map.get("DS4RT_QUANT_ROLE") != "coordinator"
        or environment_map.get("DS4RT_QUANT_TARGET_PLATFORM") != "linux/amd64"
        or environment_map.get("DS4RT_QUANT_CUDA_ARCH") != "120"
    ):
        raise ResumeError("source environment does not bind the qualified image/GPU role")

    raw_command = config.get("Cmd")
    legacy_quantizer = (
        isinstance(raw_command, list)
        and len(raw_command) >= 2
        and isinstance(raw_command[1], str)
        and raw_command[1] != QUANTIZER
        and is_legacy_quantizer_path(raw_command[1])
    )
    if (
        not isinstance(raw_command, list)
        or len(raw_command) < 3
        or any(not isinstance(value, str) for value in raw_command)
        or raw_command[0] != "python"
        or (raw_command[1] != QUANTIZER and not legacy_quantizer)
        or (legacy_quantizer and upgrade_image_id is None)
        or "--plan-only" in raw_command
    ):
        raise ResumeError("source command is not an unfinished production quantization")
    source_is_upgrade_resume = raw_command[-2:] == [
        "--execution-upgrade",
        "--resume",
    ]
    command_values = list(raw_command)
    if command_values[-2:] == ["--execution-upgrade", "--resume"]:
        command_values = command_values[:-2]
    elif command_values[-1:] == ["--resume"]:
        command_values = command_values[:-1]
    if "--resume" in command_values or "--execution-upgrade" in command_values:
        raise ResumeError("source command has a noncanonical resume suffix")
    raw_binds = host.get("Binds")
    if not isinstance(raw_binds, list) or not raw_binds:
        raise ResumeError("source container has no host bind mounts")
    binds = tuple(parse_bind(raw) for raw in raw_binds)
    if len({bind.destination for bind in binds}) != len(binds):
        raise ResumeError("source container repeats a bind destination")
    if legacy_quantizer:
        legacy_path = command_values[1]
        legacy_bind = backing_bind(legacy_path, binds)
        host_quantizer = map_container_path(legacy_path, binds)
        if (
            legacy_bind.options != "ro"
            or not host_quantizer.is_file()
            or host_quantizer.is_symlink()
        ):
            raise ResumeError(
                "legacy source quantizer is not a regular read-only bind"
            )
        command_values[1] = QUANTIZER
    command = tuple(command_values)

    output = option_value(command, "--output")
    command_preflight = option_value(command, "--preflight-report")
    offload = option_value(command, "--offload-dir")
    prefix_store = option_value(command, "--mtp-prefix-store")
    host_output = map_container_path(output, binds)
    run_state_container = PurePosixPath(
        optional_option_value(command, "--run-state-dir")
        or PurePosixPath(output).with_name(
            f".{PurePosixPath(output).name}.ds4rt-run"
        )
    )
    host_run_state = map_container_path(os.fspath(run_state_container), binds)
    plan = validate_plan(host_run_state / PLAN_FILENAME, expected_plan_sha256)
    command_checkpoint = optional_option_value(
        command, "--projection-checkpoint-dir"
    ) or os.fspath(run_state_container / "projection-checkpoints")
    command_active_source = optional_option_value(
        command, "--active-layer-source-dir"
    ) or os.fspath(run_state_container / "active-layer-source")
    split_paths_differ = plan.get("schema") in STRICT_STORAGE_PLAN_SCHEMAS and (
        plan.get("projection_checkpoint_dir") != command_checkpoint
        or plan.get("active_layer_source_dir") != command_active_source
        or plan.get("projection_checkpoint", {}).get("root") != command_checkpoint
    )
    if (
        plan.get("output") != output
        or plan.get("run_state_dir") != os.fspath(run_state_container)
        or plan.get("offload_dir") != offload
        or plan.get("mtp_prefix_store") != prefix_store
        or split_paths_differ
    ):
        raise ResumeError("source command paths differ from the saved plan")
    if plan.get("remote_workers") is not None and host.get("NetworkMode") != "host":
        raise ResumeError("distributed quantization requires host networking")
    if host_output.exists() or host_output.is_symlink():
        raise ResumeError("quantization output already exists; do not launch a raw resume")
    required_paths = [
        (os.fspath(run_state_container), "run state"),
        (offload, "offload directory"),
        (prefix_store, "MTP prefix store"),
    ]
    if plan.get("schema") in STRICT_STORAGE_PLAN_SCHEMAS:
        required_paths.extend(
            (
                (command_checkpoint, "projection checkpoint store"),
                (command_active_source, "active-layer source store"),
            )
        )
    for raw_path, label in required_paths:
        host_path = map_container_path(raw_path, binds)
        if not host_path.is_dir() or host_path.is_symlink():
            raise ResumeError(f"{label} is missing or nonregular: {host_path}")

    original_preflight_path = map_container_path(command_preflight, binds)
    original_preflight = read_json_object(
        original_preflight_path, "original coordinator preflight"
    )
    plan_preflight = plan.get("preflight")
    expected_gpus = expected_gpus_from_plan(plan)
    if environment_map.get("DS4RT_QUANT_MIN_GPUS") != str(len(expected_gpus)):
        raise ResumeError("source environment GPU count differs from the saved plan")
    if (
        not isinstance(plan_preflight, dict)
        or original_preflight.get("status") != "qualified"
        or original_preflight.get("role") != "coordinator"
        or original_preflight.get("image_digest") != parent_image_id
        or tuple(
            (gpu.get("index"), gpu.get("uuid"))
            for gpu in original_preflight.get("gpus", [])
            if isinstance(gpu, dict)
        )
        != expected_gpus
    ):
        raise ResumeError("original coordinator preflight differs from the saved plan")
    plan_preflight_is_current = (
        plan_preflight.get("path") == command_preflight
        and plan_preflight.get("sha256") == sha256_file(original_preflight_path)
        and plan_preflight.get("image_digest") == parent_image_id
    )
    if not plan_preflight_is_current:
        if not source_is_upgrade_resume:
            raise ResumeError(
                "source preflight differs from the parent plan without an upgrade chain"
            )
        upgrades = validate_execution_upgrade_chain(
            host_run_state, expected_plan_sha256
        )
        current_execution = {
            key: original_preflight.get(key)
            for key in ("image_digest", "gptqmodel", "python", "torch", "gpus")
        }
        if not any(
            upgrade.get("upgraded_execution") == current_execution
            for upgrade in upgrades
        ):
            raise ResumeError(
                "source preflight differs from the latest execution upgrade"
            )
    validate_live_gpus(live_gpus, expected_gpus, gpu_device_ids)

    network_mode = host.get("NetworkMode")
    runtime = host.get("Runtime")
    working_dir = config.get("WorkingDir")
    user = config.get("User")
    ipc_mode = host.get("IpcMode")
    shm_size = host.get("ShmSize")
    memory = host.get("Memory")
    memory_swap = host.get("MemorySwap")
    ulimits = parse_ulimits(host.get("Ulimits"))
    security_options = parse_security_options(host.get("SecurityOpt"))
    if (
        network_mode not in {"host", "bridge"}
        or runtime != "runc"
        or working_dir != "/workspace"
        or not isinstance(user, str)
        or (user and USER_RE.fullmatch(user) is None)
        or ipc_mode not in {"host", "private"}
        or isinstance(shm_size, bool)
        or not isinstance(shm_size, int)
        or shm_size <= 0
        or isinstance(memory, bool)
        or not isinstance(memory, int)
        or memory < 0
        or isinstance(memory_swap, bool)
        or not isinstance(memory_swap, int)
        or memory_swap < -1
    ):
        raise ResumeError("source runtime/resource contract has drifted")
    # The command's original preflight report is part of the immutable plan.
    # Never let the entrypoint replace it while proving that a resumed
    # container still sees the expected machine.  Publish that fresh evidence
    # beside the run state instead, under a path unique to this container.
    preflight_report = os.fspath(
        PurePosixPath(command_preflight).with_name(
            f"resume-{target_container}.json"
        )
    )
    if (
        not PurePosixPath(preflight_report).is_absolute()
        or ".." in PurePosixPath(preflight_report).parts
    ):
        raise ResumeError("fresh preflight report path is not canonical")

    environment = tuple(
        f"DS4RT_QUANT_PREFLIGHT_REPORT={preflight_report}"
        if record.startswith("DS4RT_QUANT_PREFLIGHT_REPORT=")
        else record
        for record in environment
    )
    if not any(
        record.startswith("DS4RT_QUANT_PREFLIGHT_REPORT=")
        for record in environment
    ):
        environment = (
            *environment,
            f"DS4RT_QUANT_PREFLIGHT_REPORT={preflight_report}",
        )
    image_id = parent_image_id
    resume_arguments = ("--resume",)
    if upgrade_image_id is not None:
        if (
            IMAGE_ID_RE.fullmatch(upgrade_image_id) is None
            or upgrade_image_id == parent_image_id
        ):
            raise ResumeError(
                "--upgrade-image-id must be a different content-addressed image"
            )
        image_id = upgrade_image_id
        environment = tuple(
            record
            for record in environment
            if record.split("=", 1)[0] not in IMAGE_OWNED_UPGRADE_ENV
        )
        environment = tuple(
            f"DS4RT_QUANT_IMAGE_DIGEST={image_id}"
            if record.startswith("DS4RT_QUANT_IMAGE_DIGEST=")
            else record
            for record in environment
        )
        if not any(
            record.startswith("DS4RT_QUANT_REQUIRE_IMAGE_DIGEST=")
            for record in environment
        ):
            environment = (
                *environment,
                "DS4RT_QUANT_REQUIRE_IMAGE_DIGEST=1",
            )
        command_values = list(command)
        preflight_index = command_values.index("--preflight-report") + 1
        command_values[preflight_index] = preflight_report
        command = tuple(command_values)
        resume_arguments = ("--execution-upgrade", "--resume")

    return ResumeSpec(
        source_container=source_container,
        target_container=target_container,
        image_id=image_id,
        command=command,
        resume_arguments=resume_arguments,
        environment=environment,
        binds=binds,
        network_mode=network_mode,
        runtime=runtime,
        working_dir=working_dir,
        user=user,
        ipc_mode=ipc_mode,
        shm_size=shm_size,
        memory=memory,
        memory_swap=memory_swap,
        ulimits=ulimits,
        security_options=security_options,
        plan_sha256=expected_plan_sha256,
        expected_gpus=expected_gpus,
        gpu_device_ids=gpu_device_ids,
        preflight_report=preflight_report,
    )


def verify_local_image(spec: ResumeSpec, *, runner: Run = subprocess.run) -> None:
    result = _run(
        ["docker", "image", "inspect", spec.image_id], runner=runner, check=True
    )
    try:
        values = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ResumeError("Docker returned invalid image metadata") from exc
    if (
        not isinstance(values, list)
        or len(values) != 1
        or not isinstance(values[0], dict)
        or values[0].get("Id") != spec.image_id
    ):
        raise ResumeError("local image identity differs from the stopped container")


def write_environment_file(records: Sequence[str]) -> Path:
    descriptor, raw_path = tempfile.mkstemp(prefix="ds4rt-quant-resume-", suffix=".env")
    path = Path(raw_path)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(record)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if stat.S_IMODE(path.stat().st_mode) != 0o600:
            raise ResumeError("temporary environment file is not private")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def docker_run_command(spec: ResumeSpec, environment_file: Path) -> list[str]:
    if spec.gpu_device_ids is None:
        gpu_selector = "all"
    else:
        gpu_selector = "device=" + ",".join(spec.gpu_device_ids)
        # Docker's --gpus parser treats an unquoted comma as a separator
        # between capability fields.  Shell examples quote the complete
        # multi-device selector; subprocess needs those quotes literally.
        if len(spec.gpu_device_ids) > 1:
            gpu_selector = f'"{gpu_selector}"'
    command = [
        "docker",
        "run",
        "--detach",
        "--name",
        spec.target_container,
        "--network",
        spec.network_mode,
        "--runtime",
        spec.runtime,
        "--gpus",
        gpu_selector,
        "--workdir",
        spec.working_dir,
        "--ipc",
        spec.ipc_mode,
        "--shm-size",
        str(spec.shm_size),
        "--env-file",
        os.fspath(environment_file),
    ]
    if spec.user:
        command.extend(["--user", spec.user])
    if spec.memory:
        command.extend(["--memory", f"{spec.memory}b"])
        command.extend(["--memory-swap", f"{spec.memory_swap}b"])
    for name, soft, hard in spec.ulimits:
        command.extend(["--ulimit", f"{name}={soft}:{hard}"])
    for option in spec.security_options:
        command.extend(["--security-opt", option])
    for bind in spec.binds:
        command.extend(["--volume", bind.raw])
    command.extend([spec.image_id, *spec.command, *spec.resume_arguments])
    return command


def read_fresh_preflight(
    container: str,
    path: str,
    *,
    runner: Run = subprocess.run,
) -> dict[str, Any] | None:
    probe = _run(
        ["docker", "exec", container, "test", "-s", path],
        runner=runner,
        check=False,
    )
    if probe.returncode != 0:
        return None
    program = (
        "import json,sys; p=json.load(open(sys.argv[1], encoding='utf-8')); "
        "print(json.dumps({'status':p.get('status'),'role':p.get('role'),"
        "'generated_at':p.get('generated_at'),'image_digest':p.get('image_digest'),"
        "'gpus':[{'index':g.get('index'),"
        "'uuid':g.get('uuid')} for g in p.get('gpus',[])]},sort_keys=True))"
    )
    result = _run(
        ["docker", "exec", container, "python", "-c", program, path],
        runner=runner,
        check=True,
    )
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ResumeError("fresh container preflight returned invalid JSON") from exc
    if not isinstance(report, dict):
        raise ResumeError("fresh container preflight returned a non-object")
    return report


def validate_started_container(spec: ResumeSpec, metadata: dict[str, Any]) -> None:
    config = metadata.get("Config")
    host = metadata.get("HostConfig")
    if not isinstance(config, dict) or not isinstance(host, dict):
        raise ResumeError("started resume container metadata is incomplete")
    _, expected_environment = parse_environment(list(spec.environment))
    _, actual_environment = parse_environment(config.get("Env"))
    environment_matches = (
        all(
            actual_environment.get(key) == value
            for key, value in expected_environment.items()
        )
        and set(actual_environment).difference(expected_environment)
        <= IMAGE_OWNED_UPGRADE_ENV
    )
    expected_binds = [bind.raw for bind in spec.binds]
    actual_binds = host.get("Binds")
    if (
        metadata.get("Image") != spec.image_id
        or config.get("Entrypoint") != ENTRYPOINT
        or config.get("Cmd") != [*spec.command, *spec.resume_arguments]
        or not environment_matches
        or config.get("WorkingDir") != spec.working_dir
        or config.get("User") != spec.user
        # Docker may canonicalize bind order when it materializes a container.
        # Bind order has no execution semantics; the exact source, destination,
        # and access-mode records still must match one-for-one.
        or not isinstance(actual_binds, list)
        or sorted(actual_binds) != sorted(expected_binds)
        or host.get("NetworkMode") != spec.network_mode
        or host.get("IpcMode") != spec.ipc_mode
        or host.get("Runtime") != spec.runtime
        or host.get("DeviceRequests") != docker_gpu_request(spec.gpu_device_ids)
        or host.get("ShmSize") != spec.shm_size
        or host.get("Memory") != spec.memory
        or host.get("MemorySwap") != spec.memory_swap
        or parse_ulimits(host.get("Ulimits")) != spec.ulimits
        or parse_security_options(host.get("SecurityOpt"))
        != spec.security_options
    ):
        raise ResumeError(
            "started resume container differs from the reconstructed exact launch"
        )


def wait_for_fresh_preflight(
    spec: ResumeSpec,
    *,
    timeout_seconds: float,
    runner: Run = subprocess.run,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    deadline = monotonic() + timeout_seconds
    while monotonic() < deadline:
        metadata = inspect_container(spec.target_container, runner=runner)
        validate_started_container(spec, metadata)
        state = metadata.get("State")
        if not isinstance(state, dict) or not state.get("Running"):
            status = state.get("Status") if isinstance(state, dict) else "unknown"
            exit_code = state.get("ExitCode") if isinstance(state, dict) else "unknown"
            raise ResumeError(
                "resume container exited before fresh preflight qualified "
                f"(status={status}, exit_code={exit_code}); inspect its logs"
            )
        report = read_fresh_preflight(
            spec.target_container, spec.preflight_report, runner=runner
        )
        if report is not None:
            generated_at = report.get("generated_at")
            started_at = state.get("StartedAt")
            try:
                generated_time = datetime.fromisoformat(generated_at)
                started_time = datetime.fromisoformat(
                    started_at.replace("Z", "+00:00")
                )
            except (AttributeError, TypeError, ValueError) as exc:
                raise ResumeError(
                    "fresh in-container preflight has invalid time provenance"
                ) from exc
            if generated_time.utcoffset() is None or started_time.utcoffset() is None:
                raise ResumeError(
                    "fresh in-container preflight timestamps are not timezone-aware"
                )
            if generated_time < started_time:
                # A report baked into an image or left by an abnormal runtime
                # must never satisfy this launch. Give the entrypoint time to
                # replace it with evidence generated after container start.
                sleep(0.5)
                continue
            actual_gpus = tuple(
                (gpu.get("index"), gpu.get("uuid"))
                for gpu in report.get("gpus", [])
                if isinstance(gpu, dict)
            )
            if (
                report.get("status") != "qualified"
                or report.get("role") != "coordinator"
                or report.get("image_digest") != spec.image_id
                or actual_gpus != spec.expected_gpus
            ):
                raise ResumeError(
                    "fresh in-container preflight differs from the exact resume contract"
                )
            return
        sleep(0.5)
    raise ResumeError(
        "resume container did not publish a fresh preflight before the timeout"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-container", required=True)
    parser.add_argument("--resume-container")
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument(
        "--upgrade-image-id",
        help=(
            "run the unchanged parent plan in a new content-addressed image under "
            "the quantizer's execution-upgrade contract"
        ),
    )
    parser.add_argument("--preflight-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="validate an already-started exact resume container without replacing it",
    )
    args = parser.parse_args()
    if CONTAINER_NAME_RE.fullmatch(args.source_container) is None:
        parser.error("--source-container must be one exact Docker container name")
    if args.resume_container is None:
        args.resume_container = f"{args.source_container}-resume"
    if CONTAINER_NAME_RE.fullmatch(args.resume_container) is None:
        parser.error("--resume-container must be one exact Docker container name")
    if args.resume_container == args.source_container:
        parser.error("source and resume container names must differ")
    if SHA256_RE.fullmatch(args.expected_plan_sha256) is None:
        parser.error("--expected-plan-sha256 must be exactly 64 lowercase hex digits")
    if args.upgrade_image_id is not None and IMAGE_ID_RE.fullmatch(
        args.upgrade_image_id
    ) is None:
        parser.error("--upgrade-image-id must be sha256: followed by 64 lowercase hex")
    if args.preflight_timeout_seconds <= 0:
        parser.error("--preflight-timeout-seconds must be positive")
    if args.dry_run and args.verify_existing:
        parser.error("--dry-run and --verify-existing are mutually exclusive")
    return args


def run(args: argparse.Namespace, *, runner: Run = subprocess.run) -> ResumeSpec:
    target_exists = container_exists(args.resume_container, runner=runner)
    verify_existing = bool(getattr(args, "verify_existing", False))
    if target_exists and not verify_existing:
        raise ResumeError(
            f"resume container already exists; refusing to replace it: {args.resume_container}"
        )
    if verify_existing and not target_exists:
        raise ResumeError(
            f"resume container does not exist for verification: {args.resume_container}"
        )
    metadata = inspect_container(args.source_container, runner=runner)
    live_gpus = live_host_gpus(runner=runner)
    spec = build_resume_spec(
        metadata,
        source_container=args.source_container,
        target_container=args.resume_container,
        expected_plan_sha256=args.expected_plan_sha256,
        live_gpus=live_gpus,
        upgrade_image_id=getattr(args, "upgrade_image_id", None),
    )
    verify_local_image(spec, runner=runner)
    if args.dry_run:
        return spec
    if verify_existing:
        wait_for_fresh_preflight(
            spec,
            timeout_seconds=args.preflight_timeout_seconds,
            runner=runner,
        )
        return spec

    environment_file = write_environment_file(spec.environment)
    try:
        _run(docker_run_command(spec, environment_file), runner=runner, check=True)
    finally:
        environment_file.unlink(missing_ok=True)
    wait_for_fresh_preflight(
        spec,
        timeout_seconds=args.preflight_timeout_seconds,
        runner=runner,
    )
    return spec


def main() -> int:
    args = parse_args()
    spec = run(args)
    if args.dry_run:
        outcome = "qualified dry run"
    elif args.verify_existing:
        outcome = "verified existing resume with fresh preflight"
    else:
        outcome = "started with fresh preflight"
    gpu_summary = ", ".join(
        f"cuda:{index}={uuid}" for index, uuid in spec.expected_gpus
    )
    print(
        f"{outcome}: {spec.target_container}; plan={spec.plan_sha256}; "
        f"image={spec.image_id}; {gpu_summary}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ResumeError as exc:
        print(f"resume-quantization-container: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
