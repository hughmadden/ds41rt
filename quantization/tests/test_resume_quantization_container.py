from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "resume-quantization-container.py"
SPEC = importlib.util.spec_from_file_location("resume_quantization_container", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


IMAGE = "sha256:" + "a" * 64
UPGRADE_IMAGE = "sha256:" + "b" * 64
GPUS = ((0, "GPU-zero"), (1, "GPU-one"))
SINGLE_GPU = ((0, "GPU-one"),)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(
    tmp_path: Path,
    *,
    gpus: tuple[tuple[int, str], ...] = GPUS,
    gpu_device_ids: tuple[str, ...] | None = None,
) -> tuple[dict, str]:
    artifacts = tmp_path / "artifacts"
    runtime = tmp_path / "run"
    preflight_root = tmp_path / "preflight"
    workers = tmp_path / "workers"
    model = tmp_path / "model"
    calibration = tmp_path / "calibration"
    for path in (artifacts, runtime, preflight_root, workers, model, calibration):
        path.mkdir()
    (runtime / "offload").mkdir()
    (runtime / "mtp-prefix").mkdir()

    original_preflight = {
        "status": "qualified",
        "role": "coordinator",
        "image_digest": IMAGE,
        "gpus": [
            {"index": index, "uuid": uuid} for index, uuid in gpus
        ],
    }
    preflight_path = preflight_root / "preflight.json"
    write_json(preflight_path, original_preflight)

    output = "/artifacts/flash-k2"
    run_state = artifacts / ".flash-k2.ds4rt-run"
    run_state.mkdir()
    plan = {
        "schema": MODULE.PLAN_SCHEMA,
        "output": output,
        "run_state_dir": "/artifacts/.flash-k2.ds4rt-run",
        "offload_dir": "/run/offload",
        "mtp_prefix_store": "/run/mtp-prefix",
        "preflight": {
            "path": "/preflight/preflight.json",
            "sha256": sha256_file(preflight_path),
            "image_digest": IMAGE,
            "gpus": [
                {"index": index, "uuid": uuid} for index, uuid in gpus
            ],
        },
    }
    plan["plan_sha256"] = hashlib.sha256(MODULE.canonical_json(plan)).hexdigest()
    write_json(run_state / MODULE.PLAN_FILENAME, plan)

    command = [
        "python",
        MODULE.QUANTIZER,
        "--snapshot",
        "/model/snapshot",
        "--calibration-jsonl",
        "/calibration/calibration.jsonl",
        "--preflight-report",
        "/preflight/preflight.json",
        "--output",
        output,
        "--offload-dir",
        "/run/offload",
        "--mtp-prefix-store",
        "/run/mtp-prefix",
    ]
    metadata = {
        "Image": IMAGE,
        "Config": {
            "Entrypoint": MODULE.ENTRYPOINT,
            "Cmd": command,
            "Env": [
                f"DS4RT_QUANT_IMAGE_DIGEST={IMAGE}",
                "DS4RT_QUANT_REQUIRE_IMAGE_DIGEST=1",
                "DS4RT_QUANT_ROLE=coordinator",
                "DS4RT_QUANT_TARGET_PLATFORM=linux/amd64",
                "DS4RT_QUANT_CUDA_ARCH=120",
                f"DS4RT_QUANT_MIN_GPUS={len(gpus)}",
                "DS4RT_EXL3_WORKER_TOKEN=do-not-render-this-secret",
            ],
            "User": "",
            "WorkingDir": "/workspace",
        },
        "HostConfig": {
            "NetworkMode": "host",
            "IpcMode": "host",
            "DeviceRequests": MODULE.docker_gpu_request(gpu_device_ids),
            "Privileged": False,
            "ReadonlyRootfs": False,
            "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
            "AutoRemove": False,
            "Runtime": "runc",
            "ShmSize": 64 * 1024 * 1024,
            "Memory": 0,
            "MemorySwap": 0,
            "Ulimits": None,
            "SecurityOpt": None,
            "Binds": [
                f"{workers}:/workers:ro",
                f"{artifacts}:/artifacts",
                f"{runtime}:/run",
                f"{model}:/model:ro",
                f"{calibration}:/calibration:ro",
                f"{preflight_root}:/preflight:ro",
            ],
        },
        "State": {
            "Running": False,
            "Paused": False,
            "Restarting": False,
            "Status": "exited",
        },
    }
    return metadata, plan["plan_sha256"]


def build(tmp_path: Path, **overrides: object):
    metadata, plan_sha256 = fixture(tmp_path)
    arguments = {
        "source_container": "source",
        "target_container": "source-resume",
        "expected_plan_sha256": plan_sha256,
        "live_gpus": GPUS,
    }
    arguments.update(overrides)
    return MODULE.build_resume_spec(metadata, **arguments)


def test_exact_resume_clones_command_and_hides_environment(tmp_path: Path) -> None:
    spec = build(tmp_path)
    assert spec.command[:2] == ("python", MODULE.QUANTIZER)
    assert "--resume" not in spec.command
    assert MODULE.option_value(spec.command, "--preflight-report") != (
        spec.preflight_report
    )
    assert (
        f"DS4RT_QUANT_PREFLIGHT_REPORT={spec.preflight_report}" in spec.environment
    )
    assert spec.preflight_report == "/preflight/resume-source-resume.json"
    environment_file = tmp_path / "private.env"
    command = MODULE.docker_run_command(spec, environment_file)
    assert command[-1] == "--resume"
    assert command.count("--resume") == 1
    assert command.count("--name") == 1
    assert "do-not-render-this-secret" not in " ".join(command)
    assert spec.plan_sha256


def test_local_resume_preserves_private_ipc(tmp_path: Path) -> None:
    metadata, plan_sha256 = fixture(tmp_path)
    metadata["HostConfig"]["IpcMode"] = "private"

    spec = MODULE.build_resume_spec(
        metadata,
        source_container="source",
        target_container="source-resume",
        expected_plan_sha256=plan_sha256,
        live_gpus=GPUS,
    )

    assert spec.ipc_mode == "private"


def test_exact_resume_preserves_nonroot_memory_and_runtime_limits(
    tmp_path: Path,
) -> None:
    metadata, plan_sha256 = fixture(tmp_path)
    metadata["Config"]["User"] = "1000:1000"
    metadata["HostConfig"].update(
        {
            "NetworkMode": "bridge",
            "Memory": 175 * 1024**3,
            "MemorySwap": 175 * 1024**3,
            "Ulimits": [{"Name": "memlock", "Soft": -1, "Hard": -1}],
            "SecurityOpt": ["label=disable"],
        }
    )
    spec = MODULE.build_resume_spec(
        metadata,
        source_container="source",
        target_container="source-resume",
        expected_plan_sha256=plan_sha256,
        live_gpus=GPUS,
    )
    command = MODULE.docker_run_command(spec, tmp_path / "private.env")

    assert command[command.index("--user") + 1] == "1000:1000"
    assert command[command.index("--memory") + 1] == f"{175 * 1024**3}b"
    assert command[command.index("--memory-swap") + 1] == f"{175 * 1024**3}b"
    assert command[command.index("--ulimit") + 1] == "memlock=-1:-1"
    assert command[command.index("--security-opt") + 1] == "label=disable"

    started = copy.deepcopy(metadata)
    started["Config"]["Cmd"].append("--resume")
    started["Config"]["Env"] = list(spec.environment)
    MODULE.validate_started_container(spec, started)


def test_exact_single_gpu_uuid_resume_preserves_physical_selection(
    tmp_path: Path,
) -> None:
    metadata, plan_sha256 = fixture(
        tmp_path, gpus=SINGLE_GPU, gpu_device_ids=("GPU-one",)
    )
    spec = MODULE.build_resume_spec(
        metadata,
        source_container="source",
        target_container="source-resume",
        expected_plan_sha256=plan_sha256,
        live_gpus=GPUS,
    )
    command = MODULE.docker_run_command(spec, tmp_path / "private.env")

    assert spec.expected_gpus == SINGLE_GPU
    assert spec.gpu_device_ids == ("GPU-one",)
    assert command[command.index("--gpus") + 1] == "device=GPU-one"

    started = copy.deepcopy(metadata)
    started["Config"]["Cmd"].append("--resume")
    started["Config"]["Env"] = list(spec.environment)
    MODULE.validate_started_container(spec, started)


def test_exact_multi_gpu_uuid_resume_quotes_one_docker_device_selector(
    tmp_path: Path,
) -> None:
    metadata, plan_sha256 = fixture(
        tmp_path,
        gpu_device_ids=("GPU-zero", "GPU-one"),
    )
    spec = MODULE.build_resume_spec(
        metadata,
        source_container="source",
        target_container="source-resume",
        expected_plan_sha256=plan_sha256,
        live_gpus=GPUS,
    )
    command = MODULE.docker_run_command(spec, tmp_path / "private.env")

    assert spec.gpu_device_ids == ("GPU-zero", "GPU-one")
    assert command[command.index("--gpus") + 1] == (
        '"device=GPU-zero,GPU-one"'
    )


def test_single_gpu_uuid_must_match_plan(tmp_path: Path) -> None:
    metadata, plan_sha256 = fixture(
        tmp_path, gpus=SINGLE_GPU, gpu_device_ids=("GPU-zero",)
    )
    with pytest.raises(MODULE.ResumeError, match="exact-GPU request differs"):
        MODULE.build_resume_spec(
            metadata,
            source_container="source",
            target_container="source-resume",
            expected_plan_sha256=plan_sha256,
            live_gpus=GPUS,
        )


def test_execution_upgrade_uses_new_image_and_fresh_preflight(
    tmp_path: Path,
) -> None:
    spec = build(tmp_path, upgrade_image_id=UPGRADE_IMAGE)
    environment_file = tmp_path / "private.env"
    command = MODULE.docker_run_command(spec, environment_file)

    assert spec.image_id == UPGRADE_IMAGE
    assert spec.resume_arguments == ("--execution-upgrade", "--resume")
    assert MODULE.option_value(spec.command, "--preflight-report") == (
        spec.preflight_report
    )
    assert (
        f"DS4RT_QUANT_PREFLIGHT_REPORT={spec.preflight_report}" in spec.environment
    )
    assert f"DS4RT_QUANT_IMAGE_DIGEST={UPGRADE_IMAGE}" in spec.environment
    assert f"DS4RT_QUANT_IMAGE_DIGEST={IMAGE}" not in spec.environment
    assert command[-2:] == ["--execution-upgrade", "--resume"]


@pytest.mark.parametrize(
    "source_suffix",
    [("--resume",), ("--execution-upgrade", "--resume")],
)
def test_resume_chain_normalizes_existing_resume_suffix(
    tmp_path: Path,
    source_suffix: tuple[str, ...],
) -> None:
    metadata, plan_sha256 = fixture(tmp_path)
    metadata["Config"]["Cmd"].extend(source_suffix)

    spec = MODULE.build_resume_spec(
        metadata,
        source_container="source",
        target_container="source-resume",
        expected_plan_sha256=plan_sha256,
        live_gpus=GPUS,
        upgrade_image_id=UPGRADE_IMAGE,
    )

    assert "--resume" not in spec.command
    assert "--execution-upgrade" not in spec.command
    assert spec.resume_arguments == ("--execution-upgrade", "--resume")


def test_upgrade_normalizes_read_only_legacy_quantizer_path(tmp_path: Path) -> None:
    metadata, plan_sha256 = fixture(tmp_path)
    legacy_root = tmp_path / "legacy-source"
    legacy_quantizer = legacy_root / "quantization" / "quantize_flash_gptqmodel.py"
    legacy_quantizer.parent.mkdir(parents=True)
    legacy_quantizer.write_text("# immutable legacy source\n", encoding="utf-8")
    metadata["Config"]["Cmd"][1] = "/legacy/quantization/quantize_flash_gptqmodel.py"
    metadata["HostConfig"]["Binds"].append(f"{legacy_root}:/legacy:ro")

    spec = MODULE.build_resume_spec(
        metadata,
        source_container="source",
        target_container="source-resume",
        expected_plan_sha256=plan_sha256,
        live_gpus=GPUS,
        upgrade_image_id=UPGRADE_IMAGE,
    )

    assert spec.command[1] == MODULE.QUANTIZER
    assert spec.resume_arguments == ("--execution-upgrade", "--resume")


def test_upgrade_inherits_dependency_contract_from_target_image(
    tmp_path: Path,
) -> None:
    metadata, plan_sha256 = fixture(tmp_path)
    metadata["Config"]["Env"].extend(
        [
            "DS4RT_QUANT_REQUIREMENTS_SHA256=" + "c" * 64,
            "DS4RT_QUANT_BUILD_REQUIREMENTS_SHA256=" + "d" * 64,
        ]
    )

    spec = MODULE.build_resume_spec(
        metadata,
        source_container="source",
        target_container="source-resume",
        expected_plan_sha256=plan_sha256,
        live_gpus=GPUS,
        upgrade_image_id=UPGRADE_IMAGE,
    )

    assert not any(
        record.startswith("DS4RT_QUANT_REQUIREMENTS_SHA256=")
        or record.startswith("DS4RT_QUANT_BUILD_REQUIREMENTS_SHA256=")
        for record in spec.environment
    )

    started = copy.deepcopy(metadata)
    started["Image"] = UPGRADE_IMAGE
    started["Config"]["Cmd"] = [*spec.command, *spec.resume_arguments]
    started["Config"]["Env"] = [
        *spec.environment,
        "DS4RT_QUANT_REQUIREMENTS_SHA256=" + "e" * 64,
        "DS4RT_QUANT_BUILD_REQUIREMENTS_SHA256=" + "f" * 64,
    ]
    MODULE.validate_started_container(spec, started)

    started["Config"]["Env"].append("UNEXPECTED_EXECUTION_OVERRIDE=1")
    with pytest.raises(MODULE.ResumeError, match="reconstructed exact launch"):
        MODULE.validate_started_container(spec, started)


def test_plain_resume_rejects_legacy_quantizer_path(tmp_path: Path) -> None:
    metadata, plan_sha256 = fixture(tmp_path)
    metadata["Config"]["Cmd"][1] = "/legacy/quantization/quantize_flash_gptqmodel.py"

    with pytest.raises(MODULE.ResumeError, match="unfinished production"):
        MODULE.build_resume_spec(
            metadata,
            source_container="source",
            target_container="source-resume",
            expected_plan_sha256=plan_sha256,
            live_gpus=GPUS,
        )


def test_upgrade_hardens_legacy_missing_image_digest_requirement(
    tmp_path: Path,
) -> None:
    metadata, plan_sha256 = fixture(tmp_path)
    metadata["Config"]["Env"] = [
        record
        for record in metadata["Config"]["Env"]
        if not record.startswith("DS4RT_QUANT_REQUIRE_IMAGE_DIGEST=")
    ]

    spec = MODULE.build_resume_spec(
        metadata,
        source_container="source",
        target_container="source-resume",
        expected_plan_sha256=plan_sha256,
        live_gpus=GPUS,
        upgrade_image_id=UPGRADE_IMAGE,
    )

    assert "DS4RT_QUANT_REQUIRE_IMAGE_DIGEST=1" in spec.environment


def test_plain_resume_rejects_missing_image_digest_requirement(
    tmp_path: Path,
) -> None:
    metadata, plan_sha256 = fixture(tmp_path)
    metadata["Config"]["Env"] = [
        record
        for record in metadata["Config"]["Env"]
        if not record.startswith("DS4RT_QUANT_REQUIRE_IMAGE_DIGEST=")
    ]

    with pytest.raises(MODULE.ResumeError, match="qualified image/GPU role"):
        MODULE.build_resume_spec(
            metadata,
            source_container="source",
            target_container="source-resume",
            expected_plan_sha256=plan_sha256,
            live_gpus=GPUS,
        )


def test_private_environment_file_has_mode_0600(tmp_path: Path) -> None:
    spec = build(tmp_path)
    path = MODULE.write_environment_file(spec.environment)
    try:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert "do-not-render-this-secret" in path.read_text(encoding="utf-8")
    finally:
        path.unlink()


def test_saved_plan_mismatch_fails_before_launch(tmp_path: Path) -> None:
    with pytest.raises(MODULE.ResumeError, match="expected-plan-sha256"):
        build(tmp_path, expected_plan_sha256="b" * 64)


def test_live_gpu_identity_mismatch_fails_before_launch(tmp_path: Path) -> None:
    with pytest.raises(MODULE.ResumeError, match="live coordinator GPUs"):
        build(tmp_path, live_gpus=((0, "GPU-zero"),))


def test_output_materialization_blocks_raw_resume(tmp_path: Path) -> None:
    metadata, plan_sha256 = fixture(tmp_path)
    artifacts = Path(metadata["HostConfig"]["Binds"][1].split(":", 1)[0])
    (artifacts / "flash-k2").mkdir()
    with pytest.raises(MODULE.ResumeError, match="output already exists"):
        MODULE.build_resume_spec(
            metadata,
            source_container="source",
            target_container="source-resume",
            expected_plan_sha256=plan_sha256,
            live_gpus=GPUS,
        )


def test_secret_is_not_in_environment_validation_error() -> None:
    secret = "never-show-this-value"
    with pytest.raises(MODULE.ResumeError) as captured:
        MODULE.parse_environment(
            [f"DS4RT_EXL3_WORKER_TOKEN={secret}", "INVALID-NAME=value"]
        )
    assert secret not in str(captured.value)


def test_started_container_must_preserve_exact_environment_and_command(
    tmp_path: Path,
) -> None:
    metadata, plan_sha256 = fixture(tmp_path)
    spec = MODULE.build_resume_spec(
        metadata,
        source_container="source",
        target_container="source-resume",
        expected_plan_sha256=plan_sha256,
        live_gpus=GPUS,
    )
    started = copy.deepcopy(metadata)
    started["Config"]["Cmd"].append("--resume")
    started["Config"]["Env"] = list(spec.environment)
    MODULE.validate_started_container(spec, started)
    started["HostConfig"]["Binds"].reverse()
    MODULE.validate_started_container(spec, started)
    token_index = next(
        index
        for index, record in enumerate(started["Config"]["Env"])
        if record.startswith("DS4RT_EXL3_WORKER_TOKEN=")
    )
    started["Config"]["Env"][token_index] = (
        "DS4RT_EXL3_WORKER_TOKEN=changed-secret"
    )
    with pytest.raises(MODULE.ResumeError, match="reconstructed exact launch") as error:
        MODULE.validate_started_container(spec, started)
    assert "changed-secret" not in str(error.value)


def test_container_path_mapping_rejects_unbound_path(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    bind = MODULE.parse_bind(f"{root}:/bound:ro")
    assert MODULE.map_container_path("/bound/item", [bind]) == root / "item"
    with pytest.raises(MODULE.ResumeError, match="not backed"):
        MODULE.map_container_path("/elsewhere/item", [bind])


def test_existing_target_is_never_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MODULE, "container_exists", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        MODULE,
        "inspect_container",
        lambda *args, **kwargs: pytest.fail("source must not be inspected"),
    )
    args = argparse.Namespace(
        source_container="source",
        resume_container="source-resume",
        expected_plan_sha256="a" * 64,
        preflight_timeout_seconds=30.0,
        dry_run=True,
        verify_existing=False,
    )
    with pytest.raises(MODULE.ResumeError, match="refusing to replace"):
        MODULE.run(args)


def test_verify_existing_uses_the_exact_reconstructed_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, plan_sha256 = fixture(tmp_path)
    observed: list[MODULE.ResumeSpec] = []
    monkeypatch.setattr(MODULE, "container_exists", lambda *args, **kwargs: True)
    monkeypatch.setattr(MODULE, "inspect_container", lambda *args, **kwargs: metadata)
    monkeypatch.setattr(MODULE, "live_host_gpus", lambda *args, **kwargs: GPUS)
    monkeypatch.setattr(MODULE, "verify_local_image", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        MODULE,
        "wait_for_fresh_preflight",
        lambda spec, **kwargs: observed.append(spec),
    )
    args = argparse.Namespace(
        source_container="source",
        resume_container="source-resume",
        expected_plan_sha256=plan_sha256,
        preflight_timeout_seconds=30.0,
        dry_run=False,
        verify_existing=True,
    )

    spec = MODULE.run(args)

    assert observed == [spec]
    assert spec.source_container == "source"
    assert spec.target_container == "source-resume"
    assert spec.plan_sha256 == plan_sha256
