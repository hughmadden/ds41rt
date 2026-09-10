from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WIP_PROCESS = ROOT / "scripts" / "wip-process.sh"
FINGERPRINT = "a" * 64


def wait_for(path: Path, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {path}")


def invoke(runtime: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(WIP_PROCESS), *args],
        env={**os.environ, "DS41RT_WIP_RUNTIME_ROOT": str(runtime)},
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_identity_is_bound_to_live_pid_and_start_time(tmp_path: Path) -> None:
    runtime = tmp_path / "run"
    process = subprocess.Popen(
        [str(WIP_PROCESS), "run", "test-process", "sleep", "60"],
        env={**os.environ, "DS41RT_WIP_RUNTIME_ROOT": str(runtime)},
    )
    try:
        wait_for(runtime / "test-process.pid")
        bound = invoke(runtime, "bind-identity", "test-process", FINGERPRINT)
        assert bound.returncode == 0, bound.stderr

        identity = invoke(runtime, "identity", "test-process")
        assert identity.returncode == 0, identity.stderr
        assert identity.stdout.strip() == FINGERPRINT

        identity_file = runtime / "test-process.identity"
        contents = identity_file.read_text(encoding="utf-8")
        identity_file.write_text(
            contents.replace("start_ticks=", "start_ticks=0#"), encoding="utf-8"
        )
        stale = invoke(runtime, "identity", "test-process")
        assert stale.returncode != 0
        assert stale.stdout == ""
    finally:
        os.kill(process.pid, signal.SIGTERM)
        process.wait(timeout=5)

    assert not (runtime / "test-process.pid").exists()
    assert not (runtime / "test-process.identity").exists()


def test_bind_identity_rejects_non_sha256_value(tmp_path: Path) -> None:
    result = invoke(tmp_path / "run", "bind-identity", "test-process", "not-a-hash")

    assert result.returncode == 2
    assert "invalid WIP process fingerprint" in result.stderr


def test_wip_launcher_has_separate_expert_and_deployment_identities() -> None:
    launcher = (ROOT / "scripts" / "run-wip.sh").read_text(encoding="utf-8")

    assert "wip-expert-runtime-identity.py" in launcher
    assert "ds41rt-wip-deployment-v2" in launcher
    assert 'DS41RT_RELEASE_CONFIG_SHA256="$expert_runtime_fingerprint"' in launcher
    assert "bind-identity" in launcher
    assert "'$expert_process' '$expert_runtime_fingerprint'" in launcher
    assert "reusing four fingerprint-matched resident WIP Spark experts" in launcher


def test_wip_launcher_rejects_a_stale_profile_resolver() -> None:
    launcher = (ROOT / "scripts" / "run-wip.sh").read_text(encoding="utf-8")

    resolver = launcher.index("resolve_serve_profile.py")
    policy_check = launcher.index("resolved_dspark_draft_policy", resolver)
    coordinator_dispatch = launcher.index("== starting coordinator process")
    assert resolver < policy_check < coordinator_dispatch
    assert "rebuild the slot" in launcher[policy_check:coordinator_dispatch]


def test_wip_launcher_retries_spark_headroom_after_process_teardown() -> None:
    launcher = (ROOT / "scripts" / "run-wip.sh").read_text(encoding="utf-8")

    assert "A stopped CUDA process can disappear before Linux has finished" in launcher
    assert launcher.count("for _ in $(seq 1 50); do") == 2
    assert launcher.count("sleep 0.1") >= 2


def test_wip_launcher_scopes_native_activation_capture_to_coordinator() -> None:
    launcher = (ROOT / "scripts" / "run-wip.sh").read_text(encoding="utf-8")

    env_start = launcher.index('env_file="$state_dir/coordinator.env"')
    coordinator_dispatch = launcher.index("== starting coordinator process")
    capture_dir = launcher.index(
        "DS41RT_REAL_FULL_DIAGNOSTIC_LAYER_DUMP_DIR", env_start
    )
    packed_hidden_exchange = launcher.index(
        "DS41RT_REAL_FULL_B12X_PACKED_HIDDEN_EXCHANGE", env_start
    )
    capture_layers = launcher.index(
        "DS41RT_REAL_FULL_DIAGNOSTIC_LAYER_DUMP_LAYER", env_start
    )
    assert env_start < capture_dir < coordinator_dispatch
    assert env_start < packed_hidden_exchange < coordinator_dispatch
    assert env_start < capture_layers < coordinator_dispatch


def test_wip_launcher_scopes_exact_route_capture_to_coordinator() -> None:
    launcher = (ROOT / "scripts" / "run-wip.sh").read_text(encoding="utf-8")

    env_start = launcher.index('env_file="$state_dir/coordinator.env"')
    coordinator_dispatch = launcher.index("== starting coordinator process")
    route_stats = launcher.index(
        "DS41RT_PROTOCOL_V2_EXPERT_QUEUE_STATS", env_start
    )
    row_routes = launcher.index(
        "DS41RT_PROTOCOL_V2_EXPERT_QUEUE_ROW_ROUTES", env_start
    )
    assert env_start < route_stats < coordinator_dispatch
    assert env_start < row_routes < coordinator_dispatch


def test_wip_launcher_scopes_performance_tracing_to_coordinator() -> None:
    launcher = (ROOT / "scripts" / "run-wip.sh").read_text(encoding="utf-8")

    env_start = launcher.index('env_file="$state_dir/coordinator.env"')
    coordinator_dispatch = launcher.index("== starting coordinator process")
    for variable in (
        "DS41RT_REAL_FULL_DSPARK_TRACE",
        "DS41RT_REAL_FULL_REQUEST_TIMING",
        "DS41RT_REAL_FULL_REQUEST_PREFILL_CHUNK_TOKENS",
        "DS41RT_REAL_FULL_SCHEDULER_TIMING",
        "DS41RT_REAL_FULL_SCHEDULER_SUMMARY_TIMING",
        "DS41RT_REAL_FULL_ADMISSION_STAGE_TIMING",
        "DS41RT_REAL_FULL_ATTENTION_CUDA_TIMING",
        "DS41RT_REAL_FULL_ROLLING_SPARSE_PACKS",
        "DS41RT_REAL_FULL_SPARSE_TCP_STAGE_TIMING",
        "DS41RT_REAL_FULL_NVFP4_ROUTE_TIMING",
        "DS41RT_REAL_FULL_NVFP4_ROUTE_CUDA_EVENT_TIMING",
        "DS41RT_REAL_FULL_PROTOCOL_V2_EXECUTOR_TIMING",
        "DS41RT_PROTOCOL_V2_TCP_TIMING",
    ):
        offset = launcher.index(variable, env_start)
        assert env_start < offset < coordinator_dispatch


def test_wip_launcher_shadow_trace_overrides_active_dspark() -> None:
    launcher = (ROOT / "scripts" / "run-wip.sh").read_text(encoding="utf-8")

    env_start = launcher.index('env_file="$state_dir/coordinator.env"')
    coordinator_dispatch = launcher.index("== starting coordinator process")
    shadow_option = launcher.index("--dspark-shadow-trace")
    shadow_override = launcher.index(
        'echo "DS41RT_REAL_FULL_DSPARK_SHADOW=1"', env_start
    )
    active_override = launcher.index(
        'echo "DS41RT_REAL_FULL_DSPARK=0"', env_start
    )
    trace_override = launcher.index(
        'echo "DS41RT_REAL_FULL_DSPARK_TRACE=1"', env_start
    )
    assert shadow_option < env_start
    assert env_start < active_override < coordinator_dispatch
    assert env_start < shadow_override < coordinator_dispatch
    assert env_start < trace_override < coordinator_dispatch


def test_wip_launcher_can_disable_dspark_without_editing_profile() -> None:
    launcher = (ROOT / "scripts" / "run-wip.sh").read_text(encoding="utf-8")

    env_start = launcher.index('env_file="$state_dir/coordinator.env"')
    coordinator_dispatch = launcher.index("== starting coordinator process")
    off_option = launcher.index("--dspark-off")
    off_branch = launcher.index("if ((dspark_off)); then", env_start)
    active_override = launcher.index(
        'echo "DS41RT_REAL_FULL_DSPARK=0"', off_branch
    )
    shadow_override = launcher.index(
        'echo "DS41RT_REAL_FULL_DSPARK_SHADOW=0"', off_branch
    )
    assert off_option < env_start
    assert env_start < off_branch < coordinator_dispatch
    assert off_branch < active_override < coordinator_dispatch
    assert off_branch < shadow_override < coordinator_dispatch
    assert "--dspark-off and --dspark-shadow-trace are mutually exclusive" in launcher


def test_wip_launcher_can_narrow_coordinator_execution_lanes() -> None:
    launcher = (ROOT / "scripts" / "run-wip.sh").read_text(encoding="utf-8")

    env_start = launcher.index('env_file="$state_dir/coordinator.env"')
    coordinator_dispatch = launcher.index("== starting coordinator process")
    lanes_option = launcher.index("--execution-lanes")
    lanes_override = launcher.index(
        'echo "DS41RT_REAL_FULL_MAX_EXECUTION_LANES=$execution_lanes_override"',
        env_start,
    )
    assert lanes_option < env_start
    assert env_start < lanes_override < coordinator_dispatch
    assert "--execution-lanes must be an integer in 1..8" in launcher


def test_wip_builder_streams_every_local_heredoc_into_docker() -> None:
    builder = (ROOT / "wip.sh").read_text(encoding="utf-8")
    local_heredocs = re.findall(
        r'^\s*docker exec (?P<options>.*?)"\$coordinator_container" bash -s .*<<',
        builder,
        flags=re.MULTILINE,
    )

    assert len(local_heredocs) == 4
    assert all("-i" in options.split() for options in local_heredocs)
