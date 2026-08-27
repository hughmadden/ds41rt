import json
from pathlib import Path

ROOT = Path(__file__).parents[2]
DOCKERFILE = (ROOT / "docker" / "Dockerfile.quantization").read_text(encoding="utf-8")
ENTRYPOINT = (ROOT / "docker" / "quantization-entrypoint.sh").read_text(
    encoding="utf-8"
)
BAKE = (ROOT / "docker" / "docker-bake.hcl").read_text(encoding="utf-8")
GPTQMODEL_LOCK = json.loads(
    (ROOT / "third_party" / "gptqmodel.lock.json").read_text(encoding="utf-8")
)


def test_container_uses_free_threaded_python_and_hashed_lock() -> None:
    assert "nvcr.io/nvidia/pytorch@sha256:" in DOCKERFILE
    assert "uv python install 3.14.6t" in DOCKERFILE
    assert "RUST_TOOLCHAIN=1.97.1" in DOCKERFILE
    assert "RUSTUP_INIT_AMD64_SHA256=" in DOCKERFILE
    assert "RUSTUP_INIT_ARM64_SHA256=" in DOCKERFILE
    assert "ENV PYTHON_GIL=0" in DOCKERFILE
    assert "ENV TORCH_CUDA_ARCH_LIST=12.0" in DOCKERFILE
    assert "--require-hashes" in DOCKERFILE
    assert (
        "--build-constraints /opt/ds4rt/quantization/build-requirements.lock"
        in DOCKERFILE
    )
    assert "quantization/requirements.amd64.lock" in DOCKERFILE
    assert "quantization/requirements.arm64.lock" in DOCKERFILE
    assert "normalize_nvidia_cusparselt.py --apply" in DOCKERFILE
    assert DOCKERFILE.count('uv pip check --python "${VIRTUAL_ENV}/bin/python"') == 2


def test_container_rejects_cross_architecture_build_contracts() -> None:
    assert '"amd64:coordinator:linux/amd64:120"' in DOCKERFILE
    assert '"arm64:expert:linux/arm64:121"' in DOCKERFILE
    assert "invalid native quantization target contract" in DOCKERFILE


def test_container_verifies_vendored_gptqmodel_before_import() -> None:
    verify = DOCKERFILE.index("verify-gptqmodel-source.py")
    install = DOCKERFILE.index("--editable /opt/ds4rt/third_party/gptqmodel")
    assert verify < install
    assert "DS4RT_GPTQMODEL_COMMIT" in DOCKERFILE


def test_container_contains_target_and_joint_mtp_launcher() -> None:
    assert "quantize_flash_gptqmodel.py" in DOCKERFILE
    assert "deepseek_v4_mtp_prefix_store.py" in DOCKERFILE
    assert "exl3_worker.py" in DOCKERFILE
    assert "validate_projection_checkpoint_block.py" in DOCKERFILE
    assert "validate_exl3_remote_real_family.py" in DOCKERFILE


def test_entrypoint_always_runs_preflight_before_command() -> None:
    preflight = ENTRYPOINT.index("quantization/preflight.py")
    execute = ENTRYPOINT.index('exec "$@"')
    assert preflight < execute
    assert '--output "${report_path}" >&2' in ENTRYPOINT


def test_bake_targets_pin_identical_source_and_dependency_locks() -> None:
    revision = GPTQMODEL_LOCK["revision"]
    amd64_requirements = (
        "0dd72f6171711269548da684cd1f4fda37d0da2948f371129f52fcae0bdac140"
    )
    arm64_requirements = (
        "196cfaf026ef3f7d9b81d65aaeb00a871ad69793381a40ce0d1b3decbcc31d7b"
    )
    build_requirements = (
        "9f21166fd088fd5eee2e9560c5d97b14201e0fde30d7ee27a43a56c24e104fd1"
    )
    assert BAKE.count(f'DS4RT_GPTQMODEL_COMMIT = "{revision}"') == 2
    assert BAKE.count('DS4RT_QUANT_MIN_GPUS = "2"') == 1
    assert BAKE.count('DS4RT_QUANT_MIN_GPUS = "1"') == 1
    assert BAKE.count(f'DS4RT_QUANT_REQUIREMENTS_SHA256 = "{amd64_requirements}"') == 1
    assert BAKE.count(f'DS4RT_QUANT_REQUIREMENTS_SHA256 = "{arm64_requirements}"') == 1
    assert (
        BAKE.count(f'DS4RT_QUANT_BUILD_REQUIREMENTS_SHA256 = "{build_requirements}"')
        == 2
    )
