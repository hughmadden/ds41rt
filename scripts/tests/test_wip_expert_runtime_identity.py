from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "wip-expert-runtime-identity.py"
SPEC = importlib.util.spec_from_file_location("wip_expert_runtime_identity", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
IDENTITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IDENTITY)


def payload(profile_environment: dict[str, str], ambient: dict[str, str]):
    return IDENTITY.build_payload(
        {"environment": profile_environment}, "a" * 64, {"port": "50061"}, ambient
    )


def test_coordinator_and_shell_environment_do_not_affect_expert_identity() -> None:
    base = payload({}, {})
    candidate = payload(
        {"DS41RT_COORDINATOR_W8A16_O_PROJ": "1"},
        {"OLDPWD": "/somewhere/else", "SSH_CONNECTION": "changed"},
    )

    assert IDENTITY.canonical_bytes(base) == IDENTITY.canonical_bytes(candidate)


def test_removed_transformer_tp_environment_does_not_affect_expert_identity() -> None:
    base = payload({}, {})
    candidate = payload(
        {},
        {
            "DS41RT_SPARK_TRANSFORMER_TP_RANGE": "0:61",
            "DS41RT_SPARK_LAYER_BLOCK_RANGE": "0:18",
        },
    )

    assert IDENTITY.canonical_bytes(base) == IDENTITY.canonical_bytes(candidate)


def test_expert_environment_and_profile_precedence_affect_identity() -> None:
    base = payload({}, {"DS41RT_EXPERT_INTERMEDIATE_SHARDS": "2"})
    overridden = payload(
        {"DS41RT_EXPERT_INTERMEDIATE_SHARDS": "4"},
        {"DS41RT_EXPERT_INTERMEDIATE_SHARDS": "2"},
    )

    assert base["environment"]["DS41RT_EXPERT_INTERMEDIATE_SHARDS"] == "2"
    assert overridden["environment"]["DS41RT_EXPERT_INTERMEDIATE_SHARDS"] == "4"
    assert IDENTITY.canonical_bytes(base) != IDENTITY.canonical_bytes(overridden)


def test_model_and_expert_kernel_controls_are_selected() -> None:
    selected = payload(
        {
            "DS41RT_MODEL_ID": "model-a",
            "DS41RT_B12X_SPARK_W4A16_SMALL_M_MODE": "wide",
            "DS41RT_REAL_FULL_NVFP4_ROUTE_CUDA_GRAPHS": "1",
            "DS41RT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES": "4",
        },
        {},
    )["environment"]

    assert set(selected) == {
        "DS41RT_MODEL_ID",
        "DS41RT_B12X_SPARK_W4A16_SMALL_M_MODE",
        "DS41RT_REAL_FULL_NVFP4_ROUTE_CUDA_GRAPHS",
        "DS41RT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES",
    }


def test_duplicate_settings_are_rejected() -> None:
    try:
        IDENTITY.parse_settings(["port=1", "port=2"])
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate settings were accepted")
