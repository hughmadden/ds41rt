from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import validate_ds4_staged_snapshot as validator  # noqa: E402


def test_cli_binds_staged_identity_to_requested_model_and_revision(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    revision = "a" * 64
    monkeypatch.setattr(
        validator,
        "validate_staged_exl3_checkpoint",
        lambda checkpoint, **_kwargs: {
            "schema": "ds4rt-hf-staged-snapshot-v1",
            "model_id": "tpurtell/flash-k2",
            "revision": revision,
            "checkpoint": str(checkpoint),
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_ds4_staged_snapshot.py",
            "--checkpoint",
            str(tmp_path),
            "--model-id",
            "tpurtell/flash-k2",
            "--revision",
            revision,
        ],
    )

    assert validator.main() == 0
    assert json.loads(capsys.readouterr().out)["revision"] == revision


def test_cli_requires_explicit_development_override(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    revision = "a" * 64
    calls: list[bool] = []

    def validate(
        checkpoint: Path,
        *,
        allow_development_unqualified: bool = False,
        audit_publication: bool = True,
    ) -> dict[str, str]:
        calls.append(allow_development_unqualified)
        assert audit_publication
        return {
            "schema": "ds4rt-hf-staged-snapshot-v1",
            "model_id": "tpurtell/flash-k2-dev",
            "revision": revision,
            "checkpoint": str(checkpoint),
        }

    monkeypatch.setattr(validator, "validate_staged_exl3_checkpoint", validate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_ds4_staged_snapshot.py",
            "--checkpoint",
            str(tmp_path),
            "--model-id",
            "tpurtell/flash-k2-dev",
            "--revision",
            revision,
            "--allow-development-unqualified",
        ],
    )

    assert validator.main() == 0
    assert calls == [True]
    assert json.loads(capsys.readouterr().out)["revision"] == revision


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("model_id", "tpurtell/other", "staged model ID"),
        ("revision", "b" * 64, "staged revision"),
    ],
)
def test_cli_rejects_requested_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: str,
    match: str,
) -> None:
    revision = "a" * 64
    identity = {
        "schema": "ds4rt-hf-staged-snapshot-v1",
        "model_id": "tpurtell/flash-k2",
        "revision": revision,
    }
    identity[field] = value
    monkeypatch.setattr(
        validator,
        "validate_staged_exl3_checkpoint",
        lambda _checkpoint, **_kwargs: identity,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_ds4_staged_snapshot.py",
            "--checkpoint",
            str(tmp_path),
            "--model-id",
            "tpurtell/flash-k2",
            "--revision",
            revision,
        ],
    )

    with pytest.raises(ValueError, match=match):
        validator.main()


def test_cli_startup_contract_uses_public_serving_validator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    revision = "a" * 64
    calls: list[tuple[Path, str, str]] = []

    def validate(checkpoint: Path, model_id: str, selected: str) -> dict[str, str]:
        calls.append((checkpoint, model_id, selected))
        return {
            "schema": "ds4rt-public-exl3-serving-snapshot-v1",
            "model_id": "tpurtell/flash-k2",
            "revision": revision,
        }

    monkeypatch.setattr(
        validator, "validate_public_exl3_serving_checkpoint", validate
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_ds4_staged_snapshot.py",
            "--checkpoint",
            str(tmp_path),
            "--model-id",
            "tpurtell/flash-k2",
            "--revision",
            revision,
            "--startup-contract-only",
        ],
    )

    assert validator.main() == 0
    assert calls == [(tmp_path, "tpurtell/flash-k2", revision)]


def test_startup_dispatches_generic_public_serving_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "model_id": "publisher/deepseek-v4-exl3",
        "revision": "a" * 40,
    }

    def validate_public(checkpoint: Path, model_id: str, revision: str) -> dict:
        assert checkpoint == Path("public")
        assert model_id == expected["model_id"]
        assert revision == expected["revision"]
        return expected

    monkeypatch.setattr(
        validator,
        "validate_public_exl3_serving_checkpoint",
        validate_public,
    )
    monkeypatch.setattr(
        validator,
        "validate_staged_exl3_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("wrong validator")),
    )

    assert (
        validator.validate_checkpoint(
            Path("public"),
            expected["model_id"],
            expected["revision"],
            allow_development_unqualified=False,
            startup_contract_only=True,
        )
        == expected
    )


def test_full_audit_keeps_private_staging_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {"model_id": "private/model", "revision": "a" * 64}

    def validate_staged(
        checkpoint: Path,
        *,
        allow_development_unqualified: bool,
        audit_publication: bool,
    ) -> dict:
        assert checkpoint == Path("private")
        assert allow_development_unqualified is True
        assert audit_publication is True
        return expected

    monkeypatch.setattr(
        validator,
        "validate_public_exl3_serving_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("wrong validator")),
    )
    monkeypatch.setattr(validator, "validate_staged_exl3_checkpoint", validate_staged)

    assert (
        validator.validate_checkpoint(
            Path("private"),
            "private/model",
            "a" * 64,
            allow_development_unqualified=True,
            startup_contract_only=False,
        )
        == expected
    )
