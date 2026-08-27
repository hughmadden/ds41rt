from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import sync_ds4_hf_snapshot as sync_tool  # noqa: E402
from sync_ds4_hf_snapshot import (  # noqa: E402
    MANIFEST_SCHEMA,
    REMOTE_VERIFY_SOURCE,
    load_staged_cache,
    main,
    model_cache_root,
    remote_hf_home,
    validate_hosts,
    validate_model_id,
)


def make_cache(
    hf_home: Path,
    model_id: str = "tpurtell/test-exl3",
    *,
    gptqmodel: bool = False,
    qualification: bool | None = None,
    development_unqualified: bool = False,
) -> tuple[Path, str]:
    root = model_cache_root(hf_home, model_id)
    entries = []
    config = {"model_type": "deepseek_v4"}
    if gptqmodel:
        config["quantization_config"] = {
            "meta": {"ds4rt_error_ledger": {"test": True}}
        }
    payloads = {
        "config.json": (json.dumps(config, separators=(",", ":")) + "\n").encode(),
        "model-00001-of-00001.safetensors": b"trellis-payload",
    }
    for relative, payload in sorted(payloads.items()):
        digest = hashlib.sha256(payload).hexdigest()
        blob = root / "blobs" / digest
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(payload)
        entries.append({"path": relative, "sha256": digest, "size": len(payload)})
    canonical = json.dumps(
        {"schema": MANIFEST_SCHEMA, "files": entries},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    revision = hashlib.sha256(canonical).hexdigest()
    snapshot = root / "snapshots" / revision
    snapshot.mkdir(parents=True)
    for entry in entries:
        link = snapshot / entry["path"]
        link.symlink_to(Path("../..") / "blobs" / entry["sha256"])
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "model_id": model_id,
        "revision": revision,
        "source_snapshot": "/source",
        "link_mode": "hardlink",
        "files": entries,
    }
    include_qualification = gptqmodel if qualification is None else qualification
    if include_qualification:
        qualification_root = root / (
            "ds4rt-development-evidence"
            if development_unqualified
            else "ds4rt-qualifications"
        ) / revision
        qualification_root.mkdir(parents=True)
        report_payloads = {
            "retained-native.json": json.dumps(
                {"schema": "ds4rt-exl3-retained-native-integrity-v1"}
            ).encode(),
            (
                "diagnostic-quality.json"
                if development_unqualified
                else "expert-quality.json"
            ): json.dumps(
                {
                    "schema": "ds4rt-exl3-checkpoint-quality-v1",
                    "validation_contract": {"sha256": "1" * 64},
                }
            ).encode(),
        }
        qualification_entries = []
        for name, payload in report_payloads.items():
            (qualification_root / name).write_bytes(payload)
            entry = {
                "path": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
                "schema": json.loads(payload)["schema"],
            }
            if name.endswith("quality.json"):
                entry["contract_sha256"] = "1" * 64
            qualification_entries.append(entry)
        if development_unqualified:
            manifest.update(
                {
                    "qualification_status": "development-unqualified",
                    "development_blockers": [
                        "complete token-aware all-layer quality evidence is absent",
                        "natural MTP held-out activation quality evidence is absent",
                    ],
                    "development_evidence": qualification_entries,
                }
            )
        else:
            manifest["qualification"] = qualification_entries
    manifests = root / "ds4rt-manifests"
    manifests.mkdir()
    (manifests / f"{revision}.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    refs = root / "refs"
    refs.mkdir()
    (refs / "main").write_text(revision, encoding="utf-8")
    return root, revision


def test_staged_cache_contract_binds_manifest_snapshot_and_blobs(tmp_path: Path) -> None:
    root, revision = make_cache(tmp_path)
    contract = load_staged_cache(tmp_path, "tpurtell/test-exl3", verify_hashes=True)

    assert contract.root == root
    assert contract.revision == revision
    assert contract.files == 2
    assert contract.bytes == sum(path.stat().st_size for path in (root / "blobs").iterdir())
    assert contract.qualification_reports == 0


def test_gptqmodel_cache_requires_and_hashes_qualification_evidence(
    tmp_path: Path,
) -> None:
    _, revision = make_cache(tmp_path, gptqmodel=True)
    contract = load_staged_cache(tmp_path, "tpurtell/test-exl3")

    assert contract.revision == revision
    assert contract.qualification_reports == 2

    missing_home = tmp_path / "missing"
    make_cache(missing_home, gptqmodel=True, qualification=False)
    with pytest.raises(ValueError, match="no complete qualification evidence"):
        load_staged_cache(missing_home, "tpurtell/test-exl3")


def test_development_cache_requires_explicit_local_and_remote_overrides(
    tmp_path: Path,
) -> None:
    root, revision = make_cache(
        tmp_path,
        gptqmodel=True,
        development_unqualified=True,
    )
    with pytest.raises(ValueError, match="development-unqualified"):
        load_staged_cache(tmp_path, "tpurtell/test-exl3")
    contract = load_staged_cache(
        tmp_path,
        "tpurtell/test-exl3",
        allow_development_unqualified=True,
    )
    assert contract.qualification_status == "development-unqualified"
    assert contract.qualification_reports == 0
    assert contract.development_evidence_reports == 2

    denied = subprocess.run(
        [sys.executable, "-", str(root), revision, "tpurtell/test-exl3", "0"],
        input=REMOTE_VERIFY_SOURCE,
        text=True,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert denied.returncode != 0
    assert "development-unqualified" in denied.stderr

    accepted = subprocess.run(
        [sys.executable, "-", str(root), revision, "tpurtell/test-exl3", "1"],
        input=REMOTE_VERIFY_SOURCE,
        text=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    report = json.loads(accepted.stdout)
    assert report["qualification_status"] == "development-unqualified"
    assert report["qualification_reports"] == 0
    assert report["development_evidence_reports"] == 2


def test_staged_cache_contract_rejects_tampered_blob(tmp_path: Path) -> None:
    root, _ = make_cache(tmp_path)
    blob = next((root / "blobs").iterdir())
    blob.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="blob metadata mismatch|blob digest mismatch"):
        load_staged_cache(tmp_path, "tpurtell/test-exl3", verify_hashes=True)


def test_remote_verifier_hashes_before_atomically_publishing_ref(tmp_path: Path) -> None:
    root, revision = make_cache(tmp_path)
    (root / "refs" / "main").write_text("0" * 64 + "\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-", str(root), revision, "tpurtell/test-exl3"],
        input=REMOTE_VERIFY_SOURCE,
        text=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    report = json.loads(result.stdout)
    assert report["revision"] == revision
    assert report["verified_blobs"] == 2
    assert report["qualification_reports"] == 0
    assert (root / "refs" / "main").read_text(encoding="utf-8") == revision


def test_remote_verifier_hashes_gptqmodel_qualification_before_ref(
    tmp_path: Path,
) -> None:
    root, revision = make_cache(tmp_path, gptqmodel=True)
    ref = root / "refs" / "main"
    ref.write_text("0" * 64 + "\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-", str(root), revision, "tpurtell/test-exl3"],
        input=REMOTE_VERIFY_SOURCE,
        text=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    report = json.loads(result.stdout)
    assert report["qualification_reports"] == 2
    assert ref.read_text(encoding="utf-8") == revision


def test_remote_verifier_rejects_tampered_qualification_before_ref(
    tmp_path: Path,
) -> None:
    root, revision = make_cache(tmp_path, gptqmodel=True)
    previous = "0" * 64
    ref = root / "refs" / "main"
    ref.write_text(previous + "\n", encoding="utf-8")
    quality = root / "ds4rt-qualifications" / revision / "expert-quality.json"
    quality.write_bytes(b"x" * quality.stat().st_size)

    result = subprocess.run(
        [sys.executable, "-", str(root), revision, "tpurtell/test-exl3"],
        input=REMOTE_VERIFY_SOURCE,
        text=True,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode != 0
    assert "qualification evidence digest mismatch" in result.stderr
    assert ref.read_text(encoding="utf-8") == previous + "\n"


def test_remote_verifier_does_not_publish_corrupt_revision(tmp_path: Path) -> None:
    root, revision = make_cache(tmp_path)
    previous = "0" * 64
    ref = root / "refs" / "main"
    ref.write_text(previous + "\n", encoding="utf-8")
    blob = next((root / "blobs").iterdir())
    blob.write_bytes(b"x" * blob.stat().st_size)

    result = subprocess.run(
        [sys.executable, "-", str(root), revision, "tpurtell/test-exl3"],
        input=REMOTE_VERIFY_SOURCE,
        text=True,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode != 0
    assert "blob digest mismatch" in result.stderr
    assert ref.read_text(encoding="utf-8") == previous + "\n"


def test_sync_inputs_reject_unsafe_or_duplicate_names() -> None:
    assert validate_model_id("tpurtell/model") == ("tpurtell", "model")
    assert validate_hosts("ostrich,dodo,emu,kiwi") == (
        "ostrich",
        "dodo",
        "emu",
        "kiwi",
    )
    with pytest.raises(ValueError, match="model-id"):
        validate_model_id("../unsafe")
    with pytest.raises(ValueError, match="unsafe host"):
        validate_hosts("ostrich,bad;host")
    with pytest.raises(ValueError, match="unique"):
        validate_hosts("ostrich,ostrich")


def test_remote_hf_home_sends_python_source_on_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], str | None]] = []

    def fake_run_checked(
        command: list[str], *, input_text: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, input_text))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="/home/tj/.cache/huggingface\n",
            stderr="",
        )

    monkeypatch.setattr(sync_tool, "run_checked", fake_run_checked)

    assert remote_hf_home("ostrich") == Path("/home/tj/.cache/huggingface")
    assert calls == [
        (
            ["ssh", "-o", "BatchMode=yes", "ostrich", "python3", "-"],
            sync_tool.REMOTE_HF_HOME_SOURCE + "\n",
        )
    ]


def test_dry_run_validates_local_cache_without_remote_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, revision = make_cache(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sync_ds4_hf_snapshot.py",
            "--model-id",
            "tpurtell/test-exl3",
            "--hosts",
            "ostrich,dodo,emu,kiwi",
            "--hf-home",
            str(tmp_path),
            "--dry-run",
        ],
    )

    assert main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["revision"] == revision
    assert report["hosts"] == ["ostrich", "dodo", "emu", "kiwi"]
    assert report["dry_run"] is True
    assert report["results"] == []
