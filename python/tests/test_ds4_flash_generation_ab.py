from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import threading
from typing import Any

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import validate_ds4_flash_generation_ab as generation  # noqa: E402

from validate_ds4_flash_generation_ab import (  # noqa: E402
    CASES,
    RUN_SCHEMA,
    RUNTIME_GATE,
    anthropic_completion_payload,
    anthropic_response_content,
    audit_checkpoint,
    checkpoint_quantization_recipe,
    collect_matrix_arm,
    compare_matrix,
    compare_runs,
    prepare_report_output,
    request_completion_group,
    resolve_collection_model,
    score_output,
    suite_identity,
    validate_dspark_group_evidence,
    validate_runtime_evidence,
    validate_staged_exl3_checkpoint,
    write_report,
)


def test_checkpoint_recipe_distinguishes_native_and_production_exl3(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = checkpoint / "config.json"
    config.write_text(
        json.dumps({"quantization_config": {"quant_method": "fp8"}}),
        encoding="utf-8",
    )
    assert checkpoint_quantization_recipe(checkpoint) == (
        "deepseek_v4_native_fp4_fp8_mixed_v1"
    )

    config.write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "exl3",
                    "ds4rt": {"recipe": "deepseek_v4_exl3_trellis_2bpw_v2"},
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="historical non-production"):
        checkpoint_quantization_recipe(checkpoint)
    assert checkpoint_quantization_recipe(
        checkpoint, allow_historical_control=True
    ) == ("deepseek_v4_exl3_trellis_2bpw_v2")

    config.write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "exl3",
                    "ds4rt": {
                        "recipe": "deepseek_v4_exl3_trellis_2bpw_v3_flash_activation_pilot"
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="historical non-production"):
        checkpoint_quantization_recipe(checkpoint)

    config.write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "exl3",
                    "ds4rt": {
                        "recipe": "deepseek_v4_exl3_trellis_2bpw_v4_flash_natural_route"
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    assert checkpoint_quantization_recipe(checkpoint) == (
        "deepseek_v4_exl3_trellis_2bpw_v4_flash_natural_route"
    )

    config.write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "exl3",
                    "ds4rt": {"recipe": "retired-recipe"},
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported DS4RT EXL3 recipe"):
        checkpoint_quantization_recipe(checkpoint)


def test_checkpoint_recipe_accepts_exact_compact_public_gptqmodel_config(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    declaration = {
        "bits": 2.0,
        "checkpoint_format": "exl3",
        "codebook": "mcg",
        "desc_act": False,
        "format": "exl3",
        "group_size": -1,
        "method": "exl3",
        "module_include": [
            r"^model\.layers\.\d+\.mlp\.experts\.\d+\."
            r"(?:gate_proj|up_proj|down_proj)$"
        ],
        "out_scales": "auto",
        "quant_method": "exl3",
    }
    (checkpoint / "config.json").write_text(
        json.dumps({"quantization_config": declaration}), encoding="utf-8"
    )
    (checkpoint / "quantize_config.json").write_text(
        json.dumps({**declaration, "tensor_storage": {"projection": {}}}),
        encoding="utf-8",
    )

    assert checkpoint_quantization_recipe(checkpoint) == generation.GPTQMODEL_RECIPE

    declaration["bits"] = 3.0
    (checkpoint / "config.json").write_text(
        json.dumps({"quantization_config": declaration}), encoding="utf-8"
    )
    (checkpoint / "quantize_config.json").write_text(
        json.dumps({**declaration, "tensor_storage": {"projection": {}}}),
        encoding="utf-8",
    )
    assert (
        checkpoint_quantization_recipe(checkpoint)
        == generation.GPTQMODEL_RECIPE_K3
    )

    declaration["bits"] = 2.0
    declaration["meta"] = {
        "ds4rt_inline_mixed": {
            "schema": "gptqmodel.exl3-inline-mixed",
            "schema_version": 1,
            "namespace": "base",
            "base_bits": 2,
            "upgrade_bits": 3,
            "extra_bits": {"numerator": 1, "denominator": 10},
            "target_bpw": "21/10",
            "projection_ratio": {"w1": 3, "w3": 5, "w2": 8},
            "score_kind": (
                "k2-hessian-weighted-relative-error-times-natural-gate-squared-mass-v1"
            ),
        }
    }
    (checkpoint / "config.json").write_text(
        json.dumps({"quantization_config": declaration}), encoding="utf-8"
    )
    (checkpoint / "quantize_config.json").write_text(
        json.dumps({**declaration, "tensor_storage": {"projection": {}}}),
        encoding="utf-8",
    )
    assert (
        checkpoint_quantization_recipe(checkpoint)
        == generation.GPTQMODEL_RECIPE_MIXED_K2_K3
    )

    declaration["meta"]["ds4rt_inline_mixed"]["target_bpw"] = "22/10"
    (checkpoint / "config.json").write_text(
        json.dumps({"quantization_config": declaration}), encoding="utf-8"
    )
    (checkpoint / "quantize_config.json").write_text(
        json.dumps({**declaration, "tensor_storage": {"projection": {}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="target disagrees"):
        checkpoint_quantization_recipe(checkpoint)

    declaration.pop("meta")
    declaration["bits"] = 3.0
    (checkpoint / "config.json").write_text(
        json.dumps({"quantization_config": declaration}), encoding="utf-8"
    )
    external = {**declaration, "tensor_storage": {"projection": {}}}
    external["bits"] = 2.0
    (checkpoint / "quantize_config.json").write_text(
        json.dumps(external), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unsupported DS4RT EXL3 recipe"):
        checkpoint_quantization_recipe(checkpoint)


def staged_exl3_checkpoint(
    tmp_path: Path,
    *,
    include_error_ledger: bool = True,
    standard_files_only: bool = False,
    include_readme: bool = False,
) -> Path:
    contents = {
        "config.json": json.dumps(
            {
                "quantization_config": {
                    "quant_method": "exl3",
                    "ds4rt": {
                        "recipe": "deepseek_v4_exl3_trellis_2bpw_v4_flash_natural_route"
                    },
                }
            }
        ).encode(),
        "model.safetensors.index.json": b'{"weight_map":{"x":"model.safetensors"}}',
        "quantization_config.json": b"{}",
        "ds4rt-exl3-calibration.json": b"{}",
        "ds4rt-exl3-quality.json": b"{}",
        "ds4rt-exl3-retained-native.json": b"{}",
        "ds4rt-exl3-multigpu-production.json": b"{}",
        "ds4rt-exl3-error-ledger.jsonl": b"{}\n",
        "ds4rt-exl3-error-ledger.manifest.json": b"{}",
        "model.safetensors": b"trellis",
    }
    if not include_error_ledger:
        del contents["ds4rt-exl3-error-ledger.jsonl"]
        del contents["ds4rt-exl3-error-ledger.manifest.json"]
    if include_readme:
        contents["README.md"] = b"original model card\n"
    if standard_files_only:
        for path in tuple(contents):
            if path.startswith("ds4rt-") or path == "quantization_config.json":
                del contents[path]
    entries = []
    for path, content in sorted(contents.items()):
        entries.append(
            {
                "path": path,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": len(content),
            }
        )
    canonical = json.dumps(
        {"schema": "ds4rt-hf-staged-snapshot-v1", "files": entries},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    revision = hashlib.sha256(canonical).hexdigest()
    model_root = tmp_path / "models--tpurtell--flash-exl3"
    blobs = model_root / "blobs"
    snapshot = model_root / "snapshots" / revision
    manifest_dir = model_root / "ds4rt-manifests"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    manifest_dir.mkdir()
    for entry in entries:
        blob = blobs / entry["sha256"]
        blob.write_bytes(contents[entry["path"]])
        (snapshot / entry["path"]).symlink_to(
            Path("../../blobs") / entry["sha256"]
        )
    (manifest_dir / f"{revision}.json").write_text(
        json.dumps(
            {
                "schema": "ds4rt-hf-staged-snapshot-v1",
                "model_id": "tpurtell/flash-exl3",
                "revision": revision,
                "source_snapshot": "/immutable/source",
                "link_mode": "hardlink",
                "files": entries,
            }
        ),
        encoding="utf-8",
    )
    return snapshot


def staged_gptqmodel_checkpoint(tmp_path: Path, *, bits: int = 2) -> Path:
    recipe = (
        generation.GPTQMODEL_RECIPE
        if bits == 2
        else generation.GPTQMODEL_RECIPE_K3
    )
    config = {
        "quantization_config": {
            "quant_method": "exl3",
            "bits": float(bits),
            "meta": {"ds4rt_error_ledger": {}},
        }
    }
    contents = {
        "config.json": json.dumps(config).encode(),
        "model.safetensors.index.json": (
            b'{"weight_map":{"x":"model.safetensors"}}'
        ),
        "quantize_config.json": json.dumps(config["quantization_config"]).encode(),
        "ds4rt-gptqmodel-plan.json": b"{}",
        "ds4rt-gptqmodel-run.json": b"{}",
        "ds4rt-gptqmodel-artifact.json": b"{}",
        "ds4rt-exl3-error-ledger.jsonl": b"{}\n",
        "ds4rt-exl3-error-ledger.manifest.json": b"{}",
        "model.safetensors": b"trellis",
    }
    entries = [
        {
            "path": path,
            "sha256": hashlib.sha256(content).hexdigest(),
            "size": len(content),
        }
        for path, content in sorted(contents.items())
    ]
    canonical = json.dumps(
        {"schema": "ds4rt-hf-staged-snapshot-v1", "files": entries},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    revision = hashlib.sha256(canonical).hexdigest()
    model_root = tmp_path / "models--tpurtell--flash-gptqmodel-exl3"
    blobs = model_root / "blobs"
    snapshot = model_root / "snapshots" / revision
    manifest_dir = model_root / "ds4rt-manifests"
    qualification_dir = model_root / "ds4rt-qualifications" / revision
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    manifest_dir.mkdir()
    qualification_dir.mkdir(parents=True)
    for entry in entries:
        blob = blobs / entry["sha256"]
        blob.write_bytes(contents[entry["path"]])
        (snapshot / entry["path"]).symlink_to(
            Path("../../blobs") / entry["sha256"]
        )

    source_snapshot = "/immutable/gptqmodel-source"
    metadata_sha256 = {
        name: hashlib.sha256(contents[name]).hexdigest()
        for name in generation.GPTQMODEL_IDENTITY_FILES
    }
    retained = {
        "schema": "ds4rt-exl3-retained-native-integrity-v1",
        "recipe": recipe,
        "quantization_scope": "routed_experts_only",
        "exl3_snapshot": source_snapshot,
        "gptqmodel_publication": {
            "path": source_snapshot,
            "metadata_sha256": metadata_sha256,
        },
    }
    exl3_identity = {
        "path": source_snapshot,
        "metadata_sha256": metadata_sha256,
        "shards": [
            {
                "name": "model.safetensors",
                "device": 1,
                "inode": 2,
                "size": len(contents["model.safetensors"]),
                "mtime_ns": 3,
            }
        ],
    }
    contract_body = {"allow_incomplete": False, "exl3": exl3_identity}
    contract_sha256 = hashlib.sha256(
        json.dumps(
            contract_body,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    quality = {
        "schema": "ds4rt-exl3-checkpoint-quality-v1",
        "validation_contract": {**contract_body, "sha256": contract_sha256},
    }
    qualification_entries = []
    for name, schema, report in (
        (
            "retained-native.json",
            "ds4rt-exl3-retained-native-integrity-v1",
            retained,
        ),
        (
            "expert-quality.json",
            "ds4rt-exl3-checkpoint-quality-v1",
            quality,
        ),
    ):
        payload = (json.dumps(report, sort_keys=True) + "\n").encode()
        (qualification_dir / name).write_bytes(payload)
        record = {
            "path": name,
            "schema": schema,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
        }
        if name == "expert-quality.json":
            record["contract_sha256"] = contract_sha256
        qualification_entries.append(record)

    (manifest_dir / f"{revision}.json").write_text(
        json.dumps(
            {
                "schema": "ds4rt-hf-staged-snapshot-v1",
                "model_id": "tpurtell/flash-gptqmodel-exl3",
                "revision": revision,
                "source_snapshot": source_snapshot,
                "link_mode": "hardlink",
                "files": entries,
                "qualification": qualification_entries,
            }
        ),
        encoding="utf-8",
    )
    return snapshot


def test_staged_exl3_checkpoint_binds_canonical_manifest(tmp_path: Path) -> None:
    checkpoint = staged_exl3_checkpoint(tmp_path)

    identity = validate_staged_exl3_checkpoint(checkpoint)

    assert identity["revision"] == checkpoint.name
    assert identity["model_id"] == "tpurtell/flash-exl3"
    assert identity["files"] == 10
    assert identity["bytes"] > 0

    audit = audit_checkpoint(checkpoint)
    assert audit["summary"]["gate_passed"] is True
    assert audit["artifact_identity"] == identity
    assert audit["quantization_recipe"] == (
        "deepseek_v4_exl3_trellis_2bpw_v4_flash_natural_route"
    )


def test_legacy_control_requires_explicit_override_for_missing_error_ledger(
    tmp_path: Path,
) -> None:
    checkpoint = staged_exl3_checkpoint(tmp_path, include_error_ledger=False)

    with pytest.raises(ValueError, match="lacks production evidence"):
        validate_staged_exl3_checkpoint(checkpoint)

    identity = validate_staged_exl3_checkpoint(
        checkpoint,
        allow_development_unqualified=True,
    )
    assert identity["development_override"] == {
        "kind": "wip-standard-artifact-without-private-evidence",
        "missing_production_evidence": [
            "ds4rt-exl3-error-ledger.jsonl",
            "ds4rt-exl3-error-ledger.manifest.json",
        ],
    }


def test_wip_override_accepts_standard_artifact_without_private_evidence(
    tmp_path: Path,
) -> None:
    checkpoint = staged_exl3_checkpoint(tmp_path, standard_files_only=True)

    with pytest.raises(ValueError, match="lacks production evidence"):
        validate_staged_exl3_checkpoint(checkpoint)

    identity = validate_staged_exl3_checkpoint(
        checkpoint,
        allow_development_unqualified=True,
    )
    assert identity["development_override"]["kind"] == (
        "wip-standard-artifact-without-private-evidence"
    )
    assert identity["development_override"]["missing_production_evidence"] == [
        "ds4rt-exl3-calibration.json",
        "ds4rt-exl3-error-ledger.jsonl",
        "ds4rt-exl3-error-ledger.manifest.json",
        "ds4rt-exl3-multigpu-production.json",
        "ds4rt-exl3-quality.json",
        "ds4rt-exl3-retained-native.json",
        "quantization_config.json",
    ]


def test_wip_override_accepts_model_card_size_drift_only(tmp_path: Path) -> None:
    checkpoint = staged_exl3_checkpoint(
        tmp_path,
        standard_files_only=True,
        include_readme=True,
    )
    (checkpoint / "README.md").resolve().write_bytes(b"updated public model card\n")

    with pytest.raises(ValueError, match="entry size differs from manifest"):
        validate_staged_exl3_checkpoint(checkpoint)

    identity = validate_staged_exl3_checkpoint(
        checkpoint,
        allow_development_unqualified=True,
    )
    assert identity["development_override"]["kind"] == (
        "wip-standard-artifact-without-private-evidence"
    )

    (checkpoint / "config.json").resolve().write_bytes(b"larger serving config\n")
    with pytest.raises(ValueError, match="entry size differs from manifest"):
        validate_staged_exl3_checkpoint(
            checkpoint,
            allow_development_unqualified=True,
        )


def test_collection_model_is_derived_from_staged_identity() -> None:
    identity = {
        "schema": "ds4rt-hf-staged-snapshot-v1",
        "model_id": "tpurtell/DeepSeek-V4-Flash-0731-EXL3-2bpw-v4",
        "revision": "a" * 64,
    }

    expected = "tpurtell/DeepSeek-V4-Flash-0731-EXL3-2bpw-v4-full"
    assert resolve_collection_model(None, identity) == expected
    assert resolve_collection_model(expected, identity) == expected

    with pytest.raises(ValueError, match="differs from staged checkpoint identity"):
        resolve_collection_model(
            "deepseek-ai/DeepSeek-V4-Flash-0731-full",
            identity,
        )


def test_collection_model_retains_native_default_and_override() -> None:
    identity = {
        "schema": "huggingface-native-snapshot-v1",
        "revision": "a" * 40,
    }

    assert resolve_collection_model(None, identity) == (
        "deepseek-ai/DeepSeek-V4-Flash-0731-full"
    )
    assert resolve_collection_model("local/native-full", identity) == (
        "local/native-full"
    )


def test_staged_gptqmodel_checkpoint_binds_external_qualification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = staged_gptqmodel_checkpoint(tmp_path)
    monkeypatch.setattr(
        generation,
        "validate_gptqmodel_native_exl3",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        generation,
        "validate_gptqmodel_publication",
        lambda *_args, **_kwargs: {},
    )

    assert checkpoint_quantization_recipe(checkpoint) == generation.GPTQMODEL_RECIPE
    identity = validate_staged_exl3_checkpoint(checkpoint)
    assert identity["model_id"] == "tpurtell/flash-gptqmodel-exl3"
    assert [entry["path"] for entry in identity["qualification"]] == [
        "expert-quality.json",
        "retained-native.json",
    ]


def test_staged_uniform_k3_checkpoint_binds_external_qualification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = staged_gptqmodel_checkpoint(tmp_path, bits=3)
    monkeypatch.setattr(
        generation,
        "validate_gptqmodel_native_exl3",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        generation,
        "validate_gptqmodel_publication",
        lambda *_args, **_kwargs: {},
    )

    assert (
        checkpoint_quantization_recipe(checkpoint)
        == generation.GPTQMODEL_RECIPE_K3
    )
    identity = validate_staged_exl3_checkpoint(checkpoint)
    assert identity["model_id"] == "tpurtell/flash-gptqmodel-exl3"


def test_staged_gptqmodel_checkpoint_rejects_missing_qualification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = staged_gptqmodel_checkpoint(tmp_path)
    manifest = checkpoint.parent.parent / "ds4rt-manifests" / f"{checkpoint.name}.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.pop("qualification")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        generation,
        "validate_gptqmodel_native_exl3",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        generation,
        "validate_gptqmodel_publication",
        lambda *_args, **_kwargs: {},
    )

    with pytest.raises(ValueError, match="lacks complete external qualification"):
        validate_staged_exl3_checkpoint(checkpoint)


def test_staged_gptqmodel_checkpoint_rejects_rebound_qualification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = staged_gptqmodel_checkpoint(tmp_path)
    model_root = checkpoint.parent.parent
    manifest_path = model_root / "ds4rt-manifests" / f"{checkpoint.name}.json"
    qualification_path = (
        model_root
        / "ds4rt-qualifications"
        / checkpoint.name
        / "retained-native.json"
    )
    report = json.loads(qualification_path.read_text(encoding="utf-8"))
    report["exl3_snapshot"] = "/different/publication"
    payload = (json.dumps(report, sort_keys=True) + "\n").encode()
    qualification_path.write_bytes(payload)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(
        item
        for item in manifest["qualification"]
        if item["path"] == "retained-native.json"
    )
    entry["sha256"] = hashlib.sha256(payload).hexdigest()
    entry["size"] = len(payload)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        generation,
        "validate_gptqmodel_native_exl3",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        generation,
        "validate_gptqmodel_publication",
        lambda *_args, **_kwargs: {},
    )

    with pytest.raises(ValueError, match="not bound to this publication"):
        validate_staged_exl3_checkpoint(checkpoint)


def test_staged_exl3_checkpoint_rejects_wrong_blob_target(tmp_path: Path) -> None:
    checkpoint = staged_exl3_checkpoint(tmp_path)
    config = checkpoint / "config.json"
    config.unlink()
    wrong_blob = checkpoint.parent.parent / "blobs" / ("f" * 64)
    wrong_blob.write_bytes(b"wrong")
    config.symlink_to(Path("../../blobs") / wrong_blob.name)

    with pytest.raises(ValueError, match="wrong blob"):
        validate_staged_exl3_checkpoint(checkpoint)


def test_anthropic_reference_preserves_reasoning_budget_and_visible_answer() -> None:
    case = next(case for case in CASES if case.case_id == "multiply")
    payload = json.loads(
        anthropic_completion_payload("deepseek-v4-flash", case, 256, "disabled")
    )
    assert payload["max_tokens"] == case.max_tokens
    assert payload["temperature"] == 0
    assert payload["thinking"] == {"type": "disabled"}
    visible, reasoning = anthropic_response_content(
        {
            "content": [
                {"type": "thinking", "thinking": "calculate"},
                {"type": "text", "text": "3874973"},
            ]
        }
    )
    assert visible == "3874973"
    assert reasoning == "calculate"

    omitted = json.loads(
        anthropic_completion_payload("deepseek-v4-flash", case, 256, "omit")
    )
    assert omitted["max_tokens"] == 256
    assert "thinking" not in omitted


def test_quality_scorers_keep_exact_and_structured_checks_distinct() -> None:
    exact = next(case for case in CASES if case.case_id == "multiply")
    structured = next(case for case in CASES if case.case_id == "structured-json")

    assert score_output(exact, " 3874973\n") == (True, None)
    assert score_output(exact, "The answer is 3874973")[0] is False
    assert score_output(
        structured,
        '{"line_end":47,"operation":"replace","path":"src/cache.rs","line_start":41}',
    ) == (True, None)
    passed, error = score_output(structured, "not-json")
    assert passed is False
    assert error is not None and "invalid JSON" in error


def test_suite_identity_changes_when_case_contract_changes() -> None:
    assert suite_identity(CASES) == suite_identity(tuple(CASES))
    assert suite_identity(CASES[:-1]) != suite_identity(CASES)


def run_report(
    label: str,
    outputs: list[tuple[str, bool]],
    *,
    service_instance_id: str | None = None,
) -> dict:
    if service_instance_id is None:
        service_instance_id = (
            "00000000-0000-4000-8000-000000000001"
            if label == "native"
            else "00000000-0000-4000-8000-000000000002"
        )
    records = []
    for repeat, (content, passed) in enumerate(outputs, start=1):
        records.append(
            {
                "case_id": "multiply",
                "repeat": repeat,
                "content": content,
                "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "passed": passed,
                "runtime_evidence": {
                    "backend_mode": "real-ds4-full",
                    "startup_diagnostic_mode": "request-scheduler-execution",
                    "service_instance_id": service_instance_id,
                },
            }
        )
    return {
        "schema": RUN_SCHEMA,
        "runtime_gate": RUNTIME_GATE,
        "service_instance_id": service_instance_id,
        "artifact_identity": (
            {
                "schema": "huggingface-native-snapshot-v1",
                "revision": "native-revision",
            }
            if label == "native"
            else {
                "schema": "ds4rt-hf-staged-snapshot-v1",
                "revision": "a" * 64,
            }
        ),
        "suite_sha256": "same-suite",
        "artifact_label": label,
        "expected_quantization_recipe": (
            "deepseek_v4_native_fp4_fp8_mixed_v1"
            if label == "native"
            else "deepseek_v4_exl3_trellis_2bpw_v4_flash_natural_route"
        ),
        "expected_dspark": "on",
        "concurrency": 1,
        "records": records,
    }


def test_comparison_reports_quantized_quality_regression_and_output_drift() -> None:
    native = run_report("native", [("3874973", True), ("3874973", True)])
    candidate = run_report("exl3", [("3874973", True), ("3874972", False)])

    report = compare_runs(
        native,
        candidate,
        min_candidate_pass_rate=1.0,
        max_quality_regressions=0,
    )

    assert report["summary"]["candidate_pass_rate"] == pytest.approx(0.5)
    assert report["summary"]["quality_regressions"] == 1
    assert report["summary"]["content_exact_rate"] == pytest.approx(0.5)
    assert report["summary"]["gate_passed"] is False


def test_default_comparison_uses_native_pass_rate_without_requiring_perfection() -> None:
    native = run_report("native", [("3874973", True), ("wrong", False)])
    candidate = run_report("exl3", [("3874973", True), ("wrong", False)])

    report = compare_runs(native, candidate)

    assert report["thresholds"]["min_candidate_pass_rate"] == pytest.approx(0.5)
    assert report["thresholds"]["min_candidate_pass_rate_source"] == "native_run"
    assert report["summary"]["quality_regressions"] == 0
    assert report["summary"]["gate_passed"] is True


def test_comparison_rejects_different_suites() -> None:
    native = run_report("native", [("3874973", True)])
    candidate = run_report("exl3", [("3874973", True)])
    candidate["suite_sha256"] = "different-suite"

    with pytest.raises(ValueError, match="different task suites"):
        compare_runs(
            native,
            candidate,
            min_candidate_pass_rate=1.0,
            max_quality_regressions=0,
        )


def test_matrix_requires_all_dspark_and_concurrency_cells(tmp_path: Path) -> None:
    native_dir = tmp_path / "native"
    candidate_dir = tmp_path / "candidate"
    native_dir.mkdir()
    candidate_dir.mkdir()
    for expected_dspark in ("on", "off"):
        suffix = "1" if expected_dspark == "on" else "3"
        native_service_id = f"00000000-0000-4000-8000-00000000000{suffix}"
        suffix = "2" if expected_dspark == "on" else "4"
        candidate_service_id = f"00000000-0000-4000-8000-00000000000{suffix}"
        for concurrency in range(1, 5):
            name = f"dspark-{expected_dspark}-c{concurrency}.json"
            native = run_report(
                "native",
                [("3874973", True)],
                service_instance_id=native_service_id,
            )
            candidate = run_report(
                "exl3",
                [("3874973", True)],
                service_instance_id=candidate_service_id,
            )
            for report in (native, candidate):
                report["expected_dspark"] = expected_dspark
                report["concurrency"] = concurrency
            (native_dir / name).write_text(json.dumps(native), encoding="utf-8")
            (candidate_dir / name).write_text(json.dumps(candidate), encoding="utf-8")

    report = compare_matrix(native_dir, candidate_dir)

    assert report["summary"] == {
        "cells": 8,
        "passed_cells": 8,
        "gate_passed": True,
    }
    assert report["service_lifetimes"] == {
        "native": {
            "on": "00000000-0000-4000-8000-000000000001",
            "off": "00000000-0000-4000-8000-000000000003",
        },
        "candidate": {
            "on": "00000000-0000-4000-8000-000000000002",
            "off": "00000000-0000-4000-8000-000000000004",
        },
    }
    assert report["artifact_identities"]["candidate"] == {
        "schema": "ds4rt-hf-staged-snapshot-v1",
        "revision": "a" * 64,
    }


def test_matrix_rejects_restart_between_concurrency_cells(tmp_path: Path) -> None:
    native_dir = tmp_path / "native"
    candidate_dir = tmp_path / "candidate"
    native_dir.mkdir()
    candidate_dir.mkdir()
    for expected_dspark in ("on", "off"):
        suffix = "1" if expected_dspark == "on" else "3"
        native_service_id = f"00000000-0000-4000-8000-00000000000{suffix}"
        suffix = "2" if expected_dspark == "on" else "4"
        candidate_service_id = f"00000000-0000-4000-8000-00000000000{suffix}"
        for concurrency in range(1, 5):
            name = f"dspark-{expected_dspark}-c{concurrency}.json"
            native = run_report(
                "native",
                [("3874973", True)],
                service_instance_id=native_service_id,
            )
            candidate = run_report(
                "exl3",
                [("3874973", True)],
                service_instance_id=candidate_service_id,
            )
            for report in (native, candidate):
                report["expected_dspark"] = expected_dspark
                report["concurrency"] = concurrency
            if expected_dspark == "on" and concurrency == 3:
                replacement = "00000000-0000-4000-8000-000000000003"
                native["service_instance_id"] = replacement
                native["records"][0]["runtime_evidence"][
                    "service_instance_id"
                ] = replacement
            (native_dir / name).write_text(json.dumps(native), encoding="utf-8")
            (candidate_dir / name).write_text(json.dumps(candidate), encoding="utf-8")

    with pytest.raises(ValueError, match="C1-C4 crossed service lifetimes"):
        compare_matrix(native_dir, candidate_dir)


def test_matrix_rejects_different_candidate_artifacts(tmp_path: Path) -> None:
    native_dir = tmp_path / "native"
    candidate_dir = tmp_path / "candidate"
    native_dir.mkdir()
    candidate_dir.mkdir()
    for expected_dspark in ("on", "off"):
        native_service_id = (
            "00000000-0000-4000-8000-000000000001"
            if expected_dspark == "on"
            else "00000000-0000-4000-8000-000000000003"
        )
        candidate_service_id = (
            "00000000-0000-4000-8000-000000000002"
            if expected_dspark == "on"
            else "00000000-0000-4000-8000-000000000004"
        )
        for concurrency in range(1, 5):
            name = f"dspark-{expected_dspark}-c{concurrency}.json"
            native = run_report(
                "native",
                [("3874973", True)],
                service_instance_id=native_service_id,
            )
            candidate = run_report(
                "exl3",
                [("3874973", True)],
                service_instance_id=candidate_service_id,
            )
            for report in (native, candidate):
                report["expected_dspark"] = expected_dspark
                report["concurrency"] = concurrency
            if expected_dspark == "off" and concurrency == 1:
                candidate["artifact_identity"]["revision"] = "b" * 64
            (native_dir / name).write_text(json.dumps(native), encoding="utf-8")
            (candidate_dir / name).write_text(json.dumps(candidate), encoding="utf-8")

    with pytest.raises(ValueError, match="different artifact identities"):
        compare_matrix(native_dir, candidate_dir)


def test_collect_matrix_arm_uses_one_service_for_c1_through_c4(monkeypatch) -> None:
    calls = []

    def fake_collect_run(**kwargs) -> dict:
        concurrency = kwargs["concurrency"]
        calls.append(concurrency)
        return {
            "service_instance_id": "00000000-0000-4000-8000-000000000001",
            "artifact_identity": {
                "schema": "ds4rt-hf-staged-snapshot-v1",
                "revision": "a" * 64,
            },
            "suite_sha256": "suite",
            "summary": {"samples": 4, "passed": 4},
        }

    monkeypatch.setattr(
        "validate_ds4_flash_generation_ab.collect_run", fake_collect_run
    )

    report, runs = collect_matrix_arm(
        url="http://test",
        model="model",
        artifact_label="flash-exl3-on",
        checkpoint="/checkpoint",
        cases=CASES,
        repeats=1,
        timeout=1.0,
        expected_spark_targets=4,
        expected_dspark="on",
    )

    assert calls == [1, 2, 3, 4]
    assert sorted(runs) == [1, 2, 3, 4]
    assert report["summary"] == {
        "cells": 4,
        "samples": 16,
        "passed": 16,
        "gate_passed": True,
    }


def test_collect_matrix_arm_rejects_service_restart(monkeypatch) -> None:
    def fake_collect_run(**kwargs) -> dict:
        concurrency = kwargs["concurrency"]
        return {
            "service_instance_id": (
                "00000000-0000-4000-8000-000000000001"
                if concurrency < 3
                else "00000000-0000-4000-8000-000000000002"
            ),
            "artifact_identity": {
                "schema": "ds4rt-hf-staged-snapshot-v1",
                "revision": "a" * 64,
            },
            "suite_sha256": "suite",
            "summary": {"samples": 4, "passed": 4},
        }

    monkeypatch.setattr(
        "validate_ds4_flash_generation_ab.collect_run", fake_collect_run
    )

    with pytest.raises(ValueError, match="crossed service lifetimes"):
        collect_matrix_arm(
            url="http://test",
            model="model",
            artifact_label="flash-exl3-on",
            checkpoint="/checkpoint",
            cases=CASES,
            repeats=1,
            timeout=1.0,
            expected_spark_targets=4,
            expected_dspark="on",
        )


def test_report_outputs_are_atomic_and_write_once_by_default(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "report.json"
    write_report(output, {"schema": "first", "value": 1})
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "schema": "first",
        "value": 1,
    }
    assert list(output.parent.glob(f".{output.name}.*.tmp")) == []

    with pytest.raises(ValueError, match="already exists"):
        prepare_report_output(output, replace=False)
    assert output.exists()

    prepare_report_output(output, replace=True)
    assert not output.exists()


def production_result(checkpoint: Path) -> dict[str, Any]:
    return {
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731-full",
        "metrics": {
            "backend_mode": "real-ds4-full",
            "transport_backend": "verbs-host",
            "real_full": {
                "model_id": "deepseek-ai/DeepSeek-V4-Flash-0731",
                "service_instance_id": "00000000-0000-4000-8000-000000000001",
                "snapshot_path": str(checkpoint),
                "catalog_hash": "catalog-hash",
                "quantization_recipe": "deepseek_v4_native_fp4_fp8_mixed_v1",
                "layer_count": 43,
                "dense_layer_count": 0,
                "sparse_layer_count": 43,
                "protocol": "ExpertProtocolV2",
                "status": "ready",
                "startup_diagnostic_mode": "request-scheduler-execution",
                "blocker": None,
                "failed_requirements": [],
                "request_scheduler_summary_runtime_reported": True,
                "scheduler_numeric_progression_passed": True,
                "request_numeric_progression_passed": True,
                "scheduler_full_context_device_attention_complete": True,
                "scheduler_terminal_lm_head_sample_passed": True,
                "scheduler_terminal_lm_head_uses_final_decode_device_hidden": True,
                "scheduler_terminal_lm_head_covers_full_vocabulary": True,
                "scheduler_terminal_lm_head_logits_evaluated": 129280,
                "scheduler_terminal_lm_head_vocab_size": 129280,
                "scheduler_sparse_tcp_dispatch_targets": 4,
                "scheduler_sparse_tcp_dispatch_passed": True,
                "scheduler_sparse_tcp_dispatch_all_responses_real_checkpoint_experts": True,
                "scheduler_sparse_tcp_dispatch_all_responses_real_nvfp4": True,
                "scheduler_sparse_tcp_dispatch_consumed_by_residual": True,
                "request_sparse_batches": 43,
                "request_expert_batch_rows": 12,
                "request_expert_batch_routes": 96,
                "request_expert_route_entries_match_source_rows": True,
                "request_kv_reads": 43,
                "request_committed_kv_writes": 43,
                "request_kv_reservation_bytes": 524_800,
                "request_byte_backed_scheduler_trace": False,
                "request_coordinator_graph_captures": 0,
            },
        },
    }


def dspark_group_result(
    checkpoint: Path,
    *,
    completion_tokens: int,
    concurrency: int,
    cycles: int,
) -> dict[str, Any]:
    result = production_result(checkpoint)
    result["usage"] = {"completion_tokens": completion_tokens}
    result["metrics"]["real_full"].update(
        {
            "mtp_verify_cycles": cycles,
            "dspark_joint_cycles": cycles if concurrency > 1 else 0,
            "dspark_max_joint_batch_width": concurrency if cycles else 0,
            "dspark_max_joint_cohort_width": 2 if concurrency == 4 and cycles else 0,
            "dspark_max_joint_cohort_count": 2 if concurrency == 4 and cycles else 0,
            "dspark_2x2_wavefront_cycles": cycles if concurrency == 4 else 0,
        }
    )
    return result


def test_terminal_single_token_group_has_no_dspark_execution_opportunity(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    result = dspark_group_result(
        checkpoint, completion_tokens=1, concurrency=1, cycles=0
    )

    evidence = validate_dspark_group_evidence(
        [result], expected_dspark="on", expected_concurrency=1
    )

    assert evidence["execution_required"] is False
    assert evidence["execution_observed"] is False


def test_eligible_c4_group_requires_joint_2x2_dspark_evidence(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    results = [
        dspark_group_result(
            checkpoint, completion_tokens=2, concurrency=4, cycles=3
        )
        for _ in range(4)
    ]

    evidence = validate_dspark_group_evidence(
        results, expected_dspark="on", expected_concurrency=4
    )
    assert evidence["execution_required"] is True
    assert evidence["joint_observed"] is True
    assert evidence["wavefront_2x2_observed"] is True

    for result in results:
        result["metrics"]["real_full"]["dspark_2x2_wavefront_cycles"] = 0
    with pytest.raises(ValueError, match="2x2 wavefront"):
        validate_dspark_group_evidence(
            results, expected_dspark="on", expected_concurrency=4
        )


def test_runtime_evidence_binds_live_full_scheduler_to_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    evidence = validate_runtime_evidence(
        production_result(checkpoint),
        model="deepseek-ai/DeepSeek-V4-Flash-0731-full",
        checkpoint=checkpoint,
        expected_quantization_recipe="deepseek_v4_native_fp4_fp8_mixed_v1",
        expected_spark_targets=4,
    )

    assert evidence["snapshot_path"] == str(checkpoint)
    assert evidence["service_instance_id"] == (
        "00000000-0000-4000-8000-000000000001"
    )
    assert evidence["quantization_recipe"] == "deepseek_v4_native_fp4_fp8_mixed_v1"
    assert evidence["scheduler_sparse_tcp_dispatch_targets"] == 4
    assert evidence["request_sparse_batches"] == 43


def test_runtime_evidence_requires_live_c4_dspark_2x2_wavefront(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    result = production_result(checkpoint)
    result["metrics"]["real_full"].update(
        {
            "mtp_verify_cycles": 3,
            "mtp_draft_tokens": 12,
            "mtp_accepted_draft_tokens": 8,
            "dspark_joint_cycles": 3,
            "dspark_max_joint_batch_width": 4,
            "dspark_max_joint_cohort_width": 2,
            "dspark_max_joint_cohort_count": 2,
            "dspark_2x2_wavefront_cycles": 3,
        }
    )

    evidence = validate_runtime_evidence(
        result,
        model="deepseek-ai/DeepSeek-V4-Flash-0731-full",
        checkpoint=checkpoint,
        expected_quantization_recipe="deepseek_v4_native_fp4_fp8_mixed_v1",
        expected_spark_targets=4,
        expected_dspark="on",
        expected_concurrency=4,
    )

    assert evidence["dspark_max_joint_batch_width"] == 4
    assert evidence["dspark_max_joint_cohort_width"] == 2
    assert evidence["dspark_max_joint_cohort_count"] == 2
    assert evidence["dspark_2x2_wavefront_cycles"] == 3


def test_runtime_evidence_rejects_serial_execution_claimed_as_c4_dspark(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    result = production_result(checkpoint)
    result["metrics"]["real_full"].update(
        {
            "mtp_verify_cycles": 3,
            "dspark_joint_cycles": 0,
            "dspark_max_joint_batch_width": 1,
            "dspark_max_joint_cohort_width": 1,
            "dspark_max_joint_cohort_count": 1,
            "dspark_2x2_wavefront_cycles": 0,
        }
    )

    with pytest.raises(ValueError, match="jointly execute the expected concurrency"):
        validate_runtime_evidence(
            result,
            model="deepseek-ai/DeepSeek-V4-Flash-0731-full",
            checkpoint=checkpoint,
            expected_quantization_recipe="deepseek_v4_native_fp4_fp8_mixed_v1",
            expected_spark_targets=4,
            expected_dspark="on",
            expected_concurrency=4,
        )


def test_runtime_evidence_requires_no_speculative_work_when_dspark_is_off(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    result = production_result(checkpoint)
    result["metrics"]["real_full"].update(
        {
            "mtp_verify_cycles": 0,
            "dspark_joint_cycles": 0,
        }
    )

    evidence = validate_runtime_evidence(
        result,
        model="deepseek-ai/DeepSeek-V4-Flash-0731-full",
        checkpoint=checkpoint,
        expected_quantization_recipe="deepseek_v4_native_fp4_fp8_mixed_v1",
        expected_spark_targets=4,
        expected_dspark="off",
        expected_concurrency=4,
    )

    assert evidence["mtp_verify_cycles"] == 0
    assert evidence["dspark_joint_cycles"] == 0


def test_request_completion_group_starts_every_member_concurrently(monkeypatch) -> None:
    reached = threading.Barrier(4)
    active = 0
    maximum_active = 0
    lock = threading.Lock()

    def fake_request(url: str, model: str, case, timeout: float) -> dict[str, str]:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        reached.wait(timeout=timeout)
        with lock:
            active -= 1
        return {"case_id": case.case_id}

    monkeypatch.setattr(
        "validate_ds4_flash_generation_ab.request_completion", fake_request
    )
    cases = tuple(CASES[:4])
    results = request_completion_group("http://test", "model", cases, 1.0)

    assert [result["case_id"] for result in results] == [case.case_id for case in cases]
    assert maximum_active == 4


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("startup_diagnostic_mode", "serve-fast-token-embedding-lm-head", "request scheduler"),
        ("scheduler_sparse_tcp_dispatch_targets", 3, "exactly 4 Spark targets"),
        (
            "scheduler_sparse_tcp_dispatch_all_responses_real_checkpoint_experts",
            False,
            "is not true",
        ),
        ("quantization_recipe", "retired-recipe", "quantization recipe differs"),
        ("request_sparse_batches", 0, "no jointly issued sparse expert batch"),
        ("request_kv_reads", 0, "no device-backed KV reads"),
        ("request_committed_kv_writes", 0, "no committed device-backed KV writes"),
        ("request_kv_reservation_bytes", 0, "no device-KV reservation"),
        ("request_coordinator_graph_captures", 1, "performed a coordinator graph capture"),
        ("service_instance_id", "not-a-uuid", "canonical UUIDv4"),
    ),
)
def test_runtime_evidence_rejects_nonproduction_paths(
    tmp_path: Path, field: str, value: Any, message: str
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    result = production_result(checkpoint)
    result["metrics"]["real_full"][field] = value

    with pytest.raises(ValueError, match=message):
        validate_runtime_evidence(
            result,
            model="deepseek-ai/DeepSeek-V4-Flash-0731-full",
            checkpoint=checkpoint,
            expected_quantization_recipe="deepseek_v4_native_fp4_fp8_mixed_v1",
            expected_spark_targets=4,
        )


def test_runtime_evidence_rejects_different_loaded_checkpoint(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    loaded = tmp_path / "loaded"
    expected.mkdir()
    loaded.mkdir()

    with pytest.raises(ValueError, match="differs from checkpoint"):
        validate_runtime_evidence(
            production_result(loaded),
            model="deepseek-ai/DeepSeek-V4-Flash-0731-full",
            checkpoint=expected,
            expected_quantization_recipe="deepseek_v4_native_fp4_fp8_mixed_v1",
            expected_spark_targets=4,
        )


def test_comparison_rejects_unqualified_runtime_samples() -> None:
    native = run_report("native", [("3874973", True)])
    candidate = run_report("exl3", [("3874973", True)])
    candidate["records"][0].pop("runtime_evidence")

    with pytest.raises(ValueError, match="unqualified runtime samples"):
        compare_runs(
            native,
            candidate,
            min_candidate_pass_rate=1.0,
            max_quality_regressions=0,
        )
