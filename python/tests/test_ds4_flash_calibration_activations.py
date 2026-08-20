from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import collect_ds4_flash_calibration_activations as capture  # noqa: E402


def write_checkpoint(path: Path, *, hidden_size: int = 4) -> Path:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": hidden_size,
                "num_hidden_layers": 43,
                "dspark_target_layer_ids": [11, 22, 33],
                "n_routed_experts": 256,
                "num_experts_per_tok": 6,
                "routed_scaling_factor": 1.5,
            }
        ),
        encoding="utf-8",
    )
    return path


def write_routed_capture(
    dump_dir: Path,
    *,
    layer_id: int,
    rows: int = 2,
    hidden_size: int = 4,
    routed_experts: int = 256,
) -> None:
    base = (
        f"capture_0000000001_{layer_id:012d}_layer_{layer_id:02}_"
        f"rows_{rows}_expert_input"
    )
    (dump_dir / f"{base}.bf16").write_bytes(
        bytes([layer_id % 256]) * (rows * hidden_size * 2)
    )
    route_payload = bytearray()
    for row in range(rows):
        for route in range(6):
            expert_id = (layer_id * 7 + row * 11 + route) % routed_experts
            route_payload.extend(struct.pack("<Hf", expert_id, 1.5 / 6.0))
    (dump_dir / f"{base}_routes_u16_f32.bin").write_bytes(route_payload)
    (dump_dir / f"{base}_positions_u64.bin").write_bytes(
        b"".join(struct.pack("<Q", 100 + row) for row in range(rows))
    )


def test_load_corpus_binds_raw_bytes_and_validates_records(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    raw = (
        '{"id":"prose","prompt":"A sufficiently long prompt.","max_tokens":4}\n'
        '{"id":"code","prompt":"Explain this code.","max_tokens":8}\n'
    ).encode()
    corpus.write_bytes(raw)

    records, digest = capture.load_corpus(corpus)

    assert [record["id"] for record in records] == ["prose", "code"]
    assert digest == hashlib.sha256(raw).hexdigest()

    corpus.write_text(
        '{"id":"duplicate","prompt":"one"}\n'
        '{"id":"duplicate","prompt":"two"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate corpus id"):
        capture.load_corpus(corpus)


def test_move_capture_files_checks_bf16_geometry_and_hashes(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    payload = bytes(range(16))
    captured = source / "layer_07_rows_2_expert_input.bf16"
    captured.write_bytes(payload)
    (source / "unrelated.txt").write_text("leave me", encoding="utf-8")

    records = capture.move_capture_files(
        source=source,
        destination=tmp_path / "saved",
        hidden_size=4,
    )

    assert records == [
        {
            "path": str(tmp_path / "saved" / captured.name),
            "layer_id": 7,
            "rows": 2,
            "hidden_size": 4,
            "bytes": 16,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]
    assert not captured.exists()
    assert (source / "unrelated.txt").is_file()

    (source / "layer_08_rows_2_expert_input.bf16").write_bytes(b"short")
    with pytest.raises(ValueError, match="expected 16"):
        capture.move_capture_files(
            source=source,
            destination=tmp_path / "invalid",
            hidden_size=4,
        )


def test_move_capture_files_binds_exact_route_sidecar(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    write_routed_capture(source, layer_id=7)

    records = capture.move_capture_files(
        source=source,
        destination=tmp_path / "saved",
        hidden_size=4,
        routed_experts=256,
        expected_top_k=6,
        expected_gate_sum=1.5,
        require_routes=True,
    )

    assert len(records) == 1
    record = records[0]
    assert record["capture_id"] == "0000000001_000000000007"
    assert record["routes_per_row"] == 6
    assert record["route_bytes"] == 2 * 6 * struct.calcsize("<Hf")
    assert record["route_record_format"] == capture.ROUTE_RECORD_FORMAT
    assert Path(record["route_path"]).is_file()
    assert record["position_record_format"] == capture.POSITION_RECORD_FORMAT
    assert struct.unpack(
        "<QQ", Path(record["position_path"]).read_bytes()
    ) == (100, 101)


def test_compaction_counts_all_routes_but_keeps_only_quota_rows(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    write_routed_capture(
        source,
        layer_id=0,
        rows=4,
        routed_experts=6,
    )
    retained_counts = [[0] * 6]

    records, observed = capture.compact_routed_capture_files(
        source=source,
        destination=tmp_path / "saved",
        hidden_size=4,
        layer_count=1,
        routed_experts=6,
        expected_top_k=6,
        expected_gate_sum=1.5,
        retained_routes_per_expert=1,
        retained_route_counts=retained_counts,
        record_root=tmp_path,
    )

    assert observed == [
        {
            "layer_id": 0,
            "rows": 4,
            "routes": 24,
            "expert_route_counts": [4] * 6,
        }
    ]
    assert retained_counts == [[1] * 6]
    assert len(records) == 1
    assert records[0]["source_rows"] == 4
    assert records[0]["rows"] == 1
    assert records[0]["route_bytes"] == 6 * struct.calcsize("<Hf")
    assert Path(tmp_path / records[0]["position_path"]).read_bytes() == struct.pack(
        "<Q", 100
    )
    assert not any(source.iterdir())


def test_compaction_uses_daemon_observed_counts_for_prefiltered_rows(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    write_routed_capture(
        source,
        layer_id=0,
        rows=1,
        routed_experts=6,
    )
    observed_path = (
        source
        / "capture_0000000001_000000000000_layer_00_rows_4_"
        "observed_routes_u64.bin"
    )
    observed_path.write_bytes(b"".join(struct.pack("<Q", 4) for _ in range(6)))
    retained_counts = [[0] * 6]

    records, observed = capture.compact_routed_capture_files(
        source=source,
        destination=tmp_path / "saved",
        hidden_size=4,
        layer_count=1,
        routed_experts=6,
        expected_top_k=6,
        expected_gate_sum=1.5,
        retained_routes_per_expert=1,
        retained_route_counts=retained_counts,
        record_root=tmp_path,
    )

    assert observed[0]["rows"] == 4
    assert observed[0]["expert_route_counts"] == [4] * 6
    assert records[0]["rows"] == 1
    assert records[0]["source_rows"] == 4
    assert not any(source.iterdir())


def test_retention_control_binds_quota_and_exact_progress_counts(
    tmp_path: Path,
) -> None:
    path = tmp_path / capture.RETENTION_CONTROL_FILE
    counts = [[1, 2, 3], [4, 5, 6]]

    capture.write_retention_control(
        path=path,
        layer_count=2,
        routed_experts=3,
        target=1024,
        route_counts=counts,
    )

    payload = path.read_bytes()
    assert payload[:8] == capture.RETENTION_CONTROL_MAGIC
    assert struct.unpack_from("<IIQ", payload, 8) == (2, 3, 1024)
    assert [
        value[0] for value in struct.iter_unpack("<Q", payload[24:])
    ] == [1, 2, 3, 4, 5, 6]


def test_collect_requires_live_native_evidence_and_complete_layer_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = write_checkpoint(tmp_path / "checkpoint")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"id":"mixed","prompt":"A mixed calibration prompt.","max_tokens":4}\n',
        encoding="utf-8",
    )
    dump_dir = tmp_path / "raw"
    dump_dir.mkdir()
    write_routed_capture(dump_dir, layer_id=0, rows=1)
    (
        dump_dir
        / "capture_0000000001_000000000000_layer_00_rows_1_"
        "observed_routes_u64.bin"
    ).write_bytes(
        b"".join(struct.pack("<Q", int(expert_id < 6)) for expert_id in range(256))
    )
    output = tmp_path / "capture"

    def fake_request_completion(**_: object) -> dict[str, object]:
        for layer_id in range(43):
            write_routed_capture(dump_dir, layer_id=layer_id)
        return {
            "choices": [{"message": {"content": "captured"}}],
            "usage": {"completion_tokens": 4},
        }

    evidence_calls: list[dict[str, object]] = []

    def fake_validate_runtime_evidence(
        _result: object, **kwargs: object
    ) -> dict[str, object]:
        evidence_calls.append(kwargs)
        return {"mtp_verify_cycles": 1}

    monkeypatch.setattr(capture, "request_completion", fake_request_completion)
    monkeypatch.setattr(
        capture, "validate_runtime_evidence", fake_validate_runtime_evidence
    )

    manifest = capture.collect(
        argparse.Namespace(
            corpus=corpus,
            dump_dir=dump_dir,
            output=output,
            checkpoint=checkpoint,
            url="http://127.0.0.1:8000/v1/chat/completions",
            model="deepseek-ai/DeepSeek-V4-Flash-0731-full",
            timeout=1.0,
        )
    )

    assert manifest["summary"]["prompts"] == 1
    assert manifest["layer_count"] == 43
    assert manifest["dspark_layer_count"] == 3
    assert manifest["summary"]["capture_files"] == 43
    assert manifest["summary"]["captured_rows"] == 86
    assert manifest["summary"]["routed_rows"] == 86
    assert manifest["summary"]["route_records"] == 516
    assert manifest["route_distribution"][7]["routes"] == 12
    assert manifest["summary"]["covered_layers"] == list(range(43))
    assert all(
        not Path(record["path"]).is_absolute()
        for record in manifest["prompts"][0]["capture_files"]
    )
    assert evidence_calls[0]["expected_quantization_recipe"] == capture.NATIVE_RECIPE
    assert evidence_calls[0]["expected_dspark"] == "on"
    assert evidence_calls[0]["require_dspark_execution"] is False
    assert json.loads((output / "manifest.json").read_text()) == manifest
    progress = json.loads((output / "progress.json").read_text())
    assert progress["schema"] == capture.PROGRESS_SCHEMA
    assert progress["prompts"] == manifest["prompts"]
    assert manifest["startup_files"] == []
    assert any((output / "uncommitted").iterdir())


def test_collect_resumes_from_fsynced_prompt_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = write_checkpoint(tmp_path / "checkpoint")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"id":"first","prompt":"first prompt","max_tokens":4}\n'
        '{"id":"second","prompt":"second prompt","max_tokens":4}\n',
        encoding="utf-8",
    )
    dump_dir = tmp_path / "raw"
    dump_dir.mkdir()
    output = tmp_path / "capture"
    request_count = 0

    def fake_request_completion(**_: object) -> dict[str, object]:
        nonlocal request_count
        request_count += 1
        if request_count == 2:
            raise OSError("injected request failure")
        for layer_id in range(43):
            write_routed_capture(dump_dir, layer_id=layer_id)
        return {
            "choices": [{"message": {"content": "captured"}}],
            "usage": {"completion_tokens": 4},
        }

    monkeypatch.setattr(capture, "request_completion", fake_request_completion)
    monkeypatch.setattr(
        capture,
        "validate_runtime_evidence",
        lambda *_args, **_kwargs: {"mtp_verify_cycles": 1},
    )
    args = argparse.Namespace(
        corpus=corpus,
        dump_dir=dump_dir,
        output=output,
        checkpoint=checkpoint,
        url="http://127.0.0.1:8000/v1/chat/completions",
        model="deepseek-ai/DeepSeek-V4-Flash-0731-full",
        timeout=1.0,
        retained_routes_per_expert=1,
        minimum_natural_routes_per_expert=0,
        resume=False,
    )
    with pytest.raises(OSError, match="injected request failure"):
        capture.collect(args)
    progress = json.loads((output / "progress.json").read_text())
    assert [record["id"] for record in progress["prompts"]] == ["first"]

    args.resume = True
    manifest = capture.collect(args)

    assert request_count == 3
    assert [record["id"] for record in manifest["prompts"]] == ["first", "second"]
    assert manifest["summary"]["captured_rows"] == 86
    assert manifest["summary"]["routed_rows"] == 172


def test_collect_resume_accepts_only_a_matching_appended_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = write_checkpoint(tmp_path / "checkpoint")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"id":"first","prompt":"first prompt","max_tokens":4}\n'
        '{"id":"second","prompt":"second prompt","max_tokens":4}\n',
        encoding="utf-8",
    )
    dump_dir = tmp_path / "raw"
    dump_dir.mkdir()
    output = tmp_path / "capture"
    request_count = 0

    def fake_request_completion(**_: object) -> dict[str, object]:
        nonlocal request_count
        request_count += 1
        if request_count == 2:
            raise OSError("injected request failure")
        for layer_id in range(43):
            write_routed_capture(dump_dir, layer_id=layer_id)
        return {
            "choices": [{"message": {"content": "captured"}}],
            "usage": {"completion_tokens": 4},
        }

    monkeypatch.setattr(capture, "request_completion", fake_request_completion)
    monkeypatch.setattr(
        capture,
        "validate_runtime_evidence",
        lambda *_args, **_kwargs: {"mtp_verify_cycles": 1},
    )
    args = argparse.Namespace(
        corpus=corpus,
        dump_dir=dump_dir,
        output=output,
        checkpoint=checkpoint,
        url="http://127.0.0.1:8000/v1/chat/completions",
        model="deepseek-ai/DeepSeek-V4-Flash-0731-full",
        timeout=1.0,
        retained_routes_per_expert=1,
        minimum_natural_routes_per_expert=0,
        resume=False,
        allow_corpus_extension=False,
    )
    with pytest.raises(OSError, match="injected request failure"):
        capture.collect(args)

    corpus.write_text(
        corpus.read_text(encoding="utf-8")
        + '{"id":"third","prompt":"third prompt","max_tokens":4}\n',
        encoding="utf-8",
    )
    args.resume = True
    with pytest.raises(ValueError, match="allow-corpus-extension"):
        capture.collect(args)

    args.allow_corpus_extension = True
    manifest = capture.collect(args)

    assert [record["id"] for record in manifest["prompts"]] == [
        "first",
        "second",
        "third",
    ]
    assert len(manifest["corpus_history_sha256"]) == 2
    assert manifest["corpus_history_sha256"][-1] == manifest["corpus_sha256"]


def test_collect_rejects_output_inside_live_dump_directory(tmp_path: Path) -> None:
    checkpoint = write_checkpoint(tmp_path / "checkpoint")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"id":"one","prompt":"prompt"}\n', encoding="utf-8")
    dump_dir = tmp_path / "raw"
    dump_dir.mkdir()

    with pytest.raises(ValueError, match="must not be inside"):
        capture.collect(
            argparse.Namespace(
                corpus=corpus,
                dump_dir=dump_dir,
                output=dump_dir / "capture",
                checkpoint=checkpoint,
                url="http://127.0.0.1:8000/v1/chat/completions",
                model="deepseek-ai/DeepSeek-V4-Flash-0731-full",
                timeout=1.0,
            )
        )
