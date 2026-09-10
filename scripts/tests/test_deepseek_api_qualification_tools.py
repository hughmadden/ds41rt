from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import pytest
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "python" / "tools"
sys.path.insert(0, str(TOOLS))

import bench_real_full_buffer_lifecycle as buffer_lifecycle  # noqa: E402
import bench_real_full_concurrency as concurrency  # noqa: E402
import bench_real_full_exact_decode as exact_decode  # noqa: E402
import bench_real_full_long_context_session as long_context  # noqa: E402
import bench_real_full_mtp_acceptance as mtp_acceptance  # noqa: E402
import bench_real_full_prefill_concurrency as prefill_concurrency  # noqa: E402
import deepseek_v4_benchmark as contract  # noqa: E402
import real_full_matrix as matrix  # noqa: E402


def test_flash_tokenizer_discovery_honors_selected_hf_revision(
    tmp_path: Path, monkeypatch
) -> None:
    model_root = tmp_path / "hub" / contract.FLASH_HF_CACHE_NAME
    selected = model_root / "snapshots" / "selected" / "tokenizer.json"
    selected.parent.mkdir(parents=True)
    selected.write_text("{}\n", encoding="utf-8")
    (model_root / "refs").mkdir()
    (model_root / "refs" / "main").write_text("selected\n", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)

    assert contract.default_flash_tokenizer_path() == selected


def test_simple_user_prompt_matches_deepseek_nonthinking_protocol() -> None:
    assert contract.render_simple_user_prompt("hello") == (
        "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>"
    )


def test_matrix_renderer_matches_deepseek_role_and_user_merge_protocol() -> None:
    assert matrix.render_messages(
        [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "hello"},
        ]
    ) == (
        "<｜begin▁of▁sentence｜>Be terse.<｜User｜>hello"
        "<｜Assistant｜></think>"
    )
    assert matrix.render_messages(
        [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
    ) == (
        "<｜begin▁of▁sentence｜><｜User｜>first\n\nsecond"
        "<｜Assistant｜></think>"
    )
    with pytest.raises(ValueError, match="does not support role"):
        matrix.render_messages([{"role": "tool", "content": "no"}])


def test_long_context_renderer_matches_deepseek_conversation_transitions() -> None:
    messages = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "reasoning_content": "not retained in non-thinking mode",
            "content": "Hi.",
        },
        {"role": "user", "content": "Continue."},
    ]
    expected = (
        "<｜begin▁of▁sentence｜>Be terse.<｜User｜>Hello"
        "<｜Assistant｜></think>Hi.<｜end▁of▁sentence｜>"
        "<｜User｜>Continue.<｜Assistant｜></think>"
    )
    assert long_context.render_messages(messages) == expected
    assert contract.render_nonthinking_messages(messages) == expected
    assert long_context.MODEL_ID == contract.DEFAULT_FLASH_FULL_MODEL_ID


def test_concurrency_payload_is_flash_nonthinking() -> None:
    payload = json.loads(
        concurrency.payload(
            contract.DEFAULT_FLASH_MODEL_ID,
            concurrency.FIXTURES["exact-50"],
        )
    )
    assert payload["model"] == contract.DEFAULT_FLASH_MODEL_ID
    assert payload["enable_thinking"] is False
    assert payload["max_tokens"] == 99


def test_exact_count_fixtures_have_deepseek_visible_output_contracts() -> None:
    count_13 = exact_decode.FIXTURES["count-13"]
    count_50 = exact_decode.FIXTURES["count-50"]
    assert count_13.max_tokens == 25
    assert exact_decode.expected_content(count_13).splitlines()[-1] == "13"
    assert count_50.max_tokens == 99
    assert exact_decode.expected_content(count_50).splitlines()[-1] == "50"
    assert exact_decode.prompt_content(count_50).count("beta ") == 946


def test_released_flash_tokenizer_exact_count_contract_if_available(
    tmp_path: Path,
) -> None:
    try:
        tokenizer_path = contract.default_flash_tokenizer_path()
    except FileNotFoundError:
        pytest.skip("released Flash tokenizer is not installed")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    prompt_counts = []
    completion_counts = []
    for fixture in exact_decode.FIXTURES.values():
        assert matrix.render_messages(
            [{"role": "user", "content": exact_decode.prompt_content(fixture)}]
        ) == contract.render_simple_user_prompt(
            exact_decode.prompt_content(fixture)
        )
        prompt_counts.append(
            len(
                tokenizer.encode(
                    contract.render_simple_user_prompt(
                        exact_decode.prompt_content(fixture)
                    ),
                    add_special_tokens=False,
                ).ids
            )
        )
        completion_counts.append(
            len(
                tokenizer.encode(
                    exact_decode.expected_content(fixture),
                    add_special_tokens=False,
                ).ids
            )
        )
    assert prompt_counts == [792, 977]
    assert completion_counts == [25, 99]

    snapshot_text = matrix.render_messages(
        [{"role": "user", "content": "snapshot prefix"}]
    )
    snapshot_ids = tokenizer.encode(
        snapshot_text, add_special_tokens=False
    ).ids
    payload = struct.pack(f"<{len(snapshot_ids)}I", *snapshot_ids)
    (tmp_path / "token-ids.u32le").write_bytes(payload)
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "format": "ds41rt-kv-v3",
                "producer_profile": {"cache_semantics_revision": 3},
                "token_count": len(snapshot_ids),
                "token_ids_file": "token-ids.u32le",
                "token_ids_sha256": hashlib.sha256(payload).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    loaded_ids, loaded_text, _ = matrix.load_snapshot(tmp_path, tokenizer)
    assert loaded_ids == snapshot_ids
    assert loaded_text == snapshot_text

    long_messages = [
        {
            "role": "system",
            "content": long_context.system_message("test-session", "test-key"),
        }
    ]
    corpus_ids = tokenizer.encode(
        "quoted source line\n" * 1_000, add_special_tokens=False
    ).ids
    source_message, source_end, planned_tokens = (
        long_context.append_source_to_checkpoint(
            tokenizer=tokenizer,
            messages=long_messages,
            corpus_ids=corpus_ids,
            source_start=0,
            checkpoint=256,
            probe_kind="local",
        )
    )
    assert 0 < source_end < len(corpus_ids)
    assert source_message["role"] == "user"
    assert planned_tokens == len(
        long_context.prompt_token_ids(
            tokenizer, [*long_messages, source_message]
        )
    )
    assert planned_tokens <= 256


def test_buffer_lifecycle_defaults_to_flash_full_backend(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["bench_real_full_buffer_lifecycle.py"])
    assert (
        buffer_lifecycle.parse_args().model
        == contract.DEFAULT_FLASH_FULL_MODEL_ID
    )


def test_mtp_acceptance_reports_accuracy_by_selected_proposal_depth() -> None:
    assert mtp_acceptance.positional_acceptance(
        [1, 3, 2, 4],
        [1, 2, 0, 4],
    ) == {
        1: {"attempts": 4, "accepted": 3, "acceptance_rate": 0.75},
        2: {"attempts": 3, "accepted": 2, "acceptance_rate": 2 / 3},
        3: {"attempts": 2, "accepted": 1, "acceptance_rate": 0.5},
        4: {"attempts": 1, "accepted": 1, "acceptance_rate": 1.0},
    }


def test_mtp_acceptance_rejects_impossible_lengths() -> None:
    with pytest.raises(ValueError, match="invalid speculative lengths"):
        mtp_acceptance.positional_acceptance([2], [3])


def test_mtp_acceptance_reports_complete_cycle_latency_by_physical_m() -> None:
    assert mtp_acceptance.cycle_latency_by_physical_m(
        [1, 3, 1, 2],
        [12.0, 24.0, 16.0, 18.0],
    ) == {
        2: {
            "samples": 2,
            "mean_ms": 14.0,
            "median_ms": 14.0,
            "p95_ms": 16.0,
            "min_ms": 12.0,
            "max_ms": 16.0,
        },
        3: {
            "samples": 1,
            "mean_ms": 18.0,
            "median_ms": 18.0,
            "p95_ms": 18.0,
            "min_ms": 18.0,
            "max_ms": 18.0,
        },
        4: {
            "samples": 1,
            "mean_ms": 24.0,
            "median_ms": 24.0,
            "p95_ms": 24.0,
            "min_ms": 24.0,
            "max_ms": 24.0,
        },
    }


def test_mtp_acceptance_rejects_invalid_cycle_latency() -> None:
    with pytest.raises(ValueError, match="equal length"):
        mtp_acceptance.cycle_latency_by_physical_m([1], [])
    with pytest.raises(ValueError, match="invalid speculative cycle"):
        mtp_acceptance.cycle_latency_by_physical_m([1], [0.0])


def test_prefill_concurrency_uses_flash_pages_and_model(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_real_full_prefill_concurrency.py",
            "--source",
            str(tmp_path / "source.txt"),
            "--source-tokens",
            "1",
            "--concurrency",
            "1",
            "--output",
            str(tmp_path / "result.jsonl"),
        ],
    )
    args = prefill_concurrency.parse_args()
    assert args.model == contract.DEFAULT_FLASH_FULL_MODEL_ID
    assert args.page_tokens == 256
