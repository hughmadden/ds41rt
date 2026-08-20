from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_ds4_flash_calibration_corpus as corpus  # noqa: E402


class Encoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> Encoding:
        assert add_special_tokens is False
        return Encoding([ord(character) for character in text])

    def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is False
        return "".join(chr(value) for value in ids)


def group_for_split(prefix: str, split: str, *, seed: int, index: int) -> str:
    candidate = index
    while True:
        group = f"{prefix}:{candidate}"
        if corpus.split_for_group(group, seed=seed) == split:
            return group
        candidate += 1


def test_group_split_is_stable_and_source_disjoint() -> None:
    seed = 17
    groups = [f"source:{index}" for index in range(100)]
    first = {group: corpus.split_for_group(group, seed=seed) for group in groups}
    second = {group: corpus.split_for_group(group, seed=seed) for group in groups}

    assert first == second
    assert set(first.values()) == {"calibration", "heldout"}
    assert not (
        {group for group, split in first.items() if split == "calibration"}
        & {group for group, split in first.items() if split == "heldout"}
    )


def test_wiki_selection_focuses_english_and_chinese(tmp_path: Path) -> None:
    seed = 23
    tokenizer = CharacterTokenizer()
    records: list[corpus.SourceRecord] = []
    languages = ("en", "zh", "ja", "de")
    for language_index, language in enumerate(languages):
        for index in range(5):
            article = tmp_path / f"{language}-{index}.md"
            article.write_text((language + " article body ") * 100, encoding="utf-8")
            group = group_for_split(
                f"wiki:{language}",
                "calibration",
                seed=seed,
                index=language_index * 100 + index,
            )
            records.append(
                corpus.SourceRecord(
                    domain="wiki",
                    family=("short", "paragraph", "relate", "mc")[index % 4],
                    stratum=(("short", "paragraph", "relate", "mc")[index % 4], language),
                    group=group,
                    identity=corpus.stable_digest(group, index),
                    prompt=f"Answer this {language} question {index}.",
                    metadata={"language": language, "topic": "test", "article_path": article.name},
                    article_path=article,
                )
            )

    selected = corpus.select_wiki_records(
        records,
        split="calibration",
        token_budget=1_600,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        seed=seed,
        target_tokens=400,
        min_tokens=300,
        max_tokens=500,
    )

    selected_languages = [item.source.metadata["language"] for item in selected]
    assert len(selected) == 4
    assert selected_languages.count("en") == 2
    assert selected_languages.count("zh") == 1
    assert len(set(selected_languages) - {"en", "zh"}) == 1
    assert len({item.source.group for item in selected}) == len(selected)
    assert all(300 <= item.prompt_tokens <= 500 for item in selected)


def test_code_selection_prioritizes_major_languages_without_answers() -> None:
    seed = 29
    tokenizer = CharacterTokenizer()
    records: list[corpus.SourceRecord] = []
    for index, language in enumerate(corpus.CODE_PRIORITY_LANGUAGES):
        group = group_for_split(
            f"code:{language}", "calibration", seed=seed, index=index * 100
        )
        records.append(
            corpus.SourceRecord(
                domain="code",
                family=("review", "rewrite", "ablation")[index % 3],
                stratum=(("review", "rewrite", "ablation")[index % 3], language),
                group=group,
                identity=corpus.stable_digest(group, index),
                prompt=(f"Review this {language} source. " * 18),
                metadata={
                    "repo": group.removeprefix("code:"),
                    "language": language,
                    "file": f"file-{index}",
                },
            )
        )

    selected = corpus.select_code_records(
        records,
        split="calibration",
        token_budget=2_000,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        seed=seed,
        target_tokens=400,
        min_tokens=300,
        max_tokens=768,
    )
    payload, provenance = corpus.jsonl_bytes(
        selected,
        split="calibration",
        max_output_tokens=8,
        tokenizer=tokenizer,  # type: ignore[arg-type]
    )

    assert {item.source.metadata["language"] for item in selected} == set(
        corpus.CODE_PRIORITY_LANGUAGES
    )
    decoded = [json.loads(line) for line in payload.decode().splitlines()]
    assert all(set(record) == {"id", "max_tokens", "prompt"} for record in decoded)
    assert all("answer" not in record for record in decoded)
    assert [record["prompt_sha256"] for record in provenance] == [
        hashlib.sha256(record["prompt"].encode()).hexdigest() for record in decoded
    ]
    assert all(len(record["token_ids_sha256"]) == 64 for record in provenance)


def test_code_prompt_keeps_agentic_header_and_bounded_real_code_window() -> None:
    tokenizer = CharacterTokenizer()
    source = corpus.SourceRecord(
        domain="code",
        family="review",
        stratum=("review", "Python"),
        group="code:example/repo",
        identity="identity",
        prompt=(
            "Review this implementation carefully.\n\n"
            "```python\n"
            + "value = expensive_call()\n" * 200
            + "```\n"
        ),
        metadata={"repo": "example/repo", "language": "Python", "file": "x.py"},
    )

    prompt, count = corpus.code_prompt(
        source,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        target_tokens=400,
        max_tokens=500,
    )

    assert "Review this implementation carefully." in prompt
    assert "value = expensive_call()" in prompt
    assert prompt.endswith("```\n")
    assert 350 <= count <= 500


def test_loads_math_and_structured_banks_with_stable_groups(tmp_path: Path) -> None:
    math_path = tmp_path / "banks" / "math" / "solve" / "math.jsonl"
    math_path.parent.mkdir(parents=True)
    math_path.write_text(
        json.dumps(
            {
                "type": "solve",
                "language": "en",
                "topic": "math",
                "problem_id": "problem-7",
                "prompt": "problem and worked solution",
                "source": "numinamath-cot",
                "subset": "olympiads",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    structured_path = (
        tmp_path / "banks" / "structured" / "tool_call" / "xlam.jsonl"
    )
    structured_path.parent.mkdir(parents=True)
    structured_path.write_text(
        json.dumps(
            {
                "type": "tool_call",
                "language": "en",
                "sample_id": 42,
                "schema_id": "schema-9",
                "prompt": "schema, query, and expected tool call",
                "source": "xlam-function-calling-60k",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = corpus.load_source_records(tmp_path)

    assert [(record.domain, record.group) for record in records] == [
        ("math", "math:problem-7"),
        ("structured", "struct:42"),
    ]
    assert records[0].stratum == ("numinamath-cot", "olympiads", "en")
    assert records[1].metadata["schema_id"] == "schema-9"


def test_math_selection_reserves_chinese_token_slice() -> None:
    seed = 31
    tokenizer = CharacterTokenizer()
    records: list[corpus.SourceRecord] = []
    for language, count in (("zh", 3), ("en", 12)):
        for index in range(count):
            group = group_for_split(
                f"math:{language}", "calibration", seed=seed, index=index
            )
            records.append(
                corpus.SourceRecord(
                    domain="math",
                    family="solve",
                    stratum=("source", language, str(index % 3)),
                    group=group,
                    identity=corpus.stable_digest(group, index),
                    prompt=(f"{language} worked problem {index}. " * 18),
                    metadata={"language": language, "topic": "math"},
                )
            )

    selected = corpus.select_math_records(
        records,
        split="calibration",
        token_budget=2_000,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        seed=seed,
        target_tokens=400,
        min_tokens=300,
        max_tokens=500,
    )

    chinese = [item for item in selected if item.source.metadata["language"] == "zh"]
    assert chinese
    assert sum(item.prompt_tokens for item in chinese) >= 400
    assert {item.axis for item in selected} == {"math_reasoning"}


def test_math_selection_fills_when_chinese_split_is_short() -> None:
    seed = 37
    tokenizer = CharacterTokenizer()
    records: list[corpus.SourceRecord] = []
    for language, count in (("zh", 1), ("en", 12)):
        for index in range(count):
            group = group_for_split(
                f"short-math:{language}", "calibration", seed=seed, index=index
            )
            records.append(
                corpus.SourceRecord(
                    domain="math",
                    family="solve",
                    stratum=("source", language, str(index % 3)),
                    group=group,
                    identity=corpus.stable_digest(group, index),
                    prompt=(f"{language} worked problem {index}. " * 18),
                    metadata={"language": language, "topic": "math"},
                )
            )

    selected = corpus.select_math_records(
        records,
        split="calibration",
        token_budget=2_000,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        seed=seed,
        target_tokens=400,
        min_tokens=300,
        max_tokens=500,
    )

    assert any(item.source.metadata["language"] == "zh" for item in selected)
    assert sum(item.prompt_tokens for item in selected) >= 2_000


def test_screening_selection_is_axis_stratified_calibration_subset() -> None:
    calibration: list[corpus.PreparedRecord] = []
    for axis in corpus.AXIS_WEIGHTS:
        for index in range(20):
            source = corpus.SourceRecord(
                domain="test",
                family="test",
                stratum=(axis,),
                group=f"{axis}:{index}",
                identity=corpus.stable_digest(axis, index),
                prompt=f"{axis} {index}",
                metadata={"language": "en"},
            )
            calibration.append(
                corpus.PreparedRecord(
                    source=source,
                    prompt=source.prompt,
                    prompt_tokens=100,
                    axis=axis,
                )
            )

    selected = corpus.select_screening_records(calibration, seed=41)

    assert len(selected) == len(corpus.AXIS_WEIGHTS) * 2
    assert {record.axis for record in selected} == set(corpus.AXIS_WEIGHTS)
    assert {record.source.identity for record in selected} <= {
        record.source.identity for record in calibration
    }
