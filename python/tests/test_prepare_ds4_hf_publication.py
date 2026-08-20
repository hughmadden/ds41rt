from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from prepare_ds4_hf_publication import prepare_publication  # noqa: E402
from stage_ds4_hf_snapshot import model_cache_dir, stage_snapshot  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def make_snapshot(root: Path) -> Path:
    root.mkdir()
    quantization = {
        "bits": 2.0,
        "codebook": "mcg",
        "method": "exl3",
        "quant_method": "exl3",
        "meta": {
            "fallback": None,
            "ds4rt_error_ledger": {"path": "/internal/error-ledger.jsonl"},
        },
    }
    write_json(root / "config.json", {"quantization_config": quantization})
    write_json(
        root / "quantize_config.json",
        {**quantization, "tensor_storage": {"model.layers.0.expert": {}}},
    )
    shards = (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )
    for index, name in enumerate(shards):
        (root / name).write_bytes(f"shard-{index}".encode())
    write_json(
        root / "model.safetensors.index.json",
        {
            "metadata": {"total_size": 14},
            "weight_map": {"a": shards[0], "b": shards[1]},
        },
    )
    for name in (
        ".gitattributes",
        "LICENSE",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        (root / name).write_text(f"{name}\n", encoding="utf-8")
    (root / "ds4rt-gptqmodel-plan.json").write_text("internal\n", encoding="utf-8")
    return root


def make_readme(path: Path) -> Path:
    path.write_text("---\nlicense: mit\n---\n\n# Test\n", encoding="utf-8")
    return path


def test_prepare_publication_has_only_standard_files_and_reuses_blobs(
    tmp_path: Path,
) -> None:
    snapshot = make_snapshot(tmp_path / "snapshot")
    readme = make_readme(tmp_path / "README.md")
    output = tmp_path / "public"

    report = prepare_publication(snapshot, readme, output)

    assert report["schema"] == "ds4rt-hf-standard-publication-v1"
    assert report["files"] == 11
    assert report["shards"] == 2
    assert set(report["names"]) == {
        ".gitattributes",
        "LICENSE",
        "README.md",
        "config.json",
        "generation_config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "quantize_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert "ds4rt-gptqmodel-plan.json" not in report["names"]
    assert os.stat(snapshot / "model-00001-of-00002.safetensors").st_ino == os.stat(
        output / "model-00001-of-00002.safetensors"
    ).st_ino
    assert os.stat(readme).st_ino != os.stat(output / "README.md").st_ino


def test_standard_publication_can_be_staged_without_private_evidence(
    tmp_path: Path,
) -> None:
    snapshot = make_snapshot(tmp_path / "snapshot")
    output = tmp_path / "public"
    prepare_publication(snapshot, make_readme(tmp_path / "README.md"), output)

    hf_home = tmp_path / "hf"
    result = stage_snapshot(
        output,
        "tpurtell/flash-standard-exl3",
        hf_home,
        standard_publication=True,
    )

    model_root = model_cache_dir(hf_home.resolve(), "tpurtell/flash-standard-exl3")
    staged = model_root / "snapshots" / result["revision"]
    assert result["qualification"] == []
    assert result["files"] == 11
    assert (staged / "README.md").is_symlink()
    assert not (staged / "ds4rt-gptqmodel-plan.json").exists()


def test_prepare_publication_rejects_unindexed_shard_without_output(
    tmp_path: Path,
) -> None:
    snapshot = make_snapshot(tmp_path / "snapshot")
    (snapshot / "model-00003-of-00003.safetensors").write_bytes(b"extra")
    output = tmp_path / "public"

    with pytest.raises(ValueError, match="shard set differs"):
        prepare_publication(snapshot, make_readme(tmp_path / "README.md"), output)

    assert not output.exists()


def test_prepare_publication_compacts_exact_embedded_storage_map(tmp_path: Path) -> None:
    snapshot = make_snapshot(tmp_path / "snapshot")
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    external = json.loads((snapshot / "quantize_config.json").read_text(encoding="utf-8"))
    config["quantization_config"] = external
    write_json(snapshot / "config.json", config)

    output = tmp_path / "public"
    prepare_publication(
        snapshot,
        make_readme(tmp_path / "README.md"),
        output,
    )

    published = json.loads((output / "config.json").read_text(encoding="utf-8"))
    published_external = json.loads(
        (output / "quantize_config.json").read_text(encoding="utf-8")
    )
    assert "tensor_storage" not in published["quantization_config"]
    assert "ds4rt_error_ledger" not in published["quantization_config"]["meta"]
    assert "ds4rt_error_ledger" not in published_external["meta"]
    assert published["quantization_config"] == {
        key: value
        for key, value in published_external.items()
        if key != "tensor_storage"
    }


def test_prepare_publication_rejects_mismatched_embedded_storage_map(
    tmp_path: Path,
) -> None:
    snapshot = make_snapshot(tmp_path / "snapshot")
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    external = json.loads((snapshot / "quantize_config.json").read_text(encoding="utf-8"))
    config["quantization_config"] = {**external, "bits": 3.0}
    write_json(snapshot / "config.json", config)

    with pytest.raises(ValueError, match="embedded full EXL3 config differs"):
        prepare_publication(
            snapshot,
            make_readme(tmp_path / "README.md"),
            tmp_path / "public",
        )


def test_prepare_publication_rejects_pending_model_card(tmp_path: Path) -> None:
    snapshot = make_snapshot(tmp_path / "snapshot")
    readme = make_readme(tmp_path / "README.md")
    readme.write_text(
        readme.read_text(encoding="utf-8")
        + "\n<!-- DS4RT_PUBLICATION_RESULTS_PENDING -->\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unfinished publication marker"):
        prepare_publication(snapshot, readme, tmp_path / "public")
