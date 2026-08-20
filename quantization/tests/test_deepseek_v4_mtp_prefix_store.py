from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from gptqmodel.models.definitions.deepseek_v4 import (
    DeepSeekV4MTPTargetTapEvent,
    MTP_CAPTURE_ATTENTION_MASK,
    MTP_CAPTURE_DECODE_MASK,
    MTP_CAPTURE_INPUT_IDS,
)


SCRIPT = Path(__file__).parents[1] / "deepseek_v4_mtp_prefix_store.py"
SPEC = importlib.util.spec_from_file_location("deepseek_v4_mtp_prefix_store", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _event(layer: int, raw: torch.Tensor) -> DeepSeekV4MTPTargetTapEvent:
    input_ids = torch.tensor([[11, 12, 13]])
    attention_mask = torch.tensor([[True, True, True]])
    position_ids = torch.tensor([[4, 5, 6]])
    return DeepSeekV4MTPTargetTapEvent(
        layer_index=layer,
        layer_name=f"model.layers.{layer}",
        collapsed_target_taps=(raw.mean(dim=2),),
        raw_layer_outputs=(raw,),
        layer_input_kwargs=(
            {
                MTP_CAPTURE_INPUT_IDS: input_ids,
                MTP_CAPTURE_ATTENTION_MASK: attention_mask,
                MTP_CAPTURE_DECODE_MASK: torch.tensor([[False, True, True]]),
            },
        ),
        position_ids=(position_ids,),
        attention_masks=(None,),
    )


def _store(root: Path):
    return MODULE.DeepSeekV4MTPPrefixStore(
        root,
        target_layer_ids=(40, 41, 42),
        hidden_size=4,
        hc_mult=2,
        projector=lambda taps: taps[0] + taps[1] + taps[2],
        anchor_resolver=lambda raw, input_ids, attention_mask, decode_mask, position_ids: (
            input_ids + attention_mask.long() + position_ids * 0 + raw.shape[2] - 2
        ),
        projection_device="cpu",
        projection_dtype=torch.bfloat16,
        provenance={"checkpoint": "revision", "dataset": "manifest-sha256"},
    )


def test_prefix_store_projects_three_layers_and_persists_resume_records(
    tmp_path: Path,
) -> None:
    raw_by_layer = {
        layer: torch.full((1, 3, 2, 4), float(layer), dtype=torch.bfloat16)
        for layer in (40, 41, 42)
    }
    store = _store(tmp_path)
    for layer in (40, 41, 42):
        store(_event(layer, raw_by_layer[layer]))

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["schema"] == MODULE.SCHEMA
    assert manifest["status"] == "complete"
    assert manifest["completed_layers"] == [40, 41, 42]
    assert manifest["batch_count"] == 1
    main_path = tmp_path / manifest["batches"]["000000"]["projected_main"]["path"]
    tensors = load_file(main_path)
    assert tensors["projected_main"].shape == (1, 3, 4)
    assert torch.equal(
        tensors["projected_main"],
        torch.full((1, 3, 4), 123.0, dtype=torch.bfloat16),
    )
    assert tensors["anchor_token_ids"].tolist() == [[12, 13, 14]]
    assert tensors["input_ids"].tolist() == [[11, 12, 13]]
    assert tensors["main_position_ids"].tolist() == [[4, 5, 6]]
    assert tensors["main_attention_mask"].tolist() == [[True, True, True]]
    assert tensors["dspark_decode_mask"].tolist() == [[False, True, True]]
    assert stat.S_IMODE(main_path.stat().st_mode) == 0o644
    assert stat.S_IMODE((tmp_path / "manifest.json").stat().st_mode) == 0o644

    resumed = _store(tmp_path)
    for layer in (40, 41, 42):
        resumed(_event(layer, raw_by_layer[layer]))
    assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "complete"

    stored_batches = list(
        resumed.iter_replay_batches(replay_batch_size=2, device="cpu")
    )
    assert len(stored_batches) == 1
    stored = stored_batches[0]
    assert stored.sources == (
        {
            "batch_index": 0,
            "sequence_index": 0,
            "token_index": 1,
            "absolute_position": 5,
        },
        {
            "batch_index": 0,
            "sequence_index": 0,
            "token_index": 2,
            "absolute_position": 6,
        },
    )
    replay = stored.replay_batch
    assert replay.target_taps is None
    assert replay.projected_main.shape == (2, 3, 4)
    assert replay.anchor_token_ids.tolist() == [13, 14]
    assert replay.main_attention_mask.tolist() == [
        [False, True, True],
        [True, True, True],
    ]
    assert replay.main_position_ids.tolist() == [[0, 4, 5], [4, 5, 6]]

    lazy = resumed.replay_dataset(replay_batch_size=2, device="cpu")
    assert len(lazy) == 1
    assert lazy.position_count == 2
    assert lazy.row_counts == [2]
    assert lazy.gptqmodel_calibration_summary == {
        "batch_count": 1,
        "input_ids_total_length": 10,
        "input_ids_max_length": 5,
        "total_calibration_tokens": 10,
    }
    first = lazy[0]
    second = lazy[0]
    assert first.sources == stored.sources
    assert second.sources == stored.sources
    assert torch.equal(
        first.replay_batch.projected_main,
        stored.replay_batch.projected_main,
    )

    # Production replay restores the source-sequence boundary. Both selected
    # anchors are one outer batch even when the legacy fixed batch width is 1.
    sequence_batched = resumed.replay_dataset(
        replay_batch_size=1,
        device="cpu",
        batch_by_source_sequence=True,
    )
    assert len(sequence_batched) == 1
    assert sequence_batched.row_counts == [2]
    assert sequence_batched[0].sources == stored.sources
    assert sequence_batched.replay_batching_identity["contract"] == (
        MODULE.SEQUENCE_REPLAY_BATCH_CONTRACT
    )

    sampled = resumed.replay_dataset(
        replay_batch_size=1,
        device="cpu",
        anchor_sample_count=1,
        anchor_sample_seed=17,
        batch_by_source_sequence=True,
    )
    repeated = resumed.replay_dataset(
        replay_batch_size=1,
        device="cpu",
        anchor_sample_count=1,
        anchor_sample_seed=17,
        batch_by_source_sequence=True,
    )
    assert sampled.position_count == 1
    assert sampled.anchor_selection_identity == repeated.anchor_selection_identity
    assert sampled.anchor_selection_identity == {
        "contract": MODULE.ANCHOR_SELECTION_CONTRACT,
        "seed": 17,
        "source_position_count": 2,
        "selected_position_count": 1,
        "selected_coordinates_sha256": sampled.anchor_selection_identity[
            "selected_coordinates_sha256"
        ],
    }
    assert sampled.gptqmodel_calibration_summary["total_calibration_tokens"] == 5

    opened = MODULE.DeepSeekV4MTPPrefixStore.open_complete(
        tmp_path,
        expected_manifest_sha256=MODULE.sha256_file(tmp_path / "manifest.json"),
        expected_provenance={
            "checkpoint": "revision",
            "dataset": "manifest-sha256",
        },
    )
    assert opened.read_only is True
    assert opened.target_layer_ids == (40, 41, 42)
    assert opened.manifest_sha256 == MODULE.sha256_file(
        tmp_path / "manifest.json"
    )
    assert opened.replay_dataset(replay_batch_size=2)[0].sources == stored.sources
    with pytest.raises(MODULE.PrefixStoreError, match="cannot accept target taps"):
        opened(_event(40, raw_by_layer[40]))


def test_prefix_store_read_only_open_binds_manifest_and_provenance(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    for layer in (40, 41, 42):
        raw = torch.full((1, 3, 2, 4), float(layer), dtype=torch.bfloat16)
        store(_event(layer, raw))

    with pytest.raises(MODULE.PrefixStoreError, match="manifest identity"):
        MODULE.DeepSeekV4MTPPrefixStore.open_complete(
            tmp_path,
            expected_manifest_sha256="0" * 64,
        )
    with pytest.raises(MODULE.PrefixStoreError, match="provenance mismatch"):
        MODULE.DeepSeekV4MTPPrefixStore.open_complete(
            tmp_path,
            expected_provenance={"checkpoint": "another"},
        )

    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "incomplete"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(MODULE.PrefixStoreError, match="manifest is inconsistent"):
        MODULE.DeepSeekV4MTPPrefixStore.open_complete(tmp_path)


def test_stratified_anchor_selection_is_exact_ordered_and_seeded() -> None:
    first = MODULE._stratified_anchor_indices(101, 30, 1234)
    repeated = MODULE._stratified_anchor_indices(101, 30, 1234)
    changed = MODULE._stratified_anchor_indices(101, 30, 1235)

    assert len(first) == 30
    assert list(first) == sorted(set(first))
    assert first == repeated
    assert first != changed
    for stratum, index in enumerate(first):
        assert (stratum * 101) // 30 <= index < ((stratum + 1) * 101) // 30


def test_prefix_store_rejects_existing_tap_drift(tmp_path: Path) -> None:
    store = _store(tmp_path)
    raw = torch.zeros(1, 3, 2, 4, dtype=torch.bfloat16)
    store(_event(40, raw))
    resumed = _store(tmp_path)
    with pytest.raises(MODULE.PrefixStoreError, match="differs from replay"):
        resumed(_event(40, raw + 1))


def test_prefix_store_replaces_uncommitted_partial_tap(tmp_path: Path) -> None:
    store = _store(tmp_path)
    path = tmp_path / "target-taps" / "layer_40" / "batch_000000.safetensors"
    path.parent.mkdir(parents=True)
    save_file(
        {"target_tap": torch.ones(1, 3, 4, dtype=torch.bfloat16)},
        path,
    )

    raw = torch.zeros(1, 3, 2, 4, dtype=torch.bfloat16)
    store(_event(40, raw))

    assert torch.equal(
        load_file(path)["target_tap"],
        torch.zeros(1, 3, 4, dtype=torch.bfloat16),
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["completed_layers"] == [40]
    assert manifest["batches"]["000000"]["target_taps"]["40"]["sha256"] == (
        MODULE.sha256_file(path)
    )


def test_prefix_store_requires_original_two_dimensional_mask(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for layer in (40, 41):
        store(_event(layer, torch.zeros(1, 3, 2, 4, dtype=torch.bfloat16)))
    event = _event(42, torch.zeros(1, 3, 2, 4, dtype=torch.bfloat16))
    broken_kwargs = dict(event.layer_input_kwargs[0])
    broken_kwargs[MTP_CAPTURE_ATTENTION_MASK] = torch.zeros(1, 1, 3, 3)
    broken = SimpleNamespace(**event.__dict__)
    broken.layer_input_kwargs = (broken_kwargs,)
    with pytest.raises(MODULE.PrefixStoreError, match="rank-2 attention mask"):
        store(broken)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA staging unavailable")
def test_prefix_store_stages_cpu_batches_to_materialized_cuda_runtime(
    tmp_path: Path,
) -> None:
    device = torch.device("cuda:0")

    def project(taps):
        assert all(tap.device == device for tap in taps)
        assert all(tap.dtype == torch.bfloat16 for tap in taps)
        return taps[0] + taps[1] + taps[2]

    def resolve(raw, input_ids, attention_mask, decode_mask, position_ids):
        assert all(
            value.device == device
            for value in (raw, input_ids, attention_mask, decode_mask, position_ids)
        )
        return input_ids

    store = MODULE.DeepSeekV4MTPPrefixStore(
        tmp_path,
        target_layer_ids=(40, 41, 42),
        hidden_size=4,
        hc_mult=2,
        projector=project,
        anchor_resolver=resolve,
        projection_device=device,
        projection_dtype=torch.bfloat16,
        provenance={"checkpoint": "revision", "dataset": "manifest-sha256"},
    )
    for layer in (40, 41, 42):
        raw = torch.full((1, 3, 2, 4), float(layer), dtype=torch.bfloat16)
        store(_event(layer, raw))

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    stored = load_file(
        tmp_path / manifest["batches"]["000000"]["projected_main"]["path"]
    )
    assert stored["projected_main"].device.type == "cpu"
