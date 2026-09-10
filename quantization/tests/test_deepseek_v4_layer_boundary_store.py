from __future__ import annotations

import hashlib
import json
import gc
from pathlib import Path

import pytest
import torch

from deepseek_v4_layer_boundary_store import (
    DeepSeekV4LayerBoundaryController,
    DeepSeekV4LayerBoundaryStore,
    LayerBoundaryError,
    LayerBoundaryStop,
    canonical_json_bytes,
)
from gptqmodel.utils.exl3_error_ledger import append_exl3_error_journal
from gptqmodel.looper.input_cache import DiskBackedLayerOutputWriter
from gptqmodel.utils.exl3_projection_checkpoint import (
    EXL3ProjectionCheckpointStore,
    build_projection_request,
)


FAMILY_JOIN = {
    "recipe": "boundary-test",
    "corpus": {"sha256": "a" * 64},
    "quantizer_numerics": {"sigma_reg": 0.025},
    "source": {
        "geometry": {
            "dspark_target_layer_ids": [40, 41, 42],
        }
    },
}


def _quantizer_metrics(
    *, sample_count: int = 32, scale_search_mse: float = 0.01
) -> dict[str, object]:
    return {
        "quantizer_path": "hessian_ldlq",
        "hessian_metric_status": "ok",
        "hessian_sample_count": sample_count,
        "hessian_regularization_sigma": 0.025,
        "hessian_numerical_contract": (
            "signed-block-hadamard-congruence-fp64-v1"
        ),
        "hessian_transform_compute_dtype": "torch.float64",
        "hessian_storage_dtype": "torch.float32",
        "hessian_regularization_placement": "before-fp64-congruence",
        "hessian_regularization_diagonal_addend": 0.001,
        "hessian_symmetry_restoration": "mean-with-transpose-fp64",
        "hessian_symmetry_correction_max_abs": 0.0,
        "selected_global_scale": 1.0,
        "scale_search_mse": scale_search_mse,
    }


def _projection_entries(
    checkpoint_root: Path,
    journal: Path,
    *,
    layer_index: int,
    routed_experts: int = 2,
    scale_search_mse: float = 0.01,
) -> list[dict[str, str]]:
    store = EXL3ProjectionCheckpointStore(checkpoint_root)
    entries = []
    for expert in range(routed_experts):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            module = (
                f"model.layers.{layer_index}.mlp.experts.{expert}.{projection}"
            )
            request = build_projection_request(
                module_full_name=module,
                layer_index=layer_index,
                input_weight=torch.arange(8, dtype=torch.float32).reshape(4, 2),
                hessian=torch.eye(4, dtype=torch.float32),
                sample_count=32,
                quantizer_contract={"bits": 2, "codebook": "mcg"},
                family_join=FAMILY_JOIN,
                route_evidence=None,
            )
            quantizer_metrics = _quantizer_metrics(
                scale_search_mse=scale_search_mse
            )
            ledger_record = {
                "schema": "ds41rt.exl3-error-ledger",
                "schema_version": 1,
                "record_kind": "projection",
                "module": module,
                "processor_layer_index": layer_index,
                "provenance": {"family_join": FAMILY_JOIN},
                "sample_count": 32,
                "quantizer_metrics": quantizer_metrics,
            }
            record_sha256 = append_exl3_error_journal(journal, ledger_record)
            store.commit(
                request,
                {
                    "trellis": torch.arange(8, dtype=torch.int16).reshape(1, 1, 8),
                    "suh": torch.ones(4, dtype=torch.float16),
                    "svh": torch.ones(4, dtype=torch.float16),
                    "mcg": torch.tensor([123], dtype=torch.int32),
                },
                {
                    "duration_seconds": 1.0,
                    "proxy_error": 0.1,
                    "device_names": ["cuda:0"],
                    "quantizer_metrics": quantizer_metrics,
                    "ledger_record": ledger_record,
                    "execution_contract": None,
                    "execution_result": {"kind": "test"},
                },
            )
            entries.append(
                {
                    "module": module,
                    "request_sha256": request["request_sha256"],
                    "record_sha256": record_sha256,
                }
            )
    return entries


def _mtp_projection_entry(
    checkpoint_root: Path,
    journal: Path,
    *,
    block_index: int = 0,
    expert: int = 0,
    projection: str = "gate_proj",
    ledger_overrides: dict | None = None,
) -> dict[str, str]:
    store = EXL3ProjectionCheckpointStore(checkpoint_root)
    module = f"mtp.{block_index}.mlp.experts.{expert}.{projection}"
    route_evidence = {
        "schema": "ds41rt.exl3-natural-route",
        "schema_version": 1,
        "block_namespace": "mtp",
        "logical_layer": block_index,
        "expert": expert,
    }
    request = build_projection_request(
        module_full_name=module,
        layer_index=block_index,
        input_weight=torch.arange(8, dtype=torch.float32).reshape(4, 2),
        hessian=torch.eye(4, dtype=torch.float32),
        sample_count=32,
        quantizer_contract={"bits": 2, "codebook": "mcg"},
        family_join=FAMILY_JOIN,
        route_evidence=route_evidence,
    )
    quantizer_metrics = _quantizer_metrics()
    ledger_record = {
        "schema": "ds41rt.exl3-error-ledger",
        "schema_version": 1,
        "record_kind": "projection",
        "module": module,
        "processor_layer_index": block_index,
        "block_namespace": "mtp",
        "logical_layer": block_index,
        "expert": expert,
        "projection": {
            "gate_proj": "w1",
            "up_proj": "w3",
            "down_proj": "w2",
        }[projection],
        "route_evidence": route_evidence,
        "provenance": {"family_join": FAMILY_JOIN},
        "sample_count": 32,
        "quantizer_metrics": quantizer_metrics,
    }
    if ledger_overrides:
        ledger_record.update(ledger_overrides)
    record_sha256 = append_exl3_error_journal(journal, ledger_record)
    store.commit(
        request,
        {
            "trellis": torch.arange(8, dtype=torch.int16).reshape(1, 1, 8),
            "suh": torch.ones(4, dtype=torch.float16),
            "svh": torch.ones(4, dtype=torch.float16),
            "mcg": torch.tensor([123], dtype=torch.int32),
        },
        {
            "duration_seconds": 1.0,
            "proxy_error": 0.1,
            "device_names": ["cuda:0"],
            "quantizer_metrics": quantizer_metrics,
            "ledger_record": ledger_record,
            "execution_contract": None,
            "execution_result": {"kind": "test"},
        },
    )
    return {
        "module": module,
        "request_sha256": request["request_sha256"],
        "record_sha256": record_sha256,
    }


def _metadata():
    return {
        "layer_input_kwargs": [
            {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64)}
        ],
        "position_ids": [torch.tensor([[0, 1, 2]], dtype=torch.int64)],
        "attention_masks": [torch.ones((1, 3), dtype=torch.int64)],
    }


def _boundary_store(tmp_path: Path) -> DeepSeekV4LayerBoundaryStore:
    return DeepSeekV4LayerBoundaryStore(
        tmp_path / "boundaries",
        plan_sha256="b" * 64,
        family_join=FAMILY_JOIN,
        projection_checkpoint_root=tmp_path / "projections",
        error_journal_path=tmp_path / "errors.jsonl",
        hidden_size=4,
        hc_mult=2,
        routed_experts=2,
    )


def test_layer_boundary_round_trip_and_rolling_retention(tmp_path: Path) -> None:
    store = _boundary_store(tmp_path)
    metadata = _metadata()
    for layer_index in (0, 1):
        entries = _projection_entries(
            tmp_path / "projections",
            tmp_path / "errors.jsonl",
            layer_index=layer_index,
        )
        hidden = (
            torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4)
            + layer_index
        ).to(torch.bfloat16)
        manifest = store.commit(
            layer_index=layer_index,
            layer_name=f"model.layers.{layer_index}",
            layer_outputs=[[hidden]],
            projection_entries=entries,
            **metadata,
        )
        assert manifest["activation_bytes"] == hidden.numel() * hidden.element_size()

    committed = [path for path in store.root.iterdir() if not path.name.startswith(".")]
    assert len(committed) == 1
    assert committed[0].name.startswith("layer-000001-")
    loaded = store.load_latest(**metadata)
    assert loaded is not None
    assert loaded.layer_index == 1
    assert loaded.layer_name == "model.layers.1"
    assert len(loaded.projection_entries) == 12
    assert loaded.layer_inputs.row_counts == [1]
    assert loaded.layer_inputs.lifetime_diagnostic() == {
        "issued": 0,
        "alive_tensors": 0,
        "alive_storage_bytes": 0,
    }
    retained = loaded.layer_inputs[0]
    assert torch.equal(
        retained[0],
        (torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4) + 1).to(
            torch.bfloat16
        ),
    )
    assert loaded.layer_inputs.lifetime_diagnostic()["alive_tensors"] == 1
    del retained
    gc.collect()
    assert loaded.layer_inputs.lifetime_diagnostic()["alive_tensors"] == 0


def test_layer_boundary_promotes_one_batch_replay_without_payload_copy(
    tmp_path: Path,
) -> None:
    store = _boundary_store(tmp_path)
    writer = DiskBackedLayerOutputWriter(
        tmp_path / "post-quant-replay",
        layer_index=0,
        expected_batches=2,
        provenance={"plan": "test"},
        shard_batches=1,
    )
    values = [
        torch.arange(24, dtype=torch.float32)
        .reshape(1, 3, 2, 4)
        .add(index)
        .to(torch.bfloat16)
        for index in range(2)
    ]
    for index, value in enumerate(values):
        writer.put(index, [value])
    outputs = writer.finalize()
    source_inodes = {
        record["start"]: (outputs.root / record["path"]).stat().st_ino
        for record in outputs.manifest["shards"]
    }
    store.commit(
        layer_index=0,
        layer_name="model.layers.0",
        layer_outputs=outputs,
        projection_entries=_projection_entries(
            tmp_path / "projections", tmp_path / "errors.jsonl", layer_index=0
        ),
        **_metadata(),
    )
    committed = next(
        path for path in store.root.iterdir() if not path.name.startswith(".")
    )
    promoted = sorted((committed / "activations").iterdir())
    assert [path.stat().st_ino for path in promoted] == [
        source_inodes[index] for index in range(2)
    ]
    loaded = store.load_latest(**_metadata())
    assert loaded is not None
    assert all(torch.equal(loaded.layer_inputs[index][0], value) for index, value in enumerate(values))


def test_layer_boundary_rejects_metadata_drift(tmp_path: Path) -> None:
    store = _boundary_store(tmp_path)
    metadata = _metadata()
    store.commit(
        layer_index=0,
        layer_name="model.layers.0",
        layer_outputs=[[torch.zeros((1, 3, 2, 4), dtype=torch.bfloat16)]],
        projection_entries=_projection_entries(
            tmp_path / "projections", tmp_path / "errors.jsonl", layer_index=0
        ),
        **metadata,
    )
    drifted = _metadata()
    drifted["position_ids"][0][0, 2] = 9
    with pytest.raises(LayerBoundaryError, match="metadata differs"):
        store.load_latest(**drifted)


def test_layer_boundary_rejects_activation_and_manifest_tampering(
    tmp_path: Path,
) -> None:
    store = _boundary_store(tmp_path)
    metadata = _metadata()
    store.commit(
        layer_index=0,
        layer_name="model.layers.0",
        layer_outputs=[[torch.zeros((1, 3, 2, 4), dtype=torch.bfloat16)]],
        projection_entries=_projection_entries(
            tmp_path / "projections", tmp_path / "errors.jsonl", layer_index=0
        ),
        **metadata,
    )
    committed = next(path for path in store.root.iterdir() if not path.name.startswith("."))
    activation = next((committed / "activations").iterdir())
    payload = bytearray(activation.read_bytes())
    payload[-1] ^= 1
    activation.write_bytes(payload)
    with pytest.raises(LayerBoundaryError, match="content validation"):
        store.load_latest(**metadata)

    shutil_store = _boundary_store(tmp_path / "second")
    second_metadata = _metadata()
    shutil_store.commit(
        layer_index=0,
        layer_name="model.layers.0",
        layer_outputs=[[torch.zeros((1, 3, 2, 4), dtype=torch.bfloat16)]],
        projection_entries=_projection_entries(
            tmp_path / "second" / "projections",
            tmp_path / "second" / "errors.jsonl",
            layer_index=0,
        ),
        **second_metadata,
    )
    committed = next(
        path for path in shutil_store.root.iterdir() if not path.name.startswith(".")
    )
    manifest_path = committed / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["activation_bytes"] += 2
    body = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    # Rebinding the JSON is insufficient because the committed directory name
    # also binds the original manifest digest.
    manifest["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(body)
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(LayerBoundaryError, match="manifest failed"):
        shutil_store.load_latest(**second_metadata)


def test_layer_boundary_requires_complete_projection_block(tmp_path: Path) -> None:
    store = _boundary_store(tmp_path)
    entries = _projection_entries(
        tmp_path / "projections", tmp_path / "errors.jsonl", layer_index=0
    )
    with pytest.raises(LayerBoundaryError, match="incomplete.*coverage"):
        store.commit(
            layer_index=0,
            layer_name="model.layers.0",
            layer_outputs=[[torch.zeros((1, 3, 2, 4), dtype=torch.bfloat16)]],
            projection_entries=entries[:-1],
            **_metadata(),
        )


def test_boundary_controller_selects_exl3_among_auxiliary_processors() -> None:
    class BoundaryCapable:
        def completed_layer_checkpoint_entries(self):
            return None

        def restore_completed_layer_checkpoints(self):
            return None

    exl3 = BoundaryCapable()
    assert DeepSeekV4LayerBoundaryController._processor([object(), exl3]) is exl3
    with pytest.raises(LayerBoundaryError, match="exactly one"):
        DeepSeekV4LayerBoundaryController._processor([object()])
    with pytest.raises(LayerBoundaryError, match="exactly one"):
        DeepSeekV4LayerBoundaryController._processor([exl3, BoundaryCapable()])


def test_discovers_only_contiguous_complete_projection_layers(tmp_path: Path) -> None:
    store = _boundary_store(tmp_path)
    for layer_index in (0, 1, 2):
        entries = _projection_entries(
            tmp_path / "projections",
            tmp_path / "errors.jsonl",
            layer_index=layer_index,
        )
        if layer_index == 2:
            request_sha256 = entries[-1]["request_sha256"]
            manifest, tensors = store.projection_store._paths(request_sha256)
            manifest.unlink()
            tensors.unlink()

    discovered = store.discover_completed_projection_layers()
    assert list(discovered) == [0, 1]
    assert all(len(entries) == 6 for entries in discovered.values())


def test_discovery_uses_selected_k3_and_audits_superseded_k2(
    tmp_path: Path,
) -> None:
    store = _boundary_store(tmp_path)
    journal = tmp_path / "errors.jsonl"
    entries = _projection_entries(
        tmp_path / "projections",
        journal,
        layer_index=0,
    )
    module = "model.layers.0.mlp.experts.0.gate_proj"
    original = next(entry for entry in entries if entry["module"] == module)
    for path in store.projection_store._paths(original["request_sha256"]):
        path.unlink()

    policy_sha256 = "c" * 64
    candidate_request = build_projection_request(
        module_full_name=module,
        layer_index=0,
        input_weight=torch.arange(8, dtype=torch.float32).reshape(4, 2),
        hessian=torch.eye(4, dtype=torch.float32),
        sample_count=32,
        quantizer_contract={
            "bits": 2,
            "codebook": "mcg",
            "inline_mixed": {
                "role": "candidate_k2",
                "base_bits": 2,
                "upgrade_bits": 3,
                "policy_sha256": policy_sha256,
            },
        },
        family_join=FAMILY_JOIN,
        route_evidence=None,
    )
    metrics = _quantizer_metrics()
    candidate_ledger = {
        "schema": "ds41rt.exl3-error-ledger",
        "schema_version": 1,
        "record_kind": "projection",
        "module": module,
        "processor_layer_index": 0,
        "bits": 2,
        "provenance": {"family_join": FAMILY_JOIN},
        "sample_count": 32,
        "quantizer_metrics": metrics,
    }
    append_exl3_error_journal(
        Path(f"{journal}.k2-candidates"), candidate_ledger
    )
    tensors = {
        "trellis": torch.arange(8, dtype=torch.int16).reshape(1, 1, 8),
        "suh": torch.ones(4, dtype=torch.float16),
        "svh": torch.ones(4, dtype=torch.float16),
        "mcg": torch.tensor([123], dtype=torch.int32),
    }
    result = {
        "duration_seconds": 1.0,
        "proxy_error": 0.1,
        "device_names": ["cuda:0"],
        "quantizer_metrics": metrics,
        "ledger_record": candidate_ledger,
        "execution_contract": None,
        "execution_result": {"kind": "test"},
    }
    store.projection_store.commit(candidate_request, tensors, result)

    selected_request = build_projection_request(
        module_full_name=module,
        layer_index=0,
        input_weight=torch.arange(8, dtype=torch.float32).reshape(4, 2),
        hessian=torch.eye(4, dtype=torch.float32),
        sample_count=32,
        quantizer_contract={
            "bits": 3,
            "codebook": "mcg",
            "inline_mixed": {
                "role": "selected_k3",
                "base_bits": 2,
                "upgrade_bits": 3,
                "policy_sha256": policy_sha256,
                "tier_plan_sha256": "d" * 64,
                "candidate_request_sha256": candidate_request[
                    "request_sha256"
                ],
            },
        },
        family_join=FAMILY_JOIN,
        route_evidence=None,
    )
    selected_ledger = {**candidate_ledger, "bits": 3}
    selected_record_sha256 = append_exl3_error_journal(
        journal, selected_ledger
    )
    store.projection_store.commit(
        selected_request,
        tensors,
        {**result, "ledger_record": selected_ledger},
    )

    discovered = store.discover_completed_projection_layers()

    selected = next(
        entry for entry in discovered[0] if entry["module"] == module
    )
    assert selected["request_sha256"] == selected_request["request_sha256"]
    assert selected["record_sha256"] == selected_record_sha256


def test_discovery_rejects_impossible_retained_projection_metric(
    tmp_path: Path,
) -> None:
    store = _boundary_store(tmp_path)
    _projection_entries(
        tmp_path / "projections",
        tmp_path / "errors.jsonl",
        layer_index=0,
        scale_search_mse=-1.0,
    )

    with pytest.raises(LayerBoundaryError, match="invalid numerical evidence"):
        store.discover_completed_projection_layers()


def test_decoder_discovery_validates_but_does_not_consume_mtp_checkpoints(
    tmp_path: Path,
) -> None:
    store = _boundary_store(tmp_path)
    _projection_entries(
        tmp_path / "projections",
        tmp_path / "errors.jsonl",
        layer_index=0,
    )
    _mtp_projection_entry(
        tmp_path / "projections",
        tmp_path / "errors.jsonl",
    )

    discovered = store.discover_completed_projection_layers()

    assert list(discovered) == [0]
    assert len(discovered[0]) == 6
    assert all(entry["module"].startswith("model.layers.0.") for entry in discovered[0])


@pytest.mark.parametrize(
    ("block_index", "ledger_overrides"),
    [
        (3, None),
        (0, {"block_namespace": "model"}),
        (0, {"logical_layer": 1}),
    ],
)
def test_decoder_discovery_rejects_invalid_mtp_checkpoint_evidence(
    tmp_path: Path,
    block_index: int,
    ledger_overrides: dict | None,
) -> None:
    store = _boundary_store(tmp_path)
    _projection_entries(
        tmp_path / "projections",
        tmp_path / "errors.jsonl",
        layer_index=0,
    )
    _mtp_projection_entry(
        tmp_path / "projections",
        tmp_path / "errors.jsonl",
        block_index=block_index,
        ledger_overrides=ledger_overrides,
    )

    with pytest.raises(LayerBoundaryError, match="dSpark projection discovery"):
        store.discover_completed_projection_layers()


def test_discovery_rejects_a_hole_before_a_complete_layer(tmp_path: Path) -> None:
    store = _boundary_store(tmp_path)
    entries_by_layer = {
        layer_index: _projection_entries(
            tmp_path / "projections",
            tmp_path / "errors.jsonl",
            layer_index=layer_index,
        )
        for layer_index in (0, 1, 2)
    }
    request_sha256 = entries_by_layer[1][-1]["request_sha256"]
    manifest, tensors = store.projection_store._paths(request_sha256)
    manifest.unlink()
    tensors.unlink()
    with pytest.raises(LayerBoundaryError, match="not a contiguous"):
        store.discover_completed_projection_layers()


def test_controller_prepares_discovered_layer_once(tmp_path: Path) -> None:
    store = _boundary_store(tmp_path)
    entries = _projection_entries(
        tmp_path / "projections",
        tmp_path / "errors.jsonl",
        layer_index=0,
    )

    class Cache:
        layer_input_kwargs = _metadata()["layer_input_kwargs"]
        position_ids = _metadata()["position_ids"]
        attention_masks = _metadata()["attention_masks"]

    class Processor:
        inputs_cache = Cache()
        qcfg = type("QCfg", (), {"device": "cuda:0"})()

        def __init__(self):
            self.restored = []
            self.offloaded = []

        def completed_layer_checkpoint_entries(self, _layer_index):
            return entries

        def restore_completed_layer_checkpoints(self, **kwargs):
            self.restored.append(kwargs)

        def offload_restored_layer_checkpoints(self, **kwargs):
            self.offloaded.append(kwargs)

    processor = Processor()
    controller = DeepSeekV4LayerBoundaryController(store)
    model = object()
    assert controller.restore(model=model, processors=[object(), processor]) == 0
    assert controller.is_catchup_layer(0)
    assert not processor.restored
    assert (
        controller.prepare_catchup_layer(
            model=model,
            processors=[object(), processor],
            layer_index=0,
        )
        is processor
    )
    assert len(processor.restored) == 1
    assert processor.restored[0]["model"] is model
    assert processor.restored[0]["layer_index"] == 0
    assert processor.restored[0]["materialize_device"] == "cuda:0"
    assert {
        entry["module"]: entry for entry in processor.restored[0]["projection_entries"]
    } == {entry["module"]: entry for entry in entries}
    controller.finalize_catchup_layer(
        model=model,
        processor=processor,
        layer_index=0,
    )
    assert processor.offloaded == [{"model": model, "layer_index": 0}]
    with pytest.raises(LayerBoundaryError, match="prepared twice"):
        controller.prepare_catchup_layer(
            model=model,
            processors=[object(), processor],
            layer_index=0,
        )


def test_controller_defers_boundary_prefix_until_publication(tmp_path: Path) -> None:
    store = _boundary_store(tmp_path)
    metadata = _metadata()
    entries = _projection_entries(
        tmp_path / "projections",
        tmp_path / "errors.jsonl",
        layer_index=0,
    )
    layer_outputs = [[torch.zeros((1, 3, 2, 4), dtype=torch.bfloat16)]]
    store.commit(
        layer_index=0,
        layer_name="model.layers.0",
        layer_outputs=layer_outputs,
        projection_entries=entries,
        **metadata,
    )

    class Cache:
        layer_input_kwargs = metadata["layer_input_kwargs"]
        position_ids = metadata["position_ids"]
        attention_masks = metadata["attention_masks"]

    class Processor:
        inputs_cache = Cache()

        def __init__(self):
            self.received = []
            self.restored = []

        def completed_layer_checkpoint_entries(self, _layer_index):
            return entries

        def restore_completed_layer_checkpoints(self, **kwargs):
            self.restored.append(kwargs)

        def receive_layer_inputs(self, inputs):
            self.received.append(inputs)

    processor = Processor()
    controller = DeepSeekV4LayerBoundaryController(store)
    model = object()

    assert controller.restore(model=model, processors=[processor]) == 1
    assert not processor.restored
    assert len(processor.received) == 1
    assert torch.equal(processor.received[0][0][0], layer_outputs[0][0])

    controller.materialize_deferred_prefix(
        model=model,
        processors=[processor],
    )
    assert len(processor.restored) == 1


def test_controller_can_hold_boundary_prefix_through_mtp_quantization(
    tmp_path: Path,
) -> None:
    store = _boundary_store(tmp_path)
    metadata = _metadata()
    entries = _projection_entries(
        tmp_path / "projections", tmp_path / "errors.jsonl", layer_index=0
    )
    store.commit(
        layer_index=0,
        layer_name="model.layers.0",
        layer_outputs=[[torch.zeros((1, 3, 2, 4), dtype=torch.bfloat16)]],
        projection_entries=entries,
        **metadata,
    )

    class Cache:
        layer_input_kwargs = metadata["layer_input_kwargs"]
        position_ids = metadata["position_ids"]
        attention_masks = metadata["attention_masks"]

    class Processor:
        inputs_cache = Cache()

        def __init__(self):
            self.restored = []

        def completed_layer_checkpoint_entries(self, _layer_index):
            return entries

        def restore_completed_layer_checkpoints(self, **kwargs):
            self.restored.append(kwargs)

        def receive_layer_inputs(self, _inputs):
            pass

    processor = Processor()
    controller = DeepSeekV4LayerBoundaryController(
        store, defer_publication_materialization=True
    )
    model = object()
    assert controller.restore(model=model, processors=[processor]) == 1
    controller.materialize_deferred_prefix(
        model=model, processors=[processor]
    )
    assert processor.restored == []
    controller.materialize_deferred_prefix(model=model, force=True)
    assert len(processor.restored) == 1
    assert processor.restored[0]["model"] is model
    assert processor.restored[0]["layer_index"] == 0
    assert "materialize_device" not in processor.restored[0]
    assert {
        entry["module"]: entry
        for entry in processor.restored[0]["projection_entries"]
    } == {entry["module"]: entry for entry in entries}

    controller.materialize_deferred_prefix(
        model=model,
        processors=[processor],
    )
    assert len(processor.restored) == 1


def test_boundary_controller_accounts_and_trims_after_durable_commit(
    tmp_path: Path,
) -> None:
    store = _boundary_store(tmp_path)
    metadata = _metadata()
    entries = _projection_entries(
        tmp_path / "projections",
        tmp_path / "errors.jsonl",
        layer_index=0,
    )
    layer_outputs = [[torch.zeros((1, 3, 2, 4), dtype=torch.bfloat16)]]
    events = []

    class Cache:
        layer_inputs = layer_outputs
        layer_input_kwargs = metadata["layer_input_kwargs"]
        position_ids = metadata["position_ids"]
        attention_masks = metadata["attention_masks"]

    class Processor:
        inputs_cache = Cache()

        def __init__(self):
            self.deferred = []

        def completed_layer_checkpoint_entries(self, _layer_index):
            return entries

        def defer_completed_layer_checkpoints(self, **kwargs):
            self.deferred.append(kwargs)

        def log_capture_memory_summary(self, context, *, model):
            layer = model.model.get_submodule("model.layers.0")
            devices = {
                tensor.device.type
                for tensor in (
                    *layer.parameters(),
                    *layer.buffers(),
                )
            }
            events.append(("devices", context, devices))
            events.append(("summary", context, model))

        def discard_capture_frontiers_through(self, layer_index):
            events.append(("discard", layer_index))

        def release_host_memory(self, context, *, model):
            events.append(("release", context, model))

    target = torch.nn.Module()
    target.model = torch.nn.Module()
    target.model.layers = torch.nn.ModuleList(
        [torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))]
    )

    class Turtle:
        def materialize_submodule(self, **_kwargs):
            raise AssertionError("publication materialization is not due yet")

        def sync_all_meta(self, **_kwargs):
            raise AssertionError("publication materialization is not due yet")

    class Model:
        model = target
        turtle_model = Turtle()

    model = Model()
    processor = Processor()
    manifest = DeepSeekV4LayerBoundaryController(
        store, defer_publication_materialization=True
    ).commit_layer(
        model=model,
        processor=processor,
        layer_index=0,
        layer_name="model.layers.0",
    )

    assert manifest["layer_index"] == 0
    assert processor.deferred == [
        {
            "model": model,
            "layer_index": 0,
            "projection_entries": entries,
        }
    ]
    assert events == [
        ("devices", "layer-0-before-boundary", {"cpu"}),
        ("summary", "layer-0-before-boundary", model),
        ("devices", "layer-0-after-boundary", {"meta"}),
        ("summary", "layer-0-after-boundary", model),
        ("discard", 0),
        ("release", "layer-0-after-boundary", model),
    ]
def test_boundary_controller_stops_only_after_durable_commit(tmp_path: Path) -> None:
    store = _boundary_store(tmp_path)
    metadata = _metadata()
    entries = _projection_entries(
        tmp_path / "projections",
        tmp_path / "errors.jsonl",
        layer_index=0,
    )

    class Cache:
        layer_inputs = [[torch.zeros((1, 3, 2, 4), dtype=torch.bfloat16)]]
        layer_input_kwargs = metadata["layer_input_kwargs"]
        position_ids = metadata["position_ids"]
        attention_masks = metadata["attention_masks"]

    class Processor:
        inputs_cache = Cache()

        def completed_layer_checkpoint_entries(self, _layer_index):
            return entries

    controller = DeepSeekV4LayerBoundaryController(store, stop_after_layer=0)
    with pytest.raises(LayerBoundaryStop, match="durable decoder layer 0"):
        controller.commit_layer(
            model=object(),
            processor=Processor(),
            layer_index=0,
            layer_name="model.layers.0",
        )
    assert controller.stopped_after_layer == 0
    assert store.load_latest(**metadata).layer_index == 0
