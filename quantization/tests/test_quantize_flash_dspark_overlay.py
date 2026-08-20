from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file


QUANTIZATION = Path(__file__).parents[1]
if str(QUANTIZATION) not in sys.path:
    sys.path.insert(0, str(QUANTIZATION))
SCRIPT = QUANTIZATION / "quantize_flash_dspark_overlay.py"
SPEC = importlib.util.spec_from_file_location(
    "quantize_flash_dspark_overlay",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _fixture(tmp_path: Path) -> argparse.Namespace:
    snapshot = tmp_path / ("a" * 40)
    snapshot.mkdir(parents=True)
    _write_json(
        snapshot / "config.json",
        {
            "num_hidden_layers": 3,
            "n_routed_experts": 2,
            "hidden_size": 16,
            "moe_intermediate_size": 16,
            "dspark_target_layer_ids": [0, 1, 2],
            "hc_mult": 2,
            "num_hash_layers": 0,
            "num_experts_per_tok": 1,
            "num_nextn_predict_layers": 1,
        },
    )
    weight_names = {
        f"{prefix}.ffn.experts.{expert}.{projection}.{suffix}"
        for prefix in (
            *(f"layers.{layer}" for layer in range(3)),
            *(f"mtp.{block}" for block in range(3)),
        )
        for expert in range(2)
        for projection in ("w1", "w2", "w3")
        for suffix in ("scale", "weight")
    }
    weight_names.update(
        f"{prefix}.ffn.shared_experts.{projection}.{suffix}"
        for prefix in (
            *(f"layers.{layer}" for layer in range(3)),
            *(f"mtp.{block}" for block in range(3)),
        )
        for projection in ("w1", "w2", "w3")
        for suffix in ("scale", "weight")
    )
    weight_names.update(
        f"{prefix}.ffn.gate.{suffix}"
        for prefix in (
            *(f"layers.{layer}" for layer in range(3)),
            *(f"mtp.{block}" for block in range(3)),
        )
        for suffix in ("weight", "bias")
    )
    _write_json(
        snapshot / "model.safetensors.index.json",
        {"weight_map": {name: "model.safetensors" for name in weight_names}},
    )
    (snapshot / "model.safetensors").write_bytes(b"source")
    corpus = tmp_path / "calibration.jsonl"
    corpus.write_text('{"id":"one","text":"calibration"}\n')
    lock = {
        "schema": 1,
        "repository": "https://github.com/tpurtell/GPTQModel.git",
        "revision": "b" * 40,
        "source_tree_sha256": "c" * 64,
    }
    lock_path = tmp_path / "gptqmodel.lock.json"
    _write_json(lock_path, lock)
    coordinator = {
        "status": "qualified",
        "role": "coordinator",
        "target_platform": "linux/amd64",
        "cuda_arch": "120",
        "image_digest": "sha256:" + "d" * 64,
        "gptqmodel": {
            "revision": lock["revision"],
            "source_tree_sha256": lock["source_tree_sha256"],
        },
        "python": {"gil_enabled": False},
        "torch": {"version": "test"},
        "gpus": [
            {"index": 0, "uuid": "GPU-coordinator-0"},
            {"index": 1, "uuid": "GPU-coordinator-1"},
        ],
    }
    coordinator_path = tmp_path / "coordinator.json"
    _write_json(coordinator_path, coordinator)

    parent_args = argparse.Namespace(
        snapshot=snapshot,
        calibration_jsonl=corpus,
        preflight_report=coordinator_path,
        output=tmp_path / "parent-output",
        offload_dir=tmp_path / "parent-offload",
        mtp_prefix_store=tmp_path / "prefix",
        gptqmodel_lock=lock_path,
        batch_size=1,
        mtp_replay_batch_size=4,
        mtp_execution_mode=MODULE.base.MTP_EXECUTION_EXTERNAL_OVERLAY,
        bits=3,
        coordinator_gpu_count=2,
        remote_worker=None,
    )
    parent, _ = MODULE.base.build_plan(parent_args)
    parent_path = tmp_path / "parent-plan.json"
    _write_json(parent_path, parent)

    prefix = tmp_path / "prefix"
    projected_path = prefix / "projected-main" / "batch_000000.safetensors"
    projected_path.parent.mkdir(parents=True)
    save_file(
        {
            "projected_main": torch.zeros(1, 3, 16, dtype=torch.bfloat16),
            "anchor_token_ids": torch.tensor([[1, 2, 3]]),
            "input_ids": torch.tensor([[1, 2, 3]]),
            "main_attention_mask": torch.tensor([[True, True, True]]),
            "dspark_decode_mask": torch.tensor([[False, True, True]]),
            "main_position_ids": torch.tensor([[0, 1, 2]]),
        },
        projected_path,
    )
    target_taps = {}
    for layer in (0, 1, 2):
        tap_path = prefix / "target-taps" / f"layer_{layer:02d}" / "batch_000000.safetensors"
        tap_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {"target_tap": torch.zeros(1, 3, 16, dtype=torch.bfloat16)},
            tap_path,
        )
        target_taps[str(layer)] = {
            "path": tap_path.relative_to(prefix).as_posix(),
            "bytes": tap_path.stat().st_size,
            "sha256": MODULE.sha256_file(tap_path),
            "tensors": {},
        }
    manifest = {
        "schema": "ds4rt-deepseek-v4-mtp-prefix-store-v1",
        "status": "complete",
        "target_layer_ids": [0, 1, 2],
        "hidden_size": 16,
        "hc_mult": 2,
        "provenance": {
            "plan_sha256": parent["plan_sha256"],
            "family_join": parent["ledger_provenance"]["family_join"],
        },
        "batch_count": 1,
        "completed_layers": [0, 1, 2],
        "batches": {
            "000000": {
                "target_taps": target_taps,
                "projected_main": {
                    "path": projected_path.relative_to(prefix).as_posix(),
                    "bytes": projected_path.stat().st_size,
                    "sha256": MODULE.sha256_file(projected_path),
                    "tensors": {},
                },
            }
        },
    }
    _write_json(prefix / "manifest.json", manifest)
    parent_run = Path(parent["run_state_dir"])
    parent_run.mkdir()
    _write_json(parent_run / MODULE.base.PLAN_FILENAME, parent)
    MODULE.base.publish_base_prefix_completion(
        parent,
        prefix_manifest_path=prefix / "manifest.json",
    )

    declarations = []
    for index, name in enumerate(("dodo", "emu", "kiwi", "ostrich")):
        worker = json.loads(json.dumps(coordinator))
        worker["role"] = "expert"
        worker["target_platform"] = "linux/arm64"
        worker["cuda_arch"] = "121"
        worker["image_digest"] = "sha256:" + "e" * 64
        worker["gpus"] = [{"index": 0, "uuid": f"GPU-worker-{index}"}]
        worker_path = tmp_path / f"{name}.json"
        _write_json(worker_path, worker)
        declarations.append([name, f"http://{name}:17841", str(worker_path)])

    return argparse.Namespace(
        snapshot=snapshot,
        parent_plan=parent_path,
        mtp_prefix_store=prefix,
        preflight_report=coordinator_path,
        output=tmp_path / "overlay-k3",
        offload_dir=tmp_path / "overlay-k3-offload",
        bits=3,
        coordinator_gpu_count=2,
        gptqmodel_lock=lock_path,
        mtp_anchor_sample_count=2,
        mtp_anchor_sample_seed=20260809,
        mtp_sequence_anchor_cap=0,
        remote_worker=declarations,
        remote_token_env="TEST_WORKER_TOKEN",
        remote_timeout_seconds=3600.0,
        remote_max_attempts=2,
    )


def test_overlay_plan_binds_parent_prefix_tier_and_six_slots(tmp_path: Path) -> None:
    args = _fixture(tmp_path)

    first = MODULE.build_plan(args)
    second = MODULE.build_plan(args)

    assert first == second
    assert first["schema"] == MODULE.PLAN_SCHEMA
    assert first["scope"] == MODULE.SCOPE
    assert first["exl3"]["bits"] == 3
    assert first["target_parent"]["plan_sha256"] == json.loads(
        args.parent_plan.read_text()
    )["plan_sha256"]
    assert first["prefix"]["manifest_sha256"] == MODULE.sha256_file(
        args.mtp_prefix_store / "manifest.json"
    )
    assert first["prefix"]["projected_main_bytes"] > 0
    assert first["anchor_selection"] == {
        "contract": MODULE.ANCHOR_SELECTION_CONTRACT,
        "count": 2,
        "seed": 20260809,
    }
    assert first["replay_batching"] == {
        "contract": MODULE.SEQUENCE_REPLAY_BATCH_CONTRACT,
        "source_sequence_anchor_cap": None,
        "proposal_rows_per_anchor": 5,
    }
    assert first["exl3"]["hessian_owner_policy"] == {
        "contract": MODULE.HESSIAN_OWNER_POLICY_CONTRACT,
        "device_weights": [1, 1],
    }
    assert (
        first["ledger_provenance"]["family_join"]["hessian_owner_policy"]
        == first["exl3"]["hessian_owner_policy"]
    )
    assert first["ledger_provenance"]["family_join"]["corpus"] == first["corpus"]
    assert (
        first["ledger_provenance"]["family_join"][
            "zero_route_recovery_contract"
        ]
        == MODULE.ZERO_ROUTE_RECOVERY_SCHEMA
    )
    assert first["exl3"]["zero_route_recovery"] == {
        "contract": MODULE.ZERO_ROUTE_RECOVERY_SCHEMA,
        "trigger": MODULE.ZERO_ROUTE_RECOVERY_TRIGGER,
        "sample_source": MODULE.ZERO_ROUTE_RECOVERY_SAMPLE_SOURCE,
        "capture_method": MODULE.ZERO_ROUTE_RECOVERY_CAPTURE_METHOD,
        "selection_policy": MODULE.ZERO_ROUTE_RECOVERY_SELECTION_POLICY,
        "candidate_rank_min": MODULE.ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MIN,
        "candidate_rank_max": MODULE.ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MAX,
        "target_sample_count": MODULE.ZERO_ROUTE_RECOVERY_TARGET_SAMPLE_COUNT,
        "identity_calibration_policy": MODULE.ZERO_ROUTE_RECOVERY_IDENTITY_POLICY,
    }
    assert [
        slot["device"] for slot in first["remote_workers"]["coordinator_slots"]
    ] == ["cuda:0", "cuda:1"]
    assert len(first["remote_workers"]["endpoints"]) == 4
    assert first["remote_workers"]["cuda_workers_per_device"] == 7
    assert first["projection_checkpoint"]["root"].endswith(
        "/projection-checkpoints"
    )


def test_overlay_plan_supports_coordinator_only_k3(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    args.remote_worker = None

    plan = MODULE.build_plan(args)

    assert plan["exl3"]["bits"] == 3
    assert plan["remote_workers"] is None
    topology = plan["ledger_provenance"]["family_join"]["execution_topology"]
    assert topology["contract"] == "ds4rt.exl3-coordinator-only-v1"
    assert [slot["device"] for slot in topology["coordinator_slots"]] == [
        "cuda:0",
        "cuda:1",
    ]
    assert topology["workers"] == []


def test_overlay_plan_binds_asymmetric_hessian_ownership(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    args.remote_worker = None
    args.mtp_hessian_owner_device_weights = [1, 2]

    plan = MODULE.build_plan(args)

    assert plan["exl3"]["hessian_owner_policy"] == {
        "contract": MODULE.HESSIAN_OWNER_POLICY_CONTRACT,
        "device_weights": [1, 2],
    }
    MODULE._validate_plan(plan)


def test_overlay_plan_rejects_hessian_weight_count_mismatch(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    args.mtp_hessian_owner_device_weights = [1]

    with pytest.raises(MODULE.OverlayError, match="one positive integer"):
        MODULE.build_plan(args)


def test_overlay_run_creation_is_exact_and_resumable(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    plan = MODULE.build_plan(args)

    assert MODULE.prepare_run(plan, resume=False) is False
    run_state = Path(plan["run_state_dir"])
    assert json.loads(
        (run_state / MODULE.base.PLAN_FILENAME).read_text()
    ) == plan
    assert MODULE.prepare_run(plan, resume=True) is False

    export_stage = MODULE._export_stage_path(plan)
    assert export_stage.parent == Path(plan["output"]).parent
    assert export_stage.parent != run_state
    export_stage.mkdir()
    (export_stage / "partial").write_text("incomplete")
    assert MODULE.prepare_run(plan, resume=True) is False
    assert not export_stage.exists()

    (run_state / "unexpected").write_text("drift")
    with pytest.raises(MODULE.OverlayError, match="unexpected entries"):
        MODULE.prepare_run(plan, resume=True)


def test_overlay_plan_ignores_fresh_preflight_observation_time(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    original = MODULE.build_plan(args)
    report = json.loads(args.preflight_report.read_text())
    report["generated_at"] = "2026-08-17T07:41:45.590295+00:00"
    _write_json(args.preflight_report, report)

    assert MODULE.build_plan(args) == original

    report["gpus"][0]["uuid"] = "GPU-replaced-coordinator-0"
    _write_json(args.preflight_report, report)
    assert MODULE.build_plan(args) != original


def test_mtp_activation_boundary_restores_next_block_and_rolls_forward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gptqmodel.looper.input_cache import DiskBackedLayerOutputWriter

    root = tmp_path / "mtp-activations"
    provenance = {"plan_sha256": "a" * 64, "family_join": {"recipe": "test"}}
    first_writer = DiskBackedLayerOutputWriter(
        root,
        layer_index=0,
        expected_batches=2,
        provenance={
            **provenance,
            "block_index": 0,
            "replay_contract": MODULE.MTP_REPLAY_CONTRACT,
        },
        shard_batches=1,
    )
    first_writer.put(0, [torch.zeros(1, 3, 2, 4, dtype=torch.bfloat16)])
    first_writer.put(1, [torch.ones(1, 3, 2, 4, dtype=torch.bfloat16)])
    first = first_writer.finalize()

    class Processor:
        def __init__(self) -> None:
            self.inputs_cache = SimpleNamespace(
                layer_inputs=[[torch.zeros(1)], [torch.zeros(1)]],
            )

        def receive_layer_inputs(self, inputs) -> None:
            self.inputs_cache.layer_inputs = inputs

        @staticmethod
        def completed_layer_checkpoint_entries(layer_index: int):
            return [
                {
                    "module": (
                        f"mtp.{layer_index}.mlp.experts.{expert}.{projection}"
                    )
                }
                for expert in range(2)
                for projection in ("gate_proj", "up_proj", "down_proj")
            ]

    processor = Processor()
    pruned: list[tuple[str, int]] = []
    model = SimpleNamespace(
        turtle_model=SimpleNamespace(
            prune_active_source_scope_through=(
                lambda scope, layer: pruned.append((scope, layer))
            )
        )
    )
    controller = MODULE.MTPActivationBoundaryController(
        root,
        provenance=provenance,
        block_count=3,
        hidden_size=4,
        hc_mult=2,
        proposal_rows=3,
        expert_count=2,
    )
    audited: list[tuple[Path, int, str, int]] = []
    monkeypatch.setattr(
        MODULE,
        "_audit_mtp_checkpoint_block",
        lambda run_state, *, block_index, plan_sha256, expected_projection_count: (
            audited.append(
                (
                    run_state,
                    block_index,
                    plan_sha256,
                    expected_projection_count,
                )
            )
            or {"status": "complete"}
        ),
    )

    assert controller.restore(model=model, processors=[processor]) == 1
    assert processor.inputs_cache.layer_inputs.manifest == first.manifest
    assert pruned == [("mtp", 0)]
    assert audited == [(root.parent, 0, "a" * 64, 6)]

    second_writer = DiskBackedLayerOutputWriter(
        root,
        layer_index=1,
        expected_batches=2,
        provenance={
            **provenance,
            "block_index": 1,
            "replay_contract": MODULE.MTP_REPLAY_CONTRACT,
        },
        shard_batches=1,
    )
    second_writer.put(0, [torch.full((1, 3, 2, 4), 2, dtype=torch.bfloat16)])
    second_writer.put(1, [torch.full((1, 3, 2, 4), 3, dtype=torch.bfloat16)])
    second = second_writer.finalize()
    processor.receive_layer_inputs(second)
    manifest = controller.commit_layer(
        model=model,
        processor=processor,
        layer_index=1,
        layer_name="mtp.1",
    )

    assert manifest == second.manifest
    assert not (root / "layer-000000").exists()
    assert (root / "layer-000001" / "manifest.json").is_file()


def test_overlay_execution_upgrade_binds_image_and_completed_boundary(
    tmp_path: Path,
) -> None:
    from gptqmodel.looper.input_cache import DiskBackedLayerOutputWriter

    args = _fixture(tmp_path)
    args.remote_worker = None
    plan = MODULE.build_plan(args)
    MODULE.prepare_run(plan, resume=False)
    replay = SimpleNamespace(
        anchor_selection_identity={
            "contract": MODULE.ANCHOR_SELECTION_CONTRACT,
            "seed": args.mtp_anchor_sample_seed,
            "source_position_count": 3,
            "selected_position_count": 2,
            "selected_coordinates_sha256": "1" * 64,
        },
        replay_batching_identity={
            "contract": MODULE.SEQUENCE_REPLAY_BATCH_CONTRACT,
            "source_sequence_anchor_cap": None,
            "batch_count": 1,
            "minimum_anchors": 2,
            "maximum_anchors": 2,
            "row_counts_sha256": "2" * 64,
        },
    )
    selection = MODULE._write_or_verify_selection(plan, replay)
    activation_root = Path(plan["run_state_dir"]) / MODULE.ACTIVATION_DIRNAME
    writer = DiskBackedLayerOutputWriter(
        activation_root,
        layer_index=0,
        expected_batches=1,
        provenance={
            **MODULE._activation_provenance(plan, selection),
            "block_index": 0,
            "replay_contract": MODULE.MTP_REPLAY_CONTRACT,
        },
        shard_batches=1,
    )
    writer.put(0, [torch.zeros(2, 5, 2, 16, dtype=torch.bfloat16)])
    first = writer.finalize()

    report = json.loads(args.preflight_report.read_text())
    report["image_digest"] = "sha256:" + "f" * 64
    _write_json(args.preflight_report, report)
    args.run_state_dir = Path(plan["run_state_dir"])
    args.resume = True

    saved, upgrade = MODULE.build_execution_upgrade(args)

    assert saved == plan
    assert upgrade["parent_plan_sha256"] == plan["plan_sha256"]
    assert upgrade["parent_execution"]["image_digest"] == "sha256:" + "d" * 64
    assert upgrade["upgraded_execution"]["image_digest"] == "sha256:" + "f" * 64
    assert upgrade["resume_state"]["layer_index"] == 0
    assert upgrade["resume_state"]["manifest_sha256"] == first.manifest[
        "manifest_sha256"
    ]
    assert MODULE.prepare_run(plan, resume=True) is False

    # An unchanged upgraded image retains its original authorization after the
    # rolling frontier advances; it must not rewrite quantization identity.
    writer = DiskBackedLayerOutputWriter(
        activation_root,
        layer_index=1,
        expected_batches=1,
        provenance={
            **MODULE._activation_provenance(plan, selection),
            "block_index": 1,
            "replay_contract": MODULE.MTP_REPLAY_CONTRACT,
        },
        shard_batches=1,
    )
    writer.put(0, [torch.ones(2, 5, 2, 16, dtype=torch.bfloat16)])
    writer.finalize()
    repeated_plan, repeated_upgrade = MODULE.build_execution_upgrade(args)
    assert repeated_plan == plan
    assert repeated_upgrade == upgrade

    # A later boundary may authorize a narrowly-scoped GPTQModel execution
    # repair while retaining the exact first upgrade in an auditable chain.
    lock = json.loads(args.gptqmodel_lock.read_text())
    lock["revision"] = "3" * 40
    lock["source_tree_sha256"] = "4" * 64
    _write_json(args.gptqmodel_lock, lock)
    report = json.loads(args.preflight_report.read_text())
    report["image_digest"] = "sha256:" + "5" * 64
    report["gptqmodel"] = {
        "revision": lock["revision"],
        "source_tree_sha256": lock["source_tree_sha256"],
    }
    _write_json(args.preflight_report, report)

    replacement_plan, replacement = MODULE.build_execution_upgrade(args)
    assert replacement_plan == plan
    assert replacement["previous_upgrade_sha256"] == upgrade["upgrade_sha256"]
    assert replacement["resume_state"]["layer_index"] == 1
    assert replacement["effective_memory_safety"] == {
        **plan["memory_safety"],
        "cuda_allocation_limit_bytes": (
            MODULE.MTP_RESUME_CUDA_ALLOCATION_LIMIT_BYTES
        ),
    }
    assert replacement["change_contract"]["gptqmodel_source"] == (
        "preferred-forward-device-materialization-only"
    )
    history = (
        Path(plan["run_state_dir"])
        / MODULE.EXECUTION_UPGRADE_HISTORY_DIRNAME
        / f"{upgrade['upgrade_sha256']}.json"
    )
    assert json.loads(history.read_text()) == upgrade
    _, repeated_replacement = MODULE.build_execution_upgrade(args)
    assert repeated_replacement == replacement

    with pytest.raises(MODULE.OverlayError, match="explicit --execution-upgrade"):
        MODULE.execute(plan, resume=True)


def test_overlay_execution_upgrade_rejects_hardware_drift(tmp_path: Path) -> None:
    from gptqmodel.looper.input_cache import DiskBackedLayerOutputWriter

    args = _fixture(tmp_path)
    args.remote_worker = None
    plan = MODULE.build_plan(args)
    MODULE.prepare_run(plan, resume=False)
    replay = SimpleNamespace(
        anchor_selection_identity={
            "contract": MODULE.ANCHOR_SELECTION_CONTRACT,
            "seed": args.mtp_anchor_sample_seed,
            "source_position_count": 2,
            "selected_position_count": 2,
            "selected_coordinates_sha256": "1" * 64,
        },
        replay_batching_identity={
            "contract": MODULE.SEQUENCE_REPLAY_BATCH_CONTRACT,
            "source_sequence_anchor_cap": None,
            "batch_count": 1,
            "minimum_anchors": 2,
            "maximum_anchors": 2,
            "row_counts_sha256": "2" * 64,
        },
    )
    selection = MODULE._write_or_verify_selection(plan, replay)
    writer = DiskBackedLayerOutputWriter(
        Path(plan["run_state_dir"]) / MODULE.ACTIVATION_DIRNAME,
        layer_index=0,
        expected_batches=1,
        provenance={
            **MODULE._activation_provenance(plan, selection),
            "block_index": 0,
            "replay_contract": MODULE.MTP_REPLAY_CONTRACT,
        },
        shard_batches=1,
    )
    writer.put(0, [torch.zeros(2, 5, 2, 16, dtype=torch.bfloat16)])
    writer.finalize()

    report = json.loads(args.preflight_report.read_text())
    report["image_digest"] = "sha256:" + "f" * 64
    report["gpus"][0]["uuid"] = "GPU-replaced-coordinator-0"
    _write_json(args.preflight_report, report)
    args.run_state_dir = Path(plan["run_state_dir"])
    args.resume = True

    with pytest.raises(MODULE.OverlayError, match="changes GPTQModel, Python, Torch, or GPUs"):
        MODULE.build_execution_upgrade(args)


def test_overlay_copy_tree_supports_cross_filesystem_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    (source / "nested").mkdir()
    evidence = source / "nested" / "evidence.json"
    evidence.write_text('{"status":"complete"}\n')
    evidence.chmod(0o600)

    MODULE._copy_tree(source, target)

    copied = target / "nested" / "evidence.json"
    assert copied.read_bytes() == evidence.read_bytes()
    assert copied.stat().st_ino != evidence.stat().st_ino
    assert copied.stat().st_mode & 0o777 == 0o644


def test_overlay_run_creation_accepts_precreated_empty_mounts(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    plan = MODULE.build_plan(args)
    output = Path(plan["output"])
    run_state = Path(plan["run_state_dir"])
    offload = Path(plan["offload_dir"])
    output.mkdir()
    run_state.mkdir()
    offload.mkdir()

    assert MODULE.prepare_run(plan, resume=False) is False
    assert output.exists() is False
    assert run_state.is_dir()
    assert offload.is_dir()
    assert json.loads(
        (run_state / MODULE.base.PLAN_FILENAME).read_text()
    ) == plan


def test_overlay_link_tree_publishes_container_evidence_readable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    evidence = source / "00" / "projection.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("evidence\n")
    evidence.chmod(0o600)

    MODULE._link_tree(source, target)

    published = target / "00" / "projection.json"
    assert published.read_text() == "evidence\n"
    assert stat.S_IMODE(published.stat().st_mode) == 0o644
    assert stat.S_IMODE(evidence.stat().st_mode) == 0o644


def test_overlay_selection_is_written_before_work_and_resume_bound(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    plan = MODULE.build_plan(args)
    MODULE.prepare_run(plan, resume=False)
    replay = SimpleNamespace(
        anchor_selection_identity={
            "contract": MODULE.ANCHOR_SELECTION_CONTRACT,
            "seed": args.mtp_anchor_sample_seed,
            "source_position_count": 3,
            "selected_position_count": 2,
            "selected_coordinates_sha256": "1" * 64,
        },
        replay_batching_identity={
            "contract": MODULE.SEQUENCE_REPLAY_BATCH_CONTRACT,
            "source_sequence_anchor_cap": None,
            "batch_count": 1,
            "minimum_anchors": 2,
            "maximum_anchors": 2,
            "row_counts_sha256": "2" * 64,
        },
    )

    first = MODULE._write_or_verify_selection(plan, replay)
    repeated = MODULE._write_or_verify_selection(plan, replay)

    assert first == repeated
    assert json.loads(
        (Path(plan["run_state_dir"]) / MODULE.SELECTION_FILENAME).read_text()
    ) == first
    replay.anchor_selection_identity["selected_coordinates_sha256"] = "3" * 64
    with pytest.raises(MODULE.OverlayError, match="changed across resume"):
        MODULE._write_or_verify_selection(plan, replay)


def test_overlay_plan_rejects_parent_or_prefix_drift(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    parent = json.loads(args.parent_plan.read_text())
    parent["exl3"]["codebook"] = "sqg"
    parent_body = {key: value for key, value in parent.items() if key != "plan_sha256"}
    parent["plan_sha256"] = MODULE.hashlib.sha256(
        MODULE.base.canonical_json(parent_body)
    ).hexdigest()
    _write_json(args.parent_plan, parent)
    with pytest.raises(MODULE.OverlayError, match="integer-tier target"):
        MODULE.build_plan(args)

    args = _fixture(tmp_path / "prefix-drift")
    manifest_path = args.mtp_prefix_store / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"]["plan_sha256"] = "0" * 64
    _write_json(manifest_path, manifest)
    with pytest.raises(MODULE.OverlayError, match="prefix"):
        MODULE.build_plan(args)


def test_overlay_plan_rejects_nonpositive_anchor_sample_count(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    args.mtp_anchor_sample_count = 0

    with pytest.raises(MODULE.OverlayError, match="anchor-sample-count"):
        MODULE.build_plan(args)


def test_overlay_plan_rejects_redundant_k4_for_native_mxfp4_source(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    args.bits = 4

    with pytest.raises(MODULE.OverlayError, match="K2 or K3"):
        MODULE.build_plan(args)


def test_overlay_plan_binds_optional_source_sequence_cap(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    args.mtp_sequence_anchor_cap = 128

    plan = MODULE.build_plan(args)

    assert plan["replay_batching"]["source_sequence_anchor_cap"] == 128
    assert (
        plan["ledger_provenance"]["family_join"]["replay_batching"]
        == plan["replay_batching"]
    )


def test_load_target_embedding_accepts_official_native_name(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    save_file(
        {"embed.weight": torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)},
        snapshot / "model.safetensors",
    )
    _write_json(
        snapshot / "model.safetensors.index.json",
        {"weight_map": {"embed.weight": "model.safetensors"}},
    )

    embedding = MODULE._load_target_embedding(
        snapshot,
        vocab_size=8,
        hidden_size=4,
    )

    assert embedding.dtype is torch.bfloat16
    assert embedding.tolist() == torch.arange(32).reshape(8, 4).tolist()
