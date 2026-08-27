from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

QUANTIZATION = Path(__file__).parents[1]
if str(QUANTIZATION) not in sys.path:
    sys.path.insert(0, str(QUANTIZATION))
SCRIPT = QUANTIZATION / "quantize_flash_gptqmodel.py"
SPEC = importlib.util.spec_from_file_location("quantize_flash_gptqmodel", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _FakeRotary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "main_inv_freq",
            torch.empty(4, device="meta"),
            persistent=False,
        )
        self.register_buffer(
            "compress_inv_freq",
            torch.empty(2, device="meta"),
            persistent=False,
        )


class _FakeLazyDeepSeekModel:
    def __init__(self) -> None:
        self.model = nn.Module()
        self.model.config = SimpleNamespace(
            layer_types=["sliding_attention", "compressed_attention"]
        )
        self.model.model = nn.Module()
        self.model.model.layers = nn.ModuleList([nn.Module(), nn.Module()])
        attention = nn.Module()
        attention.compressor = nn.Module()
        attention.compressor.rotary_emb = _FakeRotary()
        self.model.model.layers[1].self_attn = attention

    def shell_direct_meta_materialize(
        self,
        *,
        target_submodule: nn.Module,
        device: torch.device,
    ) -> None:
        for name, buffer in dict(
            target_submodule.named_buffers(recurse=False)
        ).items():
            if buffer.is_meta:
                target_submodule.register_buffer(
                    name,
                    torch.arange(
                        buffer.numel(),
                        device=device,
                        dtype=buffer.dtype,
                    ),
                    persistent=False,
                )


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_capture_frontier_scope_spans_work_and_restores_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variable = "GPTQMODEL_EXL3_CAPTURE_FRONTIER"
    root = tmp_path / "frontier"
    monkeypatch.delenv(variable, raising=False)

    with MODULE.capture_frontier_scope(root):
        assert MODULE.os.environ[variable] == str(root)
    assert variable not in MODULE.os.environ

    monkeypatch.setenv(variable, "conflicting-frontier")
    with pytest.raises(MODULE.LaunchError, match="conflicts"):
        with MODULE.capture_frontier_scope(root):
            pass
    assert MODULE.os.environ[variable] == "conflicting-frontier"


def test_capture_batch_scope_binds_checkpoint_cadence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_variable = "GPTQMODEL_EXL3_CAPTURE_BATCH_SPOOL"
    interval_variable = "GPTQMODEL_EXL3_CAPTURE_BATCH_CHECKPOINT_INTERVAL"
    root = tmp_path / "capture-batches"
    monkeypatch.delenv(root_variable, raising=False)
    monkeypatch.delenv(interval_variable, raising=False)

    with MODULE.capture_batch_spool_scope(root, checkpoint_interval=64):
        assert MODULE.os.environ[root_variable] == str(root)
        assert MODULE.os.environ[interval_variable] == "64"
    assert root_variable not in MODULE.os.environ
    assert interval_variable not in MODULE.os.environ


def test_lazy_direct_state_preflight_restores_constructor_buffers() -> None:
    report = MODULE.preflight_lazy_nonpersistent_buffers(
        _FakeLazyDeepSeekModel(),
        device="cpu",
        plan_sha256="a" * 64,
    )

    assert report["schema"].endswith("preflight-v1")
    assert report["first_compressed_layer"] == 1
    assert report["owner_count"] == 1
    assert report["buffer_count"] == 2
    assert report["owners"] == [
        {
            "module": "model.layers.1.self_attn.compressor.rotary_emb",
            "buffers": ["compress_inv_freq", "main_inv_freq"],
        }
    ]
    assert len(report["buffers_sha256"]) == 64


def fixture(tmp_path: Path) -> argparse.Namespace:
    tmp_path.mkdir(parents=True, exist_ok=True)
    snapshot = tmp_path / ("a" * 40)
    snapshot.mkdir()
    write_json(
        snapshot / "config.json",
        {
            "num_hidden_layers": 43,
            "n_routed_experts": 2,
            "hidden_size": 4096,
            "moe_intermediate_size": 2048,
            "dspark_target_layer_ids": [40, 41, 42],
            "num_hash_layers": 3,
            "num_experts_per_tok": 2,
            "hc_mult": 4,
            "num_nextn_predict_layers": 1,
        },
    )
    weight_map = {}
    for prefix in (
        *(f"layers.{layer}" for layer in range(43)),
        *(f"mtp.{block}" for block in range(3)),
    ):
        for expert in range(2):
            for projection in ("w1", "w2", "w3"):
                for suffix in ("scale", "weight"):
                    weight_map[
                        f"{prefix}.ffn.experts.{expert}.{projection}.{suffix}"
                    ] = "model-00001.safetensors"
        for projection in ("w1", "w2", "w3"):
            for suffix in ("scale", "weight"):
                weight_map[
                    f"{prefix}.ffn.shared_experts.{projection}.{suffix}"
                ] = "model-00001.safetensors"
    for layer in range(43):
        prefix = f"layers.{layer}.ffn.gate"
        weight_map[f"{prefix}.weight"] = "model-00001.safetensors"
        weight_map[
            f"{prefix}.{'tid2eid' if layer < 3 else 'bias'}"
        ] = "model-00001.safetensors"
    for block in range(3):
        prefix = f"mtp.{block}.ffn.gate"
        weight_map[f"{prefix}.weight"] = "model-00001.safetensors"
        weight_map[f"{prefix}.bias"] = "model-00001.safetensors"
    write_json(
        snapshot / "model.safetensors.index.json",
        {"weight_map": weight_map},
    )
    (snapshot / "model-00001.safetensors").write_bytes(b"source")
    corpus = tmp_path / "calibration.jsonl"
    corpus.write_text(
        json.dumps({"id": "english-1", "text": "Explain tensor parallelism."})
        + "\n"
        + json.dumps({"id": "chinese-1", "text": "解释张量并行。"})
        + "\n",
        encoding="utf-8",
    )
    lock = {
        "schema": 1,
        "repository": "https://github.com/tpurtell/GPTQModel.git",
        "revision": "b" * 40,
        "source_tree_sha256": "c" * 64,
    }
    lock_path = tmp_path / "gptqmodel.lock.json"
    write_json(lock_path, lock)
    preflight = tmp_path / "preflight.json"
    write_json(
        preflight,
        {
            "status": "qualified",
            "role": "coordinator",
            "target_platform": "linux/amd64",
            "cuda_arch": "120",
            "image_digest": "sha256:" + "d" * 64,
            "gptqmodel": {
                "source": "/opt/ds4rt/third_party/gptqmodel",
                "revision": lock["revision"],
                "source_tree_sha256": lock["source_tree_sha256"],
            },
            "python": {"gil_enabled": False, "version": "3.14.6"},
            "torch": {"version": "2.13.0+cu130"},
            "gpus": [
                {"index": 0, "uuid": "GPU-test-0"},
                {"index": 1, "uuid": "GPU-test-1"},
            ],
        },
    )
    return argparse.Namespace(
        snapshot=snapshot,
        calibration_jsonl=corpus,
        preflight_report=preflight,
        output=tmp_path / "output",
        offload_dir=tmp_path / "offload",
        mtp_prefix_store=tmp_path / "prefix-store",
        gptqmodel_lock=lock_path,
        bits=2,
        coordinator_gpu_count=2,
        batch_size=1,
        mtp_replay_batch_size=4,
        mtp_execution_mode=MODULE.MTP_EXECUTION_INTEGRATED,
        plan_only=True,
    )


def test_plan_binds_source_corpus_environment_and_joint_mtp(tmp_path: Path) -> None:
    args = fixture(tmp_path)

    first, texts = MODULE.build_plan(args)
    second, _ = MODULE.build_plan(args)

    assert first == second
    assert first["schema"] == MODULE.PLAN_SCHEMA
    assert (
        first["capture_batch_checkpoint_interval"]
        == MODULE.DEFAULT_CAPTURE_BATCH_CHECKPOINT_INTERVAL
    )
    assert first["ledger_provenance"]["run"][
        "capture_batch_checkpoint_interval"
    ] == MODULE.DEFAULT_CAPTURE_BATCH_CHECKPOINT_INTERVAL
    assert first["mtp_execution_mode"] == MODULE.MTP_EXECUTION_INTEGRATED
    assert first["source"]["revision"] == "a" * 40
    assert first["source"]["total_shard_bytes"] == 6
    assert first["corpus"]["examples"] == 2
    assert first["corpus"]["text_field"] == "text"
    assert texts == ["Explain tensor parallelism.", "解释张量并行。"]
    assert first["exl3"] == {
        "bits": 2,
        "codebook": "mcg",
        "seed": 787,
        "module_include": [MODULE.BASE_EXPERT_PATTERN],
        "fallback": None,
        "out_scales": "auto",
        "sigma_reg": MODULE.EXL3_SIGMA_REG,
        "hessian_capture": MODULE.EXL3_HESSIAN_CAPTURE_CONTRACT,
        "hessian_numerical": MODULE.EXL3_HESSIAN_NUMERICAL_CONTRACT,
        "hessian_symmetry": MODULE.EXL3_HESSIAN_SYMMETRY_CONTRACT,
        "zero_route_recovery": {
            "contract": MODULE.ZERO_ROUTE_RECOVERY_CONTRACT,
            "trigger": MODULE.ZERO_ROUTE_RECOVERY_TRIGGER,
            "sample_source": MODULE.ZERO_ROUTE_RECOVERY_SAMPLE_SOURCE,
            "capture_method": MODULE.ZERO_ROUTE_RECOVERY_CAPTURE_METHOD,
            "selection_policy": MODULE.ZERO_ROUTE_RECOVERY_SELECTION_POLICY,
            "candidate_rank_min": MODULE.ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MIN,
            "candidate_rank_max": MODULE.ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MAX,
            "target_sample_count": MODULE.ZERO_ROUTE_RECOVERY_TARGET_SAMPLE_COUNT,
            "identity_calibration_policy": MODULE.ZERO_ROUTE_RECOVERY_IDENTITY_POLICY,
            "scope": "all-learned-top-k-routers",
        },
    }
    join = first["ledger_provenance"]["family_join"]
    assert join["source"] == first["source"]
    assert join["corpus"] == first["corpus"]
    assert join["operator_contract"].endswith("target-plus-joint-mtp-v1")
    assert join["route_evidence_contract"] == MODULE.ROUTE_EVIDENCE_CONTRACT
    assert (
        join["zero_route_recovery_contract"]
        == MODULE.ZERO_ROUTE_RECOVERY_CONTRACT
    )
    assert join["quantizer_numerics"] == {
        "sigma_reg": MODULE.EXL3_SIGMA_REG,
        "hessian_capture": MODULE.EXL3_HESSIAN_CAPTURE_CONTRACT,
        "hessian_numerical": MODULE.EXL3_HESSIAN_NUMERICAL_CONTRACT,
        "hessian_symmetry": MODULE.EXL3_HESSIAN_SYMMETRY_CONTRACT,
    }
    assert first["projection_checkpoint"] == {
        "contract": MODULE.PROJECTION_CHECKPOINT_CONTRACT,
        "root": str(
            args.output.with_name(f".{args.output.name}.ds4rt-run")
            / MODULE.PROJECTION_CHECKPOINT_DIRNAME
        ),
    }
    assert first["run_state_dir"] == str(
        args.output.with_name(f".{args.output.name}.ds4rt-run")
    )
    assert first["projection_checkpoint_dir"] == first[
        "projection_checkpoint"
    ]["root"]
    assert first["active_layer_source_dir"] == str(
        Path(first["run_state_dir"]) / MODULE.ACTIVE_LAYER_SOURCE_DIRNAME
    )
    assert first["source"]["namespace_audit"]["base_layers"] == 43
    assert first["source"]["namespace_audit"]["mtp_blocks"] == 3
    assert (
        first["ledger_provenance"]["run"]["projection_checkpoint"]
        == first["projection_checkpoint"]
    )
    assert len(first["plan_sha256"]) == 64
    assert not args.output.exists()


def test_plan_binds_exact_inline_base_and_mtp_targets(tmp_path: Path) -> None:
    args = fixture(tmp_path)
    args.base_target_bpw = "2.1"
    args.mtp_target_bpw = "11/5"
    args.mixed_projection_ratio = (3, 5, 8)

    plan, _texts = MODULE.build_plan(args)

    assert plan["exl3"]["bits"] == 2
    assert plan["inline_mixed"]["base"]["extra_bits"] == {
        "numerator": 1,
        "denominator": 10,
    }
    assert plan["inline_mixed"]["base"]["target_bpw"] == "21/10"
    assert plan["inline_mixed"]["mtp"]["extra_bits"] == {
        "numerator": 1,
        "denominator": 5,
    }
    assert plan["inline_mixed"]["mtp"]["target_bpw"] == "11/5"
    assert plan["inline_mixed"]["mtp"]["projection_ratio"] == {
        "w1": 3,
        "w3": 5,
        "w2": 8,
    }
    assert plan["mtp_anchor_selection"] == {
        "contract": "ds4rt-mtp-anchor-stratified-v1",
        "count": 327_680,
        "seed": 20_260_809,
    }
    assert plan["mtp_replay_batching"] == {
        "contract": "ds4rt-mtp-source-sequence-anchor-batches-v1",
        "source_sequence_anchor_cap": None,
        "proposal_rows_per_anchor": 5,
    }
    assert plan["ledger_provenance"]["family_join"]["inline_mixed"][
        "base"
    ].get("tier_plan_root") is None
    MODULE._validate_plan(plan)


def test_plan_supports_local_k3_on_one_visible_coordinator_gpu(
    tmp_path: Path,
) -> None:
    args = fixture(tmp_path)
    preflight = json.loads(args.preflight_report.read_text())
    preflight["gpus"] = [{"index": 0, "uuid": "GPU-physical-one"}]
    write_json(args.preflight_report, preflight)
    args.bits = 3
    args.coordinator_gpu_count = 1

    plan, _ = MODULE.build_plan(args)

    assert plan["recipe"].endswith("3bpw_v4_flash_natural_route")
    assert plan["exl3"]["bits"] == 3
    assert plan["ledger_provenance"]["family_join"]["bits"] == 3
    assert plan["preflight"]["gpus"] == [
        {"index": 0, "uuid": "GPU-physical-one"}
    ]
    assert plan["remote_workers"] is None


def test_quantize_config_metadata_drops_local_tier_plan_root() -> None:
    class Config:
        meta = {
            "ds4rt_inline_mixed": {
                "target_bpw": "21/10",
                "tier_plan_root": "/private/coordinator/frontier",
            },
            "provenance": {"corpus": "next-v1"},
        }

    config = Config()
    MODULE._make_quantize_config_metadata_portable(config)

    assert config.meta == {
        "ds4rt_inline_mixed": {"target_bpw": "21/10"},
        "provenance": {"corpus": "next-v1"},
    }
    assert Config.meta["ds4rt_inline_mixed"]["tier_plan_root"].startswith("/")


def test_external_overlay_mode_is_bound_and_has_a_durable_handoff(
    tmp_path: Path,
) -> None:
    args = fixture(tmp_path)
    args.mtp_execution_mode = MODULE.MTP_EXECUTION_EXTERNAL_OVERLAY
    plan, _ = MODULE.build_plan(args)
    run_state = Path(plan["run_state_dir"])
    run_state.mkdir()
    prefix_root = Path(plan["mtp_prefix_store"])
    prefix_root.mkdir()
    write_json(
        prefix_root / "manifest.json",
        {
            "status": "complete",
            "target_layer_ids": [40, 41, 42],
            "completed_layers": [40, 41, 42],
            "batch_count": 1426,
        },
    )

    record = MODULE.publish_base_prefix_completion(
        plan,
        prefix_manifest_path=prefix_root / "manifest.json",
    )

    assert plan["ledger_provenance"]["run"]["mtp_execution_mode"] == (
        MODULE.MTP_EXECUTION_EXTERNAL_OVERLAY
    )
    assert record["source_batch_count"] == 1426
    assert MODULE.validate_base_prefix_completion(plan) == record

    write_json(
        prefix_root / "manifest.json",
        {
            "status": "complete",
            "target_layer_ids": [40, 41],
            "completed_layers": [40, 41],
            "batch_count": 1426,
        },
    )
    with pytest.raises(MODULE.LaunchError, match="inconsistent"):
        MODULE.validate_base_prefix_completion(plan)


def test_new_plans_default_to_integrated_mtp(tmp_path: Path) -> None:
    args = fixture(tmp_path)
    del args.mtp_execution_mode

    plan, _ = MODULE.build_plan(args)

    assert plan["mtp_execution_mode"] == MODULE.MTP_EXECUTION_INTEGRATED
    assert plan["ledger_provenance"]["run"]["mtp_execution_mode"] == (
        MODULE.MTP_EXECUTION_INTEGRATED
    )


def test_plan_binds_qualified_calibration_manifest_and_router_screen(
    tmp_path: Path,
) -> None:
    args = fixture(tmp_path)
    texts, corpus = MODULE.calibration_stream(args.calibration_jsonl)
    assert len(texts) == 2
    calibration_records = []
    for index, text in enumerate(texts):
        calibration_records.append(
            {
                "id": ("english-1", "chinese-1")[index],
                "prompt_sha256": MODULE.hashlib.sha256(text.encode()).hexdigest(),
                "token_ids_sha256": f"{index + 1:064x}",
            }
        )
    screening_records = [{"id": "screen-000"}]
    screening_sha256 = "7" * 64
    manifest = {
        "schema": MODULE.CALIBRATION_MANIFEST_SCHEMA,
        "builder": {"revision": "8" * 40, "sha256": "9" * 64},
        "training_data_snapshot": {"revision": "a" * 40},
        "tokenizer_sha256": "b" * 64,
        "screening_calibration_identity_subset": True,
        "source_group_overlap": [],
        "splits": {
            "calibration": {
                "file": args.calibration_jsonl.name,
                "sha256": corpus["file_sha256"],
                "summary": {"records": 2, "prompt_tokens": 1234},
                "records": calibration_records,
            },
            "screening": {
                "derived_from": "calibration",
                "sha256": screening_sha256,
                "summary": {"records": 1, "prompt_tokens": 123},
                "records": screening_records,
            },
        },
    }
    args.calibration_manifest = tmp_path / "manifest.json"
    write_json(args.calibration_manifest, manifest)
    route_screen = {
        "schema": MODULE.ROUTE_SCREEN_SCHEMA,
        "checkpoint": str(args.snapshot),
        "corpora": {
            "screening": {"sha256": screening_sha256, "prompts": 1}
        },
        "prompts": [{"id": "screen-000", "mtp_verify_cycles": 2}],
        "distributions": {
            "screening": {
                "layers": [
                    {
                        "layer_id": layer_id,
                        "rows": 256,
                        "routes": 1536,
                        "zero_hit_experts": 0 if layer_id < 43 else 1,
                    }
                    for layer_id in range(46)
                ]
            }
        },
    }
    args.route_screen_report = tmp_path / "routes.json"
    write_json(args.route_screen_report, route_screen)

    plan, _ = MODULE.build_plan(args)

    evidence = plan["calibration_evidence"]
    assert evidence["manifest"]["builder_revision"] == "8" * 40
    assert evidence["manifest"]["training_data_revision"] == "a" * 40
    assert evidence["router_screen"]["status"] == "qualified"
    assert evidence["router_screen"]["base_zero_hit_experts"] == 0
    assert evidence == plan["ledger_provenance"]["family_join"][
        "calibration_evidence"
    ]
    assert plan["quantization_toolchain"] == plan["ledger_provenance"][
        "family_join"
    ]["quantization_toolchain"]
    assert set(plan["quantization_toolchain"]["files"]) == {
        "collect_deepseek_v4_route_screen.py",
        "preflight.py",
        "quantize_flash_gptqmodel.py",
        "deepseek_v4_layer_boundary_store.py",
        "deepseek_v4_mtp_prefix_store.py",
    }


def test_plan_binds_manifest_and_observes_routes_inline_without_screen(
    tmp_path: Path,
) -> None:
    args = fixture(tmp_path)
    texts, corpus = MODULE.calibration_stream(args.calibration_jsonl)
    calibration_records = [
        {
            "id": identifier,
            "prompt_sha256": MODULE.hashlib.sha256(text.encode()).hexdigest(),
            "token_ids_sha256": f"{index + 1:064x}",
        }
        for index, (identifier, text) in enumerate(
            zip(("english-1", "chinese-1"), texts, strict=True)
        )
    ]
    manifest = {
        "schema": MODULE.CALIBRATION_MANIFEST_SCHEMA,
        "builder": {"revision": "8" * 40, "sha256": "9" * 64},
        "training_data_snapshot": {"revision": "a" * 40},
        "tokenizer_sha256": "b" * 64,
        "screening_calibration_identity_subset": True,
        "source_group_overlap": [],
        "splits": {
            "calibration": {
                "file": args.calibration_jsonl.name,
                "sha256": corpus["file_sha256"],
                "summary": {"records": 2, "prompt_tokens": 1234},
                "records": calibration_records,
            },
            "screening": {
                "derived_from": "calibration",
                "sha256": "7" * 64,
                "summary": {"records": 1, "prompt_tokens": 123},
                "records": [{"id": "screen-000"}],
            },
        },
    }
    args.calibration_manifest = tmp_path / "manifest.json"
    write_json(args.calibration_manifest, manifest)
    args.route_qualification = MODULE.ROUTE_QUALIFICATION_INLINE
    args.route_screen_report = None

    plan, _ = MODULE.build_plan(args)

    evidence = plan["calibration_evidence"]
    assert evidence["route_qualification"] == {
        "mode": MODULE.ROUTE_QUALIFICATION_INLINE,
        "status": "deferred-to-full-corpus-capture",
        "scope": "base-and-integrated-mtp",
        "natural_route_contract": MODULE.ROUTE_EVIDENCE_CONTRACT,
        "recovery_contract": MODULE.ZERO_ROUTE_RECOVERY_CONTRACT,
        "recovery_trigger": MODULE.ZERO_ROUTE_RECOVERY_TRIGGER,
        "target_effective_rows": 1024,
        "failure_policy": "fail-only-on-router-or-evidence-invariant",
    }
    assert "router_screen" not in evidence
    assert set(plan["quantization_toolchain"]["files"]) == {
        "preflight.py",
        "quantize_flash_gptqmodel.py",
        "deepseek_v4_layer_boundary_store.py",
        "deepseek_v4_mtp_prefix_store.py",
    }


def test_inline_route_qualification_rejects_a_screen_report(tmp_path: Path) -> None:
    args = fixture(tmp_path)
    args.calibration_manifest = tmp_path / "manifest.json"
    args.calibration_manifest.write_text("{}", encoding="utf-8")
    args.route_screen_report = tmp_path / "routes.json"
    args.route_screen_report.write_text("{}", encoding="utf-8")
    args.route_qualification = MODULE.ROUTE_QUALIFICATION_INLINE

    with pytest.raises(MODULE.LaunchError, match="mode and router-screen"):
        MODULE.build_plan(args)


def test_plan_binds_sorted_spark_workers_and_six_slot_scheduler(
    tmp_path: Path,
) -> None:
    args = fixture(tmp_path)
    coordinator = json.loads(args.preflight_report.read_text())
    declarations = []
    for name in ("spark-d", "spark-a", "spark-c", "spark-b"):
        worker = dict(coordinator)
        worker["role"] = "expert"
        worker["target_platform"] = "linux/arm64"
        worker["cuda_arch"] = "121"
        worker["image_digest"] = "sha256:" + "e" * 64
        worker["gpus"] = [{"index": 0, "uuid": f"GPU-{name}"}]
        worker_path = tmp_path / f"{name}-preflight.json"
        write_json(worker_path, worker)
        declarations.append([name, f"http://{name}:17841", str(worker_path)])
    args.remote_worker = declarations
    args.remote_token_env = "TEST_EXL3_TOKEN"
    args.remote_timeout_seconds = 3600.0
    args.remote_max_attempts = 3

    plan, _ = MODULE.build_plan(args)
    remote = plan["remote_workers"]
    assert [endpoint["name"] for endpoint in remote["endpoints"]] == [
        "spark-a",
        "spark-b",
        "spark-c",
        "spark-d",
    ]
    assert [slot["device"] for slot in remote["coordinator_slots"]] == [
        "cuda:0",
        "cuda:1",
    ]
    assert [slot["gpu_uuid"] for slot in remote["coordinator_slots"]] == [
        "GPU-test-0",
        "GPU-test-1",
    ]
    assert remote["orchestration_workers"] == 14
    assert remote["cuda_workers_per_device"] == 7
    assert remote["max_attempts"] == 3
    assert remote["token_env"] == "TEST_EXL3_TOKEN"
    assert (
        plan["ledger_provenance"]["run"]["remote_workers"]
        == plan["remote_workers"]
    )
    topology = plan["ledger_provenance"]["family_join"]["execution_topology"]
    assert topology["scheduler"] == "dynamic-pipelined-slot-projection-v2"
    expected_assignment_store = str(
        args.output.with_name(f".{args.output.name}.ds4rt-run")
        / MODULE.REMOTE_ASSIGNMENT_DIRNAME
    )
    assert remote["assignment_store"] == expected_assignment_store
    assert topology["assignment_store"] == expected_assignment_store
    assert topology["coordinator_slots"] == remote["coordinator_slots"]
    assert [worker["name"] for worker in topology["workers"]] == [
        "spark-a",
        "spark-b",
        "spark-c",
        "spark-d",
    ]


def test_execution_upgrade_preserves_parent_plan_and_is_stable(
    tmp_path: Path,
) -> None:
    args = fixture(tmp_path)
    coordinator = json.loads(args.preflight_report.read_text())
    declarations = []
    for name in ("spark-a", "spark-b", "spark-c", "spark-d"):
        worker = json.loads(json.dumps(coordinator))
        worker["role"] = "expert"
        worker["target_platform"] = "linux/arm64"
        worker["cuda_arch"] = "121"
        worker["image_digest"] = "sha256:" + "e" * 64
        worker["gpus"] = [{"index": 0, "uuid": f"GPU-{name}"}]
        worker_path = tmp_path / f"{name}-upgrade-preflight.json"
        write_json(worker_path, worker)
        declarations.append([name, f"http://{name}:17841", str(worker_path)])
    args.remote_worker = declarations
    args.remote_token_env = "TEST_EXL3_TOKEN"
    args.remote_timeout_seconds = 3600.0
    args.remote_max_attempts = 2
    parent, _ = MODULE.build_plan(args)
    MODULE.prepare_run(parent, resume=False)

    upgraded_lock = json.loads(args.gptqmodel_lock.read_text())
    upgraded_lock["revision"] = "f" * 40
    upgraded_lock["source_tree_sha256"] = "1" * 64
    write_json(args.gptqmodel_lock, upgraded_lock)
    upgraded_preflight = json.loads(args.preflight_report.read_text())
    upgraded_preflight["image_digest"] = "sha256:" + "2" * 64
    upgraded_preflight["gptqmodel"]["revision"] = upgraded_lock["revision"]
    upgraded_preflight["gptqmodel"]["source_tree_sha256"] = upgraded_lock[
        "source_tree_sha256"
    ]
    write_json(args.preflight_report, upgraded_preflight)

    resumed, texts, first = MODULE.build_execution_upgrade(args)
    repeated, repeated_texts, second = MODULE.build_execution_upgrade(args)

    assert resumed == parent == repeated
    assert texts == repeated_texts
    assert first == second
    assert first["parent_plan_sha256"] == parent["plan_sha256"]
    assert first["upgraded_execution"]["image_digest"] == "sha256:" + "2" * 64
    assert first["change_contract"]["quantization_algorithm"] == (
        "declared-seeded-true-sequential-v1"
    )
    assert first["change_contract"]["concurrent_rng"] == (
        "projection-local-generator-v1"
    )
    assert first["change_contract"][
        "capture_batch_checkpoint_interval"
    ] == MODULE.DEFAULT_CAPTURE_BATCH_CHECKPOINT_INTERVAL
    run_state = Path(parent["run_state_dir"])
    assert json.loads(
        (run_state / MODULE.EXECUTION_UPGRADE_FILENAME).read_text()
    ) == first

    upgraded_lock["revision"] = "3" * 40
    upgraded_lock["source_tree_sha256"] = "4" * 64
    write_json(args.gptqmodel_lock, upgraded_lock)
    upgraded_preflight["image_digest"] = "sha256:" + "5" * 64
    upgraded_preflight["gptqmodel"]["revision"] = upgraded_lock["revision"]
    upgraded_preflight["gptqmodel"]["source_tree_sha256"] = upgraded_lock[
        "source_tree_sha256"
    ]
    write_json(args.preflight_report, upgraded_preflight)
    _, _, replacement = MODULE.build_execution_upgrade(args)
    assert replacement["previous_failed_upgrade_sha256"] == first["upgrade_sha256"]
    archived = (
        run_state
        / MODULE.EXECUTION_UPGRADE_HISTORY_DIRNAME
        / f"{first['upgrade_sha256']}.json"
    )
    assert json.loads(archived.read_text()) == first
    assert MODULE._read_execution_upgrade(run_state, parent) == replacement

    boundary_root = run_state / MODULE.LAYER_BOUNDARY_DIRNAME
    boundary_root.mkdir()
    boundary_body = {
        "schema": "ds4rt.deepseek-v4-layer-boundary",
        "schema_version": 2,
        "payload_hash_algorithm": "xxh3-128",
        "plan_sha256": parent["plan_sha256"],
        "layer_index": 2,
        "layer_name": "model.layers.2",
        "activation_batches": 2,
        "activation_bytes": 4096,
        "completed_projection_entries": [{"module": "test"}],
    }
    boundary_digest = MODULE.hashlib.sha256(
        MODULE.canonical_json(boundary_body)
    ).hexdigest()
    boundary_manifest = {
        **boundary_body,
        "manifest_sha256": boundary_digest,
    }
    committed_boundary = (
        boundary_root / f"layer-000002-{boundary_digest[:16]}"
    )
    committed_boundary.mkdir()
    write_json(committed_boundary / "manifest.json", boundary_manifest)
    journal_payload = b'{"test":true}\n'
    (run_state / MODULE.ERROR_JOURNAL_FILENAME).write_bytes(journal_payload)
    upgraded_lock["revision"] = "6" * 40
    upgraded_lock["source_tree_sha256"] = "7" * 64
    write_json(args.gptqmodel_lock, upgraded_lock)
    upgraded_preflight["image_digest"] = "sha256:" + "8" * 64
    upgraded_preflight["gptqmodel"]["revision"] = upgraded_lock["revision"]
    upgraded_preflight["gptqmodel"]["source_tree_sha256"] = upgraded_lock[
        "source_tree_sha256"
    ]
    write_json(args.preflight_report, upgraded_preflight)
    _, _, boundary_replacement = MODULE.build_execution_upgrade(args)
    assert boundary_replacement["previous_upgrade_sha256"] == replacement[
        "upgrade_sha256"
    ]
    assert boundary_replacement["resume_state"] == {
        "contract": "latest-boundary-plus-journal-v1",
        "layer_boundary": {
            "directory": committed_boundary.name,
            "layer_index": 2,
            "layer_name": "model.layers.2",
            "manifest_sha256": boundary_digest,
            "activation_batches": 2,
            "activation_bytes": 4096,
            "completed_projection_entries": 1,
        },
        "error_journal": {
            "bytes": len(journal_payload),
            "records": 1,
            "sha256": MODULE.hashlib.sha256(journal_payload).hexdigest(),
        },
    }
    _, _, repeated_boundary_replacement = MODULE.build_execution_upgrade(args)
    assert repeated_boundary_replacement == boundary_replacement

    args.batch_size = 2
    with pytest.raises(MODULE.LaunchError, match="inputs differ"):
        MODULE.build_execution_upgrade(args)


def test_first_execution_upgrade_binds_existing_boundary_and_journal(
    tmp_path: Path,
) -> None:
    args = fixture(tmp_path)
    parent, _ = MODULE.build_plan(args)
    MODULE.prepare_run(parent, resume=False)
    run_state = Path(parent["run_state_dir"])
    boundary_root = run_state / MODULE.LAYER_BOUNDARY_DIRNAME
    boundary_root.mkdir()
    boundary_body = {
        "schema": "ds4rt.deepseek-v4-layer-boundary",
        "schema_version": 2,
        "payload_hash_algorithm": "xxh3-128",
        "plan_sha256": parent["plan_sha256"],
        "layer_index": 8,
        "layer_name": "model.layers.8",
        "activation_batches": 1426,
        "activation_bytes": 4096,
        "completed_projection_entries": [{"module": "test"}],
    }
    boundary_digest = MODULE.hashlib.sha256(
        MODULE.canonical_json(boundary_body)
    ).hexdigest()
    boundary_manifest = {
        **boundary_body,
        "manifest_sha256": boundary_digest,
    }
    committed_boundary = boundary_root / f"layer-000008-{boundary_digest[:16]}"
    committed_boundary.mkdir()
    write_json(committed_boundary / "manifest.json", boundary_manifest)
    journal_payload = b'{"test":true}\n'
    (run_state / MODULE.ERROR_JOURNAL_FILENAME).write_bytes(journal_payload)

    upgraded_lock = json.loads(args.gptqmodel_lock.read_text())
    upgraded_lock["revision"] = "f" * 40
    upgraded_lock["source_tree_sha256"] = "1" * 64
    write_json(args.gptqmodel_lock, upgraded_lock)
    upgraded_preflight = json.loads(args.preflight_report.read_text())
    upgraded_preflight["image_digest"] = "sha256:" + "2" * 64
    upgraded_preflight["gptqmodel"]["revision"] = upgraded_lock["revision"]
    upgraded_preflight["gptqmodel"]["source_tree_sha256"] = upgraded_lock[
        "source_tree_sha256"
    ]
    write_json(args.preflight_report, upgraded_preflight)

    _, _, upgrade = MODULE.build_execution_upgrade(args)

    assert upgrade.get("previous_upgrade_sha256") is None
    assert upgrade["resume_state"] == {
        "contract": "latest-boundary-plus-journal-v1",
        "layer_boundary": {
            "directory": committed_boundary.name,
            "layer_index": 8,
            "layer_name": "model.layers.8",
            "manifest_sha256": boundary_digest,
            "activation_batches": 1426,
            "activation_bytes": 4096,
            "completed_projection_entries": 1,
        },
        "error_journal": {
            "bytes": len(journal_payload),
            "records": 1,
            "sha256": MODULE.hashlib.sha256(journal_payload).hexdigest(),
        },
    }
    assert upgrade["change_contract"]["resume_state"] == (
        "content-bound-immutable-frontier"
    )


def test_plan_rejects_stale_or_duplicate_spark_workers(tmp_path: Path) -> None:
    args = fixture(tmp_path)
    worker = json.loads(args.preflight_report.read_text())
    worker["role"] = "expert"
    worker["target_platform"] = "linux/arm64"
    worker["cuda_arch"] = "121"
    worker["gpus"] = [{"index": 0, "uuid": "GPU-spark-a"}]
    worker["gptqmodel"]["source_tree_sha256"] = "e" * 64
    declarations = []
    for name in ("spark-a", "spark-b", "spark-c", "spark-d"):
        worker_copy = json.loads(json.dumps(worker))
        worker_copy["gpus"] = [{"index": 0, "uuid": f"GPU-{name}"}]
        worker_path = tmp_path / f"{name}.json"
        write_json(worker_path, worker_copy)
        declarations.append([name, f"http://{name}:17841", str(worker_path)])
    args.remote_worker = declarations
    with pytest.raises(MODULE.LaunchError, match="differs from the source lock"):
        MODULE.build_plan(args)

    worker["gptqmodel"]["source_tree_sha256"] = "c" * 64
    write_json(worker_path, worker)
    args.remote_worker = [
        ["spark-a", "http://spark-a:17841", str(worker_path)],
        ["spark-a", "http://spark-b:17841", str(worker_path)],
    ]
    with pytest.raises(MODULE.LaunchError, match="names must be unique"):
        MODULE.build_plan(args)


def test_plan_rejects_moving_snapshot_and_preflight_drift(tmp_path: Path) -> None:
    args = fixture(tmp_path)
    moving = tmp_path / "main"
    args.snapshot.rename(moving)
    args.snapshot = moving
    with pytest.raises(MODULE.LaunchError, match="immutable"):
        MODULE.build_plan(args)

    args = fixture(tmp_path / "second")
    preflight = json.loads(args.preflight_report.read_text())
    preflight["gptqmodel"]["source_tree_sha256"] = "e" * 64
    write_json(args.preflight_report, preflight)
    with pytest.raises(MODULE.LaunchError, match="preflight GPTQModel identity"):
        MODULE.build_plan(args)


def test_plan_accepts_only_content_addressed_huggingface_shard_links(
    tmp_path: Path,
) -> None:
    args = fixture(tmp_path / "model" / "snapshots")
    shard = args.snapshot / "model-00001.safetensors"
    blob_root = tmp_path / "model" / "blobs"
    blob_root.mkdir()
    blob_sha256 = "d" * 64
    shard.rename(blob_root / blob_sha256)
    shard.symlink_to(Path("..") / ".." / "blobs" / blob_sha256)

    plan, _ = MODULE.build_plan(args)
    assert plan["source"]["shards"] == [
        {
            "name": shard.name,
            "bytes": 6,
            "hf_blob_sha256": blob_sha256,
        }
    ]

    shard.unlink()
    outside = tmp_path / "outside"
    outside.write_bytes(b"source")
    shard.symlink_to(Path("..") / ".." / ".." / outside.name)
    with pytest.raises(MODULE.LaunchError, match="canonical Hugging Face blob"):
        MODULE.build_plan(args)


def test_calibration_stream_rejects_duplicate_ids(tmp_path: Path) -> None:
    corpus = tmp_path / "duplicate.jsonl"
    corpus.write_text(
        '{"id":"same","text":"one"}\n{"id":"same","text":"two"}\n',
        encoding="utf-8",
    )
    with pytest.raises(MODULE.LaunchError, match="duplicate ids"):
        MODULE.calibration_stream(corpus)


def test_calibration_stream_accepts_prompt_schema_and_rejects_ambiguity(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "prompts.jsonl"
    corpus.write_text(
        '{"id":"one","prompt":"first"}\n{"id":"two","prompt":"second"}\n',
        encoding="utf-8",
    )
    texts, identity = MODULE.calibration_stream(corpus)
    assert texts == ["first", "second"]
    assert identity["text_field"] == "prompt"

    corpus.write_text(
        '{"id":"one","text":"first","prompt":"ambiguous"}\n',
        encoding="utf-8",
    )
    with pytest.raises(MODULE.LaunchError, match="exactly one"):
        MODULE.calibration_stream(corpus)

    corpus.write_text(
        '{"id":"one","text":"first"}\n{"id":"two","prompt":"second"}\n',
        encoding="utf-8",
    )
    with pytest.raises(MODULE.LaunchError, match="mixes"):
        MODULE.calibration_stream(corpus)


def test_plan_rejects_nested_work_paths(tmp_path: Path) -> None:
    args = fixture(tmp_path)
    args.offload_dir = args.output / "offload"
    with pytest.raises(MODULE.LaunchError, match="distinct and non-nested"):
        MODULE.build_plan(args)


def test_run_resume_requires_exact_plan_and_preserves_partial_export(
    tmp_path: Path,
) -> None:
    args = fixture(tmp_path)
    plan, _ = MODULE.build_plan(args)
    run_state = Path(plan["run_state_dir"])
    stage = MODULE._export_stage_path(plan)

    assert MODULE.prepare_run(plan, resume=False) is False
    assert not args.output.exists()
    assert json.loads((run_state / MODULE.PLAN_FILENAME).read_text()) == plan
    assert (run_state / MODULE.ERROR_JOURNAL_FILENAME).read_bytes() == b""
    assert args.offload_dir.is_dir()
    assert args.mtp_prefix_store.is_dir()

    checkpoint = run_state / MODULE.PROJECTION_CHECKPOINT_DIRNAME / "keep"
    checkpoint.parent.mkdir(exist_ok=True)
    checkpoint.write_bytes(b"packed")
    journal = run_state / MODULE.ERROR_JOURNAL_FILENAME
    journal.write_bytes(b'{"durable":true}\n')
    stage.mkdir()
    (stage / "partial.safetensors").write_bytes(b"partial")

    assert MODULE.prepare_run(plan, resume=True) is False
    assert (stage / "partial.safetensors").read_bytes() == b"partial"
    assert checkpoint.read_bytes() == b"packed"
    assert journal.read_bytes() == b'{"durable":true}\n'

    changed = json.loads(json.dumps(plan))
    changed["target_batch_size"] = 2
    changed_body = {
        key: value for key, value in changed.items() if key != "plan_sha256"
    }
    changed["plan_sha256"] = MODULE.hashlib.sha256(
        MODULE.canonical_json(changed_body)
    ).hexdigest()
    with pytest.raises(MODULE.LaunchError, match="plan differs"):
        MODULE.prepare_run(changed, resume=True)


def test_run_resume_requires_initialized_error_journal(tmp_path: Path) -> None:
    args = fixture(tmp_path)
    plan, _ = MODULE.build_plan(args)
    MODULE.prepare_run(plan, resume=False)
    journal = Path(plan["run_state_dir"]) / MODULE.ERROR_JOURNAL_FILENAME
    journal.unlink()

    with pytest.raises(MODULE.LaunchError, match="error journal is unavailable"):
        MODULE.prepare_run(plan, resume=True)


def test_run_resume_fails_closed_on_unexpected_state(tmp_path: Path) -> None:
    args = fixture(tmp_path)
    plan, _ = MODULE.build_plan(args)
    MODULE.prepare_run(plan, resume=False)
    run_state = Path(plan["run_state_dir"])
    (run_state / "unowned.txt").write_text("do not delete", encoding="utf-8")

    with pytest.raises(MODULE.LaunchError, match="unexpected entries"):
        MODULE.prepare_run(plan, resume=True)
    assert (run_state / "unowned.txt").read_text(encoding="utf-8") == "do not delete"


def test_run_resume_accepts_owned_inline_mixed_state(tmp_path: Path) -> None:
    args = fixture(tmp_path)
    args.base_target_bpw = 2.1
    args.mtp_target_bpw = 2.2
    args.mixed_projection_ratio = (3, 5, 8)
    plan, _ = MODULE.build_plan(args)
    MODULE.prepare_run(plan, resume=False)
    run_state = Path(plan["run_state_dir"])
    (run_state / MODULE.INLINE_MIXED_CANDIDATE_JOURNAL_FILENAME).touch()
    (run_state / MODULE.INLINE_MIXED_TIER_PLAN_DIRNAME).mkdir()

    assert MODULE.prepare_run(plan, resume=True) is False


def test_nonmixed_run_rejects_inline_mixed_state(tmp_path: Path) -> None:
    args = fixture(tmp_path)
    plan, _ = MODULE.build_plan(args)
    MODULE.prepare_run(plan, resume=False)
    run_state = Path(plan["run_state_dir"])
    (run_state / MODULE.INLINE_MIXED_CANDIDATE_JOURNAL_FILENAME).touch()

    with pytest.raises(MODULE.LaunchError, match="unexpected entries"):
        MODULE.prepare_run(plan, resume=True)


def test_run_resume_accepts_owned_integrated_mtp_activation_state(
    tmp_path: Path,
) -> None:
    args = fixture(tmp_path)
    plan, _ = MODULE.build_plan(args)
    MODULE.prepare_run(plan, resume=False)
    mtp_activations = Path(plan["run_state_dir"]) / MODULE.MTP_ACTIVATION_DIRNAME
    mtp_activations.mkdir()

    assert MODULE.prepare_run(plan, resume=True) is False


def test_run_resume_refuses_symlinked_partial_export(tmp_path: Path) -> None:
    args = fixture(tmp_path)
    plan, _ = MODULE.build_plan(args)
    MODULE.prepare_run(plan, resume=False)
    run_state = Path(plan["run_state_dir"])
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_bytes(b"user")
    MODULE._export_stage_path(plan).symlink_to(outside, target_is_directory=True)

    with pytest.raises(MODULE.LaunchError, match="not a regular directory"):
        MODULE.prepare_run(plan, resume=True)
    assert (outside / "keep").read_bytes() == b"user"


def test_export_is_hashed_and_atomically_published(tmp_path: Path) -> None:
    args = fixture(tmp_path)
    plan, _ = MODULE.build_plan(args)
    MODULE.prepare_run(plan, resume=False)
    run_state = Path(plan["run_state_dir"])
    stage = MODULE._export_stage_path(plan)
    assert stage.parent == args.output.parent
    assert stage.parent != run_state
    stage.mkdir()
    (stage / "model.safetensors").write_bytes(b"quantized")
    nested = stage / "tokenizer"
    nested.mkdir()
    (nested / "tokenizer.json").write_text("{}\n", encoding="utf-8")

    MODULE.publish_export(plan, mtp_replay_batches=7)

    assert args.output.is_dir()
    assert not stage.exists()
    run = MODULE.validate_published_artifact(
        args.output,
        plan,
        verify_file_hashes=True,
    )
    assert run["mtp_replay_batches"] == 7
    assert MODULE.prepare_run(plan, resume=True) is True

    (args.output / "model.safetensors").write_bytes(b"corrupted")
    with pytest.raises(MODULE.LaunchError, match="failed hashing"):
        MODULE.validate_published_artifact(
            args.output,
            plan,
            verify_file_hashes=True,
        )


def test_fresh_run_rejects_nonempty_output_and_work_paths(tmp_path: Path) -> None:
    args = fixture(tmp_path)
    plan, _ = MODULE.build_plan(args)
    args.output.mkdir()
    (args.output / "existing").write_bytes(b"user")
    with pytest.raises(MODULE.LaunchError, match="output directory is not empty"):
        MODULE.prepare_run(plan, resume=False)
    assert (args.output / "existing").read_bytes() == b"user"


def test_missing_remote_token_does_not_create_run_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = fixture(tmp_path)
    worker = json.loads(args.preflight_report.read_text())
    worker["role"] = "expert"
    worker["target_platform"] = "linux/arm64"
    worker["cuda_arch"] = "121"
    worker["image_digest"] = "sha256:" + "e" * 64
    declarations = []
    for name in ("spark-a", "spark-b", "spark-c", "spark-d"):
        worker_copy = json.loads(json.dumps(worker))
        worker_copy["gpus"] = [{"index": 0, "uuid": f"GPU-{name}"}]
        worker_path = tmp_path / f"{name}.json"
        write_json(worker_path, worker_copy)
        declarations.append([name, f"http://{name}:17841", str(worker_path)])
    args.remote_worker = declarations
    args.remote_token_env = "TEST_MISSING_EXL3_TOKEN"
    args.remote_timeout_seconds = 3600.0
    args.remote_max_attempts = 2
    plan, texts = MODULE.build_plan(args)
    monkeypatch.delenv(args.remote_token_env, raising=False)

    with pytest.raises(MODULE.LaunchError, match="token env"):
        MODULE.execute(plan, texts)
    assert not Path(plan["run_state_dir"]).exists()
    assert not args.output.exists()
