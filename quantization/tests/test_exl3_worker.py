from __future__ import annotations

import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from gptqmodel.looper.exllamav3_processor import (
    EXL3Processor,
    prepare_exl3_hessian,
)
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.utils.exl3_projection_checkpoint import build_projection_request
from gptqmodel.utils.exl3_remote import (
    EXL3RemoteClient,
    RemoteEndpoint,
)
from torch import nn

QUANTIZATION = Path(__file__).parents[1]
if str(QUANTIZATION) not in sys.path:
    sys.path.insert(0, str(QUANTIZATION))
SCRIPT = QUANTIZATION / "exl3_worker.py"
SPEC = importlib.util.spec_from_file_location("ds41rt_exl3_worker", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_preflight(path: Path, *, role: str = "expert") -> None:
    path.write_text(
        json.dumps(
            {
                "status": "qualified",
                "role": role,
                "target_platform": (
                    "linux/arm64" if role == "expert" else "linux/amd64"
                ),
                "cuda_arch": "121" if role == "expert" else "120",
                "image_digest": "sha256:" + "d" * 64,
                "gptqmodel": {
                    "revision": "a" * 40,
                    "source_tree_sha256": "b" * 64,
                },
                "python": {"gil_enabled": False, "version": "3.14.6"},
                "torch": {"version": "2.13.0+cu130"},
                "gpus": [{"index": 0, "uuid": "GPU-spark-a"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_unexpected_failure_evidence_is_bounded_and_content_bound() -> None:
    try:
        raise KeyError("missing quantizer metric")
    except KeyError as error:
        message, trace = MODULE.unexpected_failure_evidence(error)

    assert "KeyError" in message
    assert "missing quantizer metric" in message
    assert "traceback_sha256=" in message
    assert "raise KeyError" in trace
    assert len(trace) <= 8240


def packed_tensors() -> dict[str, torch.Tensor]:
    return {
        "trellis": torch.zeros((8, 8, 32), dtype=torch.int16),
        "suh": torch.ones(128, dtype=torch.float16),
        "svh": torch.ones(128, dtype=torch.float16),
        "mcg": torch.tensor([-877912083], dtype=torch.int32),
    }


class Capture:
    def __init__(self, module: NamedModule, hessian: torch.Tensor) -> None:
        self.module = module
        self.H = hessian
        self.nsamples = 1024

    def finalize_hessian(self, target_device=None):
        self.H = self.H.to(target_device)
        return self.H

    def clone_module(self, copy=True, device=None):
        return self.module.module.weight.detach().to(device=device, copy=copy).float()

    def free(self):
        self.H = None


def test_prepare_exl3_hessian_restores_raw_xtx_sum() -> None:
    sample_count = 1024
    raw_xtx = torch.tensor([[7.0, 2.0], [2.0, 5.0]], dtype=torch.float32)
    normalized = raw_xtx * (2.0 / sample_count)
    capture = SimpleNamespace(
        H=normalized.clone(),
        nsamples=sample_count,
        finalize_hessian=lambda target_device=None: None,
    )

    prepared = prepare_exl3_hessian(
        capture,
        target_device=torch.device("cpu"),
        module_full_name="model.layers.0.mlp.experts.0.gate_proj",
    )

    torch.testing.assert_close(prepared, raw_xtx)


def processor_and_module(
    *,
    checkpoint_root: Path,
    endpoint: RemoteEndpoint,
    weight: torch.Tensor,
    hessian: torch.Tensor,
    expert_id: int = 31,
    coordinator_slots: list[dict] | None = None,
) -> tuple[EXL3Processor, NamedModule]:
    linear = nn.Linear(128, 128, bias=False, dtype=torch.bfloat16, device="cuda:0")
    linear.weight.data.copy_(weight)
    module = NamedModule(
        linear,
        f"mlp.experts.{expert_id}.gate_proj",
        f"model.layers.7.mlp.experts.{expert_id}.gate_proj",
        7,
    )
    processor = EXL3Processor.__new__(EXL3Processor)
    processor.qcfg = SimpleNamespace(
        meta={
            "ds41rt_error_ledger": {
                "family_join": {"source_revision": "test-source"},
                "run": {
                    "projection_checkpoint": {
                        "contract": "ds41rt.exl3-projection-checkpoint-v1",
                        "root": str(checkpoint_root),
                    },
                    "remote_workers": {
                        "contract": MODULE.REMOTE_CONTRACT,
                        "scheduler": "dynamic-pipelined-slot-projection-v2",
                        "token_env": "TEST_EXL3_WORKER_TOKEN",
                        "assignment_store": str(
                            checkpoint_root.parent / "dynamic-projection-assignments"
                        ),
                        "coordinator_slots": coordinator_slots or [],
                        "timeout_seconds": 30,
                        "max_attempts": 2,
                        "orchestration_workers": 3 + len(coordinator_slots or []),
                        "endpoints": [endpoint.__dict__],
                    },
                },
            }
        },
        dynamic=None,
    )
    processor.tasks = {
        module.name: {
            "capture": Capture(module, hessian.clone()),
            "qcfg": SimpleNamespace(
                head_bits=None,
                runtime_bits=2,
                out_scales="auto",
                codebook="mcg",
            ),
        }
    }
    processor.lm_head_name = "lm_head"
    processor.error_journal_path = str(checkpoint_root.parent / "error-journal.jsonl")
    processor._stats_lock = threading.Lock()
    processor._remote_client_initialized = False
    processor._remote_client = None
    processor._distributed_local_quant_locks = {}
    processor.durations = []
    processor.avg_losses = []
    processor.module_names = []
    processor.log = []
    processor.draw_progress = lambda *args, **kwargs: None
    processor.formatted_fwd_time = lambda: "0.000"
    processor.device_memory_report = lambda: "test"
    processor.log_new_row = lambda *args, **kwargs: None
    return processor, module


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="dual-RTX coordinator control requires two CUDA devices",
)
@pytest.mark.parametrize(
    ("expert_id", "occupy_first_slot", "expected_device"),
    [(3, False, "cuda:0"), (1, True, "cuda:1")],
)
def test_processor_executes_each_explicit_coordinator_slot(
    tmp_path: Path,
    monkeypatch,
    expert_id: int,
    occupy_first_slot: bool,
    expected_device: str,
) -> None:
    monkeypatch.setenv("TEST_EXL3_WORKER_TOKEN", "test-secret")
    endpoint = RemoteEndpoint(
        name="spark-a",
        url="http://127.0.0.1:1",
        preflight_sha256="a" * 64,
        image_digest="sha256:" + "b" * 64,
    )
    coordinator_slots = [
        {
            "device": f"cuda:{index}",
            "gpu_uuid": f"GPU-coordinator-{index}",
            "preflight_sha256": "c" * 64,
            "image_digest": "sha256:" + "d" * 64,
        }
        for index in (0, 1)
    ]
    torch.manual_seed(787)
    weight = (
        torch.randn((128, 128), dtype=torch.float32, device="cuda:0") * 0.02
    ).to(torch.bfloat16)
    activations = torch.randn((1024, 128), dtype=torch.float32, device="cuda:0")
    hessian = (2.0 / activations.shape[0]) * activations.T @ activations
    processor, module = processor_and_module(
        checkpoint_root=tmp_path / f"coordinator-checkpoints-{expert_id}",
        endpoint=endpoint,
        weight=weight,
        hessian=hessian,
        expert_id=expert_id,
        coordinator_slots=coordinator_slots,
    )

    occupied = None
    if occupy_first_slot:
        client = processor._remote_client_for_run(processor._ledger_provenance())
        occupied = client.acquire_slot("test.occupied.cuda0.projection")
        assert occupied.slot.device == "cuda:0"
    try:
        processor.process(module)
    finally:
        if occupied is not None:
            occupied.release()
    module.stream_sync()

    stat = processor.log[-1]
    assert stat["exl3_execution_contract"]["kind"] == "coordinator"
    assert stat["exl3_execution_contract"]["device"] == expected_device
    assert stat["exl3_execution_contract"]["gpu_uuid"] == (
        f"GPU-coordinator-{expected_device[-1]}"
    )
    assert stat["exl3_execution_result"]["kind"] == "coordinator"
    assert stat["exl3_error_ledger_record"]["devices"] == [expected_device]


def test_worker_identity_requires_qualified_expert_preflight(tmp_path: Path) -> None:
    report = tmp_path / "worker.json"
    write_preflight(report)
    identity = MODULE.build_worker_identity("spark-a", report)
    assert identity["name"] == "spark-a"
    assert identity["contract"] == MODULE.REMOTE_CONTRACT
    assert len(identity["preflight_sha256"]) == 64

    preflight = json.loads(report.read_text())
    preflight["generated_at"] = "2026-08-17T00:00:00+00:00"
    report.write_text(json.dumps(preflight) + "\n")
    assert MODULE.build_worker_identity("spark-a", report)[
        "preflight_sha256"
    ] == identity["preflight_sha256"]

    write_preflight(report, role="coordinator")
    with pytest.raises(MODULE.WorkerError, match="remote-worker contract"):
        MODULE.build_worker_identity("spark-a", report)


def test_authenticated_loopback_protocol_binds_identity_and_tensors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report = tmp_path / "worker.json"
    write_preflight(report)
    identity = MODULE.build_worker_identity("spark-a", report)
    token = b"test-secret"

    def fake_execute_remote_projection(**kwargs):
        assert kwargs["worker_identity"] == identity
        return (
            packed_tensors(),
            {
                "duration_seconds": 1.25,
                "proxy_error": 0.125,
                "device_names": ["remote:spark-a/cuda:0"],
                "quantizer_metrics": {"reported_metric_kind": "test"},
                "worker": identity,
            },
            False,
        )

    monkeypatch.setattr(
        MODULE,
        "execute_remote_projection",
        fake_execute_remote_projection,
    )
    server = MODULE.EXL3WorkerServer(
        ("127.0.0.1", 0),
        identity=identity,
        token=token,
        checkpoint_root=tmp_path / "checkpoints",
        max_body_bytes=64 * 1024 * 1024,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = RemoteEndpoint(
        name="spark-a",
        url=f"http://127.0.0.1:{server.server_port}",
        preflight_sha256=identity["preflight_sha256"],
        image_digest=identity["image_digest"],
    )
    client = EXL3RemoteClient(
        endpoints=[endpoint],
        token=token,
        coordinator_slots=[],
        timeout_seconds=5,
        max_attempts=1,
        max_body_bytes=64 * 1024 * 1024,
    )
    weight = torch.zeros((128, 128), dtype=torch.float32)
    hessian = torch.eye(128, dtype=torch.float32)
    request = build_projection_request(
        module_full_name="model.layers.7.mlp.experts.31.gate_proj",
        layer_index=7,
        input_weight=weight,
        hessian=hessian,
        sample_count=1024,
        quantizer_contract={
            "bits": 2,
            "codebook": "mcg",
            "hessian_capture": "raw-xtx-sum-fp32-v1",
            "apply_out_scales": None,
            "sigma_reg": 0.025,
            "seed": 787,
            "execution": client.execution_contract(endpoint),
        },
        family_join={"source_revision": "test"},
        route_evidence=None,
    )
    try:
        tensors, result, transport = client.quantize(
            endpoint=endpoint,
            request_manifest=request,
            input_weight=weight,
            hessian=hessian,
        )
        assert result["worker"] == identity
        assert transport == {
            "attempts": 1,
            "retry_errors": [],
            "worker_checkpoint_hit": False,
        }
        assert set(tensors) == {"trellis", "suh", "svh", "mcg"}

        wrong_token = EXL3RemoteClient(
            endpoints=[endpoint],
            token=b"wrong-secret",
            coordinator_slots=[],
            timeout_seconds=5,
            max_attempts=1,
        )
        with pytest.raises(RuntimeError, match="invalid error response"):
            wrong_token.qualify(endpoint)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="EXL3 requires CUDA")
def test_processor_remote_quantization_resumes_with_worker_offline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report = tmp_path / "worker.json"
    write_preflight(report)
    identity = MODULE.build_worker_identity("spark-a", report)
    token = b"test-secret"
    monkeypatch.setenv("TEST_EXL3_WORKER_TOKEN", token.decode())
    server = MODULE.EXL3WorkerServer(
        ("127.0.0.1", 0),
        identity=identity,
        token=token,
        checkpoint_root=tmp_path / "worker-checkpoints",
        max_body_bytes=64 * 1024 * 1024,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = RemoteEndpoint(
        name="spark-a",
        url=f"http://127.0.0.1:{server.server_port}",
        preflight_sha256=identity["preflight_sha256"],
        image_digest=identity["image_digest"],
    )
    torch.manual_seed(787)
    weight = (
        torch.randn((128, 128), dtype=torch.float32, device="cuda:0") * 0.02
    ).to(torch.bfloat16)
    activations = torch.randn((1024, 128), dtype=torch.float32, device="cuda:0")
    hessian = (2.0 / activations.shape[0]) * activations.T @ activations
    coordinator_checkpoints = tmp_path / "coordinator-checkpoints"
    first_processor, first_module = processor_and_module(
        checkpoint_root=coordinator_checkpoints,
        endpoint=endpoint,
        weight=weight,
        hessian=hessian,
    )
    try:
        first_processor.process(first_module)
        first_module.stream_sync()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    first_replay = first_module.module.weight.detach().cpu().clone()
    first_stat = first_processor.log[-1]
    assert first_stat["exl3_projection_checkpoint_hit"] is False
    assert first_stat["exl3_execution_contract"]["name"] == "spark-a"
    assert first_stat["exl3_execution_result"]["kind"] == "remote_worker"
    assert (
        first_stat["exl3_execution_result"]["transport"]["worker_checkpoint_hit"]
        is False
    )

    second_processor, second_module = processor_and_module(
        checkpoint_root=coordinator_checkpoints,
        endpoint=endpoint,
        weight=weight,
        hessian=hessian,
    )
    second_processor.process(second_module)
    second_module.stream_sync()
    assert second_processor.log[-1]["exl3_projection_checkpoint_hit"] is True
    assert torch.equal(
        second_module.module.weight.detach().cpu().view(torch.int16),
        first_replay.view(torch.int16),
    )
