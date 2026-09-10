from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

import pytest
import torch

from ds41rt_runtime.exl3_quantizer import (
    SafetensorsArtifactWriter,
    build_artifact_plan,
    run_layerwise_quantization,
)
from ds41rt_runtime.exl3_experts import (
    load_exl3_expert_reference_layer,
    load_exl3_expert_tp_layer,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("DS41RT_RUN_GPU_QUALIFICATION") != "1",
    reason="set DS41RT_RUN_GPU_QUALIFICATION=1 for the pinned EXL3 CUDA gate",
)


def test_pinned_exllamav3_calibrated_k2_quantization() -> None:
    tools = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools))
    import _pinned_exllamav3

    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    device = torch.device("cuda:0")
    torch.manual_seed(20260805)
    samples = torch.randn((512, 128), dtype=torch.float32, device=device)
    original = torch.randn((128, 128), dtype=torch.float32, device=device) * 0.02
    hessian = {
        "H": samples.T @ samples,
        "first_key": "ds41rt.gpu_qualification.w1",
        "count": 512,
        "finalized": False,
        "num_total": samples.numel(),
        "inf_nan": torch.zeros(2, dtype=torch.long, device=device),
        "device": device,
    }
    quant_args = {
        "seed": 20260805,
        "K": 2,
        "devices": [device],
        "device_ratios": None,
        "apply_out_scales": True,
        "debug_dir": tempfile.mkdtemp(),
        "mcg": True,
    }
    reconstructed, proxy_error, output = _pinned_exllamav3.quantize_exl3(
        original.clone(),
        hessian,
        quant_args,
        True,
    )
    cosine = torch.nn.functional.cosine_similarity(
        reconstructed.flatten(), original.flatten(), dim=0
    )
    assert not quant_args["q_fallback"]
    assert 0.0 < float(proxy_error) < 0.2
    assert float(cosine) > 0.9
    assert tuple(output["trellis"].shape) == (8, 8, 32)
    assert tuple(output["suh"].shape) == (128,)
    assert tuple(output["svh"].shape) == (128,)
    assert output["mcg"].view(torch.uint32).item() == 0xCBAC1FED


@pytest.fixture(scope="module")
def tiny_exl3_snapshot(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from test_exl3_quantizer import make_native_snapshot

    tmp_path = tmp_path_factory.mktemp("tiny-exl3")
    tools = Path(__file__).resolve().parents[1] / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    import _pinned_exllamav3

    snapshot = make_native_snapshot(tmp_path)
    output = tmp_path / "exl3"
    plan = build_artifact_plan(snapshot, max_shard_bytes=256 * 1024)
    writer = SafetensorsArtifactWriter(
        plan,
        output,
        calibration_rows=128,
        seed=19,
    )
    writer.copy_native_tensors()
    report = run_layerwise_quantization(
        plan,
        writer,
        quantize_exl3_batch=_pinned_exllamav3.quantize_exl3_batch,
        calibration_rows=128,
        seed=19,
        batch_experts=1,
        debug_dir=tmp_path / "debug",
    )
    writer.finish(report)
    return output


def test_layerwise_converter_publishes_tiny_hybrid_snapshot(
    tiny_exl3_snapshot: Path,
) -> None:
    output = tiny_exl3_snapshot
    report = __import__("json").loads(
        (output / "ds41rt-exl3-calibration.json").read_text(encoding="utf-8")
    )
    assert (output / "config.json").is_file()
    assert len(report["layers"]) == 1
    assert len(report["layers"][0]["projections"]) == 3
    assert all(
        projection["trellis_bits"] == 2
        for projection in report["layers"][0]["projections"]
    )


def test_tiny_exl3_tp4_sum_matches_unsharded_and_captures(
    tiny_exl3_snapshot: Path,
) -> None:
    tools = Path(__file__).resolve().parents[1] / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    import _pinned_sparkinfer  # noqa: F401

    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    device = torch.device("cuda:0")
    torch.manual_seed(20260805)
    hidden = (torch.randn((2, 128), device=device) * 1.0e-2).to(torch.bfloat16)
    topk_ids = torch.zeros((2, 6), dtype=torch.int32, device=device)
    topk_weights = torch.full((2, 6), 1.0 / 6.0, dtype=torch.float32, device=device)

    reference_layer = load_exl3_expert_reference_layer(
        tiny_exl3_snapshot, 0, expert_ids=[0]
    )
    reference = reference_layer.run_partial(hidden, topk_ids, topk_weights).clone()
    torch.cuda.synchronize(device)
    del reference_layer
    torch.cuda.empty_cache()

    partials = []
    source_bytes = []
    for rank in range(4):
        layer = load_exl3_expert_tp_layer(
            tiny_exl3_snapshot, 0, tp_rank=rank, expert_ids=[0]
        )
        plan, scratch = layer.plan_tp(max_tokens=2)
        output = torch.empty((2, 128), dtype=torch.float32, device=device)
        eager = layer.run_partial(
            hidden,
            topk_ids,
            topk_weights,
            plan=plan,
            scratch=scratch,
            output=output,
        ).clone()
        if rank == 0:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = layer.run_partial(
                    hidden,
                    topk_ids,
                    topk_weights,
                    plan=plan,
                    scratch=scratch,
                    output=output,
                )
            graph.replay()
            torch.cuda.synchronize(device)
            assert torch.equal(captured, eager)
        partials.append(eager)
        source_bytes.append(layer.source_bytes)
        del layer, plan, scratch, output
        torch.cuda.empty_cache()

    tp_sum = torch.stack(partials).sum(dim=0)
    relative_error = (tp_sum - reference).norm() / reference.norm().clamp_min(1.0e-9)
    cosine = torch.nn.functional.cosine_similarity(
        tp_sum.flatten(), reference.flatten(), dim=0
    )
    assert len(set(source_bytes)) == 1
    assert float(relative_error) <= 3.0e-2
    assert float(cosine) >= 0.999
