import pytest
import torch

from ds4rt_reference.deepseek_v4_mhc_capture import (
    capture_deepseek_v4_mhc_terminal,
    plan_deepseek_v4_mhc,
    prepare_deepseek_v4_mhc_terminal,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for mHC graph tests"
)


def _device_buffer(tensor: torch.Tensor) -> dict[str, int]:
    return {
        "ptr": tensor.data_ptr(),
        "bytes": tensor.numel() * tensor.element_size(),
        "device_id": tensor.device.index,
    }


def test_flash_terminal_graph_retains_raw_and_normalized_hidden_on_gpu0() -> None:
    device = torch.device("cuda", 0)
    rows, max_rows, hidden = 2, 8, 4_096
    contract = plan_deepseek_v4_mhc(variant="flash", max_rows=max_rows)
    generator = torch.Generator(device=device).manual_seed(29)
    tensors = {
        "scratch": torch.empty(
            (contract.scratch.total_bytes,), device=device, dtype=torch.uint8
        ),
        "delta": torch.randn(
            (rows, hidden),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        ),
        "residual": torch.randn(
            (rows, 4, hidden),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        ),
        "prev_post": torch.randn(
            (rows, 4), device=device, dtype=torch.float32, generator=generator
        ),
        "prev_comb": torch.randn(
            (rows, 4, 4),
            device=device,
            dtype=torch.float32,
            generator=generator,
        ),
        "terminal_residual": torch.empty(
            (rows, 4, hidden), device=device, dtype=torch.bfloat16
        ),
        "collapsed_output": torch.empty(
            (rows, hidden), device=device, dtype=torch.bfloat16
        ),
        "normalized_output": torch.empty(
            (rows, hidden), device=device, dtype=torch.bfloat16
        ),
        "aux_hidden_output": torch.empty(
            (rows, hidden), device=device, dtype=torch.bfloat16
        ),
        "hc_head_fn": torch.randn(
            (4, 4 * hidden),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        / 128,
        "hc_head_scale": torch.tensor([0.8], device=device),
        "hc_head_base": torch.linspace(-0.2, 0.2, 4, device=device),
        "norm_weight": torch.linspace(
            0.8, 1.2, hidden, device=device, dtype=torch.float32
        ).to(torch.bfloat16),
    }
    source_delta = tensors["delta"].clone()
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    ctx = {
        "buffers": {name: _device_buffer(value) for name, value in tensors.items()},
        "cuda_stream": stream.cuda_stream,
    }
    kwargs = {"variant": "flash", "rows": rows, "max_rows": max_rows}

    prepare_deepseek_v4_mhc_terminal(ctx, **kwargs)
    stream.synchronize()
    eager_terminal = tensors["terminal_residual"].clone()
    eager_collapsed = tensors["collapsed_output"].clone()
    eager_normalized = tensors["normalized_output"].clone()

    tensors["delta"].copy_(source_delta)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        capture_deepseek_v4_mhc_terminal(ctx, **kwargs)
    tensors["delta"].copy_(source_delta)
    graph.replay()
    torch.cuda.synchronize(device)

    torch.testing.assert_close(
        tensors["terminal_residual"], eager_terminal, rtol=0, atol=0
    )
    torch.testing.assert_close(
        tensors["collapsed_output"], eager_collapsed, rtol=0, atol=0
    )
    torch.testing.assert_close(
        tensors["normalized_output"], eager_normalized, rtol=0, atol=0
    )
    assert not torch.equal(
        tensors["collapsed_output"], tensors["normalized_output"]
    )

    tensors["residual"].add_(
        torch.tensor(0.125, device=device, dtype=torch.bfloat16)
    )
    tensors["delta"].copy_(source_delta)
    graph.replay()
    torch.cuda.synchronize(device)
    assert not torch.equal(tensors["terminal_residual"], eager_terminal)
    assert torch.isfinite(tensors["collapsed_output"]).all()
    assert torch.isfinite(tensors["normalized_output"]).all()
