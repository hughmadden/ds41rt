from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

import pytest
import torch

from ds41rt_runtime.exl3_quantizer import (
    analytic_isotropic_hessian,
    qualify_multigpu_batch_equivalence,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("DS41RT_RUN_MULTI_GPU_QUANTIZATION") != "1",
    reason="set DS41RT_RUN_MULTI_GPU_QUANTIZATION=1 for the two-GPU EXL3 gate",
)


def test_pinned_exllamav3_two_gpu_tiles_match_single_gpu_bit_exactly() -> None:
    tools = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools))
    import _pinned_exllamav3

    assert torch.cuda.is_available() and torch.cuda.device_count() == 2
    primary = torch.device("cuda:0")
    generator = torch.Generator(device=primary).manual_seed(20260806)
    original = torch.randn(
        (128, 512), generator=generator, dtype=torch.float32, device=primary
    )

    def quantize(devices: list[int]):
        hessian = analytic_isotropic_hessian(
            features=128,
            equivalent_rows=512,
            key="ds41rt.multigpu_qualification.w1",
            device=primary,
        )
        args = {
            "seed": 20260806,
            "K": 2,
            "devices": devices,
            "device_ratios": None,
            "apply_out_scales": True,
            "debug_dir": tempfile.mkdtemp(),
            "mcg": True,
        }
        reconstructed, proxy_error, output = _pinned_exllamav3.quantize_exl3(
            original.clone(), hessian, args, True
        )
        for device_index in devices:
            torch.cuda.synchronize(device_index)
        return (
            reconstructed.cpu(),
            float(proxy_error),
            {name: tensor.cpu() for name, tensor in output.items()},
        )

    single = quantize([0])
    torch.cuda.reset_peak_memory_stats(1)
    dual = quantize([0, 1])

    assert torch.cuda.max_memory_allocated(1) > 0
    assert torch.equal(single[0], dual[0])
    assert single[1] == dual[1]
    assert single[2].keys() == dual[2].keys()
    for name in single[2]:
        assert torch.equal(single[2][name], dual[2][name]), name


def test_pinned_exllamav3_repeated_batches_fence_two_gpu_tile_streams() -> None:
    tools = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools))
    import _pinned_exllamav3

    with tempfile.TemporaryDirectory() as temporary:
        debug_dir = Path(temporary) / "new-artifact" / ".ds41rt-exl3-debug"
        report = qualify_multigpu_batch_equivalence(
            quantize_exl3_batch=_pinned_exllamav3.quantize_exl3_batch,
            quantization_devices=(0, 1),
            device_ratios=None,
            seed=20260807,
            debug_dir=debug_dir,
            production_projection_shapes=((128, 512), (512, 128)),
        )
        assert not debug_dir.parent.exists()

    assert report is not None
    assert report["status"] == "bit-exact"
    assert report["devices"] == [0, 1]
    assert report["batches"] == 4
    assert report["tensors_per_batch"] == 4
    assert report["production_projection_shapes"] == [[128, 512], [512, 128]]
