"""Verify and import DS41RT's pinned exllamav3 quantizer source."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "third_party/exllamav3"
LOCK = REPO_ROOT / "third_party/exllamav3.lock.json"
VERIFY_SCRIPT = REPO_ROOT / "scripts/verify-exllamav3-source.py"

spec = importlib.util.spec_from_file_location("ds41rt_verify_exllamav3", VERIFY_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load exllamav3 verifier from {VERIFY_SCRIPT}")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
LOCK_DATA = verifier.verify(SOURCE, LOCK)

source_string = str(SOURCE)
if source_string not in sys.path:
    sys.path.insert(0, source_string)

from exllamav3.modules.quant.exl3_lib.quantize import (  # noqa: E402
    get_quant_stream as _get_quant_stream,
    quantize_exl3,
    quantize_exl3_batch as _quantize_exl3_batch,
)
from exllamav3.version import __version__  # noqa: E402

if __version__ != "1.3.0":
    raise RuntimeError(f"pinned exllamav3 version changed: {__version__}")

REVISION = str(LOCK_DATA["revision"])
SOURCE_TREE_SHA256 = str(LOCK_DATA["source_tree_sha256"])


def quantize_exl3_batch(
    weights,
    hessians,
    quant_args_list,
    progress_str=None,
    verbose=False,
):
    """Run batched multi-GPU search on exllamav3's own primary stream.

    The pinned-host tile fan-out uses a cached non-default stream on the
    primary device.  ``quantize_exl3_batch`` performs its scale-search tensor
    work on the caller stream, so entering the upstream function from the
    default stream otherwise races both the initial pinned copy and the
    returned tile gather.  Fence device-to-device with CUDA events by making
    the quantizer stream wait for caller inputs and the caller wait for final
    outputs.  This preserves asynchronous two-GPU execution and avoids a host
    synchronization between tile calls.
    """

    if not quant_args_list:
        return _quantize_exl3_batch(
            weights,
            hessians,
            quant_args_list,
            progress_str,
            verbose,
        )
    devices = quant_args_list[0].get("devices")
    if not isinstance(devices, list) or len(devices) <= 1:
        return _quantize_exl3_batch(
            weights,
            hessians,
            quant_args_list,
            progress_str,
            verbose,
        )
    primary = devices[0]
    caller_stream = torch.cuda.current_stream(primary)
    quantizer_stream = _get_quant_stream(primary)
    if caller_stream.cuda_stream == quantizer_stream.cuda_stream:
        return _quantize_exl3_batch(
            weights,
            hessians,
            quant_args_list,
            progress_str,
            verbose,
        )
    quantizer_stream.wait_stream(caller_stream)
    with torch.cuda.stream(quantizer_stream):
        results = _quantize_exl3_batch(
            weights,
            hessians,
            quant_args_list,
            progress_str,
            verbose,
        )
    caller_stream.wait_stream(quantizer_stream)
    return results


__all__ = [
    "REVISION",
    "SOURCE_TREE_SHA256",
    "quantize_exl3",
    "quantize_exl3_batch",
]
