#!/usr/bin/env python3
"""Check automatic sparse window widths against explicit per-query baseline launches."""
import argparse
import ctypes as C
import json
from pathlib import Path
import torch

class View(C.Structure):
    _fields_ = [('values', C.c_void_p * 4), ('scales', C.c_void_p * 4),
        ('window_end', C.c_void_p), ('pages', C.c_void_p), ('source_end', C.c_void_p),
        ('window_capacity', C.c_uint64), ('source_capacity', C.c_uint64),
        ('source_proposal_capacity', C.c_uint64), ('page_stride', C.c_uint32),
        ('compressed', C.c_uint32)]

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--baseline', required=True)
p.add_argument('--candidate', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert not a.output.exists()
torch.cuda.set_device(0)
torch.manual_seed(4931)
libs = [C.CDLL(path) for path in (a.baseline, a.candidate)]
fns = []
for lib in libs:
    assert lib.ds41rt_v41_sparse_attention_initialize() == 0
    fn = lib.ds41rt_v41_sparse_attention
    fn.argtypes = [C.c_void_p] * 5 + [C.c_int32, C.c_int32, C.POINTER(View), C.c_void_p]
    fns.append(fn)
q = (torch.randn(6, 64, 512, device='cuda') * .2).bfloat16()
sink = torch.randn(64, device='cuda')
results = []
for compressed in (0, 1, 2):
    values = [torch.randn(n, 512, device='cuda').to(torch.float8_e4m3fn).view(torch.uint8) for n in (128, 6, 256, 1)]
    scales = [torch.full((n, 16), 125, dtype=torch.uint8, device='cuda') for n in (128, 6, 256, 1)]
    if compressed == 2:
        for i in (2, 3):
            n = values[i].shape[0]
            values[i] = torch.randint(0, 256, (n, 256), dtype=torch.uint8, device='cuda')
            scales[i] = torch.full((n, 32), 56, dtype=torch.uint8, device='cuda')
    end = torch.tensor([49], dtype=torch.uint64, device='cuda')
    source_end = torch.tensor([12], dtype=torch.uint64, device='cuda')
    pages = torch.tensor([0], dtype=torch.int32, device='cuda')
    meta = torch.empty((6, 10), dtype=torch.uint64, device='cuda')
    selected = torch.full((6, 512), -1, dtype=torch.int32, device='cuda')
    selected[:, :12] = torch.arange(12, device='cuda')
    view = View((C.c_void_p * 4)(*[v.data_ptr() for v in values]),
        (C.c_void_p * 4)(*[s.data_ptr() for s in scales]), end.data_ptr(), pages.data_ptr(),
        source_end.data_ptr(), 6, 256, 1, 1, compressed)
    reference = torch.empty_like(q)
    outputs = {rows: torch.empty_like(q[:rows]) for rows in (1, 2, 6)}
    def launch(fn, rows, width, out, offset=0):
        status = fn(q[offset:].data_ptr(), sink.data_ptr(), meta[offset:].data_ptr(),
            selected[offset:].data_ptr(), out.data_ptr(), rows, width, C.byref(view),
            torch.cuda.current_stream().cuda_stream)
        assert status == 0, status
    graphs = {}
    for start in (49, 60, 63, 124, 127, 128, 2048):
        end.fill_(start)
        meta.copy_(torch.tensor([[start, 0, 6, start+r, 0, 12, 12, 0, 0, 1] for r in range(6)], dtype=torch.uint64, device='cuda'))
        for row in range(6):
            launch(fns[0], 1, min(start + row + 1, 128), reference[row:row+1], row)
        for rows, out in outputs.items():
            if rows not in graphs:
                launch(fns[1], rows, 0, out)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    launch(fns[1], rows, 0, out)
                graphs[rows] = graph
            graphs[rows].replay()
            torch.cuda.synchronize()
            assert torch.isfinite(out).all()
            assert torch.equal(out, reference[:rows]), (compressed, start, rows, float((out.float()-reference[:rows].float()).abs().max()))
            results.append(dict(compressed=compressed, start=start, rows=rows, bit_exact=True))
a.output.write_text(json.dumps(dict(scope=__doc__, cases=results), indent=2)+'\n')
print(f'PASS {len(results)} automatic-width cases, including changed-position graph replay')
