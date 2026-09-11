#!/usr/bin/env python3
"""Direct FP8 paged/window proposal attention against independent blockwise math."""
import argparse
import ctypes as C
import hashlib
import importlib.util
import json
from pathlib import Path
import torch
import tvm_ffi


class View(C.Structure):
    _fields_ = [('values', C.c_void_p * 4), ('scales', C.c_void_p * 4),
                ('window_end', C.c_void_p), ('pages', C.c_void_p), ('source_end', C.c_void_p),
                ('window_proposal_capacity', C.c_uint64), ('source_capacity', C.c_uint64),
                ('source_proposal_capacity', C.c_uint64), ('page_stride', C.c_uint32),
                ('compressed', C.c_uint32)]


def dense(t):
    return torch.empty(t.shape, dtype=t.dtype, device=t.device).copy_(t)


def oracle(q, kv, mask, sink):
    # Transcription of the pinned reference's 64-key online softmax. FP32 QK/PV
    # arithmetic, BF16 probabilities, FP32 normalizer (not BF16 probabilities).
    maximum = torch.full(q.shape[:2], -1e30, dtype=torch.float32, device=q.device)
    total = torch.zeros_like(maximum)
    out = torch.zeros_like(q, dtype=torch.float32)
    for begin in range(0, kv.shape[1], 64):
        v = kv[:, begin:begin + 64].float()
        score = torch.bmm(q.float(), v.transpose(1, 2)) * (512 ** -.5)
        score.masked_fill_(~mask[:, None, begin:begin + 64], -torch.inf)
        new = torch.maximum(maximum, score.amax(-1))
        scale = (maximum - new).exp()
        p = (score - new[..., None]).exp()
        total = total * scale + p.sum(-1)
        out = out * scale[..., None] + torch.bmm(p.bfloat16().float(), v)
        maximum = new
    total += (sink - maximum).exp()
    return (out / total[..., None]).bfloat16()


def large_pool(launch, stream):
    capacity = 16777216
    pool = torch.empty(capacity, 512, dtype=torch.uint8)
    ps = torch.empty(capacity, 16, dtype=torch.uint8)
    def packed(rows, value):
        return torch.full((rows, 512), value, dtype=torch.float32).to(torch.float8_e4m3fn).view(torch.uint8)
    ring, private, source_private = packed(128, .25), packed(1, 1.), packed(1, 3.)
    scale = torch.full((128, 16), 127, dtype=torch.uint8)
    pool[-2].copy_(packed(1, 2.)[0]); ps[-2].fill_(127)
    pool[254].copy_(packed(1, 4.)[0]); ps[254].fill_(127)
    pages = torch.full((4096,), 2**32 - 1, dtype=torch.uint32)
    end = torch.tensor([1048575], dtype=torch.uint64)
    meta = torch.tensor([[1048575, 0, 1, 1048575, 0, 1048576, 1048575, 1, 0, 1]], dtype=torch.uint64)
    selected = torch.full((1, 512), -1, dtype=torch.int32)
    selected[0, :2] = torch.tensor([1048574, 1048575], dtype=torch.int32)
    q = torch.zeros(1, 64, 512, dtype=torch.bfloat16)
    sink = torch.zeros(64, dtype=torch.float32)
    out = torch.empty_like(q)
    view = View((C.c_void_p * 4)(ring.data_ptr(), private.data_ptr(), pool.data_ptr(), source_private.data_ptr()),
                (C.c_void_p * 4)(scale.data_ptr(), scale.data_ptr(), ps.data_ptr(), scale.data_ptr()),
                end.data_ptr(), pages.data_ptr(), end.data_ptr(), 1, capacity, 1, 4096, 1)
    def run():
        assert launch(q.data_ptr(), sink.data_ptr(), meta.data_ptr(), selected.data_ptr(), out.data_ptr(),
                      1, 128, C.byref(view), stream.cuda_stream) == 0
    pages[-1] = 65535
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream): run()
    cases = []
    for page, numerator, denominator in [(65535, 37.75, 131), (0, 39.75, 131), (65536, 35.75, 130)]:
        pages[-1] = page
        graph.replay()
        torch.testing.assert_close(out, torch.full_like(out, numerator / denominator), rtol=0, atol=0)
        cases.append(dict(page=page, physical_capacity=capacity, maximum_page_stride=4096,
                          analytic_exact=True, changed_graph=True, invalid_page_masked=page==65536))
        print('PASS large', cases[-1], flush=True)
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-lib', type=Path, required=True)
    p.add_argument('--reference-dir', type=Path, required=True)
    p.add_argument('--large-only', action='store_true')
    p.add_argument('--committed-source', action='store_true', help='Global cache includes the entire query range, with no private source rows')
    p.add_argument('--bounded-replay', action='store_true', help='Qualify mutable decoder SWA lower bounds')
    p.add_argument('--split-parts', type=int, choices=range(1, 11), default=None,
                   help='Exercise the split ABI for rows <=16, sequential ABI above16')
    p.add_argument('--device', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / 'docs/ds41-reference-lock.json').read_text())
    reference = a.reference_dir / 'inference/kernel.py'
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == lock['files']['inference/kernel.py']
    spec = importlib.util.spec_from_file_location('official_sparse_reference', reference)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    assert ref.tilelang.__version__ == '0.1.8'
    torch.cuda.set_device(a.device)
    torch.manual_seed(41512640)
    torch.backends.cuda.matmul.allow_tf32 = False
    lib = C.CDLL(str(a.native_lib))
    init = lib.ds41rt_v41_sparse_attention_initialize
    init.restype = C.c_int32
    assert init() == 0
    launch = lib.ds41rt_v41_sparse_attention
    launch.argtypes = [C.c_void_p] * 5 + [C.c_int32, C.c_int32, C.POINTER(View), C.c_void_p]
    launch.restype = C.c_int32
    if a.split_parts is not None:
        original_launch = launch
        split_launch = lib.ds41rt_v41_sparse_attention_split
        split_launch.argtypes = launch.argtypes + [C.c_void_p, C.c_uint64, C.c_int32]
        split_launch.restype = C.c_int32
        scratch_storage = torch.full((16 * a.split_parts * 64 * 514 + 128,), 19.,
                                     dtype=torch.float32, device=f'cuda:{a.device}')
        scratch = scratch_storage[64:-64]
        def launch(*args):
            if 1 <= args[5] <= 16:
                return split_launch(*args, scratch.data_ptr(), scratch.numel() * 4, a.split_parts)
            return original_launch(*args)
    if a.bounded_replay:
        assert not a.large_only, 'bounded replay uses the ordinary reference corpus'
        bounded_launch = lib.ds41rt_v41_sparse_attention_bounded
        bounded_launch.argtypes = [C.c_void_p] * 5 + [C.c_int32, C.c_int32, C.POINTER(View), C.c_void_p,
            C.c_void_p, C.c_void_p, C.c_uint64, C.c_int32]
        bounded_launch.restype = C.c_int32
        window_begins = torch.zeros(4096, dtype=torch.uint64, device=f'cuda:{a.device}')
        def launch(*args):
            split = a.split_parts is not None and 1 <= args[5] <= 16
            return bounded_launch(*args, window_begins.data_ptr(),
                scratch.data_ptr() if split else 0, scratch.numel() * 4 if split else 0,
                a.split_parts if split else 1)
    replay_begin = 0
    stream = torch.cuda.Stream()
    results = []
    with torch.device('cuda'), torch.cuda.stream(stream), tvm_ffi.use_torch_stream(), torch.no_grad():
        # Every case exercises committed and/or private windows. Source history
        # uses deliberately permuted physical pages and strided private rows.
        cases = [(1, 0, 1, False), (3, 0, 2, True), (16, 63, 2, True),
                 (80, 127, 1, True), (16, 255, 2, True), (3, 1023, 1, True),
                 (4096, 0, 1, True)]
        if a.bounded_replay:
            cases += [(1, 128, 1, False), (128, 1023, 2, True), (129, 255, 1, True)]
        if a.large_only:
            assert not a.committed_source, 'committed-source uses the ordinary reference corpus'
            cases = []
            results.extend(large_pool(launch, stream))
        for rows, start, ratio, compressed in cases:
            tokens = rows
            wc = min(4096, rows + 7)
            sc = (rows + 1) // ratio
            committed = (start + tokens) // ratio if a.committed_source else start // ratio
            # Count must include a prior incomplete ratio-two group.
            sc = (start + tokens) // ratio - committed
            pc = min(4096, max(1, 5 + sc * ratio))
            cap = 8192
            values = [torch.randn(n, 512).to(torch.float8_e4m3fn).view(torch.uint8)
                      for n in (128, wc, cap, pc)]
            scales = [torch.randint(124, 128, (n, 16), dtype=torch.uint8)
                      for n in (128, wc, cap, pc)]
            pages = torch.randperm(cap // 256, dtype=torch.int64).to(torch.uint32)
            we = torch.tensor([start], dtype=torch.uint64)
            ce = torch.tensor([committed], dtype=torch.uint64)
            q = torch.randn(rows, 64, 512).bfloat16()
            sink = torch.linspace(-5, 5, 64, dtype=torch.float32)
            rawout = torch.full((rows * 64 * 512 + 32,), 19, dtype=torch.bfloat16)
            out = rawout[16:-16].view_as(q)
            meta = torch.empty(rows, 10, dtype=torch.uint64)
            selected = torch.empty(rows, 512, dtype=torch.int32)
            width = min(128, start + tokens)
            view = View((C.c_void_p * 4)(*[t.data_ptr() for t in values]),
                        (C.c_void_p * 4)(*[t.data_ptr() for t in scales]),
                        we.data_ptr(), pages.data_ptr(), ce.data_ptr(), wc, cap, pc,
                        pages.numel(), int(compressed))
            def metadata(case):
                nonlocal replay_begin
                replay_begin = start if a.bounded_replay and case else 0
                if a.bounded_replay: window_begins.fill_(replay_begin)
                host = []
                ids = []
                for row in range(rows):
                    pos = start + (row if case == 0 else rows - 1 - row)
                    causal = (pos + 1) // ratio
                    host.append([start, 0 if rows == 4096 else 2 + case, tokens, pos, 0, causal, committed, sc, 0 if rows == 4096 else 1 + case, ratio])
                    choices = list(range(max(0, causal - 510), causal))
                    if case: choices.reverse()
                    choices += [-1] * (511 - len(choices)) + [causal + 100]
                    ids.append(choices)
                meta.copy_(torch.tensor(host, dtype=torch.uint64))
                selected.copy_(torch.tensor(ids, dtype=torch.int32))
                return host, ids
            def run():
                assert launch(q.data_ptr(), sink.data_ptr(), meta.data_ptr(),
                              selected.data_ptr() if compressed else 0, out.data_ptr(),
                              rows, width, C.byref(view), stream.cuda_stream) == 0
            def check(host, ids):
                decoded = [(v.view(torch.float8_e4m3fn).float() *
                            torch.exp2(s.float() - 127).repeat_interleave(32, -1)).bfloat16()
                           for v, s in zip(values, scales)]
                concatenated = torch.cat(decoded)
                pagehost = pages.cpu().tolist()
                error = 0.
                # Independent host index construction, then gather the selected
                # BF16 vectors; never use the kernel's physical address resolver.
                for begin in range(0, rows, 8):
                    batch = []
                    masks = []
                    for row in range(begin, min(rows, begin + 8)):
                        m = host[row]
                        refs = []
                        for pos in range(max(replay_begin, m[3] - 127), m[3] + 1):
                            refs.append((0, pos % 128) if pos < start else (1, m[1] + pos - start))
                        refs += [None] * (width - len(refs))
                        if compressed:
                            for index in ids[row]:
                                if index < 0 or index >= m[5]: refs.append(None)
                                elif index < committed:
                                    physical = pagehost[index // 256] * 256 + index % 256
                                    refs.append((2, physical) if physical < cap else None)
                                else: refs.append((3, m[8] + (index - committed) * ratio))
                        idx = torch.tensor([0 if r is None else sum(t.shape[0] for t in decoded[:r[0]]) + r[1]
                                            for r in refs], dtype=torch.int64)
                        mask = torch.tensor([r is not None for r in refs], dtype=torch.bool)
                        # Invalid vectors are zero, including when FP8 backing
                        # memory is poisoned or selected IDs exceed causal bounds.
                        batch.append(concatenated[idx].masked_fill(~mask[:, None], 0))
                        masks.append(mask)
                    gathered, valid = torch.stack(batch), torch.stack(masks)
                    expected = oracle(q[begin:begin + 8], gathered, valid, sink)
                    if begin == 0:
                        # The pinned kernel executes at 16 heads to fit SM120
                        # shared memory; heads are mathematically independent.
                        idxs = torch.arange(gathered.shape[1], dtype=torch.int32).expand(gathered.shape[0], -1).clone()
                        idxs.masked_fill_(~valid, -1)
                        actual = torch.cat([ref.sparse_attn(dense(q[begin:begin + 8, h:h + 16].unsqueeze(1)),
                            gathered, sink[h:h + 16].contiguous(), idxs[:, None].contiguous(), 512 ** -.5).squeeze(1)
                            for h in range(0, 64, 16)], dim=1)
                        torch.testing.assert_close(out[begin:begin + 8], actual, rtol=.008, atol=.002)
                        torch.testing.assert_close(expected, actual, rtol=.008, atol=.002)
                    torch.testing.assert_close(out[begin:begin + 8], expected, rtol=.008, atol=.002)
                    error = max(error, (out[begin:begin + 8].float() - expected.float()).abs().max().item())
                assert (rawout[:16] == 19).all() and (rawout[-16:] == 19).all()
                return error
            host, ids = metadata(0)
            run()
            error = check(host, ids)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream): run()
            q.neg_()
            pages.copy_(pages.flip(0))
            host, ids = metadata(1)
            if a.split_parts is not None:
                scratch.fill_(float('nan'))
            before = torch.cuda.memory_allocated(a.device)
            graph.replay()
            stream.synchronize()
            assert torch.cuda.memory_allocated(a.device) == before
            error = max(error, check(host, ids))
            if a.bounded_replay:
                window_begins.fill_(start + 1)
                graph.replay(); stream.synchronize()
                assert (out == 0).all().item(), 'future replay boundary exposed cache'
                window_begins.fill_(replay_begin)
                graph.replay(); stream.synchronize()
                error = max(error, check(host, ids))
                rawargs = [q.data_ptr(), sink.data_ptr(), meta.data_ptr(),
                    selected.data_ptr() if compressed else 0, out.data_ptr(), rows, width,
                    C.byref(view), stream.cuda_stream]
                for ptr in [0, window_begins.data_ptr() + 1, out.data_ptr()]:
                    assert bounded_launch(*rawargs, ptr, 0, 0, 1) != 0
                if a.split_parts is not None:
                    assert bounded_launch(*rawargs, scratch.data_ptr(), scratch.data_ptr(),
                        scratch.numel() * 4, a.split_parts) != 0
            # Device-side descriptor failures must not expose cache bytes.
            original = meta.clone()
            fields = [(0, 1048577), (1, 2**64 - 1), (2, 0), (3, start + tokens)]
            if compressed: fields += [(4, 1), (6, committed + 1), (7, 2**64 - 1), (8, 2**64 - 1), (9, 0)]
            for field, bad in fields:
                meta.copy_(original); meta[:, field].copy_(torch.tensor([bad] * rows, dtype=torch.uint64))
                graph.replay()
                assert (out == 0).all().item(), (field, bad)
            # Invalid rows must remain zero even if exp(sink) underflows.
            saved_sink = sink.clone(); sink.fill_(-1000)
            graph.replay()
            assert (out == 0).all().item()
            sink.copy_(saved_sink)
            meta.copy_(original)
            # All reads and output retain distinct spans, even during capture.
            args = [q.data_ptr(), sink.data_ptr(), meta.data_ptr(), selected.data_ptr() if compressed else 0,
                    out.data_ptr(), rows, width, C.byref(view), stream.cuda_stream]
            for field in (0, 1, 2, 4):
                bad = args.copy(); bad[field] = 0
                assert launch(*bad) != 0
            bad = args.copy(); bad[4] = q.data_ptr(); assert launch(*bad) != 0
            for field, badvalue in [(5, 0), (5, 4097), (6, -1), (6, 129)]:
                bad = args.copy(); bad[field] = badvalue; assert launch(*bad) != 0
            if a.split_parts is not None and rows <= 16:
                split_args = args + [scratch.data_ptr(), scratch.numel() * 4, a.split_parts]
                required = rows * a.split_parts * 64 * 514 * 4
                for field, value in [(9, 0), (10, required - 1), (11, 0), (11, 11),
                                     (9, q.data_ptr()), (9, out.data_ptr()),
                                     (9, values[0].data_ptr())]:
                    bad = split_args.copy(); bad[field] = value
                    assert split_launch(*bad) != 0
            # Width zero is a supported graph-stable metadata-derived width.
            host, ids = metadata(0)  # Width zero requires ascending positions.
            derived = args.copy(); derived[6] = 0
            assert launch(*derived) == 0
            error = max(error, check(host, ids))
            results.append(dict(rows=rows, start=start, ratio=ratio, compressed=compressed,
                                max_abs=error, bounded_replay=a.bounded_replay, committed_source=a.committed_source, official_reference_rows=min(rows, 8), changed_graph=True, metadata_guards=len(fields), span_guards=True))
            print('PASS', results[-1], flush=True)
        stream.synchronize()
        if a.split_parts is not None:
            assert (scratch_storage[:64] == 19).all() and (scratch_storage[-64:] == 19).all()
    a.output.write_text(json.dumps(dict(device=a.device, split_parts=a.split_parts, reference_lock=lock,
        oracle='Independent blockwise transcription for every row; actual pinned TileLang sparse attention for first min(rows,8) rows and all64 heads in independent16-head groups, before and after replay',
        native_library_sha256=hashlib.sha256(a.native_lib.read_bytes()).hexdigest(), cases=results), indent=2) + '\n')


if __name__ == '__main__':
    main()
