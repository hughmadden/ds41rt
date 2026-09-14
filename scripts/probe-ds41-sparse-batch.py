#!/usr/bin/env python3
"""Compare an isolated device-descriptor sparse probe with per-request split launches.

The candidate exports ds41rt_probe_sparse_batch{,_initialize}; this unchecked
experimental ABI is deliberately outside the serving library. Timing excludes
metadata/descriptor upload and graph capture, and is not serving throughput.
"""
import argparse
import ctypes as C
import json
import statistics
from pathlib import Path
import torch

class View(C.Structure):
    _fields_ = [('values', C.c_void_p * 4), ('scales', C.c_void_p * 4),
        ('window_end', C.c_void_p), ('pages', C.c_void_p), ('source_end', C.c_void_p),
        ('window_capacity', C.c_uint64), ('source_capacity', C.c_uint64),
        ('source_proposal_capacity', C.c_uint64), ('page_stride', C.c_uint32),
        ('compressed', C.c_uint32)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--serving-native', action='store_true', help='Exercise public batch validation and launch ABI')
    parser.add_argument('--skip-timing', action='store_true', help='Numerical and replay checks only, for sanitizer runs')
    parser.add_argument('--max-rows', type=int, choices=[48, 64], default=48,
                        help='Include K7 layouts through 64 rows with the serving ABI')
    parser.add_argument('--device', type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    assert not args.output.exists()
    assert C.sizeof(View) == 120
    assert args.max_rows == 48 or args.serving_native, '64 rows requires the serving ABI'
    torch.cuda.set_device(args.device)
    torch.manual_seed(7183)
    baseline, candidate = (C.CDLL(p) for p in (args.baseline, args.candidate))
    assert baseline.ds41rt_v41_sparse_attention_initialize() == 0
    assert (candidate.ds41rt_v41_sparse_attention_initialize() if args.serving_native else candidate.ds41rt_probe_sparse_batch_initialize()) == 0
    old = baseline.ds41rt_v41_sparse_attention_bounded
    old.argtypes = [C.c_void_p]*5 + [C.c_int32, C.c_int32, C.POINTER(View), C.c_void_p,
        C.c_void_p, C.c_void_p, C.c_uint64, C.c_int32]
    new = candidate.ds41rt_v41_sparse_attention_batch if args.serving_native else candidate.ds41rt_probe_sparse_batch
    new.argtypes = [C.c_void_p]*5 + [C.c_int32] + [C.c_void_p]*4 + [C.c_int32]*2
    validator = candidate.ds41rt_v41_sparse_attention_batch_validate if args.serving_native else None
    if validator is not None:
        validator.argtypes = [C.c_void_p]*5 + [C.c_int32, C.POINTER(View)] + [C.c_void_p]*3 + [C.c_uint64, C.c_int32, C.c_int32]
    results, timings, rejections, capacity_checks = [], [], [], []
    for compressed in (0, 1, 2):
        parts = 10 if compressed else 2
        # Distinct allocations for each request, including private compressor rows.
        fixtures = []
        for request in range(8):
            capacities = (128, 8, 768, 3)
            values = [torch.randn(n, 512, device='cuda').to(torch.float8_e4m3fn).view(torch.uint8)
                      for n in capacities]
            scales = [torch.full((n, 16), 125, device='cuda', dtype=torch.uint8) for n in capacities]
            if compressed == 2:
                for slot in (2, 3):
                    values[slot] = torch.randint(0, 256, (capacities[slot], 256), device='cuda', dtype=torch.uint8)
                    scales[slot] = torch.full((capacities[slot], 32), 56, device='cuda', dtype=torch.uint8)
            end = torch.zeros(1, device='cuda', dtype=torch.uint64)
            source_end = torch.tensor([512], device='cuda', dtype=torch.uint64)
            pages = torch.tensor([2, 0], device='cuda', dtype=torch.int32)
            view = View((C.c_void_p*4)(*[v.data_ptr() for v in values]),
                (C.c_void_p*4)(*[s.data_ptr() for s in scales]), end.data_ptr(), pages.data_ptr(),
                source_end.data_ptr(), 8, 768, 3, 2, compressed)
            fixtures.append((view, end, values, scales, source_end, pages))
        layouts = [[1], [6], [1, 2, 3, 4, 5, 6, 1, 2], [6]*8]
        if args.max_rows == 64:
            layouts += [[8], [7]*8, [8, 7, 8, 7, 8, 7, 8, 7], [8]*8]
        for base_layout in layouts:
            layout = list(base_layout)
            rows = sum(layout)
            query = (torch.randn(rows, 64, 512, device='cuda')*.2).bfloat16()
            sink = torch.randn(64, device='cuda')
            metadata = torch.empty((rows, 10), device='cuda', dtype=torch.uint64)
            selected = torch.full((rows, 512), -1, device='cuda', dtype=torch.int32)
            selected[:, :510] = torch.arange(510, device='cuda')
            selected[:, 510:] = torch.tensor([512, 514], device='cuda')
            bounds = torch.empty(rows, device='cuda', dtype=torch.uint64)
            descriptors = torch.empty(rows*C.sizeof(View), device='cuda', dtype=torch.uint8)
            reference, output = torch.empty_like(query), torch.empty_like(query)
            scratch = torch.empty((rows, parts, 64, 514), device='cuda', dtype=torch.float32)
            graph = None
            views = []
            def launch_old():
                stream = torch.cuda.current_stream().cuda_stream
                offset = 0
                for count, view in zip(layout, views):
                    status = old(query[offset:].data_ptr(), sink.data_ptr(), metadata[offset:].data_ptr(),
                        selected[offset:].data_ptr(), reference[offset:].data_ptr(), count, 0, C.byref(view),
                        stream, bounds[offset:].data_ptr(), scratch.data_ptr(), scratch.numel()*4, parts)
                    assert status == 0, status
                    offset += count
            def launch_new():
                stream = torch.cuda.current_stream().cuda_stream
                status = new(query.data_ptr(), sink.data_ptr(), metadata.data_ptr(), selected.data_ptr(),
                    output.data_ptr(), rows, descriptors.data_ptr(), stream, bounds.data_ptr(),
                    scratch.data_ptr(), parts, compressed if args.serving_native else int(compressed == 2))
                assert status == 0, status
            for iteration, start in enumerate((49, 60, 63, 124, 127, 128, 2048, 4096, 32768, 131072, 4096)):
                shift = iteration % len(base_layout)
                layout = base_layout[shift:] + base_layout[:shift]
                views, packed, meta, lower = [], b'', [], []
                for request, count in enumerate(layout):
                    # Rotate external allocation bindings without recapturing candidate.
                    view, end, *_ = fixtures[(request+iteration) % 8]
                    position = start + request
                    end.fill_(position)
                    views.append(view)
                    packed += bytes(view)*count
                    meta.extend([[position, 0, count, position+r, 0, 515, 512, 3, 0, 1] for r in range(count)])
                    # Includes zero bounds, valid truncation and invalid whole-query bounds.
                    lower.extend([0 if iteration%3 == 0 else max(0, position-32) if iteration%3 == 1 else position+1]*count)
                metadata.copy_(torch.tensor(meta, device='cuda', dtype=torch.uint64))
                bounds.copy_(torch.tensor(lower, device='cuda', dtype=torch.uint64))
                descriptors.copy_(torch.tensor(list(packed), device='cuda', dtype=torch.uint8))
                if validator is not None:
                    host_views = (View*rows).from_buffer_copy(packed)
                    validation_args = [query.data_ptr(), sink.data_ptr(), metadata.data_ptr(), selected.data_ptr(),
                        output.data_ptr(), rows, host_views, descriptors.data_ptr(), bounds.data_ptr(),
                        scratch.data_ptr(), scratch.numel()*4, parts, compressed]
                    assert validator(*validation_args) == 0
                    if iteration == 0:
                        for name, index, value in [('descriptor-query-alias', 7, query.data_ptr()),
                            ('descriptor-cache-alias', 7, views[0].values[0]),
                            ('bounds-output-alias', 8, output.data_ptr()),
                            ('scratch-output-alias', 9, output.data_ptr()),
                            ('scratch-undersized', 10, 4), ('wrong-parts', 11, 1),
                            ('too-many-rows', 5, 65), ('wrong-format', 12, 1 if compressed != 1 else 2)]:
                            bad = list(validation_args)
                            bad[index] = value
                            assert validator(*bad) != 0, name
                            rejections.append(dict(compressed=compressed, rows=rows, case=name))
                if graph is None:
                    launch_new()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        launch_new()
                reference.fill_(float('nan'))
                output.fill_(float('nan'))
                launch_old()
                graph.replay()
                torch.cuda.synchronize()
                assert torch.isfinite(output).all(), (compressed, layout, start)
                assert torch.equal(reference, output), (compressed, layout, start,
                    (reference.float()-output.float()).abs().max().item())
                results.append(dict(compressed=compressed, layout=layout, start=start, bit_exact=True))
            if compressed == 2:
                # Reproduce the stale FP8 byte-divisor bug: upper physical pages
                # disappear if capacity is inferred as values.bytes / 512.
                for view in views:
                    view.source_capacity = 384
                launch_old()
                torch.cuda.synchronize()
                assert not torch.equal(reference, output), 'upper-pool fixture must detect halved FP4 capacity'
                for view in views:
                    view.source_capacity = 768
                launch_old()
                torch.cuda.synchronize()
                assert torch.equal(reference, output)
                capacity_checks.append(dict(rows=rows, upper_pool_page=2, detects_halved_capacity=True))
            if args.skip_timing:
                continue
            # Last state has valid bounds; interleave both arms in alternating order.
            launch_old()
            torch.cuda.synchronize()
            old_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(old_graph):
                launch_old()
            arms = [old_graph, graph]
            samples = [[], []]
            for iteration in range(12):
                for arm in ([0, 1] if iteration%2 == 0 else [1, 0]):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(20):
                        arms[arm].replay()
                    end.record()
                    end.synchronize()
                    if iteration >= 2:
                        samples[arm].append(begin.elapsed_time(end)*1000/20)
            timings.append(dict(compressed=compressed, layout=layout, rows=rows,
                scratch_bytes=scratch.numel()*4, baseline_us=statistics.median(samples[0]),
                candidate_us=statistics.median(samples[1]), samples_us=samples))
            print(timings[-1] | {'samples_us': 'omitted'}, flush=True)
    args.output.write_text(json.dumps(dict(scope=__doc__, device=args.device, max_rows=args.max_rows,
        cases=results, timings=timings, validation_rejections=rejections, capacity_checks=capacity_checks), indent=2)+'\n')
    print(f'PASS {len(results)} byte-exact cases with changed descriptor graph replay', flush=True)

if __name__ == '__main__':
    main()
