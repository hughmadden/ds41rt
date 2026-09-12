#!/usr/bin/env python3
"""Exact native sampler comparison, including Philox offsets, ties and graph reuse."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import statistics
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--skip-timing', action='store_true')
    args = parser.parse_args()
    libraries = [C.CDLL(str(path.resolve())) for path in (args.baseline, args.candidate)]
    functions = []
    for library in libraries:
        fn = library.ds41rt_v41_draft_step_rng
        fn.argtypes = [C.c_void_p]*6 + [C.c_int32, C.c_int32, C.c_void_p]
        fn.restype = C.c_int32
        functions.append(fn)
    stream = torch.cuda.Stream()
    torch.manual_seed(4137)
    vocab = 129280
    cases, timings = [], []

    def call(fn, shared, bias, rng, temperatures, adjusted, tokens, position):
        status = fn(shared.data_ptr(), bias.data_ptr(), rng.data_ptr(), temperatures.data_ptr(),
                    adjusted.data_ptr(), tokens.data_ptr(), shared.shape[0], position, stream.cuda_stream)
        assert status == 0, status

    with torch.cuda.stream(stream):
        for rows in ([1, 16] if args.quick else [1, 2, 3, 8, 16]):
            shared = torch.empty((rows, vocab), device='cuda')
            bias = torch.empty_like(shared)
            rng = torch.tensor([[0x123456789abcdef + r, (1 << 40) + 5123*r] for r in range(rows)],
                               dtype=torch.int64, device='cuda')
            temperatures = torch.zeros(rows, device='cuda')
            storage = [torch.full((rows*vocab+64,), 12345.125, device='cuda') for _ in range(2)]
            token_storage = [torch.full((rows+64,), -777, dtype=torch.int32, device='cuda') for _ in range(2)]
            adjusted = [x[32:-32].view(rows, vocab) for x in storage]
            tokens = [x[32:-32] for x in token_storage]
            for kind in (['random', 'ties'] if args.quick else ['random', 'ties', 'large', 'masked']):
                shared.normal_(); bias.normal_()
                if kind == 'ties':
                    shared.zero_(); bias.zero_()
                    # Equal maxima cross tile boundaries and include the reserved prefix.
                    shared[:, [7, 511, 512, 8191, vocab-1]] = 4
                elif kind == 'large':
                    shared.mul_(1e20); bias.mul_(1e20)
                elif kind == 'masked':
                    shared.fill_(-float('inf')); bias.zero_()
                    shared[:, [17, 506, 1024, vocab-1]] = 0
                for position in ([4] if args.quick else range(5)):
                    for temperature in (0., .7):
                        temperatures.fill_(temperature)
                        if temperature and rows > 1:
                            temperatures[::3] = 0
                            temperatures[1::3] = 1e-7
                        for i, fn in enumerate(functions):
                            call(fn, shared, bias, rng, temperatures, adjusted[i], tokens[i], position)
                        stream.synchronize()
                        assert torch.equal(adjusted[0].view(torch.int32), adjusted[1].view(torch.int32)), (rows, kind, position, 'raw logits')
                        assert torch.equal(tokens[0], tokens[1]), (rows, kind, position, temperature, tokens)
                        for x in storage:
                            assert (x[:32] == 12345.125).all() and (x[-32:] == 12345.125).all()
                        for x in token_storage:
                            assert (x[:32] == -777).all() and (x[-32:] == -777).all()
                        cases.append(dict(rows=rows, kind=kind, position=position, temperature=temperature))
            # Capture once, then change all input values and RNG metadata in-place.
            graphs = []
            for i, fn in enumerate(functions):
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    call(fn, shared, bias, rng, temperatures, adjusted[i], tokens[i], 3)
                graphs.append(graph)
            for replay in range(3):
                shared.normal_(); bias.normal_(); rng.add_(4096)
                temperatures.fill_([0., .7, 1.][replay])
                for graph in graphs:
                    graph.replay()
                stream.synchronize()
                assert torch.equal(adjusted[0].view(torch.int32), adjusted[1].view(torch.int32))
                assert torch.equal(tokens[0], tokens[1])
            if not args.skip_timing:
                for temperature in [0., .7]:
                    temperatures.fill_(temperature); stream.synchronize()
                    samples = [[], []]
                    for repeat in range(11):
                        for i in [repeat % 2, 1-repeat % 2]:
                            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                            start.record(stream)
                            for _ in range(20):
                                graphs[i].replay()
                            end.record(stream); end.synchronize()
                            samples[i].append(start.elapsed_time(end)*1000/20)
                    timings.append(dict(rows=rows, temperature=temperature,
                                        baseline_us=statistics.median(samples[0]),
                                        candidate_us=statistics.median(samples[1]), samples_us=samples))
            print('rows', rows, 'exact checks passed', flush=True)
    report = dict(baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                  candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                  exact_cases=cases, graph_replays_per_row_count=3, timings=timings)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    for timing in timings:
        print('rows', timing['rows'], 'temperature', timing['temperature'],
              'baseline/candidate us', round(timing['baseline_us'], 2),
              round(timing['candidate_us'], 2), flush=True)


if __name__ == '__main__':
    main()
