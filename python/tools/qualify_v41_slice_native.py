#!/usr/bin/env python3
"""Check exported slice execution through the production expert C ABI."""

import argparse
import json
import torch
import _pinned_sparkinfer  # noqa: F401
from _v41_expert_native import Native, library
from tests.moe.test_v41_grouped_slices import _check_grouped_slices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-lib", required=True)
    parser.add_argument("--width", type=int, choices=(64, 128, 192), required=True)
    options = parser.parse_args()
    lib = library(options.native_lib)
    owners = {}

    def check(**case):
        rows = case["rows"]
        weights, wire, ids_cpu, routing = case["native_inputs"]
        capacity = 1 if rows == 1 else 16 if rows <= 16 else 80
        if capacity not in owners:
            ids = torch.empty(capacity, 6, dtype=torch.int32, device="cuda")
            rw = torch.empty(capacity, 6, device="cuda")
            native = Native(lib, capacity, weights, wire, ids, rw)
            ids.fill_(-1)
            rw.zero_()
            native.run(1)
            graph = torch.cuda.CUDAGraph()
            # Rows stay at capacity; invalid routes mask inactive input rows.
            with torch.cuda.graph(graph):
                native.run(capacity)
            owners[capacity] = native, ids, rw, graph
        native, ids, rw, graph = owners[capacity]
        ids.fill_(-1)
        ids[:rows].copy_(ids_cpu)
        rw.zero_()
        rw[:rows].copy_(routing)
        native.output.fill_(float("nan"))
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        assert torch.cuda.memory_allocated() == allocated
        reference = torch.zeros_like(native.output)
        pairs = case["pair_gpu"]
        # Same-width independently launched grouped compute with CPU metadata.
        partial = case["out"][:, : rows * 6]
        summed = torch.zeros_like(partial[0])
        for plane in partial:
            summed += plane
        reference[pairs[:, 0] * 6 + pairs[:, 1]] = summed
        assert torch.equal(native.output, reference), case["case"]
        print(
            "NATIVE "
            + json.dumps(
                dict(
                    case=case["case"],
                    rows=rows,
                    width=options.width,
                    exact=True,
                    scratch_bytes=native.storage.numel(),
                )
            ),
            flush=True,
        )

    _check_grouped_slices(options.width, after_case=check)
    for _, _, _, graph in owners.values():
        graph.reset()


if __name__ == "__main__":
    main()
