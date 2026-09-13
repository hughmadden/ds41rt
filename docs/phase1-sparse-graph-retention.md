# Phase 1: sparse graph retention

Adaptive mode retains up to 48 complete sparse-attention graph fingerprints per
layer with least-recently-used eviction. The fingerprint preserves every external
buffer address/size and launch geometry; live proposal and selection validation
still runs before replay. Dynamic causal metadata is uploaded every time. No
inactive graph launches, and eviction/destruction drains the stream. Default
fixed-policy mode retains its previous one-graph-per-layer behavior.

Three instrumented C1 samples per case, seed 53001, RTX 400 W, standard memory
speed (14001 MHz maximum, 13365 MHz under load), same native library:

| Case | Shared layer retention | + Sparse retention (kept) | + Index retention (rejected) |
|---|---:|---:|---:|
| Code tok/s | 118.43 | 119.15 | 118.47 |
| Fable tok/s | 50.16 | 51.39 | 51.22 |
| Topic tok/s | 64.18 | 66.55 | 65.38 |

All nine outputs in each pilot exactly matched the fixed-five reference. Sparse
captures over the first nine requests fell from 15,120 to 320. Two-row shape-change
verification fell from 32.02 to 30.29 ms; six-row shape-change verification fell
from 44.18 to 42.39 ms. Six-row same-shape verification stayed about 39.5 ms.
Window/compressor and other captures remain; this does not meet A1's complete
steady-state capture goal yet.

Both new pilots passed C1/C4/C16 exact counting. Sparse-only aggregate results
were 143.99/292.55/692.26 tok/s; adding index retention produced
144.26/288.04/676.34 tok/s. These were single runs with different UUID prompts,
so they are correctness smoke checks and exploratory rates, not a controlled
concurrency regression qualification. Sparse-only process memory rose from
65,510 to 65,882 MiB across all requests, including C16; this is not an isolated
graph-allocation measurement or a worst-case retention bound measurement.

The additional index-selection experiment retained complete fingerprints across
batch restart while invalidating published output. It still captured 2,471 index
graphs during the nine requests, did not reduce total observed index time, and
did not improve complete verification or decode. It was reverted. Cancellation
and subsequent exact counting passed on that experimental variant; the kept
sparse-only variant still needs its own cancellation/retained-context gate.

Cargo check and release builds passed for both variants; combined-variant test
compilation passed. Standard serving on port 8000 was restored after each probe.
Prefill, longer context, heterogeneous concurrency, and uninstrumented comparison
remain required before enabling adaptive mode by default. See the
[measurement summary](phase1-sparse-graph-retention.json) for artifact hashes.
