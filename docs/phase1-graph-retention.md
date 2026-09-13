# Phase 1: bounded graph retention pilot

Experimental adaptive mode now retains row shapes 1–48 for the five shared
layer-graph owners (query, attention output, router, shared expert, index query).
Each layer retains at most 48 small graphs plus one large prefill graph. Captures
remain tied to exact immutable weight owners and lane storage; owner replacement
invalidates old shapes. Fixed-policy mode keeps its existing single-shape behavior.

Three instrumented C1 runs per case, seed 53001, RTX 400 W and standard memory
speed (14001 MHz maximum, 13365 MHz under load), same native library:

| Case | Previous adaptive tok/s | With graph retention tok/s |
|---|---:|---:|
| Code | 115.28 | 118.43 |
| Fable | 44.06 | 50.16 |
| Topic | 56.17 | 64.18 |

All nine outputs exactly matched the fixed-five reference. These are exploratory
medians, not release numbers or a statistical guarantee. Shape-change verification
medians fell from 39.18 to 32.02 ms at two rows and from 51.80 to 44.18 ms at six
rows. Same-shape six-row verification was roughly unchanged (39.94 to 39.58 ms).
Sparse attention still captured 15,120 graphs in each trace; other graph owners
remain to be addressed. The old trace did not enable the new shared graph capture
counter, so its zero counter is unavailable evidence, not zero actual captures.

The process grew from 65,510 to 65,732 MiB during the pilot; this includes other
runtime allocations. Worst-case C16 graph memory and prefill preservation are
still unqualified. Standard port-8000 serving was restored after the probe.

Validation: cargo check, release build, test compilation, and two actual CUDA
graph ownership tests passed. The new CUDA test exercises all 48 small shapes,
large-shape eviction, changed input data, stable replay handles, exact owner
invalidation, and the 49-entry bound. See [raw summary](phase1-graph-retention.json).
