# Batched sparse attention in serving

Measurements use one RTX PRO 6000 Blackwell at **400 W with standard memory
speed** (configured maximum 14,001 MHz; no memory overclock), plus four Sparks.

The candidate runs small multi-request split attention as two kernels per layer,
using device-resident row descriptors to change cache bindings without graph
recapture. Both fixed and adaptive policies use this path. Single-request
attention and large-prefill dispatch retain the previous path; adaptive policy
remains opt-in.

## Ownership, validation and memory

The native ABI separates host validation from launch. Before **every** replay,
Rust checks the current request/source/selection bindings, row counts, buffer
sizes and devices; native validation checks alignment and disjoint spans for all
current descriptors, inputs, output and scratch. Host validation launches no GPU
work. The wave uploads descriptors, per-row metadata and replay bounds on its
stream before the captured attention and merge. Zero bounds are explicitly
uploaded as well. All borrowed proposal and selection owners remain live until
stream completion, including the error path; graph destruction drains the stream.

The descriptor path requires multiple requests, at most 48 total rows, and at
most 16 rows per request so each request retains the original split arithmetic.
Source format and partition count are homogeneous per layer (FP8 SWA only with
two partitions, or FP4 global source with ten). Captures are keyed by total rows,
layer, sink and the stable lane-owned selection pointer. Query, output, scratch,
metadata, bounds and descriptors are wave-owned stable allocations. Request
boundaries and external cache pointers are read from current device descriptors.
At most 48 sparse graph variants are retained per layer. Existing native libraries
without the additive batch symbols retain per-request dispatch.

Scratch grows from 20.08 to 60.23 MiB per wave at capacities of at least 48 rows;
5,760 bytes of device descriptors are also allocated. `device_bytes` accounts for
both before allocation. Two lanes add about 80.32 MiB of attention workspace.
This does not change the configured KV token pool.

While integrating, the sparse binding was found to calculate physical FP4 source
capacity as `values.bytes / 512`, using the old FP8 row size. It now uses
`V41Kv::COMPRESSED_VALUE_BYTES` (256). Previously, valid source IDs mapped into
the upper half of the physical pool could be masked. The fixture deliberately
selects page 2 of a three-page pool and verifies that the old half-capacity value
changes the output; the full capacity restores byte equality. This fix applies
to both single-request and batched paths.

## Native checks

The public serving ABI passed 132 byte-exact comparisons against the previous
native library, including changed allocation bindings and request row counts
without candidate graph recapture, private and paged KV, bounds, FP4/FP8 source
formats, and positions through 131K. It rejected 96 malformed alias/shape cases.
Four upper-pool cases detect the stale FP8 capacity divisor. CUDA memcheck reports
zero errors. All five native CTests pass, and the Rust daemon release build and
check pass. The first CTest attempt omitted the source fixture mount and failed
the tokenizer-dependent XGrammar test; rerunning with its fixture mount passed.

The extended probe is reproducible with `scripts/probe-ds41-sparse-batch.py
--serving-native`; see [the isolated probe](phase1-sparse-batch-probe.md) for the
container setup. `--skip-timing` runs numerical, validation and replay checks
without a timing sweep.

## Serving measurements

**One RTX PRO 6000 Blackwell at 400 W, standard memory speed** (configured maximum
14,001 MHz; no memory overclock), plus the four unchanged resident Sparks.
Port 8000, C16 admission limit, 24 prompt snapshots, 24 retained turns, draft
limit five, and `RUST_LOG=warn`. These performance prompts use thinking disabled.
Each C1 entry is the median of three identical-input runs. Mixed C4/C16 and the
32K lifecycle probe run once per arm with deterministic token-zero prefixes.
Arms are sequential and exploratory, without a balanced clock/throttling gate.

| Workload | Previous fixed | Batched fixed | Batched adaptive |
| --- | ---: | ---: | ---: |
| C1 code tok/s | 119.11 | 120.82 | 122.06 |
| C1 fable tok/s | 46.00 | 46.31 | 52.61 |
| C1 topic tok/s | 63.00 | 63.59 | 65.83 |
| Mixed C4 aggregate tok/s | 101.63 | 109.21 | 102.19 |
| Mixed C16 aggregate tok/s | 164.47 | 167.73 | 182.80 |
| 32,815-token cold TTFT seconds | 5.893 | 5.900 | 5.887 |

The earlier adaptive warmup milestone measured C16 **151.37 tok/s**, versus
**182.80 tok/s** here (+20.8%). Its C4 was 101.90 and C1 code/fable/topic were
122.22/53.00/66.41; those C1 changes are within 1%. This historical adaptive
comparison is separate from the fresh fixed-before/fixed-after comparison.
On the same new binary, adaptive C16 exceeds fixed by 9.0%, while adaptive C4
still trails fixed by 6.4%. The policy needs further calibration; it stays opt-in.

Every arm passed the 32K needle, full-prompt reuse (32,815 cached tokens), retained
turn continuation, eight cancellations with eight surviving counting requests,
and the recovery request. Cold TTFT is an HTTP prefill proxy including overhead,
not isolated prefill throughput. It shows no material change in this one sample;
it does not close the broader prefill gate.

C1 text is exact 9/9 for all comparisons. C4 is exact 4/4. C16 is exact 15/16
for old-fixed versus batched-fixed (the earlier repeated fixed baseline also
matched 15/16), and 6/16 for batched fixed versus adaptive. Old-adaptive versus
new-adaptive also matches 6/16. Named structural checks pass; nonempty prose is
not a semantic-quality qualification. Native byte equality does not establish
whole-model trajectory equality under changing scheduling and row shapes.

A separate instrumented C4/C16 run on the same artifacts compared with the
previous adaptive warmup trace:

| C16 round metric | Before | After |
| --- | ---: | ---: |
| Median sparse attention ms | 27.62 | 8.32 |
| Median complete verification ms | 160.73 | 151.39 |
| Median complete scheduler round ms | 176.69 | 167.34 |
| Rounds with sparse captures / total | 36 / 42 | 18 / 42 |
| Median proposed / emitted tokens | 43 / 50 | 42 / 50.5 |

New row shapes still capture. Across both lanes, every batched layer/row shape
was captured at most twice in the trace. The trace omits lane identity, so it
cannot independently prove one capture per lane; the new graph key removes
request-layout and external-cache-pointer recapture by construction. Captures
remain in four of the final ten C16 rounds as shapes change. Whole verification
improves, but this is not a claim that all serving capture work is eliminated.
Routing, row counts and generated lengths differ between runs. Instrumented
aggregate C16 was 189.68 tok/s and is kept separate from the uninstrumented table.

Initial measured device memory rose 65,510 → 65,590 MiB, matching the workspace
budget. After the lifecycle probes it was 65,770 MiB before, 65,918 MiB batched
fixed, and 67,442 MiB batched adaptive; adaptive also retains the previously
introduced graph families. No KV pool size was reduced.

[The measured summary](phase1-sparse-batch-serving.json) includes hashes, paired
output checks, lifecycle accounting, memory and round timing. The original
artifacts and runner are in `/tmp/ds41-phase1-batch-serving`; the focused trace
runner is in `/tmp/ds41-phase1-batch-trace`. Use
`scripts/compare-ds41-adaptive-runs.py` for paired outputs and
`scripts/summarize-ds41-sparse-trace.py` for trace comparisons. Standard serving
was restored after both probes; no release tag, image or `main` was published.
