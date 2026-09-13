# Shorter RoCE polling quantum

One RTX PRO 6000 Blackwell at **400 W with standard memory speed** (configured
maximum 14,001 MHz; no memory overclock), plus four unchanged resident Sparks.
The experiment changes only subsequent unsuccessful-poll waits: the receiver
still yields on its first wait, and complete responses never yield.

## Matched-context experiment

An isolated binary extends the private-verifier probe to alternate 250 µs,
50 µs, and zero (yield on every unsuccessful poll). Five joint prefix shapes
run at six nearby contexts, with one code/fable request per lane. All arms
include identical local poll counters and route capture. After one warmup sweep,
three measured sweeps rotate/reverse interval/shape order. Full-prefix sentinels
bracket each sweep; cache hashes and greedy selections are checked throughout.
The interval resets to 250 µs when the probe scope exits, including on error.

| Draft lengths per lane | 250 µs interval: verify ms | 50 µs interval: verify ms | Yield every failed poll: verify ms |
| --- | ---: | ---: | ---: |
| 1 / 1 | 47.35 | 40.36 | 40.21 |
| 3 / 3 | 63.15 | 41.34 | 41.97 |
| 5 / 5 | 61.67 | 48.14 | 48.41 |
| 1 / 5 | 59.17 | 41.99 | 43.79 |
| 5 / 1 | 59.11 | 43.19 | 43.38 |

Each table entry pools 18 measured trials. Within each of the 30 context/shape
pairs, the median change from 250 to 50 µs ranges from −34.8% to −11.3%, with
median **−25.6%**. Yielding on every failed poll has median −24.2%. These are
complete target-verification/logit-collection times, excluding draft generation,
hashing, discard cleanup and publication. Probe HTTP throughput is not reported.

For the 3/3 shape, median poll/yield counts per complete two-lane verification
are 42,502/155 at 250 µs, 9,074/159 at 50 µs, and 1,121/1,041 at zero. Thus the
subsequent spin interval is exercised, and 50 µs lets the peer lane progress
with much less polling while avoiding the zero-interval rescheduling volume.
Per-context data retain all timing and poll-count samples.

All 408 executions completed, including 270 measured trials. Cache hashes and
full-prefix sentinels stayed unchanged; all **2,160 common greedy selections**
agreed. Final streamed outputs exactly matched the prior control, and the 32K
needle/reuse/retained-turn/cancellation/recovery checks passed. These checks
cover committed window and paged KV/index storage, not an independent hash of
hidden compressor carry or equality of every logit byte.

The reproduction uses `docs/phase1-poll-probe.rs`, the existing scheduler probe
patch, and `docs/phase1-poll-probe-transport.patch` on source commit `6e0df55`.
Follow the setup in [the matched-prefix experiment](phase1-matched-prefix-cost.md),
substituting the polling module and applying the transport patch. Use
`scripts/summarize-ds41-poll-probe.py` to summarize the snapshots. Frozen artifacts,
raw snapshots and the runner are in `/tmp/ds41-phase1-poll-probe`.

## Serving change and focused validation

Production selects 50 µs once per request wave only when every row is Decode or
MtpVerify. Prefill, Benchmark, empty, and mixed waves retain 250 µs. The first
unsuccessful poll still yields immediately; successful completion still returns
without yielding. There are no diagnostic counters or interval overrides in the
serving implementation. Adaptive confidence and cost coefficients are unchanged.

Fresh coordinator processes used C16 admission, 24 retained snapshots, a 2,048
prefill batch and the same GPU, native library and resident workers. Decode cases
use three repeats, temperature zero, thinking disabled and matching nonces.
Mixed C4/C16 runs include admission gaps in aggregate streaming throughput.
These sequential serving arms are exploratory, not a balanced release comparison.

| Metric (tok/s) | Before: 250 µs | After: decode-only 50 µs |
| --- | ---: | ---: |
| Fixed C1 code | 120.65 | 121.61 |
| Fixed C1 fable | 46.13 | 46.38 |
| Fixed C1 topic | 62.55 | 63.63 |
| Fixed mixed C4 | 101.51 | 113.35 |
| Fixed mixed C16 | 167.54 | 168.25 |
| Adaptive C1 code | 121.91 | 122.32 |
| Adaptive C1 fable | 52.85 | 52.99 |
| Adaptive C1 topic | 66.15 | 66.13 |
| Adaptive mixed C4 | 102.95 | 108.43 |
| Adaptive mixed C16 | 177.95 | 184.80 |
| Target-only C1 code | 42.74 | 42.76 |

All C1 responses match exactly (9/9 each speculative policy and 3/3 target-only).
C4 matches 4/4 for each policy. C16 exact matches are 14/16 fixed and 5/16 adaptive;
scheduling-dependent outputs differ. All assessed Python structure checks pass;
this does not establish equivalent prose quality. Each speculative arm passes the
32K needle, exact prefix reuse, retained turn, eight cancellations alongside eight
survivors, and recovery checks.

An initial global 50 µs candidate showed slower cold needle TTFT, so it was
superseded by the decode-only implementation. Cold needle TTFT remained variable:
fixed 5.71 → 6.12 s, adaptive 5.85 → 5.88 s. These measurements are retained in the
JSON and are not evidence of unchanged cold-start performance.

A separate A–B–B–A check then measured one exact 32,768-token cold-cache prefill
cell, with a fresh coordinator for each arm, one warmup and three measured
requests per restart. All corresponding prompt hashes match across arms. All 16
requests have zero cache hits and return `7` (checked from saved responses; the
benchmark's `passed` field alone validates the cache-miss count).

| Arm | Median effective prefill tok/s |
| --- | ---: |
| A1: original 250 µs | 7,792 |
| B1: decode-only 50 µs | 7,696 |
| B2: decode-only 50 µs | 7,735 |
| A2: original 250 µs | 7,845 |
| Pooled A (six measured requests) | 7,825 |
| Pooled B (six measured requests) | 7,721 |

The pooled difference is −1.3%, with overlapping sample ranges (A: 7,674–7,925;
B: 7,612–7,910). This is HTTP time-to-first-content throughput, including the
first output token, not isolated GPU prefill time. Prefill retains its original
polling interval; this small sample cannot prove zero regression or replace a
full prefill matrix. No full qualification suite was rerun.

The release build passed. Transport tests passed 11 tests; two tests requiring
idle live workers were ignored, with live serving exercised separately above.
The added test checks Decode/MtpVerify and preservation of 250 µs for prefill,
benchmark, mixed and empty waves. Standard service was restored after every
runner. Raw serving artifacts are under `/tmp/ds41-phase1-poll-scoped`, controls
under `/tmp/ds41-phase1-poll-serving`, and prefill data under
`/tmp/ds41-phase1-poll-prefill`.

[Machine-readable evidence](phase1-poll-quantum.json) preserves matched-context
samples, serving comparisons, prefill samples and artifact hashes. The result
improves concurrent progress, but C1 remains about 122 tok/s speculative and
42.8 tok/s target-only. The Phase 1 targets of 270/90 tok/s remain unmet.
