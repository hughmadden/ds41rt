# Dual-RTX decode investigation

2026-09-14. Two RTX PRO 6000 Blackwell GPUs, 400 W each, standard 13365 MHz
memory clocks. Optimized Rust daemon and Release CUDA library; four Sparks;
2048-row prefill, default adaptive dSpark, 16M-context pool. These are focused
diagnostics, not a replacement release qualification.

## Controlled C1 comparison

The same optimized coordinator and ARM worker binaries generated 205 tokens for
the same merge_intervals prompt. Single RTX loads five encoder expert layers;
dual RTX loads all twenty, TP2. Spark workers load all forty layers for single
RTX and only decoder layers 20–39 for dual RTX. Timing logging was enabled.

Four warm requests after one cold request had median complete-request latency
1.635 s single versus 1.380 s dual: 15.6% less time, or 18.5% higher throughput.
This metric includes request overhead and differs from content-to-finish tok/s.

Mean six-row verification layer timings, summed over each twenty-layer half:

| Stage | Single encoder ms | Dual encoder ms | Single decoder ms | Dual decoder ms |
|---|---:|---:|---:|---:|
| Production and index | 1.008 | 0.822 | 1.205 | 1.024 |
| Attention | 2.657 | 2.654 | 2.650 | 2.627 |
| Experts | 11.171 | 5.835 | 12.633 | 12.685 |
| Finish | 0.142 | 0.233 | 0.140 | 0.232 |

These are elapsed stage times, including scheduling, shared experts and transfers,
not isolated kernel durations. Encoder expert time nearly halves; decoder expert
time and attention remain essentially unchanged. Moving the remaining encoder
experts therefore cannot accelerate the entire decode loop by the bandwidth ratio.

## Routed expert reuse

Optional `DS41RT_TP2_TIMING=1` CUDA events measure each rank's routed kernel sequence
and stable-output copy, after its input dependency. Readout uses the existing
completion point; no extra device wait or cross-lane join is introduced. Enable
`RUST_LOG=info,ds41rt::timing=debug` for timing and route-count logs. Events are
allocated once at startup. Default serving does not create or record these events.

Five C1 requests followed by three same-code batches at each of C1/C2/C4/C8/C16
produced the following aggregates. Grouping is by actual verification row count,
not global concurrency; each lane can advance separately. All 93 measured curve
responses passed the static code contract.

| Verification rows | Mean distinct experts | RTX0 routed µs | RTX1 routed µs | RTX0 effective GB/s | RTX1 effective GB/s |
|---:|---:|---:|---:|---:|---:|
| 6 | 22.66 | 189.01 | 188.26 | 1127 | 1131 |
| 12 | 22.14 | 214.72 | 192.78 | 969 | 1080 |
| 24 | 23.05 | 239.14 | 224.58 | 906 | 965 |
| 48 | 22.72 | 276.98 | 223.22 | 771 | 957 |

Effective bandwidth counts each distinct expert's 9,400,320 resident bytes per
rank once, divided by elapsed GPU event time. It is not measured DRAM traffic:
cache hits, repeated reads, other tensors and concurrent work affect the result.
It cannot establish distance from peak bandwidth. GPU event intervals can also
include enqueue gaps. Identical prompts deliberately maximize sharing; mixed
requests can select more experts.

Expert sharing is present: eight times as many rows still select approximately
the same number of experts. RTX0's higher elapsed time at 48 rows warrants further
contention analysis, but these measurements do not identify a specific cause.

## Rank graph replay experiment

A temporary per-lane, per-rank graph captured routed kernels plus their output
copy. Independent-lane routed and complete-FFN correctness tests passed. Three
performance samples per point showed no consistent gain:

| Concurrency | Baseline code tok/s | Graph code tok/s | Baseline mixed tok/s | Graph mixed tok/s |
|---:|---:|---:|---:|---:|
| 1 | 148.46 | 150.04 | 147.04 | 152.76 |
| 2 | 240.33 | 247.06 | 135.20 | 134.50 |
| 4 | 445.28 | 440.85 | 172.38 | 172.06 |
| 8 | 715.40 | 718.10 | 191.27 | 200.81 |
| 16 | 1167.55 | 1165.71 | 286.67 | 280.16 |

The prototype was removed. Timing instrumentation remains opt-in. A prior Nsight
capture contained CUDA API and memcpy records but no kernel records, so it is not
used as evidence of kernel utilization.

## External Sieve reference

The [published four-RTX recipe](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/deepseek-v4.1-flash.md)
reports 375.58 tok/s for a Sieve script with K7 adaptive verification, four RTX
6000 Pros at 600 W each, stock clocks, temperature 1/top-p 0.95 and reasoning
budget 75. Its five-run median includes reasoning and answer tokens after TTFT.

We ran the same prompt with high reasoning (75), a 2000-token cap and the current
adaptive five-proposal implementation. Native serving requires temperature zero,
so this is not a matched sampling comparison. After one excluded warmup, three
runs measured 120.07, 120.29 and 120.25 tok/s (median 120.25), counting reasoning
and answer from first token to finish. Each emitted 802 tokens and stopped
naturally. Programs were not executed. Timing instrumentation was disabled.

The published result is approximately 3.12 times this local result, but hardware,
sampling, draft policy and generated output differ. It is a useful optimization
reference, not a controlled implementation comparison. Our no-thinking
merge_intervals result should not be substituted for this Sieve measurement.

Next is a fixed K7 feasibility and performance experiment against adaptive drafting.
The [adaptive policy comparison](phase2-adaptive-verification.md) identifies the
current hardware-independent cost fit as another variable to measure.
The current draft generation, terminal outputs and verification limit are built
around five proposals; changing only the CLI range would not implement K7.

[Machine-readable timings and artifact hashes](phase2-decode-analysis.json).
