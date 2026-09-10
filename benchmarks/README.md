# DS41RT Pro v1 benchmark record

The public benchmark summary is intentionally compact: it records the model,
hardware, runtime policy, request shapes, samples, and correctness gates needed
to interpret the README claims without importing the internal optimization
notebook or generated run database.

## Fixed contract

| Property | Value |
| --- | --- |
| Model | `wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1` |
| Model revision | `7a63f24905223aff19212d65226be708950823ac` |
| Coordinator | 1 × RTX PRO 6000 Blackwell 96 GB (`sm_120`) |
| Experts | 4 × DGX Spark GB10 (`sm_121`), routed-expert TP4 |
| Fabric | dedicated 400-Gb/s coordinator path, native verbs |
| Profile | `balanced`, FP8 target KV |
| Speculation | adaptive dSpark, proposal widths 1–5 |
| Sampling | temperature 0, thinking disabled |
| Runtime gates | full attention, numeric progression, zero timed captures |

## Headline gates

The largest headline rates are deliberately low-entropy stress cases, not
semantic-workload claims. Repeat decode asks for `orchid` exactly 100 times;
all five timed responses did so exactly. Retained prefill extends a
repeated-`alpha` prefix by 32K tokens after 32K cached tokens, and excludes the
time used to construct that prefix.

The semantic blend ran seven fixed cases five times (35 generations). It
accepted 3,568 of 5,577 drafted tokens and exercised every target verification
width. The fresh prefill row used cache-busted source text with one warmup and
two timed samples per size.

| Measurement | Samples/result |
| --- | --- |
| Artificial repeated-word decode | 66.81–68.09 tok/s; 67.56 aggregate |
| Artificial retained 32K + 32K prefill | 2,039.0 tok/s median |
| Semantic blend decode | 37.44–38.68 tok/s; 38.08 aggregate |
| Code slice decode | 52.50 tok/s pooled over five generations |
| Semantic-blend dSpark acceptance | 63.98% |
| Fresh 8K prefill | 1,458.9 / 1,443.4 tok/s; median 1,451.1 |
| Fresh 32K prefill | 1,661.9 / 1,665.2 tok/s; median 1,663.5 |
| Tool Eval Bench 2.3.2 | 86/100; 118/138 raw; 54 pass / 10 partial / 5 fail |

All measured samples are retained. The repeat-decode and prefill runs passed
full-attention, numeric-progression, and zero-request-time-capture gates.
The post-attention-fix Tool Eval run completed all 69 scenarios sequentially
without a backend failure. Its safety gate did not pass: TC-34, TC-42, TC-58,
and TC-60 produced warnings.

## Retained-context grid

The fresh row uses cache-busted HTTP prompts drawn from the checked-in Rust
source and constructed to exact token counts. Retained rows extend one
artificial repeated-`alpha` sequence inside the startup qualification path so
the physical compressor/KV state is actually reused. Each grid cell is the
median of two requests and reports thousands of newly computed tokens per
second. Prefix construction is excluded from retained-row throughput.

This is not an HTTP radix-cache claim: Pro's C4/C128 compressor state is
lane-local, and the external prompt cache does not restore that state.

## Decode concurrency grid

The August 21 matrix uses llama-benchy 0.4.0, a 2K prompt, exactly 128 output
tokens, three timed runs per cell, and no prefix caching after 0, 4K, or 8K
tokens of context. C2 and C4 are pooled server throughput, not the median
per-request rate. All 63 requests across the nine cells completed without an
HTTP/backend failure.

## Startup

Cold startup begins without resident expert slabs. The accepted coalesced-read
run reached API readiness in 64.75 seconds while each Spark read its 101.85-GB
source quarter at about 5.095 GB/s and exchanged the compact TP4 views. Warm
restarts that reused expert residency and capture metadata reached readiness in
60.89–61.31 seconds.

The charts in this directory are direct renderings of the arrays in
[`pro-v1.json`](pro-v1.json).
