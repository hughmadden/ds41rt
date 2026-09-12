# Official dSpark expert slice comparison

All three official dSpark stages show a substantial small-row component
improvement over the selected persistent kernel. This qualifies the composed
CUDA-graph candidate for native integration; it is not an API speed measurement
or a production tiling selection.

The fixture loads all 128 experts per stage from the official checkpoint and
GPU-packs them through the selected native packer. Both arms share those resident
weights, BF16 inputs and routing weights. The baseline uses the selected native
capacity-80 coordinator expert kernel. The candidate uses the selected native
BF16-to-FP8 quantizer, GPU route planner, grouped fused slices and ordered FP32
slice reduction. Both include the same native top-3 FP32-to-BF16 route reduction
with a zero shared-expert contribution. Shared-expert GEMM and router computation
are excluded from both arms. Live-row publication for the Python-composed
candidate is prepared before capture; the exportable pipeline may add a small
publication launch.

## Diagnostic timings

Stage 0 median microseconds per invocation, including the operations above:

| Routing | Rows | Persistent | N64 | N128 | N192 |
|---|---:|---:|---:|---:|---:|
| Shared | 5 | 1293.0 | 57.3 | 73.7 | 90.0 |
| Shared | 15 | 2100.5 | 69.6 | 84.7 | 97.1 |
| Shared | 40 | 2134.8 | 118.8 | 92.1 | 106.4 |
| Dispersed | 5 | 1166.6 | 194.3 | 197.4 | 193.3 |
| Dispersed | 15 | 1204.2 | 610.2 | 564.6 | 557.7 |
| Dispersed | 40 | 1662.9 | 1508.8 | 1446.4 | 1430.7 |

Stages 1 and 2 reproduce this pattern. Shared routing uses the same three experts
for every row; dispersed routing uses 15/45/120 distinct experts. These bound
specific reuse patterns, not the model's observed routing distribution. Five
rows correspond to one request's five draft positions within a stage.
N64 leads small shared batches; N192 leads larger dispersed batches. A policy
choice needs live inputs and concurrency validation.

Each case uses six timing rounds, reversing arm order on alternate rounds,
five warmup replays and 30 event-timed replays per arm per round. GPU0 is RTX PRO
6000 Blackwell, UUID `GPU-fe5b6dd0-a77c-c8fb-6360-e1b9d9918ac0`. Captured GPU
states accompany raw samples. These are warm diagnostic measurements with
boundary snapshots, not a continuously monitored release clock qualification.
The resident live APIs were idle; no API comparison ran concurrently.

## Correctness and provenance

All 108 comparisons pass across three stages, two routing patterns, three row
counts, two changed-input replays and three widths. The same graphs consume
changed BF16 input, IDs and routing weights. Replays allocate no Torch storage;
output tails remain untouched. Maximum relative L2 error is approximately
`3.884e-8` for FP32 routes and `1.580e-5` for final BF16 outputs. Some outputs
differ at BF16 rounding boundaries; this is not bit-exact equivalence or broad
model quality qualification.

The native baseline is frozen at
`/tmp/ds41-draft-sampling/selected-native/libds41rt_native.so`, SHA256
`0c4d4dadd683194d4da7f6728f1be1b86e72d5296c1035d9a2b26d23d4a0034f`.
The candidate uses b12x `3b19ffc1886ef0545b0e41fd4c079c5963208b6e`.
[Per-case checks, raw samples, GPU states and source hashes](ds41-draft-expert-slice-comparison.json)
retain the comparison. The earlier preliminary run lacked FP32 route checks;
only the final three runs are used here.

Reproduce with `python/tools/qualify_v41_draft_expert_slices.py --snapshot PATH
--native-lib PATH --stage 0 --output result.json`, repeating stages 1 and 2.
Use the coordinator image with the repository mounted at `/workspace/ds41rt`,
GPU0 selected, the frozen native directory mounted at `/native`, and the model
snapshot accessible read-only. The script verifies the repository's b12x source
lock; `--no-timing` runs only numerical and replay checks.

Next: export the complete coordinator pipeline, verify its native ABI and
scratch lifetimes, then compare captured live routing and full-model output
before selecting widths and measuring C1/C6/C16. The running APIs still use the
previous sampler-era native library.
