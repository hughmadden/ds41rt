# Initial dual-RTX performance measurements

2026-09-14. Optimized Rust daemon and Release CUDA library; two RTX PRO 6000
Blackwell GPUs at 400 W each and standard 13365 MHz memory clocks. Four Sparks
load decoder expert layers 20–39 only. Prefill batches remain 2048 rows, dSpark
is enabled, and the pool holds 14.96 GB / 16,809,984 source-token positions.

These are initial measurements, not release qualification. The earlier debug
daemon results are excluded. A current controlled single-RTX comparison and
profiling remain outstanding; no 2x decode or 3x prefill claim is supported.

| Concurrency | Counting tok/s | Same-code tok/s | Mixed tok/s |
|---:|---:|---:|---:|
| 1 | 186.59 | 148.46 | 147.04 |
| 2 | 296.98 | 240.33 | 135.20 |
| 4 | 533.97 | 445.28 | 172.38 |
| 8 | 823.88 | 715.40 | 191.27 |
| 16 | 1345.77 | 1167.55 | 286.67 |

Three samples per point. Counting and same-code use exact warm prompts; aggregate
timing includes admission gaps. Same-code uses the existing merge_intervals prompt
and passed static code checks in all 93 measured responses. Mixed traffic uses the
existing code/fable/topic corpus with simultaneous admission and nonce seed 56001.
Different prompt lengths and output lengths make the mixed column incomparable
to the same-code column as an efficiency ratio.

| Cold prefill tokens | Median tok/s |
|---:|---:|
| 1024 | 5002.88 |
| 4096 | 7888.39 |
| 16384 | 8393.49 |
| 32768 | 8322.90 |

Three measured samples after one warmup per cell, zero retained base, exact same
code-context corpus as the published prefill matrix. dSpark does not draft during
encoder prefill. API timing includes terminal setup; this does not explain the
modest long-prompt gain.

The 800 MiB runtime reserve passed warmed C16 counting/code/mixed requests, a
32K needle, exact prompt reuse, retained continuation, cancellation/survivor/recovery
checks, and a chart-description image request. Approximately 181/282 MiB remained
free afterward. Broader peak-memory and full-capacity qualification remain pending.

Raw logs/results are under `~/.cache/ds41rt-experiments/phase2-planner/`; hashes
and exact summaries are in [the JSON record](phase2-first-performance.json).
The measured native library predates the subsequently merged SM-diagnostic PR.
