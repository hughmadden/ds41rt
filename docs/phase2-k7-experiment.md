# Initial fixed K7 experiment

2026-09-14. Two RTX PRO 6000 Blackwell GPUs at 400 W each, standard 13365 MHz
memory clocks, four decoder-only Sparks. Optimized Rust and Release CUDA builds,
2048-row prefill, 16 concurrent requests and 24 retained prompt/turn snapshots.

K7 is opt-in through `--dspark --dspark-draft-limit 7`; add `--dspark-fixed` to
disable adaptive trimming. K5 remains the default. Limits six and seven generate
seven proposals, then cap verification at the requested limit. Limits one through
five retain the existing five-proposal generator.

These are initial measurements, not release qualification. The comparison uses
the earlier adaptive-K5 baseline, not a newly interleaved control. Each throughput
cell has three samples with the same benchmark scripts and prompt seeds.

| Concurrency | Earlier adaptive K5 code tok/s | Fixed K7 code tok/s | Earlier adaptive K5 mixed tok/s | Fixed K7 mixed tok/s |
|---:|---:|---:|---:|---:|
| 1 | 148.46 | 166.97 | 147.04 | 156.31 |
| 2 | 240.33 | 272.45 | 135.20 | 140.17 |
| 4 | 445.28 | 479.80 | 172.38 | 156.16 |
| 8 | 715.40 | 759.79 | 191.27 | 174.75 |
| 16 | 1167.55 | 1142.49 | 286.67 | 261.16 |

Fixed K7 improves low-concurrency code but does not consistently improve high
concurrency or mixed traffic. All 93 same-code responses passed the static code
contract. No counting or full release suite was rerun.

The reasoning-enabled Sieve prompt measured 122.07 tok/s median, versus 121.22
for the earlier fixed-K5 control. K7 emitted 778 tokens versus K5's 802: both
reasoning and answer wording changed, so these are not token-identical workloads.
All six generated Sieve functions passed independent prime-list checks for limits
-1, 0, 1, 2, 3, 10, 997 and 10000. The small throughput difference alone is not
evidence of a meaningful improvement.

## Memory and correctness

K7's lane-local draft workspaces require approximately 19.25 MB on RTX0 and
418.30 MB on RTX1 per lane, versus K5's 14.95/245.10 MB. Larger target-head buffers
also accommodate eight verification rows per request. With the existing 800 MiB
reserve, the automatic pool becomes **14.43 GB / 16,212,480 raw source-token
positions**, below the 16×1,048,576-context target. K5's default pool remains
14.96 GB / 16,809,984 positions. K7 is not being promoted to the default.

The 32K needle, exact prompt reuse, retained continuation, C16 cancellation,
survivor and recovery checks passed. Core tests cover seven-token correction at
every position, bonus/EOS/length stopping, nonoverlapping RNG reservations, bounded
route history and packed-counter overflow limits.

Both K5 and K7 distributed terminals matched the full vocabulary head exactly
for tokens, corrected logits and confidence across request counts 1/3/8/16 and
mixed temperatures, including independent graph replay, cancellation and reuse.
The initial standalone K7 test exposed an 80-row vocabulary limit at 16 local
requests; projection/graph limits were expanded to 128 and both tests passed.
Serving has at most eight requests per lane and the throughput runs above used
at most 64 target rows or 56 draft rows, before that standalone-limit extension.

## Next comparison

The adaptive cost formula has not yet been recalibrated. Per the requested
experiment order, measure current hardware costs **before** evaluating adaptive
K7. Include a fresh K5 control with matching binaries, then compare code and mixed
traffic. Preserve independent lane decisions and assess the KV-capacity tradeoff
before selecting a default. Single-RTX performance validation remains pending.

[Machine-readable results and raw artifact hashes](phase2-k7-experiment.json).
