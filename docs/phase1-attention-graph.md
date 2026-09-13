# Combined attention graph: rejected candidate

One RTX PRO 6000 Blackwell at **400 W, standard memory speed (no overclock)**,
four unchanged resident Sparks, official weights. Baseline: the qualified queued
chain in `1e87c39`. The candidate is preserved as
[an isolated patch](phase1-attention-graph-candidate.patch); **serving source has
been restored to the baseline**.

## Experiment

The candidate captures sparse attention, output projection, attention post-mixing
and FFN preparation into one graph. Query and request metadata uploads stay
outside capture and run before every replay. The existing sparse binding
fingerprint is extended with projection and mHC weight/storage identities. The
sparse graph owner is destroyed before its captured projection/block owners.

Cold setup warms the full chain without publishing FFN input, restores its
logical query state, then captures. Replay establishes queued mHC/FFN state;
only successful stream completion publishes normalized input. The cache remains
bounded by the existing per-layer sparse graph limits. Kernel arithmetic and the
native library are unchanged.

This increases what must be recaptured when a sparse binding changes. That is a
possible cost, not a measured explanation for the observed throughput result.
No capture-cost trace was collected for this candidate.

## Exploratory serving results

Fresh coordinator processes, matching nonces, temperature zero and thinking
disabled. C1 uses three repeats per case. Adaptive C4/C16 each use one mixed batch
per version. No compilation overlapped these runs.

| Metric, tok/s | Queued-chain baseline | Combined graph |
| --- | ---: | ---: |
| Fixed C1 code | 119.95 | 119.86 |
| Target-only C1 code | 43.50 | 43.96 |
| Adaptive C1 code | 122.97 | 123.68 |
| Adaptive C1 fable | 53.45 | 53.32 |
| Adaptive C1 topic | 66.99 | 66.90 |
| Adaptive mixed C4 | 109.92 | 110.96 |
| Adaptive mixed C16 | 190.08 | 177.77 |

All six fixed/target C1 pairs and all nine adaptive C1 pairs are exact. Adaptive
C4 matches 4/4 outputs and C16 matches 6/16; named Python structure checks pass,
while prose quality is not assessed. Both adaptive arms pass the 32K needle,
exact prefix reuse, retained turn, eight cancellations alongside eight survivors,
and recovery checks. Cold needle TTFT is 6.02 → 6.11 s, an HTTP proxy rather than
isolated prefill timing.

The approximately **6.5% C16 decline** fails this exploratory gate, alongside
negligible or small C1 gains. It is not proof that larger graphs inherently hurt
concurrency, nor a causal measurement of recapture overhead. The result does not
justify retaining the additional state/ownership machinery. No balanced rerun,
full prefill matrix, or injected-error qualification was pursued for this
rejected candidate. The qualified queued chain remains in place.

## Evidence

[Machine-readable results](phase1-attention-graph.json) preserve individual C1
samples, paired adaptive observations, limitations and artifact hashes. Apply
the candidate patch to `1e87c39` to reproduce the source. Raw artifacts and the
C1 runner are under `/tmp/ds41-phase1-attention-graph`; the adaptive runner and
results are under `/tmp/ds41-phase1-attention-graph-serving`. Each runner restored
standard serving. Candidate and restored release builds passed.

The next investigation should target a substantial measured device or remote
expert cost. Do not repeat this combined-graph prototype unchanged merely to
reduce launch counts. The Phase 1 90/270 tok/s targets remain unmet.
