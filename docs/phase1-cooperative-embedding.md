# Cooperative embedding candidate: not enabled

This candidate moved token upload and embedding completion onto the embedding
owner's stream with pinned staging, then copied embeddings into the first block
on its attention stream and waited cooperatively. Query computation retained its
existing synchronous completion. **The candidate is archived, not enabled:** C8
throughput declined consistently across three comparisons.

The [patch](phase1-cooperative-embedding-candidate.patch) applies to `fe0a090` and
preserves the implementation and focused GPU test. Serving source was restored
to that baseline after evaluation. The [evidence record](phase1-cooperative-embedding.json)
contains every initial and reverse-order result, including the unfavorable ones.

## Validation and measurements

Cargo check, test compilation and release build pass. The GPU fixture compares
two concurrent embedding owners against synchronous output at 1, 35, 129 and
4,096 rows, including repeated shapes, distinct tokens and repeated positions.
Residual and pre bytes match exactly. Image replacement, invalid-input rejection
and reuse also pass. Both initial serving arms pass retained-turn/prompt reuse,
needle retrieval, cancellation/survivor recovery and four high-thinking strict
constraint checks. The prefill matrix and full qualification suite were not rerun.

One RTX PRO 6000 Blackwell at **400 W and standard memory speed**, four unchanged
Sparks, five RTX resident expert layers, adaptive independent dSpark, 24 retained
turns, and a **16.681 GB global KV pool for 18,710,016 tokens plus private tails**.
Both versions use identical native libraries and pool settings. No builds overlap
inference. Candidate ran first initially; two follow-up pairs ran control first.

| Workload | Baseline | Candidate |
|---|---:|---:|
| C1 code median, tok/s (three samples) | 129.63 | 129.99 |
| C2 mixed median, tok/s (three batches) | 92.23 | 92.39 |
| C4 mixed median, tok/s (three batches) | 114.78 | 114.96 |
| C8 mixed median, tok/s (three batches) | 156.60 | 153.02 |
| C16 mixed median, tok/s (three batches) | 170.44 | 173.41 |
| Initial startup to API ready, seconds | 5.385 | 5.315 |
| Initial peak observed GPU memory, MiB | 96,864 | 96,876 |

All three C1 outputs match exactly. Mixed batches use ordered first-content
admission and measure earliest first content through last completion, including
admission gaps. Follow-ups retain preceding C2/C4 traffic to preserve nonce order
and warmup, but skip repeated lifecycle and constraint checks.

C8 pairs are **156.60 → 152.79**, **156.96 → 153.21**, and **154.58 → 153.02**
tok/s. The median decline is 2.3%, and reversing run order does not remove it.
C16 pairs are **177.42 → 173.41**, **170.44 → 145.23**, and **148.92 → 178.74**.
That spread does not support a stable C16 speedup or a precise regression estimate.
All C16 batches reach 16 overlapping decodes and use identical request bodies,
but prose outputs change even across repetitions of the same baseline binary.
The record preserves those outputs; no claim is made that their variation has
been causally diagnosed.

## Next implementation

The next design should enqueue a larger unit of GPU work before yielding: token
upload, embedding, handoff and query preparation on a correctly ordered stream,
with ownership retained until completion. Separate waits after each short step
can hand the host to a peer that still executes a long synchronous section. This
is a scheduling hypothesis, not a demonstrated cause of the measured decline.

Keep the direct C1/prefill contracts intact. Use cooperative query completion for
independent lanes and keep cold graph capture outside any suspended capture scope.
Retain cancellation/error draining before source or destination reuse. Reuse the
exact-byte embedding checks in the archived patch when implementing the combined
producer path, then remeasure C8 before accepting it.

Standard serving was restored after both runs. All six stopped experiment
containers were removed after saving their logs. The next release must still
complete this input/query work and the remaining synchronization audit.
