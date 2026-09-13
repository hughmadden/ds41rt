# Cooperative logits downloads

Independent decode now downloads constrained/full diagnostic logits and retained
frontier rows asynchronously through a head-owned stream and reusable pinned
host storage. Consecutive rows are copied together; selected frontier rows are
packed in request order. Head and staging ownership lasts through completion,
including a draining guard on cancellation or failure.

Commit preparation records which compact-output frontiers need full scores.
Independent lanes fetch those rows cooperatively before cache publication; direct
C1 keeps its synchronous transfer path. Constrained batches already have full
scores and retain their rows without a second device transfer. Every downloaded
frontier is checked for finite scores and agreement with the GPU's greedy choice,
then retains all vocabulary scores so a later grammar can choose another token.

Each 48-row head adds 23.67 MiB of pinned host storage, **47.34 MiB total** for the
standard two heads. No GPU workspace or additional KV reduction is introduced.
Normal unconstrained rounds still transfer only the compact GPU selections.

## Validation

The GPU fixture runs two independent download owners concurrently against a
four-row vocabulary-sized buffer. It verifies exact full-batch bytes, reordered
row selection, peer storage isolation, rejected out-of-range selection and reuse
after failure. Five score tests pass, including retained-frontier lifetime, a
changed grammar selecting a different token, non-finite rejection and mismatched
GPU/CPU selection rejection. Test compilation and release build pass.

Both serving arms pass prompt and retained-turn reuse, needle retrieval, eight
cancellations with eight survivors, recovery, and four concurrent high-thinking
strict tool/response-schema checks. All three C1 code outputs match exactly.

## Serving observations

One RTX PRO 6000 Blackwell at **400 W and standard memory speed**, four unchanged
Sparks, five RTX resident expert layers, adaptive independent dSpark, 24 retained
turns, and a **16.681 GB global KV pool for 18,710,016 tokens plus private tails**.
Both arms use the same native library and default pool. Control is `686c2b3`;
candidate ran first, with no builds during inference.

| Workload | Synchronous downloads | Cooperative downloads |
|---|---:|---:|
| C1 code median, tok/s (three samples) | 130.03 | 130.26 |
| C2 mixed, tok/s | 92.68 | 92.79 |
| C4 mixed, tok/s | 114.33 | 115.28 |
| C8 mixed, tok/s | 159.03 | 156.88 |
| C16 mixed, tok/s | 177.06 | 174.57 |
| Startup to API ready, seconds | 5.313 | 5.297 |
| Peak observed GPU memory, MiB | 96,874 | 96,890 |

Mixed batches use ordered first-content admission and include admission gaps.
These single batches remain within 1.5%; they do not demonstrate a throughput
gain. Startup remains approximately 5.3 seconds. Constraint checks establish
correctness, not a measured constraint-serving speedup. The prefill matrix and
full release qualification were not rerun.

The [evidence record](phase1-queued-logits.json) preserves artifact identities,
outputs, timing and the GPU/CPU checks. Standard serving was restored and stopped
experiment containers removed after saving their logs.

Token upload and embedding/query handoff still contain synchronous work, as do
graph capture and some component transitions. Those remain audit targets before
revisiting the earlier performance candidates; this change does not claim an
entirely asynchronous serving loop.
