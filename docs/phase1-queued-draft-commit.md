# Queued dSpark commit

Accepted dSpark KV is prepared and written on the owning decode lane's stream.
The scheduler polls completion and yields to its peer. The large existing main
workspace remains available for prefill and lane zero; lane one adds an 80-row
main workspace. Immutable projection weights and persistent request caches are
shared. Single-active-request execution retains its existing direct commit path.

The queued path copies completed target taps directly into the main producer,
then queues positions, projection/norm/RoPE, and three cache scatter kernels.
It avoids copying each produced KV array through the shared window source buffer.
Each producer owns its pinned metadata and device descriptors until completion.

Write tickets reserve only participating cache slots. Reads, writes, retention,
and recycling of those slots fail until publication; disjoint slots remain usable.
Committed frontiers change only after GPU completion and validation of all three
owners. Partial enqueue errors drain before revoking the participating requests.
Saved request identities permit cleanup even if target publication has already
cancelled its batch. The target pass and tap storage remain owned by the lane
through completion. Producer destruction drains before releasing reservations.

Two CPU tests cover reader lifetime, exclusive writers, disjoint access, and
all-or-none reservation failure. Two CUDA tests cover disjoint reader/write
progress, separate-stream writes, independent publication, rejection of premature
retention/recycling, and revocation after a simulated later transaction failure.
These GPU tests use the serving native library on GPU zero. They do not inject a
CUDA hardware fault or prove complete scheduler cancellation behavior.

This removes the dSpark commit host wait, not all commit waits. Target-model SWA
and compressed-cache publication remain synchronous, as do retirement snapshots.
Those are subsequent work in the [v2 sequence](phase1-optimization-plan.md).

## Focused serving comparison

One RTX PRO 6000 Blackwell at **400 W and standard memory speed**, four unchanged
Sparks, five RTX resident expert layers, 18 × 1,048,576-token KV pool, and 24
retained snapshots. Both versions use adaptive independent lanes and the same
native library. No builds overlapped benchmarks; the prefill matrix was not rerun.

| Workload | Synchronous dSpark commit | Queued dSpark commit |
|---|---:|---:|
| C1 code median, tok/s (three samples) | 129.90 | 130.15 |
| C2 mixed, tok/s | 91.71 | 88.59 |
| C4 mixed, tok/s | 121.66 | 122.48 |
| C8 mixed, initial pair, tok/s | 165.78 | 155.08 |
| C16 mixed, tok/s | 205.21 | 201.67 |
| C8 focused repeat 1, tok/s | 148.86 | 150.93 |
| C8 focused repeat 2, tok/s | 151.41 | 163.58 |
| Initial paired startup to API ready, seconds | 5.324 | 5.303 |
| Initial paired peak observed GPU memory, MiB | 97,014 | 97,006 |

The initial pair ran candidate first. Mixed rates cover one cold batch per
concurrency, including admission gaps. The initial C8 decline motivated two
additional C8-only comparisons with new matched nonces, control first. These did
not reproduce the decline; the measurements do not establish a consistent
throughput gain. Memory peaks include CUDA/runtime and workload allocations and
are not an isolated workspace comparison. The separate constraint-only candidate
restart reached API readiness in 6.33 seconds; one-second readiness polling limits
the precision of these HTTP startup observations.

All three C1 outputs match exactly. Both versions pass needle retrieval, prompt
reuse, retained-turn reuse, eight cancellations with eight survivors, and recovery.
Concurrent prose outputs differ; the recorded named checks are not a general
semantic-quality qualification.

The first high-thinking constraint run passed both tool cases but exhausted the
2,048-token budget on one response-format case while reasoning about a missing
schema. That test asked for an object specified only in the decoding constraint,
not in the model-visible prompt. The test now states the required object in the
prompt and keeps the strict schema unchanged. Both versions pass all four corrected
concurrent cases, covering tools, response schemas, JSON and SSE. The original
failure is preserved in the [evidence record](phase1-queued-draft-commit.json).
Failure messages now print a bounded diagnostic rather than every streaming event.

The standard service was restored after each run. This is an intermediate dev
milestone; the published release container remains unchanged.
