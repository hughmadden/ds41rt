# Independent draft workspaces

Each active decode lane now owns its dSpark chain, stream, graph cache, token and
confidence staging, and pending replay. At the default concurrency of sixteen,
each workspace supports eight requests. Both borrow the same immutable weights,
embedding and vocabulary head; persistent cache slots and RNG state remain
request-owned. There is no wait for the other lane to release a draft workspace.

Replay completion remains cooperative. Cache-slot read reservations survive until
the corresponding lane's GPU work completes, permitting disjoint commits while
rejecting overlapping writes or recycling. Request release rejects a pending draft
before removing its runtime mapping. A single active lane uses the direct path
and reuses workspace zero after the cohort drains. Concurrency one allocates only
one workspace; other limits allocate two at half the limit, rounded up.

Eight-request workspaces use forty-row storage. The native coordinator export
adds a forty-row expert variant; FP8 projection plans keep the existing 1/16/80-row
kernels and enforce a forty-row logical bound. Existing projection math and live
row selection are preserved. RoPE rotates Q in place instead of retaining a
second query buffer. Exact aliasing is allowed; partial overlap remains invalid.

The three main-context KV projections now borrow their parent's normalized input
instead of allocating unused private inputs. This saves 120 MiB at the serving
capacity. Weights and persistent windows are not duplicated. Draft-chain storage
is 468,695,984 bytes total (about 447 MiB), versus 449,504,184 bytes (429 MiB) in
the original shared implementation. The logged hypothetical shared layout with
the same in-place RoPE cleanup is 433,775,544 bytes. Startup GPU occupancy in the
first fitting candidate is 50 MiB above the earlier control; this includes CUDA
overhead and is distinct from explicit scratch accounting.

Initial versions failed the startup memory planner before issuing API requests:
two eighty-row allocations added 322 MiB of explicit scratch; forty-row storage
and then in-place RoPE reduced it, but only removing the unused commit inputs
left enough space for five local expert layers. Every failed attempt restored
standard serving. The final layout preserves the 2 GiB runtime reserve and KV pool.

The comparison uses one RTX PRO 6000 Blackwell at **400 W and standard memory
speed**, four unchanged Sparks, five RTX expert layers, KV capacity of
18 × 1,048,576 tokens and 24 retained snapshots. The control is `7244e18`; both arms
use cooperative replay, GPU top-1 and the same native library. No builds overlap
timed serving. Final raw records are under `~/.cache/ds41rt-experiments/split-draft-fit`;
earlier startup failures remain in the other `split-draft*` experiment directories.

| Workload | Shared draft workspace | Independent workspaces |
|---|---:|---:|
| C1 dSpark code, tok/s | 129.27 | 130.38 |
| 32K prefill, tok/s | 7,641.48 | 7,732.52 |
| C2 mixed, tok/s | 107.19 | 105.77 |
| C4 mixed, tok/s | 114.64 | 125.06 |
| C8 mixed, tok/s | 167.60 | 165.78 |
| C16 mixed, tok/s | 204.82 | 199.11 |
| Startup to API ready, seconds | 5.311 | 5.313 |
| Peak observed GPU memory, MiB | 96,924 | 97,030 |

C1/prefill are three-sample medians; mixed traffic is one cold batch per
concurrency, including admission gaps. The candidate ran first. C4 improves 9.1%
while C2/C8/C16 decrease 1.3%/1.1%/2.8%. These exploratory measurements do not
establish a uniform gain or a regression-free result at every concurrency. The
peak observed memory increase is 106 MiB, including runtime CUDA allocations.

Both arms pass needle retrieval, prompt/retained-turn reuse, cancellation,
survivor and recovery checks; all three matched C1 outputs are exact. Candidate high-thinking tool/strict-JSON cases pass,
including streamed and ordinary responses. The native in-place RoPE test compares
forty rows at one and sixty-four heads, both rotation directions, against separate
output bit-for-bit; it checks unchanged nonrotary coordinates, the end sentinel
and rejection of partial overlap. Native API/query/argmax checks and the Rust FP8
plan test pass, as does the release build. Standard serving is restored.

[Evidence](phase1-split-draft-workspaces.json) includes commands, artifact hashes,
samples, responses, memory/startup records, earlier planner failures and checks.

Shared main-context preparation/cache commit, cohort retirement/cancellation,
constrained logits and smaller per-layer host waits remain follow-up work.
Admission/prefill remains a deliberate shared boundary.
