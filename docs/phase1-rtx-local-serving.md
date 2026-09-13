# Initial bottom-up RTX serving integration

The native server can now route complete layers to one RTX using
`--rtx-expert-layers N` or `--rtx-expert-layers auto`. Zero remains the default
while this path is qualified. Both execution lanes share immutable official
FP4 weights and have private, preallocated workspaces. No per-token weight
loading or memory selection occurs. Local execution drains on success or
partial launch failure before publishing a bound FFN result.

Automatic placement budgets the already allocated KV cache, both local lanes,
bounded load staging and 2 GiB runtime headroom under the user's occupancy
ceiling. It chooses a contiguous prefix from layer zero. Explicit counts fail
if they cannot fit; neither mode silently shrinks the KV pool. The current
C16/18-context/24-snapshot pilot selected layers 0–3, consuming 26.89 GiB of
routed weights and 613.34 MiB for both local workspaces. A fifth layer misses
this conservative plan by about 100 MiB. Investigate exclusive workspace reuse
before reducing runtime headroom.

| Local routed layers | Startup to API readiness | Code median tok/s | Code checks |
|---|---:|---:|---:|
| None | 4.29 s | 118.60 | 3/3 |
| Layer 0 | 5.30 s | 118.96 | 3/3 |
| Auto: layers 0–3 | 8.34 s | 120.80 | 3/3 |

These are three samples per arm in sequential order, with one startup per arm
and uncontrolled host page-cache state. They are integration evidence, not a
speedup claim or a loading-speed acceptance result. Generated function bodies
passed the existing objective checks; some responses changed docstring wording,
so output hashes and token counts are not identical. The standard service was
restored after the pilot. All 40 layers remain loaded on each Spark.

The local reduction preserves a BF16 routed-output boundary, then adds shared
FFN and rounds to BF16. Ten exact GPU reference/replay cases cover ordered
routes and token sums at 1, 17, 80, 256 and 4096 rows, including exact shared
aliasing and 70 invalid-argument rejections. These reduction checks do not
qualify large-row expert arithmetic. Four memory-planner tests passed; the
release daemon and complete native coordinator builds passed.

The local kernel currently launches directly with one completion drain per
local layer. Further chaining is a later experiment. Remaining gates include
large-row expert numerics, target-only and prefill throughput, C4/C16 and
retained-context/cancellation checks, tool/needle quality, loading speed,
workspace tuning, Spark layer omission and promotion of automatic defaults.

[Machine-readable evidence](phase1-rtx-local-serving.json) records source and
artifact identities, individual samples and checks. Raw artifacts, build logs,
`serving/pilot.py`, requests and service logs are in `/tmp/ds41-rtx-full-experts`.
