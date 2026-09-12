# Queue coordinator staging on consumer streams

Query preparation, local-window projection, sparse attention and output projection
now queue input and metadata copies on their own CUDA streams. Their existing
final drains establish completion before output publication. This removes
separate blocking copy boundaries and one redundant pre-window drain, preserving
all kernels, arithmetic, cache transactions and graph bindings.

Query and output positions use retained pinned host buffers instead of temporary
vectors. Sparse replay bounds have a separate pinned arena so writing them cannot
overwrite metadata while its upload is in flight. Additional pinned storage is
24 bytes per row of lane capacity: 192 KiB for two 4,096-row lanes. Device storage
and graph counts are unchanged. Partial staging, preparation and replay failures
drain before input borrows or host arenas can be reused.

## Qualification

The full-model C2/C6/C16 lane fixture passes, including request retirement,
migration, draft execution and byte-exact serial/overlapped outputs. The existing
real-weight query fixture passes all 56 comparisons plus partial-producer failure
and reuse. Speculative API lifecycle and C2/C6/C16 distinct-sequence cancellation
and replacement pass. Eight paired target-only prompts preserve text and usage;
the inherited Unicode-format failure remains. This is regression qualification,
not broad model-quality qualification.

## Measurements

All long comparisons are sequential baseline/candidate runs on the same GPU for
each mode, counting 599 output tokens after 16k code context. Baseline APIs were
already running; candidates were freshly launched. The cold samples are retained
in [the result record](ds41-stream-staging.json), but this is not an isolated
startup comparison.

| Warm comparison | Baseline | Candidate |
|---|---:|---:|
| Target C1, second pair | 39.86 TPS | 41.80 TPS |
| dSpark C1, last two of three pairs | 120.89 TPS | 123.49 TPS |
| Short-prompt dSpark C16, initial aggregate | 609.68 TPS | 611.49 TPS |

Target C1 improves about 4.8% in the warm pair and dSpark C1 about 2.2% across
the two warm pairs. These small samples do not establish statistically precise
gains. All long outputs and usage match. Warm code prefill remains approximately
6.7–7.2k tokens/s; the two versions vary within that range, with some lower
candidate samples. Cold candidate prefill is slower than the already-running
baseline, and that startup effect has not been isolated.

Short-prompt C6 starts at 314.19 baseline versus 305.43 candidate aggregate TPS.
Repeats in candidate/baseline/baseline/candidate order are
298.59 / 312.55 / 312.38 / 319.01 TPS, with the candidate restarted before the
first repeat. Warm performance recovers, but first-use behavior remains open.
These aggregate measurements include admission gaps and are not the same
workload as the long-context C1 comparisons.

A second node-level trace confirms the change: blocking copies fall from about
383 to 103 per target step, and stream-synchronize calls from about 1,014 to 853.
Asynchronous copies rise from about 495 to 775 per step. The 280 copy replacements
and roughly 160 fewer synchronization calls match the staged paths; graph counts
remain essentially unchanged. These counts come from steady 5–19 second intervals
of separate instrumented runs, normalized by logit downloads.

The [kernel-node profile](ds41-coordinator-node-profile.md) motivated the change.
It also shows why host-side changes alone will not close the release-performance
gap: substantial RTX GEMM, attention and other kernel time remains alongside
remote expert execution.

## Selected artifacts

`ds41-staged-live-target-api-dev` and `ds41-staged-live-spec-api-dev` serve
18041/18042 with the frozen `/tmp/ds41-staged/daemon`, SHA-256
`e897c49122e8e64850ceb5283b53fe2ede574b7f496a10b4d958d11f39cd5bb2`.
Both normal endpoints pass post-selection lifecycle qualification. The native
library remains `/tmp/ds41-rmsnorm/selected-native/libds41rt_native.so`; the four
Spark workers and pinned b12x revision are unchanged. Previous RMSNorm containers
are stopped for rollback. Creation commands and raw results are retained under
`/tmp/ds41-staged`.
