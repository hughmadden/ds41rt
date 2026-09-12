# Parallel dSpark vocabulary adjustment and sampling

The sampler now partitions the 129,280-entry vocabulary into 253 tiles of 512
entries, instead of scanning it with one 256-thread block per request. A second
small kernel selects the global winner. It preserves raw adjusted logits,
lowest-token-ID tie breaking, and the existing Philox sequence/draw mapping for
both greedy and stochastic sampling. No native ABI or allocation budget changes.

The first kernel temporarily uses the first 506 output floats for partial scores
and IDs, excluding those locations from ordinary output stores. The second kernel
reads all partials, performs a block reduction and restores those raw logits from
the immutable inputs. Its block barrier orders all partial reads before any
restoration. External consumers remain serialized after stream completion, as
required by the existing interface.

Each tile retains the original 256 logical RNG lanes. Philox initialization skips
exactly the draws for earlier columns in that lane; it does not assign a new
random sequence per tile. Source and output buffers retain the existing bounds
and disjointness checks.

## Qualification

The reusable `scripts/qualify-ds41-draft-sampling.py` compares the frozen selected
native library with the candidate. Two hundred cases preserve every adjusted-logit
bit and sampled token across row counts 1, 2, 3, 8 and 16, all five positions,
greedy/mixed stochastic temperatures, large values, masked logits and ties across
tile boundaries. Fifteen graph replays additionally change inputs and RNG metadata
in-place. Output guards remain intact. Philox tests include nonzero large sequence
offsets rather than only zero-initialized state.

Compute Sanitizer racecheck reported zero hazards/errors/warnings, and memcheck
reported zero errors. Both instrumented the sampler kernels at rows 1 and 16,
including graph replay. The full-model alternating-lane fixture passed in 11.07
seconds. Native API and C2/C6/C16 cancellation/replacement checks passed. Eight
paired same-mode quality cases preserved all text and usage; the inherited Unicode
objective failure remains, so these are not broad quality qualification.

## Measurement

Isolated CUDA-event graph replay, alternating arm order, median of eleven samples:

| Requests | Greedy baseline / candidate | Stochastic baseline / candidate |
| --- | ---: | ---: |
| 1 | about 80 / 4.1 us | about 144 / 4.3 us |
| 3 | about 82 / 6.2 us | about 146 / 6.2 us |
| 8 | about 86 / 6.2 us | about 150 / 8.7 us |
| 16 | about 88 / 10.3 us | about 150 / 14.4 us |

These hot replay measurements differ from the preceding instrumented
[dSpark graph profile](ds41-dspark-node-profile.md), where the original sampler
averaged 171 us per call. Instrumentation and cache/workload differences prevent
using that number as an unprofiled saving prediction.

End-to-end short counting (599 output tokens/request), sequential run order:

| Run | C1 TPS | C6 aggregate TPS | C16 aggregate TPS |
| --- | ---: | ---: | ---: |
| Candidate, first | 130.75 | 309.73 | 625.21 |
| Baseline, first | 131.06 | 309.44 | 626.00 |
| Baseline, repeat | 131.19 | 323.30 | 629.12 |
| Candidate, repeat | 133.89 | 326.84 | 632.62 |

All arms produced identical counting text and usage. The warmed comparison is
about +2.1% C1, +1.1% C6 and +0.6% C16. This is one bounded workload and does not
close the 270-TPS dSpark release target. Target-only decode does not use this
sampler; no target-only performance gain is claimed.

[Complete checks, library hashes and samples](ds41-draft-sampling.json) retain the
evidence. The native library is frozen at
`/tmp/ds41-draft-sampling/selected-native/libds41rt_native.so`; the Rust daemon and
all four Spark worker artifacts are unchanged from the four-slot deployment.

The selected containers are `ds41-sampling-live-target-api-dev` on port 18041 and
`ds41-sampling-live-spec-api-dev` on port 18042. Both passed the native API qualifier
after rollout. Stopped `ds41-engram-slots-live-*` containers retain the preceding
native library for rollback.
