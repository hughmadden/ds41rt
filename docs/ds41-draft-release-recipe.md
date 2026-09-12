# Draft expert release recipe and capacity accounting

The normal coordinator expert exporter now delegates to b12x's qualified draft
slice recipe. b12x owns `V41DraftSlicePipeline.DEFAULT_WIDTH`; explicit width
exports remain available for comparisons. Standard builds retain the expected
`v41_coordinator_m{capacity}` artifact names and BF16-input/FP32-route ABI.
CMake tracks the delegated slice exporter as a dependency. No serving flag is
needed to enable this backend in a newly built coordinator library.

The Spark export path is unchanged. This update selects the local draft expert
backend; it does not select an io_uring Engram backend. If io_uring is later
selected, `run.sh` must apply its bundled seccomp profile automatically.

## Correct serving budget dimensions

N64 qualification exposed an existing admission error: serving passed the
4096-row prefill capacity to the generic dSpark expert-wave budget, although
its draft chain is bounded to 16 requests / 80 proposal rows. This is a budget
estimate error, not a 4096-row expert arena that the live chain actually allocated.
The initial diagnosis of two instances exhausting physical GPU memory was wrong.

The N64 arena is 182,430,416 bytes at capacity 80, but 9,340,307,984 bytes at 4096.
Multiplying the latter across three stages inflated the estimate enough to
reject startup under the 32 GiB dSpark limit. N192 arenas are 64,465,616 and
3,300,510,224 bytes respectively.

`DsparkWeights::load_serving` now derives expert capacity from the bounded
request count. Before loading weights, it reserves the complete large-context
projection owner separately and passes the remaining budget to the conservative
draft-wave admission check. The 32 GiB limit remains intact. Full committed
windows remain in the existing wave plan; the small context reservation in that
plan is conservatively retained in addition to the separately reserved full
context owner. Generic component loading keeps its original capacity contract.

N64 now starts with the original 4096-row prefill settings. Both N64 and N192
comparison arms use the same corrected daemon. This fixes admission accounting;
it does not claim a reduction in actual allocated expert storage for N192.


## N64 versus N192 on live counting

The four arms ran sequentially on RTX GPU1 in N64/N192/N192/N64 order.
Each fresh server first completed a priming C1/C6/C16 sweep, followed by a warm
sweep. Both used the corrected daemon, identical prefill settings and the same
four Spark workers. All 184 priming and measured responses preserve identical
text and usage.

| Warm arm | C1 TPS | C6 aggregate TPS | C16 aggregate TPS |
|---|---:|---:|---:|
| N64 first | 144.18 | 362.92 | 679.26 |
| N192 first | 143.74 | 363.08 | 678.71 |
| N192 second | 143.96 | 363.34 | 680.35 |
| N64 second | 143.73 | 362.42 | 680.34 |

There is no meaningful throughput advantage for N64 in this workload. N192
remains selected: it uses about one-third of the slice scratch and led larger
dispersed component cases. This does not establish optimal tiling across every
prompt category. N64 native replay checks and memcheck pass; its startup
rejection under the old budget check is retained in the diagnostic logs.

## Default build validation

A coordinator rebuild with `DS41RT_V41_EXPERT_SLICE_WIDTH` empty exports all six
standard coordinator capacities through the b12x N192 recipe. The manifest
records coordinator/BF16 input, width192 and non-experimental default selection.
Twenty native-output comparisons at live rows 1/5/15/40/80 with shared and
dispersed routing remain exact against the previously selected N192 library;
changed inputs reuse captured graphs and preserve output tails. Raw evidence is
under `/tmp/ds41-draft-native64/` and `/tmp/ds41-draft-release-default/`.

The default-built library and corrected daemon are deployed to both development
endpoints. API, C2/C6/C16 recovery and eight paired texts/usages pass, as do both
post-rollout API checks. The inherited Unicode failure remains. See the
[next-session handoff](HANDOFF_NEXT_SESSION.md) for exact selected binaries and
[the evidence summary](ds41-draft-release-recipe.json) for hashes and raw timings.
