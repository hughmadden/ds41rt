# Rejected local routed-expert graph

A bounded graph combining the local routed-expert pipeline and final shared
addition did not demonstrate a useful serving improvement. The production
source retains direct local launches. This experiment did not remove the
separate router or shared-expert completion boundaries.

ABBA comparison against `d257c1b`: five complete local layers in both versions,
adaptive dSpark, C16 admission, 18 × 1,048,576-token KV capacity and 24 retained
snapshots. One RTX PRO 6000 Blackwell at 400 W and standard memory speed, with
four unchanged Sparks. No builds overlapped benchmarks. Code used no thinking.

| Metric | Direct baseline | Graph candidate |
|---|---:|---:|
| Code median tok/s, six requests each | 123.14 | 122.79 |
| 32K prefill median tok/s, six measured requests each | 7,820.12 | 7,755.50 |
| C4 aggregate median tok/s, two batches each | 118.17 | 118.71 |
| C16 aggregate median tok/s, two batches each | 184.87 | 185.45 |

All 12 code structure checks, prefill cache checks and mixed batches passed.
All 16 prefill answers were `7`, with 32,768 new tokens and no cache hits.
The first baseline/candidate pair also passed needle retrieval, prompt reuse,
retained-turn reuse, cancellation, survivors and recovery. The second pair
omitted lifecycle repetition. These checks do not establish broad semantic
quality or a target-only speedup; target-only was not rerun for this candidate.

The candidate retained graphs for live row counts 1–48 per local layer and
execution lane. Larger prefill shapes stayed direct. Replay checked captured
input addresses and drained before publishing the result. Captures followed a
completed warmup; graph destruction followed the owning stream's drain and
preceded captured owned storage. Candidate arms logged 325 and 310 local graph
captures, with no layer/shape key appearing more than twice across two lanes.
These aggregate logs are consistent with bounded reuse, but do not individually
identify lanes. All logged shapes were within the intended bound.

The [candidate patch](phase1-rtx-local-graph-candidate.patch) preserves the tested
approach without enabling it. A subsequent draft GPU replay test and helper
refactor were archived only in the raw directory; they were not compiled or
run, and are not correctness evidence. They were abandoned when the serving
comparison failed to establish a gain.

The next completion-boundary investigation is shared-expert input staging:
`BackboneSharedWave::execute_ffn` currently performs a completed D2D copy before
launching its consumer graph. Queuing that copy with the consumer would affect
all 40 layers. It must retain a drain on both successful publication and partial
failure, and must be evaluated with complete serving measurements.

The standard service was restored. [Evidence](phase1-rtx-local-graph.json)
records individual samples, commands, artifact hashes and raw evidence paths.
