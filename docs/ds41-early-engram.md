# Earlier first-lane Engram submission

The scheduler now prepares each lane's request batch as soon as that lane's draft
proposals are known. Previously it generated both lanes' proposals before
preparing either lane. The first lane's background Engram work can therefore run
while the second lane generates its proposals. This preserves the original index
graph lifecycle; the combined graph-retention experiment was not selected.

Preparation creates private work. If second-lane preparation or proposal generation
fails, dropping the first request batch cancels its pending Engram I/O. Verification
still begins only after both batches are prepared. Both layer stacks drain before
commit, retirement or rebalancing; request histories and accepted-prefix rules are
unchanged. Timing traces now separate proposal generation from batch preparation.

The isolated candidate passed the C2/C6/C16 API qualification, including differing
budgets, cancellation and replacement. The earlier combined candidate also passed
the full-model lane fixture, native API checks and eight paired same-mode quality
comparisons; those tests are not represented as a new isolated-candidate run.

Short-counting performance (599 output tokens per request), sequential runs:

| Run | C6 aggregate TPS | C16 aggregate TPS |
| --- | ---: | ---: |
| Candidate, fresh process | 300.55 | 605.45 |
| Baseline, warm | 317.51 | 618.64 |
| Baseline, warm repeat | 318.44 | 618.23 |
| Candidate, warm repeat | 318.16 | 614.89 |

Warmed C6 matches baseline. C16 is approximately 0.6% lower in the warmed candidate
sample; no throughput improvement is claimed. The fresh-process comparison is not
a cold-storage benchmark and cannot establish first-use improvement. The change
moves submission to the earliest available lane-level token boundary and provides
the basis for measuring actual Engram lead time, queueing and completion delay.

[Measurements](ds41-early-engram.json) identify the frozen candidate. The selected
live APIs remain on coordinator stream staging while detailed I/O timing is being
collected; this source change does not yet claim a live rollout or io_uring backend.
