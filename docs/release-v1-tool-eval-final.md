# Final corrected-FP4 tool evaluation

Three hard-mode `tool-eval-bench` runs complete the release's requested tool-call
quality campaign on the corrected FP4 dSpark serving path. Every run uses C16,
thinking enabled, high reasoning effort, temperature zero, a 900-second timeout,
twelve tool turns, and the server's normal output policy without an added token
cap.

| Run | Basic | Hard | Total | Pass / partial / fail |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 119/138 | 34/38 | 153/176 | 68 / 17 / 3 |
| 2 | 123/138 | 36/38 | 159/176 | 73 / 13 / 2 |
| 3 | 120/138 | 36/38 | 156/176 | 71 / 14 / 3 |
| Mean | 120.67/138 | 35.33/38 | 156.00/176 | — |

All 264 scenarios complete without request timeouts, server failures, compressed
pool exhaustion, or worker failures. `TC-43` fails all three runs by generating
an empty required search query, and `TC-68` fails all three by selecting a tool
when none is needed. `TC-33` and `TC-61` each fail once. These are preserved as
model-quality results rather than hidden or rerun.

The user-reported official-API reference scored 117 basic points and 34 hard
points with thinking enabled, and 32 hard points without thinking. This campaign
therefore meets the requested quality bar while retaining the native schema and
serving behavior qualified separately. That external reference was supplied by
the user and was not reproduced as part of this campaign.

The coordinator used an RTX PRO 6000 Blackwell Workstation Edition with the
restored 400 W power limit and standard 14,001 MHz maximum memory clock. Four
GB10 workers supplied the dSpark expert path. The candidate daemon and native
library are identified by SHA-256 in the
[machine-readable summary](release-v1-tool-eval-final.json).

The [complete evidence archive](evidence/native-fp4-tool-eval.tar.gz) contains
all three result JSON files, raw traces, SQLite databases, generated reports,
commands, logs, summaries, launch records, container inspections, worker log
tails, hardware captures, and a per-file hash manifest. This campaign uses bind
mounted candidate artifacts in development containers; the clean final
`build.sh` and `run.sh` image qualification remains a separate release gate.
