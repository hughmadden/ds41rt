# Expert phase is not fabric latency

Reanalysis of the mHC-chain rollout traces separates coordinator work from the Spark callback. The existing summary script previously recognized only the legacy server timing label, silently omitting persistent RoCE server records. It now recognizes RoCE and coordinator stages, retains mean/median/p95/max, and explains overlapping and nested intervals.

For 10,680 one-row layer calls, coordinator medians are:

| Sequential coordinator stage | Microseconds |
|---|---:|
| Routing and wire request preparation | 50 |
| Dispatch into four client workers | 4 |
| RTX shared expert execution | 53 |
| Collection and reduction | 219 |
| Entire expert phase | 328 |

Shared expert execution overlaps the remote request. Dispatch is enqueue time, not NIC completion. Collection contains shared-output copy (5 µs), four rank-plane uploads (21 µs aggregate), receive/wait and client handling (183 µs), and reduction (6 µs). Medians are independently aggregated and do not add exactly. These collection components are nested inside collection, not extra costs.

The matching time window on ostrich contains 10,680 one-row calls: kernel 156.672 µs, compaction 4.096 µs, input upload 15 µs, output download 5 µs, complete native execution 186 µs. Server callback is 191 µs and server boundary 202 µs. Posting the send measures 1 µs and waiting for a prior send completion 1 µs at median; neither measures end-to-end fabric latency. These are one-rank statistics, not a correlated slowest-of-four decomposition. Instrumentation remains enabled and clocks are not controlled.

Source inspection confirms three synchronous Spark uploads (activation, IDs, routing), a synchronization during layer binding and another before upload, a drain before output download, and host response encoding. The mapped receive payload is exposed by transport but ignored by the native worker. The RTX also uploads each returned rank plane before reduction.

This changes the optimization priority: mapped input alone addresses only part of the 15 µs Spark upload interval. RTX routing/preparation (50 µs), rank-plane uploads (21 µs), client handling and synchronization must be tracked as well. A zero-copy path needs GPU alignment, slot lifetime and completion qualification; pointing the current kernel at a mapped payload is not sufficient evidence of correctness or speed. No pure RoCE latency claim follows from these traces.

[Filtered one- and six-row metrics and source hashes](ds41-expert-boundary-breakdown.json) retain the evidence. Reproduce with `scripts/summarize-ds41-expert-timing.py` against the listed source logs; the unfiltered command also reports prefill row counts. No runtime behavior changed in this analysis.
