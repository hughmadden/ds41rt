# Decode lane join-barrier trace

One RTX PRO 6000 Blackwell, **400 W power limit and standard memory speed**,
four unchanged Sparks, five complete local expert layers, 18 × 1,048,576-token
KV capacity and 24 retained snapshots. Traces use the adopted narrow FP8
projection library and `RUST_LOG=warn,ds41rt::timing=debug`. No builds overlapped
inference. Both arms completed and standard serving was restored.

The scheduler now records each verifier future's completion offset and its wait
until both futures return. These are host-visible timestamps after logits are
downloaded. Empty lanes are excluded. The summary verifies that both lane
completion-plus-wait timestamps identify the same join and precede round end.
Only rounds with exactly the stated number of active requests enter each row.

| Policy / active requests | Rounds | Draft median ms | Round median ms | Faster lane wait median ms | Wait p95 ms |
|---|---:|---:|---:|---:|---:|
| Joint cost / 4 | 46 | 6.85 | 81.77 | 2.94 | 5.75 |
| Joint cost / 16 | 42 | 12.71 | 183.99 | 10.18 | 12.36 |
| Incremental reuse 0.05–0.5 / 4 | 45 | 6.89 | 84.60 | 2.95 | 4.63 |
| Incremental reuse 0.05–0.5 / 16 | 41 | 12.72 | 194.38 | 10.01 | 13.64 |

Summed faster-lane wait divided by summed round time is 4.0%/5.6% for cost
C4/C16 and 3.6%/5.3% for reuse. This is not a throughput prediction: the other
lane still occupies shared hardware. It excludes subsequent commit and next-draft
barriers. Instrumentation changes execution timing, and medians include cold
shape encounters. These are diagnostic intervals, not performance table updates.

Each arm used one code request and a 32K fresh prefill warmup before the C4/C16
mixed screen. All code structure, prefill-answer and mixed completion checks
passed. Completion checks do not establish prose quality or exact paired outputs.
The raw command/artifact manifest and all parsed scheduler rounds are in
[the evidence](phase1-lane-barrier.json). Reproduce the summary with
`scripts/summarize-ds41-lane-barrier.py TRACE... --output JSON`.

Source inspection identifies a reusable ownership pattern: encoder streaming
already advances across chunk boundaries using a shared request bank, and
`PreparedLayer` owns no cache-bank borrow while waiting for the remote FFN.
Decode currently keeps the whole bank borrowed across its execution future and
commits only after joining both futures. Independent decode progress must scope
bank access to preparation, preserve each lane's device/transport owners, and
commit/replenish ready work without revoking another lane's active leases. The
current trace measures the motivation; it does not implement that scheduler.
