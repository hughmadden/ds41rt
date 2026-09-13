# Chained embedding/query: rejected candidate

One RTX PRO 6000 Blackwell, 400 W power limit, standard memory speed,
four Spark workers, five RTX resident expert layers and unchanged default KV
budget. Baseline serving source: `fe0a090` (`d92caad` adds documentation).

The candidate produces embeddings directly into block residual/pre storage,
eliminating the intermediate copies. Token IDs use owned pinned staging. It
queues embedding, mHC and query preparation on the query stream before waiting
cooperatively; subsequent layer queries also wait cooperatively. Query graph
capture follows completed warmup, with no suspension inside capture. Kernel
math and the native library are unchanged. No additional GPU workspace is used.

Exact embedding/image replacement checks and 56 real-weight query cases pass,
including changed inputs, cold/warm graphs and recovery after producer errors.
Both initial arms pass cache reuse, retained turns, cancellation/recovery and
high-thinking constrained-output checks. C1 code outputs match in all three
repeats, with median throughput 129.83 → 129.84 tok/s.

## Ordered mixed comparisons

Adaptive dSpark, independent lanes, identical nonce seeds. First pair runs
candidate then control; the two follow-ups run control then candidate. Follow-ups
retain C2/C4 for workload ordering/warmup and skip repeated lifecycle checks.
No builds overlap inference. Each cell lists control → candidate tok/s.

| Concurrency | Pair 1 | Pair 2 | Pair 3 | Median change |
| --- | ---: | ---: | ---: | ---: |
| 2 | 92.63 → 92.10 | 93.11 → 91.54 | 92.03 → 91.17 | −1.2% |
| 4 | 114.74 → 113.47 | 114.51 → 116.61 | 114.68 → 127.46 | +1.7% |
| 8 | 152.14 → 159.20 | 158.82 → 159.53 | 152.88 → 155.53 | +4.1% |
| 16 | 181.77 → 162.87 | 167.66 → 158.38 | 165.10 → 159.84 | −4.7% |

C16 declines in all three pairs. The C8 improvement does not justify adopting
this version. The serving source has been restored, and the standard service
was restored after both runners. These are exploratory comparisons, not the
final release tables. Mixed prose outputs can differ under changing batch shapes;
this evidence establishes neither a numerical defect nor the cause of the
throughput decline.

The next revision should examine the remaining synchronous cache/index/attention
work adjacent to query completion. Moving the yield boundary alone has not
passed the performance gate; do not claim a causal explanation without profiling.
See the [remaining wait audit](phase1-lane-wait-audit.md).

[Archived patch](phase1-chained-query-candidate.patch) applies to `d92caad`.
[Results](phase1-chained-query.json) retain commands, artifact identities,
individual observations and GPU test logs. Raw files are in
`/home/tj/.cache/ds41rt-experiments/chained-query`.
