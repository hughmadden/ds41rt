# Upload directly from retained RoCE response frames

Native FFN collection now takes ownership of each validated response payload and uploads directly from its pinned storage. Previously it copied the payload into a second pinned arena first. The upload guard retains all frames through the final CUDA stream drain and recycles them on drop. Error/cancellation drop also drains before releasing frames. The payload owns its pinned allocation independently of the QP receive ring, so resetting a connection cannot free an active upload source.

`receive_owned` transfers validated ownership; the existing borrowed receive API adapts to it. Executor/peer identity, frame extent, canonical row coverage and final-chunk checks remain. A sink error abandons the entire pending wave and resets QPs. Collection cannot resume that failed wave. Pageable payloads retain a synchronous-copy fallback. GPU arithmetic, wire bytes and rank-ordered reduction are unchanged.

The additional pinned arena is removed: 40 MiB at the selected 1024-row capacity, 160 MiB at capacity 4096. A preallocated owner vector holds at most four times capacity rows, covering even one-row response chunks without growth during execution. The logical payload copies removed for a 16,410-token prefill across forty layers and four ranks total about 25.04 GiB. This is source-derived copy volume, not measured DRAM traffic. The copy from registered receive storage into the pinned response frame remains; this is not GPU-direct RDMA.

## Qualification

Ten transport tests pass. The pageable fallback test covers interleaved three-row chunks, rank ordering, exact reduction, bounds errors and stable owner-vector allocation at rows 1/6/80/81/256/1024/4096/6.

The new ignored `retained_roce_uploads_drain_before_recycling_live` fixture runs against official layer-39 experts on all four Spark workers using generated FP8 activations and routes. All received payloads are asserted pinned. Device copies match the actual response bytes exactly at the same eight row shapes. Sixteen complete waves and eight failures injected after asynchronous enqueue pass; QPs reset while uploads still own their frames, guard drop drains, and the following request recovers. The fixture finishes in 7.97 seconds. This tests actual pinned-frame lifetime and transfer integrity, not independent model numerics. Production and test binaries build; existing compiler warnings remain.

Both quiet APIs pass JSON, three streaming counting requests, cancellation/recovery and unsupported-temperature rejection. All eight paired quality cases preserve prior text and usage. The inherited strict Unicode formatting failure and 7/8 cross-mode agreement remain; broad quality remains open.

## End-to-end measurements

Sequential C1 greedy counting after frozen repeated text or code context, 59 completion tokens and no prefix-cache hits. Actual prompt counts are 16,411 and 16,410 respectively. Both configurations retain healthy QPs across requests and use 1024-token prefill batches. Control precedes candidate; these are bounded status measurements rather than a clock-controlled statistical benchmark.

| Context | Mode | Control / candidate TTFT s | Control / candidate prefill tok/s | Candidate decode tok/s |
|---|---|---|---|---|
| repeated | target | 11.777 / 10.757 | 1393.4 / 1525.6 | 25.99 |
| repeated | speculative | 11.673 / 10.673 | 1405.9 / 1537.6 | 94.69 |
| code | target | 14.817 / 13.310 | 1107.5 / 1233.0 | 25.91 |
| code | speculative | 14.264 / 13.229 | 1150.5 / 1240.5 | 93.93 |

All four prompt hashes, texts and usage records match. The prefill improvement appears in both contexts and modes; no statistically established decode gain is claimed. The nearer 2–3k prefill checkpoint, original 8k target, 90/270 decode targets and concurrent serving remain open.

## Stage attribution

Warm code-context profiles of the preceding admission-reuse version and this candidate each cover seventeen prefill steps and 680 layer calls. Host upload-call totals fall from 1.009411 s to 0.008347 s; median at 1024 rows is now 12 microseconds. Collection falls from 8.012030 s to 7.048144 s. Receive/progress is approximately unchanged at 6.857/6.888 s, while reduction plus final stream drain changes from 0.139538 to 0.148103 s. Total target-step time changes from 14.360284 to 13.280360 s. These nested timings support approximately one second of saving from removing the host copy, rather than merely moving that time into the stream drain. They do not isolate pure network latency.

Sparse attention remains 3.032 s in the candidate profile; expert phases total 8.927 s. The next work still needs to address remaining response handling and expert kernel efficiency under skewed sharing. Profiles are instrumented and run sequentially; raw observations and prompt identities are retained in [the evidence record](ds41-retained-response-uploads.json).

## Deployment

Selected daemon SHA256: `58c52b150560a740b48dbd19d3e7e1df94c5b9d070867ebb284c57c939880079`. Native library remains `07db52dec294d18ee98e44b553a7c442a43cdabb65ff8103690572fc4eeaa42b`, b12x e5343cfb. Quiet APIs are `ds41-retained-{target,spec}-api-dev`, ports 18041/18042. The four capacity-4096 Spark workers retain their artifacts and configuration. Previous admission APIs remain stopped for rollback. Frozen daemon, exact launch arrays and raw tests/API traces are under `/tmp/ds41-retained-uploads`.
