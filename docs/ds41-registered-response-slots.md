# Upload final responses from registered receive slots

The local TP4 client now uploads each final response directly from its CUDA-pinned RDMA receive slot. Previously it copied the whole response into a separate pinned frame before upload. A warm 16k code profile measured 26.854 GB of full-size response frames and 1.300 seconds in that copy stage.

The endpoint is shared with each retained frame so resetting the session cannot unregister storage while a GPU upload still reads it. A retained slot stays unposted until the consumer releases it. The next request drains returned slot indices and reposts them before sending, preserving the existing FIFO receive order. Starting another request on the same session while the old payload is retained fails explicitly instead of reusing its storage.

Only final frames take this path. Earlier streamed chunks are copied/reposted immediately: the reduction owner may retain all uploads until a wave finishes, and a response can contain more chunks than the receive ring has slots. The non-local/pipelined client retains its previous behavior. Native GPU math, wire bytes, response validation and rank-ordered reduction are unchanged. This is host RDMA memory followed by a GPU upload, not GPU-direct RDMA. A small header copy remains.

## Qualification

- Release daemon/example builds and daemon check pass. All 43 regular verbs tests pass; the separately run CUDA pinned-subview test also passes.
- The actual RoCE fixture runs four local endpoints, twelve waves at rows 1/2/6/16/80, one response chunk per row. It retains every payload until the whole wave finishes, verifies only final chunks own receive slots, and rechecks every byte afterward. The 80-row case has 320 chunks across four ranks, exceeding ring depth without deadlock. It also verifies rejection/reset with an old payload retained and cancellation/reconnect recovery.
- The official layer 39 expert fixture runs all four Spark workers at rows 1/6/80/81/256/1024/4096/6. Every response is asserted to own a receive slot. Exact GPU copies, stable owner-vector capacity, failure after asynchronous enqueue, reset before upload drain and subsequent recovery pass across sixteen complete waves and eight injected failures (8.40 seconds).
- Both live APIs pass JSON/streaming, cancellation/recovery and unsupported-sampling checks. All eight paired quality-case texts and usage records match the preceding sparse-split version. The inherited strict Unicode-format failure remains: 5/6 objective checks and 8/8 pair equality; quality command exit 1. Broad model qualification remains open.

## End-to-end results

Both versions use batch 2048/reservation 4096, FP8 projection plans, sparse key splitting and unchanged four-Spark workers. The frozen code and repeated-text workloads contain 16,410/16,411 prompt tokens, have no prefix-cache hits, and generate the same 59-token counting response. Each kind/mode has two requests; the table uses the second request. Prefill is prompt tokens divided by time to first content, including API overhead.

| Context / mode | Before TTFT s | After TTFT s | Before prefill tok/s | After prefill tok/s |
|---|---:|---:|---:|---:|
| Code / target | 12.238 | 11.475 | 1341 | 1430 |
| Code / dSpark | 12.275 | 11.343 | 1337 | 1447 |
| Repeated / target | 10.856 | 10.097 | 1512 | 1625 |
| Repeated / dSpark | 10.771 | 10.018 | 1524 | 1638 |

This is roughly 7–8% better prefill in these bounded workloads. All eight benchmark outputs, usage records and prompt hashes match. Target decode remains around 38–39 tok/s. Repeated-text dSpark is about 116 tok/s. Code dSpark varies: 114.49 then 101.66 tok/s in the candidate's two requests, versus 113.63/113.28 in the earlier baseline run. A separate controlled restart/timing comparison gives 101.67 versus 101.71 tok/s on the warm request. Both take 11 verification steps, accept 48/55 draft tokens and emit 58 tokens after first content; summed verification intervals are 570,476 versus 570,164 microseconds. This does not establish the cause of run-to-run variation, but the matched comparison does not show a transport-induced decode regression.

## Stage attribution

Two-request target profiles use both coordinator and existing RoCE timing, with the second request selected. There are 1,280 full-size rank responses at 2048 rows per warm request. Receive-copy time falls from 1299.855 to 7.309 milliseconds; that interval now covers the small header copy and direct frame validation, not a full payload copy. Request send time is 548.127 versus 561.830 milliseconds. Cross-rank roundtrip intervals overlap and must not be summed as wall latency.

Across all 360 prefill layer calls including the tail, coordinator receive/progress falls from 5.657 to 4.843 seconds. Final reduction plus stream drain changes from 0.152 to 0.180 seconds. Total target-step prefill time falls from 12.211 to 11.434 seconds. The wall saving is smaller than the removed copy interval because some old copying overlapped remote execution. No claim is made that receive time measures pure network latency.

Prefill remains below the nearer 2–3k checkpoint and the 8k release target. Large-row attention, expert execution and remaining coordinator/transport overhead still need work. Concurrency 16, alternating-wave qualification, broad quality and official-only release cleanup remain open.

The live containers are `ds41-receiveslots2048-{target,spec}-api-dev`, ports 18041/18042. Daemon SHA256 is `df2295991f643b50035e04092ec64c9810a66ea1e0fd1304620452e7a6c826d0`; native remains `0aac3fd644b25369baa068947dba326d07134bad897a4912d8d484afc0678385`. The prior sparse-split containers remain stopped for rollback, and all profiling/fixture containers are stopped. Exact launch arrays, comparisons and artifact hashes are in the companion JSON; raw logs are under `/tmp/ds41-receive-ownership`.
