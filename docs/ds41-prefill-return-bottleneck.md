# Prefill expert-return bottleneck

The 1,363-token retrieval workload spends about 493 ms per 80-row target step. Late-layer attention takes about 1.6 ms per layer, while expert dispatch/collection takes about 10 ms. Across the recorded 680 layer collections, median blocking upload time is 1,645 microseconds and the remaining receive time is 7,900 microseconds. Receive time includes waiting for Spark execution, response construction and network delivery; it is not a pure network benchmark.

An experiment allocated four pinned route planes and copied validated response chunks into disjoint ranges, then queued asynchronous H2D copies on the reduction stream. Reduction, reuse and destruction drained that stream. It added 39,321,600 pinned bytes at capacity 80 and preserved all three complete real-fixture FP32 target logit arrays byte-for-byte.

The experiment did not improve end-to-end performance. Three retrieval first-content times were 8.971, 8.726 and 8.731 seconds, versus 8.597 seconds in the earlier baseline. Median upload-call time fell to 1,087 microseconds, but remaining receive time rose to 8,577 microseconds and median 80-row step time was 502 ms. These observations do not isolate every source of variance, but they provide no reason to retain the extra staging allocation. The experiment was reverted, the release daemon rebuilt, and both restored API modes returned the expected arithmetic answer. No experimental runtime code ships in this change.

## Wire-volume constraint

The current native protocol returns six FP32 route vectors of width 5120 from each of four Spark ranks for each of forty layers:

`6 × 5120 × 4 bytes × 4 ranks × 40 layers = 19,660,800 return bytes per input token`.

At 8,000 input tokens/s, this requires 157,286,400,000 return bytes/s. Raptor's `enp1s0np0` reports a 400,000 Mb/s link, equivalent to 50,000,000,000 bytes/s before protocol overhead. Even perfect use of that link limits this response representation to roughly 2,543 input tokens/s, before request traffic or GPU work. Increasing batch sizes or changing TCP to RDMA cannot by itself meet the requested 8k prefill target while preserving this payload volume.

The return representation must change. Candidates include combining route contributions before transport or moving reduction among the Sparks and returning a compact result. Such changes require a bandwidth model for every link and numerical qualification: the current reducer sums TP ranks per route, rounds each route to BF16, then sums routes, so moving those operations can change rounding. This finding does not select or qualify a replacement.

[Timing observations, logit hashes, test log and bandwidth calculation](ds41-prefill-return-bottleneck.json).
