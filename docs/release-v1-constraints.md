# Native constrained decoding

Native response schemas are now enforced through XGrammar; see the
[response qualification](release-v1-response-constraints.md). Strict tool-argument,
selection and parallel-call enforcement remain open release work. The initial
API audit found that a strict schema requesting `value: "allowed"` returned
`DISALLOWED` with HTTP 200. Preserving the schema across protocol conversion and
masking generation now fixes that reproducer, including exact cached prompts.

## Retained logits prerequisite

A retained prefix now owns its raw final FP32 vocabulary logits and the ordinary
greedy winner. New constraints can reselect the first token from those logits
without replaying an exact KV hit. Grammar state must belong to the new request,
not the retained prefix. Partial/replayed frontiers explicitly have no saved
logits; their new prefill computes the correct row.

Rows have 129,280 entries (517,120 bytes). The 24 prompt and 24 completed-turn
banks bound retained host logit payload to 24,821,760 bytes (23.67 MiB), plus active
request rows and the current verification batches. Shared rows use reference
counting. Device allocations and D2H logit transfers are unchanged. Intermediate
decode rounds do not copy rows into retained storage: only finishing requests
copy their last committed row, including the dSpark accepted-input frontier.
Ordinary exact hits use the cached winner directly.

The focused tests exercise changed masks on the same retained row, an empty
allowed-token set, ownership after the originating batch is dropped, malformed
rows and non-finite logits. Existing radix/retention tests also pass. Live checks
and performance results are recorded in the [summary](release-v1-constraints.json)
and [raw evidence](evidence/native-retained-logits.json.gz).

Both modes pass 75 live retention requests: 24 prompt hits, 24 completed-turn
resumptions, oldest-turn eviction, newer-turn survival and cancellation. Nine
Unicode cases per mode retain JSON/SSE text and usage parity, including one-token
output limits and EOS completion. The two CPU score tests and four existing
prefix tests pass; the release binary builds successfully.

Four alternating AB/BA pairs per workload, 96 tokens each, compare with the
previous Unicode serving binary on the same GPU, sequentially sharing the four
Spark workers. Median decode changes are target counting -0.80%, target code
-0.07%, dSpark counting -0.09%, and dSpark code +0.12%. The target counting delta
prompted a preserved four-pair 256-token followup: -0.08%, with runs moving in
both directions. All pairs have identical text and token counts. These focused
checks show no material decode regression; they do not establish release-wide
performance. RTX power limits are 400W, memory uses standard speed (observed
loaded clock 13,365 MHz), and the driver is recorded in the evidence.

## Shared enforcement and remaining tools

Response grammars are compiled against the pinned V4.1 tokenizer in a bounded
64-entry LRU. Request-local matchers mask first-token selection and each target
verification row; forked matchers truncate illegal dSpark proposals. Only emitted
tokens advance authoritative matcher state. Thinking precedes the constrained
answer and defaults to high effort. Completed JSON answers receive independent
schema validation before a successful completion is emitted.

Strict tool arguments, required/named selection, parallel-call limits and their
streaming behavior still need the V4.1 DSML grammar. Combined response/tool
requests currently preserve a permissive tool-block alternative: the response
schema applies to the final answer, and this is not strict argument enforcement.
Finish and qualify those tool policies before the architectural FP4 compressed
cache migration and its separate performance/needle/tool-eval gates.
