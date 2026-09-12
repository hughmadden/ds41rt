# Native constrained decoding

Native XGrammar enforcement remains an open release gate. The initial API audit
confirmed that a strict JSON schema requesting an object with `value: "allowed"`
can return the plain text `DISALLOWED` with HTTP 200. The pinned protocol converter
accepts `json_schema` while discarding its nested fields. Native generation also
currently selects unconstrained greedy tokens. This must be fixed before release;
existing tool-eval scores do not establish schema enforcement.

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

## Remaining enforcement

Preserve response schemas before protocol conversion; compile bounded cached
grammars against the pinned V4.1 tokenizer. Apply masks at first-token selection
and every target verification row. Validate speculative proposals on forked
matcher state and commit only emitted tokens to the request matcher. Support the
V4.1 spaced DSML tags and required/named/parallel tool policies, preserving JSON
parameter types and strict schema constraints. Thinking precedes constrained
answer/tool output; default high effort remains enabled.

Qualify actual API responses, both JSON and SSE, with strict schema and tool
cases, exact-prefix hits under changed constraints, partial/divergent reuse,
concurrency, cancellation and recovery in target-only and dSpark modes. Then
proceed to the architectural FP4 compressed-cache migration and its separate
performance/needle/tool-eval gates.
