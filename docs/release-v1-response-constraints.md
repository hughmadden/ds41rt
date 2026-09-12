# Native response constraints

Native generation now enforces `json_schema`, `json_object` and regex response
formats. The prior strict-schema reproducer returned `DISALLOWED` with HTTP 200;
it now returns the required JSON object in both target-only and dSpark modes.
This completes the response-format stage of XGrammar integration. Strict tool
arguments and tool-choice/parallel-call policies remain open release gates.

## Implementation

The API preserves response schemas before the pinned recipe converter discards
their nested fields. It validates schemas locally with network/file resolution
disabled. Validation defaults to Draft 2020-12, honoring an explicitly declared
older dialect. Regex uses the native constraint path while the recipe renders
ordinary text. JSON object mode permits arbitrary properties.

The native adapter expands the project-owned `ds41_json_schema` structural node
into an EBNF grammar with the requested strictness and at most sixteen whitespace
characters per JSON boundary. User schema data is excluded from recursive node
rewriting. The pinned XGrammar dependency is unchanged. Thinking is an unrestricted
token prefix until the pinned `</think>` token (128822), followed by constrained
content; EOS is token 1.

The worker lazily compiles grammars into a 64-entry LRU. Every request owns a fresh
matcher. Exact prefix hits reselect their first token from retained raw logits,
so a different schema or strictness setting cannot inherit the previous winner.
Each target verification row is masked under the corresponding forked grammar
state. Illegal dSpark proposals are truncated before target execution, and only
emitted tokens advance the authoritative matcher. Intermediate proposal state
is never installed in the retained prefix.

The API independently validates completed JSON answers. A nonconforming answer
fails instead of returning a successful stop/DONE; a stream may already contain
an unfinished prefix when that error is detected. Length-limited output retains
its normal `length` finish reason and can be incomplete JSON. A combined response
schema/tool request permits tool dispatch and validates final answers; its DSML
tool-body branch remains permissive until strict tool enforcement is implemented.

## Qualification

[Summary and hashes](release-v1-response-constraints.json) and
[raw evidence](evidence/native-response-constraints.json.gz) include the candidate
binary/library, source hashes, dependency pin, launch settings, all live records,
failed preliminary runs and build/test logs.

The native build and existing XGrammar selftest pass. Thirteen API tests pass,
including schema preservation, strictness, dialect selection, and rejection of
invalid completed output in JSON and SSE. Live qualification has sixteen paired
JSON/SSE cases per mode, covering changed constants on an identical cached prompt,
Unicode, completed-turn continuation, strict/open properties, nested local references, integers with bounds,
nullable values, string patterns/length, tuple items, default high thinking,
JSON objects, regex and post-cancellation recovery. All paired answer/reasoning texts and token
counts agree, and the repeats are exact prompt hits.

Additional live checks cover automatic, required and named tool dispatch alongside
a response schema; a one-token length limit; concurrent requests with different
schemas on the same prompt at C16 target/C2 dSpark; cancellation of a constrained
array followed by recovery; and HTTP 400 for malformed schema definitions. The
suite independently validates JSON using Python jsonschema 4.23.0. This is focused
coverage, not a claim that every JSON Schema feature always yields a completion.

Failed preliminary runs are preserved. The first reached regex and exposed
the recipe's conversion-time rejection; native routing fixed it. The second
rejected an otherwise valid `prefixItems` tuple because completion validation
used Draft 7; modern/default and explicit legacy dialect tests now cover that
correction. An additional exact-turn assertion exposed noncanonical generated
tokenization: `{"value":"allowed"}` used six generated content tokens but five
canonical tokenizer tokens. That follow-up correctly reuses the prompt frontier,
then recomputes the changed token sequence. The final suite keeps that case and
separately proves full completed-turn reuse with canonical `42` followed by a
new schema requiring `43`. No failed run is counted as a qualification pass.

Four alternating AB/BA pairs per workload compare the final candidate with
`8a7944b` using 256-token counting and code outputs, serially sharing the four
Spark workers. Median decode changes are target counting -0.01%, target code
+0.20%, dSpark counting +0.06%, and dSpark code +0.37%. All texts and token counts
match. These focused checks show no material plain-text regression and are not
release headline measurements. RTX power limits are 400W with standard memory
speed (loaded clock 13,365 MHz); hardware and driver are captured in the evidence.

Reproduce the correctness suite against an exclusive native endpoint with
`scripts/qualify-ds41-response-constraints.py --base-url URL --concurrency N
--output FILE`, using Python with `jsonschema==4.23.0`. Run target and dSpark arms
sequentially when they share Spark workers.

The full strict-tool gate, three final high-thinking tool-eval runs, FP4 cache
migration and final release performance suite remain outstanding.
