# Native tool constraint serving

Native Chat Completions requests now carry V4.1 DSML grammars for tool names,
strict argument schemas and parallel-call limits. Automatic selection permits
ordinary answers or a tool block. Required and named selection force a tool
block after reasoning. Native selection leaves the assistant outside the calls
block when rendering the prompt, so required/named requests can use the default
high thinking mode; explicit thinking overrides still apply.

Strict tools use the [typed parameter converter](release-v1-tool-grammar.md).
Non-strict tools retain generic object arguments. Automatic non-strict requests
can disable decoding assistance with `tool_decoding_assistance: false`.
Explicit strict schemas, required selection and combined response constraints
retain their grammar enforcement. `parallel_tool_calls: false` admits one call;
the default allows multiple calls. Response schemas constrain the answer branch,
while argument schemas constrain each selected tool independently.

Completion validation checks selected names, call counts, object arguments and
strict schemas before emitting a successful finish. A recursive JSON reader
rejects duplicate properties, including escaped aliases, instead of accepting
the last value. JSON and SSE share this validator. SSE may already have delivered
argument fragments before an error, but does not emit a successful finish or
`[DONE]` for invalid completed arguments. Length-limited or otherwise interrupted
output remains distinguishable from successful completion. Reasoning is not
included in argument or answer validation.

The scheduler's existing request-local matchers enforce masks on the first
token and each target verification row. dSpark proposals are checked against
forked grammar state. Retained raw logits remain independent of constraints,
allowing strictness changes on exact prompt hits without replaying that prompt.

## Qualification

The [summary](release-v1-tool-serving.json) and
[raw evidence](evidence/native-tool-serving.json.gz) record 16 native API tests,
including 44 injected JSON/character-SSE scenarios, and 101 accepted/rejected
sequence checks across 61 grammar groups, with mask/accept agreement at every
consumed token. The latter include the actual structural grammars
emitted by the API policy tests. The release binary builds successfully.

Target and dSpark each pass 14 paired live JSON/SSE tool cases. These cover
automatic, required and named selection, contradictory instructions, explicit
thinking disablement, combined response/tool schemas, typed and nested values,
references, padded Unicode, escaped parameter names, reserved delimiters and
parallel-call policy. All other cases use default high thinking. Target C16 and
dSpark C2 bursts preserve per-request schemas; cancellation and invalid-schema
recovery pass. Both modes also pass the existing 16-pair response constraint
suite. Within each mode, paired JSON/SSE arguments, reasoning and usage agree.
Tool arguments agree across modes for all 14 cases; reasoning is not always
identical across modes.

The separate cache check changes strictness without changing the rendered
prompt. Non-strict output uses `n=41`; strict output uses `n=42` with a full
301-token prompt hit. Subsequent tool-result continuations pass with partial
prefix reuse. This evidence does not establish an exact completed-turn hit for
those retokenized tool conversations.

Four alternating AB/BA pairs per workload compare ordinary, unconstrained text
with the preceding response-constraint serving binary on the same GPU. Arms and
modes run sequentially through the four shared Spark workers. The RTX power
limit is 400W and memory uses standard speed (13,365 MHz observed under load);
driver and hardware details are preserved in the evidence.

| Mode | Output limit | Counting decode change | Code decode change |
| --- | ---: | ---: | ---: |
| Target | 256 | −0.11% | +0.25% |
| dSpark | 256 | −0.56% | −0.09% |
| dSpark followup | 512 | +0.50% | +0.10% |

The initial dSpark counting dip prompted the longer followup. Results moved in
both directions, with identical text and token counts in every paired sample.
These focused checks show no material ordinary-decode regression; they do not
establish the full release performance matrix or constrained-tool throughput.

## Remaining release work

This serving integration does not close the complete tools release gate.
Root `patternProperties` and `propertyNames` are still explicitly unsupported
by the parameter converter. Grammar-level additional-key exclusions, broader
reserved-delimiter/string combinations and schema intersections still need
work; independent validation prevents a successful nonconforming completion
but does not replace generation support. Complete those cases and broader
high-thinking tool-eval/agentic qualification before the architectural
compressed FP4 migration and the final release performance campaign.
