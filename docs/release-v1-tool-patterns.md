# Overlapping tool parameter patterns

Root tool arguments support `patternProperties` alongside fixed properties,
`propertyNames`, required names and additional properties. Names are partitioned
into disjoint regular languages according to which patterns match. Every
matching value schema applies to that region. Only unmatched names use the
additional-property schema, and dynamic regions exclude fixed and materialized
required names.

Value intersections support regular grammar fragments, including qualified
numeric bounds, enums, strings, references and structured object/array cases.
Unconstrained schemas act as identities. Other intersections use a bounded
automaton product; non-regular recursion or lookahead fails compilation rather
than relaxing the schema. This also replaces the upstream unconstrained fallback
for multi-schema `allOf` in the V4.1 converter. It does not establish complete
JSON Schema support.

Regions with incompatible value constraints are removed before parameter-name
generation. Impossible optional fixed properties are removed; an impossible
required property produces a compilation error. Fixed value rules are cached
after this check. Pattern partitioning has a 256-region limit in addition to the
regular helper's state and expansion budgets.

Regular string intersections include both raw strings and canonical JSON-encoded
strings, preserving string type for reserved DSML delimiter content. Non-string
branches in `propertyNames` type arrays cannot emit unquoted attribute names.

The live tests exposed an admission error-reporting defect: an impossible
required overlap correctly failed native compilation, but protocol EOF handling
replaced its cause with a missing-tool-call validation error and HTTP 500.
The API now awaits the first backend event before returning JSON or SSE headers.
Typed schema compilation failures return HTTP 400 with the actual cause;
worker failures remain HTTP 500. Failures after admission also retain their
cause and cannot become successful completion. This adds no extra per-chunk
locking to ordinary responses.

## Qualification

The component suite passes 247 positive/negative official-tokenizer sequences
across 93 groups, with per-token mask/accept agreement. Four impossible schemas
fail compilation and the compiler recovers. Tests cover disjoint and overlapping
patterns, weaker additional schemas, fixed-property intersections, required
undeclared names, Unicode names, string lengths, reserved delimiters, references,
object/array values and incompatible intersections. The regular helper passes
135,945 language comparisons, and the native XGrammar self-test passes.
Eighteen native API unit tests pass, including injected admission errors in both
response modes and a late worker failure under required-tool validation.

Both target-only and dSpark pass eight paired JSON/SSE cases with default high
thinking, independent JSON Schema validation and exact expected arguments.
The SSE repetition proves a complete prompt-cache hit. Concurrent pattern
schemas pass at C16 target and C2 dSpark. Cancellation, invalid-schema HTTP 400,
impossible-overlap HTTP 400 in both response modes and recovery pass.

The [summary](release-v1-tool-patterns.json) and
[raw evidence](evidence/native-tool-patterns.json.gz) retain source/artifact
hashes, launch settings, all samples and the initial admission-error failure.
Four alternating AB/BA pairs per workload compare ordinary counting and code
decode against the previous serving build, at 256 output tokens and C1. All
text and token counts match. Median paired throughput changes are +0.15% and
−0.08% for target, and +0.82% and +1.13% for dSpark. These focused measurements
show no material ordinary-decode regression; they do not replace the release
matrix or measure pattern-schema compilation throughput. Both RTX GPUs use
400 W limits and standard loaded memory clocks of 13,365 MHz; modes run serially.

## Remaining release work

Broader high-thinking tool-eval qualification remains required before compressed
FP4 migration. Complex regex anchors, some schema combinations and arbitrary
dynamic-name duplicate prevention remain outside this grammar qualification;
independent completed-argument validation remains enabled. The full release
performance suite and its three final tool-eval runs still follow FP4 migration.
