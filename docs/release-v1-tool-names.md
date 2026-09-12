# Tool parameter names and Unicode grammar ranges

Root tool objects support `propertyNames` patterns, string length bounds, enums
and references, with typed additional arguments. Name patterns constrain decoded
strings. The converter emits canonical JSON escaping for attribute names, so
spaces, quotes, backslashes and controls retain their meaning. Regex repetitions
count decoded Unicode characters rather than escape bytes. Simple unanchored,
prefix and suffix patterns retain their search semantics.

Optional fixed properties whose names fail the name grammar are removed from
generation. Forbidden required names and impossible property counts produce
compilation errors. Required names supplied through `additionalProperties` are
materialized as required parameters. Name-rule caches remain separate from value
and nested-JSON caches.

## Unicode FSM corrections

Qualification exposed two defects in the pinned XGrammar FSM builder. The upper
tail of a three-byte UTF-8 range recursed with one continuation byte where two
were needed. Thus a range ending immediately before `台` incorrectly rejected
valid strings such as `北京`. Negated character classes converted exclusions to
eight-bit values and admitted excluded non-ASCII characters.

The generated project copy corrects the continuation-byte bound and computes
negation over Unicode codepoint intervals before UTF-8 conversion, excluding
surrogates. These corrections apply to response grammars as well as tool names.
The source verifier retains the pinned vendor tree; the generator checks its
replacement anchors. The vendored source is not edited.

## Qualification

The [summary](release-v1-tool-names.json) and
[raw evidence](evidence/native-tool-names.json.gz) contain 160 positive/negative
sequence checks across 75 groups using the official tokenizer, including the
existing API-generated tool grammars. Every consumed token checks agreement
between the vocabulary mask and matcher acceptance. Three unsatisfiable name
schemas fail compilation, followed by a successful compile on the same compiler.
The native build and existing grammar self-test pass.

Target and dSpark each pass eight paired JSON/SSE cases: name patterns, Unicode
length bounds, escaped enums, unanchored matches, fixed-property filtering,
references, required additional names and recovery. These use default high
thinking and compare exact expected argument objects. Both modes also pass the
existing 16-pair response constraint suite and ordinary tool control bursts
(C16 target, C2 dSpark), cancellation and recovery. This is correctness
qualification, not a new throughput measurement.

The evidence preserves the original failing Unicode checks and minimal before/
after range reproducers. An initial live unanchored-pattern request returned
`mid` when asked for `amidb`; both names satisfied that original schema. The
final generation test explicitly requires `amidb`, so it checks exact schema
enforcement. The initial result remains a model instruction-following miss,
separate from the corrected Unicode compiler defects.

## Scope still open

Root `patternProperties`, grammar-level duplicate/additional-name exclusions,
complex regex anchors and combined schema constraints still need work. Completed
argument validation remains independent of generation and rejects nonconforming
results. This change does not close the complete tools gate or replace the
broader high-thinking tool evaluation before compressed FP4 integration.
