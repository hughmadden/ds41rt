# Tool parameter-name exclusions

Additional tool arguments now exclude declared parameter names, including names
materialized from `required` under `propertyNames`. This prevents an additional
value schema from bypassing a declared value constraint. For example, with
`properties.a.const=1` and integer additional properties, the previous grammar
accepted `a=9`; the corrected grammar rejects it. The independent completion
validator remains in place.

The converter constructs bounded byte automata for regular name grammars,
subtracts the fixed-name language, then emits whole Unicode scalar transitions
back into EBNF. Rule occurrences receive independent continuations; non-tail
recursion and lookahead assertions are rejected by this helper. Construction
has explicit NFA, DFA and Unicode-expansion budgets. This helper also provides
intersection for the forthcoming overlapping `patternProperties` implementation.

The [evidence summary](release-v1-tool-exclusions.json) and
[raw results](evidence/native-tool-exclusions.json.gz) record:

- 135,945 comparisons of original, round-tripped, intersected and subtracted
  regular languages, including repetitions, negated classes, Chinese, emoji,
  NUL and UTF-8 width boundaries. Separate checks cover shared continuations
  and rejection of non-regular recursion.
- 181 positive/negative official-tokenizer sequences across 79 groups, with
  token-mask/accept agreement, plus three impossible-schema compilation errors
  and recovery. New cases cover ASCII, Unicode and escaped fixed-name
  exclusions, duplicate fixed names and a fully excluded additional-name set.
- The existing native XGrammar self-test passes. The baseline library fails
  the new `a=9` rejection case as expected.

An initial test put an additional key between fixed keys, outside the existing
ordered generation contract; the positive fixture was corrected to put it after
the fixed keys. An initial self-test invocation lacked its source fixture mount;
the corrected invocation passes. Both initial results are preserved.

These are grammar component checks. No live server library was replaced and no
new serving throughput result is claimed. Arbitrary dynamic-name duplicates,
overlapping pattern-value constraints and broader high-thinking tool evaluation
remain open. The release gate still requires those checks before FP4 migration.
