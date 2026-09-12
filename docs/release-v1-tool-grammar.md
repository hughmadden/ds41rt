# V4.1 tool grammar component

The native grammar adapter accepts an internal `ds41_tool_schema` structural
node and converts argument objects to V4.1 DSML parameters. This component is
not yet connected to native API tool requests and does not close the strict
tool release gate.

Outer strings use `string="true"` with their original content; numbers,
booleans, nulls, arrays and objects use `string="false"` with JSON content.
Nested values retain JSON syntax. Parameter names escape spaces because the
recipe parser uses a literal space to end the name attribute. Constant strings
containing a reserved DSML closing prefix use a JSON-encoded string with an
escaped `<`, preserving the decoded value without ending the parameter early.
Root constant objects preserve source property order. Reference and rule caches
distinguish root arguments, parameter values and nested JSON, including recursion.

The implementation reuses the pinned XGrammar schema parser and intermediate
representation. CMake generates a copy of its converter translation unit with
an explicit project factory extension; the vendored source and provenance lock
remain unchanged. Existing response-schema conversion keeps its original path.

The [component evidence](evidence/native-tool-grammar.json.gz) records 17 groups
using the official 129,280-token vocabulary. Positive and negative checks cover
typed scalars, required and optional arguments, nested arrays/objects, shared
and recursive references, mixed enums, padded Unicode strings, string bounds
and patterns, escaped names, reserved delimiters, root constants, empty objects
and typed/open additional arguments. Every consumed token checks agreement
between the vocabulary mask and matcher acceptance. The native release library
build and existing XGrammar self-test pass. These are CPU grammar checks, not
inference, tool-eval or performance qualification.

## Remaining work

- Implement root `patternProperties` and `propertyNames`; the component currently
  rejects them explicitly.
- Enforce duplicate-key rules and named-property exclusions for additional
  arguments, including alternate escaped spellings.
- Handle reserved DSML delimiters in constrained strings and independently
  validate completed arguments. The inherited schema converter does not enforce
  every combination of string pattern, format and length constraints.
- Wire tool selection, strict schemas, parallel-call limits and reasoning into
  native requests, then qualify JSON/SSE, cancellation, error isolation, exact
  cache hits and dSpark against live inference and high-thinking tool evals.

The architectural compressed FP4 migration follows this completed tool gate.
