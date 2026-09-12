# High-thinking agentic qualification after native grammar work

One complete C16 tool-eval-bench run scores **124/138 basic + 35/38 hard =
159/176 points**: 72 passes, 15 partials and one failure. Thinking is enabled at
high effort, temperature is zero, timeout is 900 seconds and the turn limit is
twelve. No output-token override is passed. This development server reserves a
32,768-token context, so this is not a full-context release qualification or one
of the three final runs after FP4 migration.

The [summary](release-v1-tools-agentic-high.json) and
[raw evidence](evidence/native-tools-agentic-high.json.gz) preserve the command,
all 88 scenarios and traces, partial scores, server log, benchmark source hashes
and serving artifact hashes. The serving build is the one qualified in
[pattern and admission-error testing](release-v1-tool-patterns.md).

## Trace review

- TC-43 calls `web_search {}` in response to a request to search without a query.
  The mock returns a missing-parameter error; the model then explains the error.
  The evaluator labels missing and empty queries alike as an “empty query.”
  The installed benchmark's tool definition omits `strict`, and its adapter
  forwards that definition unchanged. Native non-strict tools do not enforce
  parameter schemas, so this remains a model-level failure under that API
  contract. It is not evidence of a strict-schema enforcement failure.
- TC-66 returns valid final JSON for the nested contact schema, but first makes
  an unrelated `run_code` call. That extra call causes its partial score.
- TC-45 now passes forced tool choice with thinking enabled. TC-68 now passes
  schema-violation resistance. Both failed in the earlier high-thinking run.
- TC-62 completes its research chain in 326 seconds. The server log records
  269 benchmark admissions, 264 with prefix hits. These include partial hits;
  they do not establish universal exact completed-turn reuse.

No native admission failure appears in the captured benchmark interval. All
scenarios complete, and trace review finds no HTTP/native-error marker. Mock
tool errors are retained as part of the benchmark scenarios. Structured-output
cases score 11/12; TC-66's unrelated tool call accounts for the lost point.

The earlier run scored 120 basic and 35 hard, with an explicit 4,096-token output
cap. This run scores four more basic points, but different output settings and
one sample do not establish a general quality improvement. All fifteen partials
remain visible in the summary; no failures were discarded or rescored.

The next implementation phase is architectural compressed FP4 integration and
optimization against the preserved FP8 baseline, followed by needle and
high-thinking tool checks. Remaining schema limitations, full-context/cache
pressure checks and final release qualification remain open.
