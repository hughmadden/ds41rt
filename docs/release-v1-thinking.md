# Native reasoning defaults

An omitted thinking setting enables reasoning at high effort. The pinned
`deepseek-recipe` adapter already provides this default; native serving now
selects it explicitly and has a regression test for the request contract.
The V4.1 encoder renders high as effort 75 on its 1–100 scale.

An explicit `reasoning_effort` selects the adapter's effort level: low maps to
50, high to 75 and max to 100. `reasoning_effort: "none"` disables thinking.
An explicit `thinking.type` takes precedence over the effort's enable/disable
choice; `"thinking":{"type":"disabled"}` disables reasoning even if an effort
was also supplied. Enabled thinking without an effort uses high.

## Verification

Native API tests verify the rendered effort and thinking prefix for omitted
settings, explicit enablement, low/high/max, none and explicit disablement with
max effort. They also check that reasoning and answer text reach the separate
response fields. Existing streaming, ordinary-response and worker-failure tests
pass alongside these checks.

A live dSpark check asks the same arithmetic question with omitted settings and
explicit high effort. Both produce the same reasoning and answer; the explicit
high request reuses all 44 prompt tokens. Explicit disablement emits no reasoning
and uses an 18-token prompt. This verifies the running default independently of
the API's mocked backend test.
The [default/override evidence](release-v1-thinking-defaults.json) preserves the
test output, source hash and complete live requests and responses.

Tool-eval qualification uses thinking enabled and high effort. Earlier runs with
`--no-think` and `thinking.type: disabled` remain labeled cache diagnostics and
do not establish the required thinking-mode result. The pinned adapter rejects
required/named tool choices in thinking mode; qualification must preserve and
report such failures rather than silently disable thinking.

## Initial high-effort tool-eval result

The corrected run completes all 88 cases at C16 with thinking enabled, high
effort, temperature zero, a 900-second timeout and twelve maximum turns:
**120/138 basic + 35/38 hard = 155/176 points**. It records 70 passes, 15 partials
and three failures. Five of six structured-output cases pass (10/12 points).

TC-43 calls search with an empty required query. TC-45 receives the adapter's
HTTP 400 for forced tool choice with thinking enabled; the benchmark describes
this as no tool calls, but the raw trace identifies an API rejection. TC-68 makes
an unnecessary file-search call and adds prose around the requested JSON.
These remain explicit tool/schema qualification issues. The previous
thinking-disabled results are separate diagnostics, not substitutes for this run.

This initial rerun uses an explicit 4,096-token development output cap and does
not count toward the five final release runs. The final campaign follows the
qualified output policy after the remaining API and memory work.
[Result metadata](release-v1-thinking-agentic.json) and
[compressed raw evidence](evidence/native-thinking-high-agentic.json.gz) preserve
the exact command, all scenarios, complete tool traces and the benchmark report.

The reproducible runner defaults to thinking enabled at high effort. For the
final qualified endpoint, use five sequential runs with its measured best
concurrency and omit a development output override:

```bash
python3 scripts/qualify-ds41-tool-eval.py \
  --base-url http://127.0.0.1:8000 --output-dir /tmp/ds41-release-tool-eval \
  --reference-date 2026-09-12 --parallel 16 --runs 5
```

Each run gets a new directory, and the runner exports basic/hard/total points and
raw traces without dropping failures. `--max-tokens` is an optional explicit
development override; `--collect-only` exports an already completed run. Its
collector was exercised against the initial high-effort run.

The live default check uses the artifact identified in the
[exact-suffix manifest](release-v1-exact-suffix.json). Its already-enabled default
matches the now-explicit setting. Final output-limit policy and comprehensive
tool/structured-output qualification remain separate release gates.
