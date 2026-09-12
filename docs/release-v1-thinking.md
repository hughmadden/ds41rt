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

The live default check uses the artifact identified in the
[exact-suffix manifest](release-v1-exact-suffix.json). Its already-enabled default
matches the now-explicit setting. Final output-limit policy and comprehensive
tool/structured-output qualification remain separate release gates.
