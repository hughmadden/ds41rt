# Release semantic workload contract

The [self-contained corpus](../scripts/fixtures/release-semantic-corpus.json)
preserves the requested GLMRT prompts, output budgets, category weights and
structured-edit schema. Its metadata records the sibling repository revision
and SHA-256 hashes of both source scripts. The
[quality validator](../scripts/release_semantic_quality.py) preserves all three
original validation functions with identical Python ASTs.

The weighted cases are code, math, fable, hello, topic, natural JSON,
schema-constrained JSON and multilingual output. Each JSON case has weight 0.5;
the other six have weight 1.0. Counting, repetition, Orchid and rare-width syntax
diagnostics do not enter the weighted score. Weighted throughput is the sum of
weighted post-first-token counts divided by the sum of weighted decode times;
it is not an arithmetic average of the case throughput values.

Orchid retains the original 100-word repetition request, a 1,500-token output
budget and the original nonce-bearing prompt template. Its nonce must be
instantiated and recorded per sample. The original workload controls are
zero temperature and disabled thinking. The final tool-eval suite separately
uses high thinking, as requested.

These inputs and validators are preparation, not completed release measurements.
The serving runner must record the actual V4.1 tokenizer/model, request bodies,
cache state, outputs, timing source and all quality failures. API-observed decode
time includes streaming overhead and must not be labeled as the source engine's
internal GPU timing. No other engine's throughput or model-specific launch
settings are imported.

Verification: eight weighted cases sum to seven weight units; the schema and
prompts are extracted directly from the source declarations. Validator ASTs
match exactly. Positive/negative arithmetic and bare-versus-fenced strict JSON
checks confirm the preserved validator is callable and retains those rules.
