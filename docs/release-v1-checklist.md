# First release qualification

The current release objective is the user's September 12, 2026 release request.
This checklist tracks that scope; historical throughput targets in TO_SHIP_V1.md
are context, not substitutes for the measurements requested here. No release
gate below is satisfied merely by an older component test.

## Implementation and correctness gates

- [ ] Token radix, compression block ownership and bounded SWA replay follow
  the official model's cache semantics; verify cold, partial, exact and divergent
  prefix reuse against uncached execution, including eviction and concurrency.
- [x] Retain populated last-turn SWAs for fast exact agentic continuation,
  bounded and invalidated with radix branches.
- [x] Default to 24 exact retained turns (16 concurrent plus eight spare),
  with a configurable retention limit (user addition September 12). Audit
  prompt and completed-turn snapshots so the setting counts retained turns
  as intended, rather than silently halving capacity by keeping both frontiers.
  [Independent prompt/turn banks](release-v1-retained-turns.md) pass live 24-turn
  reuse, eviction and focused parity/concurrency/performance checks in both modes.
- [x] Measure reuse performance and expose one exact KV pool sizing option.
  [Native pool evidence](release-v1-pool.md) covers focused reuse/performance
  checks; the full retained-context release matrix remains open.
- [x] Run an initial tool-eval-bench and basic dsh agentic task after prefix
  correctness, before the full qualification campaign.
- [x] Repeat tool-eval with thinking enabled at high effort (user correction);
  retain earlier thinking-disabled runs as cache diagnostics.
- [x] Resolve Unicode correctness, distinguishing tokenization, UTF-8 streaming,
  prompt rendering and model instruction-following failures.
  [Unicode completion qualification](release-v1-unicode.md) fixes dropped terminal
  U+FFFD/incomplete UTF-8, verifies 157 official-tokenizer strings and 1,045 token
  cuts, and checks JSON/SSE in both modes. The historical arithmetic formatting
  failure remains labeled model behavior; default high thinking returns `36`.
- [x] Native vision accepts up to 16 images per prompt, with API validation,
  multi-image correctness and resource qualification.
  [Native preprocessing](release-v1-vision-preprocess.md) matches reference grids
  and complete patches across 21 format cases and handles sixteen expanded image
  spans. The [native encoder/aligner component](release-v1-vision-encoder.md)
  now has GPU operation, full-vector and precision-control evidence.
  [Image cache identity and ownership](release-v1-vision-cache-identity.md) have
  focused radix tests. [Feature replacement and request masks](release-v1-vision-features.md)
  pass CUDA and request-plumbing checks. [Native serving qualification](release-v1-vision-serving.md)
  covers image answers, changed/reordered images, partial replay, exact hits,
  maximum spans, cancellation/recovery, C16 target and C2 dSpark admission, and
  focused text regression checks. Longer target/dSpark descriptions are
  semantically consistent, not necessarily token-identical. The broader
  long-context/pool-pressure prefix gate and release performance suite remain open.
- [ ] Qualify tools, structured outputs, streaming, cancellation and recovery.
  Complete native XGrammar enforcement before the compressed-cache migration.
  [Retained-logit preparation](release-v1-constraints.md) enables first-token
  reselection under a changed grammar on exact hits. [Response constraint
  enforcement](release-v1-response-constraints.md) now covers JSON schemas,
  thinking, SSE and dSpark. [Native tool serving](release-v1-tool-serving.md)
  now enforces strict arguments, selection and parallel-call policy, including
  high thinking and independent JSON/SSE completion validation. Remaining
  parameter-schema generation cases and broader tool-eval qualification keep
  this gate open. [Additional-name exclusions](release-v1-tool-exclusions.md)
  prevent additional arguments from bypassing declared value constraints.
  [Pattern intersections](release-v1-tool-patterns.md) apply all matching regular
  value constraints and preserve native admission errors. Broader high-thinking
  tool evaluation and remaining unsupported schema combinations keep this gate open.
- [ ] Restore architectural compressed KV to FP4 E2M1 with group-16 E4M3
  scales; retain FP8 SWA and the independent FP4 index format. The previous
  FP8-only interpretation was incorrect (user clarification September 12).
  Optimize compressed packing, paged attention reads/dequantization and their
  interaction with prefill, decode and dSpark; update allocations, exact-pool
  sizing, retained snapshots and cache-format identity. Compare with the FP8
  baseline under matched hardware/power/clocks, contexts and concurrency;
  match or improve runtime performance, recording all regressions and tradeoffs.
  After performance qualification, run long-context needle checks through 1M
  and thinking-enabled high-effort tool-call evaluation, plus cold/partial/exact/
  divergent prefix and target/dSpark correctness. Final release measurements
  must use the corrected cache. [Packing primitive evidence](release-v1-compressed-kv.md)
  is preparation only; serving integration and these gates remain open.
  Track this work in order:

  1. Integrate architectural FP4 throughout compressed-cache serving and retention.
  2. Optimize and qualify prefill, decode and dSpark against the matched FP8
     baseline; reduced cache bytes alone do not satisfy the performance gate.
  3. Once performance matches or improves, qualify needle retrieval and
     high-thinking tool calls, including agentic cache reuse, before release.

- [x] Verify official maximum context/output limits. Default to those limits;
  clamp each output allowance to remaining context and the model output cap.
  Keep configurable smaller development storage reservations for side-by-side APIs.
  [Native defaults and output clamping](release-v1-model-limits.md) now have
  focused API/CLI and live target/dSpark verification; full-context qualification
  remains part of the release suite.
- [x] Reserve 16 maximum-context requests plus eight additional 1M-context KV
  equivalents; support total memory percentage and MB/GB reservation options.
  [Pool allocation and high-address GPU checks](release-v1-pool.md) pass.
  Configured-pool exhaustion/admission isolation and launcher wiring still
  belong to the remaining API/run-script qualification.
- [ ] Measure and improve startup reads/transforms/exchange/capture. Aim for
  NVMe line rate (coordinator about 14 GB/s, Sparks about 6 GB/s) and roughly
  60–90 seconds load time without sacrificing runtime performance.
- [ ] Qualify normal build/run scripts and optimized backend selection on all
  five hosts from reproducible source and dependency pins.
  Expose concurrency and KV pool parameters through the standard launch script;
  sixteen active requests and eight retained-context equivalents are defaults,
  not mandatory settings (user clarification September 12).
  Preserve port 8000 for the standard `run.sh` launch; explicit alternate
  configurations may override `ADDR` (user addition September 12).

## Release performance suite

Every result must identify source and artifact hashes, model revision, hardware,
launch settings, sampling, input data, cache state, concurrency, warmup, all
samples and errors. State the RTX power limit and standard memory speed up
front in the README results; capture driver version, enforced power limit and
loaded memory clocks per campaign. The user reports a September 11 driver update
reset power caps; do not attribute cross-campaign differences solely to code.
Preserve raw machine-readable output alongside reports.
Run the shared four-Spark target/dSpark workloads sequentially to avoid contention.

- [ ] Reuse the located eight content types and Orchid from ../glmrt-release (sources below).
- [ ] Headline maximum prefill, low-entropy decode and weighted eight-type
  decode throughput; KV size and total GPU RAM requirement.
- [ ] Memory accounting by arena, tensor, KV cache, graphs and other allocations.
- [ ] Eight content types plus low-entropy decode, target-only and dSpark.
- [ ] Prefill matrix: base contexts 0, 32K, 64K, 128K, 256K crossed with
  added 1K, 2K, 4K, 8K, 16K, 32K tokens.
- [ ] dSpark decode on retained prefixes for all eight types at base contexts
  0, 32K, 64K, 128K, 256K; prove cache reuse in measurement evidence.
- [ ] Needle retrieval through 1M tokens.
- [ ] Concurrency scaling through C16, reporting aggregate tokens per second.
- [ ] Three tool-eval-bench hard-mode runs at the measured best concurrency,
  with thinking enabled at high effort and sufficient timeout; points split
  into basic, hard and total scores. The user reduced the original five-run
  requirement to three on September 12, 2026.
- [ ] Startup time, including phase-level I/O and graph preparation evidence.
- [ ] dsh generates a single-file WebGL Frogger game; retain execution metrics
  and publish a playable link to the actual generated artifact.

## Documentation and publication

- [ ] README: headline numbers, short intro, Docker and source getting started,
  host configuration, options, performance report, engineering report, thanks.
- [ ] Place an execution-path SVG immediately below the README short intro,
  using `../glmrt-release/docs/balanced-path-execution.svg` as the visual reference
  and showing the final qualified native serving design (user addition September 12).
- [ ] Concise engineering report covering the final design: kernels, scheduling,
  memory/storage, transport, optimizations, prefix caching, vision and API.
- [ ] Complete performance report covering every suite item above.
- [x] Record dev/main/release branch practice in the operation manual.
- [ ] Validate and publish final amd64 and arm64 containers to GitHub; record
  digests and verify the published artifacts match the qualified pair.
- [ ] After documentation and image gates, advance main to qualified dev and
  create release/vX for the chosen numbered release.
- [ ] Publish a formal first-release package on GitHub with qualified artifacts
  and container references. Release notes: a short high-level overview and
  bulleted key features, linking detailed reports (user addition September 12).
- [ ] Tell the user to make the images public manually after all other work.

## Current evidence and next action

The user set the next work order on September 12: finish and push the 24-turn
retention qualification, then native vision through sixteen images together
with Unicode and tool/schema correctness. Standard `run.sh` and deployment
qualification follow those correctness gates, before the final performance
suite, reports, containers and GitHub release. The latest user ordering puts
the optimized architectural FP4 compressed-cache migration immediately after
XGrammar/tool-schema correctness and before launcher/final release qualification.

Release work began on branch dev from main. Existing user edits to .gitignore
and run-agent.sh are preserved. The two development APIs were observed running
on September 12. Docker inspection confirms both run `serve-native`, which enters
v41_native_serve.rs and its scheduler.rs. Those original artifacts report
zero cache-hit tokens and release all request state on completion. The path has four
source pools (ratio-two layers 2/8/14 and ratio-one layer 20), forty SWA rings,
and CED decoder replay bounded to 128 tokens. The candidate now adds a native prefix index,
shared source pages, retained SWA/compressor/history state and admission wiring.

Correction to the initial source audit: the token radix and dSpark tail cache
in commands/real_full belong to the older execution path. Their C4/C128 reuse
guard does not explain native serving behavior. Uncommitted changes to that
inactive path were discarded. Native release implementation must be verified
through the actual serve-native dispatch and APIs.

[Native source prefix ownership](release-v1-native-prefix.md) implements shared
KV/index page retention and copy-on-write. Native radix admission and completed
state scheduling are now wired into serve-native. [Admission qualification](release-v1-native-admission.md)
covers complete hits, retained turns, C16 cancellation/replacement, small-pool
pressure and controlled short-context decode. [Partial-prefix replay](release-v1-partial-prefix.md)
now passes focused GPU, API and C16 checks. [Large exact-resume suffixes](release-v1-exact-suffix.md)
use encoder continuation and bounded decoder prefill. [Pool sizing](release-v1-pool.md)
and [24 completed-turn retention](release-v1-retained-turns.md) now have focused
qualification. Constrained-pool admission isolation and the full release suite
remain open.
[Retained-state component evidence](release-v1-retained-state.json) remains
separate from these API results.

[Initial agentic qualification](release-v1-initial-agentic.md) records 123/138
basic and 32/38 hard points (155/176 total) from one C16 tool-eval-bench run.
A real dsh coding task passed independent file/test checks and reused complete
committed turns across five tool continuations. This satisfies the initial
agentic check, not the three final release runs or arbitrary partial-prefix reuse.

The [thinking-enabled high-effort rerun](release-v1-thinking.md) records 120/138
basic and 35/38 hard points (155/176 total). It preserves the forced-tool API
rejection and other failures. Final three-run reporting uses thinking enabled at
high effort; the earlier disabled-thinking scores remain cache diagnostics.

[Initial paired quality evidence](release-v1-initial-quality.json) records eight
cases per mode against the existing development artifacts. Both modes pass five
of six objective cases. The Chinese arithmetic case computes 36 but includes
working despite an answer-only instruction, so it fails the strict objective.
Traditional Chinese translation renders correctly in both arms. This evidence
does not establish an encoding bug or comprehensive Unicode correctness.
The open-ended English explanations differ across modes. The script exits 1
because its strict combined gate is not satisfied; do not count this as a pass.

The located release corpus is `../glmrt-release/python/tools/bench_real_full_mtp_acceptance.py`:
`WEIGHTED_CASE_IDS` selects code, math, fable, hello, topic, structured-json,
structured-json-schema and multilingual. The two JSON cases each have weight
0.5; all six other cases have weight 1.0. Preserve its actual prompts, token
budgets and schema. Counting/repetition and syntax diagnostics are excluded from
the weighted score. `../glmrt-release/python/tools/bench_real_full_repeat_decode.py`
provides Orchid. Port the corpus and measurement contract, not the other engine's
numbers or model-specific launch assumptions.
