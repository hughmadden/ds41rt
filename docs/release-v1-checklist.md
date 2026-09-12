# First release qualification

The current release objective is the user's September 12, 2026 release request.
This checklist tracks that scope; historical throughput targets in TO_SHIP_V1.md
are context, not substitutes for the measurements requested here. No release
gate below is satisfied merely by an older component test.

## Implementation and correctness gates

- [ ] Token radix, compression block ownership and bounded SWA replay follow
  the official model's cache semantics; verify cold, partial, exact and divergent
  prefix reuse against uncached execution, including eviction and concurrency.
- [ ] Retain populated last-turn SWAs for fast exact agentic continuation,
  bounded by concurrent request capacity and invalidated with radix branches.
- [ ] Measure reuse performance and expose one exact KV pool sizing option.
- [ ] Run an initial tool-eval-bench and basic dsh agentic task after prefix
  correctness, before the full qualification campaign.
- [ ] Resolve Unicode correctness, distinguishing tokenization, UTF-8 streaming,
  prompt rendering and model instruction-following failures.
- [ ] Native vision accepts up to 16 images per prompt, with API validation,
  multi-image correctness and resource qualification.
- [ ] Qualify tools, structured outputs, streaming, cancellation and recovery.
- [ ] Verify official maximum context/output limits. Default to those limits;
  clamp each output allowance to remaining context and the model output cap.
  Keep configurable smaller development storage reservations for side-by-side APIs.
- [ ] Reserve 16 maximum-context requests plus eight additional 1M-context KV
  equivalents; support total memory percentage and MB/GB reservation options.
- [ ] Measure and improve startup reads/transforms/exchange/capture. Aim for
  NVMe line rate (coordinator about 14 GB/s, Sparks about 6 GB/s) and roughly
  60–90 seconds load time without sacrificing runtime performance.
- [ ] Qualify normal build/run scripts and optimized backend selection on all
  five hosts from reproducible source and dependency pins.

## Release performance suite

Every result must identify source and artifact hashes, model revision, hardware,
launch settings, sampling, input data, cache state, concurrency, warmup, all
samples and errors. Preserve raw machine-readable output alongside reports.
Run the shared four-Spark target/dSpark workloads sequentially to avoid contention.

- [ ] Locate and reuse the eight content types and Orchid from ../glmrt-release.
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
- [ ] Five tool-eval-bench hard-mode runs at the measured best concurrency,
  with sufficient timeout; points split into basic, hard and total scores.
- [ ] Startup time, including phase-level I/O and graph preparation evidence.
- [ ] dsh generates a single-file WebGL Frogger game; retain execution metrics
  and publish a playable link to the actual generated artifact.

## Documentation and publication

- [ ] README: headline numbers, short intro, Docker and source getting started,
  host configuration, options, performance report, engineering report, thanks.
- [ ] Concise engineering report covering the final design: kernels, scheduling,
  memory/storage, transport, optimizations, prefix caching, vision and API.
- [ ] Complete performance report covering every suite item above.
- [x] Record dev/main/release branch practice in the operation manual.
- [ ] Validate and publish final amd64 and arm64 containers to GitHub; record
  digests and verify the published artifacts match the qualified pair.
- [ ] After documentation and image gates, advance main to qualified dev and
  create release/vX for the chosen numbered release.
- [ ] Tell the user to make the images public manually after all other work.

## Current evidence and next action

Release work began on branch dev from main. Existing user edits to .gitignore
and run-agent.sh are preserved. The two development APIs were observed running
on September 12. Docker inspection confirms both run `serve-native`, which enters
v41_native_serve.rs and its scheduler.rs. The native scheduler currently reports
zero cache-hit tokens and releases all request state on completion. It has four
source pools (ratio-two layers 2/8/14 and ratio-one layer 20), forty SWA rings,
and CED decoder replay bounded to 128 tokens. It needs a native prefix index,
shared source pages, retained SWA/compressor/history state and admission wiring.

Correction to the initial source audit: the token radix and dSpark tail cache
in commands/real_full belong to the older execution path. Their C4/C128 reuse
guard does not explain native serving behavior. Uncommitted changes to that
inactive path were discarded. Native release implementation must be verified
through the actual serve-native dispatch and APIs.

[Native source prefix ownership](release-v1-native-prefix.md) implements shared
KV/index page retention and copy-on-write. Native radix admission, full retained
state scheduling and end-to-end performance qualification remain open. The
native owners now support GPU retention of SWA/compressor/dSpark state and
independent Engram history restoration; [component evidence](release-v1-retained-state.json)
does not yet establish API prefix reuse.

[Initial paired quality evidence](release-v1-initial-quality.json) records eight
cases per mode against the existing development artifacts. Both modes pass five
of six objective cases. The Chinese arithmetic case computes 36 but includes
working despite an answer-only instruction, so it fails the strict objective.
Traditional Chinese translation renders correctly in both arms. This evidence
does not establish an encoding bug or comprehensive Unicode correctness.
The open-ended English explanations differ across modes. The script exits 1
because its strict combined gate is not satisfied; do not count this as a pass.
