# DeepSeek V4.1 Flash architecture audit

Status: initial source review; implementation and numerical qualification remain pending.

The authoritative sources are the [official checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/df42c109f1defefcbfcedbe7d905718a12266e40), its `config.json`, `inference/` reference, and `DeepSeek_V41_Tech_Report.pdf`; `ds41-reference-lock.json` records the reviewed file hashes.

## Execution contracts identified

| Component | Official contract | Required runtime change |
| --- | --- | --- |
| Backbone | 40 layers, width 5120, expert intermediate 2304 | Replace Pro dimensions and checkpoint schema |
| CED | 20 encoder and 20 decoder layers | Separate global cache construction from decoder SWA replay |
| Global KV sources | Layers 2, 8, 14, 20 | Request-owned shared caches with explicit source lifetimes |
| Compression | Layers 0–1 SWA only, 2–19 ratio 2, 20–39 ratio 1 | Replace C4/C128 assumptions |
| Index sources | Layers 2, 8, 14, 20, 24, 28, 32, 36 | Publish and reuse selections independently of KV ownership |
| Decoder candidates | Layer 20 selects 2048 blocks of 8 positions | Pin newest visible block and restrict later indexers to candidates |
| Attention | 64 heads, 512 dimensions, 64 RoPE dimensions, 128-token window, top-512 global positions | Respect causal group completion and shared-source positions |
| Cache formats | Window FP8; compressed FP4 with groups of 16 and E4M3 scales; indexer FP4 with groups of 32 and E8M0 scales | Separate quantization contracts rather than reuse one generic FP4 layout |
| Single-pass mHC | Four streams, previous sublayer's input-mixing coefficients, 20 Sinkhorn iterations | Carry coefficients across attention/FFN/layer boundaries and fuse the correct dependency |
| Backbone MoE | 384 routed experts, top-6, one shared expert, sqrtsoftplus scoring | Native FP4 expert AFD with text/image routing bias selection |
| Engram | Layers 1 and 14, orders 2/3/4, eight heads per order, 256 dimensions per fetched row | Deterministic hashes and asynchronous mapped table access |
| Engram gate | Per-stream normalized dot product, signed square root, sigmoid, shared value residual addition | Fused gate must preserve per-stream normalization and image masking |
| Vision | 32 layers, width 1024, 16 heads, patch size 14, bidirectional 2D RoPE | Native ViT plus 3×3 spatial folding and GELU aligner |
| dSpark | Three independent stages, 128 routed experts each, top-3, five proposal positions | Distinct RTX-resident weights using local expert kernels |
| dSpark conditioning | Means of incoming mHC streams at target layers 37/38/39 | Capture attention inputs rather than block outputs |
| dSpark heads | Shared token embedding/output head, rank-256 Markov head and confidence head | Implement proposal distribution, adaptive verification, and rollback |

## Reference limitations and production decisions to qualify

The reference runs all backbone layers during prefill and assumes either start-position-zero prefill or one-token decode in several cache producers, so arbitrary chunking and verification batches require new causal state handling.

The report's decoder bounded replay deliberately approximates SWA context; retain an exact execution baseline and measure the bounded mode explicitly rather than claiming numerical identity.

The reference's process-global shared-attention object is unsuitable for alternating concurrent requests; cache sources, selections, candidates, and speculative state need explicit request/batch ownership.

Engram token normalization must match the official tokenizer's 99092 compressed tokens, and hash multipliers use NumPy's layer-seeded generator, distinct prime ranges, and 64-bit arithmetic.

Image spans break n-gram history and receive no engram residual contribution; rejected speculative tokens must never contaminate committed history.

The reference materializes engram tables as parameters, whereas this engine must map official weights and scales and stage only requested rows with bounded prefetch buffers.

The reference generator does not invoke speculative decoding, so reference forward agreement alone cannot qualify dSpark acceptance, rollback, sampling, or concurrent serving.

## Initial environment evidence

On 2026-09-10 the coordinator exposes two RTX PRO 6000 Blackwell GPUs with 97887 MiB each; SSH, Docker, and GB10 GPUs respond on ostrich and dodo.

No access to emu or kiwi has been attempted, and no full-model execution or GPU numerical qualification has yet been performed.

## dSpark placement decision

Place all dSpark execution on the coordinator RTX, as requested, eliminating its three remote expert boundaries before target verification.

The routed expert payload lower bound is `3 stages × 128 experts × 3 projections × 5120 × 2304 × 0.5 bytes = 6.328125 GiB`, excluding scales, shared experts, attention, special heads, cache, workspace, and allocator overhead.

Shared token embedding and vocabulary head storage need not be duplicated on the same GPU, but memory planning must account for their execution workspaces and contention with target attention at concurrency 16.

No placement comparison gate is required; target verification still uses backbone AFD and proposal acceptance remains checkpoint-dependent.

## Native engram storage implementation

The official index and range-read shard headers identify `layers.1.engram.embed.{weight,scale}` in shard 47 and `layers.14.engram.embed.{weight,scale}` in shard 48, with no `model.` prefix; the reviewed headers are recorded in `ds41-engram-shard-headers.json`.

`EngramTable::from_catalog` validates the official row counts, FP8/UE8M0 representations, payload lengths, and relative shard paths before mapping both tensors read-only without prefaulting them.

`MappedRows` gathers into caller-provided buffers using checked wide offsets and performs page-budgeted deduplicated/coalesced advice, while `EngramPrefetcher` moves advice to a bounded I/O queue with request-owned cancellation tickets and nonblocking backpressure.

Advice is an OS prefetch hint and does not guarantee residency; cancellation skips queued work but does not undo page advice already issued, and the mapping contract requires immutable checkpoint files.

Loader tests include an official-size sparse shard with nonzero final rows beyond 98 GB and eager-load rejection, but decode/prefill/verification scheduling, device staging, and hardware performance qualification remain open.

## Engram addressing qualification

`EngramTokenMap` uses the upstream Rust tokenizer normalization sequence, including whitespace preservation, accent stripping, and raw forms for partial UTF-8 tokens; all 129280 token IDs match the official Python reference and collapse to 99092 compressed IDs.

`EngramHistory` retains only the preceding three compressed IDs or image barriers, uses the official layer-seeded multipliers and distinct prime ranges, and prepares hash batches without mutating committed state.

Prefix acceptance commits only accepted tokens and invalidates older batches, including zero-acceptance commits, while request identities reject batches from other histories.

`EngramTokenMap::prepare_batch` and `EngramPrefetcher::try_submit_batch` connect token IDs to mapped table page advice, omit image rows, check table identity, and bound batch storage; the serving scheduler still needs to own these objects and coordinate GPU staging.

Qualification command: `.venv/bin/python scripts/qualify-ds41-engram.py --reference-dir /tmp/ds41-reference` with Python 3.12 and PyTorch 2.13.0+cu130.

The pinned official `NgramHashState` agrees exactly across 128 interleaved batches on 16 request histories, covering image barriers, chunked input, and partial or zero acceptance; this proves host addressing equivalence rather than full serving concurrency or GPU execution.

The complete core and loader test suites pass 119 and 58 tests respectively, and the qualification tools create their transient inputs in automatically removed temporary directories.

## Engram CUDA primitives

The native library and Rust FFI expose allocation-free asynchronous engram row dequantization and fused gate/residual operations at the official dimensions.

The gate uses one block per token and residual stream, retains the residual in registers, reduces per-stream norms and the weighted dot product, applies the signed-root sigmoid, and rounds the final residual addition to BF16 with round-to-nearest-even.

The dequantizer consumes gathered 256-byte FP8 rows with eight UE8M0 scales, including subnormal scales and NaN encodings, and produces BF16 rows for the engram projection input.

`docs/ds41-engram-cuda-qualification.json` records commands, source hashes, GPU identities, library hash, and results for both RTX GPUs, including graph replay with mutated inputs and unchanged replay allocation counts.

Dequantization matches PyTorch exactly across the tested encoding cases, and the gate matches the pinned official forward within BF16 tolerance at 1, 16, 80, and 256 tokens, with maximum observed absolute difference 0.0078125.

A 104860-row GPU case validates element offsets beyond 2^31, both CPU native self-tests and all three CUDA native self-tests pass, and the compiler reports zero stack and local memory for both new kernels.

These results qualify the two CUDA primitives; engram projection GEMM, pinned host/device staging, prefetch deadlines, and request graph integration remain unfinished.

## Strict checkpoint configuration

`ds41rt-loader::OfficialV41Config` validates the official nested configuration against an embedded byte-identical copy of the pinned `config.json`, whose SHA256 is recorded in `ds41-reference-lock.json`.

The reader rejects missing or unknown fields and altered architecture/quantization values, preserves separate backbone and dSpark expert geometry, and exposes target and dSpark compression schedules separately; only the informational Transformers version may vary.

This new contract is not yet selected by the legacy `read_model_facts`/execution path, so passing its 60 loader tests does not mean the official model can already be loaded or served.

## Reference parameter inventory

`scripts/audit-ds41-reference-parameters.py` constructs the pinned reference on PyTorch's metadata device and checks all parameter names against the official index without allocating checkpoint weights.

The reference has 96,042 distinct parameters and the checkpoint has 96,085 tensors, with the only additional checkpoint entries being the 43 WO-A scale tensors that the reference converter dequantizes into BF16 weights.

Unshared dSpark reference parameters occupy 7.5434 GiB including 6.7236 GiB of routed experts, vision/aligner parameters occupy 0.9040 GiB, and mapped engram tables/scales occupy 188.8331 GiB; these are reference representations and exclude runtime caches, scratch, packing, and shared dSpark embedding/output weights.

The audit was run with the isolated reference environment using PyTorch 2.13.0+cu130, TileLang 0.1.8, and apache-tvm-ffi 0.1.6, because newer TVM-FFI versions 0.1.12/0.1.13 fail while importing this TileLang release.

This inventory is metadata evidence only and does not establish native checkpoint dtype/shape agreement, numerical correctness, or serving performance.

## Native storage contract and placement

The new `OfficialV41Catalog` validates all 96,085 native tensors from 116 explicit shape/dtype templates, checks their index/header correspondence and non-overlapping complete byte ranges, and reads headers without allocating weight payloads.

Header sources and pinned LFS identities are recorded in `ds41-native-catalog-qualification.json`; all 46 currently downloaded non-engram shards have cache filenames and lengths matching the pinned revision, and the local config/index hashes also match despite the local snapshot having a different revision label.

Native WO-A weights remain FP8 with independent 32x32 scales, the output/Markov/confidence heads are stored as BF16, vision norms are BF16, and ratio-2 compressor weight/gate projections are BF16 even where the reference converts them to FP32.

Native storage totals are 18,750,160,200 coordinator bytes including 7,932,874,632 dSpark bytes, 72,194,457,600 bytes per Spark, and 202,758,032,400 host-mapped engram bytes; runtime conversion, packing, caches, graph buffers, and scratch must be budgeted separately.

Backbone experts are split four ways along intermediate rows for W1/W3 and packed intermediate columns for W2, while every dSpark tensor remains on the coordinator RTX and engram table weights/scales are exclusively host-mapped.

The catalog exposes bounded caller-owned tensor staging, coalesced W2 column reads with caller-owned scratch, 64-bit file offsets, and direct engram mapping into the existing cancellable paired prefetcher.

Qualification passes 63 loader tests, the full 48-shard sparse-header fixture, six adversarial catalog rejection cases, and nonzero mapped first/last-row prefetch/gather checks at the full native engram dimensions; this does not qualify checkpoint payload integrity or model execution.

Run `cargo run --manifest-path rust/Cargo.toml -p ds41rt-loader --example v41_catalog -- SNAPSHOT` to inspect a complete official snapshot's native storage contract without loading its tensor payloads.

## Native expert arithmetic boundary

The official FP4 expert rounds FC1 projections to BF16 and applies routing weights to the clamped SwiGLU intermediate before BF16/FP8 quantization for FC2, so post-FC2 route weighting is numerically different.

The new b12x `silu_v41` contract implements this sequence and accumulates all local FC2 slices in FP32 before each expert's BF16 output boundary; `docs/ds41-expert-qualification.md` records four-device native/graph checks and the still-open TP-global reduction requirement.

The isolated TP-global expert boundary now has a native reducer and four-device graph qualification, but the inherited wire format uses ten-byte route entries with BF16-truncated weights and cannot carry FP32 partials, so model integration requires a wire-contract update before this path can serve V4.1.

Wire revision 3 now preserves FP32 gate weights in twelve-byte entries and provides FP32 output dtype 6, with Rust/native signature agreement and persistent-TCP qualification recorded in `docs/ds41-wire-qualification.md`; existing per-token collectives and model executors still need migration to per-route planes.

The native expert AOT export path now distinguishes logical Spark width 576 from b12x's prepared width 640, records planner-owned scratch layouts, and passes M16 C-entry arithmetic/graph checks in linked RTX and GB10 libraries; runtime weight budgets must include this padding and Rust execution integration remains open.

The checked Rust/C expert launch bridge and CUDA route reducer now cover native dSpark launch through final reduction, with an M16 numerical/graph fixture on RTX and exact standalone reduction on both RTX GPUs and ostrich; Python weight/scratch preparation and inherited transport/model execution remain to be replaced, as recorded in `ds41-native-route-reduction-qualification.md`.

Native expert scratch views and initialization now derive directly from the exported b12x planner and execute through Rust on RTX/ostrich, including Spark width-640 padding; native checkpoint weight preparation and scheduler-owned GPU allocations remain open in `ds41-native-expert-scratch-qualification.md`.

Native expert packing now preserves official FP4/E8M0 bytes through Rust/CUDA and M16 expert execution; prepared backbone weights require 80,216,064,000 bytes per Spark (8,021,606,400 bytes above checkpoint storage), plus reusable 4,700,160-byte per-expert input staging and separate runtime allocations, as recorded in `ds41-native-expert-packing-qualification.md`.

The official catalog now supplies aligned per-expert staging plans directly to native packing, with exact full-header-fixture-to-GPU checks for all four TP ranks and full dSpark on RTX; production layer ownership and execution orchestration remain open in `ds41-expert-staging-qualification.md`.

The daemon now owns four packed buffers per expert layer and releases bounded loading staging before returning, with complete synthetic RTX dSpark and ostrich TP-rank-3 layer loads and allocation-failure cleanup qualified in `ds41-owned-expert-qualification.md`; graph lifetime and serving integration remain open.

Owned native expert execution now borrows resident layers and manages stable per-wave buffers, streams and graphs, passing changed-input alternating M16 replay and smaller direct batches on RTX/ostrich; network TP and service scheduling remain open in `ds41-owned-execution-qualification.md`.
