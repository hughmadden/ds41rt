# DS41RT architecture

This document describes the V4.1 target architecture; implementation and qualification
status is tracked in PRE_MODEL_PLAN.md rather than inferred from these contracts.

## Ownership

| Component | Owner |
| --- | --- |
| API, tokenizer, admission, scheduling, sampling, request history | RTX coordinator |
| Text embeddings, single-pass mHC, normalization, output head | RTX coordinator |
| CED encoder/decoder attention, compression, indexers, cache | RTX coordinator |
| Native vision encoder, spatial aligner, image embeddings | RTX coordinator |
| Engram mapped weights/scales and background page prefetch | Host storage and I/O workers |
| Gathered engram dequantization, projection, gate/residual update | RTX coordinator |
| Backbone routers and shared experts | RTX coordinator |
| Backbone routed experts | Four Spark intermediate-dimension TP ranks |
| Entire three-stage dSpark drafter, including routed experts | RTX coordinator |

## Backbone and cache

The official backbone has 40 layers and width 5120, arranged as a 20-layer causal
encoder followed by a 20-layer decoder; routed experts use intermediate width 2304,
384 experts per layer, and top-6 routing.

Every layer has a 128-token local window; layers 2–19 add ratio-2 compressed global
attention, and layers 20–39 share ratio-1 global KV produced at the encoder/decoder
boundary.
Global KV source layers are 2, 8, 14, and 20, and index-selection sources are
2, 8, 14, 20, 24, 28, 32, and 36.
The first decoder indexer selects candidate blocks for later decoder indexers.

Window KV, compressed KV, and indexer keys have distinct native quantization
contracts; their layouts and scales must not be treated interchangeably.
Each request owns its cache references, source selections, candidate blocks,
compression tail, and speculative transaction state.

Exact prefill establishes the baseline for CED decoder SWA replay, while bounded
replay is an explicitly approximate execution policy requiring separate qualification.

## Engram and speculative execution

Engram modules at layers 1 and 14 hash 2-, 3-, and 4-grams with eight heads per order.
Normalized token IDs determine table addresses before hidden-state computation,
allowing both modules' page prefetch to start as soon as the input IDs are known.
Image spans break n-gram history and receive no engram residual contribution.

Mapped table pages are advised by bounded background workers and gathered into
bounded staging buffers, then dequantized and projected on the GPU.
Prefill and verification batches must preserve row order, image masks, and request
identity; rejecting speculative tokens must not advance committed history.

The dSpark drafter has three stages with 128 routed experts and top-3 routing per
stage, uses the backbone's embedding and output head, and conditions on incoming
residual-stream means at target layers 37, 38, and 39.
It runs entirely on the RTX, while target verification still traverses backbone AFD.

## Serving and qualification

The target topology is four expert TP ranks and one coordinator, with 16-request
admission and alternating execution waves around remote expert boundaries.
Production graph shapes and workspaces must be prepared before readiness and
reused without request-time capture or allocation.
Readiness must verify checkpoint/dependency identity, weight residency, mapped
table access, transport, numerical startup probes, and prepared graph shapes.

The official checkpoint is the only release weight source; EXL3, GPTQ, alternate
model variants, and old Pro optimization settings are being removed.
See docs/ds41-architecture-audit.md for pinned source evidence and outstanding work.
