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
| dSpark | Three independent stages, 128 routed experts each, top-3, five proposal positions | Distinct weights with shared expert kernels and transport |
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

## dSpark placement candidate

Evaluate placing all dSpark execution on the coordinator RTX to eliminate its three remote expert boundaries before target verification.

The routed expert payload lower bound is `3 stages × 128 experts × 3 projections × 5120 × 2304 × 0.5 bytes = 6.328125 GiB`, excluding scales, shared experts, attention, special heads, cache, workspace, and allocator overhead.

Shared token embedding and vocabulary head storage need not be duplicated on the same GPU, but memory planning must account for their execution workspaces and contention with target attention at concurrency 16.

Keep the remote expert placement available until measured memory and latency establish the preferred policy; target verification still uses backbone AFD and proposal acceptance remains checkpoint-dependent.
