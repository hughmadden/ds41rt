# DS41RT engineering report

DS41RT is a native inference engine for the official DeepSeek V4.1 Flash checkpoint. One RTX PRO 6000 Blackwell coordinator owns the model state and request lifecycle; four DGX Spark systems execute tensor-parallel slices of the backbone routed experts. This report describes the selected release design. Numerical and serving evidence is linked from the [release checklist](release-v1-checklist.md).

## Execution topology

The RTX owns the API, tokenizer, admission, scheduling, embeddings, mHC residual transforms, all attention, Engram lookup and projection, backbone routers, shared experts, vision, dSpark, the vocabulary head, sampling, and every cache transaction. This keeps causal state under one owner and makes accepted-prefix publication atomic across the components that advance a request.

The backbone has 40 layers at width 5,120: a 20-layer causal encoder followed by a 20-layer decoder. Each layer has a 128-token local window. Layers 2–19 also attend to ratio-two compressed global sources; layers 20–39 share a ratio-one source created at the encoder/decoder boundary. Global KV producers are layers 2, 8, 14, and 20. Index selections are produced at layers 2, 8, 14, 20, 24, 28, 32, and 36.

Backbone routed experts use 384 experts and top-6 routing. Each of the four Sparks owns one intermediate-dimension tensor-parallel rank for every routed expert. The coordinator sends the same canonical rows, expert IDs, and FP32 route weights to all ranks, gathers the four partial planes in rank order, reduces them, combines the local shared expert, and advances the residual. The dSpark drafter stays entirely on the RTX and does not traverse this boundary.

Two independent target execution lanes overlap local work with the remote expert boundary. Admission is shared across the lanes and accepts up to sixteen active requests. Requests keep generation-checked leases and versions; stale batches, duplicate identities, changed cache bindings, and mismatched transport responses fail before publication.

## Native kernels and graph reuse

The CUDA layer is split into owned operations for mHC boundaries, projections, rotary frequencies, cache packing, index scoring and selection, sparse attention, routing, shared experts, expert reduction, dSpark, vision, and terminal logits. Rust owners reserve stable buffers and validate device, extent, aliasing, request generation, and phase before launch. Results borrow the state they depend on until work drains, preventing a cache commit or slot reuse from racing an in-flight kernel.

Projection paths use the checkpoint's native weight representations and SparkInfer-selected kernels. Common row capacities are prepared for decode and prefill; the live row count remains runtime data. Dedicated small-row paths avoid paying the grouped prefill setup cost for single-token decode. Larger work uses grouped kernels and bounded 2,048-token default prefill waves.

Sparse attention stages selected KV into padded shared layouts, keeps probabilities in registers until accumulator rescaling, and passes cache descriptors as grid-constant input. The selected implementation eliminates local-memory traffic present in the earlier layout while preserving the qualified arithmetic. FP4 cache values convert directly to BF16 in the consumer; no full dequantized-history buffer is materialized.

CUDA graphs bind stable owners and are reused only while their pointer, capacity, layer, request partition, and cache identities match. New shapes recapture after draining the old work. Release startup loads all resident owners but leaves request-shape graph capture lazy, which avoids a large readiness penalty; the first use pays the capture cost and later requests replay it. A full C16 warmup adds 74 MiB of measured graph and runtime residency.

## Cache formats and ownership

The three persistent attention stores intentionally use different formats:

| Store | Representation | Role |
|---|---|---|
| Compressed global KV | FP4 E2M1 values, group-16 E4M3 scales | Long causal source history |
| Sliding-window KV | FP8 E4M3 values, group-32 E8M0 scales | Last 128 tokens at all 40 layers |
| Independent index keys | packed FP4 values with per-row scales | Learned sparse-source selection |

The default pool contains 49,216 page groups. Sources 2, 8, and 14 receive one page per group; source 20 receives two because of its compression ratio. The resulting 62,996,480 physical source rows occupy 22.43 GB, with another 43.26 MB for active SWA rings and small page-table and compressor-tail storage. Pages are reference counted and shared between active requests and retained radix branches.

Cache updates are transactional. A wave proposes private SWA rows, compression output, index keys, Engram history, and draft state. Target verification chooses the accepted prefix. Preflight validates every participant before any owner advances; the accepted rows are then committed and versions move together. An execution failure revokes the affected request state instead of exposing a partially advanced combination.

Compression operates on two-token groups. Odd trailing input is retained as a private carry so a later suffix can finish the group without changing the shared prefix. Source and index page writes accept high physical addresses and use 64-bit offsets throughout.

## Prefix reuse and agentic continuation

A token radix indexes rendered prompt tokens and image-content identities. An exact prompt hit can restore the complete target state and saved first-token logits. A partial hit aligns shared compressed sources to a complete two-token boundary and rebuilds at most the final 128 encoder tokens needed for the local windows. The new suffix then appends through copy-on-write if its last shared source page is partial. Divergent branches keep their common pages immutable.

Completed responses retain target, dSpark, compressor, window, history, and logit tails so the next agentic turn can resume without rebuilding the previous assistant output. Prompt snapshots and completed-turn snapshots have independent banks. Each defaults to 24 entries: sixteen concurrent working slots plus eight spare retained turns. LRU eviction removes the radix value, retained tails, and page references together; active owners are never eviction candidates.

The final prefix campaign proves cold, partial, exact, shorter, divergent, and multi-chunk branches; bounded replay; C2/C6/C16 isolation; cancellation and replacement; and eviction of only the twenty-fifth least-recently-used turn. Exact repeats report full prompt hits, while partial-hit counters exclude the bounded reconstruction window.

## Engram and mapped storage

Engram modules at layers 1 and 14 hash normalized 2-, 3-, and 4-grams. Token IDs determine the addresses before hidden-state execution, so bounded background workers can advise and gather mapped weight and scale pages while the GPU executes other work. Gathered rows move through bounded pinned staging, dequantize on the RTX, and enter fused projection and gating operations. The full tables remain memory-mapped on host storage rather than consuming coordinator VRAM.

Image spans reset n-gram history and receive no Engram residual. Accepted-prefix publication advances Engram history with the same request transaction as attention and draft state.

## dSpark speculative decoding

The dSpark drafter has three stages with 128 experts and top-3 routing per stage. It uses target residual information from layers 37, 38, and 39, plus the shared embedding and vocabulary head. Its projections, attention, routed and shared experts, FP8 draft windows, and output remain on the coordinator.

Draft tokens are private until the target backbone verifies them. The target accepts the matching prefix, emits those tokens, and commits the identical length to target cache, draft cache, and Engram state. Rejection cannot advance draft history. `--no-dspark` runs the same target path without proposals; this provides a direct correctness and performance control.

## Vision

The native vision owner loads the checkpoint encoder and aligner on the RTX. CPU preparation decodes JPEG, PNG, WebP, and GIF, normalizes image identity, applies the pinned resize/grid policy, and produces patch input. The API accepts up to sixteen images, with bounded encoded bytes, decoded features, decoder concurrency, redirects, and deadlines.

Image placeholders expand before context accounting. Aligned vision features replace only their matching token embeddings, and an image mask selects the model's text/image routing correction. Prefix keys include normalized image content identity, so changed or reordered images cannot inherit the wrong state. Exact retained image prompts can skip GPU encoding; a partial replay that starts inside an image reacquires the complete feature span required by that window.

The practical encoder uses BF16 matrix operations with FP32 accumulation and softmax. Qualification emphasizes complete serving behavior and semantic correctness rather than forcing byte identity with a slower reference implementation.

## API, reasoning, and constrained output

The HTTP service exposes OpenAI-compatible chat completions, streaming SSE, cancellation, usage and cache counters, model discovery, health, tools, parallel tool calls, JSON output, and supported JSON Schema constraints. Incremental UTF-8 handling carries incomplete byte sequences between token fragments so a terminal multibyte character is never dropped or replaced.

Thinking defaults to enabled at high effort. Explicit request controls select low, high, or max effort, or disable thinking. Reasoning and final answer content remain separate in both ordinary and streamed responses.

XGrammar compiles response constraints and masks logits on the native path. Strict tool schemas apply declared type, enum, numeric, string-pattern, object-property, and additional-property constraints. On an exact cache hit, retained logits are reselected under the current grammar so a new tool or response schema cannot reuse a token chosen under an older constraint.

## Memory and startup

The coordinator reserves model owners before sizing its shared cache. At the standard C16 launch, resident owners occupy 43.04 GiB and the eager architectural cache occupies 20.93 GiB for 25,165,824 tokens total (24 × 1,048,576-token contexts), for 63.97 GiB at readiness. A warmed C16 process uses 64.05 GiB. Exact pool bytes, total occupancy ceilings, context size, concurrency, and retention are independently configurable; insufficient plans fail startup instead of silently shrinking the requested service.

Each Spark reads and packs its own checkpoint shard in parallel. A cold-cache phase run brings the four workers to readiness in 38–39 seconds and the coordinator in another 5.44 seconds, for 44.65 seconds of measured core startup. The standard clean launch reaches the API in 55–56 seconds. Startup performs no weight RDMA exchange because every rank owns a local snapshot; the network path is reserved for inference activations and expert results.

Builds pin and verify the engine, model revision, SparkInfer, XGrammar, architecture, exported binaries, and notices. The amd64 coordinator and ARM64 worker images are built natively, and the worker image is distributed to the four Sparks after its identity is checked.

## Measured behavior

The corrected-FP4 release candidate reaches 2,668 prompt tok/s at its best median prefill cell, 60.68 tok/s on the weighted eight-type dSpark mix, 128.03 tok/s on low-entropy dSpark decode, and 683.70 aggregate tok/s at C16. Paired FP4/FP8 checks show no material retained-decode regression, while the compressed source allocation is substantially smaller. See the [performance](release-v1-performance.md), [memory](release-v1-memory.md), [startup](release-v1-startup.md), and [prefix](release-v1-prefix-final.md) reports for methods and raw evidence.
