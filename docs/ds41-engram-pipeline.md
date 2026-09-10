# Request-owned engram pipeline

`EngramPipeline` connects official token compression and transactional hashes to early page advice, bounded background gathers and RTX row upload for layers 1 and 14. It maps both tables through the validated official catalog; checkpoint files must remain immutable while mappings or outstanding jobs exist.

The intended scheduler sequence is:

1. Create one pipeline with explicit token capacity, gather-slot count and host staging budget, and create each request's history with `new_history`.
2. Call `prepare` with token IDs and image-span masks as soon as a decode, prefill or verification batch is known, combining at most 16 distinct requests within the configured row capacity.
3. Poll `EngramDeviceRows::poll_wave` on the CUDA thread with the current histories and layer index; it retries bounded gather admission, rejects stale request generations, uploads ready rows and recycles their host staging.
4. After verification, call `EngramWave::commit` with all histories and accepted-prefix lengths, or cancel/drop the wave to leave histories unchanged.

Preparation validates all hashes before submitting work for either layer, and does not mutate histories. Prefetch advice is best effort under backpressure; the gather worker resolves any pages that still fault. Gather admission remains pending until a slot becomes available, and both queued and completed work consume the same bounded slot pool. Image rows require correct masks from image processing and perform no table reads.

Every wave is tied to its creating pipeline and immutable request batches. Polling validates current owner/generation identities before exposing data; an error cancels the wave's remaining I/O. Commit validates every request before changing any history, preventing a stale request from partially committing a mixed wave. Accepted prefixes include image barriers, while rejected speculative suffixes never enter history. Cancellation cannot interrupt a page fault already in progress, but its result is discarded and its staging slot recycled.

`EngramLayerWeights` loads native FP8 `wkv.weight` `[25600,6144]`, UE8M0 scales `[800,192]`, and BF16 `q_weight`/`k_weight` `[4,5120]` through bounded pinned staging. These four tensors occupy 157,521,920 device bytes per layer; mapped embedding tables remain excluded from eager loading. The reusable coordinator tensor loader admits total resident bytes before allocation and preserves checkpoint representations.

`EngramLayerWeights` additionally packs the native scales into 4,915,200 bytes of MMA scale storage, bringing per-layer residency to 162,437,120 bytes without changing FP8 weight bytes. `EngramGate::execute` now owns K32 activation scratch, alpha, projected KV and residual output, enqueues native quantization/GEMM followed by the fused gate on one stream, and synchronizes before exposing output. M16 execution workspace is 1,600,516 bytes, separate from gathered-row storage. Native handles bind the compiled device/SM geometry and reject invalid capacities, pointers, overlaps and undersized scratch; module loading and initialization occur before execution.

The C/Rust launch bridge, native scale packer and complete owned projection/gate path build successfully at the new SparkInfer pin, with evidence in `ds41-native-fp8-bridge-build.json`. Coordinator release and WIP builds enable FP8 AOT and carry `V41_FP8_AOT.json`; Spark artifacts explicitly record the feature as disabled. Full container builds were not repeated for these script changes, which passed shell syntax checking.

The serving/model scheduler does not yet invoke this API. Compilation and daemon build pass with 443 warnings, including unused integration APIs; the new residency/gate ownership has not been runtime-qualified, and no model load or tests were run for this integration.
