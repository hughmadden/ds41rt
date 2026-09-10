# Request-owned engram pipeline

`EngramPipeline` connects official token compression and transactional hashes to early page advice, bounded background gathers and RTX row upload for layers 1 and 14. It maps both tables through the validated official catalog; checkpoint files must remain immutable while mappings or outstanding jobs exist.

The intended scheduler sequence is:

1. Create one pipeline with explicit token capacity, gather-slot count and host staging budget, and create each request's history with `new_history`.
2. Call `prepare` with token IDs and image-span masks as soon as a decode, prefill or verification batch is known, combining at most 16 distinct requests within the configured row capacity.
3. Poll `EngramDeviceRows::poll_wave` on the CUDA thread with the current histories and layer index; it retries bounded gather admission, rejects stale request generations, uploads ready rows and recycles their host staging.
4. After verification, call `EngramWave::commit` with all histories and accepted-prefix lengths, or cancel/drop the wave to leave histories unchanged.

Preparation validates all hashes before submitting work for either layer, and does not mutate histories. Prefetch advice is best effort under backpressure; the gather worker resolves any pages that still fault. Gather admission remains pending until a slot becomes available, and both queued and completed work consume the same bounded slot pool. Image rows require correct masks from image processing and perform no table reads.

Every wave is tied to its creating pipeline and immutable request batches. Polling validates current owner/generation identities before exposing data; an error cancels the wave's remaining I/O. Commit validates every request before changing any history, preventing a stale request from partially committing a mixed wave. Accepted prefixes include image barriers, while rejected speculative suffixes never enter history. Cancellation cannot interrupt a page fault already in progress, but its result is discarded and its staging slot recycled.

The serving/model scheduler does not yet invoke this API. Projection of the gathered BF16 embeddings and the fused engram gate still need execution wiring. Compilation and daemon build pass; this pipeline has not been runtime-qualified, and no model load or tests were run for this integration.
