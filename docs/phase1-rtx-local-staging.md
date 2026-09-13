# Local router staging and grouped expert loading

Local layers now use a separate router graph that downloads only six route IDs
per row: 24 bytes rather than the 5,328-byte remote request payload. Adaptive
history reads those IDs from the completed pinned staging buffer. Hidden input
and route weights remain on device, and local execution no longer constructs a
remote protocol request. Local and remote graph tables remain separate, with
both destroyed before their captured storage. Both paths validate row metadata.

Expert loading now queues each 16-expert group on one stream and drains before
CPU readers reuse its pinned buffers. Stream order protects the shared device
staging allocation between experts; the stream owner also drains on partial
failure. There are no packing arithmetic or allocation-budget changes.

| Five-layer adaptive ABBA comparison | Before | After |
|---|---:|---:|
| Code median tok/s, six samples each | 120.65 | 122.06 |
| 32K prefill median tok/s, six measured samples each | 7,740.49 | 7,751.45 |
| C4 aggregate median tok/s, two batches each | 117.91 | 118.28 |
| C16 aggregate median tok/s, two batches each | 183.94 | 185.44 |
| Total startup median, two launches each | 10.88 s | 11.03 s |
| Local loading/setup median, two launches each | 5.69 s | 4.52 s |

Code, prefill cache, mixed traffic and the first before/after lifecycle checks
passed. All 16 prefill responses were `7`, with matching prompts, 32,768 new
tokens and no cache reuse. The standard service was restored. Code text hashes
vary, and the named Python structure checks do not establish broader semantic
quality. The small throughput changes and variable loading times do not prove
a general speedup. Total startup remains essentially flat in this comparison;
host page-cache state was uncontrolled, and unrelated dense loading varied.

The final router GPU test passed 32 real-weight local/remote transitions at
1/16/80/256 rows. IDs, route weights and FP8 inputs match fresh remote graphs;
graph handles remain distinct and reusable. Poisoning confirms local graphs
leave staging beyond the ID region untouched. Final source additionally shares
the original row-metadata validation with the local path; that guard and the
expanded test assertion were added after freezing the timed binary. No native
kernel changed. The release build and final GPU test passed.

Keep placement opt-in. Further work includes target-only qualification, wider
quality checks, startup preservation, Spark layer omission and profiling the
remaining local execution/host completion costs before default promotion.

[Evidence](phase1-rtx-local-staging.json) records individual arms, artifacts,
source hashes and scope. Raw runs, requests, the paired output comparison,
`measured-source.patch` and the final test/build logs are under
`/tmp/ds41-rtx-local-staging`. No builds overlapped these benchmarks.
