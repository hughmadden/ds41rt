# Request image features and replay masks

Prepared image features can now replace target embeddings before the first
backbone block. Each selected BF16 row is copied into all four mHC residual
lanes; text rows and the initial mixing weights are preserved. Each execution
lane owns its upload buffers, so alternating encoder chunks do not share mutable
image staging. The text path continues to use its existing embedding graph.

Request-owned image spans determine the masks used by ordinary prefill, reserved
encoder chunks and decoder replay. The same spans supply the last three token
classifications needed to restore Engram history at a bounded replay boundary.
Masks include image start, newline and end embeddings. Explicit masks that
contradict attached image spans are rejected. Per-request ownership, batch
identity and cache-lease generations prevent feature lookup through cancelled or
recycled request slots.

Features are prepared lazily by span: a complete prefix hit can require no image
encoder work, while a replay beginning inside an image requires that image's
complete feature span. The owner exposes this preparation range to the upcoming
API integration. Image layout and feature storage are separate, so replay can
recover routing masks from layout even when it consumes an already-produced
encoder suffix. The dSpark reference seeds draft windows from target hidden
states and applies no vision routing bias to text drafts; the existing draft
path therefore consumes the corrected target taps.

## Memory

Each embedding lane reserves an additional `capacity × (5120 × 2 + 4)` bytes
for BF16 image rows and their destination indices. At the maximum 4,096-row
capacity this is 41,959,424 bytes per lane, or **80.03 MiB for two lanes**.
The allocation is included in the embedding owner's budget and occurs before
native serving measures free memory for its KV pool plan.

Complete image spans are retained on the host by active requests and copied only
for the rows being executed. One request can hold at most sixteen 1,024-row
spans: **160 MiB** of BF16 features, independent of the shared encoder's device
workspace. Release and cancellation drop these feature owners. Retained prefix
snapshots continue to hold image identity pins rather than image feature arrays.

## Validation

The CUDA embedding fixture passes complete byte comparisons at 1, 35, 129 and
4,096 rows, then returns to the smaller shape. It checks every channel in all
four mHC lanes, preservation of text and mixing weights, repeated absolute
positions belonging to different requests, graph reuse after image execution,
invalid indices/order/token types/extents and recovery. Compute Sanitizer
memcheck reports **zero errors**.

The request integration fixture runs sixteen independent image owners through
one 3,040-row batch, including 2,944 image rows. It verifies complete GPU output,
reversed request ordering, exact masks and exact Engram hashes across two reserved
chunks. Bounded-history restoration matches full-history hashing at **112 image
boundary positions**. It also rejects conflicting explicit masks and verifies
cancellation and slot reuse clear image ownership.

The existing native request lifecycle fixture still passes mapped prefetch,
gather/upload, sixteen admissions, reserved chunks, explicit image barriers,
invalid acceptance, failed-commit cleanup and fresh-generation recovery. Three
CPU image-layout tests and all ten cache-identity/radix tests pass. The first
new request-fixture run used the wrong start position in its independent Engram
comparison; that fixture error and the corrected run are preserved.

These are component and request-plumbing checks. They do **not** qualify live
vision answers or a full decoder replay through the model. API image admission
remains disabled until encoder preparation, expanded token accounting and serving
integration are connected. Cached/uncached target/dSpark image parity, sixteen
images in one API prompt, cancellation under live vision load, final memory
accounting and text-performance regression checks remain required.

## Reproduction

Build the current native CUDA library and daemon test binary, then set
`DS41RT_VISION_LIBRARY` and `DS41RT_VISION_MODEL` to the library and official
checkpoint. Run these ignored GPU tests individually:

```bash
cargo test --manifest-path rust/Cargo.toml -p ds41rt-daemon --bin ds41rt \
  v41_target_embedding::tests::native_image_replacement_and_reuse \
  -- --ignored --exact --nocapture
cargo test --manifest-path rust/Cargo.toml -p ds41rt-daemon --bin ds41rt \
  v41_requests::image_tests::native_request_image_masks_features_and_history \
  -- --ignored --exact --nocapture
```

Run the first test binary invocation under
`compute-sanitizer --tool memcheck --error-exitcode 99` for device memory checks.
The existing lifecycle test uses `DS41RT_REQUESTS_LIBRARY` and
`DS41RT_REQUESTS_MODEL` with its full AOT native library.

[Evidence metadata](release-v1-vision-features.json) records source/artifact hashes,
commands, hardware and complete logs. The new fixtures use the separately built
CUDA component library; the existing lifecycle fixture uses the unchanged full
AOT library. No shared Spark inference campaign ran during these component tests.
