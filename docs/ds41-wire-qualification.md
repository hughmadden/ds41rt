# Native expert wire precision

The expert frame now uses signature `DS41RTE3` and wire version 3, with twelve-byte route entries containing little-endian u32 row index, u32 expert ID, and an unmodified IEEE FP32 routing weight.

The old ten-byte entry truncated each routing weight to BF16; that changes V4.1's weighted intermediate before FC2 quantization, so construction and serialization no longer perform this conversion.

Output dtype code 6 denotes FP32, with checked four-byte element/row sizing for unrounded expert route planes.

The existing Rust `protocol_v2` module, public type names, native entry-point names, and framing/streaming APIs remain in place during execution migration; their wire bytes now strictly require revision 3 and reject the previous signature/version.

Rust frame producers, FFI probe fixtures, native validators, and native self-test fixtures agree on the new signature and version.

## Evidence

- The existing transport suite passes 136 tests with one pre-existing ignored test.
- Four added native V4.1 tests pass, covering exact FP32 bits in owned/borrowed frames, subnormal and maximum finite routing weights, 16-by-6-by-5120 FP32 response planes, byte-count overflow rejection, old-wire rejection, and two requests over one persistent TCP connection.
- The native CPU library rebuild completes and its two CTest self-tests pass.
- The daemon passes `cargo check` through `scripts/run-with-python-env.sh` with Python 3.12; the system Python 3.14 remains incompatible with the pinned PyO3 version.

The TCP fixture repeats route weights across each returned plane to expose conversion or ordering changes; it is a transport regression, not a substitute for native expert execution.

## Remaining integration

The production model executor still needs to launch the native route-partial API, retain those buffers until transfer completion, validate the per-route response shape, and invoke the TP4 reducer before final rounding.

Existing host reductions and Spark collectives still operate on their old token-output contract and are not qualified for FP32 per-route planes by these tests.

All coordinator and Spark binaries/native libraries must be rebuilt together before deploying revision 3; previously built revision-2 images cannot communicate with the new framing.

RDMA transfer of native FP32 planes, cross-machine numerical agreement, and complete serving remain open.
