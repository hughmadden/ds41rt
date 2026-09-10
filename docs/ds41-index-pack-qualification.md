# Native index encoding

`ds41rt_v41_index_pack` converts finite BF16 vectors of width 128 into E2M1 FP4 bytes and one E8M0 scale per 32 values. Each row occupies 64 value bytes and four scale bytes; even columns occupy the low nibble. Scale selection follows the official `fast_round_scale(max(amax, 6 * 2^-126), 1/6)` with FP32 multiplication. CUDA's saturating round-to-nearest-even conversion supplies FP4 values. This is the architecture's index format, separate from the fixed FP8 serving KV setting.

The operation allocates nothing, accepts 1–131,072 flattened vectors (up to 4096 queries with 32 heads), and checks null, alignment, address overflow and overlapping spans before launch. Each 16-lane subgroup reduces one group and writes 16 packed bytes plus its scale. Query projection is not implemented by this change.

Source graphs now pack the normalized/rotated real-weight index keys immediately after producing them. Wave-owned packed/scales buffers use the same publication and stale-proposal validation as latents. They remain proposal storage: only completed groups from accepted prefixes are eligible for future persistent writes. Existing commit returns accepted row metadata; the persistent index writer and transaction spanning all caches remain unimplemented.

`scripts/qualify-ds41-index-pack.py` checks the pinned official `inference/kernel.py` SHA-256 and runs its actual TileLang 0.1.8 quantizer, without rewriting the oracle. On each RTX PRO 6000 Blackwell, all nine cases match both packed and scale bytes exactly: random matrices at 1/3/16/80/255/4096/131072 rows, all 65,280 finite BF16 bit patterns, and FP4 midpoints with adjacent BF16 values and signed zeros. Every case also checks negated-input captured replay, zero-input replay, original-input recovery, input preservation, output boundary sentinels, unaligned byte outputs and rejected invalid pointers/shapes/overlaps. The zero-group E8M0 byte is one (2^-126).

The existing real-weight owner fixture also passes 50 cases on each GPU across layers 2/8/14/20, request counts 1/3/16, competing graphs, changed packed request order, partial and zero acceptance, stale proposals, released/reused slots, 4096-row prefill and replacement tails. It independently recomputes packed values/scales on CPU from the graph's BF16 index keys and requires exact bytes, including exact recovery after invalid proposals. Upstream real-weight compressor/index outputs retain their prior numerical tolerances against the PyTorch reference; their packed outputs are not claimed bitwise equal to quantizing numerically different reference projections.

Packing adds 68 device bytes per row. Ratio-two waves now use 5,491,264 bytes at 80 rows or 70,598,656 at 4096; ratio-one waves use 5,244,864 or 57,982,976. Shared weights, pending state and pinned metadata are unchanged. Driver/library/graph allocations and downstream persistent caches are excluded.

The native library, daemon and owner fixture builds pass. Adjacent JSON records source hashes, primitive results and fixture/build logs. The official oracle was installed separately under `/tmp/ds41-tilelang`, using `tilelang==0.1.8`, `apache-tvm-ffi==0.1.2` and `z3-solver==4.15.4.0`; newer TVM FFI rejected the pinned compiler's legacy object declarations. No production dependency was added. A local reproduction uses:

```bash
LD_LIBRARY_PATH=/tmp/ds41-tilelang/z3/lib:$PWD/.venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/cu13/lib \
PYTHONPATH=/tmp/ds41-tilelang .venv/bin/python scripts/qualify-ds41-index-pack.py \
  --native-lib /tmp/ds41-v41-aot/cmake/libds41rt_native.so \
  --reference-dir /tmp/ds41-reference --device 0 --output /tmp/ds41-index-pack-rtx0.json
```

This establishes index encoding and source graph integration. Persistent cache ownership/writes, hierarchical selection, sparse attention, full-model execution and performance targets remain open.
