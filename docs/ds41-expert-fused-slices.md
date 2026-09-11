# Fused V4.1 compute at intermediate widths 64, 128 and 192

b12x `b4a02a7` adds a low-level fused M16 expert-slice kernel using the [exact slice staging](ds41-expert-slice-staging.md). It reads native packed FP4 weights and prequantized E4M3/K32 input, computes FC1, applies the V4.1 BF16/clipping/SiLU/routing boundary, requantizes with the `1e-4` floor, and computes FC2 within the same CTA. The intermediate activation stays in shared memory. FP32 FC2 slice partials go to global output for subsequent reduction.

Widths 64 and 192 cover the logical intermediate 576 as nine and three exact slices. The width-128 comparison uses five slices and masks its final 64 padded columns. All consume the current padded-640 resident representation; this does not yet reduce resident allocation size.

## Qualification

The new `tests/moe/test_v41_fused_slice.py` passes on RTX SM120 and Spark SM121: three width cases on each device, each reusing one compiled callable for live row counts 1, 2, 6, 16 and return to 1. Changed inputs and routing weights pass captured graph replay with stable allocation. Empty-row and zero-input cases return zero, including unused output rows.

All thirty final BF16 expert-output comparisons match the independent quantization-aware oracle exactly on these synthetic fixtures. Slice reduction is performed by the test harness before BF16 conversion; this does not establish FP32 bit equality across different slice widths or qualify a native serving reduction. GPU packing uses the existing b12x preparation path. These are one-expert component fixtures, not official checkpoint or mixed-expert batch qualification.

Both images run:

```sh
python -m pytest -q -s -p no:cacheprovider \
  /workspace/ds41rt/third_party/sparkinfer/tests/moe/test_v41_fused_slice.py
```

[Case results, logs and source hashes](ds41-expert-fused-slices.json) are retained. Spark qualification precedes whitespace-only formatting; RTX qualification uses the final formatted source.

## Next integration

This is a private compute kernel, not a serving binding or planner policy. It stages synchronously and currently handles one expert's M16 group. Add multiple-expert route metadata and asynchronous staging, qualify against the existing native path and official weights, then compare 64/128/192 on representative mixed-expert FP8 decode batches. Include ordered reduction and workspace cost in that comparison. One-expert timings would not establish the optimal speculative-decode layout, so no performance claim is made here.

The deployed APIs retain their prior kernels. The temporary qualification shutdown was restored afterward.
