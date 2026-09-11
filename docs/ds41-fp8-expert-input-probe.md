# FP8 expert input transport candidate

The current expert input quantizer uses FP8 E4M3 with one UE8M0 byte per 32 values, with the V4.1 `1e-4` block-amax floor. It is independent of expert and TP rank. Moving this exact quantization to RTX can encode each BF16-rounded FFN input once before broadcasting it to all four Sparks. Input payload becomes 5,120 + 160 = 5,280 bytes per row, compared with 10,240 bytes of BF16, a 48.4375% reduction before frame/routing metadata.

A standalone CuTeDSL probe using the production `quantize_block_fp8_mx` intrinsic and V4.1 floor passed on RTX SM120 and Spark SM121. Both hosts used the identical serialized BF16 source: synthetic zero/small/outlier cases plus 80 rows from a captured real model input. Runtime row counts `[1, 16, 80, 160, 1]` reused one compiled callable. All payload and scale bytes matched the Torch quantization reference and matched across GPUs. Tail poison was preserved. A nondefault-stream graph first replayed nonzero input, then changed to zero input and produced zero payload with floor scale byte 105, without changing allocated bytes.

[Probe source, exact input/output hashes, commands and results](ds41-fp8-expert-input-probe.json). This is a component result, not a deployed input-format change. The probe writes separate payload/scale planes; the production wire layout remains to be specified. No speedup has been measured.

Remaining integration:

- Add an explicit E4M3/UE8M0 K32 input contract; existing row-scaled FP8 and debug FP8 enum values are different formats and must not be reused.
- Give the b12x plan an explicit prequantized input layout. Spark routing should copy/gather these bytes and scales into expert tiles without dequantizing and requantizing them.
- Export and bind the native contract with stable preallocated coordinator and worker buffers. Preserve BF16 rounding before quantization if fusing into the preceding RTX stage, and retain the BF16 input needed by router/shared computation.
- Compare actual official-weight FP32 route outputs against the BF16-input baseline across decode/prefill, graph replay and layer transitions before deployment. Then measure quantization, staging, wire and full API cost.

The return contract is not changed by this proposal. Cross-rank partial sums must not be reduced to FP8 merely because the input GEMM consumes FP8. The existing compact BF16 return numerical discrepancy remains a separate unresolved qualification gate.

## Intermediate width and packing

TP4's logical intermediate width is 576. Current native packing rounds it to 640 and uses N256/K128 lane-major storage, while the fused kernel processes 128-channel intermediate slices. This is an implementation choice, not proof that W4A8 inherently requires width 640. Removing the extra 64 channels would remove 10% of padded work/weight bytes, with an idealized 640/576 = 1.111 speed ratio if all other costs stayed fixed.

Candidate slice decompositions are `128*4 + 64`, `64*9`, and `192*3`. They preserve the 32-element quantization blocks, but that alone does not establish compatibility with the current producer, MMA register layout or resident packer. W13 gate/up boundaries and W2's intermediate K stride both depend on the current padding. Changing only the dimension or zero mask cannot remove its physical reads and MMA work safely.

Evaluate these alongside intermediate-slice task splitting: a 64-wide tail retains the bulk 128-wide schedule; uniform 64-wide slices offer more tasks but may increase scheduling/reduction overhead; 192-wide slices offer fewer tasks but require a different tile mapping. Preserve ordered partial accumulation and compare exact quantization boundaries, occupancy and actual kernel latency. None of these alternatives has yet been implemented or benchmarked.
