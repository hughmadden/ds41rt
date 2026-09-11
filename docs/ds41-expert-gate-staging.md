# Gate/up staging and shared-memory layout diagnostics

The deployed expert kernel is unchanged. Full K128 gate/up double-buffering loses on prefill. K64 staging, weight permutation and register-based activation conversion expose structural opportunities, but their gains are too small or inconsistent at the serving batch of 1024 to justify a rollout. The next end-to-end comparison revisits coordinator batch size after the copy/admission fixes.

All measurements use ostrich, official layer 39/rank 0/all 384 experts, generated activations and skewed/mixed/shared routes, and planned capacity 4096. The native comparator is the deployed atomic library with SHA256 `ca4d7174e0b83eba9d1e06933c61e4a3290e6c48ed1bd4de909a97e275d03628`. Every candidate passes rtol 2e-6 / atol 2e-5, finite/nonzero oracle checks, poisoned output and tail, and graph replay without allocation. This is a bounded component gate, not full-model quality qualification. Six-row results use the large atomic state and must not be reported as serving decode performance.

Timing uses eight interleaved rotations/reversals of five graph replays. Clocks are uncontrolled. Values below are median complete graph milliseconds, with each experiment's own recovered M16 control; do not compare separate runs as though their clock/cache conditions were identical.

| Experiment, N192 | 1024 skew control → candidate | 4096 skew control → candidate | 4096 mixed control → candidate |
|---|---:|---:|---:|
| Full K128 double buffer | 7.558 → 8.086 | 15.620 → 18.009 | 12.251 → 15.367 |
| K64 double buffer | 7.577 → 7.495 | 15.896 → 15.270 | 12.346 → 11.139 |
| Weight permutation, K128 | 7.603 → 7.600 | 16.028 → 15.675 | 12.320 → 12.037 |
| Weight permutation plus K64 | 7.603 → 7.781 | 16.028 → 16.697 | 12.320 → 12.218 |
| Weight permutation plus activation layout | 7.639 → 7.595 | 15.870 → 15.603 | 12.192 → 11.924 |

N128 controls and both buffering alternatives also lose against native N192 on prefill; raw samples are retained in the companion JSON. The isolated experimental checkout is `/tmp/ds41-wide-revisit/b12x`, based on b12x `907d5fd5` plus recovered atomic/wide support. Rejected runtime options are not added to production.

## Resource evidence

Full buffering adds a second complete gate/up weight tile. K64 instead splits each original tile in half so both weight buffers fit in the original allocation. Source-derived requested dynamic shared memory at M16/N192 is 41,856 bytes for the control, 67,968 for full buffering, and 43,392 for K64 buffering. These are separate from the compiled 1,024 static bytes. The larger footprint is consistent with the full-buffer slowdown; achieved occupancy was not measured in this probe.

The exported CUDA images report these registers per thread, with zero stack/local bytes in every arm:

| N192 arm | Registers |
|---|---:|
| Control | 162 |
| Full buffer | 166 |
| K64 buffer | 156 |
| Weight permutation | 162 |
| Weight permutation plus K64 | 144 |

Lower register use alone does not select the winner. `scripts/inspect-ds41-cute-object.py` extracts the CUDA ELF embedded as data in a CuTe host object, preserves the input, hashes both representations, and invokes cuobjdump. It includes ELF program headers when determining the complete image extent. Fourteen actual AOT objects were inspected; standalone CUDA ELF, empty input and two truncated-image checks also pass. cuobjdump 13.2.78 supplies the resource records. Static shared-memory reports do not include dynamic launch allocations.

## Confirmed shared-load bank conflicts

The weight-read address is normally `lane * 4 + warp` within each packed block. A GPU-side permutation rotates the four words by `warp XOR (lane // 8)`; the kernel applies the matching address change. This preserves packed FP4 bits, scales and arithmetic. The prototype performs this permutation on the GPU with tensor indexing before timing. It has not been integrated into the native load-time packer or format contract.

Nsight Compute 2026.1.1 profiles one qualified 1024-row skewed replay per arm, selected by NVTX and fused-kernel name. Both cache flushing and clock control are disabled. Three counter passes per kernel report:

| Counter | Control | Weight permutation |
|---|---:|---:|
| Shared-load bank conflicts | 77,971,146 | 20,352,127 |
| Shared-read wavefronts | 110,625,027 | 53,001,020 |
| Profiled fused-kernel duration | 7.633 ms | 7.388 ms |

The conflict reduction is substantial, but unprofiled graph timings do not reproduce a substantial 1024-row speedup. The remaining conflicts and conversion schedule deserve investigation; they are not proven to be the sole bottleneck. The activation-layout follow-up permutes FP32 intermediate storage within each 32-value block and pads the quantized activation row stride by eight words. It passes the same numerical gate but adds little performance beyond the weight permutation.

## Register-based activation conversion follow-up

The next prototype retains the post-SiLU/routing/BF16-rounded values in the gate registers. Four-lane shuffles reduce each warp's eight-channel maximum, shared memory combines the four warp maxima for each 32-channel block, and neighboring lanes supply pairs to the existing FP8 conversion primitive. The FP32 intermediate shrinks from 12,288 bytes to 1,536 bytes at M16/N192. It preserves the `1e-4` maximum clamp, power-of-two scale calculation and FP8 rounding, and passes all four existing diagnostic cases, graph replay and poison checks.

This still does not establish a useful realistic-prefill win: 1024 skewed graph time is 7.631 ms control versus 7.648 ms register conversion, or 7.608 ms with weight permutation as well. At 4096 skewed it is 15.912 → 15.636/15.598 ms. Uniform mixed improves more, 12.180 → 11.142/11.066 ms, but cannot select the serving policy. Registers rise from 162 to 168 without permutation, or 164 with it; both have zero stack/local bytes. Source-derived dynamic shared allocation drops to 31,104 bytes. No native kernel or resident-format changes are deployed from this experiment.

CUDA driver occupancy queries against the exported images on ostrich confirm maximum active blocks per SM rise from two to three at 128 threads with the declared dynamic shared allocation. This is a theoretical residency limit, not measured achieved occupancy. The numerical and timing results therefore do not support treating residency alone as the limiting factor.

With these realistic-shape gains still small, the next end-to-end check revisits larger coordinator batches after the response-copy/admission fixes. This tests whether better expert sharing can now translate into API throughput, instead of continuing to select kernels using uniform-routing gains.

## Reproduction artifacts

Launchers, patches, raw JSON, ELF objects and the Nsight report are under `/tmp/ds41-wide-revisit` on the coordinator and/or ostrich. The companion JSON records hashes, all graph samples, native identity and counter exports. Representative launch (substitute `half`, `swizzle`, or `activation` for `up`):

```bash
docker run --rm --gpus all \
  -v /tmp/ds41-wide-revisit:/audit \
  -v /tmp/ds41-direct-source/python/tools:/tools:ro \
  -v /tmp/ds41-expert-direct/cmake:/native:ro \
  -v /home/tj/.cache/huggingface:/hf:ro \
  -e PYTHONPATH=/audit/b12x:/tools \
  --entrypoint python3 ds41rt-spark-expert-dev:latest \
  /audit/up-bench.py \
  --snapshot /hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277 \
  --native-lib /native/libds41rt_native.so --layer 39 \
  --output /audit/up-result.json
```

Each launcher requires its corresponding archived prototype source, not whichever experiment happens to occupy the checkout. The Nsight launcher is `swizzle-profile.py`; select `--nvtx --nvtx-include compare/ --kernel-name regex:V41FusedSliceKernel --cache-control none --clock-control none` and metrics `gpu__time_duration.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_cmd_read.sum`. Import the resulting report with `--page raw --csv` to obtain counters.
