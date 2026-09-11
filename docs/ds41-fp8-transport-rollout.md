# FP8 input transport live on RTX and four Sparks

Root `b1e386f` and b12x `6adffce` integrate GPU encoding into the coordinator request path. The existing CuTe MXFP8 quantizer writes each row directly as 5,120 E4M3 payload bytes followed by 160 UE8M0 K32 scales, preserving the BF16 input boundary and `1e-4` amax floor. Custom output strides avoid a separate pack/copy stage, and the wire specialization skips unused MMA-scale output. Input pointers and grids are preinitialized; live row count remains a runtime argument.

The router's captured execution now includes this quantization, with a reusable 5,280-byte-per-row output allocation. BF16 input remains available for router/shared computation. Request construction copies the encoded output to host and broadcasts the same representation to all four TP ranks. This is graph-integrated encoding, not fusion into the preceding arithmetic kernel. Host transport still uses synchronous copies and per-request frame allocation; registered buffers and further overlap remain outstanding.

Native CMake chooses FP8 K32 input for Spark and BF16 for RTX dSpark. Both builds export the input quantizer. Native ABI 2 reports input representation, and the worker rejects mismatches before copying input. Return traffic remains the existing locally compacted BF16 rank partial; its numerical qualification is unchanged.

## Qualification and deployment

- Two GPU wire-layout cases pass on RTX, each exercising runtime rows 1, 2, 6, 16, 80, 256, 4096 and return to one under fixed compilation. Payload/scales match the reference exactly; tails remain poisoned; changed-input graph replay allocates nothing. Sixteen existing quantization-floor cases also pass.
- Native C ABI quantization passes the same row/graph/reference checks and rejects invalid row counts, alignment and overlapping buffers. Actual captured input bytes match the earlier intrinsic encoder exactly.
- Combined router execution passes 166 official-weight comparisons: all forty layers at rows 1 and 80 over two rebind cycles, plus three layers at rows 4096. Captured and fresh scores, routes and encoded input are exact; input addresses and graph handles remain stable.
- The final frozen Spark library passes the official two-layer Rust-owner qualifier. All 36 FP32 route/compact BF16 files remain byte-exact against the BF16-input baseline: 77,475,840 bytes. This does not close the older compact-return logit discrepancy.
- Coordinator release build and workspace checking pass. Native libraries were relinked with newly exported expert/quantizer objects and rebuilt expert wrappers, retaining unchanged nonexpert objects; these were not clean full-native builds.
- All eight paired API quality cases pass. Target and speculative text and token usage are identical before/after rollout. Both API checks pass JSON response, streaming, cancellation recovery and unsupported-sampling rejection.

`ds41-fp8-target-worker` runs on ostrich/dodo/emu/kiwi, ranks 0–3, capacity eighty and forty layers. `ds41-native-api-dev` and `ds41-native-spec-api-dev` serve ports 18041/18042 on RTX GPUs 0/1. Frozen artifacts and exact recreation commands are in the evidence. The previous workers remain stopped as `ds41-decode-target-worker`. Previous API containers used AutoRemove and disappeared on stop; rollback recreates them from retained `/tmp/ds41-compact-release` and `/tmp/ds41-compact-cmake` artifacts after restoring BF16 workers. Stop replacement containers before restoring endpoints with the same names/ports.

## Measured result

Activation payload falls from 10,240 to 5,280 bytes per row: **48.4375% smaller**. The logged six-row request is 32,448 bytes, versus 62,208 with BF16 input; routing metadata remains FP32 and response payload remains unchanged.

Three short counting streams measured target decode at **10.905, 10.886, 10.663 TPS** and speculative decode at **45.756, 45.766, 47.410 TPS**. These are below the prior run's medians of 11.431 and 47.599 TPS; no speedup is claimed. This is not a balanced A/B measurement, and the run now includes expert-M histogram logging. The byte reduction is established; its net latency benefit is not. Further work must address scheduling, compute and coordinator/transport costs rather than equating fewer bytes with higher TPS.

One-row Spark kernel medians are 237–245 µs; six-row medians are 632–645 µs. In the 80-row group, the median active-expert count is 86 and kernel time is 3.13–3.15 ms. Unique-weight byte accounting is not measured DRAM traffic.

## Decode tiling workload

For ostrich's 2,440 six-row observations in the mixed qualification workload (primarily speculative verification), the logical expert-local M distribution is:

| Expert-local M | Share of active experts | Share of routed rows |
|---|---:|---:|
| 1 | 61.32% | 36.92% |
| 2 | 23.56% | 28.37% |
| 3 | 7.61% | 13.74% |
| 4 | 4.20% | 10.11% |
| 5 | 1.83% | 5.52% |
| 6 | 1.48% | 5.34% |

Thus M≥2 carries about 63% of routed rows, while M=1 still dominates expert count and therefore unique-weight demand. Use both distributions when comparing 192-wide, 64-wide and 128-plus-tail intermediate slices. The median six-row batch exposes 22 distinct experts, materially more work than single-token/top-six execution. These logical counts precede kernel M padding and are not a proof of any tiling's occupancy or speed. Large-prefill throughput remains a guardrail for the resident layout chosen around common speculative decode.

[Deployment, hashes, API results, native probes and timing evidence](ds41-fp8-transport-rollout.json). Raw timing logs are `/tmp/ds41-fp8-{host}-{gpu,transport}.log`: stdout/stderr were collected separately because merging Docker log streams can splice records. All performance targets, C16 scheduling, long-context/prefill and broader independent quality gates remain open.
