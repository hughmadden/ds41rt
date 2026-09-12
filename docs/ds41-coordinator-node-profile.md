# Coordinator decode with graph internals visible

A launch-time Nsight Systems capture of the selected RMSNorm target API exposes
CUDA graph kernel nodes. The preceding delayed node capture exported successfully
but contained no CUDA activity, so it is excluded. Capture/export commands and
raw reports are retained under `/tmp/ds41-node-profile-full`; export completion
was confirmed before stopping the profiling container.

The analyzed interval is 5–19 seconds after launch, excluding loading and early
warmup. A short counting prompt produces 518 logit downloads during that interval.
The capture limit cuts the streaming request short; this is a profiling run,
not a quality or throughput qualification run. The native library and daemon
hashes are in [the result record](ds41-coordinator-node-profile.json).

The interval contains 963,793 kernels, including 710,024 graph-node kernels.
RTX kernel activity occupies 5.94 of 14 seconds; including copies raises that
to 6.26 seconds. Per observed target step, this corresponds to approximately
11.46 ms of kernel activity and another 0.62 ms of copy activity. The remaining
interval includes remote Spark execution, CPU scheduling and idle time. It must
not all be attributed to avoidable host overhead.

| Kernel category | Instrumented time per observed step |
|---|---:|
| Dense FP8 GEMMs | 5.49 ms |
| Sparse attention and merge | 1.46 ms |
| mHC kernels | 0.88 ms |
| RMSNorm | 0.65 ms |
| Quantization | 0.54 ms |

The BF16 LM-head cuBLAS GEMV separately averages about 0.88 ms per invocation.
These categories are selected contributors, not an exhaustive accounting.
Instrumentation affects short kernels, so these are not standalone unprofiled
kernel latency claims.

There are approximately 1,014 stream-synchronize calls, 383 blocking CUDA copies,
264 graph launches and 495 asynchronous copies per observed step. Synchronize
API duration includes waiting for GPU work and cannot be added to GPU time.
The four expert-return uploads per layer already use asynchronous copies; they
are not the blocking-copy problem.

The concrete next change queues input/position/metadata copies on the consuming
query, window, sparse-attention and output-projection streams. Their existing
completion drains preserve ownership and readiness, avoiding separate blocking
copy boundaries. Larger graph capture and GEMM/attention kernel tuning remain
necessary to approach the release targets.
