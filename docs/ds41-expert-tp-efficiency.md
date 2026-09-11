# One-row expert efficiency versus tensor-parallel width

On one GB10, keeping hidden width 5120, 384 experts, top six, native FP4/E8M0 weights, FP8 activations and the same direct/five-output-task kernel strategy:

| Represented TP | Intermediate per rank | Kernel extent | Repeated kernel median | Scaling efficiency | After cache-pressure write | Scaling efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2304 | 2304 | 821.75 µs | 100% | 837.31 µs | 100% |
| 2 | 1152 | 1152 | 408.35 µs | 100.6% (approximately ideal) | 446.19 µs | 93.8% |
| 4 | 576 | 640 | 222.27 µs | 92.4% | 264.99 µs | 79.0% |

Efficiency is `TP1 time / (TP * per-rank time)`, measuring useful scaling versus the full matrix. TP4 loses about 8% in repeated replay and 21% after the cache-pressure write in this diagnostic. Its 576-wide shard is padded to 640, adding 11.1% to that dimension; TP1/TP2 have no such intermediate padding. This is consistent with padding contributing to the warm gap, not proof that it explains all of it.

The CUDA graph contains the complete fused expert kernel returning FP32 route planes. It excludes final route reduction, TP communication and coordinator work, but includes the kernel's route preparation, activation quantization and output stores. No separate overhead subtraction establishes pure GEMM efficiency. Two reverse-order passes use fifteen intervals of twenty replays; the cache-pressure condition writes 128 MiB before timing one replay. It is not a DRAM-counter measurement or a guarantee of a particular eviction state.

All three widths passed the independent numerical oracle and changed-input graph checks. These are synthetic full-geometry measurements on one GPU, with the same random-weight distribution rather than shards of one shared weight tensor. TP1 and TP2 required isolated eligibility overlays; production remains official TP4. Do not use this experiment as evidence that the complete model fits or serves in TP1/TP2, or as end-to-end scaling evidence.

[Raw samples, probe source, eligibility overlays and machine metadata](ds41-expert-tp-efficiency.json). The exact container command is in the JSON record. The probe and overlays are `/tmp/ds41-tp-efficiency-probe.py`, `/tmp/ds41-tp-probe-{impl,policy}.py` on ostrich, with the pinned source mounted at `/b12x` and `B12X_DYNAMIC_W4A8_SHARE_INPUT=0`, `SPARKINFER_COMPILE_DISK_CACHE=0`.

## Effective weight throughput

Counting each selected expert's resident FP4 payload and E8M0 scales once gives `6 * 3 * 5120 * padded_intermediate * (1/2 + 1/32)` bytes per GPU per layer:

| TP | Packed weights and scales | Repeated effective rate | Cache-pressure effective rate |
| --- | ---: | ---: | ---: |
| 1 | 112.80 MB | 137.3 GB/s | 134.7 GB/s |
| 2 | 56.40 MB | 138.1 GB/s | 126.4 GB/s |
| 4 | 31.33 MB | 141.0 GB/s | 118.2 GB/s |

Decimal units. TP4's unpadded useful bytes are 28.20 MB, giving 126.9 GB/s of useful-weight throughput in repeated replay. These effective rates do not measure actual DRAM bandwidth: output splitting repeats FC1 reads, caches can service rereads, and activation/metadata/output traffic is excluded from the numerator. No peak-bandwidth utilization follows from this accounting alone.
