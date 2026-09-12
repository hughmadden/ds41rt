# Release v1 memory accounting

The clean-built corrected-FP4 candidate uses **63.97 GiB at readiness** with the standard C16 launch. A maximum-concurrency speculative decode raises live process residency from **65,510 MiB to 65,584 MiB**, leaving **31,655 MiB** free on the 96 GB RTX PRO 6000. Measurements used the enforced 400 W power limit and standard 14,001 MHz maximum memory clock.

The qualified artifacts are coordinator image `sha256:a13258e92dd25ddb889bd31bb77c8813c7881868b103ceec0c140e30e893c213` and Spark image `sha256:2f5d328a14f1a52a0d3b2356b041415d744635f3f3557e41403246429984093b`, built from engine revision `d5fb015d00aa7b302e3ce179b097482da678db76`. The launch uses port 8000, C16, 24 retained turns, 24 prompt snapshots, 1,048,576 context tokens, 393,216 output tokens, a 2,048-token prefill chunk, and dSpark.

## Startup allocation plan

| Allocation class | Bytes | GiB |
|---|---:|---:|
| Resident owners before the cache pool | 46,216,249,344 | 43.04 |
| Eager architectural cache pool | 22,471,251,456 | 20.93 |
| Planned total at readiness | 68,687,500,800 | 63.97 |
| Reserved runtime headroom, not eagerly allocated | 2,147,483,648 | 2.00 |
| CUDA allocatable device ceiling | 101,973,491,712 | 94.97 |

The planner measures device allocations after the model weights, two target lanes, two transports, dSpark, and vision are live. It then allocates the cache. The fresh process differs from that byte-exact plan by only 4,712,960 bytes in `nvidia-smi` process accounting.

## Cache pool

The cache has 49,216 page groups. Source layers 2, 8, and 14 each receive 49,216 pages; source layer 20 receives 98,432 because it compresses at half the ratio. Across 62,996,480 physical source rows:

| Cache storage | Bytes | Format |
|---|---:|---|
| Compressed KV values | 16,127,098,880 | FP4 E2M1 |
| Compressed KV scales | 2,015,887,360 | E4M3, group 16 |
| Independent index keys | 4,031,774,720 | FP4 packed |
| Independent index scales | 251,985,920 | per-row scale |
| Source storage total | 22,426,746,880 | shared paged ownership |
| FP8 SWA rings | 43,258,880 | 128-token windows, C16 |
| Page tables, lengths, and compressor tails | 1,245,696 | ownership metadata |
| **Eager cache total** | **22,471,251,456** | |

One retained completed state uses at most 2,922,816 bytes: 2,720,064 bytes of backbone tails and 202,752 bytes of dSpark tails. Each 24-entry bank can therefore retain 70,147,584 bytes; completed-turn and prompt-snapshot banks together have a 140,295,168-byte upper bound covered by runtime headroom. The large paged source allocation is shared by retained radix branches and is already counted in the eager pool.

## Weights and execution owners

| Owner | Bytes |
|---|---:|
| Target backbone weights | 6,896,423,360 |
| Cache-producer weights | 145,517,568 |
| Index weights | 45,916,160 |
| Shared token embedding | 1,323,827,200 |
| Engram weights and packed scales | 324,874,240 |
| Shared vocabulary head | 1,323,827,200 |
| dSpark resident tensors and packed scales | 7,950,520,200 |
| Vision weights | 970,536,960 |
| Vision buffers and workspace | 617,439,296 |
| Two target execution lanes and transports | 23,545,704,304 |
| dSpark 4K main-context owner | 498,974,736 |
| Three dSpark FP8 windows | 15,828,096 |

The target-lane figure includes two independent 4,096-row embeddings, backbone workspaces, index selection/query arenas, cache-producer proposals, Engram uploads and gates, target heads/taps, and TP4 transports. Their individual byte counts are preserved in the JSON evidence.

The remaining 2,556,849,784 bytes before cache allocation combine the dSpark draft execution chain with CUDA context, library, graph-bookkeeping, and allocator state. These cannot be separated reliably using the public CUDA process counters, so the report keeps the exact production planner fields and labels the measured residual instead of inventing a split.

## Lazy graphs and runtime state

A fresh launch reported 65,510 MiB for the coordinator process. After an untimed warmup and a complete C16 speculative counting batch, it reported 65,584 MiB, a **74 MiB** increase. The batch produced 657.64 aggregate tokens/s and exercised the maximum request-count graph shape. This is a measured graph/runtime increment for the qualified workload; new shapes may add further lazy graph state inside the 2 GiB runtime allowance.

The temporary planner probe only called production `device_bytes` and `plan` functions against the qualified library and official catalog. It was removed afterward, leaving the source tree unchanged. The service was then restarted through the standard `run.sh` path. Raw startup, planner, container, clock, C16, and restored-launch records are preserved in [`evidence/native-release-memory.tar.gz`](evidence/native-release-memory.tar.gz); the machine-readable report is [`release-v1-memory.json`](release-v1-memory.json).
