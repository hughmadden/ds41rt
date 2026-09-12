# Release v1 startup qualification

The exact clean-built release images reach API readiness in **44.65 seconds of core cold-cache startup**: the four Spark workers load in parallel and the slowest is ready after 39.08 seconds, then the coordinator is ready in 5.44 seconds. The full standard `run.sh` gate, including SSH/container orchestration and one-second health polling, measured 55–56 seconds. This is below the 60–90 second release objective without moving work into the serving hot path.

The campaign stopped every model container and issued `POSIX_FADV_DONTNEED` for all 88 resolved files in the 510.31 GB model snapshot on every host before launch. It used the qualified coordinator image `sha256:a13258e92dd25ddb889bd31bb77c8813c7881868b103ceec0c140e30e893c213`, Spark image `sha256:2f5d328a14f1a52a0d3b2356b041415d744635f3f3557e41403246429984093b`, and model revision `dba1be0a40aa45a94ad051997016db3960a90277`.

## Spark load phases

Each rank reads, uploads, and packs its own 80.216 GB resident expert shard across 40 layers. Sixteen bounded host read lanes overlap checkpoint access; CUDA uploads and native packing remain on the owning thread. The production layer timer encloses that combined pipeline because splitting it would add synchronization to the path being qualified.

| Rank / host | Init | 40-layer read/upload/pack | Median layer | Slowest layer | Service handoff | Ready |
|---|---:|---:|---:|---:|---:|---:|
| 0 / ostrich | 1.039 s | 37.893 s | 930 ms | 1,380 ms | 108 ms | 39.072 s |
| 1 / dodo | 1.020 s | 36.912 s | 910 ms | 1,147 ms | 76 ms | **38.057 s** |
| 2 / emu | 0.977 s | 37.686 s | 931 ms | 1,189 ms | 50 ms | 38.740 s |
| 3 / kiwi | 1.122 s | 37.336 s | 925 ms | 1,257 ms | 54 ms | 38.556 s |

The packed resident-output rate is 2.12–2.17 GB/s per worker, including H2D transfer and format conversion. Checkpoint payloads are accessed through mmap page faults, so `/proc/1/io` does not expose trustworthy physical NVMe bytes. The report therefore gives the cold-cache method, exact resident bytes, and combined production phase instead of claiming an unsupported disk-only bandwidth. End-to-end startup already beats the target; further splitting or parallelizing GPU transformations would need a new candidate and runtime requalification.

## Coordinator phases

The coordinator's backbone, index, and embedding weights load in 2.712 seconds after cold-cache advice. dSpark residency, Engram, vocabulary/head, vision, two execution lanes, transports, and the 20.93 GiB cache pool (25,165,824 tokens total; 24 × 1,048,576-token contexts) take the remaining 2.725 seconds. The native API then binds port 8000.

The earlier clean `run.sh` campaign measured 56.481 seconds, and a restored standard launch measured 55.152 seconds. Its one-second remote port and health polling plus SSH and container creation account for the difference from the 44.65-second core launch.

## Exchange and graph preparation

Startup performs **zero RDMA weight exchange**. Each Spark rank reads its assigned checkpoint expert tensors locally; RoCE connection/QP handoff occurs after all weights and the execution workspace are resident. This avoids serializing four storage paths through the coordinator.

Startup performs **zero CUDA graph captures**. Stable arenas and graph-capable owners are allocated before readiness, while graphs are captured lazily for the first live row/request shape. In the separate graph measurement, the first C1 low-entropy request reached first content in 0.886 seconds and completed in 5.642 seconds. Capturing the maximum C16 speculative shape increased coordinator residency by 74 MiB. Subsequent requests reuse those graphs. This policy keeps startup short and avoids capturing unused shapes.

All 160 per-layer records, timestamps, cache-eviction script, coordinator phases, hardware state, process counters, and restored standard-launch log are preserved in [`evidence/native-release-startup.tar.gz`](evidence/native-release-startup.tar.gz). The machine-readable breakdown is [`release-v1-startup.json`](release-v1-startup.json).
