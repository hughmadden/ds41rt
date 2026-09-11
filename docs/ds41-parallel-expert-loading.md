# Bounded parallel expert loading

Expert loading now uses sixteen CPU read lanes with separate pinned staging and scratch. Each bounded group is fully read and joined before the owning CUDA thread uploads and packs it. The next group is advised with `posix_fadvise`; no unrelated model layers or engram tables are prefetched. Device staging remains a single reusable allocation, and packing drains before its reuse. Read errors prevent packing the affected group, and scoped readers finish before their buffers are released.

One Spark loader uses 75,202,560 pinned host bytes and 1,179,648 read-scratch bytes. Its device staging and packed resident weight sizes are unchanged. These host buffers are temporary loading allocations, not retained for every model layer. There are no new user profiles or alternate checkpoint formats.

## Why this change

The instrumented serial loader spent 2.81–3.02 seconds reading the slower sampled layers, versus 0.054–0.060 seconds uploading and packing. Replaying a captured expert graph did not materially improve one-row execution: at capacity 80, the measured medians were 390.717 µs replay versus 391.965 µs ordinary launch, with byte-identical outputs. No new graph cache was added for that negligible gain.

The same real-weight probe loaded layers 0, 1, 2 and 0 again while a full worker remained resident. First-layer observations are excluded from this comparison because cache state differed:

| Read lanes | Layer 1, seconds | Layer 2, seconds | Repeated layer 0, seconds |
| --- | ---: | ---: | ---: |
| Serial | 3.222077 | 3.124021 | 2.983371 |
| 4 | 1.308548 | 1.324882 | 1.299042 |
| 8 | 0.995615 | 0.995533 | 0.998880 |
| 16 | 0.898317 | 0.932456 | 0.906415 |

These sequential runs did not explicitly evict the page cache and are not controlled cold-start benchmarks. A separate short `dd` read with `iflag=direct` read the entire 970,533,624-byte first checkpoint shard in 0.214226 seconds, reporting 4.5 GB/s. This is a short sequential storage observation, not sustained full-model bandwidth.

At each tested parallelism, two recorded 80-row batches produced FP32 route planes byte-identical to the previously qualified serial-loader outputs: 9,830,400 bytes per batch. The test was repeated after loading other layers and reloading layer 0. This verifies the exercised routes and staging/packing integration; it is not a hash audit of every packed expert weight. The production release daemon and ARM worker fixture build successfully.

## Full four-Spark rollout

All four workers loaded all forty layers using the optimized release loader:

| Rank / host | Previous summed layer load, s | Parallel summed layer load, s |
| --- | ---: | ---: |
| 0 / ostrich | 118.293 | 35.677 |
| 1 / dodo | 115.483 | 36.801 |
| 2 / emu | 119.453 | 37.663 |
| 3 / kiwi | 113.067 | 37.787 |

Each rank's native checkpoint share is 72,194,457,600 bytes; padding/packing produces 80,216,064,000 resident bytes. The strided W2 reads and metadata bring read syscall bytes to about 144.4 GB per process. Storage counters reported 142–157 GB per worker in this rollout, including readahead and cache effects. The summed loading intervals exclude catalog validation and process/container startup. Readiness followed the final layer by 20–25 ms; there is no additional large graph-setup interval in this worker path.

After reloading, both saved 80-row requests were sent through the production TCP service on all four ranks. All eight route planes matched the earlier serial-loader outputs byte for byte: 78,643,200 bytes total. The complete target API also passed its arithmetic/counting, JSON/SSE usage/completion, disconnect recovery and unsupported-temperature checks. Three decode observations were 10.294, 10.639 and 10.115 tokens/s, with first content at 0.255, 0.245 and 0.243 seconds. These are loading-regression checks, not a decoding speedup claim, broad quality evaluation or release-readiness proof.

The upgraded workers and API remain running at the development endpoints. [Probe results, per-layer startup records, I/O counters, output comparisons, API events and binary hashes](ds41-parallel-expert-loading.json).
