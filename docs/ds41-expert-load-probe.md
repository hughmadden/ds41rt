# Checkpoint-backed expert staging probe

The reusable `v41_load_probe` loader example measures a bounded set of real backbone experts without GPU allocation or full-model loading. It reports catalog time, synchronous staging-read time, total loop time including SHA-256, useful TP bytes, Linux I/O deltas and a digest of the six native packer inputs. Optional `--evict-read-ranges` advises eviction only for selected immutable expert ranges before timing; actual `read_bytes` distinguishes storage traffic from cached reads. Optional `--prefetch` mirrors the production four-expert lookahead.

The completed `dba1be0a40aa45a94ad051997016db3960a90277` checkpoint was probed at layer 0, experts 0–63, TP rank 0 on raptor and ostrich. Every configuration produced staged SHA-256 `d634d8f7830bb15dbdb6e7e3f248e51eebef18a00cc225ecbb776f756efcd8ca` for 300,810,240 useful bytes. The initial raptor runs were entirely page-cache reads and are not NVMe evidence.

On ostrich, the final cold-range baseline at 64 scratch rows took 0.467 seconds in staging reads and 0.988 seconds including hashing, with 568,283,136 actual storage bytes and approximately 601.6 MB returned by reads. Four-expert read-ahead took 0.430 seconds in staging reads and 0.948 seconds including hashing in the corresponding probe, with 564,432,896 storage bytes. These small samples show correct staging and modest improvement, not a qualified full-model startup speedup.

Three alternating read-ahead runs per scratch size gave median staging times of 0.447 seconds with 64 rows and 0.429 seconds with 512 rows; median loop times including hashing were 0.970 and 0.953 seconds. Production retains the existing 64-row scratch because the larger buffer's modest gain does not establish a useful full-loader tradeoff. Probe hashing adds CPU work between reads and can overlap prefetched I/O, so synchronous-read time alone is not an end-to-end loading metric.

Production `ExpertWeights::load` now advises up to four future experts while current staging/packing runs. Advice is restricted to the rank's W1/W3 row slices and each W2 column slice's physical source span, plus corresponding scales; it does not map unrelated experts or engram tables. This bounds the lookahead window, not total OS page-cache residency, and adds no GPU or explicit host staging allocation.

W2 column slicing still reads four times its useful bytes, making all-six-tensor source reads roughly twice useful TP bytes. The probe remains well below the approximately 5 GB/s NVMe target and does not include CUDA packing, all-layer startup or graph setup. Next loading work needs end-to-end measurements and a way to reduce/overlap strided W2 source traffic, potentially through a validated reusable native TP cache without changing checkpoint numerics.

The updated loader regression prefetches then verifies every byte against independent synthetic TP slicing for all four ranks and full dSpark staging; it passes. The daemon builds with the production lookahead. The Spark probe was built against refreshed current core/loader sources after stale build timestamps initially reused an older core artifact. No GPU numerical qualification or full-model performance claim follows from this read-only probe.

Example command after building the loader example in release mode:

```bash
v41_load_probe SNAPSHOT 0 64 0 64 --evict-read-ranges --prefetch
```
