# Intermediate-slice splitting: useful at low expert counts, mixed decode unresolved

An isolated CuTeDSL candidate splits each expert task into five 128-wide intermediate slices over the current padded-640 resident layout. Each slice performs fused FC1/activation/FC2 once and writes its own FP32 route plane. A second CuTe kernel sums the five planes in their original order. This avoids the current one-row output split's repeated FC1 work, at the cost of additional partial-output scratch and traffic. It does not change the resident weight layout or test 64/192 tilings.

All eight synthetic cases pass exact FP32 route comparison against the existing implementation, changed-input/route graph replay, invalid-route zero output and stable replay allocation. The BF16 fixture also exercises the independent quantization-aware oracle. The FP8 fixtures compare their initial prequantized baseline against BF16 before comparing the split candidate. These are component checks with synthetic weights, not official-weight owner or live API qualification.

## Native FP8 consumer measurements

Ostrich GB10, 48 SMs, root `1aceae0`, b12x `6adffce`. Twelve alternating-order samples per arm, twenty warm graph replays per sample. Candidate timing includes the additional scratch clearing and ordered reduction; input encoding, transport, compact return and coordinator execution are excluded.

| Request rows / routing | Active experts | Baseline median µs | Candidate median µs | Candidate latency change |
|---|---:|---:|---:|---:|
| 1, six distinct experts | 6 | 212.21 | 152.87 | −28.0% |
| 2, all six experts shared | 6 | 451.83 | 209.24 | −53.7% |
| 6, mixed sharing | 22 | 685.70 | 714.02 | +4.1% |
| 80, random routes | 276 | 7424.40 | 7334.25 | −1.2% |

The six-row fixture has twelve M1, seven M2, two M3 and one M4 experts. It matches the live workload's median active-expert count, not its full distribution. The M2 case deliberately isolates complete sharing; it is not a representative request-frequency estimate. The random 80-row case has substantially more active experts than the live 80-row median of 86, and does not qualify large prefill.

The earlier BF16 diagnostic measured 208.23→141.02, 456.86→217.61, 759.37→718.43 and 7385.87→7389.72 µs for the same four shapes. Do not treat the difference between these separate BF16/FP8 runs as an input-format performance comparison. Within-arm samples also show variability; there was no per-arm clock/throttle admission or DRAM-counter collection. These are diagnostic results, not formal release performance evidence.

## Decision and remaining work

Do not replace the common speculative-decode path wholesale with this candidate. Low active-expert counts benefit substantially, but the FP8 six-row case does not. Expert-local M alone is insufficient to choose a task split: the number of active experts also determines available parallel work.

The prototype allocates five extra FP32 route planes in addition to the existing route output: 614,400 bytes per request row, or 49,152,000 bytes at capacity eighty. FC1 activation intermediates remain fused; partial FC2 outputs are newly materialized. This scratch cost and its reads/writes argue against extending the prototype indiscriminately to prefill.

Next, compare 64/192 intermediate tiling and lower-traffic scheduling on the mixed-expert FP8 workload. Retain this candidate for low-expert-count comparison, qualify useful changes against official weights and the native owner, and measure actual memory traffic before attributing gains to weight bandwidth. Resident layout selection should favor common speculative decode, with a separate large-prefill guardrail.

Serving binaries and source remain unchanged. Ostrich's worker and both dev APIs were stopped temporarily for GPU allocation capacity and restored after the experiment. Exact source diffs, probe bodies, raw samples, hashes and reproduction details are embedded in the [evidence](ds41-expert-intermediate-split-probe.json); temporary overlays remain outside the repository.
