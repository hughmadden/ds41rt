# Accumulated asynchronous weighted comparison

The weighted eight-type median improves 77.91 → 80.77 tok/s (+3.67%) in this
matched pair. Every case passes its throughput-oriented checks. Code and math
remain essentially flat; the other six case medians improve. This is more relevant
to release decisions than the isolated counting loss, but one pair does not
establish a universal speedup or concurrent performance.

Frozen e9c07ae and 2957772 use adaptive dSpark, independent lanes, five RTX expert
layers, identical native kernels and nonce seed 55001. Each case has three samples,
with temperature zero and thinking disabled. Runs are sequential C1 workloads;
prior mixed concurrency evidence remains separate. RTX power is capped at 400 W
with standard memory speed. The graph-fusion candidate is absent from both arms.

| Case | Before tok/s | After tok/s |
|---|---:|---:|
| code | 128.70 | 128.66 |
| math | 134.61 | 135.09 |
| fable | 53.64 | 55.29 |
| hello | 77.90 | 82.23 |
| topic | 69.86 | 71.21 |
| structured-json | 89.34 | 91.73 |
| structured-json-schema | 90.68 | 94.89 |
| multilingual | 66.37 | 71.28 |

[Evidence](phase1-async-weighted.json) preserves sample rates, launch commands
and artifact identities. Full prompts, outputs, token counts and checks remain
in the raw directory. These results precede the final clean v2 package qualification.
The follow-up receive-polling experiment must preserve real-workload performance;
recovering counting alone is not sufficient justification.
