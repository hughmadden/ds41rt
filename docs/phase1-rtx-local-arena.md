# Shared local scratch: five resident layers

Each local lane now owns one arena sized to its largest kernel variant. An
execution drains before another variant can reuse that arena; the two lanes
remain independent. The full-width kernels overwrite live scratch themselves,
so no reset is added to the token loop. This reduces both lanes' local workspace
from 613.34 MiB to 325.25 MiB and permits layers 0–4 with the same 2 GiB runtime
headroom, 18 × 1,048,576-token source capacity and 24 snapshot entries.

The isolated probe passed 45 poisoned-arena transitions across capacities
1/16/80/256/1024/4096, with live rows up to 80. Ordered outputs match separate
arenas and independent grouped references exactly. Atomic outputs differ from
separate arenas by at most about 2e-7 relative L2, consistent with accumulation
order. Captured replays allocate no storage. These checks include returns from
atomic variants to ordered decode and high expert IDs through 383.

| Initial adaptive serving comparison | Remote experts | Five RTX layers |
|---|---:|---:|
| Code median tok/s, three samples | 120.59 | 121.21 |
| 32K prefill median tok/s, three measured samples | 7,917.25 | 7,704.18 |
| C4 aggregate tok/s, one batch | 115.82 | 118.38 |
| C16 aggregate tok/s, one batch | 185.88 | 180.26 |
| Startup to readiness, one launch | 4.33 s | 9.36 s |
| Peak sampled GPU memory | 61,966 MiB | 96,844 MiB |

All code, prefill-cache, mixed-request and retained-context/cancellation checks
passed. The eight prefill responses were `7`, with identical prompts, 32,768
new tokens and zero cache hits. The standard service was restored. The local
peak left about 1 GiB free according to NVIDIA's reported physical memory.

This is not performance acceptance: prefill fell about 2.7% and C16 about 3%
in sequential arms, and startup increased. Local placement stays opt-in.
Next investigate the unnecessary remote-request host download on local layers
and group-boundary loading waits. Balanced timing, target-only performance,
quality and wider context evidence remain required before changing defaults.

[Evidence](phase1-rtx-local-arena.json) contains raw rates, lifecycle results,
source/artifact hashes and probe results. Reproduce the scratch test with
`python/tools/qualify_v41_local_arena.py --native-lib LIB` on the selected RTX.
The serving runner, full results and memory samples are under
`/tmp/ds41-rtx-local-arena`; no builds ran during its benchmarks.
