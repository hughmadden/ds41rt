# DS41RT v3

V3 adds automatic one or two RTX PRO 6000 Blackwell serving and makes the dual
RTX layout a complete release path. It preserves the official DeepSeek V4.1
Flash checkpoint, four DGX Spark expert workers, and the OpenAI-compatible API
from the earlier releases.

## Highlights

- `run.sh` discovers usable peer-connected RTX cards, selects two when the
  requested memory plan fits, and falls back to one. `--rtx-gpus 1|2|auto`
  provides an explicit override.
- Dual mode hosts encoder routed experts 0–19 and every shared expert TP2 on the
  RTX pair. The four Sparks retain decoder routed experts 20–39, cutting their
  resident expert set in half.
- RTX0 owns embeddings, vision, encoder attention/SWA, and compressed-cache
  sources 2, 8, and 14. RTX1 owns decoder attention/SWA, source 20, and dSpark.
  Attention remains beside its source to avoid PCIe cache traffic.
- The two decode lanes use owned workspaces and cooperative GPU completion.
  Unconstrained greedy decoding performs parallel top-1 selection on the GPUs
  instead of returning the full vocabulary to the host.
- One-RTX dSpark defaults to placement-aware adaptive K5. Explicit K7 and the
  legacy cost profile remain available for comparison and tuning.
- The dual layout provides a 13.094 GB FP4 compressed source pool for 14,680,064
  logical tokens plus 32,768 private-tail tokens. FP8 SWA, 24 completed-turn
  snapshots, and 24 prompt snapshots remain part of the qualified policy.
- The standard launcher serves on port 8000 and exposes concurrency, KV-pool,
  memory-reservation, prefix-retention, context, output, and prefill controls.

## Demonstrated impact

- Cold +1K prefill increased from 3,084 to 4,850 tok/s, and cold +2K increased
  from 4,078 to 6,471 tok/s.
- C16 code aggregate decode increased from 992.78 to 1,181.49 tok/s; topic from
  514.96 to 596.14 tok/s; and counting from 1,161.12 to 1,333.57 tok/s.
- C16 mixed aggregate throughput increased from 196.46 to 309.06 tok/s.
- Weighted target-only decode increased from 43.04 to 46.14 tok/s, while
  weighted dSpark decode increased from 76.72 to 79.33 tok/s.
- Weighted dSpark decode over a verified 256K retained prefix increased from
  67.39 to 78.08 tok/s. All 240 retained requests verified exact cache reuse.
- Clean restart to API readiness decreased from 58.05 to 46.00 seconds.

Every local performance cell uses three fresh samples from the same release
candidate. Measurements use an enforced 400 W RTX power limit and standard
14,001 MHz maximum memory speed with no memory overclock.

Read the [README](https://github.com/tpurtell/ds41rt/blob/v3/README.md), [v3
performance report](https://github.com/tpurtell/ds41rt/blob/v3/docs/phase2-release-performance.md),
[implementation and evidence log](https://github.com/tpurtell/ds41rt/blob/v3/docs/phase2-dual-rtx.md),
and [completed checklist](https://github.com/tpurtell/ds41rt/blob/v3/docs/release-v3-checklist.md).

Container images:

- `ghcr.io/tpurtell/ds41rt-coordinator:v3` (`linux/amd64`)
- `ghcr.io/tpurtell/ds41rt-spark-expert:v3` (`linux/arm64`)

The `v3` and `latest` tags are identical for each role.
