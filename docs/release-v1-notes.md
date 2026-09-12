# First-release notes draft

Publication is pending the [release checklist](release-v1-checklist.md). The text
below is the prepared feature summary; final artifact references and report links
must be added from the qualified release. This file does not announce a release.

---

DS41RT brings native DeepSeek V4.1 Flash inference to an RTX PRO 6000 Blackwell
coordinator with four DGX Spark expert workers, with a focus on agentic workloads
and efficient conversation reuse.

Key features:

- Target-only inference and local dSpark speculative decoding, with routed
  experts distributed across four Spark workers.
- An OpenAI-compatible API with streaming, high-effort reasoning by default,
  tool calling and constrained output using supported JSON schemas.
- Token-based prefix caching, bounded sliding-window replay and retained
  completed turns for fast conversation continuation.
- Architectural FP4 compressed KV storage, FP8 sliding-window KV and optimized
  native attention kernels.
- Native vision support for up to sixteen images per prompt.
- Configurable concurrency, retained-turn limits, context/output limits and
  KV/memory budgets; defaults target sixteen active requests plus eight spare
  context equivalents and twenty-four retained completed turns.

See the README for the deployment topology and configuration. Performance
reports identify the measured hardware, power limits, memory settings and
workload conditions.
