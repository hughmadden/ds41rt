# DS41RT test-coverage port — status and method

Branch `test-coverage/upstream-port` ports unit-test coverage from vLLM, SGLang and
llama.cpp to ds41rt. Everything here runs on a GPU-less host (pg); anything needing
the coordinator, Spark workers, or real weights is recorded in `DEFERRED.md` and is
explicitly forbidden from running while the fleet is live.

## Documents

| File | Contents |
|---|---|
| `UPSTREAM-INVENTORY.md` | Full categorized census of every test file in the three upstream repos, with a relevance verdict per row. |
| `PORT-PLAN.md` | The PORT-CPU rows grouped into migration tranches by ds41rt target component. |
| `DEFERRED.md` | Relevant tests that require GPU/fleet state, keyed to the condition that unblocks them. |
| `BASELINE.md` | Pre-port test baseline on pg (what passed/failed before any port landed). |

## Method

1. Blobless shallow clones of `vllm-project/vllm`, `sgl-project/sglang`,
   `ggml-org/llama.cpp` (HEAD, 2026-09-14 AEST) into `~/dev/upstream-test-inventory/`.
2. 25 parallel census agents read every test file and produced one row per file (or
   one rolled-up row per subdir for homogeneous GPU/hardware dirs): subsystem, CPU
   viability, behaviour pinned, verdict, ds41rt target component.
3. Verdicts:
   - `PORT-CPU` — relevant to ds41rt AND runnable on a CPU-only host. These are
     migration candidates.
   - `DEFER-GPU` — relevant, but needs GPU weights/kernels/multi-node. Recorded in
     `DEFERRED.md`; never run against the live fleet from this branch.
   - `SKIP-*` — not applicable (other models, training infra, vendor hardware,
     benchmark scaffolding, duplicates).
4. Ports are *rewrites, not copies*: the upstream test's invariant is expressed
   against ds41rt's own APIs (rust crate tests, python reference tests). The
   upstream file path is cited in each ported test's doc comment.
5. Every ported suite runs on pg. Failures are triaged: genuine product bug → minimal
   fix on this branch; test/harness artifact → fix the test. No fleet contact at any
   point.

## ds41rt component map (target column)

C1 OpenAI-protocol/API surface (rust `ds41rt-api`, python server) ·
C2 constrained/structured output (native xgrammar adapter, `api/constrained.rs`) ·
C3 sampler semantics (coordinator sampling, `request_sampling_params`) ·
C4 tokenizer/detok/chat templates (`ds41rt-loader`, HF tokenizers) ·
C5 KV cache, prefix reuse, disk persistence (`ds41rt-hostcache`, native `v41_kv`) ·
C6 speculative decoding / dSpark (native `v41_dspark`, `core/dspark_policy.rs`) ·
C7 MoE routing / expert placement (native `v41_router`, `index_topk`) ·
C8 scheduler / admission / request lifecycle (`ds41rt-daemon`) ·
C9 transport / RPC / failure injection (`ds41rt-transport` protocol_v2) ·
C10 weight loading & quant formats FP4/FP8 (`ds41rt-loader`, python validators) ·
C11 vision / multimodal (native `v41_vision`) ·
C12 tool-call parsing (native `xgrammar_v41_tools`, `api/tooling.rs`).
