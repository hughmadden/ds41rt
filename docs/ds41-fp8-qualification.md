# Native 32x32 FP8 qualification

SparkInfer revision `09b39aa62b3ea36b8b6397bf58fed0f63ff19db3` adds the official checkpoint's 32x32 UE8M0 scales to the block-FP8 GEMM path without changing its FP8 weight bytes.

K32 scales remain independent; the replicated-K128 shortcut is enabled only for legacy 128x128 weights.

Native 32x32 operations request FP32 split accumulation with one final output rounding, and the split reducer now sums all four planes when the planner selects four slices instead of silently reducing only two.

## Correctness evidence, 2026-09-10

The new nine-case suite covers M=1,4,16,80,256, distinct adjacent N32/K32 scales, N=25600/K=6144 projections, bound and opaque execution, mutated CUDA graph replay, unchanged replay allocation, unchanged checkpoint bytes, and two/four-plane reduction.

The independent numerical oracle explicitly rounds activations by K32 and multiplies recovered operands with TF32 disabled; tolerances are rtol=0.01, atol=0.02, with cosine similarity above 0.99999.

| Device | GPU UUID | Results |
| --- | --- | --- |
| RTX 0 | GPU-fe5b6dd0-a77c-c8fb-6360-e1b9d9918ac0 | 9 new and 9 existing block-FP8 tests pass |
| RTX 1 | GPU-95f8f212-9131-df99-fd53-7535965197d7 | 9 new tests pass; 240 planner tests pass after correcting a stale activation-scale compile-key assertion |
| ostrich GB10 | GPU-a7c503f0-bb16-4333-4f44-92e331dbcf2e | 9 new tests pass |
| dodo GB10 | GPU-c08bcbd8-eb8e-1353-5e59-ede0a4dc974c | 9 new tests pass |

Qualification used the existing development images as tool environments, with modified SparkInfer source mounted read-only and PYTHONPATH pointing to that mount; this does not qualify the new ds41rt containers.

The environments provide CUTLASS DSL 4.6.2 and NVIDIA PyTorch 2.12.0a0+5aff3928d8.nv26.5.50603568.

Run `python3 -m pytest -q tests/gemm/test_gemm_block_fp8_v41.py` from the pinned SparkInfer tree in the CUDA environment; add `tests/gemm/test_gemm_block_fp8_linear.py` for legacy regression coverage and `tests/gemm/test_dense_gemm_expected_m.py` for planner coverage.

## Open gates

The inherited legacy test `test_block_fp8_linear_live_m_does_not_resolve_new_dense_kernel` fails numerically on ostrich when warming M=4096 then executing M=1824 with K=128/N=1536; a separate pristine checkout of revision `17f04b33e8e419caab8f845bfc9fe3c0c5328628` reproduces the failure.

That variable-M Spark issue remains unresolved and must be addressed before integrated prefill qualification; these passing fixed-shape checks do not close it.

Graph replay is allocation-stable, but FP32 split workspace is currently allocated during eager launches/capture and still needs caller-owned capacity planning for the integrated serving path.

No throughput claim, full-model numerical agreement, or end-to-end serving qualification is implied by these kernel checks.
