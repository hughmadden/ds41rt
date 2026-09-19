# WO-B capacity16 ordered split2 candidate

Prepared 19 September 2026 AEST. Default disabled; CPU contracts only. This is
separate from the rejected `codex/afd-wob-split1-aot` arithmetic candidate.
Both profiles remain opt-in and cannot be enabled together.

The frozen actual-operand oracle found original capacity1 split2 correct for all
5120 BF16 outputs, while capacity16 split1 rounded column2908 down incorrectly:
exact dot `2000684961/4294967296`, correct BF16 `0x3eef`, split1 `0x3eee`.
That evidence rejects selecting split1 merely to make row shapes agree.

`--wob-m16-split2` changes exactly WO-B K8192/N5120 at planned capacity16.
The b12x fork owns tile16x64, K128, two K partitions, direct one-M scheduling,
and FP32 partial planes. This is the original M1 MMA geometry and K association.
Capacity16 uses normal all-row TMA activation transfers: the M1 specialized
loads and stores hard-code row0 and must not be reused for a batch. The existing
native ordered FP32 reduction performs the final BF16 rounding. No persistent
scheduler or runtime dispatch is rewritten. The constructor rejects capacities
outside 1..16, mismatched expected capacity, groups, geometry, or non-SM170 plans.
The exporter applies the override only at capacity16; M1 and >=80 compiler calls
are unchanged. Live M remains a runtime scalar bounded by the native plan.

This intentionally changes small prefills as well as multirow decode/verification
that select capacity16. It does not establish invariance against >=80 prefill
plans. Original end-to-end goldens and failure receipts must remain preserved;
any new prefix baseline needs oracle classification, not silent replacement.

Use the pinned `third_party/sparkinfer.lock.json` source. CPU checks, no installs:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 /b12x/tests/gemm/test_native_aot_small_m.py
PYTHONDONTWRITEBYTECODE=1 python3 /b12x/tests/gemm/test_native_aot_split1.py
PYTHONDONTWRITEBYTECODE=1 python3 python/tests/test_v41_fp8_aot_options.py
```

The component script now selects this candidate:

```sh
DS41RT_SOURCE_REVISION=<this-commit> DS41RT_SPARKINFER_SOURCE_DIR=/b12x \
  bash /source/scripts/build-v41-wob-component.sh /build/component
```

Reuse dev image
`sha256:d62e7d302bded8b667464003e0210a11da5675c1da42ba368cea5bad7fcffb45`,
with `/source` and `/b12x` read-only, task-owned `/build`, network disabled,
4 CPUs, 16GiB memory, and an exclusive Romeo GPU window. No dependencies are
installed. The script verifies pinned source, CPU tests, actual SM170 geometry,
exports all six WO-B capacities, and links only the component ABI. It records
artifact hashes, toolchain, scratch sizes and policy in `PROVENANCE.json` and
`aot/v41_fp8.json`. Capacity16 adds 655360 bytes of ordered FP32 partial scratch
at the existing aligned activation-workspace end; the native planner consumes
the new manifest size before requests begin.

A component `.so` cannot replace the complete serving native library. Full CMake
builds may opt in with `-DDS41RT_V41_WOB_M16_SPLIT2=ON` and
`-DDS41RT_ENABLE_V41_FP8_AOT=ON`; preserve all other release settings from the
retained build. No coordinator AOT object cache was found in the retained release
trees, so a complete native build remains separate work after numerical acceptance.

Required GPU gate: run the frozen quantized operands at live rows1/2/4/8/16,
including distinct nonzero later rows, compare every output against the exact
oracle and inspect row0 column2908. Verify repeated outputs, stable workspace,
row bounds and timings. CPU policy tests cannot establish CUDA compilation,
correct arithmetic or performance. The current packet stops before GPU staging;
the operator must record the exact build container, cleanup it on timeout, and
restore the exact previously running coordinator on success or failure.
