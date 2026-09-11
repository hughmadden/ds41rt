# Full b12x upstream merge, 2026-09-11

Merged all 44 upstream commits through `81544c5b76f869f2921b7f9f0438505846307f70`
into b12x master. The merge commit is `ab88dde04baed1047bcd51f97cc72eaf7005249a`,
with our previous `b3344c9c98c6ef6363a405cb7d8cad77fe09d13f` as its first parent.
This is a full ancestry-preserving merge, not a selective import.

## What upstream adds

The changes include actual CuTe kernels and serving primitives, not just
framework integration. Their recent V4.1 work includes adaptive BF16
SIMT/tensor-core projections, bounded MXFP4 index scoring, paired attention
dequantization, dtype-aware PCIe DMA planning, and prepared-prefix Engram
hashing/lookup. Earlier incoming commits include paged top-k selection and
NVFP4 prefill improvements.

These implementations are now available for measured comparisons. Their
reported serving gains use a four-RTX PCIe/vLLM configuration and cannot be
transferred to our RTX coordinator plus four RoCE-connected Sparks. Some of
their large projection gains compare against a small-row SIMT path used for
prefill. Our WO-A already uses grouped FP8 GEMM and fused inverse-RoPE
quantization. Their paired attention optimization targets a different cache
representation from our FP8 resident cache. Their V4.1 expert recipe retains
materialized M64 execution; our native fused slices remain selected.

Use coordinator preparation profiling to choose the next integration. Useful
comparison candidates include compressor projections and large-context
indexer/top-k work. Engram prefix bounding is useful to audit, but is not a
fused gather/dequant/projection implementation.

## Merge resolutions

Resolved 28 conflicted files, preserving the following contracts:

- Native grouped AOT exports, inverse-RoPE quantization, packed wire inputs,
  route planning, and fused expert slices.
- K128 activation compatibility alongside upstream's K32 V4.1 floor. The
  public `min_amax` spelling maps to the existing canonical `amax_floor`;
  conflicting explicit values fail rather than silently changing quantization.
- Upstream `deepseek_v41` materialized expert recipe alongside the native
  `silu_v41` fused recipe, with their distinct launch requirements.
- Explicit V4.1 heterogeneous cache geometry alongside legacy NVFP4 cache
  support. GPU tests exposed and fixed an incorrect legacy gather selection
  for V4.1 compressed pages during the merge.
- Capacity-bound dispatch, caller-owned scratch, FP32 split reduction, and
  upstream immutable-input-scale options.

The new upstream GDN planned operation lacked a catalog registration. Added
its existing policy/generator and empty, schema-correct coverage entries;
unmeasured devices/queries therefore use the heuristic. This repairs the
library registry and does not add a DS41RT profiles feature. The only binary
profile component with independent edits on both branches was
`gemm.blockscaled_precision` on RTX: upstream's measured coverage replaces
our empty-geometry placeholder.

## Qualification

- Final policy/API tests: **315 passed, 1 skipped**.
- Native V4.1 GEMM, activation-floor/wire quantization, and grouped-slice
  checks: **32 passed** in the first GPU run. Three upstream expert wrapper
  failures were fixed and passed in the next run.
- Upstream expert, block-FP8 linear, and attention run: **67 passed** before
  three heterogeneous prefill failures exposed the gather-selection conflict.
  After its fix, compressed attention and dual-cache prefill: **42 passed**.
  These counts overlap and should not be added as unique tests.
- Native release rebuild and both CTests passed.
- All **133 exported object files** (7 expert/coordinator and 126 FP8) are
  byte-identical to the previous build. The native shared library is also
  byte-identical: SHA-256
  `2d5dd6611b2eb34a7a11eaabb4a34b9096c41e297e28050d491141061949fe72`.
- Real-weight native qualification passed: **32** inverse-RoPE/quantization
  cases and **174** grouped projection cases across all 43 weight sets.
- Target and speculative API smoke, streaming, cancellation/recovery, and
  unsupported-sampling checks passed.
- All eight paired quality cases preserve baseline text and usage exactly.
  The inherited Unicode-only-number formatting failure remains: **5/6**
  strict objective checks pass. The quality script consequently exits 1;
  this is not a clean broad-quality pass.

Three counting streams measured median 31.17 target and 103.43 speculative
tokens/s. These are short, predictable smoke measurements, not evidence of a
merge speedup or representative throughput. Spark workers retain their
qualified artifacts; no expert transport or deployment change is needed for
this source merge. Both coordinator APIs run the rebuilt native artifact.

The entire b12x GPU suite and every supported model/device combination were
not run. A pre-existing undefined Trellis rank-LUT helper in `intrinsics.py`
remains outside the selected native V4.1 path; this merge does not claim to
repair or qualify it.

Raw build, API, and native qualification artifacts are in
`/tmp/ds41-upstream-artifacts`; merge/test logs are in
`/tmp/ds41-upstream-merge`. The compact artifact manifest is
`ds41-b12x-upstream-merge.json` beside this note.
