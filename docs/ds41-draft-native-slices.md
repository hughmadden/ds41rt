# Native RTX dSpark expert slices

The development APIs now use the exported N192 fused slice pipeline for local
dSpark experts. The native expert ABI still accepts BF16 rows and returns FP32
route planes; Rust ownership, routing, shared experts and output reduction retain
their existing contracts. Backbone Spark workers are unchanged.

The b12x pipeline performs K32 FP8 input quantization, publishes the live count,
groups routes, executes fused FC1/SwiGLU/FC2 slices and reduces slices in order.
Quantization uses the same subgroup-four and `1e-4` floor as the native standalone
quantizer. Temporary FP8 rows occupy a separate preallocated scratch region;
there is no allocation or host synchronization during replay. Expert weights
retain the resident GPU-packed representation. N192 is the selected candidate,
not proof of globally optimal tiling; N64 still deserves a live-routing comparison.

## Qualification and measured outcome

The native ABI passes official-weight checks for all three dSpark stages at
1, 5, 15, 40 and 80 rows with both shared and dispersed routing. Changed inputs,
IDs and routing weights replay through the same graphs. Sixty native-output
comparisons pass against the persistent baseline. Final checks on every stage
also require exact FP32 route equality against the Python-composed N192 path.
Small BF16 differences from the persistent baseline remain at rounding boundaries.
Output tail guards and stable allocation checks pass. Native memcheck reports
zero errors; the full-model lane fixture passes in 11.04 seconds.

The complete native expert path for five dispersed rows takes about 193.5 us
versus 1167.7 us for the persistent kernel in the bounded component test. At
40 dispersed rows it takes about 1434.7 versus 1661.6 us. This includes input
quantization, route planning, ordered slice reduction and the common top-3 BF16
output reduction; router and shared-expert GEMMs are excluded from both arms.

Short greedy counting API results (599 output tokens/request):

| Arm/order | C1 TPS | C6 aggregate TPS | C16 aggregate TPS |
|---|---:|---:|---:|
| Candidate first | 142.57 | 347.58 | 673.85 |
| Baseline first | 134.58 | 316.37 | 632.89 |
| Baseline warm | 133.75 | 329.69 | 634.19 |
| Candidate warm | 146.23 | 370.63 | 691.24 |

All 92 performance responses have identical text and usage. C2/C6/C16
cancellation/replacement qualification and the API contract check pass. Eight
paired quality prompts retain identical text and usage; the pre-existing
Unicode objective failure remains in both arms. Both promoted endpoints pass
the post-rollout API checks. Broad quality and the 90/270 TPS release goals remain
open; these counting results do not establish code/prose or long-context speed.

## Source and reproduction

b12x `47029c50` adds the exportable BF16 draft pipeline. The integration exporter
accepts `--role coordinator`; CMake permits that role with an explicit
`-DDS41RT_V41_EXPERT_SLICE_WIDTH=192`. Coordinator atomic token output remains
rejected. Spark-role defaults and the existing expert ABI are preserved.
The default release backend/policy selection is still separate open work; the
live development library was built with the explicit N192 selection.

The selected immutable library is
`/tmp/ds41-draft-native-slices/selected-native/libds41rt_native.so`, SHA256
`1df5f6c5fd61778087fd62c21a7335421400c5bdd3631e4f783158376013f08d`.
Rust daemon remains `/tmp/ds41-engram-slots/daemon`. Live containers are
`ds41-draft-slices-live-target-api-dev` (18041) and
`ds41-draft-slices-live-spec-api-dev` (18042). Previous sampler-era containers
are stopped and available for rollback. Launch arrays and promotion/rollback
logic are retained in `/tmp/ds41-draft-native-slices/`.

Use `python/tools/qualify_v41_draft_expert_slices.py` with `--native-lib BASELINE
--candidate-lib CANDIDATE --candidate-width 192 --rows 1 5 15 40 80`, plus the
official snapshot, stage and output arguments. It compares both native libraries
and all three composed widths. `--no-timing` performs numerical/replay checks.
Full-model, build, API, concurrency, quality, promotion and sanitizer commands
and raw logs are retained under `/tmp/ds41-draft-native-slices/`.
[Summary, checks, raw timing samples and hashes](ds41-draft-native-slices.json)
record the evidence. The preceding [composed comparison](ds41-draft-expert-slice-comparison.md)
explains the controlled reuse patterns and measurement limits.
