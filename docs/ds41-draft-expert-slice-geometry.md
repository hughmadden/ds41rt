# Local dSpark expert slice geometry

The selected RTX exporter already uses W4A8 MX compute: BF16 input is quantized
inside its persistent kernel. BF16 pointers in the node trace do not imply W4A16.
The [draft profile](ds41-dspark-node-profile.md) attributes about 4.04 ms of
instrumented GPU work per draft to its three local expert kernels. The persistent
launch has 188 CTAs; it is not a single-CTA launch.

The Spark fused-slice implementation previously hardcoded intermediate 576 and
packed width 640. Local dSpark has intermediate 2304, 128 experts and top-3
routing. b12x now accepts immutable intermediate geometry in the fused kernel
and ordered slice reducer. Expert weight/scaling strides, gate-half offset,
FC2 K tiles and logical masking derive from the 128-aligned packed width.
Live rows and group counts remain runtime values. Defaults retain Spark geometry.
No policy, native export or serving dispatch has changed.

Widths 64/128/192 use 36/18/12 slices respectively for dSpark, with no tail
padding. This creates a candidate for measurement, not a selected tiling.

## Qualification

On RTX GPU0 (`GPU-fe5b6dd0-a77c-c8fb-6360-e1b9d9918ac0`), 26 tests pass across
single-expert arithmetic, grouped arithmetic, GPU routing metadata and ordered
slice reduction. Both 576/top-6 and 2304/top-3 geometries are covered. Grouped
dSpark uses a 128-expert pool, including expert 127, changed metadata and weights
on captured replay, shared runs through 80 rows, and draft row counts 5/15/40.
The 40-row case uses 120 distinct experts/groups. One compiled callable and
fixed allocations serve changing live counts; output guards and zero cases pass.
Arithmetic is checked against the existing quantization-aware FP32 oracle;
this does not assert FP32 bit equivalence with the persistent implementation.
Ordered reduction matches its sequential FP32 oracle exactly.

The expanded draft40 case initially exceeded the test's old 64-group metadata
allocation. The fixture now reserves its worst-case route capacity. All three
widths pass after that fixture correction. Compute Sanitizer memcheck reports
zero errors for grouped width192/intermediate2304/top-3, including all nine
routing cases. [Case results and source/log hashes](ds41-draft-expert-slice-geometry.json)
record the evidence; raw logs are under `/tmp/ds41-draft-expert-slices/`.

Run from the repository root:

```sh
docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES=0 \
  -v "$PWD/third_party/sparkinfer:/src:ro" -w /src --entrypoint python \
  ds41rt-coordinator-dev:latest -m pytest -q -s -p no:cacheprovider \
  tests/moe/test_v41_fused_slice.py tests/moe/test_v41_grouped_slices.py \
  tests/moe/test_v41_route_plan.py
```

For the sanitizer case, use entrypoint `/usr/local/cuda/bin/compute-sanitizer`
and arguments `--tool memcheck --error-exitcode 99 python -m pytest -q -s
-p no:cacheprovider 'tests/moe/test_v41_grouped_slices.py::test_grouped_slices[192-2304-3]'`.

## Next decision

Compare the complete RTX path with official dSpark weights: BF16 input
quantization, GPU route planning, fused slices and ordered reduction. Include
5/15/40 rows, shared and dispersed routing, stable graph replay and changed
inputs. Select a b12x plan only after correctness and balanced timing; then
qualify full-model output and C1/C6/C16 serving. Component kernel times alone
cannot establish a decode improvement. Live sampler-era APIs remain selected.
