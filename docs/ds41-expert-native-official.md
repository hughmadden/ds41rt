# Official-weight qualification of exported expert slices

All three slice widths now execute through the existing native expert ABI on Spark. The native graph includes row-count publication, GPU route grouping, fused expert compute and ordered FP32 route reduction. There is no Python planner or compute launch inside the candidate graph. The unchanged deployed library supplies the comparison arm and checkpoint GPU packer.

`qualify_v41_expert_slices.py --candidate-dir DIR` loads `DIR/w64/libds41rt_native.so`, `DIR/w128/libds41rt_native.so` and `DIR/w192/libds41rt_native.so`. `--rank` selects the corresponding official TP4 checkpoint slice. `--no-timing` retains numerical and replay checks while omitting performance samples.

## Evidence scope

- All 18 native synthetic replay cases pass on ostrich (six per width), each FP32-exact against its same-width separately launched grouped kernel and CPU metadata. These fixtures also check a synthetic numerical oracle.
- All 180 official candidate checks pass: rank 0 layers 0/1 plus ranks 1/2/3 layer 0, two captured input sets, rows 1/2/6/16/80/1, and three widths. Other-rank shards were executed on ostrich, not on their assigned production hosts.
- All 16 retained Rust-owner baseline comparisons for rank 0 layers 0/1 remain exact. Other ranks use the deployed native kernel as their differential reference, without an independent saved Rust-owner baseline.
- Width 128 is FP32-exact in every official comparison. Maximum route relative L2 is `1.113e-8` for width 64 and `2.380e-8` for width 192. Their compact BF16 outputs differ in four and two elements respectively across 5,427,200 comparisons per width, including repeated one-row cases. Maximum compact relative L2 is `1.806e-6` and `5.056e-7`. This does not resolve the separate whole-model compact-return rounding gate.

[Raw numerical checks, samples, library/source/input hashes and fixture results](ds41-expert-native-official.json) are retained. Rank-zero timing was collected before the final addition of the optional `--no-timing` switch; other-rank runs record the later tool hash. Both use the same exported candidate libraries.

## Diagnostic timing

Rank-zero layers 0/1 use all 24 permutations of the four arms, ten graph replays per sample, after correctness. Median ranges across both layers and inputs, in microseconds:

| Request rows | Deployed | Native width 64 | Native width 128 | Native width 192 |
|---|---:|---:|---:|---:|
| 1 | 204–208 | **127–150** | 179–183 | 146–166 |
| 2 | 449–451 | **252–292** | 276–316 | 259–296 |
| 6 | 667–914 | 580–634 | 574–645 | **514–576** |
| 16 | 1467–1748 | 1215–1370 | 1240–1492 | **1092–1262** |
| 80 | 4388–4564 | 3792–4085 | 3743–4007 | **3198–3459** |

These include the complete native expert launch sequence but exclude input encoding, compact return, transport and coordinator execution. One-row variation is visible; there is no per-arm clock/throttle admission or DRAM-counter evidence. These remain diagnostic timings, not release performance or an API speedup. No timing sweep was performed for other-rank shards.

The observed width ordering supports small-capacity and wider-speculative configurations. Selection still needs native worker/graph integration and paired live qualification across all four hosts. Large prefill, asynchronous staging, scratch reduction and full-model quality remain open.

## Export reproducibility fix

The sweep exposed executable-only cache entries being reused by AOT export, which requires compiler IR. V4.1 exporters were setting obsolete `SPARKINFER_COMPILE_*` environment variables; b12x reads `B12X_COMPILE_*`. The expert, coordinator FP8 and slice exporters now set the actual cache controls. The slice sweep subsequently exported all three widths in separate processes sharing the same container/cache and linked and ran each library successfully. This validates repeated slice/input-quantizer export; the complete coordinator FP8 export matrix has not been rerun here.

Example official comparison in the Spark development image:

```sh
python python/tools/qualify_v41_expert_slices.py \
  --snapshot /hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277 \
  --native-lib /native/libds41rt_native.so --candidate-dir /output \
  --inputs /inputs --baseline /baseline --rank 0 --layer 0 \
  --output /output/native-layer0.json
```

For other ranks omit `--baseline` and optionally add `--no-timing`. Libraries were built with the [native export/link procedure](ds41-expert-native-slices.md), using separate `w64`, `w128` and `w192` output directories. Serving binaries remain unchanged.
