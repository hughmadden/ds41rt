# Official-weight comparison with the deployed expert kernel

The fused slice candidates now run against the actual deployed Spark library (`505ac7f8…`, b12x `6adffce`) using identical GPU weight buffers and captured FP8 inputs. Candidate source is b12x `8e46d1c`. Qualification covers rank-zero TP4 shards of official layers zero and one, two captured input sets, and row counts 1, 2, 6, 16, 80 and return to one.

Each input set contains eighty rows. Together they reference 181 experts; only those shards are loaded into the full 384-expert address space. The existing native CUDA packer transforms checkpoint bytes on GPU. The native baseline matches all sixteen required earlier Rust-owner route-file comparisons exactly, including the repeated one-row checks. This verifies the baseline binding, input representation and loaded weight layout before comparing candidates.

## Numerical result

All seventy-two candidate checks pass: twenty-four executions per width across both layers. The final run uses the actual native compact BF16 reducer for both arms.

| Width | Maximum FP32 route relative L2 | Maximum route absolute difference | Changed compact BF16 elements | Maximum compact relative L2 |
|---|---:|---:|---:|---:|
| 64 | 1.113e-8 | 2.980e-8 | 2 / 2,170,880 | 5.914e-8 |
| 128 | 0 | 0 | 0 / 2,170,880 | 0 |
| 192 | 9.235e-9 | 5.960e-8 | 1 / 2,170,880 | 2.957e-8 |

Element totals include repeated one-row executions. Width 128 preserves the existing FP32 route arithmetic exactly on these cases. Widths 64 and 192 change the FP32 summation partition and occasionally the final BF16 rounding. They are not bit-identical replacements. These small component differences do not establish full-model token equivalence or close the separate compact-return logit discrepancy.

The qualifier enforces FP32 relative L2 below `1e-6`, compact relative L2 below `1e-5`, finite outputs, and exact FP32 equality for width 128. Graph replay uses stable allocation. All tests passed substantially below the nonzero thresholds.

## Diagnostic timing

On ostrich GB10, each arm receives twenty-four samples of ten graph replays. All twenty-four arm orderings are exercised, shuffled with seed 41. The following ranges span the two layers and two captured inputs; repeated one-row checks are included.

| Rows | Native median range µs | Selected candidate median range µs | Candidate width |
|---|---:|---:|---:|
| 1 | 206–208 | 128–141 | 64 |
| 6 | 669–932 | 500–561 | 192 |
| 80 | 4358–4572 | 3153–3411 | 192 |

**The candidate timings exclude route preparation.** They include fused compute and ordered FP32 slice reduction back to the original route order; native timing includes its internal route planner. Weight loading, transfers, compact reduction, network and coordinator work are excluded from both timing arms. Thus this establishes compute-side headroom, not a complete worker or API speedup. There is no per-arm clock/throttle admission or measured DRAM traffic, and the native six-row timing varies noticeably between runs.

Native uses its deployed capacity-one specialization for one row and capacity eighty otherwise. Candidates use fixed metadata capacities of 6, 96 or 480 groups for request capacities 1, 16 or 80. Unused groups are masked, and candidate metadata is prepared by fixture code outside the measured graph. The eighty-row captures expose 145–156 M16 groups; they do not qualify the 8k-prefill target.

## Reproduction and next step

`python/tools/qualify_v41_expert_slices.py` is the reusable official-checkpoint qualifier. It accepts `--snapshot`, `--native-lib`, `--inputs`, `--layer`, `--output` and optional `--baseline` Rust-owner files. It verifies the pinned candidate source, native geometry and checkpoint tensor shapes/dtypes, records source/input/library hashes, and rejects missing or mismatched required baseline files. `python/tools/_v41_expert_native.py` supplies the ctypes bindings. Run it in the Spark development image with the checkpoint cache and retained artifacts mounted.

[Complete records, hashes, numerical checks and raw timings](ds41-expert-official-slice-comparison.json) are retained. Two unused imports were removed from the qualifier after these runs; kernel, native binding and executable behavior are unchanged.

Next implement native route preparation and dispatch, include their cost, qualify all TP ranks and run paired live prompts before rollout. Keep width 128 as an exact arithmetic comparison while pursuing the faster 64/192 candidates. Serving retains the previous kernels; the worker and both APIs were restored after qualification.
