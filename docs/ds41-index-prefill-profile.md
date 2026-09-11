# Index scoring after direct expert accumulation

This profile selects the next optimization; it introduces no serving-kernel change and claims no further throughput gain. The baseline is root `edb1d38`, four direct-accumulation Spark workers, and the split-mHC RTX native library recorded by SHA256 in the accompanying measurements.

The same 16,411-token repeated-amber counting request, target only, produced 59 completion tokens with 16.721 s to first content under host timing instrumentation. Seventeen prefill steps took 16.610 s. The 680 layer executions attributed 8.482 s to experts, 3.613 s to index production/selection, and 3.830 s to attention/projection/FFN preparation. Sparse attention alone took 3.285 s. Expert children were routing 1.304 s, dispatch 1.391 s, shared FFN 0.162 s and collection 5.622 s. Collection includes upload 1.025 s and response wait/progress 4.453 s. These timings nest: never add parents and children, or interpret response wait as isolated RoCE latency. The first expert layer included about 538 ms of cold setup; this is not a warm steady-state trace.

## Isolated index components

`scripts/bench-ds41-index-components.py` times the deployed native overlay scorer and top-512 independently. Generated FP4 queries/keys use finite scales and all-valid committed-cache candidates; no proposal reads or causal masking reduce the work. It covers query rows 1, 6, 80, 256 and 1024 against widths 512, 4096 and 16384. Each measurement is the median of five samples of five graph replays. Poisoned outputs must be repopulated exactly against an uncaptured launch, and scoring must be finite/nonzero. These checks establish replay execution, not mathematical correctness against the model reference.

An initial temporary benchmark captured the wrong stream and timed empty graphs. Those numbers were discarded. The durable benchmark resolves the current CUDA stream at each native launch and verifies poisoned replay outputs before timing.

On the RTX PRO 6000, 1024 queries took:

| Candidate width | Scores | Top-512 |
|---:|---:|---:|
| 512 | 2.082 ms | 0.132 ms |
| 4096 | 16.855 ms | 0.363 ms |
| 16384 | 68.923 ms | 1.268 ms |

The all-valid 16k component is deliberately heavier than the average live layer. It identifies scoring as the dominant index compute cost; it does not explain the full live index stage or establish actual DRAM throughput.

## Next implementation

The native scorer assigns a 256-thread block to each query/candidate pair, repeats query decoding for every candidate, and computes all head dots using scalar arithmetic. b12x's upstream `dsa_indexer/mxfp4.py` is also a scalar direct-K implementation; it uses a different page layout, lacks our append-only proposal descriptor, and does not provide a ready tensor-core replacement. The contiguous FP8 indexer has tiled tensor-core patterns that can inform a new V4.1 CuTe scorer, but cannot be substituted without adapting packed FP4 values/scales, BF16 rounding stages, causal/page validation, and proposal overlays.

Prioritize a tiled CuTe scorer that reuses query/key decoding and handles both committed pages and proposal rows directly. Preserve 64-bit pool addressing, runtime live dimensions, preplanned scratch and graph replay. Qualify masks, malformed descriptors, high physical page IDs, BF16 score differences and selected-index agreement before an API rollout. A different reduction order needs quantified numerical evidence; a passing replay benchmark alone is insufficient. Top-k and graph-cache changes can follow measured scoring improvements.

Both selected direct-accumulation APIs remain available after profiling. The temporary instrumented target container is stopped. Raw logs, commands and diagnostic scripts are under `/tmp/ds41-index-profile`.

Reproduce the component measurement inside the coordinator development image with GPU access and the baseline library mounted:

```sh
python3 scripts/bench-ds41-index-components.py --native-lib /native/libds41rt_native.so --device 0 --output /audit/components.json
```

[Recorded stage and component timings](ds41-index-prefill-profile.json).
