# BF16 tensor-core router candidate

The running APIs still use the original router. This change provides a CuTe
projection, a separately callable native score-transform/top-k stage, an audited
AOT exporter, and repeatable component qualification. Serving integration and
full API quality/performance comparison remain outstanding.

The original score kernel launches one 256-thread CTA per token/expert dot
product. The candidate uses TMA-fed BF16 warp MMA, tile M64/N16/K64, three stages,
and FP32 output. K64 partials shorten the accumulation chain and improve numerical
accuracy. Live rows shape descriptors, masks and grids at runtime; neither model
geometry compiles on replay. The caller supplies inputs and outputs; no scratch
or load-time weight conversion is required. The existing score buffer holds
logits, then the selection kernel transforms them in place. This is a private AOT
component, not a new public planned operation.

On RTX PRO 6000 Blackwell GPU-fe5b6dd0-a77c-c8fb-6360-e1b9d9918ac0 (SM120,
188 SMs, driver 595.91.07), medians across official gates were:

| Experts | Rows | Original router µs | Candidate router µs |
|---|---:|---:|---:|
| 384 | 1 | 11.21 | 20.82 |
| 384 | 6 | 17.55 | 20.99 |
| 384 | 16 | 29.87 | 18.07 |
| 384 | 80 | 109.41 | 18.33 |
| 384 | 2048 | 2564.80 | 112.96 |
| 384 | 4096 | 5124.35 | 217.25 |
| 128 | 16 | 14.04 | 15.30 |
| 128 | 80 | 42.19 | 15.47 |
| 128 | 2048 | 880.99 | 51.90 |

These are complete GPU router timings, including selection, but exclude request
quantization, D2H copies and CPU preparation. Six interleaved measurements each
replay a ten-router graph; the ratio is original/candidate. Servers remained
loaded but idle during this component test. No end-to-end speedup is claimed.
Small-row serving must retain its original kernel until a measured crossover.

Qualification exercised all 40 backbone and three dSpark gates, eleven live-row
entries from 1 through 4096 (including a return to 6), and two input/mask mutations:
473 cases, 946 comparisons, 650,762 evaluated rows. Inputs are random BF16 with
standard deviations 1 and 0.25, not captured model activations. All score checks
pass the existing rtol=atol=2e-5. Exact-sized input allocations, guarded/poisoned
outputs, graph replay, unchanged Torch allocation counts and frozen b12x kernel
resolution pass. Output ordering is recorded separately from expert-set equality;
weights are compared by expert ID because the FFN sums weighted expert outputs.

The strict FP32-reference gate remains **failed**, rather than being silently
relaxed. Seven comparisons have ordered-ID discrepancies; two comparisons have
expert-set discrepancies, covering three rows. For all three disputed memberships,
the candidate agrees with independent FP64 dot/softplus/bias ranking. One also
agrees with the original native router. The FP64 selection margins are
1.11e-6, 1.73e-6 and 1.51e-6. These are ordinary precision differences, not evidence
of an indexing error; they still require disposition alongside full-model quality.
The earlier uninterrupted MMA accumulation had two wrong memberships even against
FP64 in the smaller corpus; K64 partial accumulation removed those cases.

Raw evidence is in `/tmp/ds41-router-gemm`; hashes and aggregated results are in
[the companion JSON](ds41-router-gemm.json). `qualification-all.json` is the final
full corpus; `qualification-diagnostic.json` and `qualification-partials.json`
preserve earlier failures. `final-aot/v41_router.json` records both generated
objects and checked pointer/scalar dispatch order. Export succeeded for both
geometries; execution through the generated C wrapper remains to be qualified.

Reproduce in the coordinator development image with the repository mounted at
`/workspace/ds41rt`, the complete Hugging Face cache at `/hf`, and an audit directory
at `/audit`. Compile the current native router source into `/audit/router.so` with
`nvcc -O3 -std=c++17 -arch=sm_120 --shared -Xcompiler=-fPIC`, adding the native include
directory. Then run:

```sh
PYTHONPATH=/workspace/ds41rt/third_party/sparkinfer python3 \
  /workspace/ds41rt/python/tools/qualify_v41_router_gemm.py \
  --snapshot /hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277 \
  --native-lib /audit/router.so --output /audit/qualification-all.json
python3 /workspace/ds41rt/python/tools/export_b12x_v41_router_aot.py \
  --output-dir /audit/final-aot
```

Next: connect preloaded AOT modules to the native router with b12x-owned crossover
selection, qualify generated C dispatch and failure boundaries, then measure
real prompt quality and 16k prefill. Preserve the current small-row path.
