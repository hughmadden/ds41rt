# Distributed layer-0 integration smoke test

The retained `real_layer_zero_executes_embedding_attention_tp4_and_mhc` test executed the production `BackboneExecution::execute_layer` on RTX GPU 0 with real native layer-0 routed experts on ostrich, dodo, emu and kiwi. Two changed batches each contained 16 requests and five rows per request. Workers served both batches over one persistent connection per rank, returning seven-row chunks.

The test loaded the official embedding table, all owned backbone lane weights, cache-producer weights/sinks and index-query weights. It initialized the four residual streams and pre-mix from real token embeddings, executed the layer-0 query, then called the combined layer method. That method produced FP8 window KV, ran window attention and output projection, normalized FFN inputs, routed actual hidden rows to the Sparks, executed the real RTX shared expert, collected/reduced FP32 rank planes and completed FFN mHC.

Both runs produced finite BF16 residuals and FP32 pre-mix values with the expected layer/position metadata. Changed token IDs changed both the routed request bytes and final residuals. A cache commit attempted after only layer 0 was rejected; all 16 committed history ends remained zero. Restarting the owned lanes then completed the second batch successfully.

All four workers received byte-identical request frames within each batch. All eight returned full FP32 planes had the expected 9,830,400-byte extent and finite values. Both coordinator and all four worker processes exited with status zero; their containers were gone afterward. Raw requests/planes and coordinator residual/pre outputs are preserved under `/tmp/ds41-layer0-output` and hashed in [the evidence record](ds41-layer0-integration.json).

The coordinator loaded its selected owned weights in 2.492 seconds with uncontrolled, likely warm filesystem caches. The layer method took 0.237 and 0.177 seconds; these measurements exclude embedding/query initialization and are **not** full-model throughput, cold NVMe loading or overlap-efficiency qualifications. Total test runtime was 4.12 seconds.

This is an integration smoke test, **not an independent numerical-reference qualification of the assembled layer**. It does not exercise compressed attention/index selection (layer 0 has neither), engram, decoder taps, target-head sampling, image inputs, a complete 40-layer pass, accepted model history, production RDMA or concurrent alternating passes. Component numerical evidence remains separate; broader composition and reference checks are still required.

## Reproduction inputs

The test compiles from the production tree through `/tmp/ds41-lane-ffn`. Set `DS41RT_LAYER0_LIBRARY`, `DS41RT_LAYER0_MODEL`, `DS41RT_LAYER0_PEERS` (four comma-separated endpoints) and optionally `DS41RT_LAYER0_OUTPUT`. This run used peers `172.22.2.1:19441` through `172.22.2.4:19441`, executor identities 1–4 and the official `dba1be0a40aa45a94ad051997016db3960a90277` snapshot.

Workers used the previously qualified `/tmp/ds41-real-tp4/artifacts/ds41-real-tp4-worker` fixture with rank and layer arguments, native library from the same artifact directory, and `ds41rt-spark-expert-dev:latest`. Each worker loaded its real TP slice for layer 0, served two requests, recorded inputs/planes and exited. The RTX test used the [isolated matching driver libraries](ds41-rtx-testing-restored.md) and `ds41rt-coordinator-dev:latest`.

Logs: `/tmp/ds41-layer0-rtx-test.log`, `/tmp/ds41-layer0-worker-{ostrich,dodo,emu,kiwi}.log`, `/tmp/ds41-layer0-fixture-build.log`. Fixture source and native/worker hashes are recorded below; no test-only bypass was added to the production execution path.

## Routed expert numerical reference

The actual recorded request hidden rows, expert IDs and routing weights now pass byte-for-byte checks against the on-wire payloads, and every rank's received frame matches. `qualify-ds41-real-tp4.py --routes-only` then ran the pinned official `Expert.forward`, activation quantizer and FP4 GEMM against the same real expert weights. Across the two batches, all **960 routes spanning 181 distinct experts** pass the established per-expert relative-L2/cosine bounds.

| Batch | Active experts | Routed relative L2 | Routed max absolute error |
| ---: | ---: | ---: | ---: |
| 0 | 154 | 0.0000154975 | 0.00048828125 |
| 1 | 141 | 0.0000297121 | 0.0009765625 |

The compared native values are rank-ordered FP32 sums rounded to BF16 per route, before summing the six routes and adding the shared expert. This numerical check covers routed expert execution from the actual distributed layer's FFN hidden inputs; it does not independently validate the upstream attention/mHC inputs, selection of expert IDs/routing weights, shared FFN or final mHC output. The complete assembled-layer numerical comparison remains open.

The qualifier's full mode also passed its existing real layer-0 shared/reduction fixture after this change. It now reads each case's own shared output rather than assuming case 0's output applies to both; the earlier fixture explicitly produced identical shared outputs, so that assumption did not invalidate its recorded result. The new routed-only mode makes no shared/reduction qualification claim. Both modes validate that fixture hidden rows, IDs and weights match recorded requests.

Reference runtime: TileLang 0.1.8 and TVM FFI 0.1.6; the pinned reference activation-quantizer vectorization override remains enabled. See [the full per-expert results and payload hashes](ds41-layer0-routes-reference.json). Logs: `/tmp/ds41-layer0-routes-reference.log`, `/tmp/ds41-tp4-qualifier-default-regression.log`. Canonical request fields were extracted into `/tmp/ds41-layer0-reference-vectors/router`; all four original rank records remain in `/tmp/ds41-layer0-output`.
