# DS4RT architecture

DS4RT is an attention–FFN-disaggregated DeepSeek V4 Pro engine for one RTX PRO
6000 Blackwell coordinator and four DGX Spark workers. This document defines
the stable v1 ownership and execution contract. Model dimensions are validated
from the selected snapshot rather than inferred from these prose values.

## Topology and ownership

| Component | Release owner |
| --- | --- |
| OpenAI-compatible API, tokenizer, scheduler, admission | Coordinator |
| Embedding, residual stream, mHC mixing, norms, LM head | Coordinator |
| MLA/C4/C128 attention, compressors, indexers, RoPE, KV | Coordinator |
| Routers, shared experts, dSpark envelope and sampling | Coordinator |
| Routed `w1`, `w2`, `w3` EXL3 projections | Four Spark TP ranks |
| Route dispatch and expert reduction | Coordinator plus all four Sparks |

The model has 61 target layers, 384 routed experts per sparse layer, top-6
routing, hidden width 7,168, and expert intermediate width 3,072. It also has
three integrated dSpark blocks. The release snapshot stores 73,728 routed
projections in calibrated EXL3 K2 form; coordinator-owned tensors retain their
published representation.

## Routed expert tensor parallelism

DS4RT does not assign whole experts to workers. Every Spark stores one quarter
of the intermediate dimension of every routed expert. The coordinator computes
one global route list and sends identical hidden rows and route weights to all
four ranks. Each rank computes its quarter-intermediate `w1`/`w3` activation
and `w2` projection, producing one hidden-width partial. Those four partials
must be reduced before the residual stream advances.

This distinction is fundamental:

- expert placement is fixed 4-way tensor parallelism;
- all ranks see the same global top-6 routes;
- rank names are `spark-0` through `spark-3`, independent of SSH host names;
- a load plan may choose a source reader, but never changes serving ownership.

Small expert batches return four BF16 partials for deterministic coordinator
accumulation. Wider batches use row-sharded Spark reduction. The crossover is
configured by `SPARK_REDUCTION_MIN_ROWS` and is 16 in the qualified release.

## One target block

The coordinator executes model-specific multi-head latent attention with
partial RoPE, the checkpoint's C0/C4/C128 compression schedule, and mHC
residual mixing. After attention it computes the router and shared expert,
dispatches the routed expert boundary, waits for the reduced result, and
finishes the next mHC boundary.

Two coordinator CUDA-graph segments surround the remote expert barrier:

1. attention, attention-post/FFN-pre mHC, routing, and shared expert;
2. reduced routed delta, FFN-post/next-attention-pre mHC.

CUDA graph bindings and layer workspaces are prepared at startup. A qualified
request performs no graph capture and no Python execution on its timed path.

## dSpark

DeepSeek V4 Pro's dSpark head is a three-block semi-autoregressive drafter.
DS4RT runs its routed experts through the same TP4 boundary as target layers.
The adaptive controller selects a proposal width from one through five using
checkpoint confidence outputs, then the target verifies the proposal. The
`-full` API model alias disables drafting without changing the target model.

## Cache and profiles

The coordinator owns physical attention state. The `balanced` release profile
uses FP8 target KV and exposes a 400,000-token logical context with 405,504
physical rows on the measured 96-GB coordinator. `long` uses native DSV4
NVFP4 KV and increases Pro capacity to 545,536 tokens. `accuracy` uses BF16 KV
and BF16 wide Spark reduction with a 200,000-token logical context.

Admission is capacity-shaped: the sum of active context plus configured output
reservations must fit the shared pool. Excess requests wait; they are not
partially admitted.

## Transport and readiness

Release serving uses host networking, Protocol V2, and native verbs transport
over the configured dedicated addresses. A secondary Spark-to-Spark rail is
enabled only when all four `SPARK_N_LANE_B` values are configured. Every role
uses unlimited memlock and explicit RDMA device access.

Health remains false until:

1. all five hosts resolve the same immutable model revision;
2. all four Spark images match the coordinator engine and SparkInfer pins;
3. routed expert weights are resident;
4. transport negotiation and the numerical startup probe pass;
5. production graph shapes are prepared; and
6. the API advertises both the drafted and `-full` model identities.

See [`docs/architecture.svg`](docs/architecture.svg) and
[`docs/request-path.svg`](docs/request-path.svg) for the corresponding system
and request-flow views.
