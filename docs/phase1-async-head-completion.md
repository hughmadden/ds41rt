# Cooperative LM-head completion

Two active decode lanes can now yield the CUDA owner thread while their own
LM-head stream finishes. The direct single-active-lane and prefill paths retain
their synchronous numerical sequence. Head weights remain shared and immutable;
each lane already owns separate execution buffers, projection workspace and stream.

The native `ds41rt_cuda_stream_query` API distinguishes pending and complete
without blocking. Asynchronous CUDA errors remain errors. Its test gates one
stream, checks that a query returns pending and an independent peer returns
complete, then releases the gate and observes completion. A CTest deadline catches
an accidentally blocking implementation. Native API and XGrammar checks pass.
The Rust check and release build pass.

`LoadStream::wait` queries completion and yields to the executor while pending.
A guard drains on cancellation or error before queued input storage can be
released. The scheduler still drains both lane futures on a failure rather than
cancelling a peer with in-flight GPU/RDMA work. No shared request-bank borrow is
held while the head awaits completion.

The first prototype yielded after copying head inputs and again after head replay.
The final adjustment queues input copies and an already-captured graph together
on the same stream, then waits once. First-use/shape-change capture still needs
completed inputs and a synchronous warmup. Full logits download, CPU argmax,
draft generation and draft-cache commit remain synchronous; this change does not
claim to remove every completion boundary.

## Initial comparison

One RTX PRO 6000 Blackwell, **400 W and standard memory speed**, four unchanged
Sparks, five complete local expert layers, 18 × 1,048,576-token KV capacity and
24 retained snapshots. Both arms use the same rebuilt native library. No builds
overlap timed serving.

| Workload | Synchronous head | First cooperative prototype |
|---|---:|---:|
| C1 code, tok/s | 128.33 | 128.55 |
| 32K prefill, tok/s | 7,611.62 | 7,751.14 |
| C2 mixed, tok/s | 101.77 | 102.31 |
| C4 mixed, tok/s | 109.81 | 110.49 |
| C8 mixed, tok/s | 160.88 | 154.05 |
| C16 mixed, tok/s | 184.22 | 194.68 |

C1/prefill are three-sample medians; mixed traffic has one batch per concurrency.
C16 improves 5.7% in this sample, while C8 decreases 4.2%. These sequential arms
are exploratory and do not establish a uniform throughput gain. Mixed output
lengths/trajectories can differ. All applicable Python structure checks pass;
prose quality is not assessed.

Both arms pass middle-needle, prompt/turn reuse, cancellation/survivor and recovery
checks. Four concurrent high-thinking tool/strict-JSON requests also pass for the
prototype, including ordinary and streamed responses. Standard serving is restored
at the end. The final queued-copy adjustment passes its focused concurrent lifecycle and
high-thinking check; C1/prefill do not take that adjusted path. Standard serving
is restored afterward. Its C2/C4/C8/C16 rates were 89.42/110.17/160.41/182.42 tok/s,
but this run omitted the earlier C1/prefill warmup sequence and has no matched
control. These numbers do not establish the final adjustment's performance effect.
Keep this as incremental work toward removing the remaining head/download and
draft completion waits; qualification of the combined change remains required.

The first native rebuild omitted `DS41RT_ENABLE_V41_LOCAL_EXPERT_AOT=ON` and the
control failed startup before any API requests. The service was restored. After
correcting that build option, export comparison found no missing `ds41rt_` symbols
relative to the working library; the only addition is the stream query API. Four
native checks pass on the corrected library. This is compiled local GPU code,
not a weight export or change to five-layer placement.

[Evidence](phase1-async-head-completion.json) retains commands, artifact identities,
export comparison, native results, C1/prefill samples, mixed responses/comparison
and high-thinking checks. Raw records are under
`~/.cache/ds41rt-experiments/async-head`; the reusable serving build is under
`~/.cache/ds41rt-experiments/native-serving-build` (about 59 MB after this build).
