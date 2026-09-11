# RoCE QPs and expert execution on one worker thread

Each Spark now runs `ds41-local-qp-worker`. The same thread owns native GPU resources, polls all admitted persistent QPs, executes an incoming expert request, and sends its response. The coordinator binaries and expert native libraries are unchanged from the first RoCE rollout. There is no inference work queue, response channel, per-request thread wakeup, request-frame clone, or owned response clone in this worker path. Host/GPU staging still exists inside native execution.

`LocalVerbsExpertConnection` factors the previous persistent server into a bootstrap constructor and a nonblocking single-request poll operation. It reuses the existing protocol, endpoint registration, chunk validation and response encoding. A separate admission thread establishes connections and transfers exclusive session ownership through a bounded two-entry channel. The GPU owner admits at most sixteen QPs and services at most one request per QP per sweep. It actively polls while connections exist and blocks on admission when none exist. TCP is bootstrap and idle liveness only on the expert fabric.

The request receive slot is reposted after the synchronous execution callback finishes. Response sends retain their existing ring completion ownership. Idle peer closure is checked after one second of inactivity; disconnected/error sessions are removed. Full send-ring backpressure can still wait for send completion, so this is not a fully asynchronous multi-wave engine. Sixteen admitted connections are not proof of C16 model scheduling or alternating-wave overlap.

## Qualification and measurements

Transport unit tests: 150 passed, one ignored. The daemon and its test target compile. The refactored persistent server passed the four-host FP8/BF16 transport fixture: twelve checked waves, 852 response chunks, QP reuse and cancellation/reconnection. Both live APIs passed JSON arithmetic, streaming counting/usage, cancellation recovery and unsupported-sampling rejection. All eight paired quality prompts preserve text and usage exactly against the queued RoCE deployment; all six objective checks pass. Three rounds of simultaneous requests to the target and speculative APIs returned all six expected answers through the shared Spark owners. This is a bounded two-coordinator check, not C16 qualification.

Three single-client counting streams per mode, sequential modes, same coordinator executables and native libraries:

| Mode | Queued RoCE TPS | Local QP owner TPS | Median change |
|---|---|---|---:|
| Target | 14.11, 13.60, 15.06 | 20.32, 20.56, 20.56 | +45.7% |
| dSpark | 59.57, 64.97, 65.90 | 79.09, 79.40, 79.59 | +22.2% |

These remain short diagnostic runs without clock/throttle admission. The first stream warms, and decode TPS excludes prompt processing and the first completion token. Neither release throughput nor broad quality is established.

One-row coordinator expert-phase median fell from 839 to **328 µs**; attention stayed at **579 µs**. Six-row expert phase fell from 1166.5 to **705 µs**, with attention **600 µs**. Spark one-row kernel medians are about **157 µs**, complete native execution **186–187 µs**, and the server execution callback **191–192 µs**. The earlier queued callback was approximately 660–690 µs. Component medians are independently aggregated, not an exact additive decomposition. Unchanged GPU kernels also measured faster in this deployment; without clock admission this must not be attributed solely to kernel efficiency.

## State and remaining work

The measured worker executable SHA256 is `3b2d40d4f49f3694556d39c61ec8f7031b5bf1a037912db3ba13a4aa0700ecf7`, staged under `/tmp/ds41-local-qp-worker` on each Spark. Its source package retains the then-unused legacy queue definitions; those definitions were subsequently removed from repository source and the daemon test target rechecked. The executed local-owner implementation is unchanged. The prior `ds41-roce-target-worker` containers remain stopped; the RTX `ds41-roce-target-api-dev` and `ds41-roce-spec-api-dev` containers remain running on ports 18041/18042.

The retained GPU staging, coordinator routing/shared FFN/response-copy overhead and coordinator attention now account for most remaining work. Direct mapped input/output GPU ownership is still unwired. Performance targets, broad numerical quality, C16, long context and prefill remain open.

[Raw comparisons, quality results, concurrent checks, per-host timing medians, worker commands and log hashes](ds41-local-qp-owner.json) are retained. Full logs and API event records are under `/tmp/ds41-local-qp-rollout`; the transport fixture rerun is `/tmp/ds41-roce-qualify/local-session-client.log`.
