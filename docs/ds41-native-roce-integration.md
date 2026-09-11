# Native V4.1 RoCE integration

The native coordinator now selects `V41Tp4Roce`, backed by the existing four persistent verbs clients. The native expert command selects the existing verbs server. TCP carries endpoint bootstrap and idle connection-liveness checks; inference requests and response chunks use the registered RDMA rings and QPs. This native path has no TCP inference fallback.

Dispatch enqueues one request per rank and returns so shared FFN execution can overlap transport/expert execution. It does not claim that enqueuing means the NIC has finished sending. The pending wave exclusively borrows the transport until completion or cancellation. Cancellation/error queues QP resets; response destinations must be discarded. The selected streaming client path does not automatically replay partially delivered requests.

Responses use the existing recycled payload owners, with native request/executor identity, contiguous row coverage, shape and final-marker validation before the coordinator sink. No extra monolithic response assembly is introduced. One admitted wave limits queued payloads to four validated rank planes; the underlying existing transport channel is unbounded and must be revisited before allowing multiple outstanding waves per owner.

GPU integration remains incomplete: the worker currently copies the registered request view into its bounded GPU-owner queue, then uses its existing host staging and host response callback. The mapped device pointer is not yet passed into native expert execution. Coordinator input D2H and response H2D also remain. RoCE transport is not a claim of GPU-direct execution or completion of the latency work.

Validation so far: transport library tests pass (150 passed, one ignored), including a new RDMA payload-owner test for interleaved rank chunks and stale, reordered and duplicate response rejection. These are CPU tests, not fabric qualification. Cross-host QP reuse, cancellation, native GPU execution and live API quality/performance must still be measured before rollout.

The two old development API containers were stopped during integration because their frozen executables still use TCP inference. Spark weight/kernel artifacts remain intact. No new throughput result is available yet.

## Four-host transport qualification

The `v41_roce_qualify` transport example now runs against all four Spark hosts without loading a checkpoint. Twelve waves at 1/2/6/16/80 rows validated every FP8 input byte and all 852 returned BF16 row chunks. After wave five, the client dropped an enqueued 80-row wave; all seven subsequent checked waves passed. Logs show four initial QP connections and four replacement connections, using RTX `mlx5_0` / 10.55.0.12 and each Spark's `rocep1s0f0` / 10.55.0.1–4. Within each group the QPs were reused across row counts. The server logs include transport retry errors on abandoned connections; recovery succeeded through the replacement QPs. This does not cover mid-GPU cancellation, multiple simultaneous API clients, or GPU direct buffer ownership.

The first attempt failed explicitly because the frozen RTX coordinator native library had `DS41RT_ENABLE_RDMA=OFF`. A separate CUDA/RDMA-enabled fixture library was built from the current native source, with expert/coordinator AOT disabled. The rerun exited zero. This fixture library cannot replace the full coordinator library: the full AOT coordinator build must be rebuilt with RDMA enabled before API rollout. Current release/WIP build scripts already request RDMA; the defect was in the frozen development artifact.

[Raw client/server logs, connection descriptors, byte checks and binary hashes](ds41-native-roce-qualification.json) are retained. These diagnostic runs include first-use initialization and deliberately emit one response message per row. They are correctness evidence, not model performance measurements.
