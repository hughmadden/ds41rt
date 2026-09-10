# Native V4.1 expert service

`ds41rt expertd-native` connects the official checkpoint loader, native FP4 packing, owned expert execution and bounded TCP response stream. It loads every backbone expert in all 40 layers for one TP rank; it does not load dSpark, which remains on the coordinator RTX.

Example invocation on a Spark, with one visible GPU and paths replaced by local deployment paths:

```bash
CUDA_VISIBLE_DEVICES=0 ./rust/target/debug/ds41rt expertd-native \
  --snapshot /path/to/official-v41-snapshot \
  --native-lib /path/to/spark/libds41rt_native.so \
  --rank 0 \
  --capacity 16 \
  --device-budget-bytes 90000000000 \
  --listen 0.0.0.0:9100
```

The snapshot must satisfy the complete official tensor/header contract. The native library must contain Spark-role V4.1 AOT kernels for the selected capacity and hardware. Supported capacities are 1, 16, 80, 256, 1024 and 4096; a request cannot exceed the selected capacity. The explicit device budget includes resident packed weights and the larger of loading staging or execution workspace; it does not include host queues, transport copies, CUDA driver allocations or other processes. The example budget is an admission allowance, not a measured free-memory claim.

A dedicated thread creates and destroys the library, resident weights and CUDA resources. One execution workspace is reused across resident layers after synchronization; captured graphs cannot be rebound. This serial worker is the TCP fallback and does not yet implement alternating GPU waves or RDMA overlap. Request admission has a 16-entry queue, separate from coordinator request concurrency. A full queue rejects admission. Each response queue holds one chunk, with additional bounded copies at the TCP handoff and writer; the worker's host exchange still retains the full FP32 route plane.

Native executor IDs are TP rank plus one, so the coordinator must configure `[1, 2, 3, 4]`. Every rank receives the same canonical six-route request. `TcpProtocolV2PersistentClient::roundtrip_chunks` supplies frames to `V41Tp4ChunkReceiver`; coordinator orchestration of those calls and device destination lifetimes remains pending. Request IDs must uniquely identify in-flight waves. Client disconnection releases response backpressure without poisoning the GPU worker; a native execution error stops it and disconnects queued work.

The new command is separate from the inherited `expertd` launch path while deployment migration is unfinished. `run.sh`, WIP startup and the default configuration have not yet switched to it. Full checkpoint serving, native coordinator dispatch, RDMA, shared execution graph capture and runtime qualification remain open. No model load, GPU test or placement benchmark was run for this change.

Local validation: `cargo check` and `cargo build` for `ds41rt-daemon` both completed successfully through `scripts/run-with-python-env.sh` using `.venv/bin/python`, with 431 warnings including remaining unused execution APIs; the binary was not used to load a model.
