# Native FP8 K32 expert input

b12x master `b53ade5` adds an explicit prequantized input representation at the native dynamic-kernel compilation boundary. It is restricted to deterministic V4.1 Spark geometry with repacked native FP4 weights. The representation participates in the compile cache key; no live row count was added. Input routing copies E4M3 payload and UE8M0 K32 scales directly into expert activation tiles, preserving the existing FC1, intermediate quantization, FC2 and output accumulation.

The native exporter accepts `--input-format fp8_k32` for Spark; its default remains BF16 until the coordinator request path is integrated. Native metadata ABI 2 reports dtype 1 (BF16) or 7 (E4M3/UE8M0 K32). The dtype occupies the previous trailing padding; metadata remains 64 bytes, with dtype at offset 60. ABI 1 libraries are rejected by the new Rust binding. New binaries must use rebuilt native libraries; running services have not been replaced.

Rust budgets and allocates input storage using that representation, independently from the unchanged BF16 compact output. Capacity-one and grouped states must agree on input dtype. The complete-batch parser validates either canonical format, and execution requires exact agreement with the native kernel before copying bytes or launching. The return format is unchanged.

## Qualification

- Three checked-in b12x GPU cases pass on Spark: 1, 16 and 80 rows with all 384 synthetic experts. FP32 route planes are byte-exact against BF16-input execution, including graph replay after changing routes and encoded inputs. Poisoning the original BF16 source proves it is not read; zeroing encoded payload makes graph output zero.
- All six capacities (1, 16, 80, 256, 1024, 4096) export successfully with FP8 input. The full native library was relinked from those expert objects, a rebuilt expert wrapper and unchanged nonexpert objects; this is not a clean full-native rebuild.
- RTX SM120 encoded two captured 80-row real-model inputs using the same CuTe quantization intrinsic and V4.1 floor. The interleaved row format is exactly 422,400 bytes per input (5,280 per row).
- A Rust owner on Spark loaded all official experts for layers zero and one, rank zero. Layer visits `[0, 1, 0]` and row sequence `[1, 16, 1, 80, 1]` passed memory-budget rejection, stable state pointers, layer rebinding, changed-input graph replay and graph rebind rejection.
- Against the saved BF16-input baseline, all 36 FP32 route and compact BF16 files are byte-exact: 77,475,840 bytes. Route planes are finite and nonzero. This independently qualifies the input change; it does not resolve the earlier compact-return logit discrepancy.
- Transport: 149 tests pass, one ignored. Two focused FFI ABI/storage tests and workspace checking pass. A broad FFI run against the local stub library had six failures involving unavailable legacy CUDA kernels/pinned allocation; it is not reported as passing.

[Artifact hashes, export manifest, source hashes, input encoder, Rust fixture and comparison evidence](ds41-fp8-native-consumer.json). Remote artifacts are `/tmp/ds41-fp8-native-{source,export,cmake}` and `/tmp/ds41-fp8-owner-output`; logs use `/tmp/ds41-fp8-*`. No end-to-end speedup is claimed. The coordinator still sends BF16 in live serving. Next integrate stable GPU-encoded request storage, rebuild matching coordinator/worker artifacts, then run paired API qualification and timing.

## Tiling measurements

Optional worker timing now emits expert-local row-count bins for M=1 through 16 plus a 17+ tail, with exact routed-row totals for that tail. The timing summarizer validates totals and reports both expert-count and route-count fractions per request-row group. Histograms describe logical expert reuse before M-tile padding. They are not yet collected from the deployed worker.

Choose resident layout using common speculative-decode traffic, including M=2 and larger expert-local batches. Keep M=1 as a baseline rather than the sole objective. Compare 128-wide slices with a 64-wide tail, uniform 64-wide slices and 192-wide slices with enough representative routes to expose their occupancy/scheduling tradeoffs. Prefill can use different M tiles over the chosen resident layout; large grouped prefill throughput remains a required guardrail. These tiling alternatives are not yet benchmarked.
