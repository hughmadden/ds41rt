# Native TP4 bridge for vLLM AFD

This shared library embeds the existing Rust DS41RT native TCP/RoCE clients
behind a small C ABI. It reuses `V41Tp4Tcp`, `V41Tp4Roce`, the canonical native
ProtocolV2 codec, row/executor/placement validation, timeout handling and
connection disposal. There is no alternate wire implementation or expert math.
The ABI is [include/afd_bridge.h](include/afd_bridge.h).

One retained OS thread owns a current-thread Tokio runtime and one or two lane
tasks. Each task owns its transport and receives at most one operation. Its
ordinary async future borrows the request and transport until completion; no
self-referential FFI handle or unsafe lifetime extension is used. The C caller
can compute the shared FFN after `submit` returns while both lanes progress.
`submit` acknowledges owned admission, **not** completed network dispatch.

Each lane holds at most one pending request or uncollected result. The result is
four rank-major BF16 planes `[4, rows, 5120]`; it is published only after the
existing receiver verifies complete, correctly ordered coverage from every
expected rank. Old or foreign-lane tickets cannot consume another operation.
Cancellation drops the pending operation, resets its connections, then publishes
`CANCELLED`; it does not claim to stop work already running on a remote GPU.
`close` cancels both tasks and joins their owner. Existing synchronous RoCE cold
bootstrap/native operations retain their upstream completion semantics; shutdown
latency on that hardware path remains a fleet qualification item.

The rank-result allocation is retained across collection, failure and
cancellation. Equal or smaller shapes reuse it; larger shapes reserve their
exact required capacity. `afd_bridge_buffer_stats` exposes this narrow
allocation invariant for the real loopback test.

The first ABI deliberately accepts an owned encoded native request and copies
validated rank responses into an owned host result. It establishes correctness
and cross-language concurrency before a GPU buffer integration. It does **not**
yet preserve the recipe's retained pinned response buffers through asynchronous
H2D or provide captured device-input staging. Neither CPU loopback timing nor
this initial host-copy ABI establishes end-to-end performance parity.

## Build and checks

From the repository's `rust` directory:

```sh
cargo build --offline --locked --release -p ds41rt-afd-bridge
cargo test --offline --locked -p ds41rt-afd-bridge --test loopback
cargo test --offline --locked --release -p ds41rt-afd-bridge --test loopback cpu_loopback_latency_probe -- --ignored --nocapture
```

The output is `target/release/libds41rt_afd_bridge.so`. CPU tests use ephemeral
loopback sockets and actual DS41RT codecs/clients; fake ranks supply tagged
BF16 planes. They require no CUDA, libibverbs, checkpoint, Python package or
running service. Dependencies are existing workspace dependencies from the
retained lockfile; adding this crate changes no dependency version.

Tests cover connection reuse, unordered completion across ranks, canonical
row ordering, executor identity, truncation/disconnect, deadlines, cancellation
and reconnect, independent lane progress, bounded admission, result leases and
stale tickets. The latency probe reports descriptive host-loopback timing only,
including caller polling and validation; it is not a GPU/kernel/fleet benchmark.

For the RoCE backend, configure `DS41RT_NATIVE_LIB` and
`DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP` exactly as the recipe before creation.
The owner selects `device` once before making QPs. TCP never loads the native
library. The bridge can connect to the retained native `ds41rt expertd-native`
worker, preserving its existing resident AOT expert implementation; replacing
that implementation is unnecessary for the first integration.
The transport permits BF16 or FP8-K32 requests, but the retained recipe's W4A8
worker requires FP8-K32 activations. BF16 fake-rank tests do not qualify a BF16
request against that kernel.

## Provenance

Base: `hughmadden/ds41rt` commit
`63235c6a7d43fffc59d3021944fefd4f9b59e032`. The campaign parent inventory identifies
the retained fleet Rust source as `b042d731`; its comparison found the runtime
source in `ds41rt-core`, `ds41rt-ffi` and `ds41rt-transport` unchanged through the
base (test/Cargo test entries changed). That inventory establishes the component
lineage, not deployment of this new bridge. This change does not alter daemon,
transport, core, FFI or native-kernel source.

## Component qualification, 2026-09-19 07:23 AEST

- Offline locked release build passed; no dependency versions changed.
- Native release loopback suite: **8 passed**, including the explicit latency
  probe; ordinary suite is seven tests plus that opt-in diagnostic. One fault
  test exercises seven independent response-failure cases.
- Parent package's real ctypes ABI suite,
  `tests/integration/test_native_bridge.py`: **9 passed** against the release
  shared library, exercising Python→Rust→TCP→fake-rank interoperability.
- Exported symbols match the header, including the buffer reuse diagnostic.
- Release shared-library SHA256:
  `0a0aeff5fbc1a0559350187d4ae3d2fc207b37d9d519285a3485d1df55d14fd1`.

The diagnostic completed 30 measured iterations after five warmups at each row
count. Median loopback times were 1,057 / 1,091 / 2,298 / 4,674 microseconds for
1 / 16 / 80 / 128 rows. These values include 1ms caller polling and complete
tagged-output validation; they are not isolated transport latency and carry no
fleet or parity claim. Both the explicit eight-party dispatch barrier and the
stalled-lane test prove independent progress without relying on timing ratios.

### Admission-path correction, 2026-09-19 07:27 AEST

Review found that a BUSY lane still parsed/copied the entire submitted request
before rejecting admission. `submit` now takes the lane lock and checks its
lease before frame inspection, holding that lock through accepted decode/send
so concurrent callers cannot double-admit. No ticket or payload allocation is
consumed by BUSY. The new real-loopback regression failed before the fix
(`INVALID` instead of `BUSY`), then passed: malformed input is rejected as BUSY
for both pending and uncollected-ready lanes, and reaches the parser only after
the result is collected. This is a structural regression check, not a timing
threshold.

Requalification: **9/9 native release tests** including the explicit diagnostic,
**9/9 parent Python ctypes integration tests**, offline locked release build.
Current release SHA256:
`9df6774bd1407ae7070c67ea910ae6e7d2e514433a8fcc8e1e39f7f09b4bdf1f`.
