# Completed expert route capture

19 September 2026 AEST. Diagnostic only; no kernel or dispatch change.

Set `DS41RT_EXPERT_TRACE_DIR` to a fresh writable directory on the expert
worker. The process reads the setting once. It captures the first successfully
written rows=1 and rows=2 requests for layer1/executor2 (rank1). Other shapes
and ranks do not trigger it. A concurrent claim has exactly one winner per
shape. A failed write logs a warning and allows a later capture attempt.

After the existing stream synchronization, before response delivery or buffer
reuse, the hook reads the completed FP32 route outputs `[rows,6,5120]` and the
actual compact BF16 output `[rows,5120]`. It also records the original packed
input, route IDs, route weights, wire request identity and kernel capacity.
Registered response destinations are captured directly. Atomic token-sum
kernels are rejected because they do not expose six route planes.

The capture adds device reads and file I/O. Do not use its timing for performance
claims; total/download timing includes these diagnostics. An unset/empty
setting returns before device reads or file creation. The setting is separate
from the coordinator's `DS41RT_ACTIVATION_TRACE_*` controls and needs no debug
logging filter to write captures.

For the current rounding investigation, retain native library f0623636 and the
existing kernel selection. Replay the exact-wire component fixture and require
all 30 output sample hashes to equal the original baseline before interpreting
the route captures. An exact sum of captured FP32 route values establishes only
rank-compaction rounding; it does not establish exact earlier FC2 arithmetic.

CPU selection, concurrent claims, extents and file integrity tests:

```
cargo test --offline --locked -p ds41rt-daemon --bin ds41rt trace_tests
cargo test --offline --locked -p ds41rt-transport
```

The GPU serving-path test is explicitly ignored by default. To run it on a
released GPU with the required native library/model/trace-directory variables,
select its full test name with `--ignored`. Missing variables fail that test;
they are not reported as a passing GPU check.
