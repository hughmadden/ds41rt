# Combined attention graph revisit

The release retains the asynchronous implementation at 2957772 and omits this
optional graph fusion. Adaptive C1 improved consistently by 0.5–1.3%, but the
concurrent results did not establish a benefit sufficient to justify the extra
capture and lifecycle machinery. This is not proof of a general C16 regression:
the control itself ranged from 148.73 to 189.78 tok/s.

The three matched pairs used adaptive dSpark, independent lanes, five RTX expert
layers, identical native kernels, and the same fixed-input mixed corpus. Order
was A/B, B/A, A/B. Prose output lengths can differ; this is not an exact-output
throughput comparison. Each C1 median contains three samples.

| Metric | Pair 1 | Pair 2 | Pair 3 |
|---|---:|---:|---:|
| Adaptive C1 change | +0.88% | +0.50% | +1.25% |
| C2 change | +3.42% | −8.30% | −5.63% |
| C6 change | −2.12% | −0.21% | +4.87% |
| C8 change | −3.99% | +2.84% | +0.21% |
| C16 change | −9.28% | −13.11% | +6.01% |

Target-only C1 in the initial pair improved 44.25 → 44.76 tok/s. Full GPU fixtures
at 32 and 80 rows produced 16 exact saved outputs, including pending cancellation
and reuse. Initial serving lifecycle, constraints and indexed-context checks
passed; the two follow-up pairs repeated only the focused performance cases.

An instrumented 12-second capture contained no CUDA stream/device synchronization,
synchronous memcpy or CUDA allocation/free calls. Graph instantiations fell from
17,868 to 16,562, but their CPU time was essentially flat: 812.80 → 822.20 ms.
Instrumentation used 14 GiB KV for profiler headroom and is not throughput evidence.

[Machine-readable evidence](phase1-attention-graph-revisit.json) preserves all
pairs, commands, binary identities and trace API totals. The
[candidate patch](phase1-attention-graph-revisit-candidate.patch) applies to
2957772. The original [graph experiment](phase1-attention-graph.md) remains separate.
Next compare fixed-output counting against the frozen pre-expansion control to
check accumulated asynchronous changes, then perform the scoped v2 qualification.
