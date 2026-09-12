# Continuous alternating encoder lanes

The paired scheduler waited for both chunks to complete layer 19 and commit before
starting the next pair. The new scheduler gives each lane every other prompt chunk.
After a lane commits, it can begin its next chunk while the other lane is still
working. The final short chunk follows the same schedule.

There are still only two mutable target passes, two RoCE transport waves and at
most two active request/Engram batches. Reservation notifications keep the virtual
Engram history in prompt order. Per-layer notifications release the following
chunk only after its predecessor has finished attention readers and published that
layer's KV. Completion notifications serialize retained-suffix capture, source-20
publication and history/cache commits. No shared request or suffix borrow crosses
an await. The model, native kernels, worker artifacts and token chunk boundaries
are unchanged.

Cancellation owns a guard for each active chunk. After its execution future drops,
the guard revokes the request and discards the lane before reuse. An inactive lane
has already committed and is idle. Disconnect checks occur before reserving a
chunk and before publishing its final boundary/commit.

## Qualification

The real RTX/four-Spark TargetPass fixture compares serial, paired and continuous
execution for 65+65, 79+1, 65+65+1 and 64+64+27 rows. A separate 31+33+35+37+39
run reuses both lanes across multiple former pair boundaries. All retained
residual/pre bytes match serial execution, committed positions match, and decoder
replay begins at the expected final-128 boundary. Dropping a live stream revokes
admission and resets both passes; the same owners then pass the five-chunk check.
The release daemon and test binary build successfully.

Both API lifecycle checks pass. All eight prefill benchmark outputs/usages and
eight quality-case outputs/usages per mode match the selected tail scheduler.
The inherited Unicode-format failure and differing target/speculative explanation
remain. These checks do not establish broad bounded-decoder-replay accuracy or
concurrency-16 serving.

## Timing and observed overlap

The controlled API comparison runs three requests per workload and mode, using
identical native kernels and workers. Means of the final two requests are:

| Effective 16k prefill tok/s | Paired + tail | Continuous lanes |
| --- | ---: | ---: |
| Target, code | 6,323 | 6,514 |
| dSpark, code | 6,270 | 6,584 |
| Target, repeated text | 7,881 | 7,887 |
| dSpark, repeated text | 7,685 | 8,033 |

This is a small-sample comparison with visible run-to-run variation, not a
statistical throughput guarantee. The first candidate run was mixed, which is
why the controlled repeat is retained separately.

The warm code trace confirms chunks 2, 4 and 6 begin about 13 ms before the
preceding chunk commits; the tail begins about 37 ms early. Warm repeated-text
runs show about 11 ms of overlap at these former pair boundaries. Existing
within-pair overlap remains. Start times are inferred from host completion
timestamps minus measured chunk durations; they describe scheduling overlap,
not independent GPU kernel durations.

A diagnostic rank-zero worker log sample (4,960 requests of 2,048 rows over an
8-minute interval, mixed benchmark cases) reports median kernel 9.90 ms,
compaction 0.315 ms, input upload 0.600 ms and output download 0.358 ms.
These are per-field medians, not an additive critical-path breakdown. The sample
bounds the likely payoff of download-only overlap; it does not measure all-rank
network completion or establish memory-bandwidth efficiency.

The separate 599-token decode comparison preserves text and usage: target
37.82 → 37.93 tok/s and dSpark 117.81 → 119.56 tok/s. This is a regression
check, not evidence of a decode optimization. Results and artifact hashes are in
[the result record](ds41-encoder-stream.json).

The selected development APIs use `/tmp/ds41-stream/daemon` and the unchanged
`/tmp/ds41-head-reuse/cmake/libds41rt_native.so`:
`ds41-stream2048-live-target-api-dev` on port 18041 and
`ds41-stream2048-live-spec-api-dev` on port 18042. Launch arrays, component
commands, API checks, traces and post-rollout smoke responses are retained under
`/tmp/ds41-stream`. The prior tail-scheduler containers remain stopped for rollback.
Workers, b12x pin and 2,048-row live chunk size are unchanged.

## Scope

This removes the coordinator's pair-completion barrier. Worker GPU execution,
response compaction, host transfer and send still need their own overlap audit;
this change does not implement streaming expert output or Spark-side cross-rank
reduction. The external vLLM/EXL3 comparison remains queued behind overlap work.
