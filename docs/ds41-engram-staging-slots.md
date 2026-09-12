# Engram staging for both alternating lanes

The two-lane scheduler needs four simultaneous host staging leases: two Engram
layers for each lane. The serving pipeline previously allocated only two. The
first lane occupied both leases, so the second lane's gather was admitted only
when consumption reached each Engram layer. The consumer then slept on the
existing 1-ms polling timer even when the gather completed in tens of microseconds.

The pool now reserves `2 * ENGRAM_LAYERS.len()` slots. This allows both lanes to
submit both gathers during preparation. No I/O backend, tensor format, GPU buffer,
commit boundary, cancellation rule or request migration rule changes. With the
4096-row serving capacity, the additional two reusable staging buffers cost about
52.5 MiB of host RAM and remain inside the existing 256-MiB staging budget.

## Attribution

With `RUST_LOG=info,ds41rt::timing=debug`, each gather records queue duration,
gather duration, thread-local minor/major faults and input blocks. Consumption
records how long the completed data has already been ready. These clocks and
OS-counter queries are disabled when the timing trace is disabled. Gather counters
include staging/output page faults; they do not uniquely identify file-mapping
faults and do not account for the separate advisory-prefetch thread.

Three C6 counting runs per configuration contained 400 row-18 Engram deliveries
per run (100 complete two-lane rounds, two Engram layers):

| Configuration | Pending polls, each run | Deliveries that waited, each run |
| --- | ---: | ---: |
| Two slots | 200 | 200 |
| Four slots | 0 | 0 |

With two slots, warmed median gathers took about 35–40 us; the affected consumer
wait intervals were about 2.1–2.2 ms. Data had already been ready for almost the
whole interval. These intervals can overlap useful work in the other lane and
must not simply be summed into a predicted throughput improvement.

The first two-slot process recorded 411,704 minor faults across the request run;
47,676 occurred during the measured row-18 decode gathers. There were no process
storage-read bytes or major faults. Warm repeats had no gather faults or input
blocks. This workload's immediate issue was admission and polling, not NVMe read
latency. This does not establish locality for other prompts.

## Qualification and throughput

The real-model alternating-lane fixture with four staging slots passed in 11.19
seconds. C2/C6/C16 differing-budget, cancellation and replacement API checks passed.
All measured counting runs generated identical per-request text and usage.

Untraced 599-token counting runs, in execution order:

| Run | C6 aggregate TPS | C16 aggregate TPS |
| --- | ---: | ---: |
| Candidate, fresh process | 304.24 | 615.38 |
| Baseline, warm | 318.83 | 620.42 |
| Baseline, warm repeat | 318.93 | 620.27 |
| Candidate, warm repeat | 323.91 | 623.83 |

The warmed candidate is approximately 1.6% higher at C6 and 0.6% higher at C16 than
the mean of the two baseline samples. The first candidate run is not comparable
to a newly started baseline and does not quantify cold-start performance. Separate
traced warm runs improved from about 310 to 314–316 TPS. No C1 or broad-category
throughput improvement is claimed.

[Results, counter summaries and artifact identity](ds41-engram-staging-slots.json)
retain the measurements. Native kernels and all four Spark workers are unchanged.
The [I/O-method comparison](ds41-engram-io-comparison.md) remains separate: io_uring
is not selected or required by this change.

The four-slot daemon was initially selected on target port 18041 and dSpark port 18042 as
`ds41-engram-slots-live-target-api-dev` and `ds41-engram-slots-live-spec-api-dev`.
Both modes passed the native API qualifier after rollout. The stopped
`ds41-staged-live-*` containers retain the previous daemon for rollback. The later
[sampler rollout](ds41-draft-sampling.md) changed the selected container names and
native library while retaining this Rust daemon.
