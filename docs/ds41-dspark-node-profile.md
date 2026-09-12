# dSpark graph-node profile

A launch-time Nsight Systems capture of the selected four-slot coordinator ran a
single counting request on the dSpark API. Graph internals are visible. The
analysis starts five seconds after process launch and covers 231 completed draft
graphs on stream 623. The API request completed; this instrumented capture is a
kernel attribution experiment, not a release-throughput measurement.

The draft graph is already a single captured three-stage transformer plus
terminal projection and dependent sampling. Its measured kernel activity sums
to 7.216 ms per draft graph:

| Component | Instrumented kernel time per draft |
| --- | ---: |
| Three local routed expert calls | 4.036 ms |
| Shared vocabulary head | 0.866 ms |
| Five vocabulary adjustment/sample calls | 0.855 ms |
| Three draft attention calls | 0.726 ms |
| Other graph kernels | 0.733 ms |

The routed expert kernel launches 188 persistent CTAs, one per RTX SM; its grid
is `(1,1,188)`, not one CTA. It remains the largest draft-side kernel contributor.
Draft attention launches 20 CTAs at C1 (`5 x 4`). Both deserve further kernel
investigation; their performance is not changed by the sampling experiment.

The original sampling kernel launches one 256-thread block per request, scanning
all 129,280 logits and writing adjusted raw logits while selecting a token. C1
therefore uses one block, five times per draft graph. A parallel vocabulary-tile
implementation was subsequently [qualified and selected](ds41-draft-sampling.md).
The original kernel measures about
80 us in an isolated unprofiled greedy replay benchmark, versus approximately
171 us here; instrumentation and workload/cache differences matter. Do not apply
the instrumented saving directly to an end-to-end TPS prediction.

[Kernel totals and artifact identity](ds41-dspark-node-profile.json) retain the
attribution. The raw report and SQLite export are under
`/tmp/ds41-dspark-node-profile`; export completion was verified before stopping
the disposable profiling container. The report predates the sampler change.
