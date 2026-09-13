# Matched-prefix verification costs

One RTX PRO 6000 Blackwell at **400 W, standard memory speed** (configured maximum
14,001 MHz, no memory overclock), plus four unchanged Sparks. This is an isolated
private-verifier experiment. The probe hook is not installed in serving.

## Same committed state, different joint prefixes

The probe captures six nearby contexts from a code/fable pair, with one request
on each execution lane. It generates the full draft once, then tests all 25 pairs
of one-to-five retained draft tokens against the same committed context. The
mandatory anchor is present in addition to those lengths. Committed positions
range from 89–114 on one lane and 64–79 on the other.

Every trial prepares fresh private request batches, drains both target futures,
collects logits/routes, then discards both private proposals. It never commits
trial tokens or changes the draft RNG/history. All 25 shapes run once for warmup,
then three measured sweeps rotate/reverse their order. Full-prefix sentinels
bracket each sweep. The hook completes before ordinary verification and commit
resume. There were 648 complete verifier executions, including 450 measured
trials (six contexts × 25 shapes × three repetitions).

Checks passed:

- Committed window rows, paged FP4 KV/index rows, valid page tables and published
  lengths retain their SHA-256 hashes after every trial. Hashing starts before
  the first trial and reads only initialized rows.
- Every full-prefix sentinel retains the same greedy selections.
- All **3,600 common-row greedy selections** in measured shorter-prefix trials
  agree with the full-prefix reference; repeat expert unions are also identical.
- The two final streamed responses exactly match the control. Both control and
  probe runs pass the 32K needle, prompt reuse, retained turn, eight cancellations
  with eight counting survivors, and recovery request.

These checks do not independently hash hidden compressor carry or prove equality
of every logit byte. The completed-discard ownership contract, repeated full
sentinels and final outputs provide additional evidence of successful rollback.

## The observed cost curve is flatter than the model

Median full verification plus logit collection, in milliseconds. Rows are lane 0
draft lengths; columns are lane 1 draft lengths. Each cell pools 18 measured
trials from six contexts. Draft generation, hashing, discard cleanup and final
publication are excluded; route capture is enabled for every trial.

| Lane 0 / lane 1 | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 51.88 | 53.54 | 56.29 | 58.14 | 61.02 |
| 2 | 55.52 | 56.90 | 58.99 | 60.38 | 62.50 |
| 3 | 59.77 | 58.85 | 60.08 | 60.74 | 62.01 |
| 4 | 60.25 | 60.16 | 60.56 | 61.06 | 63.40 |
| 5 | 61.88 | 60.75 | 60.87 | 61.24 | 61.93 |

The existing cost formula was evaluated using **actual post-verification expert
unions**, giving it more information than the online accepted-history forecast.
This isolates a cost-shape mismatch from route-forecast error:

| Trim both requests | Median measured saving vs full | Model-predicted saving vs full |
| --- | ---: | ---: |
| Five drafts → three | 3.3% | 18.2% |
| Five drafts → one | 16.0% | 37.4% |

Savings are calculated within each matched context before taking the median;
they are not ratios of pooled table cells. Trimming can still be worthwhile, but
the current formula substantially overprices its benefit here. Do not fit a
universal floor from six nearby contexts or treat these data as a serving-TPS
measurement. The probe deliberately blocks publication while repeating work.

The current RoCE receiver yields on its first wait, then spins for up to 250 µs
before yielding again. A narrower subsequent quantum is a concrete next
experiment: test it inside the same matched-prefix probe before a serving gate.
The initial-yield improvement is already installed; this is a different question
about subsequent waits. This observation does not establish polling as the cause
of the measured plateau. No polling or adaptive-policy change was promoted.

## Reproduction and artifacts

In an isolated checkout of the matching source:

```bash
cp docs/phase1-prefix-probe.rs rust/crates/ds41rt-daemon/src/v41_native_serve/prefix_probe.rs
git apply docs/phase1-prefix-probe-scheduler.patch
DS41RT_PYTHON=.venv/bin/python scripts/run-with-python-env.sh \
  cargo build --release --manifest-path rust/Cargo.toml -p ds41rt-daemon
```

Use the candidate binary with the same qualified native library and ordinary
single-RTX/four-Spark configuration. Mount an empty audit directory and set
`DS41RT_PREFIX_PROBE_DIR` to it; enable `ds41rt::draft_policy=debug` to retain
confidence. The hook limits itself to six eligible snapshots, requires one
unconstrained request per lane, and begins after both requests have generated
at least 16 tokens. Snapshot files are created exclusively. Without that
variable it does not run, but the hook still belongs only in an isolated build.

`scripts/bench-ds41-adaptive-mixed.py --concurrency 2` drives the focused pair and
lifecycle checks. The new concurrency argument retains the previous default
`4 16` and its nonce sequence. Use the same CLI and nonce seed for the control.
Summarize with:

```bash
python scripts/summarize-ds41-prefix-probe.py /path/to/snapshots --output /path/to/summary.json
```

[The measured data](phase1-matched-prefix-cost.json) includes per-context timings,
expert unions, hashes, checks and limitations. Frozen artifacts, raw snapshots
and the control/probe runner are in `/tmp/ds41-phase1-prefix-probe`. The standard
service was restored, the hook removed from serving source, and the ordinary
binary rebuilt from the restored serving source. The previously qualified
frozen build remains available unchanged. No image,
release tag or `main` update was published.
