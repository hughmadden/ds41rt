# Compact GPU greedy head selection

Unconstrained decode now selects each row's top token on the GPU and downloads
only its token ID and score: **8 bytes instead of 517,120 bytes per row**. The
head projection, reduction and pinned-host copies run on the lane's stream.
With two active lanes, completion yields the owner thread so its peer can advance.
Lowest-token-ID tie breaking and rejection of any non-finite logit are preserved.

Constrained lanes and explicit logit diagnostics retain full host logits. A
finishing request also downloads its one committed frontier row, preserving the
ability to apply a different grammar when reusing an exact cached frontier.
Prefill retains its existing head path. Shared draft replay and draft-cache
commit still need work to remove their blocking completion waits.

## Focused comparison

One RTX PRO 6000 Blackwell at **400 W, standard memory speed, no overclock**,
four unchanged Sparks, five complete RTX expert layers, KV capacity of
18 × 1,048,576 tokens and 24 retained snapshots. Both arms use the same rebuilt
native library; the control binary is from `f7114ad`. No builds overlap serving.

| Workload | Full host logits | GPU top-1 |
|---|---:|---:|
| C1 target-only code, tok/s | 44.28 | 44.30 |
| C1 dSpark code, tok/s | 128.58 | 129.69 |
| 32K prefill, tok/s | 7,815.36 | 7,450.90 |
| C2 mixed, tok/s | 101.80 | 103.58 |
| C4 mixed, tok/s | 110.02 | 111.77 |
| C8 mixed, tok/s | 155.84 | 170.51 |
| C16 mixed, tok/s | 190.14 | 192.58 |

C1 and prefill are three-sample medians; mixed traffic has one cold batch per
concurrency, including admission gaps. Target-only and dSpark C1 each preserve
exact output in all three matched requests. Mixed traffic can change trajectories
and lengths; these sequential samples do not establish a uniform throughput gain.

The lower initial prefill median prompted a focused reverse-order check with
three samples per arm: control **7,581.66**, candidate **7,680.17 tok/s**. The
direction reverses and the sample ranges overlap. This does not establish a
prefill regression or improvement; preserve both measurements rather than
claiming that the unchanged prefill path became faster. Target-only C1 is flat.

Both arms pass needle retrieval, prompt and retained-turn reuse, cancellation,
survivor and recovery checks. Four concurrent high-thinking tool/strict-JSON
requests pass on the candidate, covering streamed and ordinary responses. The
native test covers seven full-vocabulary rows, negative/tied scores, lowest finite
values, and NaN/positive-infinity/negative-infinity anywhere in a row. Four Rust
score tests and the release build pass.

[Evidence](phase1-compact-head-selection.json) records artifact hashes, commands,
samples, responses, lifecycle and constraint checks. Raw logs and frozen artifacts
are under `~/.cache/ds41rt-experiments/compact-head`, with separate target-only and
reverse-prefill directories. Standard serving is restored after each experiment.
