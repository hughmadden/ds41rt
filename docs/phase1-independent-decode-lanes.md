# Independent decode lane candidate

`--independent-decode-lanes` is an experimental opt-in scheduler. It supports
fixed drafts, the independent confidence cutoff and lane-local incremental reuse,
and also target-only execution. The cross-lane `--dspark-adaptive` cost policy is
incompatible. Defaults are unchanged; runtime qualification is pending.

Each lane owns its target pass and transport and loops through preparation,
verification, commit, token delivery and its next round independently. Request,
draft and prefix state use short synchronous borrows on the CUDA owner thread.
The common target execution body releases its request-bank borrow after preparing
a layer and before awaiting its remote FFN. Prepared layer state retains its
lane buffers; no cache-bank reference or RefCell guard crosses that await.
The ordinary paired path uses the same numerical operation sequence.

Independent commits share the existing greedy verification, grammar handling,
draft-window update, accepted-route observation and retained-logit code. Client
sends are asynchronous outside all shared-state borrows. A fixed three-element
chunk array avoids adding per-token heap allocation. The paired scheduler still
uses blocking sends outside its executor, as before.

There is no decode-round join. Lanes request a common drain when a request
finishes/disconnects or queued input can use a free slot. This preserves the
existing admission/prefill and owner migration boundary: prefill uses both lanes.
A lane with in-flight work finishes/discards it before returning. Both futures
are drained on errors; neither future is cancelled merely because its peer
failed. The outer scheduler retains its existing transport reset and request
cleanup behavior. A yield after each committed round lets ready peer work run.

Debug target `ds41rt::lane_schedule` records verifier issue and commit events by
lane and lane-local round number, allowing a trace to demonstrate the next issue
before the peer's prior commit. The counters restart after each common drain.

The daemon check and release build pass. The serving screen in
`/tmp/ds41-independent-lanes` compares paired and independent scheduling with
identical 0.05–0.5 reuse policy, Rust binary and native library. It includes code,
32K prefill, C4/C16 mixed work, retained-context needle/reuse, cancellation and
recovery checks. Scheduling diagnostics are enabled; this first screen is not
an uninstrumented performance adoption claim.

Hardware is one RTX PRO 6000 Blackwell, **400 W and standard memory speed**,
four unchanged Sparks, five complete local expert layers, 18 × 1,048,576-token
KV capacity and 24 retained snapshots. No builds overlap timed serving.

The initial launch failed while copying artifacts because the 92 GiB `/tmp`
tmpfs was full, before inference. Standard serving was restored and verified.
The experiment and old `ds41-attention-output-rtx1` fixture were moved to
`~/.cache/ds41rt-experiments`, preserving `/tmp` paths via symlinks and freeing
7.4 GiB. Artifact copies were repaired and hashes matched before relaunch. The
runner now writes its initial manifest before stopping the service and covers
the stop with its restoration guard. Both comparison arms run after this host
memory change; prior loading timings are not directly comparable controls.
