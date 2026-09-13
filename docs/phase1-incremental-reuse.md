# Lane-local incremental expert reuse policy

The candidate adds `--dspark-reuse-floor LOW` to
`--dspark --dspark-confidence-cutoff HIGH`. LOW must be positive and no greater
than HIGH. This opt-in experiment leaves defaults unchanged and cannot be
combined with the existing joint cost policy.

Each lane starts with all mandatory anchors and its qualified minimum of one
draft per request where available. Then it walks draft positions in increasing
order, considering requests in their stable lane order at each position. For a
candidate row, it measures the additional predicted unique experts averaged
across forty layers, divides by six, and interpolates the cumulative-confidence
cutoff from LOW (complete reuse) to HIGH (six new experts per layer). Admission
updates the expert counts immediately. Rejection restores counts and prevents
that request's later positions from being admitted. This is order-dependent
incremental selection, not an optimal search.

The forecast uses bounded accepted target route history, separately by request,
layer and lane. It does not know future target routes. Missing history retains
full available prefixes for that lane; it never waits for another lane. A lane
containing grammar-constrained work keeps its existing grammar/fixed truncation.
The policy collects accepted route history but has no cross-lane cost term.
A positive floor represents nonzero work for reused experts; the current screen
does not separately price a kernel bucket crossing or expert route-group boundary.

Core tests verify that an admitted row immediately discounts a later request,
that rejected rows add no reusable experts, and that cross-lane inputs and invalid
cutoffs are rejected. The initial test build needed an explicit slice coercion in
the fixtures; corrected tests pass. The daemon check and release build pass.
The minimum-output case with no available drafts uses an empty confidence vector.

The policy is selected before this lane's Engram preparation, but the existing
verifier/commit round barrier remains. Removing that barrier is separate ongoing
scheduler work. No claim of fully independent lane execution is made here.

The frozen serving screen is running in `/tmp/ds41-reuse-policy`, with joint-cost
controls before/after and LOW/HIGH pairs 0.05/0.5 and 0.10/0.5. Each arm uses the
same native library and Rust binary, three no-thinking code samples, a 32K prefill
warmup and C4/C16 mixed completion checks. Hardware: one RTX PRO 6000 Blackwell,
**400 W, standard memory speed**, four unchanged Sparks, five complete local
expert layers, 18 × 1,048,576-token KV capacity and 24 retained snapshots. No
builds overlap timed inference; the runner restores standard serving on exit.
Performance and serving qualification of this candidate remain pending.
