# Native dSpark numerical and graph qualification

The synthetic qualifier `scripts/qualify-ds41-dspark.py` passed on both local RTX PRO 6000 Blackwell GPUs using the coordinator native library, without loading checkpoint weights or contacting Sparks. It validates the pinned reference `inference/model.py` hash and compares native results against the reference operations expressed with strict FP32 PyTorch matrix products over synthetic BF16 inputs; it does not execute a complete reference Transformer.

Confidence was checked at 1, 16 and 80 rows and Markov projection at 1, 3 and 16 rows. Maximum recorded absolute error across these cases was 7.153e-7, below the fixed 2e-5 absolute/relative tolerances. The composed five-position greedy sequence matched reference tokens exactly and its corrected logits, saved Markov embeddings and raw confidence matched their reference values. A complete native sequence captured through the Python harness passed three greedy replays with changed logits/hidden inputs and an exact stochastic replay. This exercises native kernels, embedding gather and the dedicated cuBLAS workspace in a graph; it does not qualify the Rust terminal owner.

Seed/range reuse and reversing request order preserved sampled outputs exactly. A two-category smoke check yielded 0.682861 for a nominal 0.7 probability over 4,096 deterministic seeded samples on each GPU, within the predeclared 0.04 tolerance. This detects gross distribution errors but does not establish full distribution or tail quality. Both GPUs used the same sampling seeds, so those sample sets are not independent statistical trials. Invalid confidence overlap, excessive Markov rows and an invalid draft position were rejected by native entry points.

Three core RNG tests passed: cancellation does not recycle reserved ranges, exhaustion cannot wrap or advance state, and reservations follow requests across batch order. Rust GPU ownership/allocation/cancellation, graph allocation stability, alternating-wave overlap, upstream shared-head execution and complete serving remain open. The first harness run encountered unsupported PyTorch CUDA UInt64 indexing; the corrected harness uses identical positive 64-bit metadata represented as Int64 for reordering. No production numerical fix was needed.

Reproduce inside the coordinator build environment with:

```bash
python3 scripts/qualify-ds41-dspark.py --native-lib /audit/cmake/libds41rt_native.so --reference-dir /reference --devices 0,1 --output /audit/dspark-qualification.json
cargo test --manifest-path rust/Cargo.toml -p ds41rt-core dspark_rng
```

The native library/reference mounts and their hashes must match the recorded `ds41-dspark-qualification.json`; rebuilding unrelated native sources changes the library hash. Temporary outputs are listed in `TO_DELETE_SCAFFOLDING.md`, while the reusable qualifier stays in the repository.
