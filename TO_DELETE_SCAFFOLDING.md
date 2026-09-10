# Temporary modelless scaffolding

Delete listed artifacts at the start of Goal 2 after preserving any reusable qualification tooling.

- Delete `/tmp/ds41-meta-audit/` generated full reference parameter inventories after native checkpoint catalog qualification; retain `scripts/audit-ds41-reference-parameters.py` as reusable audit tooling.

- Delete `rust/crates/ds41rt-loader/examples/qualify_v41_mapped_fixture.rs`, `/tmp/ds41-native-header-fixture/`, and `/tmp/ds41-native-catalog-adversarial.py` after Goal-2 checkpoint-backed loader qualification; retain the production catalog contract, its regression tests, and the `v41_catalog` inspection example.

- Delete `/tmp/ds41-expert-qualification/`, `/tmp/ds41-fix-expert-accum.py`, and the `/tmp/ds41-v41-moe.*` staging directories on ostrich/dodo after Goal-2 checkpoint-backed expert qualification; retain the native expert regression suite and recorded qualification results.

- Delete `/tmp/ds41-v41-aot/` and ostrich's `/tmp/ds41-v41-aot-src.odALRiKi/`, including the temporary C-entry qualification shim, after the production Rust expert binding replaces that fixture; retain the AOT exporter, CMake integration, and recorded qualification evidence.

- Delete `/tmp/ds41-v41-rust-bridge/` and `/tmp/ds41-v41-abi-mutations.ggwbi658/`, plus the Rust cdylib and Python bridge fixtures inside the existing local/ostrich AOT staging directories, after production native weight/scratch preparation replaces them; retain the C/Rust bridge and recorded evidence.

- Delete `qualify_reduce.py`, `debug_reduce.py`, `qualify_full_expert.py`, their logs/graph dumps under `/tmp/ds41-v41-aot/`, and the copied ostrich reduction fixture with the existing AOT scaffolding after checkpoint-backed execution replaces these synthetic checks.

- Delete `/tmp/ds41-v41-scratch-abi.iaz5zkwn/`, `/tmp/ds41-v41-scratch-spark-abi/`, and `check_scratch_layout.py` / `qualify_native_scratch.py` in the local/ostrich AOT staging trees after checkpoint-backed expert execution replaces the native scratch fixtures; retain the production exporter, C/Rust APIs and evidence.

- Delete `qualify_pack.py` / `qualify_native_weights.py` and their synthetic-data logs in the existing local/ostrich AOT staging directories after checkpoint-backed expert preparation replaces them; retain native packing APIs and recorded qualification evidence.

- Delete `/tmp/ds41-staged-packer/`, its `ds41-staged-packer-fixture` binary in `rust/target/debug/`, and `staged-pack/`, `prepare_staged_fixture.py`, `check_staged_pack.py`, `staged-fixture.json` and related logs under `/tmp/ds41-v41-aot/` after checkpoint-backed expert loading replaces them; the existing sparse-header fixture now also contains selected synthetic expert payloads and remains scheduled for deletion.

- Delete `/tmp/ds41-owned-weights/`, its binary in `rust/target/debug/`, `owned-weights/`, `prepare_owned_fixture.py`, `check_owned_weights.py`, `owned-fixture.json`, `allocation_audit.cc`, `liballocation_audit.so` and associated logs under `/tmp/ds41-v41-aot/`, plus `owned-weights-src/`, `owned-fixture/` and corresponding fixture artifacts in ostrich's existing AOT staging tree after checkpoint-backed layer execution replaces them; retain production layer ownership and recorded evidence.

- Delete `/tmp/ds41-owned-execution/`, its binary in `rust/target/debug/`, `owned-execution/`, `prepare_owned_execution.py`, `check_owned_execution.py` and related logs under `/tmp/ds41-v41-aot/`, plus corresponding `owned-execution-src/` and fixture artifacts in ostrich's AOT staging tree after checkpoint-backed scheduled execution replaces these two-wave synthetic checks; retain production execution ownership and qualification evidence.

- Delete `/tmp/ds41-native-tp4/`, its binary in `rust/target/debug/`, `tp4-execution/`, `prepare_tp4_execution.py`, `check_tp4_execution.py` and related logs under `/tmp/ds41-v41-aot/`, ostrich's `native-tp4-src/` and temporary worker binary, and dodo's `/tmp/ds41-native-tp4/` after checkpoint-backed TP serving replaces the synthetic network workers; retain production request/response contracts and host adapter.

- Delete `/tmp/ds41-v41-aot/chunk-check.log` and `chunk-daemon-check.log` with the temporary AOT staging tree; these are compilation logs, not runtime qualification evidence.

- Delete `/tmp/ds41-v41-aot/streaming-tcp-check.log` and `streaming-tcp-daemon-check.log` with the temporary AOT staging tree; retain the production TCP streaming interfaces.

- Delete `/tmp/ds41-v41-aot/native-service-check.log` and `native-service-build.log` with the temporary AOT staging tree; retain the production native worker and launch command.

- Delete `/tmp/ds41-v41-aot/native-dispatch-check.log`, `native-dispatch-daemon-check.log` and `native-dispatch-build.log` with the temporary AOT staging tree; retain the production TP dispatch and coordinator wave owners.

- Delete `/tmp/ds41-v41-aot/engram-staging-check.log` and `engram-staging-build.log` with the temporary AOT staging tree; retain reusable mapped-row staging and native memory owners.

- Delete `/tmp/ds41-v41-aot/engram-gather-worker-check.log`, `engram-gather-worker-build.log` and `engram-gather-worker-final-build.log` with the temporary AOT staging tree; retain the production cancellable gather worker and completion polling.

- Delete `/tmp/ds41-v41-aot/engram-pipeline-check.log` and `engram-pipeline-build.log` with the temporary AOT staging tree; retain the request-owned pipeline and history transaction APIs.

- Delete `/tmp/ds41-v41-aot/engram-resident-check.log` and `engram-resident-build.log` with the temporary AOT staging tree; retain bounded coordinator tensor reads, RTX residency and gate ownership.

- Delete `/tmp/ds41-v41-fp8-aot/` and the `v41_fp8/` build artifacts under `/tmp/ds41-v41-aot/cmake/` with the temporary AOT staging trees; retain the production exporter, CMake integration and recorded build evidence.

- Include `bridge-build.log`, `rust-bridge-build.log`, `owned-projection-build.log` and `final-owned-build.log` in the preceding FP8 staging cleanup; these record builds, not numerical qualification.

- Include `/tmp/ds41-v41-fp8-aot/engram-graph-build.log` in the FP8 staging cleanup; retain production graph ownership and its build evidence.

- Include `/tmp/ds41-v41-fp8-aot/dspark-residency-build.log` in the FP8 staging cleanup; retain production dSpark residency and static budget evidence.

- Include `dspark-confidence-native-build.log`, `dspark-confidence-rust-build.log` and `dspark-confidence-final-build.log` under `/tmp/ds41-v41-fp8-aot/` in staging cleanup; retain the production confidence kernel, bindings and owned execution.

- Include `dspark-markov-build.log`, `dspark-markov-owned-build.log` and `dspark-markov-native-build.log` under `/tmp/ds41-v41-fp8-aot/` in staging cleanup; retain production Markov ownership and native handle bindings.

- Include `dspark-terminal-rust-build.log`, `dspark-terminal-native-build.log` and `dspark-terminal-native-final-build.log` under `/tmp/ds41-v41-fp8-aot/` in staging cleanup; retain production terminal composition and sampling bindings.

- Include `dspark-rng-rust-build.log` and `dspark-rng-native-build.log` under `/tmp/ds41-v41-fp8-aot/` in staging cleanup; retain production RNG reservation and fused sampling.

- Delete `/tmp/ds41-v41-aot/dspark-qualification.json` and `dspark-qualification.log`, `dspark-qualification-final.log`, `dspark-qualification-complete.log`, and `dspark-rng-core-tests.log` under `/tmp/ds41-v41-fp8-aot/` with staging cleanup; retain the reusable qualifier, production RNG tests and committed evidence.

- Delete `/tmp/ds41-v41-aot/dspark-head-qualification.json` and `dspark-head-rust-build.log`, `dspark-head-rust-final-build.log`, `dspark-head-rust-qualified-build.log`, `dspark-head-native-build.log`, `dspark-head-qualification.log`, and `dspark-head-qualification-final.log` under `/tmp/ds41-v41-fp8-aot/` with staging cleanup; retain shared-head ownership and qualification tools.

- Delete `/tmp/ds41-v41-aot/dspark-hc-qualification.json` and `dspark-hc-rust-build.log`, `dspark-hc-rust-final-build.log`, `dspark-hc-native-build.log`, and `dspark-hc-qualification.log` under `/tmp/ds41-v41-fp8-aot/` with staging cleanup; retain native mHC kernels, Rust bindings and the reusable qualifier.

- Delete `/tmp/ds41-v41-aot/hc-mixes-qualification.json` and `hc-mixes-rust-build.log`, `hc-mixes-native-build.log`, and `hc-mixes-qualification.log` under `/tmp/ds41-v41-fp8-aot/` with staging cleanup; retain native coefficient generation, owned sublayer boundaries and reusable qualification.

- Delete `/tmp/ds41-tp4-four-host/` on this host, ostrich, dodo, emu and kiwi during Goal 2 cleanup, including sparse snapshot/source archives, current standalone fixture copies, fresh ARM/CUDA build outputs and regenerated oracle artifacts; the four `ds41-tp4-physical-r*` containers were already removed after successful exit.

- Delete `/tmp/ds41-v41-aot/router-qualification.json` and `router-rust-build.log`, `router-native-build.log`, and `router-qualification.log` under `/tmp/ds41-v41-fp8-aot/` with staging cleanup; retain the native router, Rust ownership and reusable qualifier.

- Delete `/tmp/ds41-v41-aot/shared-fp8-rtx0.json`, `shared-fp8-rtx1.json`, and all `shared-fp8-*.log` under `/tmp/ds41-v41-fp8-aot/` with staging cleanup, including failed diagnostic runs; retain the reusable shared-FP8 qualifier, b12x floor regression tests and committed evidence.

- Delete `/tmp/ds41-expert-floor/` on this host and ostrich, including before-fix/input-only source mounts, transferred sources and Spark AOT artifacts, plus `expert-floor-*.log` under `/tmp/ds41-v41-fp8-aot/`; retain production expert regression tests and committed qualification evidence.

- Delete `/tmp/ds41-ffn-boundary/`, the `ds41-ffn-boundary-fixture` binary under `rust/target/debug/`, and `dspark-ffn-*.log` under `/tmp/ds41-v41-fp8-aot/` during Goal 2 cleanup; the sparse checkpoint contains only synthetic payloads, and production FFN ownership and committed evidence must remain.

- Delete `/tmp/ds41-dspark-cache/`, `rust/target/debug/ds41-dspark-cache-fixture`, and `dspark-cache-*.log` under `/tmp/ds41-v41-fp8-aot/` during Goal 2 cleanup; retain production cache ownership, native kernels and committed evidence.

- Delete `/tmp/ds41-projection-owner/`, `rust/target/debug/ds41-projection-owner-fixture`, `/tmp/ds41-v41-aot/dspark-projections-rtx*.json`, and `dspark-projection-*.log` under `/tmp/ds41-v41-fp8-aot/`, including failed initial split-export/build diagnostics; retain production owners, reusable qualification/regression tests and committed evidence.

- Delete `/tmp/ds41-v41-aot/attention-ops-rtx*.json` and `attention-ops-*.log` under `/tmp/ds41-v41-fp8-aot/`, including initial failed decorator and multiply-contraction diagnostics; retain production kernels/bindings, the reusable pinned-reference qualifier and committed evidence.

- Delete `/tmp/ds41-attention-output-owner/`, `rust/target/debug/ds41-attention-output-owner-fixture`, `/tmp/ds41-v41-aot/grouped-output-rtx*.json` and `grouped-output-*.log` under `/tmp/ds41-v41-fp8-aot/`, including initial module-doc and unconverted synthetic-weight diagnostics; retain production code, the reusable qualifier and committed evidence.

- Delete `/tmp/ds41-v41-aot/dspark-attention-rtx*.json` and `dspark-attention-*.log` under `/tmp/ds41-v41-fp8-aot/` with Goal 2 staging cleanup; retain native kernels/bindings, the reusable blockwise reference qualifier and committed evidence.

- Delete `/tmp/ds41-master-merge/` on ostrich, `master-merge-*.log` under `/tmp/ds41-v41-fp8-aot/`, and `/tmp/ds41-v41-aot/master-merge-projections-rtx*.json` during Goal 2 cleanup; retain the dependency pin, merged production/regression code and committed review/evidence.

- Delete `/tmp/ds41-attention-wave/`, `rust/target/debug/ds41-attention-wave-fixture`, `/tmp/ds41-v41-aot/attention-wave-ops-rtx*.json` and `attention-wave-*.log` under `/tmp/ds41-v41-fp8-aot/`, including the initial live-row/capacity admission failure; retain production owners, fused KV kernels, the reusable primitive qualifier and committed evidence.

- Profile-removal editing helpers `/tmp/remove-serve-profiles.py`, `/tmp/fix-kv-entry.py` and `/tmp/ds41-no-profiles-{tests,rust}.log` are disposable after review. No model data was loaded for this change.

- `/tmp/ds41-main-context/` contains disposable sparse synthetic checkpoint shards, copied Rust fixture modules, generators and expected BF16 outputs; remove it and `rust/target/debug/ds41-main-context-fixture` during Goal 2 cleanup.
- `/tmp/ds41-main-context-*.log` records producer compilation and both RTX qualification runs and can be removed after review of the checked-in qualification artifacts.
