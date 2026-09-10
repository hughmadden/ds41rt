# Temporary modelless scaffolding

Delete listed artifacts at the start of Goal 2 after preserving any reusable qualification tooling.

- Delete `/tmp/ds41-meta-audit/` generated full reference parameter inventories after native checkpoint catalog qualification; retain `scripts/audit-ds41-reference-parameters.py` as reusable audit tooling.

- Delete `rust/crates/ds41rt-loader/examples/qualify_v41_mapped_fixture.rs`, `/tmp/ds41-native-header-fixture/`, and `/tmp/ds41-native-catalog-adversarial.py` after Goal-2 checkpoint-backed loader qualification; retain the production catalog contract, its regression tests, and the `v41_catalog` inspection example.

- Delete `/tmp/ds41-expert-qualification/`, `/tmp/ds41-fix-expert-accum.py`, and the `/tmp/ds41-v41-moe.*` staging directories on ostrich/dodo after Goal-2 checkpoint-backed expert qualification; retain the native expert regression suite and recorded qualification results.

- Delete `/tmp/ds41-v41-aot/` and ostrich's `/tmp/ds41-v41-aot-src.odALRiKi/`, including the temporary C-entry qualification shim, after the production Rust expert binding replaces that fixture; retain the AOT exporter, CMake integration, and recorded qualification evidence.

- Delete `/tmp/ds41-v41-rust-bridge/` and `/tmp/ds41-v41-abi-mutations.ggwbi658/`, plus the Rust cdylib and Python bridge fixtures inside the existing local/ostrich AOT staging directories, after production native weight/scratch preparation replaces them; retain the C/Rust bridge and recorded evidence.
