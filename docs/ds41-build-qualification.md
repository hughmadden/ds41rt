# Build and naming qualification

The engine now uses `ds41rt`, `DS41RT_*`, `Ds41rt*`, `ds41rt.config`, and ds41 container/image names across first-party code, Rust crates, native ABI, Python packages, scripts, and documentation.

V4 algorithm names remain until their implementations are replaced, and third-party sources and provenance hashes are unchanged.

Protocol V2 uses the eight-byte `DS41RTE2` signature consistently in Rust and native validation, correcting the inherited native validator's stale GLM signature; debug frames use `DS41RF01`.

`./build.sh --spark-hosts ostrich,dodo` restricts build validation and image distribution to the available hosts without changing the runtime's four-rank topology, and dirty builds generate source manifests automatically.

## Verified on 2026-09-10

- Shell syntax checks pass for build, run, WIP, stop, release helpers, and the image-distribution helper.
- All 59 selected release/WIP identity tests pass after renaming.
- All eight build-host selection cases pass, including partial fleets, duplicates, unknown hosts, and empty entries.
- CPU native library compilation succeeds and both native ABI and XGrammar self-tests pass.
- Rust API, core, FFI, loader, and transport suites pass 497 tests with one ignored test using Python 3.12 and native auto-discovery disabled.
- The existing daemon host-test profile passes 608 tests but six V4 planner/capture tests fail because the selected Python 3.12 lacks Torch, with 17 ignored and 113 filtered tests.

## Remaining gates

Container builds, GPU/native CUDA execution, the six Torch-dependent tests, and full run/restart/stop checks remain pending and are not covered by the CPU results above.

The host default Python 3.14 exceeds the pinned PyO3 version's supported range; select Python 3.12 using `DS41RT_PYTHON="$(uv python find 3.12)" scripts/run-with-python-env.sh COMMAND` for current Rust qualification.

An attempted unfiltered daemon run against the CPU native library correctly rejected CUDA-dependent tests, so that run does not qualify GPU behavior.
