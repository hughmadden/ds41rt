# Build and naming qualification

The engine now uses `ds41rt`, `DS41RT_*`, `Ds41rt*`, `ds41rt.config`, and ds41 container/image names across first-party code, Rust crates, native ABI, Python packages, scripts, and documentation.

V4 algorithm names remain until their implementations are replaced, and dependency provenance is updated separately when native V4.1 support is added.

Protocol V2 uses the eight-byte `DS41RTE2` signature consistently in Rust and native validation, correcting the inherited native validator's stale GLM signature; debug frames use `DS41RF01`.

`./build.sh --spark-hosts ostrich,dodo` restricts build validation and image distribution to the available hosts without changing the runtime's four-rank topology, and dirty builds generate source manifests automatically.

## Verified on 2026-09-10

- Shell syntax checks pass for build, run, WIP, stop, release helpers, and the image-distribution helper.
- All 59 selected release/WIP identity tests pass after renaming.
- All eight build-host selection cases pass, including partial fleets, duplicates, unknown hosts, and empty entries.
- CPU native library compilation succeeds and both native ABI and XGrammar self-tests pass.
- Rust API, core, FFI, loader, and transport suites pass 497 tests with one ignored test using Python 3.12 and native auto-discovery disabled.
- After installing cached PyTorch 2.13.0+cu130 into the Python 3.12 environment, the daemon host-test profile passes 609 tests but five V4 planner tests still require the pinned CUTLASS package, with 17 ignored and 113 filtered tests.

## Remaining gates

The CPU results above do not cover container or GPU execution; subsequent successful build/CUDA evidence is recorded below, while the five CUTLASS-dependent tests and full run/restart/stop remain open.

The host default Python 3.14 exceeds the pinned PyO3 version's supported range; select Python 3.12 using `DS41RT_PYTHON="$(uv python find 3.12)" scripts/run-with-python-env.sh COMMAND` for current Rust qualification.

An attempted unfiltered daemon run against the CPU native library correctly rejected CUDA-dependent tests, so that run does not qualify GPU behavior.

## CUDA native library follow-up

A native CUDA build using `/usr/local/cuda-13.3/bin/nvcc` and architecture 120 now succeeds, with all three native/CUDA/XGrammar self-tests passing; the CPU build also still passes both self-tests.

The new engram CUDA primitives are qualified on both local RTX GPUs as recorded in `ds41-engram-cuda-qualification.json`, while container, Spark, and full serving qualification remains open.

## Successful native container build, 2026-09-10

`./build.sh --spark-hosts ostrich,dodo` completed with exit status zero, automatically generating source manifest `981dd404ee3f7e6653324f37e2307d75f5251ac86709160ae2b37879a2f5deb4` for the dirty checkout at `575a9df2d47631371d7c766ba55dfd975b1b7ebd`.

Both coordinator and Spark development/inference images compiled successfully, including Rust and CUDA/AOT binaries, and release artifacts with source/provenance checksums were exported to `dist/`.

The Spark inference image was built natively on ostrich and distributed to dodo over RDMA; no emu or kiwi access was used.

The coordinator image and both deployed Spark images launch `ds41rt --help` successfully, and coordinator `doctor` sees the RTX and CUDA driver; its absent Rust compiler and unresolved default placeholder Spark hostnames are not full-stack qualification.

`ds41-container-build-qualification.json` records the image IDs, architectures, revision labels, and source identity.

This successful build precedes the new strict configuration reader, and the default model configuration/execution path still needs migration from V4; full run/restart/stop and official checkpoint execution remain open.

The later native-expert precision migration uses signature `DS41RTE3` and wire revision 3, so the earlier container build recorded here must be replaced before deploying the updated transport.
