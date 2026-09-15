# BASELINE — pre-port test state on pg (2026-09-14 AEST)

Host: pg (32-core x86-64, no GPU, 124 GB RAM). Worktree `~/dev/ds41rt-testcov-wt`,
branch `test-coverage/upstream-port` at `5173e44`.

## How to reproduce

```bash
cd rust
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=~/dev/ds41rt-persistence/rust/target   # reuse build cache
# torch (CPU) in an isolated dir; system python is PEP-668 externally managed
export PYTHONPATH=$HOME/.local/share/ds41rt-testdeps:$PWD/../python/reference
cargo test --workspace --no-fail-fast
```

Python: `~/dev/ds41rt-testcov-wt/.venv` (uv) with pytest/tokenizers/pydantic; system
python3.12 + `~/.local/share/ds41rt-testdeps` (CPU torch 2.14.0) for the daemon's
embedded pyo3 planner.

## Results

| Suite | Result |
|---|---|
| ds41rt-core | 124 passed, 0 failed |
| ds41rt-api | 146 passed, 1 ignored |
| ds41rt-daemon | 773 passed, **6 failed (all pre-existing environment limits, not product bugs)**, 40 ignored |
| ds41rt-ffi / loader / transport / hostcache (+ integration tests) | all green (770+ tests incl. hostcache HC-2..HC-13 concurrency/functional/soak suites) |
| python (`python/tests`, venv) | 254 passed, 0 failed — requires `git submodule update --init --recursive` (sparkinfer/xgrammar pins) |

## The 6 pre-existing daemon failures (environment-limited)

1. `commands::real_full::coordinator_kernels::target_attention::tests::{flash_native_nvfp4_target_storage_uses_exact_compact_physical_pages, flash_smoke_target_storage_pads_c128_indices_to_sm120_prefill_tile, flash_target_mhc_is_bounded_by_active_frontier_not_context, flash_target_storage_keeps_one_workspace_and_persistent_hc_kv, target_graph_metadata_is_bounded_by_sequence_not_shared_pool}` — the Python planner
   module lazily imports the `b12x` reference package, which is only installed on the
   GPU dev hosts. **Unblock: UC-5** (install `b12x` or run on a dev host that has it).
2. `commands::real_full::sparse_mlp::math::tests::real_checkpoint_nvfp4_decode_matches_python_fixture` —
   needs `tests/fixtures/nvfp4/real_tensor_decode.json`, a fixture generated from a real
   checkpoint (tool: `python/tools/verify_nvfp4_modelopt_real_tensor_decode_fixture.py`).
   Not in git history; **Unblock: UC-5** (generate on a weights host).

These 6 are recorded as known-environment failures; any NEW failure in the ported
suites is triaged against this baseline.
