# Upstream-derived test coverage

Ports of upstream unit-test coverage onto ds41rt harnesses, plus the small
product fixes those tests exposed. Upstream sources (immutable):

| Repo | Commit |
|---|---|
| vllm-project/vllm | `64856080b05054fc62754f37234f7b483b753445` |
| sgl-project/sglang | `832ec39cc0324cb0e7823dc8385e27a30c356bdd` |
| ggml-org/llama.cpp | `6011c34ce6099646ccdf0d39a61c6e681477c178` |

## What was ported

| Area | Harness | Upstream sources | Count |
|---|---|---|---|
| OpenAI protocol errors, streaming/SSE, stop strings, tool calls | `ds41rt-api` (legacy router) | vllm entrypoints suites, llama.cpp server patterns | 55 |
| Protocol/streaming invariants on the **production native router** | `ds41rt-api` `upstream_native_v41.rs` | vllm non-object-body, llama.cpp SSE contract | 5 |
| Sampler param validation + logits-processor reference | `ds41rt-api` | vllm sample/, logits_processors/ | 35 |
| Spec-decode acceptance math + bookkeeping | `ds41rt-core` | vllm spec_decode/, sglang spec/ | 25 |
| Hostcache eviction / radix invariants | `ds41rt-hostcache` | vllm core prefix-cache, sglang radix tree | 10 |
| Transport fault injection | `ds41rt-transport` | sgl-router failover/timeout/drain | 14 |
| Admission control | `ds41rt-daemon` | vllm engine admission, sglang scheduler | 39 |
| Container adversarial invariants | `ds41rt-loader` | llama.cpp gguf-py, test-gguf | 11 |
| Sampler reference oracle (llama.cpp exact vectors) | `python/tests` | llama.cpp test-sampling.cpp | 52 |
| FP4/E4M3/FP8 pack-math oracle | `python/tests` | vllm nvfp4/per-token-group quant | 51 |
| Quant-config validation | `python/tests` | vllm quantization config args | 39 |
| JSON-schema→grammar semantics | `python/tests` | llama.cpp test-json-schema* | 58 (+7 skip) |

## Running

```bash
# Rust (CPU-only; no GPU or weights required)
cargo test --workspace --no-fail-fast

# Python
cd python && pip install -e '.[test]'   # pytest, numpy, xgrammar==0.2.6
python -m pytest tests/
```

## Coverage classes

Two classes are labeled per file — per GROUP for mixed suites — and counted
separately (re-review 2026-09-16):

1. **Product regression coverage** (~250 tests) — exercises ds41rt code;
   failures mean product regressions.
2. **Standalone reference oracles** (~140 tests) — document upstream behavior
   with no ds41rt dependency (llama.cpp sampler vectors, FP4/FP8 pack math,
   contract-model groups in transport/admission/spec suites). They are the
   comparison references for the deferred GPU-parity tests and cannot detect
   product regressions alone.

## Skips and known failures

- `test_upstream_json_schema_grammar.py` skips 7 invariants xgrammar 0.2.6
  cannot enforce (`pattern`, `minLength`, `maxLength`, allOf merging,
  `uniqueItems`, `not`); re-check on a submodule bump. The module
  `pytest.importorskip`s xgrammar so a clean env skips rather than errors.
- Six pre-existing daemon tests fail identically on both revisions (the
  Python planner expects a `b12x` API — `dsa_indexer.SOURCE_LAYOUT_PAGED` —
  the v3-pinned sparkinfer `3882b935` does not provide; pre-dates this
  change): `flash_native_nvfp4_target_storage_uses_exact_compact_physical_pages`,
  `flash_smoke_target_storage_pads_c128_indices_to_sm120_prefill_tile`,
  `flash_target_mhc_is_bounded_by_active_frontier_not_context`,
  `flash_target_storage_keeps_one_workspace_and_persistent_hc_kv`,
  `target_graph_metadata_is_bounded_by_sequence_not_shared_pool`,
  `real_checkpoint_nvfp4_decode_matches_python_fixture`.

## Product fixes included (each verified by the ported tests)

1. `api`: 400 error bodies bounded (JsonRejection, native-router serde
   errors, and ApiError messages — validly typed huge fields included).
2. `api`: stop-string selection by earliest *completion* position with
   list-order ties (vllm `check_stop_strings` semantics).
3. `loader`: safetensors tensor rank bound (policy max 32).
4. `daemon`: dual-RTX serving passes the shared `/v1/stats` handle through
   (v3 merge repair had dropped it).

## Deferred

Fleet/GPU-dependent ports (kernel parity, sampler↔native parity,
FP4 wire-format parity against checkpoint tensors, hostcache serve-path
validation) are planned against a discrete-event census of the upstream
suites; not part of this change.
