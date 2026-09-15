# STATUS — test-coverage/upstream-port campaign

## Phase 1 — DONE (commit cdb4770)
Inventory: 25 slices, 1916 rows (~800 PORT-CPU / ~615 DEFER-GPU / ~500 SKIP).
UPSTREAM-INVENTORY.md, PORT-PLAN.md (904 rows by component), DEFERRED.md (695 rows,
UC-keyed + semantic-parity questions), BASELINE.md (pg env reproduction).

## Phase 2 — DONE: all 12 port waves landed and committed
| Area | File | Tests |
|---|---|---|
| C1 streaming/SSE | api/src/tests/upstream_streaming.rs | 13 |
| C1 protocol errors | api/src/tests/upstream_protocol_errors.rs | 10 |
| C1/C4 stop strings | api/src/tests/upstream_stop_strings.rs | 8 |
| C12 tool calls | api/src/tests/upstream_tool_calls.rs | 23 |
| C3 sampler (rust) | api/src/tests/upstream_sampler.rs | 35 |
| C6 spec acceptance | core/tests/upstream_spec_acceptance.rs | 25 |
| C5 hostcache eviction | hostcache/tests/upstream_eviction_invariants.rs | 10 |
| C9 transport faults | transport/tests/upstream_fault_injection.rs | 14 |
| C8 admission | daemon/tests/admission_upstream_port.rs | 39 |
| C10 containers | loader/tests/upstream_container_invariants.rs | 11 |
| C3 sampler oracle | python/tests/test_upstream_sampler_reference.py | 52 |
| C10 quant config | python/tests/test_upstream_quant_config.py | 39 |
| C10 FP4/FP8 pack math | python/tests/test_upstream_fp4_pack_math.py | 40 |
| C2 json-schema grammar | python/tests/test_upstream_json_schema_grammar.py | 58+7 skip |

Final python suite: 443 passed, 7 skipped (documented xgrammar 0.2.6 limits).
Final rust workspace: see BASELINE.md known-env failures only (6, UC-5).

## Bug fixes (confirmed by ported upstream tests, minimal)
1. api: unbounded 400 error body (JsonRejection echo; vllm #49239 class) -> 512-char bound.
2. api: stop-string selection by earliest start -> earliest completion (vllm check_stop_strings).
3. loader: safetensors header accepted arbitrary tensor rank -> max-rank 32 (gguf-py class).

## Triage notes
- a67f01f swept four in-progress agent drafts; each was superseded and fixed in
  later commits (54e4707 admission note, transport file, sampler, hostcache).
- DSML parser divergences + unset-temperature-greedy: DEFERRED.md semantic-parity
  questions 1-4 (UC-3 fleet window).
- xgrammar 0.2.6 unsupported invariants: skip-recorded in the C2 test file.
