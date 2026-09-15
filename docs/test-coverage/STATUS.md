# STATUS — test-coverage/upstream-port campaign (updated as phases land)

## Phase 1 — DONE (commit cdb4770)
- Inventory: 25 slices, 1916 rows (~800 PORT-CPU / ~615 DEFER-GPU / ~500 SKIP).
  UPSTREAM-INVENTORY.md, PORT-PLAN.md (904 rows by component), DEFERRED.md (695 rows, UC-keyed).
- Baseline on pg: workspace green except 6 pre-existing env-limited daemon tests
  (5 need `b12x` pkg, 1 needs real-checkpoint fixture — both UC-5). See BASELINE.md.
- Env: venv at .venv (pytest/tokenizers/pydantic); CPU torch at ~/.local/share/ds41rt-testdeps;
  system python PEP-668 so torch is NOT system-installed — always set
  PYTHONPATH=$HOME/.local/share/ds41rt-testdeps:<repo>/python/reference for daemon tests.

## Phase 2 — IN PROGRESS: port waves
Wave 1 (running): api protocol errors / streaming conformance / sampler (rust+python) /
hostcache eviction invariants / spec-decode acceptance math.
Wave 2 (running): tool-call semantics / stop strings / transport fault injection /
admission control / quant+container invariants.
Agents write new files only; orchestrator wires `mod` lines into ds41rt-api/src/tests.rs
(and core tests.rs if needed) centrally after each wave, then runs full suites.
Rules: no product-code fixes by port agents — suspected bugs become #[ignore =
"BUG(candidate): ..."] + reported; orchestrator triages and fixes centrally.

## Phase 3 — pending: triage + fixes + final commit
## Phase 4 — pending: fleet-clear handoff (DEFERRED.md already committed)
