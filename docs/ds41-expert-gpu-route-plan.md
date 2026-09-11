# GPU route preparation for slice experiments

b12x `f239d97` adds private `V41RoutePlan`, a three-launch CuTe implementation of stable per-expert route packing, exclusive route/group prefixes, and M16 metadata scatter. Caller-owned capacity buffers remain stable. Live rows come from a device scalar and can change within the same captured graph; no live quantity enters compilation. Invalid expert IDs receive inverse-map value -1, while duplicate routes are preserved. Inactive metadata counts are cleared on every replay. Packed expert-pool offsets use Int64.

RTX qualification: `tests/moe/test_v41_route_plan.py` passes four capacity cases (1, 16, 80, 4096), each replaying eight changing workloads against an exact CPU oracle. Cases cover long single-expert runs, duplicate IDs within rows, invalid IDs, zero rows, changing weights, and returning to one row. Source formatting and Ruff F checks pass.

Command from the root checkout:

```sh
docker run --rm --gpus 'device=0' --entrypoint bash \
  -v /home/tj/Developer/ds41rt/third_party/sparkinfer:/src -w /src \
  ds41rt-coordinator-dev:latest \
  -lc 'PYTHONPATH=/src python -m pytest -q tests/moe/test_v41_route_plan.py'
```

This is metadata-only correctness evidence on RTX, with no performance claim. Spark qualification, integration with the candidate inverse-map slice reduction, combined planner/compute timings, and native export/dispatch remain outstanding. Existing official comparison timings still exclude candidate route preparation. Serving binaries are unchanged.

Tiling decisions should prioritize the observed speculative-decode distribution of expert-local M and active expert count, retaining prefill throughput as a separate guardrail. A dedicated prefill configuration remains possible; neither the current low-M results nor the 4096-row metadata test establish prefill GEMM performance. The initial prefix stage uses a simple per-expert prefix summation and the pack scratch scales as experts × route capacity; both need measured assessment before production use.
