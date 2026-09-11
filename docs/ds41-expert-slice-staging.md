# Exact 64/128/192 weight staging for the tiling experiment

b12x `a86b722` adds CuTeDSL staging primitives for both sides of a fused expert intermediate slice:

- FC1: N64/N128/N192 by K128, copied into compact lane-major shared storage.
- FC2: N128 by K64/K128/K192, with packed UE8M0 scale words gathered at the correct byte offsets.

Both support 32-aligned starts that cross the resident N256/K128 tile boundaries. All global offset products use 64-bit arithmetic. Weight payload is copied at the requested width; only the last scale word is padded to four bytes for MMA byte selection. No activation quantization or weight conversion is performed by these primitives.

The current resident format is unchanged. These helpers are not yet called by the serving GEMM. They are a prerequisite for decoupling its currently coupled 128-wide intermediate staging and FC2 contraction, not evidence of a faster GEMM or a padding-free resident allocation.

## GPU qualification

`tests/moe/test_w4a8_slice_staging.py` passes **12 tests on RTX SM120 and 12 on Spark SM121**. Across both devices, the tests exercise 144 changed-data/position graph replays using already-compiled callables. They compare every copied weight word and packed scale word against an independent scalar interpretation of the layout, check shared-memory guard regions, and require stable allocated bytes during replay.

The cases include N and K boundary crossings, independently padded W13 half positions, K starts inside a packed scale word, return to earlier positions with changed payloads, and sparse allocations whose N-tile × K-stride × tile-size addresses exceed `INT32_MAX` words. The large-offset cases reserve approximately 8 GiB for weight storage without initializing the entire allocation. They test the actual address arithmetic rather than merely inspecting its types.

Commands use `ds41rt-coordinator-dev:latest` on RTX GPU 0 and `ds41rt-spark-expert-dev:latest` on ostrich, with the b12x source mounted read-only and its path supplied in `PYTHONPATH`:

```sh
python -m pytest -q -p no:cacheprovider \
  /workspace/ds41rt/third_party/sparkinfer/tests/moe/test_w4a8_slice_staging.py
```

Results: `/tmp/ds41-slice-staging-rtx.log` (12 passed, 3.71 s) and `/tmp/ds41-slice-staging-spark.log` (12 passed, 4.08 s). The final commit only relocates an import and clarifies a docstring after those runs; executable behavior is unchanged. This is functional qualification, with no bandwidth or latency claim. The full-model worker on ostrich was temporarily stopped for allocation capacity; its retained binary still uses b12x `6adffce`.

Next, connect these helpers to separate FC1 N and FC2 K tile geometry, preserve the V4.1 quantization boundaries, and compare the common mixed-expert FP8 decode workload before selecting a resident format. The [intermediate-split result](ds41-expert-intermediate-split-probe.md) remains the warning against choosing solely from the one-row case.
