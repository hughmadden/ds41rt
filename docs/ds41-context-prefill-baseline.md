# 16k context baseline

Native serving now accepts `--max-context-tokens` (default 32768) and reserves compressed cache pages for all sixteen slots at startup. Each slot is rounded separately to 256-row pages, respecting ratio-two sources at layers 2/8/14 and ratio-one source 20. Fixed FP8 local windows remain unchanged. This removes the previous 4096-token admission/pool ceiling; it does not establish concurrency or maximum-context qualification.

The pure geometry test covers uneven page boundaries, sixteen fully provisioned slots, maximum supported cache geometry and invalid limits. The release daemon builds. The test binary requires the coordinator image's Python shared library and passes there.

Live baseline uses the existing 80-row prefill batches, unchanged native library and four unchanged RoCE expert workers. One sequential request per mode/context, repeated `amber` filler followed by counting 1 through 20, greedy generation, 59 completion tokens, no prefix-cache hits. Both 16k requests return the expected counting output; subsequent 4k requests exercise release and reuse. This synthetic workload is a status measurement, not broad quality or representative corpus throughput.

Artifacts: `/tmp/ds41-context/{daemon,create-commands.json,measure.py,measure.log,results.json}`. Deployed daemon SHA256: `f05ff9de6c9b3b14e779eb82bafb8d4aeb32ec9e684692206e5163bcfdf060e4`. Native library SHA256: `cf202558ee672d1c34a273e9c62d09b1ca8f6bad1367f1cae2c63ceedc5f3560`.

Time to first content includes API/tokenization and the first generated token; prompt tokens divided by this time is effective prefill throughput, not isolated GPU prefill. Decode excludes time through first content and includes EOS/HTTP overhead. Larger prefill batches and detailed stage attribution remain next work; no claim that low active parameter count alone guarantees a throughput target.

| Prompt tokens | Mode | TTFT (s) | Effective prefill tok/s | Decode tok/s |
|---:|---|---:|---:|---:|
| 16411 | target | 24.536 | 668.8 | 25.95 |
| 16411 | speculative | 24.343 | 674.1 | 88.24 |
| 3977 | target | 5.507 | 722.2 | 26.41 |
| 3977 | speculative | 5.397 | 736.9 | 94.75 |
