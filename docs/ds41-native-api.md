# First live native V4.1 API

`ds41rt serve-native` connects the official checkpoint's target pass to `/v1/chat/completions`, `/v1/models` and `/health`. The new path uses DeepSeek's `deepseek-recipe` and `deepseek-recipe-encoding` Rust crates pinned to 0.1.0 for V4.1 request formatting and output parsing, including thinking/tool syntax. It does not route through the old V4 full backend. The registry lockfile also resolves tokenizers 0.23.2; all loader engram tests pass with it.

One dedicated thread owns the CUDA allocations and the current-thread execution runtime. A bounded 16-job channel connects HTTP handlers to that owner. The server initializes weights, caches, engram and TP4 transport before binding HTTP. Each request gets a fresh cache/engram lease; prefill is chunked at 80 rows, followed by full target decode passes. Outputs use the streaming tokenizer decoder so byte-fragment tokens are not decoded independently. Request tokens publish through the coupled cache/engram commit. Client disconnects are checked before/after passes and failed proposals are discarded; request admission is released when generation ends.

JSON and SSE use the same official response parser. SSE honors requested usage and ends with `[DONE]`. Worker errors or missing finish metadata cannot become successful EOF completions. Prompt/output text is not logged by the server. The current endpoint accepts the exact model ID `deepseek-ai/DeepSeek-V4.1-Flash`.

## Actual five-host results

The production daemon served an HTTP request for “What is 2 + 2? Answer with just the number.” and returned `4`, stop, 18 prompt tokens and 2 completion tokens. Three SSE runs returned the exact comma-separated sequence 1 through 20. All streams supplied requested usage and normal completion. Disconnecting a counting stream after its first content was followed by a successful fresh `2 + 2` request. Nonzero-temperature input returned HTTP 400. These are real target executions on RTX plus four resident all-layer Spark workers.

| Counting stream | First content latency | Observed decode tokens/s |
| --- | ---: | ---: |
| First shape warmup | 1.332 s | 5.361 |
| Repeat 1 | 1.325 s | 5.845 |
| Repeat 2 | 1.311 s | 5.813 |

The request has 21 prompt tokens and 59 locally counted completion tokens, including EOS. Decode rate is `(completion_tokens - 1) / (finish_event_time - first_content_time)`. It includes transport/HTTP overhead and EOS, excludes time through first content, and uses a Cargo dev build with one client. These are initial short-context API observations, not a release benchmark or evidence for 90/270/8000 TPS. The approximately 5.8 TPS repeated result establishes a substantial optimization gap; the layer/transfer/capture timing breakdown is still required.

The user-authorized hosted API returned the same visible answers for both prompts using `deepseek-flash`, temperature 0 and thinking disabled. The host model-list endpoint exposed `deepseek-flash` and `deepseek-v4-pro`; it did not establish an exact checkpoint revision. Hosted usage excludes the terminating token in these responses (1/58 completion tokens versus local 2/59). Do not interpret visible text agreement as matching logits or weights. No credential is included in either evidence file.

- [Local requests, SSE events, timing and hashes](ds41-native-api-qualification.json)
- [Hosted requests and responses](ds41-native-api-hosted-comparison.json)
- All 108 API unit tests pass, including new prompt/JSON/SSE and worker-failure checks; all three focused loader engram tests pass.

## Development entry point

The current local server is at `http://127.0.0.1:18041`, with a `ds41-native-api-dev` container and four `ds41-full-target-worker` Spark containers. Recheck their live state before reuse. The daemon command is:

```sh
ds41rt serve-native --native-lib /path/to/libds41rt_native.so \
  --snapshot /path/to/official/snapshot \
  --peers 172.22.2.1:19441,172.22.2.2:19441,172.22.2.3:19441,172.22.2.4:19441 \
  --listen 127.0.0.1:18041
python3 scripts/qualify-ds41-native-api.py --output /tmp/native-api-results.json
```

The qualifier issues short real requests, records three counting streams, tests disconnect recovery and rejects unsupported sampling. It is not a broad quality corpus.

## Remaining release work

This initial service executes one request at a time; the queue does not implement concurrent GPU batching. It currently supports greedy text generation, at most 4096 prompt-plus-output tokens, and no vision replacement. Nonzero temperature and images are rejected. Longer context, sampling, concurrency-16 scheduling/alternating waves, dSpark, native vision, comprehensive protocol/error/shutdown checks, broader quality comparison and performance tuning remain required. Root release build/run scripts still need migration to this native service; this development command does not establish release readiness. Spark all-layer startup was around two minutes in the earlier observed run and still needs separate load/setup timing and optimization.
