# Native vision serving

The native API accepts up to sixteen images and runs the BF16 vision encoder,
aligner and target embedding replacement in both target-only and dSpark modes.
The practical BF16 attention path remains the default; FP32 accumulation and
softmax are retained. The [encoder component report](release-v1-vision-encoder.md)
preserves the numerical differences from the reference and its failed strict
diagnostic thresholds. The serving checks below establish image-answer behavior,
not bitwise equivalence to that reference.

## Admission and ownership

OpenAI image content accepts image data URLs and HTTP/HTTPS URLs. Preparation
supports base64, unpadded base64 and percent-encoded payloads, using the native
JPEG/PNG/WebP/GIF loader and the pinned model's resize policy. Encoded input is
bounded to 32 MiB per image and 64 MiB per request; the HTTP request body limit is
96 MiB. Existing loader limits bound dimensions and codec allocation. Downloads
allow at most three redirects, use a ten-second connection timeout and a
thirty-second request deadline, and share a sixty-second preparation budget.

Each image request reserves a backend queue slot before CPU preparation. Up to
four blocking decoders run at once; additional admitted requests wait within
the bounded queue. Cancellation releases the queue reservation, but a running
blocking decoder keeps its semaphore permit until it actually finishes. Invalid
images never reach the CUDA worker. The API drops the converted conversation's
copy of image payloads before preparation and generation.

Recipe 0.1.0 renders `<｜image｜>`. The pinned model tokenizer instead assigns
129264 to `<｜deepseek_image｜>`, so image-bearing prompts translate that spelling
at the API handoff. Text-only prompts are unchanged. The first live attempt
caught this mismatch and failed before inference; the corrected spelling is
covered by the API admission test.

The worker expands image spans before context/output accounting and installs
their layouts before prefix restoration. Content identities select radix keys.
Only spans intersecting new input or the bounded encoder replay window are
encoded; replay can begin inside an image and still obtains its complete feature
span. Exact prompt and completed-turn hits can skip the encoder entirely. CPU
decoding still runs to establish normalized image identity.

One shared vision owner reserves **1,587,976,256 device bytes** before KV-pool
sizing. The [embedding lanes](release-v1-vision-features.md) reserve another
80.03 MiB at 4,096-row capacity. Each active request can own at most 160 MiB of
host image features, or 2.5 GiB at C16. Retained prefixes pin image identities,
not those feature arrays. Request release and cancellation drop feature owners.

## Qualification

Tests use two RTX PRO 6000 Blackwell GPUs with **400 W enforced power limits and
standard memory speed**, driver 595.91.07, and the same four Spark workers as
the text baseline. Workloads run serially across modes. Development servers use
32,768-token contexts, 2,048-token prefill chunks, 24 retained turns, target C16
and dSpark C2. These are development settings, not changes to release defaults.

Nine native API tests pass, including image formats and byte limits, HTTP bodies
with and without declared lengths, sixteen-image admission, rejection of
seventeen images, queue saturation/recovery, and cancellation while downloading
or waiting for a decoder. Existing streaming, error propagation, output-limit
and high-thinking default tests continue to pass.

Live checks cover mountain and logo recognition, a solid black image, changed
image identity, reordered pairs, a changed second image that replays through
the first image's span, completed-turn reuse, and sixteen-image exact repeats.
Both modes give the expected semantic answers. Exact repeats within each mode
produce identical text and report complete prompt-cache hits; logs confirm zero
encoded images. Longer descriptions differ between target and dSpark, so these
results do not claim cross-mode token-for-token parity.
Separate image requests that omit thinking settings also return the correct
answer with nonempty reasoning content, exercising the default high effort.

Both modes also process sixteen maximum-length spans, totaling 16,384 image
tokens. A generated 1×4,096 solid-red PNG exercises the 1,024-token-per-image
bound. Both answer `Red`, reuse all 16,419 prompt tokens on an exact repeat,
and successfully handle a fresh text request after cancellation during a second
maximum-span request. The evidence includes concurrent maximum-span admission
at each canary's configured concurrency.

## Text regression check

Four alternating AB/BA pairs per workload compare against the retained-turn
baseline, using 96 generated tokens and warmed prefixes. Every compared text
and token count matches.

| Mode | Counting decode change | Code decode change |
| --- | ---: | ---: |
| Target | −0.31% | +0.34% |
| dSpark | −0.61% | −0.50% |

These short checks show no material decode regression. They are not the full
release prefill/decode matrix or headline throughput claims. The performance
canary precedes the final API-only changes that release the conversation payload
earlier and queue decoder waiters; its CUDA execution and text inputs are the
same. The final binary has separate API and concurrent image-admission checks.

## Reproduction and remaining release work

`scripts/qualify-ds41-native-vision.py` accepts `--base-url`, `--fixtures-dir`
and a fresh `--output` path. Use an isolated cache or a fresh `--session` so the
partial-replay case does not accidentally become an exact hit from a prior run.
External fixture hashes are recorded without redistributing the images. The
maximum-span and concurrency drivers are preserved with the campaign evidence.

[Evidence metadata](release-v1-vision-serving.json) identifies source and binary
hashes, launch settings, complete results, logs and the compressed audit bundle.
The broader long-context/pool-pressure prefix gate, Unicode/tool/schema audit,
standard launch scripts and full release performance/publication work remain
open.
