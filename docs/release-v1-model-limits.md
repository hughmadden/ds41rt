# Native model and output limits

`serve-native` defaults to 1,048,576 total context tokens and 393,216 generated
tokens. The pinned model config specifies the context maximum, and the
[official API reference](https://api-docs.deepseek.com/api/create-chat-completion/)
identifies 384K as exactly 393,216 output tokens. The
[official model details](https://api-docs.deepseek.com/quick_start/pricing/)
identify V4.1 Flash and its 1M context. Sources were checked September 12, 2026.

An omitted or null `max_tokens` uses the configured output maximum. A positive
request above that maximum clamps to the maximum. After tokenizing the rendered
conversation, admission further clamps output to the remaining context:

```
allowed_output = min(requested_or_default_output, configured_output_max,
                     configured_context_max - actual_prompt_tokens)
```

Thinking and answer tokens share this allowance. Reaching it finishes with
`length`; an earlier end token finishes with `stop`. Empty prompts and prompts
that leave no output space fail admission. Zero `max_tokens` fails HTTP validation.
The prompt count includes rendered control, tool and reasoning instructions.

`--max-context-tokens` and `--max-output-tokens` select positive smaller launch
limits; values above the model maxima are rejected. For development comparisons,
pass `--max-context-tokens 32768` explicitly to reserve smaller source storage.
The maximum output option may exceed the configured context; each request still
clamps to its remaining context. `/v1/models` reports both configured limits.

The engine's omitted output default deliberately follows the user's release
requirement to allow the full model maximum. The hosted API uses smaller
mode-dependent defaults; those are not imposed here.

## Focused verification

API tests cover the model default, lower launch caps, positive oversized requests,
zero rejection, model metadata and forwarding of output allowances. The shared
admission calculation is tested at the final context token, with invalid prompts
and integer-overflow inputs. CLI tests verify defaults and reject out-of-range
options. Existing native thinking, streaming, ordinary-response and backend-error
tests also pass.

Live target-only and dSpark tests use a 256-token development context. Target-only
keeps the default output maximum; dSpark uses an explicit 128-token output maximum.
The target-only counting request consumes 25 prompt tokens and stops after 231
output tokens, exactly at the context boundary. dSpark stops at its lower output
cap. Both accept omitted limits and positive requests of 8,192 and 4,294,967,295,
respect an explicit eight-token limit, and report `length` in ordinary and
streaming responses. Their output matches the preceding artifacts when those
artifacts receive the equivalent explicit allowance.

[Evidence metadata](release-v1-model-limits.json) links the binary, source hashes,
launch commands, checks and complete requests/responses. These are focused limit
and output-parity checks; 1M inference, long output qualification, final pool
sizing and the complete release performance campaign remain open.
