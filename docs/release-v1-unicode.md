# Native Unicode completion and streaming

Native serving now flushes the tokenizer's pending suffix at EOS or the output
token limit. It uses the same lossy UTF-8 policy as whole-sequence decoding:
incomplete terminal bytes become U+FFFD (`�`), and a literal U+FFFD is preserved.
Completed characters continue to stream incrementally as before.

The upstream streaming helper waits while decoded text ends in U+FFFD because
additional token bytes might complete a character. Native serving previously
finished without draining that state. A live request to copy `�` therefore
returned an empty answer despite generating the character and EOS. The same
request now returns `�`. A parrot emoji limited to one or two of its three
tokens also previously disappeared; it now produces the terminal replacement
character. All three tokens still produce `🦜`.

The scheduler emits the terminal suffix before its finish event and accounts
for buffered tokens once, including EOS. Prefix-cache keys and model token IDs
do not change. The new work runs at completion, not on every generated token.

## Validation

The byte-level fixture compares incremental decoding plus completion against
Rust's whole-input UTF-8 conversion at every byte boundary. It covers ASCII,
Chinese, accents and combining marks, joined emoji, flags, Hebrew, Arabic,
Devanagari, Thai, NUL, tabs/newlines and literal replacement characters. It also
checks that finishing twice emits nothing twice and that a reset decoder can
be reused.

The pinned official tokenizer passes exact encode/decode round trips for
**157 strings** and streaming/whole-decode agreement at **1,045 token cutoffs**.
The corpus includes deterministic samples across the Unicode scalar range and
the UTF-8 length/surrogate boundaries. These deterministic tests distinguish
encoding correctness from a model choosing different text.

Ten native API tests pass. The Unicode fixture feeds character-sized chunks
through both buffered JSON and SSE, preserving non-ASCII text, combining marks,
joined emoji, U+FFFD, interior newlines and NUL in reasoning and answer content.
It also checks the prompt passed to the worker. The pinned protocol intentionally
consumes newlines adjoining `</think>`; that separator behavior is tested
separately from preservation of the reasoning text itself.

Both target and dSpark canaries pass nine live cases in buffered and streaming
form, with equal content and token usage. Every complete echo is exact; only
the deliberately truncated emoji differs from its complete source character.
The old empty-output reproduction, full responses, SSE events and model inputs
are preserved in the evidence.

## The earlier arithmetic result

The historical “Unicode” quality case asks in Traditional Chinese to calculate
18 + 27 − 9 and output only the number. With thinking explicitly disabled, both
modes still output the correct result plus working, failing the exact-answer
instruction. With the default high-thinking setting, both return `36` in answer
content and put the working in reasoning content. This is model/protocol
instruction-following behavior, not evidence of corrupted Chinese encoding.
Both variants remain in the report rather than dropping the failing one.

## Performance and artifacts

The short comparison uses RTX PRO 6000 Blackwell GPUs at **400 W enforced power
limits and standard memory speed**, driver 595.91.07. It alternates four AB/BA
pairs of a 96-token counting request against the vision-qualified baseline,
serially across the shared four Spark workers. Text and token counts agree in
every pair. Median decode changes are **−0.34% target** and **+0.44% dSpark**,
within normal variation. This is a focused regression check, not the full
release performance matrix.

Run `scripts/qualify-ds41-unicode.py --base-url URL --output NEW_PATH` for the
live cases. The loader's ignored official-tokenizer test uses
`DS41RT_UNICODE_MODEL` to locate the pinned snapshot. The ordinary loader/API
fixtures run without model weights or CUDA.

[Evidence metadata](release-v1-unicode.json) records source/artifact hashes,
launch commands, test logs, complete live output and the compressed audit
bundle. Two initial API-fixture failures are retained: it incorrectly supplied
streaming-only options to a nonstreaming request, then incorrectly expected the
reasoning delimiter's newline to remain content. Correcting those fixture
expectations required no protocol/runtime changes.

The separate tool/schema, long-context/pool-pressure, launch-script and full
release qualification gates remain open.
