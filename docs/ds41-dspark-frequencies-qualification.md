# Native dSpark rotary frequencies

The native generator matches the pinned reference's CUDA construction of RoPE64/base10000 frequencies with YaRN disabled: FP32 power and reciprocal, an FP32 outer product with absolute positions, and FP32 cosine/sine. It accepts one U64 position per row, has no persistent workspace, and is allocation-free during capture/replay.

On both RTX PRO 6000 GPUs, all 1,048,576 checkpoint context positions (32 complex pairs each) matched the official CUDA reference bitwise, including 256 changed-position replays of a 4096-row graph. Separate 1/16/80/255/1023/4095-row graphs covered the end of the context range; overlap and invalid-row guards passed. The qualifier extracts the function from the hash-verified reference and constructs its table inside `torch.device('cuda')`, as the official model loader does.

Draft attention now derives the five positions from each generation-checked committed end, uploads them alongside its descriptors, and generates frequencies inside its captured graph before query/KV rotation. Caller-supplied draft frequencies have been removed. Committed main context takes initialized U64 positions and generates its own frequencies in the shared main/KV graph. Its caller must still supply positions matching the accepted-token chunk mappings.

Both composed owners were rebuilt and requalified on both RTX GPUs with freshly generated CUDA-reference expected outputs. Main context passed six capacities through 4096, changed inputs, ring appends/wraparound and stage-identity rejection; attention passed all three stages at 1/3/16 requests, changed inputs and committed positions, and existing lease/owner/ring guards. All compared BF16 outputs in these structured fixtures matched bitwise. This does not establish full-checkpoint numerics or throughput.

Capacity-80 main context adds 640 device bytes for positions; each capacity-80 attention wave adds 640 device bytes and 640 pinned host bytes. The partial two-wave dSpark device estimate becomes 8,853,577,584 bytes, still excluding the shared vocabulary head and driver allocations. Larger prefill producer capacities require their own explicit budget.

Decoder taps still need to be gathered from the mean of four hyperconnection streams at each target layer's attention input, after its engram update, in target-layer order 37/38/39. The complete three-stage attention/mHC/FFN and verification lifecycle and serving integration remain open. No real checkpoint was loaded, and no throughput claim is made.
