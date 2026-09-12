# Image identity in the native prefix radix

The prefix cache can now distinguish image content while preserving its compact
u32 token radix. Each prepared image's complete SHA-256 identity includes its
normalized BF16 patches and two-dimensional grid. A per-server registry maps
that identity to a symbol outside the model vocabulary. Cache comparisons replace
only the image span's keys; model execution and Engram history continue to receive
the original token IDs. Key count and token positions remain unchanged, including
partial matches inside an image and compression-boundary replay calculations.

Active requests and retained prompt/turn snapshots hold reference-counted image
identity pins. The registry retains weak references and recycles a symbol only
after all its owners have expired. Radix eviction removes orphaned edges before
another admission can recycle their symbols. Collection runs on image admission;
text requests borrow the original token slice and allocate no additional token
buffer. The registry's live identities are bounded by active requests and the
configured retention banks, with at most sixteen image bindings per request.

The scheduler and cache retention/restore interfaces carry those pins. This
checkpoint establishes cache identity and lifetime behavior; the separate
[serving qualification](release-v1-vision-serving.md) covers their live use with
image feature replacement, routing/Engram masks and bounded replay.

## Validation

Ten focused tests pass: six image-key tests and the four existing radix/retention
tests. They cover actual loader-produced image identities, identical normalized
images from different source sizes, changed and reordered images, all 32 hash
bytes, partial-image matches, generated text suffixes, both retention banks,
active-owner pins, symbol reuse below a shared text edge, sixteen-image bounds,
invalid spans, key-space exhaustion and recovery, and 1,000 successive expired
admissions without registry growth. The million-token text case confirms that
key lookup borrows the original buffer. Existing 24-turn retention, partial-replay
selection and LRU ownership tests continue to pass.

```bash
cargo test --manifest-path rust/Cargo.toml -p ds41rt-daemon --bin ds41rt \
  v41_native_serve::prefix
```

[Evidence metadata](release-v1-vision-cache-identity.json) records the source and
test artifact hashes and complete build/test logs. These checks are CPU-side;
live image-cache behavior and focused text-performance checks are recorded in
the serving report.
