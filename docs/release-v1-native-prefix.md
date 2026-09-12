# Native prefix cache implementation

The release API uses `serve-native`, entering `v41_native_serve.rs` and its
`scheduler.rs`. The older `commands/real_full` radix/cache is not on that path.
The current native scheduler always admits a fresh cache lease, reports zero
cache-hit tokens and releases its request on completion.

## Compressed source ownership

The native source pool now supports retaining and restoring initialized prefixes
without copying their GPU data. Every active request table and retained prefix
owns references to physical pages. Dropping a retained prefix releases its
references; a page becomes free only when its final owner disappears. Prefix
handles cannot be restored into another pool. The pool tracks committed row
counts and rejects retaining an uninitialized portion of a partially filled page.

KV values/scales and index keys/scales share one page table. When an append
touches a shared partial page, the transaction reserves a private replacement,
copies all four planes on the commit stream, writes accepted rows, then publishes
device metadata and host ownership. Stream draining and the existing request
invalidation path protect failed transactions. Appending after a completed page
uses a new page without copying the completed page.

No GPU data copy is added to appends on exclusive pages. Prefix restoration
uploads the page table and initialized row count. A shared partial tail copies
256 × (64 + 4 + 512 + 16) = 152,576 bytes, independent of total prefix length.
Host reference accounting occurs at page allocation/publication/release; this
is not an end-to-end performance result.

## Qualification and remaining integration

The component qualification uses the selected native library whose SHA256 is
`ccd82d862473f71d01622bcf5decd9d5789444210db1be4cc245dbdcb44b8a07`.
Tests cover exact and divergent retained prefixes, all four stored planes,
copy-on-write isolation, completed page sharing, restoration after original
request release, eviction, pool exhaustion and recovery when an exclusive tail
can be appended without free pages. See the accompanying JSON evidence.

This is a source-pool prerequisite, not enabled API prefix reuse. Still required:
native token radix publication and lookup; compressor pending-row and Engram
history restoration; retained last-turn SWA/dSpark windows; bounded replay for
other hits; eviction coupling and pool sizing; API numerical/agentic/concurrency
qualification; controlled ordinary decode and prefix-resume performance checks.
The existing development containers have not been replaced by this change.

Reproduce component tests from the repository root:

```bash
LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/cu13/lib" \
DS41RT_NATIVE_LIB=/tmp/ds41-draft-release-default/selected-native/libds41rt_native.so \
DS41RT_PYTHON=.ds41rt-cache/reference-venv/bin/python \
scripts/run-with-python-env.sh cargo test --manifest-path rust/Cargo.toml \
  -p ds41rt-daemon --bin ds41rt source_cache -- --include-ignored --test-threads=1
```

The native library path is a development artifact; substitute a matching built
library when reproducing after a reboot. Run the resulting test executable under
`compute-sanitizer --tool memcheck --error-exitcode 99` with the same environment
and test arguments to qualify device bounds.
