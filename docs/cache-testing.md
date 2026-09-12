# Page-cache control for benchmarks

Install the fixed-function helper once, entering your sudo password when requested:

```bash
./scripts/drop-page-cache.sh --install
```

Installation does not flush caches. Afterward, the same user can run:

```bash
./scripts/drop-page-cache.sh --check
./scripts/drop-page-cache.sh
```

The second command synchronizes writes and requests `drop_caches=1`, dropping
eligible clean filesystem page-cache pages system-wide. It affects other
workloads on the host. Active pages can remain; measure residency with `mincore`
and record physical reads and major/minor faults rather than assuming a fully
cold cache. It does not request inode/dentry-cache eviction or clear GPU memory.
The Linux [drop_caches documentation](https://docs.kernel.org/admin-guide/sysctl/vm.html#drop-caches)
explains the kernel control.

Without an installed helper, invoking the command builds and installs it through
sudo before flushing. `--install` also updates an existing helper. A C compiler
and sudo are required for installation, not subsequent runs.

The shell script itself is never setuid. Its small C helper is installed at
`/usr/local/libexec/ds41rt-bench/drop-page-cache`, owned by root, mode `4750`, with
the installing user's primary group. Only root and that group can execute it.
All parent directories must be root-owned and not group/world-writable. The
installer replaces the helper atomically; no privileged binary is placed in the
writable repository. Each install/update requires ordinary sudo authorization;
no passwordless sudo rule is installed.

The helper accepts no paths, commands, environment configuration or alternate
sysctl values. Its only privileged operation is the fixed sync/cache-drop path.
To remove this access:

```bash
sudo rm /usr/local/libexec/ds41rt-bench/drop-page-cache
```

For I/O comparisons, distinguish:

- Cached file data with already-faulted mappings.
- Cached file data with fresh process mappings.
- Verified nonresident file data, requiring storage reads.

Use file-range eviction when possible to avoid disturbing unrelated workloads.
Keep storage-test fixtures on the intended disk: `/tmp` is tmpfs on the current
development coordinator, so it is unsuitable for NVMe throughput tests.

Unprivileged safety checks (never install or flush):

```bash
python3 -m unittest discover -s scripts/tests -p test_drop_page_cache.py
```
