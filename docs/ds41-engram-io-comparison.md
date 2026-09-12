# Engram CPU delivery and prefetch overlap

The September 12, 2026 experiments favor retaining warm mmap access and moving
completed row delivery early. io_uring is promising for rows that need storage
reads; merely warming the file cache leaves substantial first-mapping work for
the consumer. No production backend has been changed by this experiment.

Both official Engram layers were tested on raptor's NVMe checkpoint, with five
shuffled repetitions of 1, 6, 18, and 48 tokens, each requesting 24 uniformly
sampled addresses. Addresses were sorted and deduplicated. Every completed result
was compared byte-for-byte with buffered reference reads. These synthetic hashes
are not a trace of actual prompt locality or cross-request sharing.

At 18 tokens (432 distinct addresses, 114,048 packed bytes), layer 14 median
completed CPU delivery was:

| Method | Warm mapping/data | Cached data, fresh mapping | File-range eviction |
| --- | ---: | ---: | ---: |
| mmap | 5.5 us | 1,113 us | 34.15 ms |
| mmap + WILLNEED | 188 us | 1,064 us | 2.17 ms |
| mmap + POPULATE_READ | 154 us | 904 us | 35.32 ms |
| buffered pread | 194 us | 193 us | 32.99 ms |
| io_uring, depth 256 | 219 us | 218 us | 1.54 ms |
| io_uring, registered buffer | 219 us | 225 us | 1.72 ms |
| io_uring then mmap copy | 206 us | 1,180 us | 2.68 ms |
| Linux native AIO | 230 us | 234 us | 33.91 ms |
| POSIX aio_read | 1,288 us | 1,402 us | 34.54 ms |

Layer 1 gave similar results. POSIX AIO was measured separately, with its library
worker primed before timing. io_uring workers were also primed. The native AIO
case uses buffered io_submit/io_getevents; it is not an O_DIRECT benchmark.
Registered buffers did not consistently improve these small row reads.

Queue depth matters: the depth-64 io_uring trial lost to WILLNEED on evicted rows;
depth 256 improved the 18-token median to 1.68 ms versus 2.28 ms for WILLNEED in
that sweep. Depth 1024 was not uniformly better across token counts or buffer
registration choices. The implementation submits bounded batches and waits for
each batch, so these are measurements of that implementation, not API ceilings.

## Lead time

A separate seven-repeat experiment starts a prepared CPU worker, then delays the
consumer by 0, 100, 500, 1,000 or 3,000 us. Reports retain actual elapsed lead time
and total completion time. It does not simulate competing GPU/CPU work or the
runtime's queue contention. Worker creation, mapping, I/O-ring setup and reference
reads are outside the timer; dispatch/wakeup and completion are inside it.

For the same 18-token layer-14 batch, residual consumer wait after **3 ms** lead:

| Work completed in background | Cached data, fresh mapping | File-range eviction |
| --- | ---: | ---: |
| WILLNEED advice only; consumer gathers mmap | 1.30 ms | 1.39 ms |
| io_uring reads only; consumer gathers mmap | 1.19 ms | 1.46 ms |
| WILLNEED plus completed mmap gather | 1.5 us | 1.4 us |
| io_uring directly into completed staging | 1.5 us | 1.5 us |
| io_uring plus completed mmap gather | 1.4 us | 1.4 us |

At **1 ms** lead on evicted data, background direct io_uring still waited about
0.99 ms, versus 1.67 ms for WILLNEED plus gather and 1.97 ms for io_uring plus mmap
gather. Warm mmap background gather completed within the 100-us lead in these
trials. Paying asynchronous read overhead for every already-mapped row would
sacrifice the cheapest path.

The current runtime already has a background gather worker. It also has separate
WILLNEED advice, one FIFO gather worker, blocking GPU uploads and a 1-ms consumer
polling sleep. Actual queueing, early submission, first-layer lead time, completion
notification and upload staging need runtime measurements before choosing a
replacement. This experiment does not prove advisory io_uring alone is best.

## Cache verification and scope

`fresh` primes file data through pread, then creates a new mmap; `warm` additionally
copies the sampled mmap rows before timing. All I/O modes use buffered reads;
file access is advised random. `cold` requests page-aligned file-range eviction
and uses mincore to record actual residency. Across the two-layer final sweep,
11–28% of sampled pages remained resident. **These are partially evicted trials,
not fully cold storage.** Warm/fresh trials had all sampled pages resident.

The installed fixed-function cache-drop helper was exercised in a separate
15-trial global-drop run. It succeeded but some sampled pages remained resident.
Global-drop timings differed materially from file-range eviction: at six tokens,
WILLNEED and io_uring were about 10 ms, versus about 21 ms for serial mmap/pread.
Do not mix these results with the file-range sweep or assume that a successful
cache-drop write proves zero residency.

The native benchmark maps the shard rather than separate tensor regions, copies
unique rows into an interleaved scratch layout, and issues per-page advice rather
than the runtime's coalesced runs. It excludes hash generation, sorting, duplicate
scatter, staging allocation, GPU uploads and projections. Memory faults and input
blocks are process-wide over the measured interval. Reference reads occur before
eviction, and mincore itself does not touch the mapped payload.

## Reproduction

Build with a C++17 compiler and liburing development headers/library:

```bash
mkdir -p .ds41rt-cache/engram-io
c++ -std=c++17 -O2 -Wall -Wextra -Werror \
  scripts/native/bench-engram-io.cpp -luring -pthread -lrt \
  -o .ds41rt-cache/engram-io/bench
python3 scripts/bench-ds41-engram-io.py \
  --binary .ds41rt-cache/engram-io/bench --snapshot /path/to/official/snapshot \
  --depth 256 --output .ds41rt-cache/engram-io/results.json
```

For background comparison, add `--tokens 18 --delivery gather --modes mmap
willneed uring uring-mmap --lead-us 0 100 500 1000 3000 --repeats 7`.
For advisory-only delivery use `--delivery advisory --modes willneed uring-mmap`.
`--cache cold-global` explicitly invokes the installed system-wide cache helper;
normal runs only request eviction of sampled file ranges. Use the actual storage
checkpoint, not a tmpfs fixture, for storage conclusions.

[Machine-readable summaries](ds41-engram-io-comparison.json) retain sample counts,
medians, ranges, residency, fault/I/O counts, raw artifact hashes and available
binary hashes. The container needs the narrowly extended
[seccomp profile](../docker/seccomp-io-uring.md) to test io_uring; this was verified
in a disposable container without changing the live APIs.
