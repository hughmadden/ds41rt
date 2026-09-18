# Ordered Spark expert AOT comparison profile

19 September 2026 AEST. This is an unqualified comparison build. The release
recipe default stays unchanged until measured against the ordered build.

## Why this profile exists

The retained native recipe at DS41RT `b042d7314972cedae80b58822f6d562ea7f0c2b4`
and SparkInfer `3882b935ede761d6c73a5d6fd68e690f1e3f5380` exports width 64 at
capacity 1 and width 192 elsewhere. Capacities 256, 1024 and 4096 use direct
FP32 atomic token accumulation (ABI 3). A worker configured at capacity 1024
uses its separate ordered capacity-80 state for rows 2–80, then the atomic
capacity-1024 state for rows 81–1024. This is selected by live row count, not
the request's decode/prefill label.

`V41FusedSliceKernel` calls `red_add_global_f32` for every slice/route
contribution. Different block arrival orders change FP32 addition order; the
final BF16 conversion can expose a one-step difference. The native bridge
probe observed this at rows 128 and 1024 across identical packed inputs.
Integer route planning itself uses stable ballots/prefixes; this finding does
not implicate route ordering or network corruption.

The ordered profile reuses the existing ABI-2 path: each block stores its FP32
slice partial, `V41SliceReduce` adds slices in fixed order into original route
order, and `compact_routes<6>` adds routes with `__fadd_rn` before one BF16
conversion. It keeps the current fused gate/up/SwiGLU/down computation and
width choices. It does not imply numerical equivalence to the Triton engine,
which also uses ordered reduction but has different intermediate quantization.

## Build on a native Spark compiler environment

Use a clean committed DS41RT checkout with both locked source dependencies
populated. Run the retained development image's artifact builder with read-only
source and a separate output directory. The following is the command **inside**
that environment, where `/source` and `/output` are those mounts:

```sh
DS41RT_V41_EXPERT_SLICE_WIDTH='1:64,16:192,80:192,256:192,1024:192,4096:192' \
DS41RT_V41_EXPERT_ATOMIC_MIN_CAPACITY=4097 \
  /source/scripts/build-release-artifacts.sh /source expert 121 /output
```

These are the existing CMake experimental slice controls, now passed through
the release and WIP artifact builders. Threshold 4097 is explicitly above the
largest supported capacity; omitting the threshold with the same width map is
also ordered. Setting the threshold without a width override is rejected.
For an atomic comparison through the same experimental exporter, use the same
width map with threshold 256. Clearing both variables preserves the standard
recipe exporter and its original threshold 256.

Direct CMake callers use the same two names with `-D`. Use separate build and
output directories for ordered and atomic arms, retaining each manifest and
linked-library hash. The top-level `build.sh` also rebuilds/distributes both
roles; the targeted command above deliberately uses the existing artifact
builder directly. Environment variables set on the host must be explicitly
passed into its build container; the top-level wrapper does not forward them.

Before packaging or launching, inspect `/output/V41_EXPERT_AOT.json`: all six
variants must have `native_abi_version: 2`, `output_kind: "fp32_routes"`, and
the requested widths. Record the clean DS41RT SHA, SparkInfer lock SHA,
development image ID and output library SHA alongside the manifest. Package
with the existing `docker/Dockerfile.release` flow under a distinct candidate
tag; keep the retained atomic image available for rollback.

## Memory and validation gates

At capacity 1024, width 192 gives three intermediate slices. Per rank and lane,
the ordered path allocates 360 MiB of FP32 slice planes and 120 MiB of FP32
route output, excluding route metadata, input and compact BF16 output. The
atomic path uses a 20 MiB FP32 token output instead. Two lanes therefore add
about 920 MiB per rank for these buffers. Capacity 4096 multiplies these figures
by four; the worker allocates for its selected capacity, not every exported
capacity. Recheck the actual per-worker budget before launch.

Offline profile tests:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest scripts/tests/test_ordered_expert_profile.py
```

The stub tests check unchanged recipe selection, explicit ordered/atomic
compiler arguments, rejected combinations and real shell-builder forwarding.
They do not compile or execute GPU math.

The smallest live discriminator is identical packed input at rows 80 and 81,
first with one lane and then two, followed by 128 and 1024. For each shape run
at least eight identical replays, compare each rank's BF16 bytes within its
lane and across lanes, and retain mismatch coordinates plus values. Also
compare the full per-rank outputs against loaded-checkpoint reference results;
repeatability alone is not accuracy.

Measure atomic and ordered arms with identical weights, routes, row counts,
lane count, width and warmup. Report per-rank kernel/reduction time, total
layer latency and memory. No performance preservation is claimed: the ordered
path trades atomic contention for extra memory traffic. If that cost is too
large, the next bounded optimization is to fuse ordered slice and route
reduction into compact BF16 output, removing the 120 MiB route-output
intermediate while preserving the current summation order.
