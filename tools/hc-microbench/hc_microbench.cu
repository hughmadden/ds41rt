// hc_microbench.cu — PCIe copy-shape microbenchmark for the DS41RT host snapshot cache.
//
// Question answered: the host-cache moves a snapshot as thousands of ~22 KB
// device<->host memcpys (four segments per 91 KB page). Idle restore runs at
// ~0.8 GB/s on a PCIe 5 x16 link; under prefill load stores complete at
// 20-80 MB/s. Is the cost per-submission (fixable by batching) or the
// link/host memory (not)?
//
// For each variant below, both directions (D2H then H2D) are timed with CUDA
// events on a single non-blocking stream, best-of-`--repeat` and mean, and the
// copied bytes are verified after every repeat (host-side memcmp against a
// host-generated golden image for D2H; a trusted linear read-back compared
// against the source buffer for H2D) so a variant that copies the wrong bytes
// is reported as FAILED, not fast.
//
// Variants:
//   seg    one cudaMemcpyAsync per --seg bytes (the status quo)
//   page   one cudaMemcpyAsync per --page bytes
//   2d     cudaMemcpy2DAsync, rows of --seg bytes, pitch --page both sides
//          (models a per-arena run of consecutive pages; copies
//          floor(bytes/page) * seg bytes, not the whole buffer)
//   batch  cudaMemcpyBatchAsync (CUDA >= 12.8) with the same segment list as
//          variant seg; guarded by CUDART_VERSION and a runtime probe
//   bulk   one memcpy of the whole buffer (upper bound)
//   scatter  variant seg with the segment order randomly permuted on the
//          device side (models free-list fragmentation)
//
// Usage: hc_microbench [--bytes MB] [--seg B] [--page B] [--repeat N]
//                      [--min-free-mb N]
//                      [--variants seg,page,2d,batch,bulk,scatter]
//
// No dependencies beyond the CUDA runtime and the C/C++ standard library.
// This program only queries and uses its own allocations; it never touches
// other processes, containers, or driver state.

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>


// cudaMemcpyBatchAsync across toolkits: 12.8 takes a trailing size_t *failIdx, 13.x does not
// (cuda_runtime_api.h 13.2 line 6540). Both require a valid attribute entry per copy
// (attrs == NULL is cudaErrorInvalidValue); one entry with stream-ordered source access covers all.
static cudaError_t batch_copy(void **dsts, void **srcs, std::size_t *sizes,
                              std::size_t count, cudaStream_t stream) {
#if CUDART_VERSION >= 12080
  cudaMemcpyAttributes attr{};
  attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  std::size_t attr_idx = 0;
#if CUDART_VERSION >= 13000
  return cudaMemcpyBatchAsync(dsts, srcs, sizes, count, &attr, &attr_idx, 1, stream);
#else
  std::size_t fail_idx = 0;
  return cudaMemcpyBatchAsync(dsts, srcs, sizes, count, &attr, &attr_idx, 1, &fail_idx, stream);
#endif
#else
  (void)dsts; (void)srcs; (void)sizes; (void)count; (void)stream;
  return cudaErrorNotSupported;
#endif
}

namespace {

// ---------------------------------------------------------------------------
// Fatal error handling
// ---------------------------------------------------------------------------

[[noreturn]] void die(const char *what, cudaError_t err) {
  std::fprintf(stderr, "error: %s: %s\n", what, cudaGetErrorString(err));
  std::exit(1);
}

void cuda_check(cudaError_t err, const char *what) {
  if (err != cudaSuccess) die(what, err);
}

[[noreturn]] void die_msg(const std::string &msg) {
  std::fprintf(stderr, "error: %s\n", msg.c_str());
  std::exit(2);
}

// ---------------------------------------------------------------------------
// Ground-truth byte pattern: a pure function of word index, identical on host
// and device, so the host-generated golden image is independent of every code
// path under test.
// ---------------------------------------------------------------------------

__host__ __device__ inline std::uint32_t pattern_word(std::size_t i,
                                                      std::uint32_t seed) {
  std::uint32_t x = static_cast<std::uint32_t>(i) * 2654435761u;
  x ^= x >> 15;
  x *= 0x85ebca6bu;
  x ^= seed * 0xc2b2ae35u;
  x ^= x >> 13;
  return x;
}

__global__ void fill_pattern(std::uint32_t *buf, std::size_t n_words,
                             std::uint32_t seed) {
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_words;
       i += stride) {
    buf[i] = pattern_word(i, seed);
  }
}

void fill_host_pattern(std::uint8_t *buf, std::size_t bytes,
                       std::uint32_t seed) {
  const std::size_t n = bytes / sizeof(std::uint32_t);
  auto *words = reinterpret_cast<std::uint32_t *>(buf);
  for (std::size_t i = 0; i < n; ++i) words[i] = pattern_word(i, seed);
}

// Pattern seeds: seed 0 is the device/golden image every D2H repeat restores
// and is verified against; seed 1 is the independent pattern every H2D repeat
// re-patterns the pinned source buffer with.
constexpr std::uint32_t kDevSeed = 0, kHostSeed = 1;

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

struct Args {
  std::size_t bytes = 256ull << 20;  // --bytes is given in MB
  std::size_t seg = 22784;
  std::size_t page = 91136;
  int repeat = 3;
  std::size_t min_free_mb = 1024;
  std::vector<std::string> variants;  // --variants names; empty = all
};

std::vector<std::string> split_commas(const std::string &s) {
  std::vector<std::string> out;
  std::size_t begin = 0;
  while (true) {
    const std::size_t comma = s.find(',', begin);
    out.push_back(s.substr(
        begin, comma == std::string::npos ? std::string::npos : comma - begin));
    if (comma == std::string::npos) break;
    begin = comma + 1;
  }
  return out;
}

std::size_t parse_u64(const char *s, const char *what) {
  char *end = nullptr;
  const unsigned long long v = std::strtoull(s, &end, 10);
  if (end == s || *end != '\0') die_msg(std::string("bad value for ") + what);
  return static_cast<std::size_t>(v);
}

Args parse_args(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto need_value = [&](const char *flag) -> const char * {
      if (++i >= argc) die_msg(std::string("missing value for ") + flag);
      return argv[i];
    };
    if (k == "--bytes") {
      a.bytes = parse_u64(need_value("--bytes"), "--bytes") << 20;
    } else if (k == "--seg") {
      a.seg = parse_u64(need_value("--seg"), "--seg");
    } else if (k == "--page") {
      a.page = parse_u64(need_value("--page"), "--page");
    } else if (k == "--repeat") {
      a.repeat = static_cast<int>(parse_u64(need_value("--repeat"), "--repeat"));
    } else if (k == "--min-free-mb") {
      a.min_free_mb = parse_u64(need_value("--min-free-mb"), "--min-free-mb");
    } else if (k == "--variants") {
      a.variants = split_commas(need_value("--variants"));
    } else {
      die_msg("unknown argument: " + k);
    }
  }
  if (a.bytes == 0) die_msg("--bytes must be > 0");
  if (a.seg == 0 || a.page == 0) die_msg("--seg and --page must be > 0");
  if (a.seg > a.page)
    die_msg("--seg must be <= --page (2D variant uses it as width <= pitch)");
  if (a.repeat < 1) die_msg("--repeat must be >= 1");
  return a;
}

// ---------------------------------------------------------------------------
// Copy plans: every variant is described as a list of triples
// (host_off, dev_off, len) with the invariant "after the copy,
// device bytes [dev_off, dev_off+len) hold what host bytes
// [host_off, host_off+len) held before it". Verification is uniform in this
// invariant; only submission strategy differs per variant.
// ---------------------------------------------------------------------------

struct CopyTriple {
  std::size_t host_off;
  std::size_t dev_off;
  std::size_t len;
};

enum class Variant { Seg, Page, TwoD, Batch, Bulk, Scatter };
enum class Direction { D2H, H2D };

const char *variant_name(Variant v) {
  switch (v) {
    case Variant::Seg: return "seg";
    case Variant::Page: return "page";
    case Variant::TwoD: return "2d";
    case Variant::Batch: return "batch";
    case Variant::Bulk: return "bulk";
    case Variant::Scatter: return "scatter";
  }
  return "?";
}

// Every variant, in run order; --variants selects a subsequence by name.
const Variant kAllVariants[] = {Variant::Seg,   Variant::Page, Variant::TwoD,
                                Variant::Batch, Variant::Bulk, Variant::Scatter};

// Resolves the --variants comma list to variants; an empty list selects all.
std::vector<Variant> select_variants(const std::vector<std::string> &names) {
  if (names.empty())
    return std::vector<Variant>(kAllVariants,
                                kAllVariants +
                                    sizeof(kAllVariants) / sizeof(kAllVariants[0]));
  std::vector<Variant> out;
  out.reserve(names.size());
  for (const std::string &name : names) {
    bool found = false;
    for (const Variant v : kAllVariants) {
      if (name == variant_name(v)) {
        out.push_back(v);
        found = true;
        break;
      }
    }
    if (!found) die_msg("unknown variant in --variants: '" + name + "'");
  }
  return out;
}

const char *direction_name(Direction d) {
  return d == Direction::D2H ? "D2H" : "H2D";
}

// Debug-only self-check (live in run.sh's default build; compiled out under
// NDEBUG): every triple lies inside `total` on both sides and the triples tile [0, total)
// exactly — no gaps, no overlaps — on each side. A plan-construction bug
// (e.g. a permutation pairing a full slot with the short tail slot's offset)
// fails here at the source instead of as cudaErrorInvalidArgument at
// submission time.
void assert_plan_invariants(const std::vector<CopyTriple> &plan,
                            std::size_t total) {
  for (const CopyTriple &t : plan) {
    assert(t.host_off <= total && t.len <= total - t.host_off &&
           "plan: host range out of bounds");
    assert(t.dev_off <= total && t.len <= total - t.dev_off &&
           "plan: device range out of bounds");
  }
  [[maybe_unused]] const auto tiles_exactly = [&plan,
                                               total](std::size_t CopyTriple::*member) {
    std::vector<std::pair<std::size_t, std::size_t>> ranges;
    ranges.reserve(plan.size());
    for (const CopyTriple &t : plan) ranges.emplace_back(t.*member, t.len);
    std::sort(ranges.begin(), ranges.end());
    std::size_t off = 0;
    for (const auto &[begin, len] : ranges) {
      if (begin != off) return false;
      off += len;
    }
    return off == total;
  };
  assert(tiles_exactly(&CopyTriple::host_off) &&
         "plan: host ranges must tile [0, total) exactly");
  assert(tiles_exactly(&CopyTriple::dev_off) &&
         "plan: device ranges must tile [0, total) exactly");
}

// Segments covering [0, total) with the given stride; the last one carries
// the remainder. With `perm != nullptr`, host slot i pairs with device slot
// perm[i]. The permutation must permute whole slot identities — offset AND
// length together — so every slot keeps its own length on both sides: pairing
// a full slot with the short tail slot's offset would leave
// (slots-1)*stride + stride > total, i.e. an out-of-bounds triple (make_perm
// keeps the tail slot fixed for exactly this reason).
std::vector<CopyTriple> segmented_plan(std::size_t total, std::size_t stride,
                                       const std::vector<std::size_t> *perm) {
  std::vector<CopyTriple> plan;
  for (std::size_t off = 0, i = 0; off < total; off += stride, ++i) {
    const std::size_t len = (total - off < stride) ? total - off : stride;
    const std::size_t dev_off =
        perm ? (*perm)[i] * stride : off;
    plan.push_back({off, dev_off, len});
  }
  assert_plan_invariants(plan, total);
  return plan;
}

std::vector<CopyTriple> make_plan(Variant v, const Args &a,
                                  const std::vector<std::size_t> *perm) {
  switch (v) {
    case Variant::Seg:
    case Variant::Batch:
      return segmented_plan(a.bytes, a.seg, nullptr);
    case Variant::Page:
      return segmented_plan(a.bytes, a.page, nullptr);
    case Variant::Scatter:
      return segmented_plan(a.bytes, a.seg, perm);
    case Variant::Bulk:
      return {{0, 0, a.bytes}};
    case Variant::TwoD: {
      // One segment-slot per consecutive page, both sides pitched by `page`.
      std::vector<CopyTriple> plan;
      for (std::size_t off = 0; off + a.seg <= a.bytes; off += a.page)
        plan.push_back({off, off, a.seg});
      return plan;
    }
  }
  return {};
}

std::size_t plan_bytes(const std::vector<CopyTriple> &plan) {
  std::size_t n = 0;
  for (const auto &t : plan) n += t.len;
  return n;
}

// ---------------------------------------------------------------------------
// Submission
// ---------------------------------------------------------------------------

struct Buffers {
  std::uint8_t *dev;
  std::uint8_t *pinned;  // transfer buffer
  // Scratch arrays for cudaMemcpyBatchAsync, sized to the segment count.
  std::vector<void *> dsts;
  std::vector<void *> srcs;
  std::vector<std::size_t> sizes;
};

// Issues the variant's copies on `stream`. Returns the number of CUDA API
// submissions (what the per-submission-cost question is about).
std::size_t submit(Variant v, Direction dir, const std::vector<CopyTriple> &plan,
                   const Args &a, Buffers &b, cudaStream_t stream) {
  const cudaMemcpyKind kind =
      dir == Direction::D2H ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
  auto endpoints = [&](const CopyTriple &t) -> std::pair<void *, const void *> {
    std::uint8_t *host = b.pinned + t.host_off;
    std::uint8_t *dev = b.dev + t.dev_off;
    return dir == Direction::D2H
               ? std::make_pair(static_cast<void *>(host),
                                static_cast<const void *>(dev))
               : std::make_pair(static_cast<void *>(dev),
                                static_cast<const void *>(host));
  };

  switch (v) {
    case Variant::Seg:
    case Variant::Page:
    case Variant::Scatter: {
      for (const auto &t : plan) {
        auto [dst, src] = endpoints(t);
        cuda_check(cudaMemcpyAsync(dst, src, t.len, kind, stream),
                   "cudaMemcpyAsync");
      }
      return plan.size();
    }
    case Variant::Bulk: {
      auto [dst, src] = endpoints(plan.front());
      cuda_check(cudaMemcpyAsync(dst, src, plan.front().len, kind, stream),
                 "cudaMemcpyAsync");
      return 1;
    }
    case Variant::TwoD: {
      // plan rows are (k*page, k*page, seg) — a single pitched call.
      auto [dst, src] = endpoints(plan.front());
      cuda_check(cudaMemcpy2DAsync(dst, a.page, src, a.page, a.seg,
                                   plan.size(), kind, stream),
                 "cudaMemcpy2DAsync");
      return 1;
    }
    case Variant::Batch: {
#if CUDART_VERSION >= 12080
      for (std::size_t i = 0; i < plan.size(); ++i) {
        auto [dst, src] = endpoints(plan[i]);
        b.dsts[i] = dst;
        b.srcs[i] = const_cast<void *>(src);
        b.sizes[i] = plan[i].len;
      }
      cuda_check(batch_copy(b.dsts.data(), b.srcs.data(), b.sizes.data(),
                            plan.size(), stream),
                 "cudaMemcpyBatchAsync");
      return 1;
#else
      return 0;  // unreachable: caller probes availability first
#endif
    }
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Verification, uniform in the copy-plan invariant.
//   D2H: pinned[host_off..] must equal golden[dev_off..] (golden is the
//        host-generated image of the device buffer).
//   H2D: device_image[dev_off..] (trusted linear read-back of the device
//        buffer after the timed copies) must equal pinned[host_off..].
// ---------------------------------------------------------------------------

bool verify_plan(const std::vector<CopyTriple> &plan, Direction dir,
                 const std::uint8_t *pinned, const std::uint8_t *golden,
                 const std::uint8_t *device_image) {
  for (const auto &t : plan) {
    const std::uint8_t *expect =
        dir == Direction::D2H ? golden + t.dev_off : pinned + t.host_off;
    const std::uint8_t *actual =
        dir == Direction::D2H ? pinned + t.host_off : device_image + t.dev_off;
    if (std::memcmp(actual, expect, t.len) != 0) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Permutation (fixed seed: reproducible across runs and hosts)
// ---------------------------------------------------------------------------

// Fixed-seed permutation of the `n_permutable` full-length slots; all slots
// at index >= n_permutable (the short tail slot) keep their identity. A slot
// is an (offset, length) identity: permuting raw offsets while keeping the
// host slot's length would pair a full slot with the tail offset and overrun
// the buffer (the scatter abort this guards against).
std::vector<std::size_t> make_perm(std::size_t n, std::size_t n_permutable) {
  std::vector<std::size_t> perm(n);
  for (std::size_t i = 0; i < n; ++i) perm[i] = i;
  std::mt19937_64 rng(0x5EEDu);
  for (std::size_t i = n_permutable; i > 1; --i)
    std::swap(perm[i - 1], perm[rng() % i]);
  return perm;
}

struct Row {
  bool available = true;
  bool ok = true;
  std::size_t submissions = 0;
  std::size_t copied_bytes = 0;
  double best_ms = 0.0;
  double mean_ms = 0.0;
};

// One variant x direction row: `repeat` timed, verified iterations. Every
// repeat is independent of every other: a D2H repeat restores the device
// buffer to the golden image (untimed) before timing and poisons the receive
// buffer; an H2D repeat re-patterns the pinned source with the known seed-1
// image (untimed) before timing, so verification never depends on what a
// previous variant or repeat left in either buffer.
Row run_row(Variant v, Direction dir, const std::vector<CopyTriple> &plan,
            const Args &a, Buffers &b, std::uint8_t *device_image,
            const std::uint8_t *golden, cudaStream_t stream,
            cudaEvent_t start_ev, cudaEvent_t end_ev) {
  Row row;
  row.copied_bytes = plan_bytes(plan);
  double sum_ms = 0.0;
  row.best_ms = 1e30;
  for (int r = 0; r < a.repeat; ++r) {
    if (dir == Direction::D2H) {
      // Untimed restore: whatever earlier variants left in the device buffer
      // (e.g. the 2d H2D run's zeros outside its plan) is overwritten with the
      // golden image this repeat's verification compares against.
      cuda_check(cudaMemcpyAsync(b.dev, golden, a.bytes,
                                 cudaMemcpyHostToDevice, stream),
                 "restore cudaMemcpyAsync");
      cuda_check(cudaStreamSynchronize(stream), "restore cudaStreamSynchronize");
      std::memset(b.pinned, 0xA5, a.bytes);  // poison the receive buffer
    } else {
      // Untimed re-pattern: the pinned source must hold the known seed-1
      // image, not whatever the previous D2H repeat delivered into it.
      fill_host_pattern(b.pinned, a.bytes, kHostSeed);
      // Bytes outside the plan stay zero; verification checks plan triples
      // only.
      cuda_check(cudaMemsetAsync(b.dev, 0, a.bytes, stream),
                 "cudaMemsetAsync");
    }
    cuda_check(cudaEventRecord(start_ev, stream), "cudaEventRecord");
    row.submissions = submit(v, dir, plan, a, b, stream);
    cuda_check(cudaEventRecord(end_ev, stream), "cudaEventRecord");
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    float ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&ms, start_ev, end_ev),
               "cudaEventElapsedTime");
    if (dir == Direction::H2D) {
      // Untimed trusted read-back of the whole device buffer.
      cuda_check(cudaMemcpy(device_image, b.dev, a.bytes, cudaMemcpyDeviceToHost),
                 "verify cudaMemcpy");
    }
    if (!verify_plan(plan, dir, b.pinned, golden, device_image)) row.ok = false;
    sum_ms += ms;
    if (ms < row.best_ms) row.best_ms = ms;
  }
  row.mean_ms = sum_ms / a.repeat;
  return row;
}

void print_row(Variant v, Direction dir, const Row &row) {
  const double gbps =
      row.best_ms > 0.0 ? row.copied_bytes / (row.best_ms * 1e6) : 0.0;
  const double subs_per_s =
      row.best_ms > 0.0 ? row.submissions / (row.best_ms * 1e-3) : 0.0;
  if (!row.available) {
    std::printf("| %s | %s | - | - | - | - | - | unavailable |\n",
                variant_name(v), direction_name(dir));
    return;
  }
  std::printf("| %s | %s | %zu | %zu | %.3f | %.3f | %.1f | %s |\n",
              variant_name(v), direction_name(dir), row.copied_bytes,
              row.submissions, row.mean_ms, row.best_ms, gbps,
              row.ok ? "ok" : "FAILED");
  std::printf("<!-- %s-%s submissions/s at best: %.0f -->\n", variant_name(v),
              direction_name(dir), subs_per_s);
}

}  // namespace

int main(int argc, char **argv) {
  const Args args = parse_args(argc, argv);

  // Environment report (PCIe gen/width is added by run.sh via nvidia-smi).
  int rt_version = 0, drv_version = 0;
  cuda_check(cudaRuntimeGetVersion(&rt_version), "cudaRuntimeGetVersion");
  cuda_check(cudaDriverGetVersion(&drv_version), "cudaDriverGetVersion");
  cudaDeviceProp prop{};
  cuda_check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
  std::printf("# hc_microbench\n\n");
  std::printf("- device: `%s` (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  std::printf("- CUDA runtime %d.%d, driver %d.%d\n", rt_version / 1000,
              (rt_version % 1000) / 10, drv_version / 1000,
              (drv_version % 1000) / 10);
  std::printf("- bytes=%zu MB, seg=%zu B, page=%zu B, repeat=%d\n\n",
              args.bytes >> 20, args.seg, args.page, args.repeat);

  // Memory guard: the production coordinator holds most of the card.
  std::size_t free_bytes = 0, total_bytes = 0;
  cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  std::printf("- free device memory: %.1f / %.1f GB\n\n",
              free_bytes / 1e9, total_bytes / 1e9);
  if (free_bytes < args.min_free_mb << 20) {
    std::fprintf(stderr,
                 "error: free device memory %.1f MB is below --min-free-mb "
                 "%zu MB; refusing to allocate (a production coordinator may "
                 "be holding the card)\n",
                 free_bytes / 1e6, args.min_free_mb);
    return 2;
  }

  // Allocations.
  Buffers bufs{};
  cuda_check(cudaMalloc(reinterpret_cast<void **>(&bufs.dev), args.bytes),
             "cudaMalloc");
  cuda_check(cudaHostAlloc(reinterpret_cast<void **>(&bufs.pinned), args.bytes,
                           cudaHostAllocPortable),
             "cudaHostAlloc");
  std::uint8_t *golden =
      static_cast<std::uint8_t *>(std::malloc(args.bytes));
  std::uint8_t *device_image =
      static_cast<std::uint8_t *>(std::malloc(args.bytes));
  if (!golden || !device_image) die_msg("host malloc failed");
  const std::size_t max_slots = (args.bytes + args.seg - 1) / args.seg;
  bufs.dsts.resize(max_slots);
  bufs.srcs.resize(max_slots);
  bufs.sizes.resize(max_slots);

  cudaStream_t stream{};
  cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
             "cudaStreamCreate");
  cudaEvent_t start_ev{}, end_ev{};
  cuda_check(cudaEventCreate(&start_ev), "cudaEventCreate");
  cuda_check(cudaEventCreate(&end_ev), "cudaEventCreate");

  // Ground truth: device buffer and host golden hold the same pattern
  // (seed 0). A one-off bulk read cross-checks the fill kernel; each H2D
  // repeat re-patterns the pinned buffer with the independent seed 1.
  fill_host_pattern(golden, args.bytes, kDevSeed);
  const std::size_t n_words = args.bytes / sizeof(std::uint32_t);
  fill_pattern<<<1024, 256, 0, stream>>>(
      reinterpret_cast<std::uint32_t *>(bufs.dev), n_words, kDevSeed);
  cuda_check(cudaGetLastError(), "fill_pattern launch");
  cuda_check(cudaStreamSynchronize(stream), "fill_pattern sync");
  cuda_check(cudaMemcpy(device_image, bufs.dev, args.bytes,
                        cudaMemcpyDeviceToHost),
             "setup cudaMemcpy");
  if (std::memcmp(device_image, golden, args.bytes) != 0)
    die_msg("setup check failed: device fill kernel disagrees with the host "
            "golden image");

  fill_host_pattern(bufs.pinned, args.bytes, kHostSeed);

  // Fixed-seed permutation of the full-length segment slots (reproducible
  // scatter); the short tail slot, if any, keeps its identity so every slot
  // keeps its own length on both sides of the scatter plan.
  const std::size_t full_slots = args.bytes / args.seg;
  const std::vector<std::size_t> perm = make_perm(max_slots, full_slots);

  // cudaMemcpyBatchAsync availability: compile-time guard plus a runtime
  // probe (a driver may predate the API even when the toolkit is new). The
  // probe is a real 1-byte D2H copy; any error means "unavailable" and the
  // sticky error is cleared so it cannot poison later calls.
  bool batch_available = false;
#if CUDART_VERSION >= 12080
  {
    void *dsts[1] = {bufs.pinned};
    void *srcs[1] = {bufs.dev};
    std::size_t sizes[1] = {1};
    const cudaError_t err = batch_copy(dsts, srcs, sizes, 1, stream);
    cudaStreamSynchronize(stream);
    batch_available = (err == cudaSuccess);
    cudaGetLastError();  // clear any sticky error from the probe
  }
#endif

  std::printf("| variant | direction | bytes | submissions | mean ms | best ms "
              "| GB/s | check |\n");
  std::printf("|---|---|---|---|---|---|---|---|\n");
  const std::vector<Variant> variants = select_variants(args.variants);
  bool any_failed = false;
  for (const Variant v : variants) {
    if (v == Variant::Batch && !batch_available) {
      Row row;
      row.available = false;
      print_row(v, Direction::D2H, row);
      print_row(v, Direction::H2D, row);
      continue;
    }
    const std::vector<CopyTriple> plan = make_plan(v, args, &perm);
    if (plan.empty()) {
      std::printf("<!-- %s: nothing to copy at this size, skipped -->\n",
                  variant_name(v));
      continue;
    }
    for (const Direction dir : {Direction::D2H, Direction::H2D}) {
      const Row row =
          run_row(v, dir, plan, args, bufs, device_image, golden, stream,
                  start_ev, end_ev);
      any_failed = any_failed || !row.ok;
      print_row(v, dir, row);
    }
  }
  std::printf("\nGB/s is computed from the best repeat; per-variant "
              "submissions/s at best is in the HTML comments under each row. "
              "PCIe generation/width: see the `nvidia-smi` line run.sh adds "
              "above this table.\n");

  cudaEventDestroy(start_ev);
  cudaEventDestroy(end_ev);
  cudaStreamDestroy(stream);
  cudaFree(bufs.dev);
  cudaFreeHost(bufs.pinned);
  std::free(golden);
  std::free(device_image);
  if (any_failed) {
    std::fprintf(stderr, "error: at least one variant FAILED verification\n");
    return 1;
  }
  return 0;
}
