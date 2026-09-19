//! CPU stub-library coverage of the REAL compressor wave paths.
//!
//! A tiny host-compiled stub `ds41rt` shared library (existing CPU mock
//! convention: trivial bodies, no GPU numerics) backs `NativeLibrary`, so the
//! production `CompressorWeights::wave` and `CompressorWave::execute` code runs
//! end-to-end on CPU against recorded, host-backed allocations. This covers the
//! production-path behaviors the geometry unit tests cannot: the one-policy
//! budget/layout/allocation contract, budget failure before any allocator call,
//! missing pad-export rejection, and the exact enqueue call order with padded
//! projection rows and live downstream rows — including that only the stale
//! input tail is ever zeroed. No GPU claim is made: kernels only record calls.
use super::pad_rows::INPUT_ROW_BYTES;
use super::*;
use anyhow::{Context, Result};
use ds41rt_ffi::NativeLibrary;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

const STUB_C: &str = r#"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { void* ptr; size_t bytes; int device_id; uint64_t flags; } buf_t;
typedef struct { void* ptr; size_t bytes; uint64_t flags; } host_t;

#define KLOG_CAP 512
#define ALLOC_CAP 256
static int32_t klog_count = 0;
static char klog_name[KLOG_CAP][40];
static void* klog_ptr[KLOG_CAP][4];
static int64_t klog_val[KLOG_CAP][4];
static int32_t dev_alloc_count = 0;
static int64_t dev_alloc_bytes[ALLOC_CAP];
static int32_t host_alloc_count = 0;
static int64_t host_alloc_bytes[ALLOC_CAP];
static int32_t h2d_count = 0;
static int32_t stream_seq = 0;

static void klog(const char* name, void* a, void* b, void* c, void* d,
                 int64_t v0, int64_t v1) {
    if (klog_count < KLOG_CAP) {
        snprintf(klog_name[klog_count], sizeof klog_name[0], "%s", name);
        klog_ptr[klog_count][0] = a; klog_ptr[klog_count][1] = b;
        klog_ptr[klog_count][2] = c; klog_ptr[klog_count][3] = d;
        klog_val[klog_count][0] = v0; klog_val[klog_count][1] = v1;
    }
    klog_count++;
}

void stub_reset(void) {
    klog_count = 0; dev_alloc_count = 0; host_alloc_count = 0;
    h2d_count = 0;
}
int32_t stub_klog_count(void) { return klog_count; }
const char* stub_klog_name(int32_t i) {
    return (i >= 0 && i < klog_count && i < KLOG_CAP) ? klog_name[i] : "";
}
void* stub_klog_ptr(int32_t i, int32_t j) {
    return (i >= 0 && i < klog_count && i < KLOG_CAP && j >= 0 && j < 4)
        ? klog_ptr[i][j] : 0;
}
int64_t stub_klog_val(int32_t i, int32_t j) {
    return (i >= 0 && i < klog_count && i < KLOG_CAP && j >= 0 && j < 4)
        ? klog_val[i][j] : -1;
}
int32_t stub_dev_alloc_count(void) { return dev_alloc_count; }
int64_t stub_dev_alloc_bytes(int32_t i) {
    return (i >= 0 && i < dev_alloc_count && i < ALLOC_CAP) ? dev_alloc_bytes[i] : -1;
}
int32_t stub_host_alloc_count(void) { return host_alloc_count; }
int64_t stub_host_alloc_bytes(int32_t i) {
    return (i >= 0 && i < host_alloc_count && i < ALLOC_CAP) ? host_alloc_bytes[i] : -1;
}
int32_t stub_h2d_count(void) { return h2d_count; }

int32_t ds41rt_rdma_rc_endpoint_try_poll(void* handle, uint32_t s, uint32_t r, void* out) {
    (void)handle; (void)s; (void)r; (void)out; return 0;
}
int32_t ds41rt_alloc_device_buffer(size_t bytes, buf_t* out) {
    if (!out) return 1;
    out->ptr = calloc(1, bytes ? bytes : 1);
    out->bytes = bytes; out->device_id = 0; out->flags = 0;
    if (!out->ptr) return 1;
    if (dev_alloc_count < ALLOC_CAP) dev_alloc_bytes[dev_alloc_count] = (int64_t)bytes;
    dev_alloc_count++;
    return 0;
}
int32_t ds41rt_free_device_buffer(buf_t* buf) {
    if (buf && buf->ptr) { free(buf->ptr); buf->ptr = 0; }
    return 0;
}
int32_t ds41rt_alloc_host_buffer(size_t bytes, host_t* out) {
    if (!out) return 1;
    out->ptr = calloc(1, bytes ? bytes : 1);
    out->bytes = bytes; out->flags = 0;
    if (!out->ptr) return 1;
    if (host_alloc_count < ALLOC_CAP) host_alloc_bytes[host_alloc_count] = (int64_t)bytes;
    host_alloc_count++;
    return 0;
}
int32_t ds41rt_free_host_buffer(host_t* buf) {
    if (buf && buf->ptr) { free(buf->ptr); buf->ptr = 0; }
    return 0;
}
int32_t ds41rt_cuda_stream_create(void** out) {
    if (!out) return 1;
    *out = (void*)(uintptr_t)(0x51000000ull + (uint64_t)(++stream_seq));
    return 0;
}
int32_t ds41rt_cuda_stream_destroy(void* stream) { (void)stream; return 0; }
int32_t ds41rt_cuda_stream_synchronize(void* stream) { (void)stream; return 0; }
int32_t ds41rt_cuda_stream_query(void* stream, int32_t* ready) {
    (void)stream; if (ready) *ready = 1; return 0;
}
int32_t ds41rt_copy_h2d(buf_t dst, const void* src, size_t bytes) {
    if (!dst.ptr || !src) return 1;
    if (bytes > dst.bytes) return 1;
    memcpy(dst.ptr, src, bytes);
    h2d_count++;
    return 0;
}

int32_t ds41rt_v41_compressor_create(void* workspace, uint64_t bytes, void** output) {
    (void)workspace; (void)bytes;
    *output = (void*)(uintptr_t)1;
    return 0;
}
int32_t ds41rt_v41_compressor_destroy(void* handle) { (void)handle; return 0; }
int32_t ds41rt_v41_compressor_project(void* handle, const uint16_t* input,
        const uint16_t* weight, void* output, int32_t rows, int32_t ratio, void* stream) {
    (void)handle;
    klog("project", (void*)input, (void*)weight, output, stream, rows, ratio);
    return 0;
}
#ifndef DS41RT_WAVE_STUB_WITHOUT_PAD_EXPORT
// Faithful stub of the native contract: validated no-op for equal rows,
// otherwise zero EXACTLY the BF16 tail [rows,padded_rows) of the host-backed
// scratch — the live prefix is never touched.
int32_t ds41rt_v41_compressor_pad_input(uint16_t* input, int32_t rows,
        int32_t padded_rows, void* stream) {
    if (rows < 1 || rows > padded_rows || (padded_rows != 2 && padded_rows != 16)) return 1;
    if (!input) return 1;
    if (rows != padded_rows) {
        memset((uint8_t*)input + (size_t)rows * 10240, 0,
               (size_t)(padded_rows - rows) * 10240);
    }
    klog("pad_input", input, 0, 0, stream, rows, padded_rows);
    return 0;
}
#endif
int32_t ds41rt_v41_compressor_pool(const float* kv, const float* scores,
        const float* pending_kv, const float* pending_scores, const uint64_t* predecessors,
        const uint16_t* norm_weight, uint16_t* output, int32_t rows, int32_t slots,
        void* stream) {
    (void)pending_kv; (void)pending_scores; (void)predecessors; (void)norm_weight;
    klog("pool", (void*)kv, (void*)scores, output, stream, rows, slots);
    return 0;
}
int32_t ds41rt_v41_index_key_project(void* handle, const uint16_t* input,
        const uint16_t* weight, uint16_t* output, int32_t rows, void* stream) {
    (void)handle;
    klog("index_project", (void*)input, (void*)weight, output, stream, rows, 0);
    return 0;
}
int32_t ds41rt_v41_index_pack(const uint16_t* input, uint8_t* packed,
        uint8_t* scales, int32_t rows, void* stream) {
    klog("index_pack", (void*)input, packed, scales, stream, rows, 0);
    return 0;
}
int32_t ds41rt_v41_index_store(const uint8_t* packed, const uint8_t* scales,
        const uint64_t* destinations, uint8_t* cache, uint8_t* cache_scales,
        int32_t rows, uint64_t capacity, void* stream) {
    (void)packed; (void)scales; (void)destinations; (void)cache;
    (void)cache_scales; (void)rows; (void)capacity; (void)stream;
    return 0;
}
int32_t ds41rt_v41_index_weights_project(void* handle, const uint16_t* input,
        const uint16_t* weight, uint16_t* output, int32_t rows, void* stream) {
    (void)handle; (void)input; (void)weight; (void)output; (void)rows; (void)stream;
    return 0;
}
int32_t ds41rt_v41_index_query_prepare(const uint16_t* input, const float* frequencies,
        const uint16_t* weights, uint8_t* packed, uint8_t* scales,
        uint16_t* scaled_weights, int32_t rows, void* stream) {
    (void)input; (void)frequencies; (void)weights; (void)packed;
    (void)scales; (void)scaled_weights; (void)rows; (void)stream;
    return 0;
}
int32_t ds41rt_v41_compressed_kv_pack(const uint16_t* input, const float* frequencies,
        uint8_t* values, uint8_t* scales, int32_t rows, void* stream) {
    (void)frequencies;
    klog("kv_pack", (void*)input, values, scales, stream, rows, 0);
    return 0;
}
int32_t ds41rt_v41_compressed_kv_store(const uint8_t* values, const uint8_t* scales,
        const uint64_t* destinations, uint8_t* cache, uint8_t* cache_scales,
        int32_t rows, uint64_t capacity, void* stream) {
    (void)values; (void)scales; (void)destinations; (void)cache;
    (void)cache_scales; (void)rows; (void)capacity; (void)stream;
    return 0;
}
int32_t ds41rt_v41_backbone_frequencies(const uint64_t* positions,
        float* output, int32_t rows, int32_t layer, void* stream) {
    klog("frequencies", (void*)positions, output, 0, stream, rows, layer);
    return 0;
}
int32_t ds41rt_v41_attention_norm(const uint16_t* input, const uint16_t* weight,
        const float* frequencies, uint16_t* output, int32_t rows, int32_t dim, void* stream) {
    (void)frequencies;
    klog("norm", (void*)input, (void*)weight, output, stream, rows, dim);
    return 0;
}
int32_t ds41rt_v41_attention_rope(const uint16_t* input, const float* frequencies,
        uint16_t* output, int32_t rows, int32_t heads, int32_t inverse, void* stream) {
    (void)input; (void)frequencies; (void)output; (void)rows; (void)heads;
    (void)inverse; (void)stream;
    return 0;
}
int32_t ds41rt_v41_attention_kv(const uint16_t* input, const uint16_t* weight,
        const float* frequencies, uint16_t* output, int32_t rows, void* stream) {
    (void)input; (void)weight; (void)frequencies; (void)output; (void)rows; (void)stream;
    return 0;
}
int32_t ds41rt_v41_dspark_frequencies(const uint64_t* positions, float* output,
        int32_t rows, void* stream) {
    (void)positions; (void)output; (void)rows; (void)stream;
    return 0;
}
int32_t ds41rt_v41_dspark_embed(const uint16_t* table, const int32_t* tokens,
        uint16_t* residual, float* pre, int32_t requests, void* stream) {
    (void)table; (void)tokens; (void)residual; (void)pre; (void)requests; (void)stream;
    return 0;
}
int32_t ds41rt_v41_target_embed(const uint16_t* table, const int32_t* tokens,
        uint16_t* residual, float* pre, int32_t rows, void* stream) {
    (void)table; (void)tokens; (void)residual; (void)pre; (void)rows; (void)stream;
    return 0;
}
int32_t ds41rt_v41_dspark_tap(const uint16_t* input, uint16_t* output,
        int32_t rows, int32_t layer, void* stream) {
    (void)input; (void)output; (void)rows; (void)layer; (void)stream;
    return 0;
}
int32_t ds41rt_v41_dspark_terminal_layout(const uint16_t* residual,
        const float* pre, uint16_t* output, float* output_pre, int32_t requests, void* stream) {
    (void)residual; (void)pre; (void)output; (void)output_pre; (void)requests; (void)stream;
    return 0;
}
"#;

static STUB_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Compile the stub shared library. A missing/broken compiler or any I/O
/// failure FAILS the tests: these are the only production-path stub tests, and
/// a silently skipped build would report the suite green without coverage.
/// Every call builds a distinct library file so parallel tests get independent
/// stub state.
fn build_stub(with_pad_export: bool) -> Result<PathBuf> {
    let cc = Command::new("cc")
        .arg("--version")
        .output()
        .context("running the host C compiler `cc` for the compressor wave stub")?;
    anyhow::ensure!(
        cc.status.success(),
        "host C compiler `cc --version` failed: {}",
        String::from_utf8_lossy(&cc.stderr)
    );
    let unique = STUB_COUNTER.fetch_add(1, Ordering::Relaxed);
    let directory = std::env::temp_dir().join(format!(
        "ds41rt-compressor-wave-stub-{}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir_all(&directory)
        .with_context(|| format!("creating stub directory {}", directory.display()))?;
    let source = directory.join("stub.c");
    std::fs::write(&source, STUB_C)
        .with_context(|| format!("writing stub source {}", source.display()))?;
    let library = directory.join(if with_pad_export {
        "libds41rt_wave_stub.so"
    } else {
        "libds41rt_wave_stub_nopad.so"
    });
    let mut command = Command::new("cc");
    command
        .arg("-shared")
        .arg("-fPIC")
        .arg("-o")
        .arg(&library)
        .arg(&source);
    if !with_pad_export {
        command.arg("-DDS41RT_WAVE_STUB_WITHOUT_PAD_EXPORT");
    }
    let output = command
        .output()
        .context("running the host C compiler for the compressor wave stub")?;
    anyhow::ensure!(
        output.status.success(),
        "building the compressor wave stub library failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(library)
}

/// Recorded stub state of one loaded stub library.
struct StubWave {
    library: NativeLibrary,
}

impl StubWave {
    fn load(with_pad_export: bool) -> Self {
        let path = build_stub(with_pad_export).expect("the host C stub must build");
        Self {
            library: unsafe { NativeLibrary::load(path) }.expect("load stub library"),
        }
    }
    fn call<T: Copy>(&self, name: &[u8]) -> T {
        unsafe { self.library.raw_fn::<T>(name) }.unwrap_or_else(|error| {
            panic!("stub export {}: {error}", String::from_utf8_lossy(name))
        })
    }
    fn reset(&self) {
        let reset: unsafe extern "C" fn() = self.call(b"stub_reset");
        unsafe { reset() };
    }
    fn names(&self) -> Vec<String> {
        let count: unsafe extern "C" fn() -> i32 = self.call(b"stub_klog_count");
        let name: unsafe extern "C" fn(i32) -> *const std::ffi::c_char =
            self.call(b"stub_klog_name");
        (0..unsafe { count() })
            .map(|i| unsafe {
                std::ffi::CStr::from_ptr(name(i))
                    .to_string_lossy()
                    .into_owned()
            })
            .collect()
    }
    fn ptr(&self, entry: i32, index: i32) -> *mut std::ffi::c_void {
        let get: unsafe extern "C" fn(i32, i32) -> *mut std::ffi::c_void =
            self.call(b"stub_klog_ptr");
        unsafe { get(entry, index) }
    }
    fn val(&self, entry: i32, index: i32) -> i64 {
        let get: unsafe extern "C" fn(i32, i32) -> i64 = self.call(b"stub_klog_val");
        unsafe { get(entry, index) }
    }
    fn dev_alloc_bytes(&self) -> Vec<i64> {
        let count: unsafe extern "C" fn() -> i32 = self.call(b"stub_dev_alloc_count");
        let get: unsafe extern "C" fn(i32) -> i64 = self.call(b"stub_dev_alloc_bytes");
        (0..unsafe { count() }).map(|i| unsafe { get(i) }).collect()
    }
    fn host_alloc_count(&self) -> i32 {
        let count: unsafe extern "C" fn() -> i32 = self.call(b"stub_host_alloc_count");
        unsafe { count() }
    }
    fn dev_alloc_count(&self) -> i32 {
        let count: unsafe extern "C" fn() -> i32 = self.call(b"stub_dev_alloc_count");
        unsafe { count() }
    }
    fn h2d_count(&self) -> i32 {
        let count: unsafe extern "C" fn() -> i32 = self.call(b"stub_h2d_count");
        unsafe { count() }
    }
    fn entry(&self, name: &str) -> i32 {
        self.entry_nth(name, 0)
    }
    fn entry_nth(&self, name: &str, occurrence: usize) -> i32 {
        self.names()
            .iter()
            .enumerate()
            .filter(|(_, logged)| logged == &name)
            .nth(occurrence)
            .unwrap_or_else(|| panic!("{name} call is missing: {:?}", self.names()))
            .0 as i32
    }
    fn assert_order(&self, expected: &[&str]) {
        assert_eq!(self.names(), expected, "recorded production call order");
    }
}

fn stub_state(library: &NativeLibrary, layer: usize) -> Result<CompressorState<'_>> {
    CompressorState::new(library, layer, 2, 4, usize::MAX)
}

fn fill(input: ds41rt_ffi::Ds41rtDeviceBuffer, byte: u8) {
    let bytes = unsafe { std::slice::from_raw_parts_mut(input.ptr.cast::<u8>(), input.bytes) };
    bytes.fill(byte);
}

fn row_bytes(input: ds41rt_ffi::Ds41rtDeviceBuffer, row: usize) -> Vec<u8> {
    unsafe { std::slice::from_raw_parts(input.ptr.cast::<u8>().add(row * INPUT_ROW_BYTES), INPUT_ROW_BYTES) }
        .to_vec()
}

#[test]
fn wave_budget_failure_precedes_any_allocator_call() {
    let stub = StubWave::load(true);
    let weights = CompressorWeights::with_stub_tensors(&stub.library, 2).unwrap();
    stub.reset();
    let needed = CompressorWave::device_bytes(2, 16, PadRowsPolicy::Pad(16)).unwrap();
    let error = weights
        .wave(16, needed - 1, PadRowsPolicy::Pad(16))
        .err()
        .expect("an under-budget wave must fail");
    assert!(
        error.to_string().contains("exceeds budget"),
        "unexpected error: {error:#}"
    );
    // The budget rejection happens before ANY allocation or upload callback.
    assert_eq!(stub.dev_alloc_count(), 0, "device allocator must not run");
    assert_eq!(stub.host_alloc_count(), 0, "host allocator must not run");
    assert_eq!(stub.h2d_count(), 0, "no upload may run");
    // An invalid row capacity fails the same way, also without allocating.
    assert!(
        weights.wave(0, usize::MAX, PadRowsPolicy::Off).is_err(),
        "invalid row capacity must fail"
    );
    assert_eq!(stub.dev_alloc_count(), 0);
}

#[test]
fn wave_allocates_exact_geometry_from_one_captured_policy() {
    let stub = StubWave::load(true);
    let weights = CompressorWeights::with_stub_tensors(&stub.library, 2).unwrap();
    // Ratio two, capacity one: the historical allocation list.
    stub.reset();
    let off = weights.wave(1, usize::MAX, PadRowsPolicy::Off).unwrap();
    let off_allocs = stub.dev_alloc_bytes();
    drop(off);
    // The same capacity under the pad-16 policy: only the input/projected/
    // scores entries grow, by exactly (pad-capacity)*(10240+2048+2048) bytes.
    stub.reset();
    let padded = weights.wave(1, usize::MAX, PadRowsPolicy::Pad(16)).unwrap();
    let pad_allocs = stub.dev_alloc_bytes();
    let mut expected = off_allocs.clone();
    expected[1] += 15 * 10240;
    expected[2] += 15 * 2048;
    expected[3] += 15 * 2048;
    assert_eq!(pad_allocs, expected, "only the padded trio may grow");
    // The growth is exactly (pad-capacity)*(input+projected+scores row bytes).
    let grown: i64 = pad_allocs.iter().sum::<i64>() - off_allocs.iter().sum::<i64>();
    assert_eq!(grown as usize, 15 * (10240 + 2048 + 2048));
    assert_eq!(
        pad_allocs.iter().sum::<i64>() as usize,
        CompressorWave::device_bytes(2, 1, PadRowsPolicy::Pad(16)).unwrap(),
        "allocated device bytes must equal the budget formula"
    );
    assert_eq!(
        off_allocs.iter().sum::<i64>() as usize,
        CompressorWave::device_bytes(2, 1, PadRowsPolicy::Off).unwrap()
    );
    assert_eq!(off_allocs[0] as usize, 4 * 1024 * 1024, "workspace first");
    drop(padded);
    // Ratio one never grows even with the policy on: identical allocation list.
    let weights1 = CompressorWeights::with_stub_tensors(&stub.library, 20).unwrap();
    stub.reset();
    let ratio1 = weights1
        .wave(1, usize::MAX, PadRowsPolicy::Pad(16))
        .unwrap();
    let ratio1_allocs = stub.dev_alloc_bytes();
    drop(ratio1);
    stub.reset();
    let ratio1_off = weights1.wave(1, usize::MAX, PadRowsPolicy::Off).unwrap();
    assert_eq!(ratio1_allocs, stub.dev_alloc_bytes());
    drop(ratio1_off);
}

#[test]
fn wave_without_pad_export_rejects_policy_before_wave_allocations() {
    let stub = StubWave::load(false);
    let weights = CompressorWeights::with_stub_tensors(&stub.library, 2).unwrap();
    // The OFF policy keeps working against an old library without the export.
    stub.reset();
    let off = weights
        .wave(16, usize::MAX, PadRowsPolicy::Off)
        .expect("off policy must keep working without the pad export");
    drop(off);
    // Selecting the policy must fail clearly, before the wave allocations (the
    // workspace/kernel handle is the only allocation made) and long before any
    // enqueue could run.
    stub.reset();
    let error = weights
        .wave(16, usize::MAX, PadRowsPolicy::Pad(2))
        .err()
        .expect("selected policy without the export must fail");
    assert!(
        error
            .to_string()
            .contains("ds41rt_v41_compressor_pad_input"),
        "unexpected error: {error:#}"
    );
    assert_eq!(
        stub.dev_alloc_count(),
        1,
        "only the workspace may be allocated before the export check"
    );
    assert_eq!(stub.names().len(), 0, "no kernel launch may run");
}

#[test]
fn execute_zeroes_only_stale_tail_and_launches_in_padded_order() {
    let stub = StubWave::load(true);
    let weights = CompressorWeights::with_stub_tensors(&stub.library, 2).unwrap();
    let mut wave = weights
        .wave(16, usize::MAX, PadRowsPolicy::Pad(16))
        .unwrap();
    let mut state = stub_state(&stub.library, 2).unwrap();
    let lease = state.begin_request(0, 11).unwrap();
    let input = wave.input();
    assert_eq!(input.bytes, 16 * INPUT_ROW_BYTES, "padded input allocation");
    fill(input, 0xA5);
    stub.reset();
    let chunk = [CompressorChunk {
        lease,
        position: 0,
        tokens: 1,
    }];
    let output = unsafe { wave.execute(&state, &chunk) }.expect("stub execute");
    assert_eq!(output.completed.len(), 0, "incomplete ratio-two row");
    assert_eq!(stub.dev_alloc_count(), 0, "enqueue must not allocate device buffers");
    assert_eq!(stub.host_alloc_count(), 0, "enqueue must not allocate host buffers");
    // Exact production order on one serialized stream: frequencies, the tail
    // memset, WKV and Wgate projections at PADDED rows, then every downstream
    // consumer at LIVE rows.
    stub.assert_order(&[
        "frequencies",
        "pad_input",
        "project",
        "project",
        "pool",
        "index_project",
        "norm",
        "index_pack",
        "kv_pack",
    ]);
    let frequencies = stub.entry("frequencies");
    let pad = stub.entry("pad_input");
    let (wkv, wgate) = (stub.entry("project"), stub.entry("project") + 1);
    let pool = stub.entry("pool");
    let index_project = stub.entry("index_project");
    let norm = stub.entry("norm");
    let index_pack = stub.entry("index_pack");
    let kv_pack = stub.entry("kv_pack");
    assert_eq!(stub.val(frequencies, 0), 1, "live frequency rows");
    assert_eq!(stub.val(frequencies, 1), 2, "producer layer");
    assert_eq!(stub.val(pad, 0), 1, "pad live rows");
    assert_eq!(stub.val(pad, 1), 16, "pad padded rows");
    assert_eq!(stub.val(wkv, 0), 16, "WKV rows padded");
    assert_eq!(stub.val(wkv, 1), 2, "WKV ratio");
    assert_eq!(stub.val(wgate, 0), 16, "Wgate rows padded");
    assert_eq!(stub.val(wgate, 1), 2, "Wgate ratio");
    assert_eq!(stub.val(pool, 0), 1, "pool keeps live rows");
    assert_eq!(stub.val(pool, 1), 2, "pool slot count");
    assert_eq!(stub.val(index_project, 0), 1, "index projection live rows");
    assert_eq!(stub.val(norm, 0), 1, "index norm live rows");
    assert_eq!(stub.val(norm, 1), 128, "index norm dim");
    assert_eq!(stub.val(index_pack, 0), 1, "index pack live rows");
    assert_eq!(stub.val(kv_pack, 0), 1, "kv pack live rows");
    // Distinct buffer identities, one shared stream for every launch.
    let stream = stub.ptr(pad, 3);
    assert!(!stream.is_null());
    for entry in [
        frequencies,
        pad,
        wkv,
        wgate,
        pool,
        index_project,
        norm,
        index_pack,
        kv_pack,
    ] {
        assert_eq!(stub.ptr(entry, 3), stream, "serialized stream identity");
    }
    let input_ptr = stub.ptr(wkv, 0);
    let projected = stub.ptr(wkv, 2);
    let scores = stub.ptr(wgate, 2);
    let pooled = stub.ptr(pool, 2);
    let mut operands = vec![input_ptr, projected, scores, pooled];
    operands.extend([
        stub.ptr(frequencies, 1),
        stub.ptr(index_project, 2),
        stub.ptr(norm, 2),
        stub.ptr(index_pack, 1),
        stub.ptr(kv_pack, 1),
    ]);
    let mut unique = operands.clone();
    unique.dedup();
    assert_eq!(
        unique.len(),
        operands.len(),
        "operand pointers are distinct"
    );
    assert_eq!(input_ptr, input.ptr, "the wave input is the padded operand");
    // Only the tail was zeroed: the live prefix keeps its sentinel bytes and
    // every padded row is exactly zero.
    assert_eq!(row_bytes(input, 0), vec![0xA5; INPUT_ROW_BYTES], "live prefix untouched");
    for row in 1..16 {
        assert_eq!(row_bytes(input, row), vec![0u8; INPUT_ROW_BYTES], "tail row {row} zeroed");
    }
}

#[test]
fn execute_smaller_live_rows_after_larger_zeroes_stale_tail() {
    let stub = StubWave::load(true);
    let weights = CompressorWeights::with_stub_tensors(&stub.library, 2).unwrap();
    let mut wave = weights
        .wave(16, usize::MAX, PadRowsPolicy::Pad(16))
        .unwrap();
    let mut state = stub_state(&stub.library, 2).unwrap();
    let input = wave.input();
    // A larger live batch first: rows [12,16) are zeroed by its pad call.
    let lease = state.begin_request(0, 21).unwrap();
    fill(input, 0x3C);
    let larger = [CompressorChunk {
        lease,
        position: 0,
        tokens: 12,
    }];
    unsafe { wave.execute(&state, &larger) }.expect("larger stub execute");
    assert_eq!(row_bytes(input, 11), vec![0x3C; INPUT_ROW_BYTES], "live prefix untouched");
    assert_eq!(row_bytes(input, 12), vec![0u8; INPUT_ROW_BYTES], "larger tail zeroed");
    assert_eq!(row_bytes(input, 15), vec![0u8; INPUT_ROW_BYTES], "larger tail zeroed");
    state.release(lease).unwrap();
    // A smaller live batch on the SAME wave: only row zero is fresh input; the
    // stale rows [1,16) left by the larger batch must be re-zeroed.
    let lease = state.begin_request(0, 22).unwrap();
    let fresh = unsafe { std::slice::from_raw_parts_mut(input.ptr.cast::<u8>(), INPUT_ROW_BYTES) };
    fresh.fill(0x77);
    let smaller = [CompressorChunk {
        lease,
        position: 0,
        tokens: 1,
    }];
    unsafe { wave.execute(&state, &smaller) }.expect("smaller stub execute");
    assert_eq!(
        row_bytes(input, 0),
        vec![0x77; INPUT_ROW_BYTES],
        "fresh live row untouched"
    );
    for row in 1..16 {
        assert_eq!(
            row_bytes(input, row),
            vec![0u8; INPUT_ROW_BYTES],
            "stale tail row {row} re-zeroed"
        );
    }
    let pad_calls = stub
        .names()
        .iter()
        .filter(|name| *name == "pad_input")
        .count();
    assert_eq!(pad_calls, 2, "both executions recorded their pad call");
    // The second (smaller) execution padded from live row 1 again.
    let second_pad = stub.entry_nth("pad_input", 1);
    assert_eq!(stub.val(second_pad, 0), 1);
    assert_eq!(stub.val(second_pad, 1), 16);
}

#[test]
fn execute_skips_pad_zero_for_off_equal_above_and_ratio_one() {
    // OFF policy: exact historical shape — no memset, live projection rows.
    let stub = StubWave::load(true);
    let weights = CompressorWeights::with_stub_tensors(&stub.library, 2).unwrap();
    let mut wave = weights.wave(17, usize::MAX, PadRowsPolicy::Off).unwrap();
    let mut state = stub_state(&stub.library, 2).unwrap();
    let lease = state.begin_request(0, 31).unwrap();
    let chunk = [CompressorChunk {
        lease,
        position: 0,
        tokens: 1,
    }];
    unsafe { wave.execute(&state, &chunk) }.unwrap();
    stub.assert_order(&[
        "frequencies",
        "project",
        "project",
        "pool",
        "index_project",
        "norm",
        "index_pack",
        "kv_pack",
    ]);
    assert_eq!(stub.val(stub.entry("project"), 0), 1, "live rows when off");
    drop(wave);
    state.release(lease).unwrap();

    // Pad policy with live rows equal to and above the pad: no memset, GEMM
    // rows exactly the live count.
    let mut wave = weights
        .wave(17, usize::MAX, PadRowsPolicy::Pad(16))
        .unwrap();
    let mut state = stub_state(&stub.library, 2).unwrap();
    for (request, tokens) in [(32u64, 16u32), (33, 17)] {
        let lease = state.begin_request(0, request).unwrap();
        stub.reset();
        let chunk = [CompressorChunk {
            lease,
            position: 0,
            tokens,
        }];
        unsafe { wave.execute(&state, &chunk) }.unwrap();
        let names = stub.names();
        assert!(
            !names.iter().any(|name| name == "pad_input"),
            "no pad memset at or above the pad: {names:?}"
        );
        assert_eq!(stub.val(stub.entry("project"), 0), tokens as i64);
        assert_eq!(stub.val(stub.entry("pool"), 0), tokens as i64);
        state.release(lease).unwrap();
    }
    drop(wave);

    // Ratio one never pads: single WKV projection at live rows, no Wgate, no
    // pooling — the historical norm path.
    let weights1 = CompressorWeights::with_stub_tensors(&stub.library, 20).unwrap();
    let mut wave = weights1
        .wave(16, usize::MAX, PadRowsPolicy::Pad(16))
        .unwrap();
    let mut state = stub_state(&stub.library, 20).unwrap();
    let lease = state.begin_request(0, 34).unwrap();
    stub.reset();
    let chunk = [CompressorChunk {
        lease,
        position: 0,
        tokens: 1,
    }];
    unsafe { wave.execute(&state, &chunk) }.unwrap();
    stub.assert_order(&[
        "frequencies",
        "project",
        "norm",
        "index_project",
        "norm",
        "index_pack",
        "kv_pack",
    ]);
    assert_eq!(stub.val(stub.entry("project"), 0), 1, "ratio-one live rows");
    assert_eq!(stub.val(stub.entry("project"), 1), 1, "ratio-one ratio");
    assert_eq!(stub.val(stub.entry("norm"), 1), 512, "pooled norm dim");
}
