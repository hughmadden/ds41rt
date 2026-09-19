//! Native CSA2 projection and immutable-state ratio-two pooling.
use crate::{Ds41rtDeviceBuffer, NativeLibrary};
use anyhow::{ensure, Context, Result};
use std::ffi::{c_int, c_void};
type Create = unsafe extern "C" fn(*mut c_void, u64, *mut *mut c_void) -> i32;
type Destroy = unsafe extern "C" fn(*mut c_void) -> i32;
type Project = unsafe extern "C" fn(
    *mut c_void,
    *const u16,
    *const u16,
    *mut c_void,
    i32,
    i32,
    *mut c_void,
) -> i32;
type IndexProject =
    unsafe extern "C" fn(*mut c_void, *const u16, *const u16, *mut u16, i32, *mut c_void) -> i32;
type PadInput = unsafe extern "C" fn(*mut u16, i32, i32, *mut c_void) -> i32;
type QueryPrepare = unsafe extern "C" fn(
    *const u16,
    *const f32,
    *const u16,
    *mut u8,
    *mut u8,
    *mut u16,
    i32,
    *mut c_void,
) -> i32;
type IndexPack = unsafe extern "C" fn(*const u16, *mut u8, *mut u8, i32, *mut c_void) -> i32;
type IndexStore = unsafe extern "C" fn(
    *const u8,
    *const u8,
    *const u64,
    *mut u8,
    *mut u8,
    i32,
    u64,
    *mut c_void,
) -> i32;
type Pool = unsafe extern "C" fn(
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const u64,
    *const u16,
    *mut u16,
    i32,
    i32,
    *mut c_void,
) -> i32;
pub struct V41Compressor<'a> {
    _library: &'a NativeLibrary,
    handle: *mut c_void,
    destroy: Destroy,
    project: Project,
    index_project: IndexProject,
    weights_project: IndexProject,
    query_prepare: QueryPrepare,
    index_pack: IndexPack,
    index_store: IndexStore,
    pool: Pool,
    pad_input: Option<PadInput>,
    /// Device of the exclusively owned workspace captured at handle creation;
    /// every launched operand must live on this device.
    workspace_device: c_int,
}
fn buffer(b: Ds41rtDeviceBuffer, bytes: usize) -> Result<()> {
    ensure!(
        !b.ptr.is_null() && b.bytes >= bytes,
        "compressor buffer is null or undersized"
    );
    Ok(())
}
/// Pure extent/shape/device validation for the optional pad-input export. Kept
/// separate from the native call so CPU tests can check rows/padded-rows/buffer
/// rules with synthetic device buffers. No CUDA state is queried: the workspace
/// device is the ordinal captured at handle creation.
fn validate_pad_input_buffers(
    input: Ds41rtDeviceBuffer,
    rows: usize,
    padded_rows: usize,
    workspace_device: c_int,
) -> Result<()> {
    ensure!(
        padded_rows == 2 || padded_rows == 16,
        "invalid compressor pad rows"
    );
    ensure!(
        (1..=padded_rows).contains(&rows),
        "invalid compressor pad live rows"
    );
    buffer(input, padded_rows * 10240)?;
    ensure!(
        input.device_id == workspace_device,
        "compressor pad input device {} differs from the workspace device {}",
        input.device_id,
        workspace_device
    );
    ensure!(
        input.ptr as usize % 2 == 0,
        "compressor pad input is not 2-byte aligned"
    );
    ensure!(
        (input.ptr as usize).checked_add(padded_rows * 10240).is_some(),
        "compressor pad input extent overflows the address space"
    );
    Ok(())
}
impl NativeLibrary {
    /// # Safety
    /// Workspace stays on the current device, exclusively used and live through
    /// handle/graph destruction. Drain all work before dropping this handle.
    pub unsafe fn v41_compressor(
        &self,
        workspace: Ds41rtDeviceBuffer,
    ) -> Result<V41Compressor<'_>> {
        buffer(workspace, V41Compressor::WORKSPACE_BYTES)?;
        let create: Create = unsafe { *self.lib.get(b"ds41rt_v41_compressor_create")? };
        let destroy: Destroy = unsafe { *self.lib.get(b"ds41rt_v41_compressor_destroy")? };
        let project: Project = unsafe { *self.lib.get(b"ds41rt_v41_compressor_project")? };
        let index_project: IndexProject =
            unsafe { *self.lib.get(b"ds41rt_v41_index_key_project")? };
        let index_pack: IndexPack = unsafe { *self.lib.get(b"ds41rt_v41_index_pack")? };
        let index_store: IndexStore = unsafe { *self.lib.get(b"ds41rt_v41_index_store")? };
        let weights_project: IndexProject =
            unsafe { *self.lib.get(b"ds41rt_v41_index_weights_project")? };
        let query_prepare: QueryPrepare =
            unsafe { *self.lib.get(b"ds41rt_v41_index_query_prepare")? };
        let pool: Pool = unsafe { *self.lib.get(b"ds41rt_v41_compressor_pool")? };
        // Optional EXPERIMENTAL export: older native libraries keep working with
        // the pad-rows policy off; selecting the policy without this export must
        // fail clearly before any enqueue (checked via `pad_input_supported`).
        let pad_input: Option<PadInput> = unsafe {
            self.lib
                .get(b"ds41rt_v41_compressor_pad_input")
                .ok()
                .map(|symbol| *symbol)
        };
        let mut handle = std::ptr::null_mut();
        let status = unsafe { create(workspace.ptr, workspace.bytes as u64, &mut handle) };
        ensure!(status == 0, "native compressor create status {status}");
        Ok(V41Compressor {
            _library: self,
            handle,
            destroy,
            project,
            index_project,
            weights_project,
            query_prepare,
            index_pack,
            index_store,
            pool,
            pad_input,
            workspace_device: workspace.device_id,
        })
    }
}
impl V41Compressor<'_> {
    /// # Safety
    /// BF16 hidden [rows,5120], weight [32,5120], disjoint BF16 [rows,32] output
    /// live on the handle device. Serialize use of the handle/workspace.
    pub unsafe fn weights_project(
        &self,
        input: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            (1..=4096).contains(&rows),
            "invalid index weight projection rows"
        );
        buffer(input, rows * 10240)?;
        buffer(weight, 32 * 5120 * 2)?;
        buffer(output, rows * 64)?;
        let status = unsafe {
            (self.weights_project)(
                self.handle,
                input.ptr.cast(),
                weight.ptr.cast(),
                output.ptr.cast(),
                rows as i32,
                stream,
            )
        };
        ensure!(
            status == 0,
            "native index weight projection status {status}"
        );
        Ok(())
    }
    /// # Safety
    /// BF16 queries [rows,32,128], FP32 frequencies [rows,32,2], BF16 head
    /// weights [rows,32], disjoint FP4/E8M0 [rows,32,64/4] and BF16 [rows,32]
    /// outputs live on the stream device. Inputs are finite. No normalization
    /// is applied here: queries come from the model's normalized query-rank input.
    pub unsafe fn query_prepare(
        &self,
        input: Ds41rtDeviceBuffer,
        frequencies: Ds41rtDeviceBuffer,
        weights: Ds41rtDeviceBuffer,
        packed: Ds41rtDeviceBuffer,
        scales: Ds41rtDeviceBuffer,
        scaled_weights: Ds41rtDeviceBuffer,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!((1..=4096).contains(&rows), "invalid index query rows");
        for (b, n) in [
            (input, 8192),
            (frequencies, 256),
            (weights, 64),
            (packed, 2048),
            (scales, 128),
            (scaled_weights, 64),
        ] {
            buffer(b, rows * n)?;
        }
        let status = unsafe {
            (self.query_prepare)(
                input.ptr.cast(),
                frequencies.ptr.cast(),
                weights.ptr.cast(),
                packed.ptr.cast(),
                scales.ptr.cast(),
                scaled_weights.ptr.cast(),
                rows as i32,
                stream,
            )
        };
        ensure!(status == 0, "native index query prepare status {status}");
        Ok(())
    }

    /// # Safety
    /// All disjoint spans are live on the stream device. Each in-range U64
    /// destination is unique and owned by the committing request; other values
    /// skip writes. Validate acceptance and reserve physical pages before launch.
    pub unsafe fn index_store(
        &self,
        packed: Ds41rtDeviceBuffer,
        scales: Ds41rtDeviceBuffer,
        destinations: Ds41rtDeviceBuffer,
        cache: Ds41rtDeviceBuffer,
        cache_scales: Ds41rtDeviceBuffer,
        rows: usize,
        capacity: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            (1..=4096).contains(&rows) && (1..=64 * 1048576).contains(&capacity),
            "invalid index store shape"
        );
        buffer(packed, rows * 64)?;
        buffer(scales, rows * 4)?;
        buffer(destinations, rows * 8)?;
        buffer(cache, capacity * 64)?;
        buffer(cache_scales, capacity * 4)?;
        let status = unsafe {
            (self.index_store)(
                packed.ptr.cast(),
                scales.ptr.cast(),
                destinations.ptr.cast(),
                cache.ptr.cast(),
                cache_scales.ptr.cast(),
                rows as i32,
                capacity as u64,
                stream,
            )
        };
        ensure!(status == 0, "native index store status {status}");
        Ok(())
    }
    /// # Safety
    /// Finite BF16 vectors [rows,128] and disjoint packed [rows,64] / scale
    /// [rows,4] byte outputs are live on the stream device. Flatten query heads
    /// into rows when needed. This produces proposals, not committed cache rows.
    pub unsafe fn index_pack(
        &self,
        input: Ds41rtDeviceBuffer,
        packed: Ds41rtDeviceBuffer,
        scales: Ds41rtDeviceBuffer,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!((1..=131072).contains(&rows), "invalid index packing rows");
        buffer(input, rows * 256)?;
        buffer(packed, rows * 64)?;
        buffer(scales, rows * 4)?;
        let status = unsafe {
            (self.index_pack)(
                input.ptr.cast(),
                packed.ptr.cast(),
                scales.ptr.cast(),
                rows as i32,
                stream,
            )
        };
        ensure!(status == 0, "native index packing status {status}");
        Ok(())
    }
    /// # Safety
    /// Initialized unrotated BF16 latents [rows,512], weight [128,512] and
    /// disjoint output [rows,128] are live on the handle's device. Serialize use
    /// with all other operations sharing the handle/workspace.
    pub unsafe fn index_project(
        &self,
        input: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!((1..=4096).contains(&rows), "invalid index projection rows");
        buffer(input, rows * 1024)?;
        buffer(weight, 128 * 512 * 2)?;
        buffer(output, rows * 256)?;
        let status = unsafe {
            (self.index_project)(
                self.handle,
                input.ptr.cast(),
                weight.ptr.cast(),
                output.ptr.cast(),
                rows as i32,
                stream,
            )
        };
        ensure!(status == 0, "native index projection status {status}");
        Ok(())
    }

    pub const WORKSPACE_BYTES: usize = 4 * 1024 * 1024;
    /// # Safety
    /// BF16 input [rows,5120] and weight [512,5120] are initialized on this
    /// device. Output is BF16 for ratio one, FP32 for ratio two. Buffers and
    /// exclusive workspace remain live through completion, with disjoint output.
    pub unsafe fn project(
        &self,
        input: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        ratio: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            (1..=4096).contains(&rows) && (1..=2).contains(&ratio),
            "invalid compressor projection shape"
        );
        buffer(input, rows * 10240)?;
        buffer(weight, 512 * 10240)?;
        buffer(output, rows * 512 * if ratio == 2 { 4 } else { 2 })?;
        let status = unsafe {
            (self.project)(
                self.handle,
                input.ptr.cast(),
                weight.ptr.cast(),
                output.ptr,
                rows as i32,
                ratio as i32,
                stream,
            )
        };
        ensure!(status == 0, "native compressor projection status {status}");
        Ok(())
    }
    /// Whether the loaded native library exports `ds41rt_v41_compressor_pad_input`.
    /// Waves that select the EXPERIMENTAL pad-rows policy must check this before
    /// enqueueing so a stale native library fails clearly instead of silently
    /// falling back to unpadded geometry.
    pub fn pad_input_supported(&self) -> bool {
        self.pad_input.is_some()
    }
    /// # Safety
    /// Caller-owned writable BF16 [padded_rows,5120] scratch is live on the
    /// handle's workspace device through the recorded memset (including CUDA
    /// graph capture). Only the tail [rows,padded_rows) is zeroed; the live
    /// prefix is never touched. This performs no allocation and leaves the
    /// workspace and weights untouched; no CUDA state is queried. Requires
    /// 1<=rows<=padded_rows with padded_rows in {2,16}, a non-null, 2-byte
    /// aligned input of at least padded_rows*10240 bytes on the workspace
    /// device whose padded extent cannot overflow the address space;
    /// rows==padded_rows is a validated no-op.
    pub unsafe fn pad_input(
        &self,
        input: Ds41rtDeviceBuffer,
        rows: usize,
        padded_rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        let pad_input = self.pad_input.context(
            "native library does not export ds41rt_v41_compressor_pad_input; \
             the DS41RT_COMPRESSOR_PAD_ROWS policy requires a native build that provides it",
        )?;
        validate_pad_input_buffers(input, rows, padded_rows, self.workspace_device)?;
        let status = unsafe {
            pad_input(
                input.ptr.cast::<u16>(),
                rows as i32,
                padded_rows as i32,
                stream,
            )
        };
        ensure!(status == 0, "native compressor pad input status {status}");
        Ok(())
    }
    /// # Safety
    /// FP32 projections [rows,512], committed pending values [slots,512], U64
    /// predecessors [rows], BF16 weight [512] and output [rows,512] are live on
    /// the stream device. Output is disjoint. A completed row references its
    /// chronological predecessor: pending slot, or slots + earlier input row.
    /// Incomplete rows use u64::MAX. Validate request leases and causal positions
    /// before preparing descriptors; commit only accepted projections afterward.
    pub unsafe fn pool(
        &self,
        kv: Ds41rtDeviceBuffer,
        scores: Ds41rtDeviceBuffer,
        pending_kv: Ds41rtDeviceBuffer,
        pending_scores: Ds41rtDeviceBuffer,
        predecessors: Ds41rtDeviceBuffer,
        norm_weight: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        slots: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            (1..=4096).contains(&rows) && (1..=16).contains(&slots),
            "invalid compressor pooling shape"
        );
        for b in [kv, scores] {
            buffer(b, rows * 2048)?;
        }
        for b in [pending_kv, pending_scores] {
            buffer(b, slots * 2048)?;
        }
        buffer(predecessors, rows * 8)?;
        buffer(norm_weight, 1024)?;
        buffer(output, rows * 1024)?;
        let status = unsafe {
            (self.pool)(
                kv.ptr.cast(),
                scores.ptr.cast(),
                pending_kv.ptr.cast(),
                pending_scores.ptr.cast(),
                predecessors.ptr.cast(),
                norm_weight.ptr.cast(),
                output.ptr.cast(),
                rows as i32,
                slots as i32,
                stream,
            )
        };
        ensure!(status == 0, "native compressor pool status {status}");
        Ok(())
    }
}
impl Drop for V41Compressor<'_> {
    fn drop(&mut self) {
        let status = unsafe { (self.destroy)(self.handle) };
        if status != 0 {
            eprintln!("native compressor destruction status {status}");
        }
    }
}

#[cfg(test)]
mod pad_input_tests {
    use super::*;
    use std::path::PathBuf;
    use std::process::Command;
    use std::sync::atomic::{AtomicU64, Ordering};

    // CPU mock convention: a tiny stub shared library built with the host C
    // compiler. It implements the exact compressor export surface with trivial
    // bodies and records every pad-input call so tests can verify the optional
    // symbol resolution, argument pass-through, and extent/shape validation
    // without a GPU. No fake GPU numerics are produced: the stub never computes.
    const STUB_C: &str = r#"
#include <stdint.h>

static int32_t pad_calls = 0;
static void* pad_last_input = 0;
static int32_t pad_last_rows = -1;
static int32_t pad_last_padded_rows = -1;
static void* pad_last_stream = 0;

int32_t ds41rt_rdma_rc_endpoint_try_poll(void* endpoint, uint32_t max_send,
        uint32_t max_recv, void* out) {
    (void)endpoint; (void)max_send; (void)max_recv; (void)out;
    return 0;
}
int32_t ds41rt_v41_compressor_create(void* workspace, uint64_t bytes, void** output) {
    (void)workspace; (void)bytes;
    *output = (void*)(uintptr_t)1;
    return 0;
}
int32_t ds41rt_v41_compressor_destroy(void* handle) {
    (void)handle;
    return 0;
}
int32_t ds41rt_v41_compressor_project(void* handle, const uint16_t* input,
        const uint16_t* weight, void* output, int32_t rows, int32_t ratio, void* stream) {
    (void)handle; (void)input; (void)weight; (void)output;
    (void)rows; (void)ratio; (void)stream;
    return 0;
}
int32_t ds41rt_v41_index_key_project(void* handle, const uint16_t* input,
        const uint16_t* weight, uint16_t* output, int32_t rows, void* stream) {
    (void)handle; (void)input; (void)weight; (void)output; (void)rows; (void)stream;
    return 0;
}
int32_t ds41rt_v41_index_weights_project(void* handle, const uint16_t* input,
        const uint16_t* weight, uint16_t* output, int32_t rows, void* stream) {
    (void)handle; (void)input; (void)weight; (void)output; (void)rows; (void)stream;
    return 0;
}
int32_t ds41rt_v41_index_pack(const uint16_t* input, uint8_t* packed,
        uint8_t* scales, int32_t rows, void* stream) {
    (void)input; (void)packed; (void)scales; (void)rows; (void)stream;
    return 0;
}
int32_t ds41rt_v41_index_store(const uint8_t* packed, const uint8_t* scales,
        const uint64_t* destinations, uint8_t* cache, uint8_t* cache_scales,
        int32_t rows, uint64_t capacity, void* stream) {
    (void)packed; (void)scales; (void)destinations; (void)cache;
    (void)cache_scales; (void)rows; (void)capacity; (void)stream;
    return 0;
}
int32_t ds41rt_v41_compressor_pool(const float* kv, const float* scores,
        const float* pending_kv, const float* pending_scores, const uint64_t* predecessors,
        const uint16_t* norm_weight, uint16_t* output, int32_t rows, int32_t slots,
        void* stream) {
    (void)kv; (void)scores; (void)pending_kv; (void)pending_scores;
    (void)predecessors; (void)norm_weight; (void)output;
    (void)rows; (void)slots; (void)stream;
    return 0;
}
int32_t ds41rt_v41_index_query_prepare(const uint16_t* input, const float* frequencies,
        const uint16_t* weights, uint8_t* packed, uint8_t* scales,
        uint16_t* scaled_weights, int32_t rows, void* stream) {
    (void)input; (void)frequencies; (void)weights; (void)packed;
    (void)scales; (void)scaled_weights; (void)rows; (void)stream;
    return 0;
}
#ifndef DS41RT_STUB_WITHOUT_PAD_EXPORT
int32_t ds41rt_v41_compressor_pad_input(uint16_t* input, int32_t rows,
        int32_t padded_rows, void* stream) {
    pad_calls += 1;
    pad_last_input = input;
    pad_last_rows = rows;
    pad_last_padded_rows = padded_rows;
    pad_last_stream = stream;
    return 0;
}
#endif
int32_t stub_pad_input_calls(void) { return pad_calls; }
void* stub_pad_input_last_input(void) { return pad_last_input; }
int32_t stub_pad_input_last_rows(void) { return pad_last_rows; }
int32_t stub_pad_input_last_padded_rows(void) { return pad_last_padded_rows; }
void* stub_pad_input_last_stream(void) { return pad_last_stream; }
"#;

    static STUB_COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Build the CPU stub shared library. A missing/broken compiler or any I/O
    /// failure is a TEST FAILURE, never a silent skip: these tests are the only
    /// coverage of the optional pad-input symbol resolution, and a silently
    /// skipped build would report the suite green without exercising it.
    fn build_stub(without_pad_export: bool) -> Result<PathBuf> {
        let cc = Command::new("cc")
            .arg("--version")
            .output()
            .context("running the host C compiler `cc` for the pad-input stub")?;
        ensure!(
            cc.status.success(),
            "host C compiler `cc --version` failed: {}",
            String::from_utf8_lossy(&cc.stderr)
        );
        let unique = STUB_COUNTER.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "ds41rt-pad-input-stub-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory)
            .with_context(|| format!("creating stub directory {}", directory.display()))?;
        let source = directory.join("stub.c");
        std::fs::write(&source, STUB_C)
            .with_context(|| format!("writing stub source {}", source.display()))?;
        let library = directory.join(if without_pad_export {
            "libds41rt_stub_nopad.so"
        } else {
            "libds41rt_stub.so"
        });
        let mut command = Command::new("cc");
        command
            .arg("-shared")
            .arg("-fPIC")
            .arg("-o")
            .arg(&library)
            .arg(&source);
        if without_pad_export {
            command.arg("-DDS41RT_STUB_WITHOUT_PAD_EXPORT");
        }
        let output = command
            .output()
            .context("running the host C compiler for the pad-input stub")?;
        ensure!(
            output.status.success(),
            "building the pad-input stub library failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        Ok(library)
    }

    fn workspace_buffer() -> Ds41rtDeviceBuffer {
        Ds41rtDeviceBuffer {
            ptr: 0x1000_0000 as *mut c_void,
            bytes: V41Compressor::WORKSPACE_BYTES,
            device_id: 0,
            flags: 0,
        }
    }

    fn input_buffer(rows: usize) -> Ds41rtDeviceBuffer {
        Ds41rtDeviceBuffer {
            ptr: 0x2000_0000 as *mut c_void,
            bytes: rows * 10240,
            device_id: 0,
            flags: 0,
        }
    }

    #[test]
    fn old_library_without_pad_export_still_constructs_and_pad_use_fails_clearly() {
        let path = build_stub(true).expect("the host C stub must build");
        let library = unsafe { NativeLibrary::load(path) }.expect("load stub library");
        let kernel = unsafe { library.v41_compressor(workspace_buffer()) }
            .expect("old library must keep working when the policy is off");
        assert!(!kernel.pad_input_supported());
        let error = unsafe {
            kernel.pad_input(input_buffer(2), 1, 2, std::ptr::null_mut())
        }
        .expect_err("pad use on a library without the export must fail clearly");
        assert!(
            error
                .to_string()
                .contains("ds41rt_v41_compressor_pad_input"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn stub_passes_exact_pad_arguments_and_validates_shape_and_extent() {
        let path = build_stub(false).expect("the host C stub must build");
        let library = unsafe { NativeLibrary::load(path) }.expect("load stub library");
        let kernel = unsafe { library.v41_compressor(workspace_buffer()) }
            .expect("construct compressor against stub");
        assert!(kernel.pad_input_supported());
        let stream = 0x55 as *mut c_void;
        unsafe {
            kernel
                .pad_input(input_buffer(16), 3, 16, stream)
                .expect("valid pad call");
        }
        type Getter<T> = unsafe extern "C" fn() -> T;
        let calls: Getter<i32> = unsafe { *library.lib.get(b"stub_pad_input_calls").unwrap() };
        let last_input: Getter<*mut c_void> =
            unsafe { *library.lib.get(b"stub_pad_input_last_input").unwrap() };
        let last_rows: Getter<i32> = unsafe { *library.lib.get(b"stub_pad_input_last_rows").unwrap() };
        let last_padded: Getter<i32> =
            unsafe { *library.lib.get(b"stub_pad_input_last_padded_rows").unwrap() };
        let last_stream: Getter<*mut c_void> =
            unsafe { *library.lib.get(b"stub_pad_input_last_stream").unwrap() };
        assert_eq!(unsafe { calls() }, 1);
        assert_eq!(unsafe { last_input() }, input_buffer(16).ptr);
        assert_eq!(unsafe { last_rows() }, 3);
        assert_eq!(unsafe { last_padded() }, 16);
        assert_eq!(unsafe { last_stream() }, stream);
        // rows == padded_rows is a validated no-op but still reaches the export.
        unsafe {
            kernel
                .pad_input(input_buffer(2), 2, 2, stream)
                .expect("equal rows is a valid no-op");
        }
        assert_eq!(unsafe { calls() }, 2);
        assert_eq!(unsafe { last_rows() }, 2);
        // Rejected shapes/extents never reach the native call.
        for (rows, padded) in [(0, 2), (3, 2), (1, 3), (1, 15), (17, 16)] {
            unsafe {
                kernel
                    .pad_input(input_buffer(16), rows, padded, stream)
                    .expect_err("invalid pad shape must be rejected");
            }
        }
        let mut null_buffer = input_buffer(2);
        null_buffer.ptr = std::ptr::null_mut();
        unsafe {
            kernel
                .pad_input(null_buffer, 1, 2, stream)
                .expect_err("null input must be rejected");
            kernel
                .pad_input(input_buffer(1), 1, 2, stream)
                .expect_err("undersized input extent must be rejected");
        }
        assert_eq!(unsafe { calls() }, 2, "rejected calls must not reach the stub");
    }

    #[test]
    fn stub_rejects_wrong_device_misalignment_and_address_overflow_before_the_export() {
        let path = build_stub(false).expect("the host C stub must build");
        let library = unsafe { NativeLibrary::load(path) }.expect("load stub library");
        // The workspace buffer (device 0) fixes the handle's workspace device.
        let kernel = unsafe { library.v41_compressor(workspace_buffer()) }
            .expect("construct compressor against stub");
        let stream = 0x77 as *mut c_void;
        type Getter<T> = unsafe extern "C" fn() -> T;
        let calls: Getter<i32> = unsafe { *library.lib.get(b"stub_pad_input_calls").unwrap() };
        let before = unsafe { calls() };
        // A fully valid call on the workspace device still reaches the export.
        unsafe {
            kernel
                .pad_input(input_buffer(16), 2, 16, stream)
                .expect("valid pad call on the workspace device");
        }
        assert_eq!(unsafe { calls() }, before + 1);
        // Operand on a different device than the handle workspace.
        let mut foreign = input_buffer(16);
        foreign.device_id = workspace_buffer().device_id + 1;
        let error = unsafe { kernel.pad_input(foreign, 2, 16, stream) }
            .expect_err("pad input on a foreign device must be rejected");
        assert!(
            error.to_string().contains("differs"),
            "unexpected error: {error:#}"
        );
        // Misaligned (odd) input pointer.
        let mut misaligned = input_buffer(16);
        misaligned.ptr = (misaligned.ptr as usize + 1) as *mut c_void;
        let error = unsafe { kernel.pad_input(misaligned, 2, 16, stream) }
            .expect_err("misaligned pad input must be rejected");
        assert!(
            error.to_string().contains("aligned"),
            "unexpected error: {error:#}"
        );
        // Even, aligned pointer whose padded extent wraps the address space for
        // both supported pad sizes.
        for (padded_rows, back) in [(16usize, 163839usize), (2, 20479)] {
            let mut overflow = input_buffer(padded_rows);
            overflow.ptr = usize::MAX.wrapping_sub(back) as *mut c_void;
            assert_eq!(overflow.ptr as usize % 2, 0, "fixture must stay aligned");
            assert_eq!(overflow.bytes, padded_rows * 10240);
            let error = unsafe { kernel.pad_input(overflow, 1, padded_rows, stream) }
                .expect_err("pad input extent overflow must be rejected");
            assert!(
                error.to_string().contains("overflows"),
                "unexpected error: {error:#}"
            );
        }
        assert_eq!(
            unsafe { calls() },
            before + 1,
            "rejected calls must never reach the native export"
        );
    }
}
