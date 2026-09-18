//! C ABI over the unmodified native DS41RT TP4 clients. See include/afd_bridge.h.
mod engine;
use engine::{Bridge, Config};
use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    ptr, slice,
};

pub const OK: i32 = 0;
pub const INVALID: i32 = 1;
pub const BUSY: i32 = 2;
pub const STALE: i32 = 3;
pub const NOT_READY: i32 = 4;
pub const BUFFER_TOO_SMALL: i32 = 5;
pub const FAILED: i32 = 6;
pub const CANCELLED: i32 = 7;
pub const INTERNAL: i32 = 8;

#[allow(non_camel_case_types)]
pub struct afd_bridge_handle {
    bridge: Bridge,
}
type ApiResult = Result<(), (i32, String)>;

fn guarded(action: impl FnOnce() -> i32) -> i32 {
    catch_unwind(AssertUnwindSafe(action)).unwrap_or(INTERNAL)
}
unsafe fn handle<'a>(raw: *mut afd_bridge_handle) -> Result<&'a Bridge, (i32, String)> {
    unsafe { raw.as_ref() }
        .map(|h| &h.bridge)
        .ok_or((INVALID, "null bridge handle".into()))
}
unsafe fn input<'a>(raw: *const u8, len: usize) -> Result<&'a [u8], (i32, String)> {
    if raw.is_null() || len > isize::MAX as usize {
        return Err((INVALID, "null or oversized input".into()));
    }
    Ok(unsafe { slice::from_raw_parts(raw, len) })
}
fn finish(bridge: &Bridge, result: ApiResult) -> i32 {
    match result {
        Ok(()) => OK,
        Err((status, error)) => {
            if let Ok(mut last) = bridge.last_error.lock() {
                *last = error;
            }
            status
        }
    }
}
fn with_slot<T>(
    bridge: &Bridge,
    lane: u32,
    ticket: u64,
    action: impl FnOnce(&mut engine::Slot) -> Result<T, (i32, String)>,
) -> Result<T, (i32, String)> {
    let owner = bridge
        .slots
        .get(lane as usize)
        .ok_or((INVALID, "lane out of range".into()))?;
    let mut slot = owner
        .lock()
        .map_err(|_| (INTERNAL, "lane mutex poisoned".into()))?;
    if ticket == 0 || slot.ticket != ticket {
        return Err((STALE, "ticket is not current".into()));
    }
    action(&mut slot)
}
unsafe fn copy_output(
    bytes: &[u8],
    output: *mut u8,
    capacity: usize,
    written: *mut usize,
) -> ApiResult {
    if written.is_null() {
        return Err((INVALID, "null size output".into()));
    }
    unsafe {
        *written = bytes.len();
    }
    if capacity < bytes.len() {
        return Err((BUFFER_TOO_SMALL, "output buffer is too small".into()));
    }
    if !bytes.is_empty() {
        if output.is_null() {
            return Err((INVALID, "null data output".into()));
        }
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), output, bytes.len());
        }
    }
    Ok(())
}

#[no_mangle]
pub extern "C" fn afd_bridge_abi_version() -> u32 {
    1
}

/// # Safety
/// Pointers obey the ownership and extent contract in include/afd_bridge.h.
#[no_mangle]
pub unsafe extern "C" fn afd_bridge_create(
    json: *const u8,
    len: usize,
    out: *mut *mut afd_bridge_handle,
) -> i32 {
    guarded(|| {
        if out.is_null() {
            return INVALID;
        }
        unsafe {
            *out = ptr::null_mut();
        }
        let bytes = match unsafe { input(json, len) } {
            Ok(b) => b,
            Err((s, _)) => return s,
        };
        let config: Config = match serde_json::from_slice(bytes) {
            Ok(c) => c,
            Err(_) => return INVALID,
        };
        let bridge = match Bridge::new(config) {
            Ok(b) => b,
            Err(_) => return INVALID,
        };
        unsafe {
            *out = Box::into_raw(Box::new(afd_bridge_handle { bridge }));
        }
        OK
    })
}

/// # Safety
/// Pointers obey include/afd_bridge.h.
#[no_mangle]
pub unsafe extern "C" fn afd_bridge_submit(
    raw: *mut afd_bridge_handle,
    lane: u32,
    frame: *const u8,
    len: usize,
    ticket: *mut u64,
) -> i32 {
    guarded(|| {
        let b = match unsafe { handle(raw) } {
            Ok(b) => b,
            Err((s, _)) => return s,
        };
        finish(
            b,
            (|| {
                if ticket.is_null() {
                    return Err((INVALID, "null ticket output".into()));
                }
                let frame = unsafe { input(frame, len)? };
                let generation = b.submit(lane as usize, frame)?;
                unsafe {
                    *ticket = generation;
                }
                Ok(())
            })(),
        )
    })
}

/// # Safety
/// Pointers obey include/afd_bridge.h.
#[no_mangle]
pub unsafe extern "C" fn afd_bridge_poll(
    raw: *mut afd_bridge_handle,
    lane: u32,
    ticket: u64,
    state: *mut u32,
) -> i32 {
    guarded(|| {
        let b = match unsafe { handle(raw) } {
            Ok(b) => b,
            Err((s, _)) => return s,
        };
        finish(
            b,
            with_slot(b, lane, ticket, |slot| {
                if state.is_null() {
                    return Err((INVALID, "null state output".into()));
                }
                unsafe {
                    *state = slot.state;
                }
                Ok(())
            }),
        )
    })
}

/// # Safety
/// Pointers obey include/afd_bridge.h.
#[no_mangle]
pub unsafe extern "C" fn afd_bridge_collect(
    raw: *mut afd_bridge_handle,
    lane: u32,
    ticket: u64,
    output: *mut u8,
    capacity: usize,
    written: *mut usize,
) -> i32 {
    guarded(|| {
        let b = match unsafe { handle(raw) } {
            Ok(b) => b,
            Err((s, _)) => return s,
        };
        finish(
            b,
            with_slot(b, lane, ticket, |slot| {
                match slot.state {
                    engine::READY => {}
                    engine::FAILED => return Err((FAILED, slot.error.clone())),
                    engine::CANCELLED => return Err((CANCELLED, slot.error.clone())),
                    _ => return Err((NOT_READY, "result is not ready".into())),
                }
                unsafe {
                    copy_output(&slot.output, output, capacity, written)?;
                }
                slot.state = engine::IDLE;
                Ok(())
            }),
        )
    })
}

/// # Safety
/// Pointers obey include/afd_bridge.h.
#[no_mangle]
pub unsafe extern "C" fn afd_bridge_cancel(
    raw: *mut afd_bridge_handle,
    lane: u32,
    ticket: u64,
) -> i32 {
    guarded(|| {
        let b = match unsafe { handle(raw) } {
            Ok(b) => b,
            Err((s, _)) => return s,
        };
        finish(b, b.cancel(lane as usize, ticket))
    })
}

/// # Safety
/// Pointers obey include/afd_bridge.h.
#[no_mangle]
pub unsafe extern "C" fn afd_bridge_error(
    raw: *mut afd_bridge_handle,
    lane: u32,
    ticket: u64,
    output: *mut u8,
    capacity: usize,
    written: *mut usize,
) -> i32 {
    guarded(|| {
        let b = match unsafe { handle(raw) } {
            Ok(b) => b,
            Err((s, _)) => return s,
        };
        let result = if ticket == 0 {
            match b.last_error.lock() {
                Ok(error) => unsafe { copy_output(error.as_bytes(), output, capacity, written) },
                Err(_) => Err((INTERNAL, "diagnostic mutex poisoned".into())),
            }
        } else {
            with_slot(b, lane, ticket, |slot| unsafe {
                copy_output(slot.error.as_bytes(), output, capacity, written)
            })
        };
        // A size query must not overwrite the diagnostic it is querying.
        result.map_or_else(|(status, _)| status, |_| OK)
    })
}

/// # Safety
/// Pointers obey include/afd_bridge.h.
#[no_mangle]
pub unsafe extern "C" fn afd_bridge_buffer_stats(
    raw: *mut afd_bridge_handle,
    lane: u32,
    capacity: *mut usize,
    growths: *mut u64,
) -> i32 {
    guarded(|| {
        let b = match unsafe { handle(raw) } {
            Ok(b) => b,
            Err((s, _)) => return s,
        };
        finish(
            b,
            (|| {
                if capacity.is_null() || growths.is_null() {
                    return Err((INVALID, "null buffer statistics output".into()));
                }
                let owner = b
                    .slots
                    .get(lane as usize)
                    .ok_or((INVALID, "lane out of range".into()))?;
                let slot = owner
                    .lock()
                    .map_err(|_| (INTERNAL, "lane mutex poisoned".into()))?;
                unsafe {
                    *capacity = slot.buffer_capacity;
                    *growths = slot.buffer_growths;
                }
                Ok(())
            })(),
        )
    })
}

/// # Safety
/// The caller owns the handle exclusively and will never use it again.
#[no_mangle]
pub unsafe extern "C" fn afd_bridge_close(raw: *mut afd_bridge_handle) -> i32 {
    guarded(|| {
        if raw.is_null() {
            return INVALID;
        }
        unsafe {
            drop(Box::from_raw(raw));
        }
        OK
    })
}
