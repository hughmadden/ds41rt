//! Optional, reusable GPU events for the native expert boundary.
use anyhow::Result;
use ds41rt_ffi::NativeLibrary;
use std::ffi::c_void;

struct Event<'a> {
    library: &'a NativeLibrary,
    raw: *mut c_void,
}
impl<'a> Event<'a> {
    fn new(library: &'a NativeLibrary) -> Result<Self> {
        Ok(Self {
            library,
            raw: library.cuda_event_create()?,
        })
    }
}
impl Drop for Event<'_> {
    fn drop(&mut self) {
        if let Err(error) = unsafe { self.library.cuda_event_destroy(self.raw) } {
            tracing::error!(%error, "destroying expert timing event");
        }
    }
}

/// The wave owns these events and drains its stream before dropping them.
pub(super) struct ExpertTiming<'a> {
    events: [Event<'a>; 3],
}
impl<'a> ExpertTiming<'a> {
    pub fn new(library: &'a NativeLibrary) -> Result<Self> {
        Ok(Self {
            events: [
                Event::new(library)?,
                Event::new(library)?,
                Event::new(library)?,
            ],
        })
    }
    pub unsafe fn record(&self, index: usize, stream: *mut c_void) -> Result<()> {
        let event = &self.events[index];
        unsafe { event.library.cuda_event_record(event.raw, stream) }
    }
    /// Call only after the owning stream's existing completion barrier.
    pub unsafe fn elapsed_us(&self) -> Result<(f32, f32)> {
        let [start, kernel_end, compact_end] = &self.events;
        Ok(unsafe {
            (
                1000.0
                    * start
                        .library
                        .cuda_event_elapsed_ms(start.raw, kernel_end.raw)?,
                1000.0
                    * start
                        .library
                        .cuda_event_elapsed_ms(kernel_end.raw, compact_end.raw)?,
            )
        })
    }
}
