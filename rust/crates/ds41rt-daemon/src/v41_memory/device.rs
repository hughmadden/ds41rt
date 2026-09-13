//! Explicit device owners for the dual-RTX path. Legacy owners stay unchanged.
use super::*;
use anyhow::ensure;

#[derive(Clone, Copy)]
pub(crate) struct Device<'a> {
    pub library: &'a NativeLibrary,
    pub id: i32,
}

impl Device<'_> {
    /// Only synchronous enqueue/query work belongs inside this closure. The
    /// previous device is restored before the caller can yield its async task.
    pub fn run<T>(&self, work: impl FnOnce() -> Result<T>) -> Result<T> {
        struct Restore<'a> { library: &'a NativeLibrary, previous: i32, armed: bool }
        impl Drop for Restore<'_> {
            fn drop(&mut self) {
                if self.armed {
                    if let Err(error) = self.library.cuda_set_device(self.previous) {
                        tracing::error!(%error, "restoring CUDA device during unwind");
                    }
                }
            }
        }
        let previous = self.library.cuda_get_device()?;
        if previous == self.id { return work(); }
        self.library.cuda_set_device(self.id)?;
        let mut restore = Restore { library: self.library, previous, armed: true };
        let result = work();
        self.library.cuda_set_device(previous)?;
        restore.armed = false;
        result
    }
}

pub(crate) struct Allocation<'a> {
    pub device: Device<'a>,
    pub buffer: Ds41rtDeviceBuffer,
}
impl<'a> Allocation<'a> {
    pub fn new(device: Device<'a>, bytes: usize) -> Result<Self> {
        let buffer = device.run(|| device.library.alloc_device_buffer(bytes))?;
        Ok(Self { device, buffer })
    }
}
impl Drop for Allocation<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.device.run(|| self.device.library.free_device_buffer(&mut self.buffer)) {
            tracing::error!(%error, "freeing device-owned allocation");
        }
    }
}

pub(crate) struct Stream<'a> {
    pub device: Device<'a>,
    pub raw: *mut c_void,
}
impl<'a> Stream<'a> {
    pub fn new(device: Device<'a>) -> Result<Self> {
        Ok(Self { device, raw: device.run(|| device.library.cuda_stream_create())? })
    }
    fn ready(&self) -> Result<bool> {
        self.device.run(|| unsafe { self.device.library.cuda_stream_query(self.raw) })
    }
    fn drain(&self) -> Result<()> {
        self.device.run(|| unsafe { self.device.library.cuda_stream_synchronize(self.raw) })
    }
}
impl Drop for Stream<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.drain() {
            tracing::error!(%error, "draining device-owned stream");
        }
        if let Err(error) = self.device.run(|| unsafe { self.device.library.cuda_stream_destroy(self.raw) }) {
            tracing::error!(%error, "destroying device-owned stream");
        }
    }
}

pub(crate) struct Event<'a> { pub device: Device<'a>, pub raw: *mut c_void }
impl<'a> Event<'a> {
    pub fn new(device: Device<'a>) -> Result<Self> {
        Ok(Self { device, raw: device.run(|| device.library.cuda_event_create())? })
    }
    pub fn record(&mut self, producer: &Stream<'a>) -> Result<()> {
        ensure!(self.device.id == producer.device.id
            && std::ptr::eq(self.device.library, producer.device.library), "event producer device mismatch");
        self.device.run(|| unsafe { self.device.library.cuda_event_record(self.raw, producer.raw) })
    }
}
impl Drop for Event<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.device.run(|| unsafe { self.device.library.cuda_event_destroy(self.raw) }) {
            tracing::error!(%error, "destroying device-owned event");
        }
    }
}

/// One transfer direction for one lane; stream and event are allocated at startup.
pub(crate) struct PeerTransfer<'a> { destination: Stream<'a>, ready: Event<'a> }
impl<'a> PeerTransfer<'a> {
    pub fn new(source: Device<'a>, destination: Device<'a>) -> Result<Self> {
        ensure!(std::ptr::eq(source.library, destination.library) && source.id != destination.id,
            "peer transfer requires distinct devices from one library");
        destination.run(|| destination.library.cuda_enable_peer(source.id))?;
        let stream = Stream::new(destination)?;
        let ready = Event { device: source,
            raw: source.run(|| source.library.cuda_event_create())? };
        Ok(Self { destination: stream, ready })
    }

    /// # Safety
    /// All source writes must be ordered on `producer`; no aliases may access
    /// either allocation incompatibly while this future is live. Allocation and
    /// producer borrows survive until completion or cancellation drains the copy.
    pub async unsafe fn copy(&mut self, source: &Allocation<'a>, destination: &mut Allocation<'a>,
        producer: &Stream<'a>, bytes: usize) -> Result<()> {
        unsafe { self.copy_then(source, destination, producer, bytes, |_, _| Ok(())).await }
    }

    /// # Safety
    /// Same buffer contract as `copy`. `then` enqueues only on the supplied
    /// destination stream; its captured storage must survive completion. The
    /// callback itself is retained until the completion/cancellation drain.
    pub async unsafe fn copy_then(&mut self, source: &Allocation<'a>, destination: &mut Allocation<'a>,
        producer: &Stream<'a>, bytes: usize,
        mut then: impl FnMut(Ds41rtDeviceBuffer, *mut c_void) -> Result<()>) -> Result<()> {
        let library = self.destination.device.library;
        ensure!(source.device.id == self.ready.device.id
            && producer.device.id == source.device.id
            && destination.device.id == self.destination.device.id
            && std::ptr::eq(library, source.device.library)
            && std::ptr::eq(library, destination.device.library)
            && std::ptr::eq(library, producer.device.library)
            && bytes <= source.buffer.bytes && bytes <= destination.buffer.bytes,
            "peer transfer owner or extent mismatch");
        struct Drain<'s, 'a> { stream: &'s Stream<'a>, complete: bool }
        impl Drop for Drain<'_, '_> {
            fn drop(&mut self) {
                if !self.complete {
                    if let Err(error) = self.stream.drain() {
                        tracing::error!(%error, "draining cancelled peer transfer");
                    }
                }
            }
        }
        let mut drain = Drain { stream: &self.destination, complete: false };
        self.ready.device.run(|| unsafe { library.cuda_event_record(self.ready.raw, producer.raw) })?;
        self.destination.device.run(|| unsafe {
            library.cuda_stream_wait_event(self.destination.raw, self.ready.raw)?;
            library.copy_peer_async(destination.buffer, source.buffer, bytes, self.destination.raw)?;
            then(destination.buffer, self.destination.raw)
        })?;
        while !self.destination.ready()? { tokio::task::yield_now().await; }
        drain.complete = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB and two CUDA devices"]
    fn peer_owners_restore_device_and_copy_independent_lanes() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        lib.cuda_set_device(0)?;
        let a = Device { library: &lib, id: 0 };
        let b = Device { library: &lib, id: 1 };
        let failed: Result<()> = b.run(|| anyhow::bail!("deliberate error"));
        assert!(failed.is_err());
        assert_eq!(lib.cuda_get_device()?, 0);
        const BYTES: usize = 1024 * 1024;
        let source_a = Allocation::new(a, BYTES)?;
        let source_b = Allocation::new(b, BYTES)?;
        let mut target_a = Allocation::new(a, BYTES)?;
        let mut target_b = Allocation::new(b, BYTES)?;
        let producer_a = Stream::new(a)?;
        let producer_b = Stream::new(b)?;
        let mut ab = PeerTransfer::new(a, b)?;
        let mut ba = PeerTransfer::new(b, a)?;
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        for seed in [13u8, 127, 241] {
            let input_a: Vec<u8> = (0..BYTES).map(|i| (i as u8).wrapping_add(seed)).collect();
            let input_b: Vec<u8> = input_a.iter().map(|v| !v).collect();
            a.run(|| lib.copy_h2d(source_a.buffer, &input_a))?;
            b.run(|| lib.copy_h2d(source_b.buffer, &input_b))?;
            runtime.block_on(async {
                let (x, y) = tokio::join!(
                    unsafe { ab.copy(&source_a, &mut target_b, &producer_a, BYTES) },
                    unsafe { ba.copy(&source_b, &mut target_a, &producer_b, BYTES) });
                x?; y?;
                Ok::<_, anyhow::Error>(())
            })?;
            assert_eq!(lib.cuda_get_device()?, 0);
            let mut output = vec![0u8; BYTES];
            a.run(|| lib.copy_d2h(&mut output, target_a.buffer))?;
            assert_eq!(output, input_b);
            b.run(|| lib.copy_d2h(&mut output, target_b.buffer))?;
            assert_eq!(output, input_a);
        }
        drop((ab, ba, producer_a, producer_b, source_a, source_b, target_a, target_b));
        assert_eq!(lib.cuda_get_device()?, 0);
        Ok(())
    }
}
