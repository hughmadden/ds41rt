//! Explicit device owners for the dual-RTX path. Legacy owners stay unchanged.
use super::*;
use anyhow::ensure;
use std::{future::Future, mem::ManuallyDrop, pin::Pin, task::{Context as TaskContext, Poll}};

#[derive(Clone, Copy)]
pub(crate) struct Device<'a> {
    pub library: &'a NativeLibrary,
    pub id: i32,
}

#[cfg(test)]
pub(crate) fn cache_test_device(library: &NativeLibrary) -> Result<Device<'_>> {
    let id: i32 = std::env::var("DS41RT_CACHE_TEST_DEVICE").unwrap_or_else(|_| "0".into()).parse()?;
    ensure!(matches!(id,0 | 1), "invalid cache fixture GPU");
    library.cuda_set_device(0)?;
    if id == 1 {
        for gpu in 0..2 { Device { library, id: gpu }.run(|| library.cuda_enable_peer(1-gpu))?; }
    }
    Ok(Device { library,id })
}

impl<'a> Device<'a> {
    /// Scope a synchronous method body. Never retain this guard across a yield.
    pub fn enter(&self) -> Result<DeviceScope<'a>> {
        let previous = self.library.cuda_get_device()?;
        if previous != self.id { self.library.cuda_set_device(self.id)?; }
        Ok(DeviceScope { library: self.library, previous, changed: previous != self.id })
    }
    /// Scope every poll and cancellation cleanup, never an entire async wait.
    pub fn future<F: Future>(&self, future: F) -> DeviceFuture<'a, F> {
        DeviceFuture { device: *self, future: ManuallyDrop::new(future) }
    }

    /// Legacy components may own streams without storing their device ordinal.
    /// Construct and destroy the entire component on its selected device.
    pub fn own<T>(&self, initialize: impl FnOnce() -> Result<T>) -> Result<DeviceOwner<'a, T>> {
        Ok(DeviceOwner { device: *self, value: ManuallyDrop::new(self.run(initialize)?) })
    }
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

pub(crate) struct DeviceScope<'a> { library: &'a NativeLibrary, previous: i32, changed: bool }
impl Drop for DeviceScope<'_> {
    fn drop(&mut self) {
        if self.changed {
            if let Err(error) = self.library.cuda_set_device(self.previous) {
                tracing::error!(%error, "restoring synchronous CUDA device scope");
            }
        }
    }
}

pub(crate) struct DeviceOwner<'a, T> { pub device: Device<'a>, value: ManuallyDrop<T> }
impl<T> DeviceOwner<'_, T> {
    /// Metadata access is ordinary; GPU work must use this owner's device scope.
    pub fn get(&self) -> &T { &self.value }
    pub fn get_mut(&mut self) -> &mut T { &mut self.value }
}
impl<T> std::ops::Deref for DeviceOwner<'_, T> {
    type Target = T;
    fn deref(&self) -> &T { self.get() }
}
impl<T> std::ops::DerefMut for DeviceOwner<'_, T> {
    fn deref_mut(&mut self) -> &mut T { self.get_mut() }
}
impl<T> Drop for DeviceOwner<'_, T> {
    fn drop(&mut self) {
        if let Err(error) = self.device.run(|| {
            unsafe { ManuallyDrop::drop(&mut self.value); }
            Ok(())
        }) { tracing::error!(%error, "dropping device-owned component"); }
    }
}

pub(crate) struct DeviceFuture<'a, F: Future> { device: Device<'a>, future: ManuallyDrop<F> }
impl<T, F: Future<Output = Result<T>>> Future for DeviceFuture<'_, F> {
    type Output = Result<T>;
    fn poll(self: Pin<&mut Self>, context: &mut TaskContext<'_>) -> Poll<Self::Output> {
        // The field is never moved after pinning; Drop destroys it in place.
        let this = unsafe { self.get_unchecked_mut() };
        match this.device.run(|| Ok(unsafe { Pin::new_unchecked(&mut *this.future) }.poll(context))) {
            Ok(poll) => poll,
            Err(error) => Poll::Ready(Err(error)),
        }
    }
}
impl<F: Future> Drop for DeviceFuture<'_, F> {
    fn drop(&mut self) {
        if let Err(error) = self.device.run(|| {
            unsafe { ManuallyDrop::drop(&mut self.future); }
            Ok(())
        }) { tracing::error!(%error, "dropping device-scoped future"); }
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
    pub(crate) fn drain(&self) -> Result<()> {
        self.device.run(|| unsafe { self.device.library.cuda_stream_synchronize(self.raw) })
    }
    /// Retain queued work through cooperative completion or cancellation drain.
    pub(crate) async fn wait(&self) -> Result<()> {
        struct Drain<'s, 'a> { stream: &'s Stream<'a>, complete: bool }
        impl Drop for Drain<'_, '_> {
            fn drop(&mut self) {
                if !self.complete {
                    if let Err(error) = self.stream.drain() {
                        tracing::error!(%error, "draining interrupted device stream");
                    }
                }
            }
        }
        let mut guard = Drain { stream: self, complete: false };
        while !self.ready()? { tokio::task::yield_now().await; }
        guard.complete = true;
        Ok(())
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
    #[ignore = "requires DS41RT_NATIVE_LIB and two CUDA GPUs"]
    fn device_scoped_futures_restore_poll_and_cancellation_context() -> Result<()> {
        use std::{cell::Cell, rc::Rc, task::Waker};
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        lib.cuda_set_device(0)?;
        let d1 = Device { library: &lib, id: 1 };
        let dropped_on = Rc::new(Cell::new(-1));
        struct Probe<'a> { library: &'a NativeLibrary, dropped_on: Rc<Cell<i32>> }
        impl Drop for Probe<'_> {
            fn drop(&mut self) { self.dropped_on.set(self.library.cuda_get_device().unwrap()); }
        }
        let mut owner = d1.own(|| {
            assert_eq!(lib.cuda_get_device()?,1);
            Ok(Probe { library: &lib, dropped_on: dropped_on.clone() })
        })?;
        assert_eq!(lib.cuda_get_device()?,0);
        let mut context = TaskContext::from_waker(Waker::noop());
        let future = d1.future(async {
            assert_eq!(lib.cuda_get_device()?,1);
            let _guard = Probe { library: &lib, dropped_on: dropped_on.clone() };
            let _stream = LoadStream { library: &lib, raw: lib.cuda_stream_create()? };
            tokio::task::yield_now().await;
            assert_eq!(lib.cuda_get_device()?,1);
            Ok(())
        });
        let mut future = Box::pin(future);
        assert!(future.as_mut().poll(&mut context).is_pending());
        assert_eq!(lib.cuda_get_device()?,0);
        drop(future);
        assert_eq!(dropped_on.get(),1);
        assert_eq!(lib.cuda_get_device()?,0);
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        let d0 = Device { library: &lib, id: 0 };
        runtime.block_on(async {
            tokio::try_join!(d0.future(async {
                for _ in 0..8 { assert_eq!(lib.cuda_get_device()?,0); tokio::task::yield_now().await; }
                Ok(())
            }), d1.future(async {
                for _ in 0..8 { assert_eq!(lib.cuda_get_device()?,1); tokio::task::yield_now().await; }
                Ok(())
            }))?;
            Ok::<_,anyhow::Error>(())
        })?;
        owner.get_mut().dropped_on.set(-1);
        drop(owner);
        assert_eq!(dropped_on.get(),1);
        assert_eq!(lib.cuda_get_device()?,0);
        Ok(())
    }
    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB, libcudart.so.13 and two CUDA devices"]
    fn cancelled_peer_chain_drains_before_releasing_borrows() -> Result<()> {
        use std::future::Future;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::task::{Context, Poll, Waker};
        struct RuntimeLibrary(*mut c_void);
        impl Drop for RuntimeLibrary {
            fn drop(&mut self) { unsafe { libc::dlclose(self.0); } }
        }
        unsafe extern "C" fn hold(data: *mut c_void) {
            let release = unsafe { &*data.cast::<AtomicBool>() };
            while !release.load(Ordering::Acquire) { std::thread::yield_now(); }
        }
        let cuda = RuntimeLibrary(unsafe { libc::dlopen(c"libcudart.so.13".as_ptr(), libc::RTLD_NOW) });
        ensure!(!cuda.0.is_null(), "CUDA runtime library unavailable");
        let symbol = unsafe { libc::dlsym(cuda.0, c"cudaLaunchHostFunc".as_ptr()) };
        ensure!(!symbol.is_null(), "CUDA callback API unavailable");
        let launch: unsafe extern "C" fn(*mut c_void, unsafe extern "C" fn(*mut c_void), *mut c_void) -> i32 =
            unsafe { std::mem::transmute(symbol) };
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        lib.cuda_set_device(0)?;
        let d0 = Device { library: &lib, id: 0 };
        let d1 = Device { library: &lib, id: 1 };
        let source = Allocation::new(d0, 4096)?;
        let mut destination = Allocation::new(d1, 4096)?;
        let output = Allocation::new(d1, 4096)?;
        let producer = Stream::new(d0)?;
        let independent = Stream::new(d1)?;
        let mut transfer = PeerTransfer::new(d0, d1)?;
        lib.copy_h2d(source.buffer, &[73u8; 4096])?;
        let release = AtomicBool::new(false);
        // Always release during unwinding, before any stream owner drains.
        struct Release<'a>(&'a AtomicBool);
        impl Drop for Release<'_> { fn drop(&mut self) { self.0.store(true, Ordering::Release); } }
        let _release = Release(&release);
        ensure!(unsafe { launch(producer.raw, hold, (&release as *const AtomicBool).cast_mut().cast()) } == 0,
            "could not stall producer");
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        let _entered = runtime.enter();
        let mut future = Box::pin(unsafe { transfer.copy_then(&source, &mut destination, &producer, 4096,
            |copied, stream| lib.copy_d2d_async(output.buffer, copied, 4096, stream)) });
        // On assertion failure release before the future's cancellation guard.
        let _unwind_release = Release(&release);
        let mut context = Context::from_waker(Waker::noop());
        assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
        assert_eq!(lib.cuda_get_device()?, 0);
        assert!(independent.ready()?);
        assert!(!release.load(Ordering::Acquire));
        std::thread::scope(|scope| {
            scope.spawn(|| {
                std::thread::sleep(std::time::Duration::from_millis(50));
                release.store(true, Ordering::Release);
            });
            drop(future);
            // Without cancellation draining, Drop returns while the gate is held.
            assert!(release.load(Ordering::Acquire));
        });
        assert!(producer.ready()?);
        assert_eq!(lib.cuda_get_device()?, 0);
        let mut bytes = [0u8; 4096];
        d1.run(|| lib.copy_d2h(&mut bytes, output.buffer))?;
        assert_eq!(bytes, [73u8; 4096]);
        Ok(())
    }

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
