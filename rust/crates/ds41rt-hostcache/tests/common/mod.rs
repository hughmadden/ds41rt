//! Shared test support: a deterministic RNG and an independent shadow model of the stub copy
//! engine's documented behaviour, so the functional and concurrency suites can compare the
//! engine's observable state against a second implementation of the same contract.
#![allow(dead_code)]

use ds41rt_hostcache::copy::{CopyModel, Stream};

/// xorshift64*: deterministic, so a failing schedule reproduces from its seed.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// The slot a stream occupies in per-stream arrays.
pub fn stream_index(stream: Stream) -> usize {
    match stream {
        Stream::Store => 0,
        Stream::Restore => 1,
    }
}

/// Nanoseconds to move `bytes` at `bytes_per_ns`, rounded up.
pub fn transfer_ns(bytes: usize, bytes_per_ns: f64) -> u64 {
    (bytes as f64 / bytes_per_ns).ceil() as u64
}

/// A copy the shadow model has queued.
struct ShadowCopy {
    stream: Stream,
    d2h: bool,
    src: usize,
    dst: usize,
    bytes: usize,
    completion: u64,
    seq: u64,
}

/// An independent model of the stub: the same completion formula, the same execution-time byte
/// movement, and the same global completion order. It exists to disagree with the engine when
/// the engine is wrong.
pub struct Shadow {
    pub device: Vec<u8>,
    pub host: Vec<u8>,
    pub now: u64,
    pub events: Vec<Option<u64>>,
    last: [u64; 2],
    pending: Vec<ShadowCopy>,
    seq: u64,
    model: CopyModel,
}

impl Shadow {
    pub fn new(model: CopyModel, device_bytes: usize, host_bytes: usize) -> Self {
        Self {
            device: vec![0; device_bytes],
            host: vec![0; host_bytes],
            now: 0,
            events: Vec::new(),
            last: [0; 2],
            pending: Vec::new(),
            seq: 0,
            model,
        }
    }

    pub fn write_device(&mut self, addr: usize, bytes: &[u8]) {
        self.device[addr..addr + bytes.len()].copy_from_slice(bytes);
    }

    pub fn issue(&mut self, stream: Stream, d2h: bool, src: usize, dst: usize, bytes: usize) {
        let index = stream_index(stream);
        let rate = if d2h {
            self.model.d2h_bytes_per_ns
        } else {
            self.model.h2d_bytes_per_ns
        };
        let start = self.last[index].max(self.now);
        let completion = start
            .saturating_add(self.model.per_copy_latency_ns)
            .saturating_add(transfer_ns(bytes, rate));
        self.last[index] = completion;
        self.seq += 1;
        self.pending.push(ShadowCopy {
            stream,
            d2h,
            src,
            dst,
            bytes,
            completion,
            seq: self.seq,
        });
    }

    pub fn record(&mut self, stream: Stream) -> usize {
        self.events.push(Some(self.last[stream_index(stream)]));
        self.events.len() - 1
    }

    pub fn advance(&mut self, nanos: u64) {
        self.now = self.now.saturating_add(nanos);
        self.execute_due();
    }

    pub fn wait(&mut self, event: usize, budget: u64) -> bool {
        let completion = self.events[event];
        let target = match completion {
            Some(completion) => completion.min(self.now.saturating_add(budget)),
            None => self.now.saturating_add(budget),
        };
        if target > self.now {
            self.now = target;
            self.execute_due();
        }
        matches!(completion, Some(completion) if self.now >= completion)
    }

    pub fn pending(&self, stream: Stream) -> usize {
        self.pending
            .iter()
            .filter(|copy| copy.stream == stream)
            .count()
    }

    fn execute_due(&mut self) {
        self.pending.sort_by_key(|copy| (copy.completion, copy.seq));
        let mut i = 0;
        while i < self.pending.len() {
            if self.pending[i].completion <= self.now {
                let copy = self.pending.remove(i);
                if copy.d2h {
                    self.host[copy.dst..copy.dst + copy.bytes]
                        .copy_from_slice(&self.device[copy.src..copy.src + copy.bytes]);
                } else {
                    self.device[copy.dst..copy.dst + copy.bytes]
                        .copy_from_slice(&self.host[copy.src..copy.src + copy.bytes]);
                }
            } else {
                i += 1;
            }
        }
    }
}
