//! Shared test support for the integration suites: a deterministic RNG so a failing schedule
//! reproduces from its seed. The independent shadow model of the stub copy engine lives in
//! `ds41rt_hostcache::copy::testing`, where the unit tests can reach it too.

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
