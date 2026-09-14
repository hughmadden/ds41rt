//! Deterministic interleaver for concurrency suites.
//!
//! Cooperative tasks yield at every shared-state boundary; the interleaver picks the next task from
//! a seeded PRNG so a run is reproducible from its seed. A failing property prints the seed and the
//! schedule. Sixteen lanes plus two retention lanes plus the sink and restore workers are the
//! production population.
pub trait Task {
    /// Run until the next yield point. Returns `false` when finished.
    fn step(&mut self) -> bool;
    fn name(&self) -> &str;
}
