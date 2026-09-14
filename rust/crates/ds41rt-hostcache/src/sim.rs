//! The simulator (packet HC-4): a model of the engine around the cache, so the suites can drive
//! `HostCache` exactly as the daemon will, on a virtual clock, deterministically.
//!
//! - **Device model**: a page pool per compressor with generations (a freed index becomes a new
//!   identity), copy-on-write prefix sharing, `SourcePoolExhausted` when full, two retention
//!   banks using `ds41rt_core::prefix::Retention` with the engine's bank sizes, and the
//!   `make_room` eviction the engine performs under exhaustion.
//! - **Lanes**: requests advance through admission → lookup (device bank, then cache) → restore
//!   or prefill (tokens / prefill rate) → decode (new tokens / decode rate) → retire (turn
//!   retained); prefill retains a prompt snapshot. Timing is the modelled cost of each step.
//! - **Interleaver**: a seeded, logged order of lane steps and copy completions so a failure
//!   reproduces from its seed.
//! - **Workloads**: `AgentLoop` (sessions × turns with context growth and think time), `Burst`
//!   (many cold prompts), `Churn` (more live sessions than the device holds, the pressure case).
//!
//! The simulator asserts the crate invariants after every step: pool bytes within quota, pinned
//! snapshots never evicted, stored bytes equal restored bytes (content fidelity through the stub
//! engine), and every ticket reported once.
use crate::cache::HostCache;
use crate::copy::StubCopyEngine;
use serde::Serialize;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EngineModel {
    pub device_pool_tokens: u64,
    pub retain_prompts: usize,
    pub retain_turns: usize,
    pub prefill_tokens_per_s: f64,
    pub decode_tokens_per_s: f64,
    pub lanes: usize,
}

impl Default for EngineModel {
    /// The capacity-1024 configuration of record.
    fn default() -> Self {
        Self {
            device_pool_tokens: 2_500_000,
            retain_prompts: 24,
            retain_turns: 24,
            prefill_tokens_per_s: 5_377.0,
            decode_tokens_per_s: 74.1,
            lanes: 8,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum Workload {
    AgentLoop {
        sessions: usize,
        turns: usize,
        context_tokens: u32,
        new_tokens_per_turn: u32,
        think_ns: u64,
    },
    Burst {
        prompts: usize,
        tokens: u32,
    },
    Churn {
        sessions: usize,
        turns: usize,
        context_tokens: u32,
        live_ratio: f64,
    },
}

/// What a run measured.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct RunReport {
    pub steps: u64,
    pub prefilled_tokens: u64,
    pub restored_tokens: u64,
    pub device_hits: u64,
    pub host_hits: u64,
    pub misses: u64,
    pub ttft_ns: Vec<u64>,
    pub invariant_failures: Vec<String>,
    pub schedule_log: Vec<String>,
}

pub struct Simulator {
    _private: (),
}

impl Simulator {
    pub fn new(model: EngineModel, cache: HostCache<StubCopyEngine>, seed: u64) -> Self {
        let _ = (model, cache, seed);
        unimplemented!("HC-4")
    }
    pub fn run(&mut self, workload: &Workload) -> RunReport {
        let _ = workload;
        unimplemented!("HC-4")
    }
    pub fn cache(&self) -> &HostCache<StubCopyEngine> {
        unimplemented!("HC-4")
    }
    pub fn now_ns(&self) -> u64 {
        unimplemented!("HC-4")
    }
}
