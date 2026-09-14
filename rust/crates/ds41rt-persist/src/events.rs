//! Retention events: what the scheduler does to the prefix cache, replayable from traces.
//!
//! `restore` at admission; `retain(Prompt)` after prefill of an uncached prompt; `retain(Turn)` at
//! retirement of a cacheable finished request; `evict_one` on bank overflow; `make_room` under
//! pool pressure; a `Step` per scheduler step with whether it carried decode work (the idle
//! flusher runs only on decode-free steps). Traces are the captured token-timing JSON from the
//! fleet (`cap2-*.json` shape: turns with `t_start`, `ttft`, `t_end`, `events`).
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum RetentionEvent {
    Admit { request: u64, tokens: Vec<u32>, at_ns: u64 },
    PrefillDone { request: u64, at_ns: u64 },
    RetainPrompt { request: u64, at_ns: u64 },
    RetireTurn { request: u64, cacheable: bool, at_ns: u64 },
    Step { at_ns: u64, decode_tokens: u32 },
    PoolPressure { needed_pages: usize, at_ns: u64 },
    Cancel { request: u64, at_ns: u64 },
}

pub trait Clock {
    fn now_ns(&self) -> u64;
    fn advance(&mut self, nanos: u64);
}
