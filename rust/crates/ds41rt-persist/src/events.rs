//! Retention events: what the scheduler does to the prefix cache, replayable from traces.
//!
//! `restore` at admission; `retain(Prompt)` after prefill of an uncached prompt; `retain(Turn)` at
//! retirement of a cacheable finished request; `evict_one` on bank overflow; `make_room` under
//! pool pressure; a `Step` per scheduler step with whether it carried decode work (the idle
//! flusher runs only on decode-free steps). Traces are the captured token-timing JSON from the
//! fleet (`cap2-*.json` shape: turns with `t_start`, `ttft`, `t_end`, `events`).
use serde::{Deserialize, Serialize};

/// One retention event: what the scheduler does to the prefix cache.
///
/// Emission semantics:
/// - `Admit` is emitted once per turn at admission, carrying the session's cumulative prompt.
/// - `PrefillDone` is emitted at the turn's first token, when the trace reports a TTFT.
/// - `RetainPrompt` follows `PrefillDone` at the same instant when the prompt was not fully cached.
/// - `RetireTurn` is emitted at the turn's end, marking whether the finished request is cacheable.
/// - `Step` is emitted once per scheduler step; `decode_tokens` is zero on the decode-free steps
///   the idle flusher rides on.
/// - `PoolPressure` and `Cancel` are produced by the live scheduler, never by [`TraceReplay`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum RetentionEvent {
    /// A request entered the scheduler with the session's cumulative prompt.
    Admit {
        /// The request's id.
        request: u64,
        /// The session's cumulative prompt token ids at admission.
        tokens: Vec<u32>,
        /// Admission time in nanoseconds.
        at_ns: u64,
    },
    /// The request's prefill finished and its first token is about to be emitted.
    PrefillDone {
        /// The request's id.
        request: u64,
        /// Time of the first token in nanoseconds.
        at_ns: u64,
    },
    /// The request's prompt was not fully cached and must be retained.
    RetainPrompt {
        /// The request's id.
        request: u64,
        /// Time of the prefill that produced the prompt, in nanoseconds.
        at_ns: u64,
    },
    /// The request retired; `cacheable` says whether its prefix may be retained.
    RetireTurn {
        /// The request's id.
        request: u64,
        /// Whether the finished request's prefix is worth retaining.
        cacheable: bool,
        /// Retirement time in nanoseconds.
        at_ns: u64,
    },
    /// One scheduler step; `decode_tokens` is zero on a decode-free step.
    Step {
        /// Step time in nanoseconds.
        at_ns: u64,
        /// Decode tokens carried by the step; zero means the idle flusher may run.
        decode_tokens: u32,
    },
    /// The pools are under pressure and the scheduler must make room.
    PoolPressure {
        /// Pages the scheduler needs to free.
        needed_pages: usize,
        /// Time of the pressure in nanoseconds.
        at_ns: u64,
    },
    /// A request was cancelled before it retired.
    Cancel {
        /// The request's id.
        request: u64,
        /// Cancellation time in nanoseconds.
        at_ns: u64,
    },
}

/// A monotone source of nanoseconds for replay and scheduling.
///
/// Invariant: `now_ns` never decreases; `advance` moves it forward and `advance_to` clamps to the
/// later of the current reading and its target.
pub trait Clock {
    /// The current reading in nanoseconds.
    fn now_ns(&self) -> u64;
    /// Move the reading forward by `nanos`, saturating at `u64::MAX`.
    fn advance(&mut self, nanos: u64);
}

/// Nanoseconds per second; trace times are seconds from the run start.
pub const NANOS_PER_SECOND: u64 = 1_000_000_000;
/// Cadence of the decode-free `Step`s the idle flusher rides on.
pub const IDLE_STEP_NANOS: u64 = 10_000_000;

/// A `Clock` whose time only moves when a caller advances it.
///
/// Invariant: `now_ns` is the sum of every `advance` and `advance_to` target applied so far,
/// saturated at `u64::MAX`; it never moves backwards, so a replay driven by this clock is
/// bit-for-bit reproducible.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VirtualClock {
    now_ns: u64,
}

impl VirtualClock {
    /// A clock reading zero.
    pub fn new() -> Self {
        Self::default()
    }

    /// A clock already reading `now_ns`.
    pub fn starting_at(now_ns: u64) -> Self {
        Self { now_ns }
    }

    /// Move to `at_ns`; a time already passed is a no-op, so the clock never runs backwards.
    pub fn advance_to(&mut self, at_ns: u64) {
        self.now_ns = self.now_ns.max(at_ns);
    }
}

impl Clock for VirtualClock {
    fn now_ns(&self) -> u64 {
        self.now_ns
    }

    fn advance(&mut self, nanos: u64) {
        self.now_ns = self.now_ns.saturating_add(nanos);
    }
}

/// One recorded turn: a request's prompt, its timing and its emitted token chunks.
#[derive(Clone, Debug, Deserialize)]
pub struct TraceTurn {
    /// The turn's own prompt text; the session's cumulative prompt is the concatenation of every
    /// turn's `q`. The trace carries no token ids, so ids are synthesised from this.
    pub q: String,
    /// Admission time, seconds from the run start.
    pub t_start: f64,
    /// Time to first token, seconds from the run start; `None` when the turn never prefilled.
    pub ttft: Option<f64>,
    /// Retirement time, seconds from the run start.
    pub t_end: f64,
    /// `(t_seconds, text)` per emitted chunk, in emission order.
    pub events: Vec<(f64, String)>,
    /// Provider usage block; only `prompt_cache_hit_tokens` is read.
    #[serde(default)]
    pub usage: TraceUsage,
}

/// The subset of a provider usage block the replay needs.
#[derive(Clone, Copy, Debug, Default, Deserialize)]
pub struct TraceUsage {
    /// Prompt tokens served from the provider's own cache, when the provider reports it.
    #[serde(default)]
    pub prompt_cache_hit_tokens: Option<u64>,
    /// Prompt tokens the provider had to prefill, when the provider reports it. This is the turn's
    /// own prompt length: `prompt_tokens` is the whole session's cumulative prompt.
    #[serde(default)]
    pub prompt_cache_miss_tokens: Option<u64>,
}

/// One recorded run: a list of sessions, each a list of turns.
pub type TraceSession = Vec<TraceTurn>;

/// A recorded token-timing trace: `{"mode": ..., "runs": [[turn, ...], ...]}`.
#[derive(Clone, Debug, Deserialize)]
pub struct Trace {
    /// `"solo"` for one session, `"c4"` for four concurrent sessions.
    pub mode: String,
    /// One entry per session.
    pub runs: Vec<TraceSession>,
}

/// A trace that could not be turned into events.
#[derive(Debug, thiserror::Error)]
pub enum TraceError {
    /// The JSON did not match the trace shape.
    #[error("trace is not valid token-timing JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// A session held no turns, so it has no admission to anchor request ids.
    #[error("session {session} is empty")]
    EmptySession {
        /// Index of the offending session.
        session: usize,
    },
    /// A turn's `t_end` preceded its `t_start`.
    #[error("session {session} turn {turn} retires at {t_end_ns} ns, before it was admitted at {t_start_ns} ns")]
    RetiresBeforeAdmission {
        /// Index of the offending session.
        session: usize,
        /// Index of the offending turn.
        turn: usize,
        /// Admission time in nanoseconds.
        t_start_ns: u64,
        /// Retirement time in nanoseconds.
        t_end_ns: u64,
    },
}

/// Turns a recorded token-timing trace into the ordered retention events it implies.
///
/// Invariant: the returned vector is sorted by `at_ns`, and within one instant by the fixed
/// emission order `Admit`, `PrefillDone`, `RetainPrompt`, `Step`, `RetireTurn`, `PoolPressure`,
/// `Cancel`, so replaying the same trace twice yields identical events. `TraceReplay` never emits
/// `PoolPressure` or `Cancel`; those two ranks exist so the live scheduler's events sort with the
/// replayed ones.
#[derive(Clone, Debug)]
pub struct TraceReplay {
    events: Vec<RetentionEvent>,
}

impl TraceReplay {
    /// Replay `trace`, interleaving every session by time.
    pub fn from_trace(trace: &Trace) -> Result<Self, TraceError> {
        let mut events = Vec::new();
        for (session, turns) in trace.runs.iter().enumerate() {
            replay_session(session, turns, &mut events)?;
        }
        events.extend(idle_steps(&events));
        events.sort_by_key(event_order);
        Ok(Self { events })
    }

    /// Parse a trace from JSON and replay it.
    pub fn from_json(json: &str) -> Result<Self, TraceError> {
        Self::from_trace(&serde_json::from_str(json)?)
    }

    /// The replayed events, ordered by time.
    pub fn events(&self) -> &[RetentionEvent] {
        &self.events
    }

    /// Consume the replay, yielding its events.
    pub fn into_events(self) -> Vec<RetentionEvent> {
        self.events
    }

    /// The number of events in the replay.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the replay holds no events.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

/// A synthetic trace shape, for cases the recorded traces do not cover.
///
/// Invariant: the same `seed` always produces the same trace, so a failing synthetic run is
/// reproducible from its seed alone.
#[derive(Clone, Copy, Debug)]
pub struct SyntheticShape {
    /// Number of concurrent sessions.
    pub sessions: usize,
    /// Turns per session.
    pub turns_per_session: usize,
    /// Prompt tokens per turn.
    pub prompt_tokens: u64,
    /// Prompt tokens served from cache on every turn but the first.
    pub cached_tokens: u64,
    /// Decode chunks emitted per turn.
    pub chunks_per_turn: u64,
    /// Seconds between a turn's admission and its first token.
    pub ttft_seconds: f64,
    /// Seconds between consecutive decode chunks.
    pub chunk_seconds: f64,
    /// Seconds a session waits after retiring a turn before admitting the next.
    pub gap_seconds: f64,
}

impl Default for SyntheticShape {
    fn default() -> Self {
        Self {
            sessions: 4,
            turns_per_session: 8,
            prompt_tokens: 14_800,
            cached_tokens: 14_700,
            chunks_per_turn: 16,
            ttft_seconds: 0.08,
            chunk_seconds: 0.04,
            gap_seconds: 0.02,
        }
    }
}

/// Build a deterministic synthetic trace: `sessions` sessions of `turns_per_session` turns each.
///
/// The shape is the one the recorded `c4` traces show — a long shared prompt, a short prefill on
/// every turn after the first, and a burst of decode chunks — so suites that need more sessions or
/// turns than the captures hold can still run.
pub fn synthetic(seed: u64, sessions: usize, turns_per_session: usize) -> Trace {
    synthetic_with(
        seed,
        SyntheticShape {
            sessions,
            turns_per_session,
            ..SyntheticShape::default()
        },
    )
}

/// Build a deterministic synthetic trace from an explicit shape.
pub fn synthetic_with(seed: u64, shape: SyntheticShape) -> Trace {
    let mut rng = SplitMix64::new(seed);
    let runs = (0..shape.sessions)
        .map(|session| synthetic_session(&mut rng, session, shape))
        .collect();
    Trace {
        mode: if shape.sessions == 1 {
            "solo".into()
        } else {
            format!("c{}", shape.sessions)
        },
        runs,
    }
}

/// One synthetic session: turns laid end to end, each with a jittered start.
fn synthetic_session(rng: &mut SplitMix64, session: usize, shape: SyntheticShape) -> TraceSession {
    let mut turns = Vec::with_capacity(shape.turns_per_session);
    let mut cursor = session as f64 * shape.gap_seconds;
    for turn in 0..shape.turns_per_session {
        let t_start = cursor + rng.unit() * shape.gap_seconds;
        let ttft = t_start + shape.ttft_seconds;
        let events = (0..shape.chunks_per_turn)
            .map(|chunk| {
                (
                    ttft + (chunk + 1) as f64 * shape.chunk_seconds,
                    format!("c{chunk}"),
                )
            })
            .collect();
        let t_end = ttft + shape.chunks_per_turn as f64 * shape.chunk_seconds;
        let cached = if turn == 0 { 0 } else { shape.cached_tokens };
        turns.push(TraceTurn {
            q: format!("synthetic session {session} turn {turn}"),
            t_start,
            ttft: Some(ttft),
            t_end,
            events,
            usage: TraceUsage {
                prompt_cache_hit_tokens: Some(cached),
                prompt_cache_miss_tokens: Some(shape.prompt_tokens - cached),
            },
        });
        cursor = t_end + shape.gap_seconds;
    }
    turns
}

/// The decode-free `Step`s the idle flusher rides on: one every [`IDLE_STEP_NANOS`] while no
/// session is producing tokens.
///
/// A session is producing tokens from its `PrefillDone` until its `RetireTurn`; the gaps between
/// those windows are exactly the steps on which the engine may flush stale entries. The first
/// idle step lands one cadence after the run starts, so a run that never goes idle emits none.
fn idle_steps(events: &[RetentionEvent]) -> Vec<RetentionEvent> {
    let Some(end_ns) = events.iter().map(event_time).max() else {
        return Vec::new();
    };
    let mut busy = busy_windows(events);
    busy.sort_unstable();

    let mut idle = Vec::new();
    let mut at_ns = IDLE_STEP_NANOS;
    let mut window = busy.iter().peekable();
    while at_ns <= end_ns {
        // Skip every window that ended before this instant; the next one is the only candidate.
        while window.peek().is_some_and(|(_, end)| *end <= at_ns) {
            window.next();
        }
        let producing = window.peek().is_some_and(|(start, _)| *start <= at_ns);
        if !producing {
            idle.push(RetentionEvent::Step {
                at_ns,
                decode_tokens: 0,
            });
        }
        at_ns += IDLE_STEP_NANOS;
    }
    idle
}

/// The `[prefill, retire)` windows during which some session was producing tokens.
fn busy_windows(events: &[RetentionEvent]) -> Vec<(u64, u64)> {
    let mut prefill: std::collections::BTreeMap<u64, u64> = std::collections::BTreeMap::new();
    let mut busy = Vec::new();
    for event in events {
        match event {
            RetentionEvent::PrefillDone { request, at_ns } => {
                prefill.insert(*request, *at_ns);
            }
            RetentionEvent::RetireTurn { request, at_ns, .. } => {
                if let Some(start) = prefill.remove(request) {
                    busy.push((start, *at_ns));
                }
            }
            _ => {}
        }
    }
    busy
}

/// The time an event happened.
fn event_time(event: &RetentionEvent) -> u64 {
    event_order(event).0
}

/// Emit one session's events. Request ids are `session * 2^32 + turn`, so they are unique across
/// sessions without a shared counter and stay stable when sessions are replayed in any order.
fn replay_session(
    session: usize,
    turns: &[TraceTurn],
    out: &mut Vec<RetentionEvent>,
) -> Result<(), TraceError> {
    if turns.is_empty() {
        return Err(TraceError::EmptySession { session });
    }
    let mut tokens = Vec::new();
    for (turn, record) in turns.iter().enumerate() {
        let request = request_id(session, turn);
        let t_start_ns = seconds_to_nanos(record.t_start);
        let t_end_ns = seconds_to_nanos(record.t_end);
        if t_end_ns < t_start_ns {
            return Err(TraceError::RetiresBeforeAdmission {
                session,
                turn,
                t_start_ns,
                t_end_ns,
            });
        }
        tokens.extend(record.q.bytes().map(token_id));
        out.push(RetentionEvent::Admit {
            request,
            tokens: tokens.clone(),
            at_ns: t_start_ns,
        });
        if let Some(ttft) = record.ttft {
            let ttft_ns = seconds_to_nanos(ttft);
            out.push(RetentionEvent::PrefillDone {
                request,
                at_ns: ttft_ns,
            });
            if prompt_was_uncached(record) {
                out.push(RetentionEvent::RetainPrompt {
                    request,
                    at_ns: ttft_ns,
                });
            }
        }
        for (at, _) in &record.events {
            out.push(RetentionEvent::Step {
                at_ns: seconds_to_nanos(*at),
                decode_tokens: 1,
            });
        }
        out.push(RetentionEvent::RetireTurn {
            request,
            cacheable: true,
            at_ns: t_end_ns,
        });
    }
    Ok(())
}

/// The request id of `turn` in `session`: unique across sessions, stable across replays.
fn request_id(session: usize, turn: usize) -> u64 {
    ((session as u64) << 32) | turn as u64
}

/// A stable token id for one byte of prompt text.
///
/// The trace carries no tokenizer, so the only lossless, collision-free mapping available is one id
/// per byte: `byte + 1` preserves the prompt's exact byte sequence and its exact length, which
/// [`prompt_was_uncached`] needs. Hashing chunks would collide and change that length, so it is not
/// used.
fn token_id(byte: u8) -> u32 {
    u32::from(byte) + 1
}

/// The turn's own prompt length in tokens: the provider's miss count when it reports one, else the
/// synthesised token count of the turn's own prompt text.
///
/// `prompt_tokens` is the session's cumulative prompt, so it is never used here.
fn prompt_len(record: &TraceTurn) -> u64 {
    record
        .usage
        .prompt_cache_miss_tokens
        .unwrap_or(record.q.len() as u64)
}

/// Whether the turn's prompt was not fully served from cache, and so must be retained.
///
/// A turn with no reported hit count is treated as uncached, since the provider cannot prove it was
/// served from cache.
pub fn prompt_was_uncached(record: &TraceTurn) -> bool {
    match record.usage.prompt_cache_hit_tokens {
        None => true,
        Some(hit) => hit < prompt_len(record),
    }
}

/// Seconds to nanoseconds, saturating rather than wrapping on absurd inputs.
fn seconds_to_nanos(seconds: f64) -> u64 {
    if !seconds.is_finite() || seconds <= 0.0 {
        return 0;
    }
    (seconds * NANOS_PER_SECOND as f64)
        .round()
        .min(u64::MAX as f64) as u64
}

/// Total order over events: by time, then by the fixed emission order within one instant.
fn event_order(event: &RetentionEvent) -> (u64, u8) {
    let (at_ns, rank) = match event {
        RetentionEvent::Admit { at_ns, .. } => (*at_ns, 0),
        RetentionEvent::PrefillDone { at_ns, .. } => (*at_ns, 1),
        RetentionEvent::RetainPrompt { at_ns, .. } => (*at_ns, 2),
        RetentionEvent::Step { at_ns, .. } => (*at_ns, 3),
        RetentionEvent::RetireTurn { at_ns, .. } => (*at_ns, 4),
        RetentionEvent::PoolPressure { at_ns, .. } => (*at_ns, 5),
        RetentionEvent::Cancel { at_ns, .. } => (*at_ns, 6),
    };
    (at_ns, rank)
}

/// SplitMix64: a tiny, well-distributed generator used only to jitter synthetic traces.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(
        q: &str,
        t_start: f64,
        ttft: Option<f64>,
        t_end: f64,
        events: &[(f64, &str)],
    ) -> TraceTurn {
        TraceTurn {
            q: q.into(),
            t_start,
            ttft,
            t_end,
            events: events.iter().map(|(t, s)| (*t, (*s).to_string())).collect(),
            usage: TraceUsage::default(),
        }
    }

    #[test]
    fn virtual_clock_accumulates_and_never_rewinds() {
        let mut clock = VirtualClock::new();
        assert_eq!(clock.now_ns(), 0);
        clock.advance(5);
        clock.advance(7);
        assert_eq!(clock.now_ns(), 12);
        clock.advance_to(3);
        assert_eq!(clock.now_ns(), 12);
        clock.advance_to(40);
        assert_eq!(clock.now_ns(), 40);
        assert_eq!(VirtualClock::starting_at(9).now_ns(), 9);
    }

    #[test]
    fn virtual_clock_saturates_instead_of_wrapping() {
        let mut clock = VirtualClock::starting_at(u64::MAX - 1);
        clock.advance(10);
        assert_eq!(clock.now_ns(), u64::MAX);
    }

    #[test]
    fn admit_carries_the_cumulative_prompt() {
        let trace = Trace {
            mode: "solo".into(),
            runs: vec![vec![
                turn("ab", 0.0, Some(0.5), 1.0, &[(0.5, "x")]),
                turn("c", 1.0, Some(1.5), 2.0, &[(1.5, "y")]),
            ]],
        };
        let replay = TraceReplay::from_trace(&trace).expect("replays");
        let admits: Vec<&Vec<u32>> = replay
            .events()
            .iter()
            .filter_map(|e| match e {
                RetentionEvent::Admit { tokens, .. } => Some(tokens),
                _ => None,
            })
            .collect();
        assert_eq!(admits[0].len(), 2);
        assert_eq!(
            admits[1].len(),
            3,
            "the second turn extends the session's prompt"
        );
    }

    #[test]
    fn retain_prompt_follows_an_uncached_prefill() {
        let mut cached = turn("abcd", 0.0, Some(0.5), 1.0, &[]);
        cached.usage.prompt_cache_hit_tokens = Some(4);
        let mut partial = turn("abcd", 1.0, Some(1.5), 2.0, &[]);
        partial.usage.prompt_cache_hit_tokens = Some(3);
        let trace = Trace {
            mode: "solo".into(),
            runs: vec![vec![cached, partial]],
        };
        let replay = TraceReplay::from_trace(&trace).expect("replays");
        let retains = replay
            .events()
            .iter()
            .filter(|e| matches!(e, RetentionEvent::RetainPrompt { .. }))
            .count();
        assert_eq!(retains, 1, "only the partially cached prompt is retained");
    }

    #[test]
    fn a_turn_without_ttft_emits_no_prefill() {
        let trace = Trace {
            mode: "solo".into(),
            runs: vec![vec![turn("q", 0.0, None, 1.0, &[(0.5, "x")])]],
        };
        let replay = TraceReplay::from_trace(&trace).expect("replays");
        assert!(!replay
            .events()
            .iter()
            .any(|e| matches!(e, RetentionEvent::PrefillDone { .. })));
        assert!(replay
            .events()
            .iter()
            .any(|e| matches!(e, RetentionEvent::Step { .. })));
    }

    #[test]
    fn an_empty_session_is_rejected() {
        let trace = Trace {
            mode: "solo".into(),
            runs: vec![vec![]],
        };
        assert!(matches!(
            TraceReplay::from_trace(&trace),
            Err(TraceError::EmptySession { session: 0 })
        ));
    }

    #[test]
    fn a_turn_retiring_before_admission_is_rejected() {
        let trace = Trace {
            mode: "solo".into(),
            runs: vec![vec![turn("q", 2.0, Some(2.5), 1.0, &[])]],
        };
        assert!(matches!(
            TraceReplay::from_trace(&trace),
            Err(TraceError::RetiresBeforeAdmission {
                session: 0,
                turn: 0,
                ..
            })
        ));
    }

    #[test]
    fn malformed_json_is_rejected() {
        assert!(matches!(
            TraceReplay::from_json("not json"),
            Err(TraceError::Json(_))
        ));
    }

    #[test]
    fn synthetic_is_deterministic_per_seed() {
        let a = synthetic(7, 3, 4);
        let b = synthetic(7, 3, 4);
        let c = synthetic(8, 3, 4);
        assert_eq!(a.runs.len(), 3);
        assert_eq!(a.runs[0].len(), 4);
        assert_eq!(a.runs[0][0].t_start, b.runs[0][0].t_start);
        assert_ne!(a.runs[0][0].t_start, c.runs[0][0].t_start);
    }

    #[test]
    fn synthetic_solo_mode_names_one_session() {
        assert_eq!(synthetic(1, 1, 2).mode, "solo");
        assert_eq!(synthetic(1, 4, 2).mode, "c4");
    }

    #[test]
    fn seconds_convert_saturating() {
        assert_eq!(seconds_to_nanos(0.0), 0);
        assert_eq!(seconds_to_nanos(-1.0), 0);
        assert_eq!(seconds_to_nanos(f64::NAN), 0);
        assert_eq!(seconds_to_nanos(1.5), 1_500_000_000);
        assert_eq!(seconds_to_nanos(f64::INFINITY), 0);
    }
}
