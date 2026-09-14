//! Functional suite for P0-4: trace replay, the deterministic interleaver and the counters.
//!
//! The two recorded traces under `tests/data/` are the contract: `cap2-solo.json` is one session of
//! twelve turns, `cap2-c4.json` is four sessions that overlap in time.
use ds41rt_persist::events::{
    prompt_was_uncached, synthetic, synthetic_with, Clock, RetentionEvent, SyntheticShape, Trace,
    TraceError, TraceReplay, TraceTurn, TraceUsage, VirtualClock, IDLE_STEP_NANOS,
    NANOS_PER_SECOND,
};
use ds41rt_persist::metrics::{Counter, Counters, Metered};
use ds41rt_persist::sched::{explore, Interleaver, ReplayError, Shared, XorShift64};
use proptest::prelude::*;
use std::cell::Cell;

#[allow(dead_code)]
mod common;

use common::{yielders, PRODUCTION_TASKS};

const SOLO: &str = include_str!("data/cap2-solo.json");
const C4: &str = include_str!("data/cap2-c4.json");

/// The number of turns in the recorded solo capture.
const SOLO_TURNS: usize = 12;
/// The number of sessions in the recorded c4 capture.
const C4_SESSIONS: usize = 4;

fn replay(json: &str) -> TraceReplay {
    TraceReplay::from_json(json).expect("the recorded trace replays")
}

fn admits(events: &[RetentionEvent]) -> Vec<(u64, u64)> {
    events
        .iter()
        .filter_map(|event| match event {
            RetentionEvent::Admit { request, at_ns, .. } => Some((*request, *at_ns)),
            _ => None,
        })
        .collect()
}

fn steps(events: &[RetentionEvent]) -> Vec<(u64, u32)> {
    events
        .iter()
        .filter_map(|event| match event {
            RetentionEvent::Step {
                at_ns,
                decode_tokens,
            } => Some((*at_ns, *decode_tokens)),
            _ => None,
        })
        .collect()
}

/// The `PrefillDone` time of `request`, if it prefilled.
fn prefill_of(events: &[RetentionEvent], request: u64) -> Option<u64> {
    events.iter().find_map(|event| match event {
        RetentionEvent::PrefillDone { request: r, at_ns } if *r == request => Some(*at_ns),
        _ => None,
    })
}

/// The first decode `Step` inside `request`'s own `[Admit, RetireTurn)` window.
///
/// The window starts at admission, not at `PrefillDone`: anchoring it at the prefill would make the
/// ordering assertion tautological, since the first step after the prefill is by construction after
/// it. Anchoring at admission is what makes a late `PrefillDone` fail the test.
fn first_decode_step(events: &[RetentionEvent], request: u64) -> Option<u64> {
    let admit = events.iter().find_map(|event| match event {
        RetentionEvent::Admit {
            request: r, at_ns, ..
        } if *r == request => Some(*at_ns),
        _ => None,
    })?;
    let retire = events.iter().find_map(|event| match event {
        RetentionEvent::RetireTurn {
            request: r, at_ns, ..
        } if *r == request => Some(*at_ns),
        _ => None,
    })?;
    events.iter().find_map(|event| match event {
        RetentionEvent::Step {
            at_ns,
            decode_tokens,
        } if *decode_tokens == 1 && *at_ns >= admit && *at_ns < retire => Some(*at_ns),
        _ => None,
    })
}

/// The `[PrefillDone, RetireTurn)` windows during which some request was producing tokens.
fn busy_windows(events: &[RetentionEvent]) -> Vec<(u64, u64)> {
    let mut prefill = std::collections::BTreeMap::new();
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

// ---------------------------------------------------------------- recorded traces

#[test]
fn solo_replay_admits_every_turn_in_monotone_time() {
    let replay = replay(SOLO);
    let admits = admits(replay.events());
    assert_eq!(admits.len(), SOLO_TURNS);
    assert!(
        admits.windows(2).all(|pair| pair[0].1 <= pair[1].1),
        "admission times must be monotone: {admits:?}"
    );
    assert!(
        admits
            .iter()
            .map(|(request, _)| request)
            .all(|request| request >> 32 == 0),
        "a solo run has one session, so every request id is in session 0"
    );
}

#[test]
fn solo_replay_prefills_before_the_first_step_of_each_turn() {
    let replay = replay(SOLO);
    let events = replay.events();
    for (request, _) in admits(events) {
        let prefill = prefill_of(events, request)
            .unwrap_or_else(|| panic!("request {request} never prefilled"));
        let first_step = first_decode_step(events, request)
            .unwrap_or_else(|| panic!("request {request} never stepped"));
        assert!(
            prefill <= first_step,
            "request {request} emitted its first token at {first_step} before its prefill at {prefill}"
        );
    }
}

#[test]
fn solo_replay_retires_every_turn_as_cacheable() {
    let replay = replay(SOLO);
    let retires: Vec<bool> = replay
        .events()
        .iter()
        .filter_map(|event| match event {
            RetentionEvent::RetireTurn { cacheable, .. } => Some(*cacheable),
            _ => None,
        })
        .collect();
    assert_eq!(retires.len(), SOLO_TURNS);
    assert!(retires.iter().all(|cacheable| *cacheable));
}

#[test]
fn solo_replay_emits_one_decode_step_per_token_event() {
    let replay = replay(SOLO);
    let trace: Trace = serde_json::from_str(SOLO).expect("the capture parses");
    let chunks: usize = trace
        .runs
        .iter()
        .flatten()
        .map(|turn| turn.events.len())
        .sum();
    let decode_steps = steps(replay.events())
        .iter()
        .filter(|(_, tokens)| *tokens == 1)
        .count();
    assert_eq!(decode_steps, chunks);
}

#[test]
fn solo_replay_events_are_ordered_by_time() {
    let replay = replay(SOLO);
    let times: Vec<u64> = replay
        .events()
        .iter()
        .map(|event| match event {
            RetentionEvent::Admit { at_ns, .. }
            | RetentionEvent::PrefillDone { at_ns, .. }
            | RetentionEvent::RetainPrompt { at_ns, .. }
            | RetentionEvent::RetireTurn { at_ns, .. }
            | RetentionEvent::Step { at_ns, .. }
            | RetentionEvent::PoolPressure { at_ns, .. }
            | RetentionEvent::Cancel { at_ns, .. } => *at_ns,
        })
        .collect();
    assert!(
        times.windows(2).all(|pair| pair[0] <= pair[1]),
        "events must be time-ordered"
    );
}

#[test]
fn c4_replay_interleaves_four_sessions() {
    let replay = replay(C4);
    let sessions: std::collections::BTreeSet<u64> = admits(replay.events())
        .iter()
        .map(|(request, _)| request >> 32)
        .collect();
    assert_eq!(sessions.len(), C4_SESSIONS);
    assert_eq!(sessions, (0..C4_SESSIONS as u64).collect());

    let admits = admits(replay.events());
    assert!(
        admits
            .windows(2)
            .any(|pair| pair[0].0 >> 32 != pair[1].0 >> 32),
        "the four sessions must interleave, not run one after another"
    );
}

#[test]
fn c4_replay_keeps_each_session_in_its_own_order() {
    let replay = replay(C4);
    let mut last: std::collections::BTreeMap<u64, u64> = std::collections::BTreeMap::new();
    for (request, at_ns) in admits(replay.events()) {
        let session = request >> 32;
        if let Some(previous) = last.insert(session, at_ns) {
            assert!(previous <= at_ns, "session {session} admitted out of order");
        }
    }
}

#[test]
fn c4_replay_retains_the_prompts_that_missed_the_cache() {
    let replay = replay(C4);
    let trace: Trace = serde_json::from_str(C4).expect("the capture parses");
    let expected = trace
        .runs
        .iter()
        .flatten()
        .filter(|turn| prompt_was_uncached(turn))
        .count();
    let retains = replay
        .events()
        .iter()
        .filter(|event| matches!(event, RetentionEvent::RetainPrompt { .. }))
        .count();
    assert_eq!(retains, expected);
    assert_eq!(
        expected, 0,
        "every c4 session starts from a warm shared prefix, so no prompt needs retaining"
    );
}

#[test]
fn solo_replay_retains_the_cold_first_prompt() {
    let replay = replay(SOLO);
    let retains: Vec<u64> = replay
        .events()
        .iter()
        .filter_map(|event| match event {
            RetentionEvent::RetainPrompt { request, .. } => Some(*request),
            _ => None,
        })
        .collect();
    assert_eq!(
        retains,
        vec![0],
        "only the session's first, cold prompt is retained"
    );
}

#[test]
fn replay_is_idempotent() {
    let first = replay(C4);
    let second = replay(C4);
    assert_eq!(first.events(), second.events());
}

#[test]
fn replay_rejects_a_trace_with_an_empty_session() {
    let trace = Trace {
        mode: "solo".into(),
        runs: vec![Vec::new()],
    };
    assert!(matches!(
        TraceReplay::from_trace(&trace),
        Err(TraceError::EmptySession { session: 0 })
    ));
}

#[test]
fn replay_rejects_a_turn_that_retires_before_it_is_admitted() {
    let json = r#"{"mode":"solo","runs":[[{"q":"q","t_start":2.0,"ttft":2.5,"t_end":1.0,
        "events":[],"usage":{}}]]}"#;
    assert!(matches!(
        TraceReplay::from_json(json),
        Err(TraceError::RetiresBeforeAdmission {
            session: 0,
            turn: 0,
            ..
        })
    ));
}

#[test]
fn replay_rejects_malformed_json() {
    assert!(matches!(
        TraceReplay::from_json("{"),
        Err(TraceError::Json(_))
    ));
}

#[test]
fn replay_accepts_a_turn_without_a_ttft() {
    let json = r#"{"mode":"solo","runs":[[{"q":"q","t_start":0.0,"ttft":null,"t_end":1.0,
        "events":[[0.5,"x"]],"usage":{}}]]}"#;
    let replay = TraceReplay::from_json(json).expect("a null ttft is a legal turn");
    assert!(!replay
        .events()
        .iter()
        .any(|event| matches!(event, RetentionEvent::PrefillDone { .. })));
    assert_eq!(
        steps(replay.events())
            .iter()
            .filter(|(_, tokens)| *tokens == 1)
            .count(),
        1
    );
}

#[test]
fn replay_accepts_a_turn_without_a_usage_block() {
    let json = r#"{"mode":"solo","runs":[[{"q":"q","t_start":0.0,"ttft":0.5,"t_end":1.0,
        "events":[],"usage":{}}]]}"#;
    let replay = TraceReplay::from_json(json).expect("usage is optional");
    assert!(replay
        .events()
        .iter()
        .any(|event| matches!(event, RetentionEvent::RetainPrompt { .. })));
}

#[test]
fn a_fully_cached_prompt_without_a_miss_count_is_not_retained() {
    // The provider reports a hit count but no miss count, so the turn's own prompt length is the
    // fallback. A hit equal to that length means the prompt was fully cached.
    let mut turn = TraceTurn {
        q: "abcdefghij".into(),
        t_start: 0.0,
        ttft: Some(0.5),
        t_end: 1.0,
        events: Vec::new(),
        usage: TraceUsage {
            prompt_cache_hit_tokens: Some(10),
            prompt_cache_miss_tokens: None,
        },
    };
    assert!(!prompt_was_uncached(&turn));
    let replay = TraceReplay::from_trace(&Trace {
        mode: "solo".into(),
        runs: vec![vec![turn.clone()]],
    })
    .expect("replays");
    assert!(
        !replay
            .events()
            .iter()
            .any(|event| matches!(event, RetentionEvent::RetainPrompt { .. })),
        "a fully cached prompt must not be retained"
    );

    // One token short of the prompt length is a partial miss, so it must be retained.
    turn.usage.prompt_cache_hit_tokens = Some(9);
    assert!(prompt_was_uncached(&turn));
}

#[test]
fn replay_reports_its_own_size() {
    let replay = replay(SOLO);
    assert_eq!(replay.len(), replay.events().len());
    assert!(!replay.is_empty());
    assert_eq!(replay.clone().into_events(), replay.events().to_vec());
}

// ---------------------------------------------------------------- synthetic traces

#[test]
fn synthetic_is_deterministic_per_seed() {
    let first = synthetic(11, 4, 6);
    let second = synthetic(11, 4, 6);
    let other = synthetic(12, 4, 6);
    assert_eq!(first.runs.len(), 4);
    assert!(first.runs.iter().all(|session| session.len() == 6));
    assert_eq!(first.runs[0][0].t_start, second.runs[0][0].t_start);
    assert_ne!(first.runs[0][0].t_start, other.runs[0][0].t_start);
}

#[test]
fn synthetic_replays_to_the_same_events_every_time() {
    let first = TraceReplay::from_trace(&synthetic(3, 4, 5)).expect("replays");
    let second = TraceReplay::from_trace(&synthetic(3, 4, 5)).expect("replays");
    assert_eq!(first.events(), second.events());
}

#[test]
fn synthetic_covers_the_idle_step_cadence() {
    let shape = SyntheticShape {
        sessions: 2,
        turns_per_session: 3,
        ..SyntheticShape::default()
    };
    let replay = TraceReplay::from_trace(&synthetic_with(5, shape)).expect("replays");
    let idle = steps(replay.events())
        .iter()
        .filter(|(_, tokens)| *tokens == 0)
        .count();
    assert!(
        idle > 0,
        "a synthetic run must exercise the decode-free steps"
    );
    assert_eq!(IDLE_STEP_NANOS, 10_000_000);
}

#[test]
fn idle_steps_land_only_on_the_cadence_outside_busy_windows() {
    let replay = TraceReplay::from_trace(&synthetic(5, 2, 3)).expect("replays");
    let events = replay.events();
    let busy = busy_windows(events);
    let idle: Vec<u64> = steps(events)
        .iter()
        .filter(|(_, tokens)| *tokens == 0)
        .map(|(at_ns, _)| *at_ns)
        .collect();
    assert!(!idle.is_empty(), "the synthetic run must go idle");
    for at_ns in idle {
        assert_eq!(
            at_ns % IDLE_STEP_NANOS,
            0,
            "idle step at {at_ns} is off the {IDLE_STEP_NANOS} ns cadence"
        );
        assert!(
            !busy
                .iter()
                .any(|(start, end)| *start <= at_ns && at_ns < *end),
            "idle step at {at_ns} lands inside a busy window {busy:?}"
        );
    }
}

#[test]
fn synthetic_sessions_interleave() {
    let replay = TraceReplay::from_trace(&synthetic(9, 4, 6)).expect("replays");
    let admits = admits(replay.events());
    assert!(admits
        .windows(2)
        .any(|pair| pair[0].0 >> 32 != pair[1].0 >> 32));
}

#[test]
fn synthetic_names_its_mode_after_its_session_count() {
    assert_eq!(synthetic(1, 1, 2).mode, "solo");
    assert_eq!(synthetic(1, 4, 2).mode, "c4");
}

// ---------------------------------------------------------------- virtual clock

#[test]
fn virtual_clock_advances_only_when_told() {
    let mut clock = VirtualClock::new();
    assert_eq!(clock.now_ns(), 0);
    clock.advance(NANOS_PER_SECOND);
    assert_eq!(clock.now_ns(), NANOS_PER_SECOND);
    clock.advance_to(0);
    assert_eq!(
        clock.now_ns(),
        NANOS_PER_SECOND,
        "the clock never runs backwards"
    );
    clock.advance_to(3 * NANOS_PER_SECOND);
    assert_eq!(clock.now_ns(), 3 * NANOS_PER_SECOND);
}

#[test]
fn virtual_clock_drives_a_replay_to_the_same_times() {
    let replay = replay(SOLO);
    let mut clock = VirtualClock::new();
    for event in replay.events() {
        let at_ns = match event {
            RetentionEvent::Admit { at_ns, .. }
            | RetentionEvent::PrefillDone { at_ns, .. }
            | RetentionEvent::RetainPrompt { at_ns, .. }
            | RetentionEvent::RetireTurn { at_ns, .. }
            | RetentionEvent::Step { at_ns, .. }
            | RetentionEvent::PoolPressure { at_ns, .. }
            | RetentionEvent::Cancel { at_ns, .. } => *at_ns,
        };
        clock.advance_to(at_ns);
        assert_eq!(clock.now_ns(), at_ns);
    }
}

// ---------------------------------------------------------------- interleaver

#[test]
fn interleaver_runs_every_task_to_completion() {
    let mut interleaver = Interleaver::new(yielders(&[3, 0, 7, 1]), 4);
    let schedule = interleaver.run();
    assert_eq!(interleaver.remaining(), 0);
    // Each task costs one step per yield plus one final step that reports it finished.
    assert_eq!(schedule.len(), 4 + 1 + 8 + 2);
    assert_eq!(interleaver.schedule(), schedule.as_slice());
}

#[test]
fn interleaver_replays_a_schedule_bit_for_bit() {
    let schedule = Interleaver::new(yielders(&[5, 2, 9, 0]), 17).run();
    let mut replay = Interleaver::new(yielders(&[5, 2, 9, 0]), 0);
    replay
        .replay(&schedule)
        .expect("the recorded schedule replays");
    assert_eq!(replay.schedule(), schedule.as_slice());
}

#[test]
fn interleaver_is_deterministic_per_seed() {
    let first = Interleaver::new(yielders(&[4, 4, 4, 4]), 21).run();
    let second = Interleaver::new(yielders(&[4, 4, 4, 4]), 21).run();
    assert_eq!(first, second);
}

#[test]
fn interleaver_rejects_a_schedule_that_names_an_unknown_task() {
    let mut interleaver = Interleaver::new(yielders(&[1]), 1);
    assert!(matches!(
        interleaver.replay(&[9]),
        Err(ReplayError::UnknownTask { index: 9, .. })
    ));
}

#[test]
fn interleaver_rejects_a_schedule_that_reuses_a_finished_task() {
    let mut interleaver = Interleaver::new(yielders(&[0]), 1);
    assert!(matches!(
        interleaver.replay(&[0, 0]),
        Err(ReplayError::TaskFinished { step: 1, .. })
    ));
}

#[test]
fn interleaver_rejects_a_schedule_that_stops_early() {
    let mut interleaver = Interleaver::new(yielders(&[3]), 1);
    assert!(matches!(
        interleaver.replay(&[0]),
        Err(ReplayError::Incomplete { unfinished: 1, .. })
    ));
}

#[test]
fn interleaver_exposes_its_task_names() {
    let interleaver = Interleaver::new(yielders(&[1, 1]), 1);
    assert_eq!(interleaver.len(), 2);
    assert_eq!(interleaver.name(0), Some("t0"));
    assert_eq!(interleaver.name(2), None);
}

#[test]
fn explore_reports_the_first_failing_seed_with_its_schedule() {
    let failure = explore(
        0..64,
        || yielders(&[2, 3]),
        |seed, _schedule| {
            if seed == 13 {
                Err(format!("seed {seed} lost an update"))
            } else {
                Ok(())
            }
        },
    )
    .expect("seed 13 fails");
    assert_eq!(failure.seed, 13);
    assert!(failure.message.contains("lost an update"));
    assert!(!failure.schedule.is_empty());
}

#[test]
fn explore_returns_none_when_no_seed_fails() {
    assert!(explore(0..64, || yielders(&[2, 3]), |_, _| Ok(())).is_none());
}

#[test]
fn xorshift_is_deterministic_and_never_zero() {
    let mut first = XorShift64::new(99);
    let mut second = XorShift64::new(99);
    for _ in 0..1_000 {
        let value = first.next_u64();
        assert_eq!(value, second.next_u64());
        assert_ne!(value, 0);
    }
}

#[test]
fn shared_hands_out_handles_to_one_cell() {
    let shared = Shared::new(0u64);
    let other = shared.handle();
    *other.borrow_mut() += 3;
    shared.yield_point();
    assert_eq!(shared.get(), 3);
}

// ---------------------------------------------------------------- counters

#[test]
fn counters_merge_field_by_field() {
    let mut total = Counters {
        stores: 1,
        store_bytes: 100,
        ..Counters::default()
    };
    total.merge(&Counters {
        stores: 2,
        restores: 5,
        ..Counters::default()
    });
    total.merge(&Counters {
        store_bytes: 50,
        restores: 1,
        ..Counters::default()
    });
    assert_eq!(total.stores, 3);
    assert_eq!(total.store_bytes, 150);
    assert_eq!(total.restores, 6);
}

#[test]
fn counters_delta_is_the_inverse_of_merge() {
    let before = Counters {
        lookups: 10,
        lookup_hits: 7,
        ..Counters::default()
    };
    let mut after = before;
    after.merge(&Counters {
        lookups: 4,
        lookup_hits: 3,
        ..Counters::default()
    });
    assert_eq!(
        after.delta(&before),
        Some(Counters {
            lookups: 4,
            lookup_hits: 3,
            ..Counters::default()
        })
    );
}

#[test]
fn counters_delta_rejects_a_regression() {
    let before = Counters {
        stores: 5,
        ..Counters::default()
    };
    let after = Counters {
        stores: 4,
        ..Counters::default()
    };
    assert_eq!(after.delta(&before), None);
}

#[test]
fn counters_address_one_field_by_name() {
    let mut counters = Counters::default();
    counters.add(Counter::EvictionsDisk, 2);
    counters.add(Counter::EvictionsDisk, 3);
    assert_eq!(counters.get(Counter::EvictionsDisk), 5);
    assert_eq!(counters.get(Counter::Stores), 0);
}

#[test]
fn counter_order_matches_the_named_fields() {
    // A swap of two same-typed fields would keep `get`/`add` self-consistent but break the mapping
    // to the named struct fields, so check both directions against the field names.
    for &counter in &ALL_COUNTERS {
        let mut by_name = Counters::default();
        by_name.add(counter, 1);
        assert_eq!(
            counter_field(&by_name, counter),
            1,
            "add({counter:?}) wrote the wrong field"
        );
        for &other in &ALL_COUNTERS {
            if other != counter {
                assert_eq!(
                    counter_field(&by_name, other),
                    0,
                    "add({counter:?}) also wrote {other:?}"
                );
            }
        }

        let mut by_field = Counters::default();
        *counter_field_mut(&mut by_field, counter) = 1;
        assert_eq!(
            by_field.get(counter),
            1,
            "get({counter:?}) read the wrong field"
        );
    }
}

#[test]
fn metered_counts_around_a_closure() {
    let counters = Cell::new(Counters::default());
    let metered = Metered::new(&counters, Counter::Restores);
    assert_eq!(metered.count(|| 7), 7);
    metered.count_by(4, || ());
    assert_eq!(counters.get().restores, 5);
}

#[test]
fn metered_counts_a_closure_that_fails() {
    let counters = Cell::new(Counters::default());
    let metered = Metered::new(&counters, Counter::ChecksumFailures);
    let outcome: Result<(), &str> = metered.count(|| Err("torn page"));
    assert!(outcome.is_err());
    assert_eq!(counters.get().checksum_failures, 1);
}

// ---------------------------------------------------------------- performance floors

/// Replaying a trace must stay far above this rate; the floor only catches a broken fast path.
const MIN_REPLAY_EVENTS_PER_SECOND: f64 = 100_000.0;
/// The interleaver must schedule far more than this many steps per second.
const MIN_INTERLEAVER_STEPS_PER_SECOND: f64 = 100_000.0;

#[test]
fn replay_throughput_clears_its_floor() {
    let trace = synthetic(1, 4, 12);
    let events = TraceReplay::from_trace(&trace).expect("replays").len();
    let iterations = 200;
    let started = std::time::Instant::now();
    for _ in 0..iterations {
        let replay = TraceReplay::from_trace(&trace).expect("replays");
        assert_eq!(replay.len(), events);
    }
    let elapsed = started.elapsed().as_secs_f64();
    let rate = (iterations * events) as f64 / elapsed;
    assert!(
        rate >= MIN_REPLAY_EVENTS_PER_SECOND,
        "replay ran at {rate:.0} events/s, below the {MIN_REPLAY_EVENTS_PER_SECOND:.0} floor"
    );
}

#[test]
fn interleaver_throughput_clears_its_floor() {
    let iterations = 200;
    let mut steps = 0usize;
    let started = std::time::Instant::now();
    for seed in 0..iterations {
        steps += Interleaver::new(yielders(&[8; PRODUCTION_TASKS]), seed as u64)
            .run()
            .len();
    }
    let elapsed = started.elapsed().as_secs_f64();
    let rate = steps as f64 / elapsed;
    assert!(
        rate >= MIN_INTERLEAVER_STEPS_PER_SECOND,
        "the interleaver ran at {rate:.0} steps/s, below the {MIN_INTERLEAVER_STEPS_PER_SECOND:.0} floor"
    );
}

// ---------------------------------------------------------------- properties

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Merging counters is associative and commutative, and `default` is its identity.
    #[test]
    fn prop_merge_is_a_commutative_monoid(
        a in any::<[u16; 14]>(),
        b in any::<[u16; 14]>(),
        c in any::<[u16; 14]>(),
    ) {
        let (a, b, c) = (counters_of(a), counters_of(b), counters_of(c));
        let mut left = a;
        left.merge(&b);
        left.merge(&c);
        let mut right = c;
        right.merge(&b);
        right.merge(&a);
        prop_assert_eq!(left, right);

        let mut identity = a;
        identity.merge(&Counters::default());
        prop_assert_eq!(identity, a);
    }

    /// `delta` inverts `merge` exactly when no counter regressed.
    #[test]
    fn prop_delta_inverts_merge(base in any::<[u16; 14]>(), added in any::<[u16; 14]>()) {
        let base = counters_of(base);
        let added = counters_of(added);
        let mut after = base;
        after.merge(&added);
        prop_assert_eq!(after.delta(&base), Some(added));
    }

    /// A counter that went backwards has no meaningful delta.
    #[test]
    fn prop_delta_rejects_a_regression(before in 1u64..u64::MAX, drop in 1u64..u64::MAX) {
        let earlier = Counters { stores: before, ..Counters::default() };
        let now = Counters { stores: before.saturating_sub(drop), ..Counters::default() };
        prop_assert_eq!(now.delta(&earlier), None);
    }

    /// Every named counter addresses a distinct field.
    #[test]
    fn prop_named_counters_are_distinct(index in 0usize..14, amount in 1u64..1_000) {
        let counter = counter_at(index);
        let mut counters = Counters::default();
        counters.add(counter, amount);
        prop_assert_eq!(counters.get(counter), amount);
        for other in 0..14 {
            let other = counter_at(other);
            if other != counter {
                prop_assert_eq!(counters.get(other), 0, "counter {:?} shares a field", other);
            }
        }
    }

    /// A replayed schedule reproduces the run bit-for-bit, for any task shape and seed.
    #[test]
    fn prop_replay_reproduces_the_schedule(
        yields in prop::collection::vec(0usize..8, 1..12),
        seed in any::<u64>(),
    ) {
        let schedule = Interleaver::new(yielders(&yields), seed).run();
        let mut replay = Interleaver::new(yielders(&yields), 0);
        prop_assert!(replay.replay(&schedule).is_ok());
        prop_assert_eq!(replay.schedule(), schedule.as_slice());
    }

    /// The same seed always produces the same schedule.
    #[test]
    fn prop_schedules_are_a_function_of_the_seed(
        yields in prop::collection::vec(0usize..6, 1..10),
        seed in any::<u64>(),
    ) {
        let first = Interleaver::new(yielders(&yields), seed).run();
        let second = Interleaver::new(yielders(&yields), seed).run();
        prop_assert_eq!(first, second);
    }

    /// A schedule never names a task that has already finished: replaying it must succeed.
    #[test]
    fn prop_schedules_never_reuse_a_finished_task(
        yields in prop::collection::vec(0usize..6, 1..10),
        seed in any::<u64>(),
    ) {
        let schedule = Interleaver::new(yielders(&yields), seed).run();
        let mut replay = Interleaver::new(yielders(&yields), 0);
        prop_assert!(replay.replay(&schedule).is_ok(), "schedule {:?} is not a legal run", schedule);
    }

    /// Synthetic traces are a pure function of their seed and shape.
    #[test]
    fn prop_synthetic_is_deterministic(
        seed in any::<u64>(),
        sessions in 1usize..6,
        turns in 1usize..8,
    ) {
        let first = synthetic(seed, sessions, turns);
        let second = synthetic(seed, sessions, turns);
        prop_assert_eq!(first.runs.len(), sessions);
        prop_assert!(first.runs.iter().all(|session| session.len() == turns));
        let first = TraceReplay::from_trace(&first).expect("replays");
        let second = TraceReplay::from_trace(&second).expect("replays");
        prop_assert_eq!(first.events(), second.events());
    }

    /// Every replay is ordered by time, whatever the trace.
    #[test]
    fn prop_replays_are_time_ordered(
        seed in any::<u64>(),
        sessions in 1usize..5,
        turns in 1usize..6,
    ) {
        let replay = TraceReplay::from_trace(&synthetic(seed, sessions, turns)).expect("replays");
        let times: Vec<u64> = replay.events().iter().map(event_time).collect();
        prop_assert!(times.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    /// A turn's prefill never precedes its admission.
    #[test]
    fn prop_prefill_follows_admission(
        seed in any::<u64>(),
        sessions in 1usize..5,
        turns in 1usize..6,
    ) {
        let replay = TraceReplay::from_trace(&synthetic(seed, sessions, turns)).expect("replays");
        let mut admitted: std::collections::BTreeMap<u64, u64> = std::collections::BTreeMap::new();
        for event in replay.events() {
            match event {
                RetentionEvent::Admit { request, at_ns, .. } => {
                    admitted.insert(*request, *at_ns);
                }
                RetentionEvent::PrefillDone { request, at_ns } => {
                    prop_assert!(admitted.get(request).is_some_and(|start| start <= at_ns));
                }
                _ => {}
            }
        }
    }
}

/// Build a `Counters` from fourteen raw values, in `Counter` order.
fn counters_of(values: [u16; 14]) -> Counters {
    let mut counters = Counters::default();
    for (index, value) in values.into_iter().enumerate() {
        counters.add(counter_at(index), u64::from(value));
    }
    counters
}

/// Every counter, in `Counter` order.
const ALL_COUNTERS: [Counter; 14] = [
    Counter::Stores,
    Counter::StoreBytes,
    Counter::StoreDeclinedPressure,
    Counter::StoreDeclinedGate,
    Counter::Restores,
    Counter::RestoreBytes,
    Counter::RestoreCancelled,
    Counter::Lookups,
    Counter::LookupHits,
    Counter::LookupMissesFast,
    Counter::IdleFlushCandidates,
    Counter::IdleFlushEnqueued,
    Counter::EvictionsDisk,
    Counter::ChecksumFailures,
];

/// The `index`-th counter, in `Counter` order.
fn counter_at(index: usize) -> Counter {
    ALL_COUNTERS[index]
}

/// The struct field `counter` names.
fn counter_field(counters: &Counters, counter: Counter) -> u64 {
    match counter {
        Counter::Stores => counters.stores,
        Counter::StoreBytes => counters.store_bytes,
        Counter::StoreDeclinedPressure => counters.store_declined_pressure,
        Counter::StoreDeclinedGate => counters.store_declined_gate,
        Counter::Restores => counters.restores,
        Counter::RestoreBytes => counters.restore_bytes,
        Counter::RestoreCancelled => counters.restore_cancelled,
        Counter::Lookups => counters.lookups,
        Counter::LookupHits => counters.lookup_hits,
        Counter::LookupMissesFast => counters.lookup_misses_fast,
        Counter::IdleFlushCandidates => counters.idle_flush_candidates,
        Counter::IdleFlushEnqueued => counters.idle_flush_enqueued,
        Counter::EvictionsDisk => counters.evictions_disk,
        Counter::ChecksumFailures => counters.checksum_failures,
    }
}

/// A mutable reference to the struct field `counter` names.
fn counter_field_mut(counters: &mut Counters, counter: Counter) -> &mut u64 {
    match counter {
        Counter::Stores => &mut counters.stores,
        Counter::StoreBytes => &mut counters.store_bytes,
        Counter::StoreDeclinedPressure => &mut counters.store_declined_pressure,
        Counter::StoreDeclinedGate => &mut counters.store_declined_gate,
        Counter::Restores => &mut counters.restores,
        Counter::RestoreBytes => &mut counters.restore_bytes,
        Counter::RestoreCancelled => &mut counters.restore_cancelled,
        Counter::Lookups => &mut counters.lookups,
        Counter::LookupHits => &mut counters.lookup_hits,
        Counter::LookupMissesFast => &mut counters.lookup_misses_fast,
        Counter::IdleFlushCandidates => &mut counters.idle_flush_candidates,
        Counter::IdleFlushEnqueued => &mut counters.idle_flush_enqueued,
        Counter::EvictionsDisk => &mut counters.evictions_disk,
        Counter::ChecksumFailures => &mut counters.checksum_failures,
    }
}

/// The time an event happened.
fn event_time(event: &RetentionEvent) -> u64 {
    match event {
        RetentionEvent::Admit { at_ns, .. }
        | RetentionEvent::PrefillDone { at_ns, .. }
        | RetentionEvent::RetainPrompt { at_ns, .. }
        | RetentionEvent::RetireTurn { at_ns, .. }
        | RetentionEvent::Step { at_ns, .. }
        | RetentionEvent::PoolPressure { at_ns, .. }
        | RetentionEvent::Cancel { at_ns, .. } => *at_ns,
    }
}
