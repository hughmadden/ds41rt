//! Performance suite for P0-4: trace replay throughput and interleaver step rate.
//!
//! The floors these benches measure are asserted in `tests/functional_p0_4.rs`, so a broken fast
//! path fails `cargo test` rather than waiting for a human to read a criterion report.
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ds41rt_persist::events::{synthetic, TraceReplay};
use ds41rt_persist::sched::Interleaver;
use std::hint::black_box;

#[allow(dead_code)]
#[path = "../tests/common/mod.rs"]
mod common;

use common::{lanes, PRODUCTION_TASKS};

/// Turns per session in the replay bench: the recorded captures hold twelve.
const BENCH_TURNS: usize = 12;

/// Sessions in the replay bench: the recorded `c4` capture holds four.
const BENCH_SESSIONS: usize = 4;

/// Times each lane yields before finishing.
const LANE_YIELDS: usize = 8;

/// Steps one full run costs: every lane's yields plus its finishing step.
const STEPS_PER_RUN: u64 = (PRODUCTION_TASKS * (LANE_YIELDS + 1)) as u64;

/// Replay a synthetic trace of `sessions` × `BENCH_TURNS` turns.
fn bench_replay(c: &mut Criterion) {
    let trace = synthetic(1, BENCH_SESSIONS, BENCH_TURNS);
    let events = TraceReplay::from_trace(&trace)
        .expect("the synthetic trace replays")
        .len();
    let mut group = c.benchmark_group("events/replay");
    group.throughput(Throughput::Elements(events as u64));
    group.bench_function(
        BenchmarkId::new("synthetic", format!("{BENCH_SESSIONS}x{BENCH_TURNS}")),
        |b| {
            b.iter(|| {
                let replay = TraceReplay::from_trace(black_box(&trace)).expect("replays");
                black_box(replay.len())
            })
        },
    );
    group.finish();
}

/// Run `PRODUCTION_TASKS` lanes to completion under a fresh seed each iteration.
fn bench_interleaver(c: &mut Criterion) {
    let mut group = c.benchmark_group("sched/steps");
    group.throughput(Throughput::Elements(STEPS_PER_RUN));
    group.bench_function(BenchmarkId::new("lanes", PRODUCTION_TASKS), |b| {
        let mut seed = 0u64;
        b.iter(|| {
            seed = seed.wrapping_add(1);
            let schedule =
                Interleaver::new(lanes("lane", PRODUCTION_TASKS, LANE_YIELDS), seed).run();
            black_box(schedule.len())
        })
    });
    group.finish();
}

criterion_group!(benches, bench_replay, bench_interleaver);
criterion_main!(benches);
