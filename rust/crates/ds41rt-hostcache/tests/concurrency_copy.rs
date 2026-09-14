//! Concurrency (interleaving) suite for the copy engine (packet HC-3). The crate is
//! single-threaded by design, so "concurrency" here means two lanes, one per stream, driven in a
//! seeded order that mixes issues, records, waits and advances. A shadow model checks the
//! engine's observable state after every step, and the schedule is logged so a failure
//! reproduces from its seed.
mod common;

use common::{Rng, Shadow};
use ds41rt_hostcache::copy::{CopyEngine, CopyModel, DeviceRange, Event, Stream, StubCopyEngine};
use ds41rt_hostcache::pool::{HostRange, PinnedMemory};

const DEVICE_BYTES: usize = 1 << 12;
const HOST_BYTES: usize = 1 << 12;
const STRIDE: usize = 64;
const SLOTS: usize = DEVICE_BYTES / STRIDE;
const STEPS: usize = 120;

/// Drive one seeded schedule and return its log. Panics with the log if any invariant breaks.
fn run_schedule(seed: u64) -> Vec<String> {
    let model = CopyModel::default();
    let mut engine = StubCopyEngine::new(model, DEVICE_BYTES, HOST_BYTES);
    let chunk = engine.allocate_chunk(HOST_BYTES).unwrap();
    let mut shadow = Shadow::new(model, DEVICE_BYTES, HOST_BYTES);
    let mut rng = Rng::new(seed);
    let mut log = Vec::new();
    let mut events: Vec<(Stream, Event)> = Vec::new();
    let mut slot = 0usize;

    for step in 0..STEPS {
        let lane = if rng.below(2) == 0 {
            Stream::Store
        } else {
            Stream::Restore
        };
        match rng.below(5) {
            0 => {
                let bytes = 1 + rng.below(STRIDE as u64) as usize;
                let addr = (slot % SLOTS) * STRIDE;
                slot += 1;
                if lane == Stream::Store {
                    let pattern: Vec<u8> = (0..bytes).map(|i| (i as u8) ^ (step as u8)).collect();
                    engine.write_device(
                        DeviceRange {
                            addr: addr as u64,
                            bytes,
                        },
                        &pattern,
                    );
                    shadow.write_device(addr, &pattern);
                    engine
                        .d2h(
                            lane,
                            DeviceRange {
                                addr: addr as u64,
                                bytes,
                            },
                            HostRange {
                                chunk: chunk.id,
                                offset: addr,
                                bytes,
                            },
                        )
                        .unwrap();
                    shadow.issue(lane, true, addr, addr, bytes);
                } else {
                    engine
                        .h2d(
                            lane,
                            HostRange {
                                chunk: chunk.id,
                                offset: addr,
                                bytes,
                            },
                            DeviceRange {
                                addr: addr as u64,
                                bytes,
                            },
                        )
                        .unwrap();
                    shadow.issue(lane, false, addr, addr, bytes);
                }
                log.push(format!("{step}: issue {lane:?} {bytes}B at {addr}"));
            }
            1 => {
                let event = engine.record(lane).unwrap();
                let shadow_event = shadow.record(lane);
                assert_eq!(event.0 as usize, shadow_event, "event ids diverged");
                events.push((lane, event));
                log.push(format!("{step}: record {lane:?} -> {event:?}"));
            }
            2 => {
                let nanos = rng.below(20_000);
                engine.advance(nanos);
                shadow.advance(nanos);
                log.push(format!("{step}: advance {nanos}"));
            }
            3 => {
                if let Some((event_lane, event)) = events.last().copied() {
                    let budget = rng.below(20_000);
                    let got = engine.wait(event, budget).unwrap();
                    let want = shadow.wait(event.0 as usize, budget);
                    assert_eq!(got, want, "wait mismatch on {event:?}");
                    log.push(format!(
                        "{step}: wait {event_lane:?} {event:?} budget {budget} -> {got}"
                    ));
                } else {
                    log.push(format!("{step}: wait skipped (no events)"));
                }
            }
            _ => log.push(format!("{step}: now {}", engine.now_ns())),
        }
        check_invariants(&mut engine, &shadow, &events, &log);
    }
    log
}

/// Compare the engine with the shadow and assert the per-stream event prefix invariant.
fn check_invariants(
    engine: &mut StubCopyEngine,
    shadow: &Shadow,
    events: &[(Stream, Event)],
    log: &[String],
) {
    assert_eq!(engine.now_ns(), shadow.now, "clock diverged\n{log:?}");
    for stream in [Stream::Store, Stream::Restore] {
        assert_eq!(
            engine.pending(stream),
            shadow.pending(stream),
            "pending diverged on {stream:?}\n{log:?}"
        );
    }
    let device = engine.read_device(DeviceRange {
        addr: 0,
        bytes: DEVICE_BYTES,
    });
    assert_eq!(device, shadow.device, "device diverged\n{log:?}");
    let host = engine.read_host(HostRange {
        chunk: 0,
        offset: 0,
        bytes: HOST_BYTES,
    });
    assert_eq!(host, shadow.host, "host diverged\n{log:?}");

    for stream in [Stream::Store, Stream::Restore] {
        let mut incomplete = false;
        for (event_stream, event) in events {
            if *event_stream != stream {
                continue;
            }
            if engine.completed(*event).unwrap() {
                assert!(
                    !incomplete,
                    "event {event:?} completed after an incomplete one on {stream:?}\n{log:?}"
                );
            } else {
                incomplete = true;
            }
        }
    }
}

#[test]
fn interleaved_lanes_keep_the_invariants() {
    for seed in 1..=64 {
        let log = run_schedule(seed);
        assert_eq!(log.len(), STEPS);
    }
}

#[test]
fn schedules_reproduce_from_their_seed() {
    assert_eq!(run_schedule(7), run_schedule(7));
}
