//! Criterion benches for the simulator (HC-4): scheduler steps per second at eight lanes under
//! the capacity-1024 configuration of record, for the agent-loop and churn workloads. The
//! throughput unit is the scheduler step, the unit of work the interleaver spends.
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ds41rt_hostcache::sim::testing::RecordingCache;
use ds41rt_hostcache::sim::{EngineModel, Simulator, Workload};
use std::hint::black_box;

const SEED: u64 = 0x5EED;

fn steps_of(workload: &Workload) -> u64 {
    Simulator::new(EngineModel::default(), RecordingCache::new(), SEED)
        .run(workload)
        .steps
}

fn bench_steps(c: &mut Criterion) {
    let model = EngineModel::default();
    let workloads = [
        (
            "agent_loop",
            Workload::AgentLoop {
                sessions: 8,
                turns: 12,
                context_tokens: 32_768,
                new_tokens_per_turn: 256,
                think_ns: 0,
            },
        ),
        (
            "churn",
            Workload::Churn {
                sessions: 48,
                turns: 6,
                context_tokens: 8_192,
                live_ratio: 1.0,
            },
        ),
    ];
    let mut group = c.benchmark_group("sim_steps_per_s");
    for (name, workload) in workloads {
        let steps = steps_of(&workload);
        group.throughput(Throughput::Elements(steps));
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &workload,
            |b, workload| {
                b.iter(|| {
                    let mut sim =
                        Simulator::new(model.clone(), RecordingCache::new(), black_box(SEED));
                    black_box(sim.run(black_box(workload)))
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_steps);
criterion_main!(benches);
