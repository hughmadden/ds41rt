//! Placeholder: P1-1 replaces this with named criterion groups (lookup ops/s at 10k keys,
//! insert/remove transactions per second, open and rebuild time at 10k objects).
use criterion::{criterion_group, criterion_main, Criterion};

fn placeholder(_c: &mut Criterion) {}

criterion_group!(benches, placeholder);
criterion_main!(benches);
