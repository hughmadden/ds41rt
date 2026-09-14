//! Placeholder: the packet that owns this bench replaces it with named criterion groups.
use criterion::{criterion_group, criterion_main, Criterion};

fn placeholder(_c: &mut Criterion) {}

criterion_group!(benches, placeholder);
criterion_main!(benches);
