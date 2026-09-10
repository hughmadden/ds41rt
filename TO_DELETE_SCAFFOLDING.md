# Temporary committed scaffolding

Track temporary code that would otherwise ship in Git, and remove it when production-path or reusable regression coverage replaces it.
Useful small fixtures, sparse checkpoint staging and cached builds outside the repository may stay while they shorten iteration; checkpoint availability alone is not a reason to delete them.
The former external-artifact inventory remains available in Git history.

- [ ] Remove `rust/crates/ds41rt-loader/examples/qualify_v41_mapped_fixture.rs` after real-checkpoint mapped-engram qualification replaces its temporary fixture-generation role.
- [ ] Review model-specific synthetic entry points during integration and remove temporary tracked wrappers once production entry points supply equivalent coverage.
- [ ] Keep reusable numerical qualifiers, production catalog inspection, regression tests and recorded qualification evidence.

Current `/tmp/ds41-*` component fixtures and AOT builds are development caches rather than shipped source, and may be retained until they are no longer useful.
