# DS41RT v3 publication record

DS41RT v3 was published from qualified release commit `681256121deebbd16f5946facd09964278b6b339`. The `v3` annotated tag resolves to that commit. At publication, `dev`, `main`, and `release/v3` also resolved to the same source and documentation tree.

The [GitHub release](https://github.com/tpurtell/ds41rt/releases/tag/v3) is public, final rather than draft or prerelease, and includes the high-level release notes, qualified container references, source archives, and the sanitized qualification evidence asset. GitHub reports release ID `387678878` and publication time `2026-09-12T19:22:30Z`.

| Published artifact | Identity |
|---|---|
| annotated tag object | `2ade3b93dff30aa72813c251d3d3577748f7d032` |
| tag target / release source | `681256121deebbd16f5946facd09964278b6b339` |
| qualification evidence asset | `sha256:43192d5c9d47e7fb23e35ccc9c1c80009aa8a6d2fb7e6db6fc1e45c800859701` |
| coordinator OCI index | `sha256:0d8a29160924dc62694d65f46e5101bf39071fb28e7611344489dde416bfe950` |
| Spark expert manifest | `sha256:672f82a1a99872cdc8014811b99c0967e0955c8c3e1e29b91bd48e7b06d3566d` |

GitHub Pages now serves `main:/docs`. Build `1211219703` completed for release commit `681256121deebbd16f5946facd09964278b6b339` with no error. The [playable Frogger artifact](https://tpurtell.github.io/ds41rt/frogger.html) returns HTTP 200 and exactly matches [`frogger.html`](frogger.html): 15,763 bytes and SHA-256 `187d2e5a22e7477d59752b1e3982b98e6fa0a5df760f4b03cc2acceb8db6c185`.

The standard five-host v3 service remained healthy after publication. `/v1/models` reports only `deepseek-ai/DeepSeek-V4.1-Flash` with the official 1,048,576-token context and 393,216-token output limits. The coordinator and four rank-specific expert containers continue to run from the v3 images.

Both GHCR packages are intentionally still private. The repository owner will perform the requested final manual visibility change:

- [ds41rt-coordinator package settings](https://github.com/users/tpurtell/packages/container/package/ds41rt-coordinator/settings)
- [ds41rt-spark-expert package settings](https://github.com/users/tpurtell/packages/container/package/ds41rt-spark-expert/settings)

[Machine-readable publication record](release-v1-publication.json) preserves the release, tag, branch, Pages, container, evidence, and remaining-owner-action state.
