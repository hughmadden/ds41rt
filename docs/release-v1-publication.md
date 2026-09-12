# DS41RT v1 publication record

The first DS41RT release is published as [DS41RT v1](https://github.com/tpurtell/ds41rt/releases/tag/v1). The earlier `v3` identity came from continuing two local DS4RT tag numbers and was replaced in place. Legacy history is now explicitly named `ds4rt-v1` and `ds4rt-v2`; there is no DS41RT `v2` or `v3` release.

The release remains final rather than draft or prerelease. It retains GitHub release ID `387678878` and its original publication time, `2026-09-12T19:22:30Z`. Its notes identify the project as the first DS41RT release, summarize the high-level features, link the detailed reports, and cite the exact corrected container digests.

The attached `native-release-v1-build-run.tar.gz` asset is 182,245 bytes with SHA-256 `ce65c575ef311599498eea24c8b47c50547e179da4b46aa9065a3bba262957ea`. GitHub asset ID `560049283` reports the same digest. The obsolete v3-named asset was removed.

Published containers:

- `ghcr.io/tpurtell/ds41rt-coordinator:v1` and `latest`: `sha256:67f2954e18f69b39f8fbb68164f7d9e2b8f4c4b9e3242281ecfcb8afec8552e9`
- `ghcr.io/tpurtell/ds41rt-spark-expert:v1` and `latest`: `sha256:1f1bff295a1d112c8a2fb80918b5abcafcf10eec4936717e234d3d727635a0be`

Raw-manifest comparison confirms `v1` and `latest` are identical for each role. Both images report version `v1` and measured source revision `9ea5c96468da690fe7dd01471d4fa2fb8555a606`. The source tag includes later performance and publication documentation; no inference source changed after the measured image.

The corrected standard build restores 7,743.47 prompt tok/s at the headline prefill cell, 150.51 tok/s warm dSpark counting, and 742.91 aggregate tok/s at C16. The complete three-sample rerun is in the [performance report](release-v1-performance.md). The full qualification suite was not repeated.

`dev`, `main`, `release/v1`, and annotated tag `v1` are advanced to the publication record. The obsolete `release/v3` branch and `v3` tag are absent. The README-backed Pages deployment remains public. Both GHCR packages remain private for the repository owner's requested manual visibility change.
