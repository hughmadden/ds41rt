//! EXPERIMENTAL ratio-two small-row projection geometry policy (default OFF).
//!
//! Opt-in process configuration `DS41RT_COMPRESSOR_PAD_ROWS`: unset or `0` keeps
//! every historical allocation size and native call shape; `2` or `16` pads the
//! ratio-two WKV/Wgate projections of live batches at or below the configured row
//! count up to that many rows. Scope is strictly the ratio-two compressor input /
//! projected / scores buffers and the WKV/Wgate launch rows; the ratio-one path,
//! index projections, packing, pooling, descriptors and commit row counts are
//! untouched, and live rows above the pad fall back to the old geometry. Logical
//! capacity and all downstream row counts remain the live row count.
//!
//! The policy is captured once per wave from the process environment and then
//! carried explicitly; the hot enqueue path never reads the environment, never
//! allocates, and never synchronizes. The per-enqueue tail memset is recorded on
//! the wave's serialized stream, so it is legal inside CUDA graph capture.
use anyhow::Result;

/// Process configuration knob for the experimental pad-rows policy.
pub(crate) const PAD_ROWS_ENV: &str = "DS41RT_COMPRESSOR_PAD_ROWS";
/// BF16 elements per projected input row: [5120] at two bytes.
pub(crate) const INPUT_ROW_BYTES: usize = 10240;
/// Exact per-row bytes added when a ratio-two allocation is padded:
/// input + projected + scores. Every other buffer keeps the logical capacity.
pub(crate) const PADDED_ROW_BYTES: usize = INPUT_ROW_BYTES + 2048 + 2048;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PadRowsPolicy {
    Off,
    Pad(usize),
}

impl PadRowsPolicy {
    /// Capture the process policy once (per wave or per planning pass). An unset
    /// or `0` value is off; only `2` and `16` are accepted. An empty,
    /// whitespace-only, other or non-Unicode value is INVALID configuration and
    /// is rejected here, before any wave allocation happens — it never silently
    /// selects the off policy.
    pub(crate) fn from_env() -> Result<Self> {
        match std::env::var(PAD_ROWS_ENV) {
            Ok(value) => Self::parse(Some(&value)),
            Err(std::env::VarError::NotPresent) => Ok(Self::Off),
            Err(std::env::VarError::NotUnicode(_)) => anyhow::bail!(
                "invalid {PAD_ROWS_ENV}: value is not valid Unicode; \
                 allowed values are unset, 0, 2 or 16"
            ),
        }
    }

    fn parse(value: Option<&str>) -> Result<Self> {
        match value.map(str::trim) {
            None | Some("0") => Ok(Self::Off),
            Some("2") => Ok(Self::Pad(2)),
            Some("16") => Ok(Self::Pad(16)),
            Some("") => anyhow::bail!(
                "invalid {PAD_ROWS_ENV}: value is empty; \
                 allowed values are unset, 0, 2 or 16"
            ),
            Some(other) => anyhow::bail!(
                "invalid {PAD_ROWS_ENV}={other:?}: allowed values are unset, 0, 2 or 16"
            ),
        }
    }

    pub(crate) fn enabled(self) -> bool {
        !matches!(self, Self::Off)
    }

    fn pad(self) -> Option<usize> {
        match self {
            Self::Off => None,
            Self::Pad(rows) => Some(rows),
        }
    }

    /// Physical allocation rows for the input / projected / scores buffers.
    /// Ratio-one waves and the off policy keep the exact logical capacity.
    pub(crate) fn allocation_rows(self, ratio: usize, capacity: usize) -> usize {
        match self.pad().filter(|_| ratio == 2) {
            Some(pad) => capacity.max(pad),
            None => capacity,
        }
    }

    /// Extra device bytes the policy adds to the historical wave size:
    /// exactly `extra_rows * (10240 + 2048 + 2048)`, ratio two only.
    pub(crate) fn extra_bytes(self, ratio: usize, capacity: usize) -> usize {
        (self.allocation_rows(ratio, capacity) - capacity) * PADDED_ROW_BYTES
    }

    /// Launch geometry for the WKV/Wgate projections of a live batch.
    /// `zero` is the exact byte range `[offset, offset + length)` of the input
    /// tail to zero on the serialized stream before the projections; `rows` is
    /// the GEMM row count handed to the existing compressor project export.
    pub(crate) fn projection(self, ratio: usize, live_rows: usize) -> PadProjection {
        let Some(pad) = self.pad().filter(|&pad| ratio == 2 && live_rows < pad) else {
            return PadProjection {
                zero: None,
                rows: live_rows,
            };
        };
        PadProjection {
            zero: Some((
                live_rows * INPUT_ROW_BYTES,
                (pad - live_rows) * INPUT_ROW_BYTES,
            )),
            rows: pad,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PadProjection {
    /// Tail byte range of the input to zero, if any.
    pub(crate) zero: Option<(usize, usize)>,
    /// Row count for the WKV/Wgate projection launches.
    pub(crate) rows: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATIO2_LAYER: usize = 2;
    const RATIO1_LAYER: usize = 20;

    #[test]
    fn config_parses_off_by_default_and_accepts_only_2_or_16() {
        assert_eq!(PadRowsPolicy::parse(None).unwrap(), PadRowsPolicy::Off);
        assert_eq!(PadRowsPolicy::parse(Some("0")).unwrap(), PadRowsPolicy::Off);
        assert_eq!(PadRowsPolicy::parse(Some("2")).unwrap(), PadRowsPolicy::Pad(2));
        assert_eq!(PadRowsPolicy::parse(Some("16")).unwrap(), PadRowsPolicy::Pad(16));
        // Surrounding whitespace is trimmed away.
        assert_eq!(PadRowsPolicy::parse(Some(" 2 ")).unwrap(), PadRowsPolicy::Pad(2));
        assert_eq!(PadRowsPolicy::parse(Some(" 0 ")).unwrap(), PadRowsPolicy::Off);
        for invalid in ["", " ", "  ", "1", "3", "15", "17", "02", "-2", "abc", "4096"] {
            let error = PadRowsPolicy::parse(Some(invalid)).unwrap_err();
            assert!(
                error.to_string().contains(PAD_ROWS_ENV),
                "{invalid:?}: unexpected error: {error:#}"
            );
        }
    }

    /// Run `PadRowsPolicy::from_env` in an isolated child process with an exact
    /// `DS41RT_COMPRESSOR_PAD_ROWS` value, so the parent test process never
    /// mutates its own environment while tests run in parallel. The child is the
    /// test binary itself, gated by `PAD_ROWS_CHILD` and filtered to this test.
    fn from_env_in_child(value: Option<&std::ffi::OsStr>) -> String {
        const MARKER: &str = "DS41RT_PAD_ROWS_POLICY_CHILD";
        if let Ok(mode) = std::env::var(MARKER) {
            // Child: report the captured policy and exit before any other tests.
            assert_eq!(mode, "1");
            let report = match PadRowsPolicy::from_env() {
                Ok(PadRowsPolicy::Off) => "ok-off".to_string(),
                Ok(PadRowsPolicy::Pad(rows)) => format!("ok-{rows}"),
                Err(error) => format!("err-{}", error.to_string().contains(PAD_ROWS_ENV)),
            };
            println!("{report}");
            std::process::exit(0);
        }
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .arg("--exact")
            .arg("v41_compressor::pad_rows::tests::from_env_rejects_invalid_environment_values")
            .arg("--nocapture")
            .env(MARKER, "1")
            .env_remove(PAD_ROWS_ENV);
        if let Some(value) = value {
            command.env(PAD_ROWS_ENV, value);
        }
        let output = command.output().expect("spawn isolated child test binary");
        assert!(
            output.status.success(),
            "isolated child failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .find(|line| line.starts_with("ok-") || line.starts_with("err-"))
            .expect("child must report the captured policy")
            .to_string()
    }

    #[test]
    fn from_env_rejects_invalid_environment_values() {
        use std::os::unix::ffi::OsStringExt;
        // Unset stays off; a non-Unicode, empty or whitespace-only value is
        // invalid configuration, never a silent off.
        assert_eq!(from_env_in_child(None), "ok-off");
        assert_eq!(
            from_env_in_child(Some(std::ffi::OsStr::new(""))),
            "err-true"
        );
        assert_eq!(from_env_in_child(Some(std::ffi::OsStr::new("  "))), "err-true");
        assert_eq!(
            from_env_in_child(Some(std::ffi::OsStr::new("2"))),
            "ok-2"
        );
        // Invalid UTF-8 bytes (no interior NUL: execve values are C strings).
        let non_unicode = std::ffi::OsString::from_vec(vec![0xff, 0xfe, 0x01]);
        assert_eq!(
            from_env_in_child(Some(non_unicode.as_os_str())),
            "err-true"
        );
    }

    #[test]
    fn off_policy_preserves_exact_old_allocation_and_projection_geometry() {
        for capacity in [1, 2, 3, 15, 16, 17, 4096] {
            for ratio in [1, 2] {
                let policy = PadRowsPolicy::Off;
                assert_eq!(policy.allocation_rows(ratio, capacity), capacity);
                assert_eq!(policy.extra_bytes(ratio, capacity), 0);
                for live in [1, 2, 3, 15, 16, 17] {
                    assert_eq!(
                        policy.projection(ratio, live),
                        PadProjection {
                            zero: None,
                            rows: live
                        }
                    );
                }
            }
        }
    }

    #[test]
    fn ratio_one_is_never_padded_even_with_the_policy_on() {
        for pad in [PadRowsPolicy::Pad(2), PadRowsPolicy::Pad(16)] {
            for capacity in 1..=4096usize {
                assert_eq!(pad.allocation_rows(1, capacity), capacity);
                assert_eq!(pad.extra_bytes(1, capacity), 0);
            }
            for live in [1, 2, 3, 15, 16, 17] {
                assert_eq!(
                    pad.projection(1, live),
                    PadProjection {
                        zero: None,
                        rows: live
                    }
                );
            }
        }
    }

    #[test]
    fn allocation_rows_grow_only_when_capacity_is_below_the_pad() {
        for pad_rows in [2usize, 16] {
            let policy = PadRowsPolicy::Pad(pad_rows);
            for capacity in 1..=4096usize {
                let expected = if capacity < pad_rows { pad_rows } else { capacity };
                assert_eq!(policy.allocation_rows(2, capacity), expected, "capacity {capacity}");
            }
        }
    }

    #[test]
    fn extra_bytes_are_exactly_padded_rows_times_projection_row_bytes() {
        for pad_rows in [2usize, 16] {
            let policy = PadRowsPolicy::Pad(pad_rows);
            for capacity in 1..=4096usize {
                let extra_rows = capacity.max(pad_rows) - capacity;
                let expected = extra_rows * (10240 + 2048 + 2048);
                assert_eq!(policy.extra_bytes(2, capacity), expected, "capacity {capacity}");
                assert_eq!(expected % PADDED_ROW_BYTES, 0);
            }
        }
    }

    #[test]
    fn projection_geometry_covers_boundary_live_rows_for_both_pads() {
        for (pad_rows, boundaries) in [
            (2usize, &[1usize, 2, 3][..]),
            (16, &[1, 2, 3, 15, 16, 17][..]),
        ] {
            let policy = PadRowsPolicy::Pad(pad_rows);
            for &live in boundaries {
                let geometry = policy.projection(2, live);
                if live < pad_rows {
                    assert_eq!(
                        geometry,
                        PadProjection {
                            zero: Some((live * 10240, (pad_rows - live) * 10240)),
                            rows: pad_rows
                        },
                        "pad {pad_rows} live {live}"
                    );
                } else {
                    assert_eq!(
                        geometry,
                        PadProjection {
                            zero: None,
                            rows: live
                        },
                        "pad {pad_rows} live {live}"
                    );
                }
            }
        }
    }

    #[test]
    fn tail_zero_range_targets_only_the_stale_tail_bytes() {
        // live=1 pad=2: zero [10240, 20480); live=3 pad=16: zero [30720, 163840).
        assert_eq!(
            PadRowsPolicy::Pad(2).projection(2, 1).zero,
            Some((10240, 10240))
        );
        assert_eq!(
            PadRowsPolicy::Pad(16).projection(2, 3).zero,
            Some((3 * 10240, 13 * 10240))
        );
        // Equal rows: no memset, GEMM still at the pad count.
        assert_eq!(
            PadRowsPolicy::Pad(2).projection(2, 2),
            PadProjection { zero: None, rows: 2 }
        );
        assert_eq!(
            PadRowsPolicy::Pad(16).projection(2, 16),
            PadProjection { zero: None, rows: 16 }
        );
    }

    #[test]
    fn smaller_live_rows_after_bigger_rows_rezero_the_stale_tail() {
        // A wave whose previous enqueue used more rows must compute the zero
        // range from the current live rows only; the prefix is never touched.
        let policy = PadRowsPolicy::Pad(16);
        let previous = policy.projection(2, 12);
        assert_eq!(previous.zero, Some((12 * 10240, 4 * 10240)));
        let current = policy.projection(2, 1);
        assert_eq!(current.zero, Some((10240, 15 * 10240)));
        assert_eq!(current.rows, 16);
    }

    #[test]
    fn wave_device_bytes_matches_historical_formula_plus_exact_extra() {
        for pad in [PadRowsPolicy::Off, PadRowsPolicy::Pad(2), PadRowsPolicy::Pad(16)] {
            for capacity in 1..=4096usize {
                for (layer, ratio, per_row) in [
                    (RATIO2_LAYER, 2, 10240 + 2048 + 2048 + 8 + 1024 + 264 + 512 + 68 + 8 + 288),
                    (RATIO1_LAYER, 1, 10240 + 1024 + 1024 + 264 + 512 + 68 + 8 + 288),
                ] {
                    let bytes =
                        super::super::CompressorWave::device_bytes(layer, capacity, pad).unwrap();
                    let expected = 4 * 1024 * 1024 + capacity * per_row
                        + if ratio == 2 { pad.extra_bytes(2, capacity) } else { 0 };
                    assert_eq!(bytes, expected, "layer {layer} capacity {capacity} pad {pad:?}");
                }
            }
        }
    }

    #[test]
    fn wave_device_bytes_rejects_bad_rows_and_bad_layers() {
        for rows in [0, 4097] {
            for pad in [PadRowsPolicy::Off, PadRowsPolicy::Pad(2)] {
                super::super::CompressorWave::device_bytes(RATIO2_LAYER, rows, pad)
                    .expect_err("invalid capacity must be rejected");
            }
        }
        super::super::CompressorWave::device_bytes(3, 16, PadRowsPolicy::Pad(2))
            .expect_err("non-compressor layer must be rejected");
    }
}
