//! Startup-only source-pool sizing. No allocation policy runs in the token loop.
use crate::v41_backbone_cache::BackboneCache;
use anyhow::{ensure, Context, Result};
use std::{fmt, str::FromStr};

const RETAINED_CONTEXTS: usize = 8;
// Extra pages cover retained partial tails and active copy-on-write frontiers.
const MAX_GROUPS: usize = 131_072;
const GROUP_BYTES: usize = 5 * 256 * (68 + ds41rt_ffi::V41Kv::COMPRESSED_ROW_BYTES);
// Future retained windows, request scratch and graph/runtime allocations.
// Kept outside the eagerly allocated cache; this is not a CUDA process quota.
pub(super) const RUNTIME_HEADROOM: usize = 2 * 1024 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ByteSize(pub usize);
#[derive(Clone, Copy, Debug)]
pub(crate) enum Reservation {
    Bytes(ByteSize),
    Percent(u64), // millionths of one percent
}
fn decimal(value: &str) -> Result<u64> {
    let (whole, fraction) = value.split_once('.').unwrap_or((value, ""));
    ensure!(
        !whole.is_empty()
            && whole.bytes().all(|c| c.is_ascii_digit())
            && fraction.len() <= 6
            && fraction.bytes().all(|c| c.is_ascii_digit()),
        "expected a positive decimal with at most six fractional digits"
    );
    let whole: u64 = whole.parse()?;
    let fractional: u64 = if fraction.is_empty() {
        0
    } else {
        fraction.parse()?
    };
    let scaled = whole
        .checked_mul(1_000_000)
        .and_then(|v| v.checked_add(fractional * 10u64.pow(6 - fraction.len() as u32)))
        .context("memory size overflow")?;
    ensure!(scaled > 0, "memory size must be positive");
    Ok(scaled)
}
impl FromStr for ByteSize {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        let value = value.trim();
        let end = value
            .find(|c: char| !c.is_ascii_digit() && c != '.')
            .unwrap_or(value.len());
        let amount = decimal(&value[..end])?;
        let unit = value[end..].trim().to_ascii_lowercase();
        let multiplier: u128 = match unit.as_str() {
            "" | "b" => 1,
            "mb" => 1_000_000,
            "gb" => 1_000_000_000,
            "mib" => 1_048_576,
            "gib" => 1_073_741_824,
            _ => anyhow::bail!("memory size unit must be B, MB, GB, MiB or GiB"),
        };
        let bytes = usize::try_from(u128::from(amount) * multiplier / 1_000_000)?;
        ensure!(bytes > 0, "memory size rounds to zero bytes");
        Ok(Self(bytes))
    }
}
impl fmt::Display for ByteSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}B", self.0)
    }
}
impl FromStr for Reservation {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        if let Some(percent) = value.trim().strip_suffix('%') {
            let percent = decimal(percent.trim())?;
            ensure!(
                percent <= 100_000_000,
                "memory percentage must be in (0,100]"
            );
            Ok(Self::Percent(percent))
        } else {
            Ok(Self::Bytes(value.parse()?))
        }
    }
}
impl Reservation {
    fn bytes(self, total: usize) -> Result<usize> {
        let bytes = match self {
            Self::Bytes(size) => size.0,
            Self::Percent(percent) => (total as u128 * u128::from(percent) / 100_000_000) as usize,
        };
        ensure!(
            bytes > 0 && bytes <= total,
            "memory reservation exceeds device total or rounds to zero"
        );
        Ok(bytes)
    }
}

#[derive(Debug)]
pub(super) struct PoolPlan {
    pub pages: [usize; 4],
    pub global_bytes: usize,
    pub cache_bytes: usize,
    pub occupied_before: usize,
    pub reservation_bytes: usize,
}
impl PoolPlan {
    fn from_groups(
        slots: usize,
        groups: usize,
        occupied_before: usize,
        reservation_bytes: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=MAX_GROUPS).contains(&groups),
            "source pool exceeds physical page capacity"
        );
        let pages = [groups, groups, groups, groups * 2];
        Ok(Self {
            pages,
            global_bytes: groups * GROUP_BYTES,
            cache_bytes: BackboneCache::device_bytes(slots, pages)?,
            occupied_before,
            reservation_bytes,
        })
    }
    pub fn new(
        slots: usize,
        context: usize,
        retained_turns: usize,
        exact: Option<ByteSize>,
        reservation: Option<Reservation>,
        free: usize,
        total: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=1_048_576).contains(&context),
            "invalid pool context limit"
        );
        ensure!(
            free <= total && total > 0,
            "invalid device memory information"
        );
        ensure!((1..=16).contains(&slots), "invalid concurrency limit");
        let occupied = total - free;
        let ceiling = reservation
            .map(|r| r.bytes(total))
            .transpose()?
            .unwrap_or(total);
        let available = ceiling.checked_sub(occupied).and_then(|v| v.checked_sub(RUNTIME_HEADROOM))
            .context("memory reservation leaves no space after existing allocations and runtime headroom")?;
        ensure!(retained_turns <= 128, "invalid retained-turn limit");
        // At most two retained frontiers per turn, plus one active tail per slot.
        let spare_groups = slots + 2 * retained_turns;
        let per_context = context.div_ceil(512);
        // Explicit budgets may trade aggregate context capacity for memory.
        // Provision at least one page per active owner plus private tail space.
        let minimum = slots + spare_groups;
        let groups = if let Some(exact) = exact {
            ensure!(
                exact.0 / GROUP_BYTES <= MAX_GROUPS,
                "exact KV pool exceeds physical page capacity"
            );
            exact.0 / GROUP_BYTES
        } else if reservation.is_some() {
            // Tables saturate at the maximum logical per-request context. Find
            // the largest whole page group whose entire cache fits the budget.
            let (mut low, mut high) = (0, MAX_GROUPS);
            while low < high {
                let mid = (low + high).div_ceil(2);
                if Self::from_groups(slots, mid, occupied, ceiling)?.cache_bytes <= available {
                    low = mid;
                } else {
                    high = mid - 1;
                }
            }
            low
        } else {
            per_context * (slots + RETAINED_CONTEXTS) + spare_groups
        };
        ensure!(groups >= minimum,
            "KV pool needs at least {} global bytes for {slots} active owners plus copy-on-write headroom; increase the memory budget",
            minimum * GROUP_BYTES);
        let plan = Self::from_groups(slots, groups, occupied, ceiling)?;
        ensure!(plan.cache_bytes <= available,
            "KV cache needs {} bytes but reservation leaves {available} after existing allocations and {} bytes of runtime headroom",
            plan.cache_bytes, RUNTIME_HEADROOM);
        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sizes_and_reservations_validate_units_and_overflow() {
        for (value, bytes) in [
            ("37GB", 37_000_000_000),
            ("1.5GiB", 1_610_612_736),
            ("1MB", 1_000_000),
            ("1MiB", 1_048_576),
            ("123", 123),
        ] {
            assert_eq!(value.parse::<ByteSize>().unwrap().0, bytes);
        }
        for value in [
            "0",
            "-1GB",
            "NaN",
            "1TB",
            "10%",
            "1.0000001GB",
            "18446744073709551616GB",
        ] {
            assert!(value.parse::<ByteSize>().is_err(), "{value}");
        }
        assert_eq!(
            "87.5%"
                .parse::<Reservation>()
                .unwrap()
                .bytes(8_000_000_000)
                .unwrap(),
            7_000_000_000
        );
        for value in ["0%", "100.1%", "inf%"] {
            assert!(value.parse::<Reservation>().is_err());
        }
    }
    #[test]
    fn default_pool_covers_twenty_four_contexts_and_private_tails() {
        let p = PoolPlan::new(16, 1_048_576, 24, None, None, 96 << 30, 96 << 30).unwrap();
        assert_eq!(p.pages, [49_216, 49_216, 49_216, 98_432]);
        assert_eq!(p.global_bytes, 22_426_746_880);
        assert!(p.cache_bytes > p.global_bytes);
        let small = PoolPlan::new(16, 32768, 24, None, None, 8 << 30, 96 << 30).unwrap();
        assert_eq!(small.pages, [1600, 1600, 1600, 3200]);
        assert!(PoolPlan::new(16, 1_048_576, 24, None, None, 32 << 30, 96 << 30).is_ok());
        assert!(PoolPlan::new(16, 1_048_576, 24, None, None, 22 << 30, 96 << 30).is_err());
    }
    #[test]
    fn exact_and_total_budgets_round_down_without_undercutting_admission() {
        let total = 96 << 30;
        let free = 56 << 30;
        let default = PoolPlan::new(16, 1_048_576, 24, None, None, free, total).unwrap();
        let exact = PoolPlan::new(
            16,
            1_048_576,
            24,
            Some(ByteSize(default.global_bytes + 99)),
            None,
            free,
            total,
        )
        .unwrap();
        assert_eq!(exact.pages, default.pages);
        let small =
            PoolPlan::new(2, 1_048_576, 24, Some(ByteSize(1 << 30)), None, free, total).unwrap();
        assert!(small.global_bytes <= 1 << 30);
        let c2 = PoolPlan::new(2, 1_048_576, 24, None, None, free, total).unwrap();
        assert_eq!(c2.pages[0], 2048 * 10 + 50);
        let reservation = Some("80GiB".parse().unwrap());
        let p = PoolPlan::new(16, 1_048_576, 24, None, reservation, free, total).unwrap();
        assert!(p.cache_bytes + p.occupied_before + RUNTIME_HEADROOM <= 80 << 30);
        let next =
            PoolPlan::from_groups(16, p.pages[0] + 1, p.occupied_before, p.reservation_bytes)
                .unwrap();
        assert!(next.cache_bytes + p.occupied_before + RUNTIME_HEADROOM > 80 << 30);
        assert!(PoolPlan::new(
            16,
            1_048_576,
            24,
            Some(ByteSize(1 << 20)),
            None,
            free,
            total
        )
        .is_err());
        assert!(PoolPlan::new(
            16,
            1_048_576,
            24,
            Some(ByteSize(default.global_bytes)),
            Some("50GiB".parse().unwrap()),
            free,
            total
        )
        .is_err());
        assert!(PoolPlan::new(
            16,
            1_048_576,
            24,
            None,
            Some("101GiB".parse().unwrap()),
            free,
            total
        )
        .is_err());
    }
}
