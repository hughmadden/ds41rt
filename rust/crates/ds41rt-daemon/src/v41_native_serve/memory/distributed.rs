//! Startup accounting for colocated compressed sources on a pair of GPUs.
//! Budgets are residual bytes after all non-global allocations and headroom.
//! No GPU allocation, device selection, or serving-loop policy occurs here.

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct SourcePoolPlan {
    pub groups: usize,
    pub pages: [usize; 4],
    pub device_bytes: [usize; 2],
    pub unused_bytes: [usize; 2],
}

impl SourcePoolPlan {
    pub fn new(
        available: [usize; 2],
        owners: [usize; 4],
        page_bytes: usize,
        minimum_groups: usize,
        maximum_groups: usize,
        exact_global_bytes: Option<usize>,
    ) -> Result<Self, &'static str> {
        if page_bytes == 0 || minimum_groups == 0 || minimum_groups > maximum_groups {
            return Err("invalid compressed-pool geometry");
        }
        let ratios = [1usize, 1, 1, 2];
        let mut per_device = [0usize; 2];
        for (owner, ratio) in owners.into_iter().zip(ratios) {
            let bytes = page_bytes.checked_mul(ratio).ok_or("source page size overflow")?;
            let sum = per_device.get_mut(owner).ok_or("source GPU must be zero or one")?;
            *sum = sum.checked_add(bytes).ok_or("source group size overflow")?;
        }
        let group_bytes = page_bytes.checked_mul(5).ok_or("global group size overflow")?;
        let fits = per_device.iter().zip(available).filter(|(bytes, _)| **bytes != 0)
            .map(|(bytes, budget)| budget / bytes).min().ok_or("no source owners")?
            .min(maximum_groups);
        let groups = match exact_global_bytes {
            // Match the existing planner: byte requests round down to complete
            // groups, never borrow unused bytes from the other device.
            Some(bytes) => bytes / group_bytes,
            None => fits,
        };
        if groups < minimum_groups || groups > fits {
            return Err("compressed pool does not fit per-GPU capacity or minimum admission");
        }
        let device_bytes = per_device.map(|bytes| bytes * groups);
        Ok(Self {
            groups,
            pages: [groups, groups, groups, groups.checked_mul(2).ok_or("page count overflow")?],
            device_bytes,
            unused_bytes: [available[0] - device_bytes[0], available[1] - device_bytes[1]],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asymmetric_sources_obey_the_tighter_card() {
        let p = SourcePoolPlan::new([3_000, 10_000], [0, 0, 0, 1], 100, 1, 100, None).unwrap();
        assert_eq!(p.pages, [10, 10, 10, 20]);
        assert_eq!(p.device_bytes, [3_000, 2_000]);
        assert_eq!(p.unused_bytes, [0, 8_000]);
        assert!(SourcePoolPlan::new([3_000, 10_000], [0, 0, 0, 1], 100, 1, 100, Some(5_500)).is_err());
    }

    #[test]
    fn swapped_devices_and_ratio_one_source_limit() {
        let p = SourcePoolPlan::new([800, 30_000], [1, 1, 1, 0], 100, 4, 100, None).unwrap();
        assert_eq!(p.groups, 4);
        assert_eq!(p.device_bytes, [800, 1_200]);
        assert!(SourcePoolPlan::new([799, 30_000], [1, 1, 1, 0], 100, 4, 100, None).is_err());
    }

    #[test]
    fn exact_rounding_and_physical_limit() {
        let p = SourcePoolPlan::new([10_000; 2], [0, 0, 0, 1], 100, 1, 8, Some(4_499)).unwrap();
        assert_eq!(p.groups, 8);
        assert!(SourcePoolPlan::new([10_000; 2], [0, 0, 0, 1], 100, 1, 8, Some(4_500)).is_err());
    }

    #[test]
    fn invalid_geometry_rejects_without_arithmetic_wrap() {
        for (owners, page, min, max) in [([0, 0, 0, 2], 100, 1, 8),
            ([0, 0, 0, 1], 0, 1, 8), ([0, 0, 0, 1], usize::MAX, 1, 8),
            ([0, 0, 0, 1], 100, 0, 8), ([0, 0, 0, 1], 100, 9, 8)] {
            assert!(SourcePoolPlan::new([usize::MAX; 2], owners, page, min, max, None).is_err());
        }
    }
}
