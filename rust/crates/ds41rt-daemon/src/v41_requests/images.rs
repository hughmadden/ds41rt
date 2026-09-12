//! Request-owned image layout and lazily prepared complete embedding spans.
use anyhow::{ensure, Context, Result};
use ds41rt_loader::{V41ImageSpan, V41_MAX_IMAGES};
use std::borrow::Cow;

const ROW_BYTES: usize = 5120 * 2;
struct Span {
    start: usize,
    end: usize,
    features: Option<Box<[u8]>>,
}
#[derive(Default)]
pub(crate) struct RequestImages {
    spans: Vec<Span>,
}
impl RequestImages {
    pub fn new(images: &[V41ImageSpan]) -> Result<Self> {
        Self::from_ranges(
            &images
                .iter()
                .map(|s| (s.start, s.image.grid().tokens()))
                .collect::<Vec<_>>(),
        )
    }
    fn from_ranges(ranges: &[(usize, usize)]) -> Result<Self> {
        ensure!(ranges.len() <= V41_MAX_IMAGES, "too many request images");
        let mut spans = Vec::with_capacity(ranges.len());
        let mut previous = 0;
        for &(start, rows) in ranges {
            let end = start
                .checked_add(rows)
                .context("request image span overflow")?;
            ensure!(
                rows > 0 && rows <= 1024 && start >= previous && end <= 1_048_576,
                "invalid request image span"
            );
            spans.push(Span {
                start,
                end,
                features: None,
            });
            previous = end;
        }
        Ok(Self { spans })
    }
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }
    pub fn needed(&self, start: usize, end: usize) -> Result<Vec<usize>> {
        ensure!(
            start <= end && end <= 1_048_576,
            "invalid image preparation range"
        );
        Ok(self
            .spans
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                (start < end && s.start < end && s.end > start && s.features.is_none()).then_some(i)
            })
            .collect())
    }
    pub fn install(&mut self, index: usize, features: Vec<u8>) -> Result<()> {
        let span = self
            .spans
            .get_mut(index)
            .context("request image index out of bounds")?;
        ensure!(
            span.features.is_none() && features.len() == (span.end - span.start) * ROW_BYTES,
            "request image features already prepared or extent differs"
        );
        span.features = Some(features.into_boxed_slice());
        Ok(())
    }
    pub fn row(&self, position: u64) -> Option<&[u8]> {
        let position = usize::try_from(position).ok()?;
        let span = self
            .spans
            .iter()
            .find(|s| position >= s.start && position < s.end)?;
        let offset = (position - span.start) * ROW_BYTES;
        span.features
            .as_deref()
            .map(|f| &f[offset..offset + ROW_BYTES])
    }
    pub fn mask(&self, start: u64, rows: usize) -> Result<Option<Vec<bool>>> {
        if self.is_empty() {
            return Ok(None);
        }
        let start = usize::try_from(start)?;
        let end = start
            .checked_add(rows)
            .context("request image mask overflow")?;
        ensure!(end <= 1_048_576, "image mask exceeds model context");
        let mut mask = None;
        for span in &self.spans {
            let a = start.max(span.start);
            let b = end.min(span.end);
            if a < b {
                mask.get_or_insert_with(|| vec![false; rows])[a - start..b - start].fill(true);
            }
        }
        Ok(mask)
    }
    pub fn resolve_mask<'m>(
        &self,
        start: u64,
        rows: usize,
        explicit: Option<&'m [bool]>,
    ) -> Result<Option<Cow<'m, [bool]>>> {
        ensure!(
            explicit.is_none_or(|m| m.len() == rows),
            "image mask length differs"
        );
        if self.is_empty() {
            return Ok(explicit.map(Cow::Borrowed));
        }
        let derived = self.mask(start, rows)?;
        if let Some(explicit) = explicit {
            ensure!(
                explicit
                    .iter()
                    .enumerate()
                    .all(|(i, &v)| v == derived.as_ref().is_some_and(|m| m[i])),
                "explicit image mask differs from request image layout"
            );
        }
        Ok(derived.map(Cow::Owned))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chunk_and_replay_masks_include_every_image_delimiter() -> Result<()> {
        let images = RequestImages::from_ranges(&[(63, 10), (128, 7)])?;
        assert_eq!(
            images.mask(60, 16)?.unwrap(),
            [vec![false; 3], vec![true; 10], vec![false; 3]].concat()
        );
        assert_eq!(
            images.mask(70, 64)?.unwrap(),
            [vec![true; 3], vec![false; 55], vec![true; 6]].concat()
        );
        assert_eq!(images.mask(132, 3)?.unwrap(), vec![true; 3]);
        assert!(images.mask(135, 4096)?.is_none());
        // These are the three recent tokens used to seed bounded Engram replay.
        assert_eq!(images.mask(71, 3)?.unwrap(), vec![true, true, false]);
        assert!(images.resolve_mask(63, 3, Some(&[false; 3])).is_err());
        assert!(images.resolve_mask(73, 3, Some(&[true; 3])).is_err());
        assert!(images.resolve_mask(63, 3, Some(&[true; 3]))?.is_some());
        assert!(images.resolve_mask(63, 3, Some(&[true; 2])).is_err());
        assert!(images.mask(u64::MAX, 4).is_err());
        Ok(())
    }
    #[test]
    fn features_are_lazy_bounded_and_addressed_by_request_position() -> Result<()> {
        let mut images =
            RequestImages::from_ranges(&(0..16).map(|i| (i * 1024, 1024)).collect::<Vec<_>>())?;
        assert_eq!(images.needed(16384, 16384)?, Vec::<usize>::new());
        assert_eq!(images.needed(16384, 16400)?, Vec::<usize>::new());
        assert_eq!(images.needed(1023, 1025)?, vec![0, 1]);
        assert!(images.row(1023).is_none());
        assert!(images.install(0, vec![0; ROW_BYTES]).is_err());
        let mut features = vec![0; 1024 * ROW_BYTES];
        features[1023 * ROW_BYTES..].fill(42);
        images.install(0, features)?;
        assert_eq!(images.row(1023).unwrap(), &[42; ROW_BYTES]);
        assert_eq!(images.row(0).unwrap(), &[0; ROW_BYTES]);
        assert!(images.row(1024).is_none());
        assert_eq!(images.needed(1023, 1025)?, vec![1]);
        assert!(images.install(0, vec![0; 1024 * ROW_BYTES]).is_err());
        assert!(images.install(16, Vec::new()).is_err());
        assert!(RequestImages::from_ranges(&[(0, 1); 17]).is_err());
        for ranges in [
            vec![(usize::MAX, 1)],
            vec![(1, 0)],
            vec![(0, 1025)],
            vec![(0, 5), (4, 5)],
            vec![(1_048_576, 1)],
        ] {
            assert!(RequestImages::from_ranges(&ranges).is_err());
        }
        Ok(())
    }
    #[test]
    fn text_and_explicit_component_masks_remain_borrowed() -> Result<()> {
        let images = RequestImages::default();
        assert!(images.mask(0, 1_048_576)?.is_none());
        assert!(images.resolve_mask(0, 1_048_576, None)?.is_none());
        let explicit = [true, false, true];
        let resolved = images.resolve_mask(0, 3, Some(&explicit))?.unwrap();
        assert!(matches!(resolved, Cow::Borrowed(_)));
        assert_eq!(resolved.as_ptr(), explicit.as_ptr());
        Ok(())
    }
}
