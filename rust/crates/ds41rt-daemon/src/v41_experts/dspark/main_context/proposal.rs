//! Bind produced context to canonical request groups and accepted prefixes.
use super::DsparkMainContext;
use crate::v41_backbone_router::ExpertRow;
use crate::v41_dspark_cache::{DsparkWindow, WindowChunk, WindowLease};
use anyhow::{ensure, Result};
use ds41rt_ffi::Ds41rtDeviceBuffer;

struct Group {
    request: u64,
    position: u64,
    source: u32,
    rows: u32,
}
fn groups(rows: &[ExpertRow]) -> Result<Vec<Group>> {
    ensure!(
        !rows.is_empty() && rows.len() <= 4096,
        "invalid main proposal rows"
    );
    let mut groups: Vec<Group> = Vec::new();
    for (index, row) in rows.iter().enumerate() {
        if let Some(last) = groups.last_mut().filter(|g| g.request == row.request_id) {
            ensure!(
                last.position.checked_add(u64::from(last.rows)) == Some(row.position),
                "main proposal positions are not contiguous"
            );
            last.rows += 1;
        } else {
            ensure!(
                groups.len() < 16 && groups.iter().all(|g| g.request != row.request_id),
                "main proposal repeats a request or exceeds sixteen requests"
            );
            groups.push(Group {
                request: row.request_id,
                position: row.position,
                source: index as u32,
                rows: 1,
            });
        }
    }
    Ok(groups)
}
fn validate_acceptance(groups: &[Group], accepted: &[u32]) -> Result<()> {
    ensure!(
        groups.len() == accepted.len(),
        "main acceptance count differs from request count"
    );
    ensure!(
        groups.iter().zip(accepted).all(|(g, &n)| n <= g.rows),
        "main acceptance exceeds proposed prefix"
    );
    Ok(())
}

/// Holds exclusive producer ownership until publication or discard. The caller
/// must also prevalidate its target-cache/engram transaction before publication.
pub(crate) struct MainProposal<'p, 'w, 'a> {
    main: &'p mut DsparkMainContext<'w, 'a>,
    batch: u64,
    groups: Vec<Group>,
}
impl<'w, 'a> DsparkMainContext<'w, 'a> {
    /// # Safety
    /// Rows and input belong to the given completed target batch. Input obeys
    /// execute_input's device, finite-value and exclusive-access contract.
    pub unsafe fn execute_rows<'p>(
        &'p mut self,
        input: Ds41rtDeviceBuffer,
        batch: u64,
        rows: &[ExpertRow],
    ) -> Result<MainProposal<'p, 'w, 'a>> {
        self.ready = None;
        let groups = groups(rows)?;
        let positions = rows.iter().map(|r| r.position).collect::<Vec<_>>();
        unsafe {
            self.execute_input(input, &positions)?;
        }
        Ok(MainProposal {
            main: self,
            batch,
            groups,
        })
    }
}
impl MainProposal<'_, '_, '_> {
    pub fn output(&self) -> Result<Ds41rtDeviceBuffer> {
        self.main.output()
    }
    #[cfg(test)]
    pub fn kv_output(&self, stage: usize) -> Result<Ds41rtDeviceBuffer> {
        self.main.kv_output(stage)
    }

    /// Validation failures preserve the proposal for correction; starting a GPU
    /// commit consumes it. Zero acceptance leaves the corresponding rings alone.
    /// # Safety
    /// Windows are on this producer's device with no outstanding consumers. The
    /// caller owns the matching target transaction and must revoke its requests
    /// if a later operation fails after these caches have advanced.
    pub unsafe fn commit(
        &mut self,
        batch: u64,
        windows: &mut [&mut DsparkWindow<'_>; 3],
        leases: [&[WindowLease]; 3],
        accepted: &[u32],
    ) -> Result<()> {
        let chunks = self.validate_commit(batch, windows, leases, accepted)?;
        if chunks[0].is_empty() {
            self.main.ready = None;
            return Ok(());
        }
        unsafe {
            self.main
                .commit(windows, [&chunks[0], &chunks[1], &chunks[2]])
        }
    }
    /// Check publication without consuming the proposal or changing a cache.
    pub fn validate_commit(
        &self,
        batch: u64,
        windows: &[&mut DsparkWindow<'_>; 3],
        leases: [&[WindowLease]; 3],
        accepted: &[u32],
    ) -> Result<[Vec<WindowChunk>; 3]> {
        ensure!(self.batch == batch, "foreign main proposal batch");
        let rows = self
            .main
            .ready
            .ok_or_else(|| anyhow::anyhow!("main proposal consumed"))?;
        validate_acceptance(&self.groups, accepted)?;
        ensure!(
            leases.iter().all(|l| l.len() == self.groups.len()),
            "main proposal lease count differs"
        );
        let mut chunks: [Vec<WindowChunk>; 3] = std::array::from_fn(|_| Vec::new());
        for stage in 0..3 {
            for (i, group) in self.groups.iter().enumerate() {
                let lease = leases[stage][i];
                ensure!(
                    windows[stage].request_id(lease)? == group.request,
                    "main proposal cache request differs"
                );
                let end = windows[stage].committed_end(lease)?;
                ensure!(
                    end.is_none_or(|end| end == group.position),
                    "main proposal cache position changed"
                );
                ensure!(
                    end == windows[0].committed_end(leases[0][i])?,
                    "main proposal stage ends differ"
                );
                if accepted[i] != 0 {
                    chunks[stage].push(WindowChunk {
                        lease,
                        position: group.position,
                        source_row: group.source,
                        tokens: accepted[i],
                    });
                }
            }
            if !chunks[stage].is_empty() {
                windows[stage].validate_write(&chunks[stage], rows)?;
            }
        }
        Ok(chunks)
    }
}
impl Drop for MainProposal<'_, '_, '_> {
    fn drop(&mut self) {
        self.main.ready = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn row(request_id: u64, position: u64) -> ExpertRow {
        ExpertRow {
            request_id,
            position,
            kind: ds41rt_transport::ExpertV2SourceKind::MtpVerify,
        }
    }
    #[test]
    fn main_prefixes_keep_flattened_offsets_and_reject_ambiguous_rows() {
        let plan = groups(&[row(8, 10), row(8, 11), row(9, 2)]).unwrap();
        assert_eq!((plan[0].source, plan[0].rows, plan[1].source), (0, 2, 2));
        validate_acceptance(&plan, &[1, 0]).unwrap();
        validate_acceptance(&plan, &[0, 0]).unwrap();
        assert!(validate_acceptance(&plan, &[3, 0]).is_err());
        assert!(validate_acceptance(&plan, &[1]).is_err());
        assert!(groups(&[row(8, 10), row(8, 12)]).is_err());
        assert!(groups(&[row(8, 10), row(9, 2), row(8, 11)]).is_err());
        assert!(groups(&[row(8, u64::MAX), row(8, 0)]).is_err());
    }
}
