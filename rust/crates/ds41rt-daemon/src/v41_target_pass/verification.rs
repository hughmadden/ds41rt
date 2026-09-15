//! Static dispatch for independent verification lanes in either GPU layout.
use super::{CacheStage, DistributedTargetPass, NativeTp4Wave, RequestBatch, Requests,
    Result, TargetCache, TargetPass};
use crate::v41_memory::device::DeviceOwner;
use std::cell::RefCell;

/// Futures remain concrete: selecting a layout does not box lane work or add a
/// common stream. Each implementation retains its own device/transport owners.
pub(crate) trait VerificationTarget<'a>: TargetCache<'a> {
    type Transport;
    fn set_route_capture(&mut self, enabled: bool) -> Result<()>;
    fn captured_routes(&self) -> &[Vec<[u32; 6]>];
    async unsafe fn execute_shared(&mut self, requests: &RefCell<&mut Requests<'a>>,
        batch: &mut RequestBatch, transport: &mut Self::Transport, placement: u64,
        selected: &[usize]) -> Result<()>;
    async unsafe fn execute_shared_greedy(&mut self, requests: &RefCell<&mut Requests<'a>>,
        batch: &mut RequestBatch, transport: &mut Self::Transport, placement: u64,
        selected: &[usize]) -> Result<Vec<(u32, f32)>>;
    async fn download_logits(&mut self, batch: &RequestBatch, rows: &[usize]) -> Result<Vec<u8>>;
    fn enqueue_cache_commit(&mut self, requests: &Requests<'a>, batch: &RequestBatch,
        accepted: &[u32]) -> Result<()>;
    fn poll_cache_commit(&self) -> Result<bool>;
    fn abort_cache_commit(&mut self, requests: &mut Requests<'a>) -> Result<()>;
    fn discard(&mut self, batch: &mut RequestBatch) -> Result<()>;
}

impl<'a> VerificationTarget<'a> for TargetPass<'_, 'a> {
    type Transport = NativeTp4Wave<'a>;
    fn set_route_capture(&mut self, enabled: bool) -> Result<()> {
        TargetPass::set_route_capture(self, enabled); Ok(())
    }
    fn captured_routes(&self) -> &[Vec<[u32; 6]>] { TargetPass::captured_routes(self) }
    async unsafe fn execute_shared(&mut self, requests: &RefCell<&mut Requests<'a>>,
        batch: &mut RequestBatch, transport: &mut Self::Transport, placement: u64,
        selected: &[usize]) -> Result<()> {
        unsafe { TargetPass::execute_shared(self, requests, batch, transport, placement, selected).await?; }
        Ok(())
    }
    async unsafe fn execute_shared_greedy(&mut self, requests: &RefCell<&mut Requests<'a>>,
        batch: &mut RequestBatch, transport: &mut Self::Transport, placement: u64,
        selected: &[usize]) -> Result<Vec<(u32, f32)>> {
        unsafe { TargetPass::execute_shared_greedy(self, requests, batch, transport, placement, selected).await }
    }
    async fn download_logits(&mut self, batch: &RequestBatch, rows: &[usize]) -> Result<Vec<u8>> {
        TargetPass::download_logits(self, batch, rows).await
    }
    fn enqueue_cache_commit(&mut self, requests: &Requests<'a>, batch: &RequestBatch,
        accepted: &[u32]) -> Result<()> { TargetPass::enqueue_cache_commit(self, requests, batch, accepted) }
    fn poll_cache_commit(&self) -> Result<bool> { TargetPass::poll_cache_commit(self) }
    fn abort_cache_commit(&mut self, requests: &mut Requests<'a>) -> Result<()> {
        TargetPass::abort_cache_commit(self, requests)
    }
    fn discard(&mut self, batch: &mut RequestBatch) -> Result<()> { TargetPass::discard(self, batch) }
}

impl<'a> VerificationTarget<'a> for DistributedTargetPass<'_, 'a> {
    type Transport = DeviceOwner<'a, NativeTp4Wave<'a>>;
    fn set_route_capture(&mut self, enabled: bool) -> Result<()> {
        DistributedTargetPass::set_route_capture(self, enabled)
    }
    fn captured_routes(&self) -> &[Vec<[u32; 6]>] { DistributedTargetPass::captured_routes(self) }
    async unsafe fn execute_shared(&mut self, requests: &RefCell<&mut Requests<'a>>,
        batch: &mut RequestBatch, transport: &mut Self::Transport, placement: u64,
        selected: &[usize]) -> Result<()> {
        anyhow::ensure!(batch.cache()?.stage() == CacheStage::Full, "verification requires full phase");
        unsafe { self.execute(requests, batch, transport, placement, selected, None, None, false).await }
    }
    async unsafe fn execute_shared_greedy(&mut self, requests: &RefCell<&mut Requests<'a>>,
        batch: &mut RequestBatch, transport: &mut Self::Transport, placement: u64,
        selected: &[usize]) -> Result<Vec<(u32, f32)>> {
        anyhow::ensure!(batch.cache()?.stage() == CacheStage::Full, "verification requires full phase");
        unsafe { self.execute(requests, batch, transport, placement, selected, None, None, true).await?; }
        self.greedy_output(batch)
    }
    async fn download_logits(&mut self, batch: &RequestBatch, rows: &[usize]) -> Result<Vec<u8>> {
        DistributedTargetPass::download_logits(self, batch, rows).await
    }
    fn enqueue_cache_commit(&mut self, requests: &Requests<'a>, batch: &RequestBatch,
        accepted: &[u32]) -> Result<()> { DistributedTargetPass::enqueue_cache_commit(self, requests, batch, accepted) }
    fn poll_cache_commit(&self) -> Result<bool> { DistributedTargetPass::poll_cache_commit(self) }
    fn abort_cache_commit(&mut self, requests: &mut Requests<'a>) -> Result<()> {
        DistributedTargetPass::abort_cache_commit(self, requests)
    }
    fn discard(&mut self, batch: &mut RequestBatch) -> Result<()> { DistributedTargetPass::discard(self, batch) }
}
