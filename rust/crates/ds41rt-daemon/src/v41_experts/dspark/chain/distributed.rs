//! Complete lane-owned GPU1 draft chain with peer embeddings and TP2 vocabulary.
use super::*;
use crate::v41_experts::dspark::DistributedDsparkTerminal;
use crate::v41_memory::device::{Device, DeviceOwner};
use crate::v41_tensors::VocabularyShard;
#[cfg(test)]
mod tests;

pub(crate) struct DistributedDsparkChain<'w, 'a> {
    chain: DeviceOwner<'a, DsparkChain<'w, 'a>>,
    terminal: DistributedDsparkTerminal<'w, 'a>,
    ready: bool,
}
impl<'w, 'a> DistributedDsparkChain<'w, 'a> {
    pub fn device_bytes(weights: &DsparkWeights<'a>, requests: u32, split: usize) -> Result<[usize; 2]> {
        let mut bytes = DistributedDsparkTerminal::device_bytes(requests as usize, split)?;
        bytes[1] = bytes[1].checked_add(weights.chain_bytes(requests)?).context("distributed draft budget overflow")?;
        Ok(bytes)
    }
    pub fn new(devices: [Device<'a>; 2], weights: &'w DsparkWeights<'a>,
        embedding: &'w NativeRtxTensors<'a>, shards: [&'w VocabularyShard<'a>; 2],
        requests: u32, budgets: [usize; 2]) -> Result<Self> {
        let required = devices[1].run(|| Self::device_bytes(weights, requests, shards[0].tokens().end))?;
        ensure!(required.iter().zip(budgets).all(|(need, budget)| *need <= budget), "distributed draft exceeds budget");
        ensure!(weights.tensor("mtp.2.norm.weight")?.device_id == devices[1].id, "draft weights must reside on rank 1");
        let terminal_bytes = DistributedDsparkTerminal::device_bytes(requests as usize, shards[0].tokens().end)?;
        Ok(Self {
            chain: devices[1].own(|| weights.embedded_chain(embedding, requests, weights.chain_bytes(requests)?))?,
            terminal: DistributedDsparkTerminal::new(devices, weights, shards, requests as usize, terminal_bytes)?,
            ready: false,
        })
    }
    pub fn stage_tokens(&mut self, tokens: &[i32]) -> Result<()> {
        self.ready = false;
        self.chain.stage_tokens(tokens)
    }
    pub fn stage_sampling(&mut self, rngs: &mut [&mut DsparkRng], temperatures: &[f32]) -> Result<()> {
        self.ready = false;
        self.terminal.stage_sampling(rngs, temperatures)
    }
    /// # Safety
    /// Cache windows belong to GPU1 and describe the same packed requests and
    /// committed positions in all three stages. Their producers are complete.
    /// Borrowed weights and window storage remain live and no external access
    /// races draft work through return/cancellation.
    pub async unsafe fn execute(&mut self, windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3]) -> Result<[Ds41rtDeviceBuffer; 3]> {
        let device = self.chain.device;
        device.future(unsafe { self.execute_on(windows, bindings) }).await
    }
    async unsafe fn execute_on(&mut self, windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3]) -> Result<[Ds41rtDeviceBuffer; 3]> {
        self.ready = false;
        let count = bindings[0].len();
        self.terminal.validate_sampling(count)?;
        let chain = self.chain.get_mut();
        let reads = chain.prepare(windows, bindings)?;
        if let Some(&(_, owners)) = chain.graphs.get(&count) {
            ensure!(owners == reads.each_ref().map(|read| read.owner), "distributed draft capture owner differs");
        }
        let mut pending = PendingDraft { count, reads, warming: !chain.has_graph(count), armed: true,
            library: chain.stream.library, stream: chain.stream.raw };
        // This guard is live before any submission and drops under the GPU1
        // future scope, including cancellation during a later head phase.
        let queued = (|| -> Result<()> { unsafe {
            let lib = chain.stream.library;
            lib.copy_h2d_async(chain.tokens.buffer, &chain.token_staging.bytes_mut()[..count * 4], chain.stream.raw)?;
            for stage in 0..3 { chain.stages[stage].upload_on(&pending.reads[stage], bindings[stage], chain.stream.raw)?; }
            if pending.warming { chain.enqueue(&pending.reads, count)?; }
            else { lib.cuda_graph_launch(chain.graphs[&count].0, chain.stream.raw)?; }
            let source = chain.stages[2].output_storage();
            let target = self.terminal.inputs();
            chain.ops.terminal_layout(source[0], source[1], target[0], target[1], count as u32, chain.stream.raw)?;
            lib.copy_d2d_async(target[2], chain.tokens.buffer, count * 4, chain.stream.raw)
        } })();
        let drained = chain.stream.wait().await;
        queued.and(drained)?;
        if pending.warming { unsafe { chain.capture_ready(&pending.reads, count)?; } }
        pending.armed = false;
        drop(pending); // All transformer/cache reads are complete.
        let output = unsafe { self.terminal.execute(count).await? };
        self.ready = true;
        Ok(output)
    }
    pub fn output(&self) -> Result<[Ds41rtDeviceBuffer; 3]> {
        ensure!(self.ready, "distributed draft output unpublished");
        self.terminal.output()
    }
}
