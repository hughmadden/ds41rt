//! Static draft-chain dispatch for the shared request lifecycle.
use super::*;
use crate::v41_experts::dspark::DistributedDsparkChain;

pub(crate) trait DraftChain {
    fn stage_tokens(&mut self, tokens: &[i32]) -> Result<()>;
    fn stage_sampling(&mut self, rngs: &mut [&mut ds41rt_core::DsparkRng], temperatures: &[f32]) -> Result<()>;
    /// # Safety
    /// Windows, bindings and staged inputs obey the concrete chain's contract;
    /// backing storage remains live until completion or chain destruction.
    unsafe fn begin_replay(&mut self, windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3]) -> Result<()>;
    fn poll_replay(&mut self) -> Result<Option<(Vec<u32>, Vec<f32>)>>;
}
macro_rules! chain {
    ($ty:ident) => {
        impl DraftChain for $ty<'_, '_> {
            fn stage_tokens(&mut self, tokens: &[i32]) -> Result<()> { self.stage_tokens(tokens) }
            fn stage_sampling(&mut self, rngs: &mut [&mut ds41rt_core::DsparkRng], temperatures: &[f32]) -> Result<()> {
                self.stage_sampling(rngs, temperatures)
            }
            unsafe fn begin_replay(&mut self, windows: [&DsparkWindow<'_>; 3],
                bindings: [&[(WindowLease, u64)]; 3]) -> Result<()> {
                unsafe { self.begin_replay(windows, bindings) }
            }
            fn poll_replay(&mut self) -> Result<Option<(Vec<u32>, Vec<f32>)>> { self.poll_replay() }
        }
    };
}
chain!(DsparkChain);
chain!(DistributedDsparkChain);
