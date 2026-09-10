//! Shifted backbone attention/FFN sequencing with exact query-result identity.
use crate::v41_attention_binding::QueryBinding;
use crate::v41_attention_output::AttentionOutput;
use crate::v41_attention_query::{AttentionQueryOutput, AttentionQueryWave};
use crate::v41_hc::HcSublayer;
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary};
#[derive(Clone, Copy)]
enum Phase {
    Idle,
    Attention(QueryBinding, usize),
    Ffn(QueryBinding, usize),
    Ready(QueryBinding, usize),
}
pub(crate) struct FfnInput<'a> {
    pub residual: Ds41rtDeviceBuffer,
    pub incoming_pre: Ds41rtDeviceBuffer,
    pub values: Ds41rtDeviceBuffer,
    pub layer: usize,
    pub tokens: &'a [u64],
    binding: QueryBinding,
}
impl FfnInput<'_> {
    pub fn binding(&self) -> QueryBinding {
        self.binding
    }
}
pub(crate) struct BlockOutput<'a> {
    binding: QueryBinding,
    pub residual: Ds41rtDeviceBuffer,
    pub pre: Ds41rtDeviceBuffer,
    pub layer: usize,
    pub tokens: &'a [u64],
}
impl BlockOutput<'_> {
    pub fn binding(&self) -> QueryBinding {
        self.binding
    }
}
pub(crate) struct BackboneBlockWave<'w, 'a> {
    library: &'a NativeLibrary,
    layer: usize,
    attention: HcSublayer<'w, 'a>,
    ffn: HcSublayer<'w, 'a>,
    capacity: usize,
    tokens: Vec<u64>,
    phase: Phase,
}
impl<'w, 'a> BackboneBlockWave<'w, 'a> {
    pub fn device_bytes(capacity: usize) -> Result<usize> {
        Ok(2 * HcSublayer::device_bytes(capacity)?)
    }
    pub(crate) fn new(
        library: &'a NativeLibrary,
        layer: usize,
        attention: HcSublayer<'w, 'a>,
        ffn: HcSublayer<'w, 'a>,
        capacity: usize,
    ) -> Self {
        Self {
            library,
            layer,
            attention,
            ffn,
            capacity,
            tokens: Vec::new(),
            phase: Phase::Idle,
        }
    }
    /// Residual [capacity,4,5120] and incoming FP32 pre [capacity,4]. Engram
    /// contributions and dSpark tap reads precede begin_attention at the caller.
    pub fn inputs(&self) -> [Ds41rtDeviceBuffer; 2] {
        self.attention.inputs()
    }
    pub fn reset(&mut self) {
        self.phase = Phase::Idle;
        self.tokens.clear();
        self.attention.invalidate();
        self.ffn.invalidate();
    }
    /// # Safety
    /// Inputs are finite in token order with all producer writes complete. Query
    /// storage and both boundary owners have exclusive use on this device.
    pub unsafe fn begin_attention<'q>(
        &mut self,
        query: &'q mut AttentionQueryWave<'_, '_>,
        tokens: &[u64],
    ) -> Result<AttentionQueryOutput<'q>> {
        self.reset();
        ensure!(
            query.layer() == self.layer
                && !tokens.is_empty()
                && tokens.len() <= self.capacity
                && tokens.iter().all(|&p| p < 1048576),
            "block attention layer or tokens differ"
        );
        ensure!(
            query.input().device_id == self.inputs()[0].device_id,
            "block query device differs"
        );
        let normalized = unsafe { self.attention.begin(tokens.len()) }?;
        if let Err(e) = self
            .library
            .copy_d2d(query.input(), normalized, normalized.bytes)
        {
            self.reset();
            return Err(e);
        }
        let out = match unsafe { query.execute_tokens(tokens) } {
            Ok(o) => o,
            Err(e) => {
                self.reset();
                return Err(e);
            }
        };
        self.tokens.extend_from_slice(tokens);
        self.phase = Phase::Attention(out.binding()?, tokens.len());
        Ok(out)
    }
    /// # Safety
    /// Attention output is complete and immutable; no external writes race the
    /// preserved residual or mixing coefficients. A rejected finish resets phase.
    pub unsafe fn begin_ffn(&mut self, output: &AttentionOutput<'_>) -> Result<FfnInput<'_>> {
        let result = (|| -> Result<(Ds41rtDeviceBuffer, QueryBinding)> {
            let Phase::Attention(binding, rows) = std::mem::replace(&mut self.phase, Phase::Idle)
            else {
                anyhow::bail!("block attention is not pending");
            };
            ensure!(
                output.binding()? == binding
                    && output.layer == self.layer
                    && output.rows == rows
                    && output.projected.device_id == self.inputs()[0].device_id,
                "block attention result differs"
            );
            self.library.copy_d2d(
                self.attention.sublayer_result(),
                output.projected,
                rows * 10240,
            )?;
            let completed = unsafe { self.attention.finish()? };
            for (dst, src) in self.ffn.inputs().into_iter().zip(completed) {
                self.library.copy_d2d(dst, src, src.bytes)?;
            }
            let normalized = unsafe { self.ffn.begin(rows)? };
            self.phase = Phase::Ffn(binding, rows);
            Ok((normalized, binding))
        })();
        match result {
            Ok((values, binding)) => Ok(FfnInput {
                residual: {
                    let mut b = self.ffn.inputs()[0];
                    b.bytes = values.bytes * 4;
                    b
                },
                incoming_pre: {
                    let mut b = self.ffn.inputs()[1];
                    b.bytes = values.bytes / 10240 * 16;
                    b
                },
                values,
                layer: self.layer,
                tokens: &self.tokens,
                binding,
            }),
            Err(e) => {
                self.reset();
                Err(e)
            }
        }
    }
    /// # Safety
    /// Result is the finite completed shared+routed FFN output for the returned
    /// FfnInput binding, in its token order, on this device. It stays immutable
    /// through the copy. Transport/expert reduction must establish this contract.
    pub unsafe fn finish_ffn(
        &mut self,
        binding: QueryBinding,
        result: Ds41rtDeviceBuffer,
    ) -> Result<BlockOutput<'_>> {
        let completed = (|| -> Result<()> {
            let Phase::Ffn(expected, rows) = std::mem::replace(&mut self.phase, Phase::Idle) else {
                anyhow::bail!("block FFN is not pending");
            };
            ensure!(
                binding == expected && result.device_id == self.inputs()[0].device_id,
                "block FFN result binding differs"
            );
            self.library
                .copy_d2d(self.ffn.sublayer_result(), result, rows * 10240)?;
            unsafe {
                self.ffn.finish()?;
            }
            self.phase = Phase::Ready(binding, rows);
            Ok(())
        })();
        if let Err(e) = completed {
            self.reset();
            return Err(e);
        }
        self.output()
    }
    pub fn output(&self) -> Result<BlockOutput<'_>> {
        let (binding, rows) = match self.phase {
            Phase::Ready(binding, rows) => (binding, rows),
            _ => return Err(anyhow::anyhow!("block output unpublished")),
        };
        ensure!(self.tokens.len() == rows, "block token count differs");
        let [residual, pre] = self.ffn.output().context("block FFN output unavailable")?;
        Ok(BlockOutput {
            binding,
            residual,
            pre,
            layer: self.layer,
            tokens: &self.tokens,
        })
    }
}
