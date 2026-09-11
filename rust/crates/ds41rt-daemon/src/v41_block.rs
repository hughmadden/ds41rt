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
    Prepared(QueryBinding, usize, bool),
    Attention(QueryBinding, usize),
    Ffn(QueryBinding, usize),
    Ready(QueryBinding, usize),
}
/// Next-layer input after required engram work, before attention or dSpark taps.
pub(crate) struct PreparedBlockInput<'a> {
    pub residual: Ds41rtDeviceBuffer,
    pub pre: Ds41rtDeviceBuffer,
    pub layer: usize,
    pub tokens: &'a [u64],
    previous: QueryBinding,
}
impl PreparedBlockInput<'_> {
    pub fn previous_binding(&self) -> QueryBinding { self.previous }
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
    /// Reuse this lane's two mHC workspaces for the adjacent layer. Completed
    /// residual/pre values are copied before installing both validated bindings.
    /// No device allocation or graph capture occurs here.
    pub fn advance(
        &mut self,
        next: &'w crate::v41_backbone_hc::BackboneHcWeights<'a>,
    ) -> Result<()> {
        let result = (|| -> Result<()> {
            let (binding, rows) = match self.phase {
                Phase::Ready(binding, rows) => (binding, rows),
                _ => anyhow::bail!("block advance requires a completed FFN"),
            };
            ensure!(self.layer < 39 && next.layer() == self.layer + 1
                && binding.layer() == self.layer && self.tokens.len() == rows,
                "block advance layer or identity differs");
            let [attention, ffn] = next.prepare_bindings(&self.attention, &self.ffn)?;
            for (source, destination) in self.ffn.output()?.into_iter().zip(self.inputs()) {
                self.library.copy_d2d(destination, source, source.bytes)?;
            }
            // Backbone mHC executes on its own drained streams; it has no
            // external captured mHC graphs to invalidate during this rebind.
            unsafe {
                self.attention.install_binding(attention);
                self.ffn.install_binding(ffn);
            }
            self.layer = next.layer();
            self.phase = Phase::Prepared(binding, rows, ![1,14].contains(&self.layer));
            Ok(())
        })();
        if result.is_err() { self.reset(); }
        result
    }
    /// Start another sequence after completion, or after an explicit reset on
    /// cancellation. Token initialization is required before attention resumes.
    pub fn restart(
        &mut self,
        first: &'w crate::v41_backbone_hc::BackboneHcWeights<'a>,
    ) -> Result<()> {
        let result = (|| -> Result<()> {
            ensure!(first.layer() == 0
                && (matches!(self.phase, Phase::Idle)
                    || (self.layer == 39 && matches!(self.phase, Phase::Ready(..)))),
                "block restart requires layer zero and an idle or completed sequence");
            let [attention, ffn] = first.prepare_bindings(&self.attention, &self.ffn)?;
            unsafe {
                self.attention.install_binding(attention);
                self.ffn.install_binding(ffn);
            }
            self.layer = 0;
            self.reset();
            Ok(())
        })();
        if result.is_err() { self.reset(); }
        result
    }
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
    /// Copy a completed adjacent layer into stable next-block input storage.
    /// Layers 1 and 14 remain unavailable until their engram update completes.
    pub fn initialize_previous(&mut self, previous: &BlockOutput<'_>) -> Result<()> {
        self.reset();
        let rows = previous.tokens.len();
        ensure!(previous.layer < 39 && self.layer == previous.layer + 1
            && previous.binding.layer() == previous.layer
            && rows > 0 && rows <= self.capacity
            && previous.tokens.iter().all(|&p| p < 1048576)
            && previous.residual.bytes == rows * 40960 && previous.pre.bytes == rows * 16,
            "previous block layer, tokens or extents differ");
        let sources = [previous.residual, previous.pre];
        let destinations = self.inputs();
        ensure!(sources.iter().zip(destinations).all(|(s,d)| s.device_id == d.device_id),
            "previous block device differs");
        for (source, destination) in sources.into_iter().zip(destinations) {
            self.library.copy_d2d(destination, source, source.bytes)?;
        }
        self.tokens.extend_from_slice(previous.tokens);
        self.phase = Phase::Prepared(previous.binding, rows, ![1,14].contains(&self.layer));
        Ok(())
    }
    /// # Safety
    /// Gathered engram rows/masks correspond to this block's exact request and
    /// token order, with current history leases and completed producer writes.
    /// Gather and gate storage remain live and exclusive until the call drains.
    pub unsafe fn apply_engram(
        &mut self,
        gate: &mut crate::v41_engram::layer::EngramGate<'_, '_>,
        gathered: &crate::v41_engram::EngramDeviceView,
    ) -> Result<()> {
        let result = (|| -> Result<()> {
            let (binding, rows) = match self.phase {
                Phase::Prepared(binding, rows, false) => (binding, rows),
                _ => anyhow::bail!("block is not awaiting engram"),
            };
            ensure!(gate.layer() == self.layer && gathered.rows == rows,
                "block engram layer or rows differ");
            let output = unsafe { gate.execute_captured(self.inputs()[0], gathered) }?;
            self.library.copy_d2d(self.inputs()[0], output, output.bytes)?;
            self.phase = Phase::Prepared(binding, rows, true);
            Ok(())
        })();
        if result.is_err() { self.reset(); }
        result
    }
    /// Metadata for gather association before engram makes query inputs ready.
    pub fn pending_engram(&self) -> Result<(usize, &[u64])> {
        ensure!(matches!(self.phase, Phase::Prepared(_, _, false)),
            "block is not awaiting engram");
        Ok((self.layer, &self.tokens))
    }
    pub fn prepared_input(&self) -> Result<PreparedBlockInput<'_>> {
        let (previous, rows) = match self.phase {
            Phase::Prepared(binding, rows, true) => (binding, rows),
            _ => anyhow::bail!("block input is not prepared; engram may be pending"),
        };
        let [mut residual, mut pre] = self.inputs();
        residual.bytes = rows * 40960;
        pre.bytes = rows * 16;
        Ok(PreparedBlockInput { residual, pre, layer: self.layer,
            tokens: &self.tokens, previous })
    }
    /// # Safety
    /// Query and block storage have exclusive use on the same device. Required
    /// dSpark tap reads must finish before this call overwrites any input.
    pub unsafe fn begin_prepared_attention<'q>(
        &mut self,
        query: &'q mut AttentionQueryWave<'_, '_>,
    ) -> Result<AttentionQueryOutput<'q>> {
        let tokens = match self.prepared_input() {
            Ok(input) => input.tokens.to_vec(),
            Err(e) => { self.reset(); return Err(e); }
        };
        unsafe { self.begin_attention(query, &tokens) }
    }
    /// Copy text embeddings into the first block's stable inputs. This leaves
    /// output unpublished. Image replacement belongs to the separate vision path.
    pub fn initialize_embedding(
        &mut self,
        embedding: &crate::v41_target_embedding::TargetEmbedding<'_>,
    ) -> Result<()> {
        self.reset();
        let rows = embedding.positions.len();
        ensure!(self.layer == 0 && rows > 0
            && rows <= self.capacity && embedding.token_ids.len() == rows
            && embedding.residual.bytes == rows * 40960 && embedding.pre.bytes == rows * 16,
            "initial block embedding geometry differs");
        for (destination, source) in self.inputs().into_iter()
            .zip([embedding.residual, embedding.pre]) {
            ensure!(destination.device_id == source.device_id, "initial block device differs");
            self.library.copy_d2d(destination, source, source.bytes)?;
        }
        Ok(())
    }
    /// # Safety
    /// Query and block storage have exclusive use on the embedding device.
    /// Embeddings are finite text inputs; image replacement uses the vision path.
    pub unsafe fn begin_embedded_attention<'q>(
        &mut self,
        query: &'q mut AttentionQueryWave<'_, '_>,
        embedding: &crate::v41_target_embedding::TargetEmbedding<'_>,
    ) -> Result<AttentionQueryOutput<'q>> {
        self.reset();
        ensure!(query.layer() == 0, "initial block query layer differs");
        self.initialize_embedding(embedding)?;
        unsafe { self.begin_attention(query, embedding.positions) }
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
        let timing = std::time::Instant::now();
        let out = match unsafe {
            query.execute_tokens_prepared(tokens, |stream, input| {
                self.attention.enqueue_begin(tokens.len(), Some(input), stream)?;
                Ok(())
            })
        } {
            Ok(o) => o,
            Err(e) => {
                self.reset();
                return Err(e);
            }
        };
        self.tokens.extend_from_slice(tokens);
        tracing::debug!(target: "ds41rt::timing", layer=self.layer, rows=tokens.len(), total_us=timing.elapsed().as_micros() as u64, "target query preparation");
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
            let stream = self.attention.stream_raw();
            let enqueued = (|| -> Result<Ds41rtDeviceBuffer> {
                unsafe { self.attention.enqueue_finish(Some(output.projected), stream)?; }
                // Keep the existing residual/pre buffers, but copy on the same
                // stream after attention post and before FFN pre/mixing.
                for ((dst, src), bytes) in self.ffn.inputs().into_iter()
                    .zip(self.attention.output_storage())
                    .zip([rows * 40960, rows * 16]) {
                    unsafe { self.library.copy_d2d_async(dst, src, bytes, stream)?; }
                }
                unsafe { self.ffn.enqueue_begin(rows, None, stream) }
            })();
            // Always drain, including partial enqueue failure: both mHC owners
            // must be safe to reset/rebind when this function returns.
            let drained = unsafe { self.library.cuda_stream_synchronize(stream) };
            let normalized = enqueued.and_then(|value| drained.map(|()| value))?;
            unsafe { self.attention.complete()?; }
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
