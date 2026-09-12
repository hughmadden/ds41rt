//! Serialized native ViT/aligner owner. No request KV or text execution state.
use crate::v41_memory::{DeviceAllocation, LoadStream};
use crate::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer as Buffer, NativeLibrary, V41VisionOps};
use ds41rt_loader::{OfficialV41Catalog, V41Image, V41ImageGrid};

#[derive(Clone, Copy)]
#[repr(usize)]
enum Slot {
    Patches,
    Hidden,
    Norm,
    Qkv,
    Q,
    K,
    V,
    Attention,
    Gates,
    Activated,
    Linear,
    Scores,
    ValueOrProbability,
    Merged,
    AlignedA,
    AlignedB,
    Span,
    Frequencies,
}
pub(crate) struct VisionRuntime<'a> {
    // Drain the stream before dropping the BLAS handle and any buffer/weights.
    stream: LoadStream<'a>,
    ops: V41VisionOps<'a>,
    _workspace: DeviceAllocation<'a>,
    weights: NativeRtxTensors<'a>,
    buffers: Vec<DeviceAllocation<'a>>,
    capacity: usize,
    fp32_attention: bool,
    ready: Option<V41ImageGrid>,
}
impl<'a> VisionRuntime<'a> {
    fn names() -> Vec<String> {
        let mut names = vec![
            "vision.patch_embed.proj.weight".into(),
            "vision.patch_embed.proj.bias".into(),
            "vision.norm.weight".into(),
            "aligner.w1.weight".into(),
            "aligner.w1.bias".into(),
            "aligner.w2.weight".into(),
            "aligner.w2.bias".into(),
            "image_start".into(),
            "image_end".into(),
            "image_newline".into(),
        ];
        for layer in 0..32 {
            for suffix in [
                "norm1.weight",
                "attn.wqkv.weight",
                "attn.wqkv.bias",
                "attn.wo.weight",
                "attn.wo.bias",
                "norm2.weight",
                "mlp.w1.weight",
                "mlp.w2.weight",
            ] {
                names.push(format!("vision.blocks.{layer}.{suffix}"));
            }
        }
        names
    }
    fn buffer_sizes(capacity: usize) -> Result<Vec<usize>> {
        ensure!(
            (1..=V41VisionOps::MAX_PATCHES).contains(&capacity),
            "invalid vision patch capacity"
        );
        Ok(vec![
            capacity * 588 * 2,
            capacity * 2048,
            capacity * 2048,
            capacity * 6144,
            capacity * 2048,
            capacity * 2048,
            capacity * 2048,
            capacity * 2048,
            capacity * 5632 * 2,
            capacity * 2816 * 2,
            (capacity * 3072).max(1024 * 5120) * 4,
            capacity * 16 * 128 * 4,
            capacity * 16 * 128 * 2,
            1024 * 9216 * 2,
            1024 * 5120 * 2,
            1024 * 5120 * 2,
            1024 * 5120 * 2,
            16 * 4,
        ])
    }
    pub fn device_bytes(catalog: &OfficialV41Catalog, capacity: usize) -> Result<usize> {
        let names = Self::names();
        for name in &names {
            ensure!(
                catalog.tensor(name)?.metadata.dtype == ds41rt_core::DType::Bf16,
                "vision tensor must retain BF16: {name}"
            );
        }
        Ok(NativeRtxTensors::plan(catalog, &names)?
            + Self::buffer_sizes(capacity)?.iter().sum::<usize>()
            + V41VisionOps::WORKSPACE_BYTES)
    }
    pub fn new(
        lib: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        capacity: usize,
        budget: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(catalog, capacity)? <= budget,
            "vision owner exceeds device budget"
        );
        let names = Self::names();
        let weights = NativeRtxTensors::load(
            lib,
            catalog,
            &names,
            NativeRtxTensors::plan(catalog, &names)?,
            16 << 20,
        )?;
        let workspace = DeviceAllocation::new(lib, V41VisionOps::WORKSPACE_BYTES)?;
        let ops = unsafe { lib.v41_vision_ops(workspace.buffer)? };
        let buffers = Self::buffer_sizes(capacity)?
            .into_iter()
            .map(|n| DeviceAllocation::new(lib, n))
            .collect::<Result<Vec<_>>>()?;
        let stream = LoadStream {
            library: lib,
            raw: lib.cuda_stream_create()?,
        };
        let value = Self {
            stream,
            ops,
            _workspace: workspace,
            weights,
            buffers,
            capacity,
            fp32_attention: false,
            ready: None,
        };
        let frequencies: Vec<u8> = (0..16)
            .flat_map(|i| (1.0f32 / 10000.0f32.powf(i as f32 / 16.0)).to_le_bytes())
            .collect();
        lib.copy_h2d(value.buffer(Slot::Frequencies), &frequencies)?;
        Ok(value)
    }
    fn buffer(&self, slot: Slot) -> Buffer {
        self.buffers[slot as usize].buffer
    }
    fn linear(
        &self,
        input: Slot,
        name: &str,
        bias: bool,
        output: Slot,
        rows: usize,
        k: usize,
        n: usize,
    ) -> Result<()> {
        unsafe {
            self.ops.linear(
                self.buffer(input),
                self.weights.get(&format!("{name}.weight"))?,
                if bias {
                    Some(self.weights.get(&format!("{name}.bias"))?)
                } else {
                    None
                },
                self.buffer(Slot::Linear),
                self.buffer(output),
                rows,
                k,
                n,
                self.stream.raw,
            )
        }
    }
    fn norm(&self, input: Slot, name: &str, output: Slot, rows: usize) -> Result<()> {
        unsafe {
            self.ops.norm(
                self.buffer(input),
                self.weights.get(name)?,
                self.buffer(output),
                rows,
                self.stream.raw,
            )
        }
    }
    fn element(
        &self,
        input: Slot,
        other: Option<Slot>,
        output: Slot,
        rows: usize,
        mode: i32,
    ) -> Result<()> {
        unsafe {
            self.ops.element(
                self.buffer(input),
                other.map(|s| self.buffer(s)),
                self.buffer(output),
                rows,
                mode,
                self.stream.raw,
            )
        }
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    /// Returns a completed span borrowed from this owner's reusable device storage.
    /// Copy or consume it before encoding another image or dropping this owner.
    pub fn encode(&mut self, image: &V41Image) -> Result<Buffer> {
        self.encode_patches(image.patches(), image.grid(), None)
    }
    // Optional observer is only for numerical qualification, and sees completed
    // outputs. Normal execution queues the entire image before one final drain.
    fn encode_patches(
        &mut self,
        patches: &[u8],
        grid: V41ImageGrid,
        mut observer: Option<&mut dyn FnMut(&str, Buffer) -> Result<()>>,
    ) -> Result<Buffer> {
        self.ready = None;
        let rows = grid
            .vit_height
            .checked_mul(grid.vit_width)
            .ok_or_else(|| anyhow::anyhow!("vision grid overflow"))?;
        ensure!(
            grid.vit_height > 0
                && grid.vit_width > 0
                && rows <= self.capacity
                && grid.llm_height == grid.vit_height.div_ceil(3)
                && grid.llm_width == grid.vit_width.div_ceil(3)
                && grid.tokens() <= 1024
                && patches.len() == rows * 588 * 2,
            "invalid vision patches or grid"
        );
        let mut destination = self.buffer(Slot::Patches);
        destination.bytes = patches.len();
        self.stream.library.copy_h2d(destination, patches)?;
        let run = (|| -> Result<()> {
            let observe = |name: &str,
                           slot: Slot,
                           count: usize,
                           observer: &mut Option<&mut dyn FnMut(&str, Buffer) -> Result<()>>|
             -> Result<()> {
                if let Some(observer) = observer.as_deref_mut() {
                    self.synchronize()?;
                    let mut buffer = self.buffer(slot);
                    buffer.bytes = count;
                    observer(name, buffer)?;
                }
                Ok(())
            };
            self.linear(
                Slot::Patches,
                "vision.patch_embed.proj",
                true,
                Slot::Hidden,
                rows,
                588,
                1024,
            )?;
            observe("patch", Slot::Hidden, rows * 2048, &mut observer)?;
            for layer in 0..32 {
                let prefix = format!("vision.blocks.{layer}");
                self.norm(
                    Slot::Hidden,
                    &format!("{prefix}.norm1.weight"),
                    Slot::Norm,
                    rows,
                )?;
                self.linear(
                    Slot::Norm,
                    &format!("{prefix}.attn.wqkv"),
                    true,
                    Slot::Qkv,
                    rows,
                    1024,
                    3072,
                )?;
                unsafe {
                    self.ops.rope(
                        self.buffer(Slot::Qkv),
                        self.buffer(Slot::Frequencies),
                        self.buffer(Slot::Q),
                        self.buffer(Slot::K),
                        self.buffer(Slot::V),
                        grid.vit_height,
                        grid.vit_width,
                        self.stream.raw,
                    )?;
                }
                unsafe {
                    self.ops.attention(
                        self.buffer(Slot::Q),
                        self.buffer(Slot::K),
                        self.buffer(Slot::V),
                        self.buffer(Slot::Scores),
                        self.buffer(Slot::ValueOrProbability),
                        self.buffer(Slot::Linear),
                        self.buffer(Slot::Attention),
                        rows,
                        self.fp32_attention,
                        self.stream.raw,
                    )?;
                }
                self.linear(
                    Slot::Attention,
                    &format!("{prefix}.attn.wo"),
                    true,
                    Slot::Norm,
                    rows,
                    1024,
                    1024,
                )?;
                self.element(Slot::Hidden, Some(Slot::Norm), Slot::Hidden, rows, 0)?;
                self.norm(
                    Slot::Hidden,
                    &format!("{prefix}.norm2.weight"),
                    Slot::Norm,
                    rows,
                )?;
                self.linear(
                    Slot::Norm,
                    &format!("{prefix}.mlp.w1"),
                    false,
                    Slot::Gates,
                    rows,
                    1024,
                    5632,
                )?;
                self.element(Slot::Gates, None, Slot::Activated, rows, 1)?;
                self.linear(
                    Slot::Activated,
                    &format!("{prefix}.mlp.w2"),
                    false,
                    Slot::Norm,
                    rows,
                    2816,
                    1024,
                )?;
                self.element(Slot::Hidden, Some(Slot::Norm), Slot::Hidden, rows, 0)?;
                observe(
                    &format!("block-{layer}"),
                    Slot::Hidden,
                    rows * 2048,
                    &mut observer,
                )?;
            }
            self.norm(Slot::Hidden, "vision.norm.weight", Slot::Norm, rows)?;
            observe("norm", Slot::Norm, rows * 2048, &mut observer)?;
            let aligned = grid.llm_height * grid.llm_width;
            unsafe {
                self.ops.merge(
                    self.buffer(Slot::Norm),
                    self.buffer(Slot::Merged),
                    grid.vit_height,
                    grid.vit_width,
                    self.stream.raw,
                )?;
            }
            observe("merged", Slot::Merged, aligned * 18432, &mut observer)?;
            self.linear(
                Slot::Merged,
                "aligner.w1",
                true,
                Slot::AlignedA,
                aligned,
                9216,
                5120,
            )?;
            self.element(Slot::AlignedA, None, Slot::AlignedA, aligned, 2)?;
            self.linear(
                Slot::AlignedA,
                "aligner.w2",
                true,
                Slot::AlignedB,
                aligned,
                5120,
                5120,
            )?;
            observe("aligned", Slot::AlignedB, aligned * 10240, &mut observer)?;
            unsafe {
                self.ops.span(
                    self.buffer(Slot::AlignedB),
                    self.weights.get("image_start")?,
                    self.weights.get("image_newline")?,
                    self.weights.get("image_end")?,
                    self.buffer(Slot::Span),
                    grid.llm_height,
                    grid.llm_width,
                    self.stream.raw,
                )?;
            }
            Ok(())
        })();
        // Drain even on a failed launch before allowing buffers to be reused.
        run.and(self.synchronize())?;
        self.ready = Some(grid);
        let mut output = self.buffer(Slot::Span);
        output.bytes = grid.tokens() * 10240;
        Ok(output)
    }
}

#[cfg(test)]
mod tests;
