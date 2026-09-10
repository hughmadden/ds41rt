//! Official backbone mHC parameters around attention and routed FFN execution.
use crate::v41_hc::{HcBinding, HcSublayer};
use crate::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Result};
use ds41rt_ffi::NativeLibrary;
use ds41rt_loader::OfficialV41Catalog;
pub(crate) struct BackboneHcWeights<'a> {
    library: &'a NativeLibrary,
    layer: usize,
    tensors: NativeRtxTensors<'a>,
}
impl<'a> BackboneHcWeights<'a> {
    pub fn layer(&self) -> usize { self.layer }
    pub(crate) fn prepare_bindings<'w>(
        &'w self,
        attention: &HcSublayer<'w, 'a>,
        ffn: &HcSublayer<'w, 'a>,
    ) -> Result<[HcBinding<'w, 'a>; 2]> {
        Ok([
            attention.prepare_binding(&self.tensors, Self::names(self.layer, true)?)?,
            ffn.prepare_binding(&self.tensors, Self::names(self.layer, false)?)?,
        ])
    }
    fn names(layer: usize, attention: bool) -> Result<[String; 4]> {
        ensure!(layer < 40, "invalid backbone mHC layer");
        let kind = if attention { "attn" } else { "ffn" };
        Ok([
            format!("layers.{layer}.hc_{kind}_fn"),
            format!("layers.{layer}.hc_{kind}_scale"),
            format!("layers.{layer}.hc_{kind}_base"),
            format!("layers.{layer}.{kind}_norm.weight"),
        ])
    }
    pub fn device_bytes(catalog: &OfficialV41Catalog, layer: usize) -> Result<usize> {
        let names = [Self::names(layer, true)?, Self::names(layer, false)?].concat();
        let bytes = NativeRtxTensors::plan(catalog, &names)?;
        ensure!(bytes == 3_952_856, "unexpected backbone mHC tensor sizes");
        Ok(bytes)
    }
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: usize,
        budget: usize,
        staging: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(catalog, layer)? <= budget,
            "backbone mHC weights exceed budget"
        );
        let names = [Self::names(layer, true)?, Self::names(layer, false)?].concat();
        Ok(Self {
            library,
            layer,
            tensors: NativeRtxTensors::load(library, catalog, &names, budget, staging)?,
        })
    }
    pub fn block(
        &self,
        capacity: usize,
        budget: usize,
    ) -> Result<crate::v41_block::BackboneBlockWave<'_, 'a>> {
        ensure!(
            crate::v41_block::BackboneBlockWave::device_bytes(capacity)? <= budget,
            "backbone block exceeds budget"
        );
        Ok(crate::v41_block::BackboneBlockWave::new(
            self.library,
            self.layer,
            self.boundary(true, capacity, HcSublayer::device_bytes(capacity)?)?,
            self.boundary(false, capacity, HcSublayer::device_bytes(capacity)?)?,
            capacity,
        ))
    }
    pub fn boundary(
        &self,
        attention: bool,
        capacity: usize,
        budget: usize,
    ) -> Result<HcSublayer<'_, 'a>> {
        HcSublayer::new(
            self.library,
            &self.tensors,
            Self::names(self.layer, attention)?,
            capacity,
            budget,
        )
    }
}
