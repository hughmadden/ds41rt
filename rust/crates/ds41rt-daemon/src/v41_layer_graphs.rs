//! Captured shapes per backbone layer, sharing the containing lane's buffers.
use anyhow::{ensure, Result};
use ds41rt_ffi::NativeLibrary;
use std::ffi::c_void;
use std::collections::BTreeMap;

struct Entry<'w, W> {
    weights: &'w W,
    graph: *mut c_void,
    rows: u32,
}

/// The containing owner drains graph launches before replacement or destruction.
/// References retain every captured weight owner until its graph is destroyed.
pub(crate) struct LayerGraphs<'w, 'a, W> {
    library: &'a NativeLibrary,
    entries: [Option<Entry<'w, W>>; 40],
    retained: [BTreeMap<u32, Entry<'w, W>>; 40],
    retain_small: bool,
}
impl<'w, 'a, W> LayerGraphs<'w, 'a, W> {
    pub fn new(library: &'a NativeLibrary) -> Self {
        Self {
            library,
            entries: std::array::from_fn(|_| None),
            retained: std::array::from_fn(|_| BTreeMap::new()),
            retain_small: false,
        }
    }
    pub fn get(&self, layer: usize, weights: &W) -> Option<(*mut c_void, u32)> {
        let entry = self.entries.get(layer)?.as_ref()?;
        std::ptr::eq(entry.weights, weights).then_some((entry.graph, entry.rows))
    }
    /// A decode lane contains at most eight requests with six rows each.
    /// Retain those shapes plus at most one current large-prefill graph.
    pub fn enable_small_shapes(&mut self) { self.retain_small = true; }
    pub fn get_shape(&self, layer: usize, weights: &W, rows: u32) -> Option<(*mut c_void, u32)> {
        if let Some(graph) = self.get(layer, weights).filter(|(_, n)| *n == rows) { return Some(graph); }
        let entry = self.retained.get(layer)?.get(&rows)?;
        std::ptr::eq(entry.weights, weights).then_some((entry.graph, entry.rows))
    }
    /// # Safety
    /// All uses of the replaced graph are complete. On success this cache owns
    /// graph, captured for exactly these weights/rows and this lane's buffers.
    /// On failure the caller still owns graph.
    pub unsafe fn insert(
        &mut self,
        layer: usize,
        weights: &'w W,
        rows: u32,
        graph: *mut c_void,
    ) -> Result<()> {
        ensure!(
            layer < 40 && rows > 0 && rows <= 4096 && !graph.is_null(),
            "invalid layer graph binding"
        );
        let different_owner = self.entries[layer].as_ref().is_some_and(|e| !std::ptr::eq(e.weights, weights))
            || self.retained[layer].values().any(|e| !std::ptr::eq(e.weights, weights));
        if !self.retain_small || different_owner {
            unsafe { self.remove(layer)?; }
        } else {
            if let Some(old) = self.retained[layer].remove(&rows) {
                unsafe { self.library.cuda_graph_exec_destroy(old.graph)?; }
            }
            if let Some(old) = self.entries[layer].take() {
                if old.rows <= 48 && old.rows != rows {
                    if let Some(replaced) = self.retained[layer].insert(old.rows, old) {
                        unsafe { self.library.cuda_graph_exec_destroy(replaced.graph)?; }
                    }
                } else {
                    unsafe { self.library.cuda_graph_exec_destroy(old.graph)?; }
                }
            }
        }
        self.entries[layer] = Some(Entry {
            weights,
            graph,
            rows,
        });
        tracing::debug!(target: "ds41rt::graph_capture", layer, rows,
            owner=std::any::type_name::<W>(), retained=self.retained[layer].len()+1,
            "native layer graph captured");
        Ok(())
    }
    /// # Safety
    /// All launches using the layer's graph have completed.
    pub unsafe fn remove(&mut self, layer: usize) -> Result<()> {
        ensure!(layer < 40, "invalid layer graph index");
        let mut error = None;
        for entry in self.entries[layer].take().into_iter()
            .chain(std::mem::take(&mut self.retained[layer]).into_values()) {
            if let Err(e) = unsafe { self.library.cuda_graph_exec_destroy(entry.graph) } { error.get_or_insert(e); }
        }
        error.map_or(Ok(()), Err)
    }
    /// # Safety
    /// All launches using any retained graph have completed.
    pub unsafe fn clear(&mut self) -> Result<()> {
        let mut first_error = None;
        for layer in 0..40 {
            if let Err(error) = unsafe { self.remove(layer) } {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}
// Destruction is explicit in the containing wave after its stream drains.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v41_memory::{DeviceAllocation, LoadStream};

    #[test]
    fn cuda_small_shape_bank_replays_changes_and_bounds_large_shapes() -> Result<()> {
        let Some(path) = std::env::var_os("DS41RT_LAYER_GRAPH_TEST_LIBRARY") else {
            eprintln!("skip CUDA shape bank test: library unset");
            return Ok(());
        };
        let library = unsafe { NativeLibrary::load(path)? };
        let weights = DeviceAllocation::new(&library, 128 * 4)?;
        let other = DeviceAllocation::new(&library, 128 * 4)?;
        let output = DeviceAllocation::new(&library, 128 * 4)?;
        let stream = LoadStream { library: &library, raw: library.cuda_stream_create()? };
        let mut bank = LayerGraphs::new(&library);
        bank.enable_small_shapes();
        let mut handles = BTreeMap::new();
        for (cycle, rows) in (1..=48).chain([80, 128, 2, 6, 3, 2, 6, 48]).enumerate() {
            let expected = vec![(cycle + 1) as u8; 128 * 4];
            library.copy_h2d(weights.buffer, &expected)?;
            if bank.get_shape(0, &weights, rows).is_none() {
                unsafe {
                    library.cuda_graph_begin_capture(stream.raw)?;
                    library.copy_d2d_async(output.buffer, weights.buffer, rows as usize * 4, stream.raw)?;
                    let graph = library.cuda_graph_end_capture(stream.raw)?;
                    bank.insert(0, &weights, rows, graph)?;
                }
            }
            let (graph, _) = bank.get_shape(0, &weights, rows).unwrap();
            if rows <= 48 {
                assert_eq!(*handles.entry(rows).or_insert(graph), graph);
            }
            assert!(bank.get_shape(0, &other, rows).is_none());
            unsafe {
                library.cuda_graph_launch(graph, stream.raw)?;
                library.cuda_stream_synchronize(stream.raw)?;
            }
            let mut actual = vec![0; rows as usize * 4];
            let mut view = output.buffer;
            view.bytes = actual.len();
            library.copy_d2h(&mut actual, view)?;
            assert_eq!(actual, expected[..actual.len()]);
            assert!(bank.retained[0].len() + usize::from(bank.entries[0].is_some()) <= 49);
        }
        assert!(bank.get_shape(0, &weights, 80).is_none());
        assert!(bank.get_shape(0, &weights, 128).is_some());
        unsafe {
            library.cuda_graph_begin_capture(stream.raw)?;
            library.copy_d2d_async(output.buffer, other.buffer, 8, stream.raw)?;
            let graph = library.cuda_graph_end_capture(stream.raw)?;
            bank.insert(0, &other, 2, graph)?;
        }
        assert!(bank.get_shape(0, &weights, 2).is_none());
        assert!(bank.get_shape(0, &weights, 6).is_none());
        assert!(bank.get_shape(0, &other, 2).is_some());
        unsafe { bank.clear()?; }
        assert!(bank.entries.iter().all(Option::is_none));
        assert!(bank.retained.iter().all(BTreeMap::is_empty));
        Ok(())
    }

    #[test]
    fn cuda_layer_graphs_reuse_storage_and_retain_exact_weight_bindings() -> Result<()> {
        let Some(path) = std::env::var_os("DS41RT_LAYER_GRAPH_TEST_LIBRARY") else {
            eprintln!("skip CUDA layer graph test: DS41RT_LAYER_GRAPH_TEST_LIBRARY unset");
            return Ok(());
        };
        let library = unsafe { NativeLibrary::load(path)? };
        // Real CUDA copies isolate graph ownership from SM120 FP8 arithmetic.
        let weights = (0..40)
            .map(|_| DeviceAllocation::new(&library, 4096 * 4))
            .collect::<Result<Vec<_>>>()?;
        let replacement = DeviceAllocation::new(&library, 4096 * 4)?;
        let streams = (0..2)
            .map(|_| {
                Ok(LoadStream {
                    library: &library,
                    raw: library.cuda_stream_create()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let outputs = (0..2)
            .map(|_| DeviceAllocation::new(&library, 4096 * 4))
            .collect::<Result<Vec<_>>>()?;
        let mut banks = [LayerGraphs::new(&library), LayerGraphs::new(&library)];
        let mut captures = 0;
        let mut replays = 0;
        for rows in [1u32, 80, 4096] {
            for cycle in 0..2u32 {
                for layer in 0..40 {
                    let expected: Vec<u8> = (0..rows)
                        .flat_map(|row| (row + 4096 * (layer as u32 + 40 * cycle)).to_ne_bytes())
                        .collect();
                    library.copy_h2d(weights[layer].buffer, &expected)?;
                    let mut handles = [std::ptr::null_mut(); 2];
                    for lane in 0..2 {
                        let bank = &mut banks[lane];
                        let current = bank.get(layer, &weights[layer]);
                        if current.is_none_or(|(_, count)| count != rows) {
                            unsafe {
                                bank.remove(layer)?;
                                library.cuda_graph_begin_capture(streams[lane].raw)?;
                                library.copy_d2d_async(
                                    outputs[lane].buffer,
                                    weights[layer].buffer,
                                    expected.len(),
                                    streams[lane].raw,
                                )?;
                                let graph = library.cuda_graph_end_capture(streams[lane].raw)?;
                                bank.insert(layer, &weights[layer], rows, graph)?;
                            }
                            captures += 1;
                        } else {
                            // A repeated same-shape pass must keep the graph handle.
                            assert_eq!(cycle, 1);
                        }
                        let (graph, count) = bank.get(layer, &weights[layer]).unwrap();
                        assert_eq!(count, rows);
                        assert!(bank.get(layer, &replacement).is_none());
                        handles[lane] = graph;
                        unsafe {
                            library.cuda_graph_launch(graph, streams[lane].raw)?;
                        }
                        replays += 1;
                    }
                    for lane in 0..2 {
                        unsafe {
                            library.cuda_stream_synchronize(streams[lane].raw)?;
                        }
                        let mut actual = vec![0; expected.len()];
                        library.copy_d2h(&mut actual, outputs[lane].buffer)?;
                        assert_eq!(actual, expected);
                        assert_eq!(
                            banks[lane].get(layer, &weights[layer]).unwrap().0,
                            handles[lane]
                        );
                    }
                }
                assert!(banks
                    .iter()
                    .all(|bank| bank.entries.iter().flatten().count() == 40));
            }
        }
        assert_eq!(captures, 240);
        assert_eq!(replays, 480);
        let bank = &mut banks[0];
        let original = bank.get(0, &weights[0]).unwrap();
        // Reject invalid metadata without touching the existing graph.
        unsafe {
            assert!(bank.insert(40, &replacement, 1, original.0).is_err());
            assert!(bank.insert(0, &replacement, 0, original.0).is_err());
            assert!(bank.insert(0, &replacement, 4097, original.0).is_err());
            assert!(bank
                .insert(0, &replacement, 1, std::ptr::null_mut())
                .is_err());
        }
        assert_eq!(bank.get(0, &weights[0]), Some(original));
        // Replacing an owner at the same layer evicts its old capture only.
        library.copy_h2d(replacement.buffer, &123456u32.to_ne_bytes())?;
        unsafe {
            library.cuda_graph_begin_capture(streams[0].raw)?;
            library.copy_d2d_async(outputs[0].buffer, replacement.buffer, 4, streams[0].raw)?;
            let graph = library.cuda_graph_end_capture(streams[0].raw)?;
            bank.insert(0, &replacement, 1, graph)?;
            assert!(bank.get(0, &weights[0]).is_none());
            assert!(bank.get(39, &weights[39]).is_some());
            library.cuda_graph_launch(bank.get(0, &replacement).unwrap().0, streams[0].raw)?;
            library.cuda_stream_synchronize(streams[0].raw)?;
        }
        let mut actual = [0u8; 4];
        library.copy_d2h(&mut actual, outputs[0].buffer)?;
        assert_eq!(u32::from_ne_bytes(actual), 123456);
        for bank in &mut banks {
            unsafe {
                bank.clear()?;
            }
            assert!(bank.entries.iter().all(Option::is_none));
            unsafe {
                bank.clear()?;
            }
        }
        eprintln!("PASS 240 captures, 480 replays, two independent lanes, 40 layers, rows 1/80/4096, changed payloads, owner replacement and invalid binding guards");
        Ok(())
    }
}
