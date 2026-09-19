//! CPU tests for the compressor producer capture: completed-identity guards,
//! pure extent planning, producer-exact predecessor resolution, and file
//! writing through a fake device-copy callback. No device, bank, or
//! environment is involved.
use super::input_trace::*;
use super::{CompressorLease, CompressorLatentRow};
use anyhow::Result;
use ds41rt_ffi::Ds41rtDeviceBuffer;

/// Fake device: registrations stand in for live allocations, reads are logged,
/// and every destination byte is a pure function of its device address so file
/// contents pin the exact source extents that were copied.
#[derive(Default)]
struct FakeDevice {
    allocations: Vec<(usize, usize)>,
    reads: Vec<(usize, usize, usize)>,
    contents: std::collections::BTreeMap<usize, Vec<u8>>,
}
impl FakeDevice {
    fn register(&mut self, bytes: usize) -> Ds41rtDeviceBuffer {
        let base = 0x3000_0000 + self.allocations.len() * 0x2000_0000;
        self.allocations.push((base, bytes));
        Ds41rtDeviceBuffer {
            ptr: base as *mut std::ffi::c_void,
            bytes,
            device_id: 0,
            flags: 0,
        }
    }
    fn copy(&mut self, src: Ds41rtDeviceBuffer, dst: &mut [u8]) -> Result<()> {
        let (base, bytes) = self
            .allocations
            .iter()
            .copied()
            .find(|&(base, limit)| {
                let address = src.ptr as usize;
                (address >= base) && (address < base + limit)
            })
            .ok_or_else(|| anyhow::anyhow!("read targets no registered allocation"))?;
        let offset = src.ptr as usize - base;
        anyhow::ensure!(
            offset + src.bytes <= bytes,
            "fake read [{offset}, {}) exceeds {} byte allocation",
            offset + src.bytes,
            bytes
        );
        self.reads.push((base, offset, src.bytes));
        if let Some(contents) = self.contents.get(&base) {
            dst.copy_from_slice(&contents[offset..offset + src.bytes]);
            return Ok(());
        }
        let address = src.ptr as usize;
        for (index, byte) in dst.iter_mut().enumerate() {
            *byte = ((address + index) & 0xff) as u8;
        }
        Ok(())
    }
    fn expected(&self, base: usize, offset: usize, bytes: usize) -> Vec<u8> {
        (0..bytes)
            .map(|index| ((base + offset + index) & 0xff) as u8)
            .collect()
    }
}

struct ChunkSpec {
    /// First flattened wave row of this chunk (the prepared offset).
    index: usize,
    request_id: u64,
    slot: usize,
    generation: u64,
    version: u64,
    position: u64,
    tokens: u32,
}

/// Producer-exact host descriptor staging for a ratio-two wave.
fn staged_descriptors(chunks: &[ChunkSpec], slot_count: usize) -> Vec<u64> {
    let mut descriptors = Vec::new();
    for chunk in chunks {
        for j in 0..chunk.tokens as usize {
            let pos = chunk.position + j as u64;
            let row = chunk.index + j;
            descriptors.push(if pos % 2 == 0 {
                DESCRIPTOR_SENTINEL
            } else if j == 0 {
                chunk.slot as u64
            } else {
                (slot_count + row - 1) as u64
            });
        }
    }
    descriptors
}

fn trace_chunks(chunks: &[ChunkSpec]) -> Vec<CompressorTraceChunk> {
    chunks
        .iter()
        .map(|chunk| CompressorTraceChunk {
            index: chunk.index,
            request_id: chunk.request_id,
            slot: chunk.slot,
            generation: chunk.generation,
            version: chunk.version,
            position: chunk.position,
            tokens: chunk.tokens,
            offset: chunk.index,
        })
        .collect()
}

/// Producer-exact completed latents: odd positions (ratio two) or every
/// position (ratio one) complete a latent whose first token is `pos + 1 - ratio`.
fn latents(chunks: &[ChunkSpec], ratio: usize) -> Vec<CompressorLatentRow> {
    let mut latents = Vec::new();
    for chunk in chunks {
        for j in 0..chunk.tokens as usize {
            let pos = chunk.position + j as u64;
            let row = chunk.index + j;
            if ratio == 1 || pos % 2 == 1 {
                latents.push(CompressorLatentRow {
                    lease: CompressorLease {
                        owner: 1,
                        slot: chunk.slot,
                        generation: chunk.generation,
                    },
                    source_row: row as u32,
                    position: pos + 1 - ratio as u64,
                });
            }
        }
    }
    latents
}

struct Fixture {
    device: FakeDevice,
    buffers: CompressorTraceBuffers,
    weights: CompressorTraceWeights,
    chunks: Vec<CompressorTraceChunk>,
    rows: usize,
    slot_count: usize,
    descriptors_host: Vec<u64>,
    input: usize,
}

/// Build a completed wave fixture; device descriptor bytes mirror the staged
/// host upload unless a test overrides them.
fn build_fixture(chunks: &[ChunkSpec], slot_count: usize, ratio: usize) -> Fixture {
    let mut device = FakeDevice::default();
    let rows: usize = chunks.iter().map(|chunk| chunk.tokens as usize).sum();
    let capacity = rows.max(4);
    let input = device.register(capacity * 10240);
    let projected = device.register(capacity * if ratio == 2 { 2048 } else { 1024 });
    let scores = (ratio == 2).then(|| device.register(capacity * 2048));
    let descriptors = (ratio == 2).then(|| device.register(capacity * 8));
    let output = device.register(capacity * 1024);
    let frequencies = device.register(capacity * 256);
    let positions = device.register(capacity * 8);
    let kv_values = device.register(capacity * 256);
    let kv_scales = device.register(capacity * 32);
    let pending_kv = (ratio == 2).then(|| device.register(slot_count * 2048));
    let pending_scores = (ratio == 2).then(|| device.register(slot_count * 2048));
    let descriptors_host = if ratio == 2 {
        staged_descriptors(chunks, slot_count)
    } else {
        Vec::new()
    };
    if let Some(descriptors) = descriptors {
        device.contents.insert(
            descriptors.ptr as usize,
            descriptors_host
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect(),
        );
    }
    let wkv = device.register(512 * 5120 * 2);
    let norm = device.register(512 * 2);
    let wgate = (ratio == 2).then(|| device.register(512 * 5120 * 2));
    Fixture {
        buffers: CompressorTraceBuffers {
            input,
            projected,
            scores,
            output,
            frequencies,
            positions,
            descriptors,
            kv_values,
            kv_scales,
            pending_kv,
            pending_scores,
        },
        weights: CompressorTraceWeights {
            wkv,
            norm,
            wgate,
            names: [
                "layers.2.attn.compressor.wkv.weight".into(),
                "layers.2.attn.compressor.norm.weight".into(),
                "layers.2.attn.compressor.wgate.weight".into(),
            ],
        },
        device,
        chunks: trace_chunks(chunks),
        rows,
        slot_count,
        descriptors_host,
        input: input.ptr as usize,
    }
}

fn scratch_directory(label: &str) -> std::path::PathBuf {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "ds41rt-compressor-input-trace-{label}-{}-{unique}",
        std::process::id()
    ))
}

const WEIGHTS_DIRECTORY: &str = "weights";

fn manifest_of(directory: &std::path::Path, layer: usize) -> serde_json::Value {
    let bytes = std::fs::read(directory.join(format!("layer{layer}-compressor-inputs.json")))
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

fn find_read<'a>(reads: &'a [PlannedRead], name: &str) -> Option<&'a PlannedRead> {
    reads.iter().find(|read| read.name == name)
}

#[test]
fn guards_reject_pending_work_and_missing_or_foreign_identity() {
    // Pending query or commit work is never traced: the producer output could
    // still be mutating.
    assert!(ensure_completed_trace_ready(true, false, Some(7), 7, 2, 2)
        .unwrap_err()
        .to_string()
        .contains("query"));
    assert!(ensure_completed_trace_ready(false, true, Some(7), 7, 2, 2)
        .unwrap_err()
        .to_string()
        .contains("commit"));
    // A wave with no completed output has no identity to trace.
    assert!(ensure_completed_trace_ready(false, false, None, 7, 2, 2)
        .unwrap_err()
        .to_string()
        .contains("incomplete"));
    // Foreign owners and mismatched layers are rejected.
    assert!(ensure_completed_trace_ready(false, false, Some(8), 7, 2, 2)
        .unwrap_err()
        .to_string()
        .contains("owner"));
    assert!(ensure_completed_trace_ready(false, false, Some(7), 7, 2, 8)
        .unwrap_err()
        .to_string()
        .contains("layer"));
    ensure_completed_trace_ready(false, false, Some(7), 7, 2, 2).unwrap();
}

#[test]
fn planner_records_exact_live_extents_and_unique_files() {
    let specs = [ChunkSpec {
        index: 0,
        request_id: 41,
        slot: 1,
        generation: 3,
        version: 7,
        position: 40,
        tokens: 2,
    }];
    let fixture = build_fixture(&specs, 4, 2);
    let reads = plan_compressor_reads(2, 2, fixture.rows, 4, &fixture.buffers, None).unwrap();
    let expect = |name: &str, dtype: &str, shape: &[usize], bytes: usize| {
        let read = find_read(&reads, name).unwrap();
        assert_eq!(read.dtype, dtype, "{name}");
        assert_eq!(read.shape, shape, "{name}");
        assert_eq!(read.bytes, bytes, "{name}");
        assert_eq!(read.file, format!("layer2-compressor-{name}.bin"));
    };
    expect("input", "bfloat16", &[2, 5120], 2 * 10240);
    expect("projected", "float32", &[2, 512], 2 * 2048);
    expect("scores", "float32", &[2, 512], 2 * 2048);
    expect("output", "bfloat16", &[2, 512], 2 * 1024);
    expect("frequencies", "float32", &[2, 32, 2], 2 * 256);
    expect("positions", "uint64", &[2], 16);
    expect("descriptors-device", "uint64", &[2], 16);
    expect("kv-values", "fp4e2m1", &[2, 512], 2 * 256);
    expect("kv-scales", "fp8e4m3", &[2, 32], 2 * 32);
    // Pending planes are whole [slot_count, 512] state, not live rows.
    expect("pending-kv", "float32", &[4, 512], 4 * 2048);
    expect("pending-scores", "float32", &[4, 512], 4 * 2048);
    // With weights supplied, the fixed checkpoint extents join the plan.
    let reads =
        plan_compressor_reads(2, 2, fixture.rows, 4, &fixture.buffers, Some(&fixture.weights))
            .unwrap();
    let weight = |name: &str, bytes: usize| {
        let read = find_read(&reads, name).unwrap();
        assert_eq!(read.dtype, "bfloat16");
        assert_eq!(read.bytes, bytes);
    };
    weight("wkv-weight", 512 * 5120 * 2);
    weight("wgate-weight", 512 * 5120 * 2);
    weight("norm-weight", 512 * 2);
    // Every file name is unique so create_new can never collide.
    let names: std::collections::HashSet<_> = reads.iter().map(|read| read.file.clone()).collect();
    assert_eq!(names.len(), reads.len());
    // Ratio one plans BF16 projection and no pooling scratch at all.
    let ratio1 = build_fixture(
        &[ChunkSpec {
            index: 0,
            request_id: 41,
            slot: 0,
            generation: 1,
            version: 1,
            position: 40,
            tokens: 2,
        }],
        4,
        1,
    );
    let reads = plan_compressor_reads(20, 1, ratio1.rows, 4, &ratio1.buffers, None).unwrap();
    assert!(reads.iter().all(|read| !read.name.starts_with("pending")));
    assert!(reads.iter().all(|read| read.name != "scores"));
    assert!(reads.iter().all(|read| read.name != "descriptors-device"));
    assert_eq!(
        reads
            .iter()
            .find(|read| read.name == "projected")
            .unwrap()
            .bytes,
        2 * 1024
    );
    // Ratio-one weights plan without wgate; carrying one is an identity error.
    let mut wrong_ratio1 = ratio1.weights.clone();
    wrong_ratio1.wgate = Some(wrong_ratio1.wkv);
    let error =
        plan_compressor_reads(20, 1, ratio1.rows, 4, &ratio1.buffers, Some(&wrong_ratio1))
            .unwrap_err();
    assert!(error.to_string().contains("ratio-one weights own wgate"), "{error}");
}

#[test]
fn planner_rejects_bad_slots_extents_and_budget() {
    let fixture = build_fixture(
        &[ChunkSpec {
            index: 0,
            request_id: 41,
            slot: 1,
            generation: 3,
            version: 7,
            position: 40,
            tokens: 2,
        }],
        4,
        2,
    );
    // Zero rows never select and slot counts past the device bound fail.
    assert!(plan_compressor_reads(2, 2, 0, 4, &fixture.buffers, None).is_err());
    assert!(plan_compressor_reads(2, 2, 2, 17, &fixture.buffers, None).is_err());
    // Missing ratio-two scratch fails loudly instead of skipping the file.
    let no_scores = CompressorTraceBuffers {
        scores: None,
        ..fixture.buffers
    };
    let error = plan_compressor_reads(2, 2, 2, 4, &no_scores, None).unwrap_err();
    assert!(error.to_string().contains("scores absent"), "{error}");
    // One byte short of the live input extent.
    let undersized = CompressorTraceBuffers {
        input: Ds41rtDeviceBuffer {
            bytes: 2 * 10240 - 1,
            ..fixture.buffers.input
        },
        ..fixture.buffers
    };
    let error = plan_compressor_reads(2, 2, 2, 4, &undersized, None).unwrap_err();
    assert!(error.to_string().contains("input"), "{error}");
    // Row arithmetic overflow fails before any allocation check.
    let huge = CompressorTraceBuffers {
        input: Ds41rtDeviceBuffer {
            bytes: usize::MAX,
            ..fixture.buffers.input
        },
        ..fixture.buffers
    };
    let error = plan_compressor_reads(2, 2, usize::MAX, 4, &huge, None).unwrap_err();
    assert!(error.to_string().contains("overflow"), "{error}");
    // A capture over the fixed budget fails explicitly instead of dumping.
    // 4220 rows of ratio-two operands (15,920 bytes per row) plus the whole
    // 16-slot pending planes cross the 64 MiB bound.
    let rows = 4220usize;
    let big = CompressorTraceBuffers {
        input: Ds41rtDeviceBuffer { bytes: rows * 10240, ..fixture.buffers.input },
        projected: Ds41rtDeviceBuffer { bytes: rows * 2048, ..fixture.buffers.projected },
        scores: Some(Ds41rtDeviceBuffer { bytes: rows * 2048, ..fixture.buffers.scores.unwrap() }),
        output: Ds41rtDeviceBuffer { bytes: rows * 1024, ..fixture.buffers.output },
        frequencies: Ds41rtDeviceBuffer { bytes: rows * 256, ..fixture.buffers.frequencies },
        positions: Ds41rtDeviceBuffer { bytes: rows * 8, ..fixture.buffers.positions },
        descriptors: Some(Ds41rtDeviceBuffer {
            bytes: rows * 8,
            ..fixture.buffers.descriptors.unwrap()
        }),
        kv_values: Ds41rtDeviceBuffer { bytes: rows * 256, ..fixture.buffers.kv_values },
        kv_scales: Ds41rtDeviceBuffer { bytes: rows * 32, ..fixture.buffers.kv_scales },
        pending_kv: Some(Ds41rtDeviceBuffer {
            bytes: 16 * 2048,
            ..fixture.buffers.pending_kv.unwrap()
        }),
        pending_scores: Some(Ds41rtDeviceBuffer {
            bytes: 16 * 2048,
            ..fixture.buffers.pending_scores.unwrap()
        }),
    };
    let error = plan_compressor_reads(2, 2, rows, 16, &big, None).unwrap_err();
    assert!(error.to_string().contains("budget"), "{error}");
    // A weight tensor with the wrong extent is an identity failure.
    let wrong_weights = CompressorTraceWeights {
        wkv: Ds41rtDeviceBuffer {
            bytes: 512 * 5120 * 2 - 2,
            ..fixture.weights.wkv
        },
        ..fixture.weights.clone()
    };
    let error = plan_compressor_reads(2, 2, 2, 4, &fixture.buffers, Some(&wrong_weights))
        .unwrap_err();
    assert!(error.to_string().contains("expected"), "{error}");
    // Ratio-one waves must not carry ratio-two scratch.
    let mut mixed = build_fixture(
        &[ChunkSpec {
            index: 0,
            request_id: 41,
            slot: 0,
            generation: 1,
            version: 1,
            position: 40,
            tokens: 2,
        }],
        4,
        1,
    );
    mixed.buffers.descriptors = Some(mixed.buffers.positions);
    let error = plan_compressor_reads(20, 1, mixed.rows, 4, &mixed.buffers, None).unwrap_err();
    assert!(error.to_string().contains("ratio-two scratch"), "{error}");
}

#[test]
fn write_inputs_maps_singleton_and_two_request_waves() -> Result<()> {
    // Singleton: one request, positions 40-41, one completed latent whose
    // first token is 40 (logical compressed row 20, the p41 evidence shape).
    let specs = [ChunkSpec {
        index: 0,
        request_id: 41,
        slot: 1,
        generation: 3,
        version: 7,
        position: 40,
        tokens: 2,
    }];
    let mut fixture = build_fixture(&specs, 4, 2);
    let completed = latents(&specs, 2);
    let directory = scratch_directory("singleton");
    std::fs::create_dir_all(&directory)?;
    let weights_directory = directory.join(WEIGHTS_DIRECTORY);
    let captured = write_compressor_inputs(
        &mut |src, dst| fixture.device.copy(src, dst),
        &directory,
        2,
        2,
        fixture.slot_count,
        99,
        &fixture.chunks,
        &completed,
        Some(&fixture.descriptors_host),
        &fixture.buffers,
        &weights_directory,
        Some(&fixture.weights),
    )?;
    let manifest = manifest_of(&directory, 2);
    assert_eq!(manifest["kind"], serde_json::json!("compressor-inputs"));
    assert_eq!(manifest["layer"], serde_json::json!(2));
    assert_eq!(manifest["ratio"], serde_json::json!(2));
    assert_eq!(manifest["rows"], serde_json::json!(2));
    assert_eq!(manifest["slot_count"], serde_json::json!(4));
    assert_eq!(manifest["proposal_snapshot"], serde_json::json!(99));
    assert_eq!(manifest["captured_bytes"].as_u64(), Some(captured as u64));
    // Actual chunk identity with lease, version and absolute positions.
    let chunks = manifest["chunks"].as_array().unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0]["request_id"], serde_json::json!(41));
    assert_eq!(
        chunks[0]["lease"],
        serde_json::json!({"slot": 1, "generation": 3})
    );
    assert_eq!(chunks[0]["version"], serde_json::json!(7));
    assert_eq!(chunks[0]["prepared_offset"], serde_json::json!(0));
    assert_eq!(chunks[0]["flattened_row_range"], serde_json::json!([0, 2]));
    assert_eq!(chunks[0]["absolute_positions"], serde_json::json!([40, 41]));
    // Row 0 is an even position: sentinel, no latent. Row 1 is odd with j=1
    // and pools earlier wave row 0, completing the p41 latent.
    let rows = manifest["wave_rows"].as_array().unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["request_id"], serde_json::json!(41));
    assert_eq!(rows[0]["absolute_position"], serde_json::json!(40));
    assert_eq!(
        rows[0]["device_descriptor"],
        serde_json::json!(DESCRIPTOR_SENTINEL)
    );
    assert_eq!(
        rows[0]["predecessor"],
        serde_json::json!({"kind": "invalid_sentinel"})
    );
    assert_eq!(rows[0]["completed_latent"], serde_json::json!(null));
    assert_eq!(rows[1]["absolute_position"], serde_json::json!(41));
    assert_eq!(rows[1]["device_descriptor"], serde_json::json!(4 + 1 - 1));
    assert_eq!(
        rows[1]["predecessor"],
        serde_json::json!({"kind": "earlier_wave_row", "wave_row": 0})
    );
    assert_eq!(rows[1]["completed_latent"]["first_token"], serde_json::json!(40));
    assert_eq!(
        rows[1]["completed_latent"]["logical_compressed_row"],
        serde_json::json!(20)
    );
    assert_eq!(rows[1]["completed_latent"]["request_id"], serde_json::json!(41));
    // Descriptors compare actual device bytes against the staged host bytes.
    assert_eq!(
        manifest["descriptors"]["device_matches_host"],
        serde_json::json!(true)
    );
    assert_eq!(manifest["descriptors"]["rows"], serde_json::json!(2));
    // Whole pending planes with UNSCORED padding slots.
    assert_eq!(manifest["pending"]["shape"], serde_json::json!([4, 512]));
    let slots = manifest["pending"]["slots"].as_array().unwrap();
    assert_eq!(slots.len(), 4);
    assert_eq!(slots[1]["state"], serde_json::json!("active"));
    assert_eq!(slots[1]["request_id"], serde_json::json!(41));
    assert_eq!(slots[0]["state"], serde_json::json!("unscored"));
    assert_eq!(slots[2]["state"], serde_json::json!("unscored"));
    assert_eq!(
        std::fs::metadata(directory.join("layer2-compressor-pending-kv.bin"))?.len(),
        (4 * 2048) as u64
    );
    // Weight tensors land in the shared weights directory with full names.
    let tensors = manifest["weights"]["tensors"].as_array().unwrap();
    assert_eq!(tensors.len(), 3);
    assert_eq!(
        tensors[0]["tensor"],
        serde_json::json!("layers.2.attn.compressor.wkv.weight")
    );
    assert_eq!(tensors[0]["shape"], serde_json::json!([512, 5120]));
    assert_eq!(tensors[0]["bytes"], serde_json::json!(512 * 5120 * 2));
    assert_eq!(
        std::fs::metadata(weights_directory.join("layer2-compressor-wkv-weight.bin"))?.len(),
        (512 * 5120 * 2) as u64
    );
    assert_eq!(
        std::fs::metadata(weights_directory.join("layer2-compressor-norm-weight.bin"))?.len(),
        (512 * 2) as u64
    );
    // Input rows are the wave's D2D copy: address-derived bytes pin extents.
    assert_eq!(
        std::fs::read(directory.join("layer2-compressor-input.bin"))?,
        fixture.device.expected(fixture.input, 0, 2 * 10240)
    );
    std::fs::remove_dir_all(&directory)?;

    // Two requests: pending-slot and earlier-wave predecessors both appear.
    let specs = [
        ChunkSpec {
            index: 0,
            request_id: 41,
            slot: 1,
            generation: 3,
            version: 7,
            position: 41,
            tokens: 2,
        },
        ChunkSpec {
            index: 2,
            request_id: 97,
            slot: 3,
            generation: 1,
            version: 5,
            position: 42,
            tokens: 3,
        },
    ];
    let mut fixture = build_fixture(&specs, 4, 2);
    let completed = latents(&specs, 2);
    let directory = scratch_directory("paired");
    std::fs::create_dir_all(&directory)?;
    write_compressor_inputs(
        &mut |src, dst| fixture.device.copy(src, dst),
        &directory,
        2,
        2,
        fixture.slot_count,
        100,
        &fixture.chunks,
        &completed,
        Some(&fixture.descriptors_host),
        &fixture.buffers,
        std::path::Path::new(WEIGHTS_DIRECTORY),
        None,
    )?;
    let manifest = manifest_of(&directory, 2);
    let rows = manifest["wave_rows"].as_array().unwrap();
    assert_eq!(rows.len(), 5);
    // Row 0 (request 41, odd start): pending slot 1.
    assert_eq!(rows[0]["request_id"], serde_json::json!(41));
    assert_eq!(rows[0]["absolute_position"], serde_json::json!(41));
    assert_eq!(
        rows[0]["predecessor"],
        serde_json::json!({"kind": "pending_slot", "slot": 1})
    );
    assert_eq!(
        rows[0]["completed_latent"]["logical_compressed_row"],
        serde_json::json!(20)
    );
    // Row 1 (position 42, even): sentinel, no latent.
    assert_eq!(
        rows[1]["predecessor"],
        serde_json::json!({"kind": "invalid_sentinel"})
    );
    assert_eq!(rows[1]["completed_latent"], serde_json::json!(null));
    // Rows 2-4 belong to request 97; row 3 (odd, j=1) pools earlier wave row 2.
    assert_eq!(rows[2]["request_id"], serde_json::json!(97));
    assert_eq!(rows[2]["chunk"], serde_json::json!(2));
    assert_eq!(rows[3]["absolute_position"], serde_json::json!(43));
    assert_eq!(
        rows[3]["predecessor"],
        serde_json::json!({"kind": "earlier_wave_row", "wave_row": 2})
    );
    assert_eq!(rows[3]["completed_latent"]["first_token"], serde_json::json!(42));
    assert_eq!(
        rows[3]["completed_latent"]["logical_compressed_row"],
        serde_json::json!(21)
    );
    assert_eq!(rows[4]["completed_latent"], serde_json::json!(null));
    // Both members map through their chunks with prepared offsets.
    let chunks = manifest["chunks"].as_array().unwrap();
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0]["flattened_row_range"], serde_json::json!([0, 2]));
    assert_eq!(chunks[1]["prepared_offset"], serde_json::json!(2));
    assert_eq!(chunks[1]["flattened_row_range"], serde_json::json!([2, 5]));
    assert_eq!(chunks[1]["absolute_positions"], serde_json::json!([42, 43, 44]));
    // Both member slots active, others UNSCORED padding.
    let slots = manifest["pending"]["slots"].as_array().unwrap();
    assert_eq!(slots[1]["request_id"], serde_json::json!(41));
    assert_eq!(slots[3]["request_id"], serde_json::json!(97));
    assert_eq!(slots[0]["state"], serde_json::json!("unscored"));
    assert_eq!(slots[2]["state"], serde_json::json!("unscored"));
    // Weights were marked already written for this root: no new weight reads
    // and no weight files appear.
    assert_eq!(
        manifest["weights"]["already_written_for_root"],
        serde_json::json!(true)
    );
    assert!(!directory.join("layer2-compressor-wkv-weight.bin").exists());
    assert!(!directory.join(WEIGHTS_DIRECTORY).exists());
    std::fs::remove_dir_all(&directory)?;
    Ok(())
}

#[test]
fn write_inputs_records_device_host_descriptor_mismatch_from_device() -> Result<()> {
    let specs = [ChunkSpec {
        index: 0,
        request_id: 41,
        slot: 1,
        generation: 3,
        version: 7,
        position: 40,
        tokens: 2,
    }];
    let mut fixture = build_fixture(&specs, 4, 2);
    let completed = latents(&specs, 2);
    // The staged host upload says row 1 pools earlier wave row 0, but the
    // actual device bytes say pending slot 3: the device value wins and the
    // mismatch is recorded, never silently reinterpreted from the host.
    let mut device_descriptors = fixture.descriptors_host.clone();
    device_descriptors[1] = 3;
    fixture.device.contents.insert(
        fixture.buffers.descriptors.unwrap().ptr as usize,
        device_descriptors
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect(),
    );
    let directory = scratch_directory("mismatch");
    std::fs::create_dir_all(&directory)?;
    write_compressor_inputs(
        &mut |src, dst| fixture.device.copy(src, dst),
        &directory,
        2,
        2,
        fixture.slot_count,
        101,
        &fixture.chunks,
        &completed,
        Some(&fixture.descriptors_host),
        &fixture.buffers,
        std::path::Path::new(WEIGHTS_DIRECTORY),
        None,
    )?;
    let manifest = manifest_of(&directory, 2);
    assert_eq!(
        manifest["descriptors"]["device_matches_host"],
        serde_json::json!(false)
    );
    let rows = manifest["wave_rows"].as_array().unwrap();
    assert_eq!(rows[1]["device_descriptor"], serde_json::json!(3));
    assert_eq!(
        rows[1]["predecessor"],
        serde_json::json!({"kind": "pending_slot", "slot": 3})
    );
    // The device file holds the actual bytes; the host file the staged ones.
    let device: Vec<u64> =
        std::fs::read(directory.join("layer2-compressor-descriptors-device.bin"))?
            .chunks_exact(8)
            .map(|chunk| u64::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
    assert_eq!(device, device_descriptors);
    let host: Vec<u64> = std::fs::read(directory.join("layer2-compressor-descriptors-host.bin"))?
        .chunks_exact(8)
        .map(|chunk| u64::from_ne_bytes(chunk.try_into().unwrap()))
        .collect();
    assert_eq!(host, fixture.descriptors_host);
    std::fs::remove_dir_all(&directory)?;
    Ok(())
}

/// Run one capture through a counting fake copy callback so every guard can
/// prove it rejected before any copy was issued.
fn try_write(
    ratio: usize,
    slot_count: usize,
    chunks: &[CompressorTraceChunk],
    descriptors: Option<&[u64]>,
    latents: &[CompressorLatentRow],
    buffers: &CompressorTraceBuffers,
) -> (anyhow::Result<usize>, usize) {
    let mut copies = 0;
    let result = write_compressor_inputs(
        &mut |_, _| {
            copies += 1;
            Ok(())
        },
        std::path::Path::new("unused"),
        2,
        ratio,
        slot_count,
        1,
        chunks,
        latents,
        descriptors,
        buffers,
        std::path::Path::new(WEIGHTS_DIRECTORY),
        None,
    );
    (result, copies)
}

#[test]
fn write_inputs_guards_invalid_identity_before_any_copy() {
    let specs = [ChunkSpec {
        index: 0,
        request_id: 41,
        slot: 1,
        generation: 3,
        version: 7,
        position: 40,
        tokens: 2,
    }];
    let fixture = build_fixture(&specs, 4, 2);
    let completed = latents(&specs, 2);
    // Host descriptor identity must cover exactly the live rows.
    let (result, copies) = try_write(
        2,
        4,
        &fixture.chunks,
        Some(&fixture.descriptors_host[..1]),
        &completed,
        &fixture.buffers,
    );
    let error = result.unwrap_err().to_string();
    assert!(error.contains("descriptor"), "{error}");
    assert_eq!(copies, 0);
    // Ratio one has no descriptors at all.
    let (result, copies) = try_write(
        1,
        4,
        &fixture.chunks,
        Some(&fixture.descriptors_host),
        &completed,
        &fixture.buffers,
    );
    let error = result.unwrap_err().to_string();
    assert!(error.contains("ratio"), "{error}");
    assert_eq!(copies, 0);
    // A completed latent past the wave is rejected.
    let mut beyond = completed.clone();
    beyond.push(CompressorLatentRow {
        lease: CompressorLease {
            owner: 1,
            slot: 1,
            generation: 3,
        },
        source_row: 9,
        position: 60,
    });
    let (result, copies) = try_write(
        2,
        4,
        &fixture.chunks,
        Some(&fixture.descriptors_host),
        &beyond,
        &fixture.buffers,
    );
    let error = result.unwrap_err().to_string();
    assert!(error.contains("exceeds the wave"), "{error}");
    assert_eq!(copies, 0);
    // Duplicate slots are identity failures.
    let mut duplicate = fixture.chunks.clone();
    duplicate.push(CompressorTraceChunk {
        index: 2,
        request_id: 97,
        slot: 1,
        generation: 1,
        version: 1,
        position: 42,
        tokens: 1,
        offset: 2,
    });
    let mut descriptors = fixture.descriptors_host.clone();
    descriptors.push(DESCRIPTOR_SENTINEL);
    let (result, copies) = try_write(2, 4, &duplicate, Some(&descriptors), &completed, &fixture.buffers);
    let error = result.unwrap_err().to_string();
    assert!(error.contains("duplicate"), "{error}");
    assert_eq!(copies, 0);
    // Empty chunks and non-contiguous offsets are rejected.
    let mut empty = fixture.chunks.clone();
    empty[0].tokens = 0;
    let (result, copies) = try_write(2, 4, &empty, Some(&[]), &completed, &fixture.buffers);
    let error = result.unwrap_err().to_string();
    assert!(error.contains("empty"), "{error}");
    assert_eq!(copies, 0);
    let mut gapped = fixture.chunks.clone();
    gapped[0].offset = 3;
    let (result, copies) = try_write(
        2,
        4,
        &gapped,
        Some(&fixture.descriptors_host),
        &completed,
        &fixture.buffers,
    );
    let error = result.unwrap_err().to_string();
    assert!(error.contains("contiguous"), "{error}");
    assert_eq!(copies, 0);
    // No chunks at all is invalid.
    let (result, copies) = try_write(2, 4, &[], None, &completed, &fixture.buffers);
    let error = result.unwrap_err().to_string();
    assert!(error.contains("no chunks"), "{error}");
    assert_eq!(copies, 0);
    // A ratio-two plan over ratio-one buffers fails on scratch identity,
    // again before any copy.
    let ratio1 = build_fixture(
        &[ChunkSpec {
            index: 0,
            request_id: 41,
            slot: 0,
            generation: 1,
            version: 1,
            position: 40,
            tokens: 2,
        }],
        4,
        1,
    );
    let mut descriptors_host = fixture.descriptors_host.clone();
    descriptors_host.truncate(2);
    let (result, copies) = try_write(
        2,
        4,
        &ratio1.chunks,
        Some(&descriptors_host),
        &completed,
        &ratio1.buffers,
    );
    let error = result.unwrap_err().to_string();
    assert!(error.contains("absent at ratio two"), "{error}");
    assert_eq!(copies, 0);
}

#[test]
fn budget_is_rejected_before_any_copy() {
    // 4300 rows of ratio-two operands plus whole pending planes exceed the
    // 64 MiB budget: rejected with zero device reads.
    let specs = [ChunkSpec {
        index: 0,
        request_id: 41,
        slot: 1,
        generation: 3,
        version: 7,
        position: 40,
        tokens: 4300,
    }];
    let slot_count = 16;
    let descriptors_host = staged_descriptors(&specs, slot_count);
    let chunks = trace_chunks(&specs);
    let completed = latents(&specs, 2);
    let mut device = FakeDevice::default();
    let rows = 4300usize;
    let buffers = CompressorTraceBuffers {
        input: device.register(rows * 10240),
        projected: device.register(rows * 2048),
        scores: Some(device.register(rows * 2048)),
        output: device.register(rows * 1024),
        frequencies: device.register(rows * 256),
        positions: device.register(rows * 8),
        descriptors: Some(device.register(rows * 8)),
        kv_values: device.register(rows * 256),
        kv_scales: device.register(rows * 32),
        pending_kv: Some(device.register(slot_count * 2048)),
        pending_scores: Some(device.register(slot_count * 2048)),
    };
    let error = write_compressor_inputs(
        &mut |src, dst| device.copy(src, dst),
        std::path::Path::new("unused"),
        2,
        2,
        slot_count,
        1,
        &chunks,
        &completed,
        Some(&descriptors_host),
        &buffers,
        std::path::Path::new(WEIGHTS_DIRECTORY),
        None,
    )
    .unwrap_err();
    assert!(error.to_string().contains("budget"), "{error}");
    assert!(
        device.reads.is_empty(),
        "no read may precede the budget check"
    );
}

#[test]
fn create_new_refuses_to_overwrite_any_capture_file() -> Result<()> {
    let specs = [ChunkSpec {
        index: 0,
        request_id: 41,
        slot: 1,
        generation: 3,
        version: 7,
        position: 40,
        tokens: 2,
    }];
    let mut fixture = build_fixture(&specs, 4, 2);
    let completed = latents(&specs, 2);
    let directory = scratch_directory("new");
    std::fs::create_dir_all(&directory)?;
    let mut write = || {
        write_compressor_inputs(
            &mut |src, dst| fixture.device.copy(src, dst),
            &directory,
            2,
            2,
            fixture.slot_count,
            1,
            &fixture.chunks,
            &completed,
            Some(&fixture.descriptors_host),
            &fixture.buffers,
            std::path::Path::new(WEIGHTS_DIRECTORY),
            None,
        )
    };
    let first = write();
    assert!(first.is_ok(), "{first:?}");
    // A second capture into the same directory fails loudly: one invocation
    // can never silently replace a prior trace.
    let error = write().unwrap_err();
    assert!(
        error.to_string().contains("layer2-compressor"),
        "{error}"
    );
    std::fs::remove_dir_all(&directory)?;
    // A pre-existing target file also refuses replacement.
    std::fs::create_dir_all(&directory)?;
    std::fs::write(
        directory.join("layer2-compressor-kv-scales.bin"),
        [0u8; 64],
    )?;
    let error = write().unwrap_err();
    assert!(error.to_string().contains("kv-scales"), "{error}");
    std::fs::remove_dir_all(&directory)?;
    Ok(())
}

#[test]
fn write_inputs_covers_ratio_one_without_pooling_scratch() -> Result<()> {
    let specs = [ChunkSpec {
        index: 0,
        request_id: 41,
        slot: 0,
        generation: 2,
        version: 3,
        position: 40,
        tokens: 2,
    }];
    let mut fixture = build_fixture(&specs, 4, 1);
    let completed = latents(&specs, 1);
    let directory = scratch_directory("ratio1");
    std::fs::create_dir_all(&directory)?;
    write_compressor_inputs(
        &mut |src, dst| fixture.device.copy(src, dst),
        &directory,
        20,
        1,
        fixture.slot_count,
        102,
        &fixture.chunks,
        &completed,
        None,
        &fixture.buffers,
        &directory.join(WEIGHTS_DIRECTORY),
        Some(&fixture.weights),
    )?;
    let manifest = manifest_of(&directory, 20);
    assert_eq!(manifest["ratio"], serde_json::json!(1));
    assert_eq!(manifest["descriptors"], serde_json::json!(null));
    assert_eq!(manifest["pending"], serde_json::json!(null));
    assert_eq!(
        manifest["buffers"]["projected"]["bytes"],
        serde_json::json!(2 * 1024)
    );
    // Every ratio-one row completes its own latent: logical row == position.
    let rows = manifest["wave_rows"].as_array().unwrap();
    assert_eq!(
        rows[0]["completed_latent"]["logical_compressed_row"],
        serde_json::json!(40)
    );
    assert_eq!(
        rows[1]["completed_latent"]["logical_compressed_row"],
        serde_json::json!(41)
    );
    assert_eq!(
        rows[0]["predecessor"],
        serde_json::json!({"kind": "invalid_sentinel"})
    );
    // Ratio one has no wgate among the checkpoint weights.
    let tensors = manifest["weights"]["tensors"].as_array().unwrap();
    assert_eq!(tensors.len(), 2);
    assert!(tensors
        .iter()
        .all(|tensor| !tensor["tensor"].as_str().unwrap().contains("wgate")));
    std::fs::remove_dir_all(&directory)?;
    Ok(())
}
