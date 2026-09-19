//! CPU tests for the sparse-attention operand capture: pure planning,
//! kernel-exact referenced-row resolution, and file writing through a fake
//! device-copy callback. No device, bank, or environment is involved.
use super::input_trace::*;
use super::RequestLaunch;
use anyhow::Result;
use ds41rt_ffi::{Ds41rtDeviceBuffer, V41SparseSource, V41SparseWindow};

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
    fn expected_u64(&self, base: usize, offset: usize) -> u64 {
        u64::from_ne_bytes(self.expected(base, offset, 8)[..8].try_into().unwrap())
    }
    fn reads_from(&self, base: usize) -> Vec<(usize, usize)> {
        self.reads
            .iter()
            .filter(|(read_base, _, _)| *read_base == base)
            .map(|&(_, offset, bytes)| (offset, bytes))
            .collect()
    }
}

struct RequestSpec {
    rows: usize,
    request_id: u64,
    positions: &'static [u64],
    /// Some((m[5] causal source length, m[6] private start)) when a paged
    /// source is present, exactly as the compressor proposal metadata emits.
    source_bounds: Option<(u64, u64)>,
    pages: Vec<u32>,
    source_capacity_rows: usize,
    /// Upload a bounded replay slice for this request.
    bounded: bool,
}

struct Fixture {
    device: FakeDevice,
    launches: Vec<RequestLaunch>,
    provenance: Vec<RequestProvenance>,
    buffers: WaveOperandBuffers,
    metadata: Vec<u64>,
    rows: usize,
    window_ends: Vec<usize>,
    source_ends: Vec<Option<usize>>,
    source_values: Vec<Option<usize>>,
    source_scales: Vec<Option<usize>>,
    pages: Vec<usize>,
    query: usize,
    selected: usize,
    replay_begins: usize,
    descriptors: usize,
}

fn build_fixture(specs: &[RequestSpec], with_selection: bool) -> Fixture {
    let mut device = FakeDevice::default();
    let rows: usize = specs.iter().map(|spec| spec.rows).sum();
    let query = device.register(rows * 65536);
    let sink = device.register(256);
    let metadata_buffer = device.register(rows * 80);
    let selected = with_selection.then(|| device.register(rows * 2048));
    let replay_begins = device.register(rows * 8);
    let descriptors = device.register(rows * 120);
    let mut launches = Vec::with_capacity(specs.len());
    let mut provenance = Vec::with_capacity(specs.len());
    let mut metadata = Vec::with_capacity(rows * 10);
    let (mut window_ends, mut source_ends, mut source_values, mut source_scales, mut pages) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut row_offset = 0usize;
    for spec in specs {
        let window_values = device.register(128 * 512);
        let window_scales = device.register(128 * 16);
        let window_end = device.register(8);
        let proposal_capacity = 32usize;
        let proposal_values = device.register(proposal_capacity * 512);
        let proposal_scales = device.register(proposal_capacity * 16);
        let bounds = spec.bounded.then(|| Ds41rtDeviceBuffer {
            ptr: (replay_begins.ptr as usize + row_offset * 8) as *mut _,
            bytes: spec.rows * 8,
            device_id: 0,
            flags: 0,
        });
        let source = spec.source_bounds.map(|(length, start)| {
            let capacity = spec.source_capacity_rows;
            let values = device.register(capacity * 256);
            let scales = device.register(capacity * 32);
            let proposal_capacity = 16usize;
            let proposals = device.register(proposal_capacity * 256);
            let proposal_scales = device.register(proposal_capacity * 32);
            let table = device.register(spec.pages.len() * 4);
            let end = device.register(8);
            V41SparseSource {
                values,
                scales,
                proposals,
                proposal_scales,
                pages: table,
                end,
                capacity,
                proposal_capacity,
                page_stride: spec.pages.len(),
            }
        });
        for &position in spec.positions {
            metadata.extend([row_offset as u64, 0, 128, position]);
            match spec.source_bounds {
                Some((length, start)) => metadata.extend([0, length, start, 1000, 0, 1]),
                None => metadata.extend([0; 6]),
            }
        }
        let source_parts = source.as_ref().map(|source| {
            (
                source.values.ptr as usize,
                source.scales.ptr as usize,
                source.end.ptr as usize,
                source.pages.ptr as usize,
            )
        });
        launches.push(RequestLaunch {
            window: V41SparseWindow {
                values: window_values,
                scales: window_scales,
                proposals: proposal_values,
                proposal_scales,
                end: window_end,
                proposal_capacity,
                replay_begins: bounds,
            },
            source,
            rows: spec.rows,
            width: 0,
        });
        provenance.push(RequestProvenance {
            request_id: spec.request_id,
            positions: spec.positions.to_vec(),
        });
        window_ends.push(window_end.ptr as usize);
        let (values, scales, end, table) = source_parts.unwrap_or((0, 0, 0, 0));
        source_values.push((values != 0).then_some(values));
        source_scales.push((scales != 0).then_some(scales));
        source_ends.push((end != 0).then_some(end));
        pages.push(table);
        row_offset += spec.rows;
    }
    device.contents.insert(
        metadata_buffer.ptr as usize,
        metadata
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect(),
    );
    Fixture {
        device,
        launches,
        provenance,
        buffers: WaveOperandBuffers {
            query,
            sink,
            metadata: metadata_buffer,
            selected,
            replay_begins,
            descriptors,
        },
        metadata,
        rows,
        window_ends,
        source_ends,
        source_values,
        source_scales,
        pages,
        query: query.ptr as usize,
        selected: selected.map(|buffer| buffer.ptr as usize).unwrap_or(0),
        replay_begins: replay_begins.ptr as usize,
        descriptors: descriptors.ptr as usize,
    }
}

fn selection(rows: usize, entries: &[(usize, usize, i32)]) -> Vec<u8> {
    // entries: (row, key, id); every other slot stays masked padding (-1).
    let mut selected = vec![-1i32; rows * 512];
    for &(row, key, id) in entries {
        selected[row * 512 + key] = id;
    }
    selected.iter().flat_map(|id| id.to_ne_bytes()).collect()
}

fn scratch_directory(label: &str) -> std::path::PathBuf {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "ds41rt-sparse-input-trace-{label}-{}-{unique}",
        std::process::id()
    ))
}

fn manifest_of(directory: &std::path::Path) -> serde_json::Value {
    let bytes = std::fs::read(directory.join("layer2-attention-inputs.json")).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[test]
fn trigger_arms_once_and_errors_cannot_leak_it() {
    let mut trigger = InputTraceTrigger::default();
    assert!(!trigger.armed());
    assert_eq!(trigger.take(), None);
    let directory = std::path::Path::new("/trace/lane0-batch7-Full-4rows");
    trigger.arm(directory);
    assert!(trigger.armed());
    // The staged execution consumes the trigger before doing any work, so a
    // failure inside that execution can never arm a later one.
    assert_eq!(trigger.take().as_deref(), Some(directory));
    assert!(!trigger.armed());
    assert_eq!(trigger.take(), None);
    // Clearing an armed trigger resets it; re-arming works afterwards.
    trigger.arm(directory);
    trigger.disarm();
    assert!(!trigger.armed());
    trigger.arm(directory);
    assert!(trigger.armed());
}

#[test]
fn planner_records_four_row_identity_without_source_or_selection() {
    let fixture = build_fixture(
        &[RequestSpec {
            rows: 4,
            request_id: 41,
            positions: &[40, 41, 42, 43],
            source_bounds: None,
            pages: vec![],
            source_capacity_rows: 1,
            bounded: false,
        }],
        false,
    );
    let reads =
        plan_operand_reads(2, fixture.rows, &fixture.launches, &fixture.buffers, false).unwrap();
    let find = |name: &str| reads.iter().find(|read| read.name == name);
    // Wave-wide operands: query, sink, metadata; no selection or descriptors.
    let query = find("query").unwrap();
    assert_eq!(query.bytes, 4 * 65536);
    assert_eq!(query.shape, vec![4, 64, 512]);
    assert_eq!(query.dtype, "bfloat16");
    assert_eq!(query.file, "layer2-attention-query.bin");
    let sink = find("sink").unwrap();
    assert_eq!(sink.bytes, 256);
    assert_eq!(sink.dtype, "float32");
    assert_eq!(find("metadata").unwrap().bytes, 4 * 80);
    assert!(find("selected").is_none());
    assert!(find("descriptors-device").is_none());
    // Source-free layer: no page table, no source proposals.
    assert!(reads.iter().all(|read| !read.name.starts_with("source-")));
    // Unuploaded bounds become an explicit-zero record, never a device read.
    let replay = find("replay-begins").unwrap();
    assert_eq!(replay.bytes, 4 * 8);
    assert_eq!(replay.synthetic, Some(SyntheticContent::ExplicitZero));
    // Fixed ring extents and the live proposal capacity are both recorded.
    let ring = find("ring-values").unwrap();
    assert_eq!(ring.bytes, 128 * 512);
    assert_eq!(ring.shape, vec![128, 512]);
    assert_eq!(find("window-proposal-values").unwrap().bytes, 32 * 512);
    // Every file name is unique so create_new can never collide.
    let names: std::collections::HashSet<_> = reads.iter().map(|read| read.file.clone()).collect();
    assert_eq!(names.len(), reads.len());
}

#[test]
fn planner_rejects_extents_past_allocations_and_the_byte_budget() {
    let mut fixture = build_fixture(
        &[RequestSpec {
            rows: 1,
            request_id: 7,
            positions: &[10],
            source_bounds: None,
            pages: vec![],
            source_capacity_rows: 1,
            bounded: false,
        }],
        false,
    );
    // One byte short of the live query extent.
    fixture.buffers.query.bytes -= 1;
    let error = plan_operand_reads(2, 1, &fixture.launches, &fixture.buffers, false).unwrap_err();
    assert!(error.to_string().contains("query"), "{error}");
    // Row-count arithmetic overflow fails before any allocation check.
    fixture.buffers.query.bytes = usize::MAX;
    let error =
        plan_operand_reads(2, usize::MAX, &fixture.launches, &fixture.buffers, false).unwrap_err();
    assert!(error.to_string().contains("overflow"), "{error}");
    // A capture over the fixed budget fails explicitly instead of dumping.
    let mut fixture = build_fixture(
        &[RequestSpec {
            rows: 1,
            request_id: 7,
            positions: &[10],
            source_bounds: Some((8, 0)),
            pages: vec![1],
            source_capacity_rows: 1,
            bounded: false,
        }],
        true,
    );
    let huge = TRACE_BYTE_BUDGET / 512 + 1;
    fixture.launches[0].window.proposal_capacity = huge;
    fixture.launches[0].window.proposals.bytes = huge * 512;
    fixture.launches[0].window.proposal_scales.bytes = huge * 16;
    let error = plan_operand_reads(2, 1, &fixture.launches, &fixture.buffers, false).unwrap_err();
    assert!(error.to_string().contains("budget"), "{error}");
}

#[test]
fn resolver_masks_duplicates_private_and_permuted_pages_exactly_like_the_kernel() {
    // One row; m[5]=1000 causal length, m[6]=512 private start, one permuted
    // page table entry mapping logical page 0 to physical page 3.
    let mut selected = vec![-1i32; 512];
    for (key, id) in [
        (0, 5i32),
        (1, 300),
        (2, -1),
        (3, 999),
        (4, 512),
        (5, 600),
        (6, 5),
        (7, 4095),
        (8, 255),
        (9, 44),
    ] {
        selected[key] = id;
    }
    let metadata: Vec<u64> = vec![0, 0, 128, 40, 0, 1000, 512, 4, 0, 1];
    let resolved =
        resolve_referenced_source_rows(1, 0, &selected, &metadata, &[3], 1, 1000).unwrap();
    // id 5 -> 3*256+5 = 773 (selected twice, deduplicated); id 44 -> 812.
    assert_eq!(resolved.physical, vec![773, 812]);
    assert_eq!(
        resolved.logical_to_physical.get(&5).copied(),
        Some(773),
        "permuted pages must map through pages[id/256]"
    );
    assert_eq!(resolved.logical_to_physical.get(&44).copied(), Some(812));
    assert_eq!(resolved.logical_to_physical.len(), 2);
    // Every dropped slot is accounted for by the mask that drops it.
    assert_eq!(resolved.masked, 503); // 512 - 10 set keys
    assert_eq!(resolved.private, 3); // 999, 512, 600 at/past m[6], under m[5]
    assert_eq!(resolved.past_page_stride, 1); // 300 >= 1*256 and < m[6]
    assert_eq!(resolved.past_causal_length, 1); // 4095 >= m[5]
    assert_eq!(resolved.past_capacity, 1); // 255 -> physical 1023 >= 1000
}

#[test]
fn resolver_rejects_inconsistent_selection_metadata_and_page_extents() {
    let metadata: Vec<u64> = vec![0, 0, 128, 40, 0, 1000, 512, 4, 0, 1];
    let selected = vec![-1i32; 512];
    // Selection shorter than the request's flattened rows.
    let error =
        resolve_referenced_source_rows(2, 0, &selected, &metadata, &[3], 1, 1000).unwrap_err();
    assert!(error.to_string().contains("selection"), "{error}");
    // Metadata shorter than the rows.
    let error = resolve_referenced_source_rows(2, 0, &vec![-1i32; 1024], &metadata, &[3], 1, 1000)
        .unwrap_err();
    assert!(error.to_string().contains("metadata"), "{error}");
    // Page table length must equal the stride.
    let error =
        resolve_referenced_source_rows(1, 0, &selected, &metadata, &[3, 4], 1, 1000).unwrap_err();
    assert!(error.to_string().contains("stride"), "{error}");
}

#[test]
fn write_inputs_captures_referenced_rows_only_and_reports_identity() -> Result<()> {
    let mut fixture = build_fixture(
        &[
            RequestSpec {
                rows: 2,
                request_id: 41,
                positions: &[40, 41],
                source_bounds: Some((1000, 512)),
                pages: vec![3],
                source_capacity_rows: 1000,
                bounded: true,
            },
            RequestSpec {
                rows: 2,
                request_id: 97,
                positions: &[900, 901],
                // m[5]=m[6]=100: every id under 100 is a committed paged row
                // and the private overlay range is empty.
                source_bounds: Some((100, 100)),
                pages: vec![9],
                source_capacity_rows: 3000,
                bounded: false,
            },
        ],
        true,
    );
    // Flattened selection across all four rows.
    let selected_bytes = selection(
        fixture.rows,
        &[
            // request 0, row 0
            (0, 0, 5),
            (0, 1, 300),
            (0, 2, -1),
            (0, 3, 999),
            (0, 4, 512),
            (0, 5, 600),
            (0, 6, 5),
            (0, 7, 4095),
            (0, 8, 255),
            (0, 9, 44),
            // request 0, row 1 repeats a physical row
            (1, 0, 44),
            // request 1, row 2
            (2, 0, 7),
            (2, 1, 7),
            (2, 2, 150),
        ],
    );
    let directory = scratch_directory("main");
    std::fs::create_dir_all(&directory)?;
    let selected_ptr = fixture.selected;
    let pages0_ptr = fixture.pages[0];
    let pages1_ptr = fixture.pages[1];
    let pages0: Vec<u8> = 3u32.to_ne_bytes().to_vec();
    let pages1: Vec<u8> = 9u32.to_ne_bytes().to_vec();
    let captured_bytes = write_attention_inputs(
        &mut |src, dst| {
            if src.ptr as usize == selected_ptr {
                dst.copy_from_slice(&selected_bytes);
                return Ok(());
            }
            if src.ptr as usize == pages0_ptr {
                dst.copy_from_slice(&pages0);
                return Ok(());
            }
            if src.ptr as usize == pages1_ptr {
                dst.copy_from_slice(&pages1);
                return Ok(());
            }
            fixture.device.copy(src, dst)
        },
        &directory,
        2,
        fixture.rows,
        &fixture.provenance,
        &fixture.launches,
        &fixture.metadata,
        &fixture.buffers,
        None,
    )?;

    // --- no full pool copy: exactly the referenced physical rows are read ---
    let pool = fixture.source_values[0].unwrap();
    assert_eq!(
        fixture.device.reads_from(pool),
        vec![(773 * 256, 256), (812 * 256, 256)],
        "only the two referenced physical rows, 256 bytes each"
    );
    assert_eq!(
        fixture.device.reads_from(fixture.source_values[1].unwrap()),
        vec![(2311 * 256, 256)]
    );

    // --- referenced-row files hold the exact physical row bytes ---
    let expected_values: Vec<u8> = [773usize, 812]
        .iter()
        .flat_map(|&physical| fixture.device.expected(pool, physical * 256, 256))
        .collect();
    assert_eq!(
        std::fs::read(directory.join("layer2-attention-request0-source-referenced-values.bin"))?,
        expected_values
    );
    let scales_pool = fixture.source_scales[0].unwrap();
    let expected_scales: Vec<u8> = [773usize, 812]
        .iter()
        .flat_map(|&physical| fixture.device.expected(scales_pool, physical * 32, 32))
        .collect();
    assert_eq!(
        std::fs::read(directory.join("layer2-attention-request0-source-referenced-scales.bin"))?,
        expected_scales
    );

    // --- manifest identity, ranges, scalars and skip accounting ---
    let manifest = manifest_of(&directory);
    assert_eq!(manifest["schema"], serde_json::json!(1));
    assert_eq!(
        manifest["kind"],
        serde_json::json!("sparse-attention-inputs")
    );
    assert_eq!(manifest["layer"], serde_json::json!(2));
    assert_eq!(manifest["rows"], serde_json::json!(4));
    assert_eq!(
        manifest["launch_kind"],
        serde_json::json!("single_per_request")
    );
    assert_eq!(manifest["parts"], serde_json::json!(null));
    assert_eq!(
        manifest["captured_bytes"].as_u64(),
        Some(captured_bytes as u64)
    );
    let requests = manifest["requests"].as_array().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0]["request_id"], serde_json::json!(41));
    assert_eq!(
        requests[0]["flattened_row_range"],
        serde_json::json!([0, 2])
    );
    assert_eq!(requests[0]["positions"], serde_json::json!([40, 41]));
    assert_eq!(requests[0]["width"], serde_json::json!(0));
    assert_eq!(requests[0]["split_parts"], serde_json::json!(10));
    assert_eq!(requests[1]["request_id"], serde_json::json!(97));
    assert_eq!(
        requests[1]["flattened_row_range"],
        serde_json::json!([2, 4])
    );
    assert_eq!(requests[1]["split_parts"], serde_json::json!(10));
    // Actual device window_end/source_end scalars are address-derived.
    assert_eq!(
        requests[0]["window"]["window_end"]["value"].as_u64(),
        Some(fixture.device.expected_u64(fixture.window_ends[0], 0))
    );
    assert_eq!(
        requests[0]["source"]["source_end"]["value"].as_u64(),
        Some(
            fixture
                .device
                .expected_u64(fixture.source_ends[0].unwrap(), 0)
        )
    );
    // Request 0 uploaded bounds: the actual live replay values.
    assert_eq!(
        requests[0]["replay_begins"]["uploaded"],
        serde_json::json!(true)
    );
    assert_eq!(
        requests[0]["replay_begins"]["values"],
        serde_json::json!([
            fixture.device.expected_u64(fixture.replay_begins, 0),
            fixture.device.expected_u64(fixture.replay_begins, 8),
        ])
    );
    // Request 1 passed no bounds pointer: explicit zeros, no device read of
    // its uninitialized rows.
    assert_eq!(
        requests[1]["replay_begins"]["uploaded"],
        serde_json::json!(false)
    );
    assert_eq!(
        requests[1]["replay_begins"]["values"],
        serde_json::json!([0, 0])
    );
    assert_eq!(
        std::fs::read(directory.join("layer2-attention-request1-replay-begins.bin"))?,
        vec![0u8; 16]
    );
    assert_eq!(
        fixture.device.reads_from(fixture.replay_begins),
        vec![(0, 16)]
    );
    // Referenced-row accounting matches the kernel-exact masks.
    let referenced = &requests[0]["source"]["referenced"];
    assert_eq!(referenced["count"], serde_json::json!(2));
    assert_eq!(referenced["physical_rows"], serde_json::json!([773, 812]));
    assert_eq!(
        referenced["logical_to_physical"],
        serde_json::json!([{"logical": 5, "physical": 773}, {"logical": 44, "physical": 812}])
    );
    assert_eq!(referenced["skipped"]["private"], serde_json::json!(3));
    assert_eq!(
        referenced["skipped"]["past_causal_length"],
        serde_json::json!(1)
    );
    assert_eq!(
        referenced["skipped"]["past_page_stride"],
        serde_json::json!(1)
    );
    assert_eq!(referenced["skipped"]["past_capacity"], serde_json::json!(1));
    // 1024 slots, 11 set keys of which one is an explicit -1: 1013 padding
    // slots plus that explicit mask.
    assert_eq!(
        referenced["skipped"]["masked"],
        serde_json::json!(512 * 2 - 11 + 1)
    );
    assert_eq!(
        manifest["wave"]["query"]["bytes"],
        serde_json::json!(4 * 65536)
    );
    assert_eq!(
        manifest["wave"]["selected"]["bytes"],
        serde_json::json!(4 * 2048)
    );
    std::fs::remove_dir_all(&directory)?;
    Ok(())
}

#[test]
fn write_inputs_records_batch_descriptors_and_compares_bytes_only() -> Result<()> {
    let mut fixture = build_fixture(
        &[
            RequestSpec {
                rows: 2,
                request_id: 41,
                positions: &[40, 41],
                source_bounds: Some((1000, 512)),
                pages: vec![3],
                source_capacity_rows: 1000,
                bounded: true,
            },
            RequestSpec {
                rows: 2,
                request_id: 97,
                positions: &[900, 901],
                source_bounds: Some((1000, 512)),
                pages: vec![3],
                source_capacity_rows: 1000,
                bounded: false,
            },
        ],
        true,
    );
    let directory = scratch_directory("batch");
    std::fs::create_dir_all(&directory)?;
    // Host staging bytes that match what the device replay of the descriptor
    // buffer returns: the address-derived pattern of that allocation.
    let host: Vec<u8> = fixture
        .device
        .expected(fixture.descriptors, 0, fixture.rows * 120);
    write_attention_inputs(
        &mut |src, dst| fixture.device.copy(src, dst),
        &directory,
        2,
        fixture.rows,
        &fixture.provenance,
        &fixture.launches,
        &fixture.metadata,
        &fixture.buffers,
        Some(&host),
    )?;
    let manifest = manifest_of(&directory);
    assert_eq!(
        manifest["launch_kind"],
        serde_json::json!("descriptor_batch")
    );
    assert_eq!(manifest["parts"], serde_json::json!(10));
    assert_eq!(
        manifest["descriptors"]["bytes"].as_u64(),
        Some((fixture.rows * 120) as u64)
    );
    assert_eq!(
        manifest["descriptors"]["device_matches_host"],
        serde_json::json!(true)
    );
    assert!(manifest["descriptors"]["comparison"]
        .as_str()
        .unwrap()
        .contains("bytes"));
    assert_eq!(
        std::fs::read(directory.join("layer2-attention-descriptors-device.bin"))?,
        host
    );
    assert_eq!(
        std::fs::metadata(directory.join("layer2-attention-descriptors-host.bin"))?.len(),
        (fixture.rows * 120) as u64
    );
    // One wave-wide bounds upload serves every flattened row.
    let requests = manifest["requests"].as_array().unwrap();
    assert_eq!(
        requests[0]["replay_begins"]["uploaded"],
        serde_json::json!(true)
    );
    assert_eq!(
        requests[1]["replay_begins"]["uploaded"],
        serde_json::json!(true)
    );
    assert_eq!(
        requests[1]["replay_begins"]["file"],
        serde_json::json!("layer2-attention-replay-begins.bin")
    );
    assert_eq!(
        std::fs::read(directory.join("layer2-attention-replay-begins.bin"))?.len(),
        fixture.rows * 8
    );
    std::fs::remove_dir_all(&directory)?;
    Ok(())
}

#[test]
fn create_new_refuses_to_overwrite_any_capture_file() -> Result<()> {
    let mut fixture = build_fixture(
        &[RequestSpec {
            rows: 1,
            request_id: 41,
            positions: &[40],
            source_bounds: None,
            pages: vec![],
            source_capacity_rows: 1,
            bounded: false,
        }],
        false,
    );
    let directory = scratch_directory("new");
    std::fs::create_dir_all(&directory)?;
    let mut write = || {
        write_attention_inputs(
            &mut |src, dst| fixture.device.copy(src, dst),
            &directory,
            2,
            fixture.rows,
            &fixture.provenance,
            &fixture.launches,
            &fixture.metadata,
            &fixture.buffers,
            None,
        )
    };
    let first = write();
    assert!(first.is_ok(), "{first:?}");
    // A second capture into the same directory fails loudly.
    let error = write().unwrap_err();
    assert!(error.to_string().contains("layer2-attention"), "{error}");
    std::fs::remove_dir_all(&directory)?;
    // A pre-existing target file also refuses replacement.
    std::fs::create_dir_all(&directory)?;
    std::fs::write(directory.join("layer2-attention-sink.bin"), [0u8; 256])?;
    let error = write().unwrap_err();
    assert!(error.to_string().contains("sink"), "{error}");
    std::fs::remove_dir_all(&directory)?;
    Ok(())
}

#[test]
fn unuploaded_bounds_are_explicit_zeros_and_never_read() -> Result<()> {
    let mut fixture = build_fixture(
        &[RequestSpec {
            rows: 3,
            request_id: 5,
            positions: &[7, 8, 9],
            source_bounds: None,
            pages: vec![],
            source_capacity_rows: 1,
            bounded: false,
        }],
        false,
    );
    let directory = scratch_directory("zeros");
    std::fs::create_dir_all(&directory)?;
    write_attention_inputs(
        &mut |src, dst| fixture.device.copy(src, dst),
        &directory,
        0,
        fixture.rows,
        &fixture.provenance,
        &fixture.launches,
        &fixture.metadata,
        &fixture.buffers,
        None,
    )?;
    // The bounds allocation is never touched: its rows were not uploaded, and
    // the capture synthesizes the implicit zero the kernel would read.
    assert!(fixture.device.reads_from(fixture.replay_begins).is_empty());
    assert_eq!(
        std::fs::read(directory.join("layer0-attention-request0-replay-begins.bin"))?,
        vec![0u8; 24]
    );
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(
        directory.join("layer0-attention-inputs.json"),
    )?)?;
    assert_eq!(
        manifest["requests"][0]["replay_begins"]["synthetic"],
        serde_json::json!("explicit_zero")
    );
    // Source-free (layer < 2) captures are fully supported.
    assert_eq!(manifest["requests"][0]["source"], serde_json::json!(null));
    assert_eq!(manifest["wave"]["selected"], serde_json::json!(null));
    std::fs::remove_dir_all(&directory)?;
    Ok(())
}

#[test]
fn captured_metadata_controls_source_reads_even_when_host_differs() -> Result<()> {
    let mut fixture = build_fixture(
        &[RequestSpec {
            rows: 1,
            request_id: 7,
            positions: &[10],
            source_bounds: Some((8, 8)),
            pages: vec![0],
            source_capacity_rows: 256,
            bounded: false,
        }],
        true,
    );
    fixture
        .device
        .contents
        .insert(fixture.selected, selection(1, &[(0, 0, 5)]));
    fixture
        .device
        .contents
        .insert(fixture.pages[0], 0u32.to_ne_bytes().to_vec());
    // Device metadata permits committed row 5. A stale host view would drop it.
    fixture.metadata[5] = 0;
    let directory = scratch_directory("device-metadata");
    std::fs::create_dir_all(&directory)?;
    write_attention_inputs(
        &mut |src, dst| fixture.device.copy(src, dst),
        &directory,
        2,
        fixture.rows,
        &fixture.provenance,
        &fixture.launches,
        &fixture.metadata,
        &fixture.buffers,
        None,
    )?;
    assert_eq!(
        fixture.device.reads_from(fixture.source_values[0].unwrap()),
        vec![(5 * 256, 256)]
    );
    let manifest = manifest_of(&directory);
    assert_eq!(manifest["metadata_device_matches_host"], false);
    assert_eq!(
        manifest["requests"][0]["source"]["referenced"]["physical_rows"],
        serde_json::json!([5])
    );
    std::fs::remove_dir_all(directory)?;
    Ok(())
}

#[test]
fn invalid_identity_and_resolver_overflow_fail_without_copying() {
    assert!(
        resolve_referenced_source_rows(usize::MAX / 512 + 1, 0, &[], &[], &[], 0, 0)
            .unwrap_err()
            .to_string()
            .contains("overflow")
    );
    let mut fixture = build_fixture(
        &[RequestSpec {
            rows: 1,
            request_id: 7,
            positions: &[10],
            source_bounds: None,
            pages: vec![],
            source_capacity_rows: 1,
            bounded: false,
        }],
        false,
    );
    fixture.provenance[0].positions.clear();
    let mut copies = 0;
    let error = write_attention_inputs(
        &mut |_, _| {
            copies += 1;
            Ok(())
        },
        std::path::Path::new("unused"),
        0,
        1,
        &fixture.provenance,
        &fixture.launches,
        &fixture.metadata,
        &fixture.buffers,
        None,
    )
    .unwrap_err();
    assert!(error.to_string().contains("position count"));
    assert_eq!(copies, 0);
}

#[test]
fn referenced_pool_budget_is_checked_before_any_pool_read() {
    let rows = 500;
    let positions = Box::leak((0..rows as u64).collect::<Vec<_>>().into_boxed_slice());
    let mut fixture = build_fixture(
        &[RequestSpec {
            rows,
            request_id: 7,
            positions,
            source_bounds: Some((256000, 256000)),
            pages: (0..1000).collect(),
            source_capacity_rows: 262144,
            bounded: false,
        }],
        true,
    );
    fixture.device.contents.insert(
        fixture.selected,
        (0..256000i32).flat_map(|id| id.to_ne_bytes()).collect(),
    );
    fixture.device.contents.insert(
        fixture.pages[0],
        (0..1000u32).flat_map(|page| page.to_ne_bytes()).collect(),
    );
    let error = write_attention_inputs(
        &mut |src, dst| fixture.device.copy(src, dst),
        std::path::Path::new("unused"),
        2,
        rows,
        &fixture.provenance,
        &fixture.launches,
        &fixture.metadata,
        &fixture.buffers,
        None,
    )
    .unwrap_err();
    assert!(error.to_string().contains("budget"), "{error}");
    assert!(fixture
        .device
        .reads_from(fixture.source_values[0].unwrap())
        .is_empty());
    assert!(fixture
        .device
        .reads_from(fixture.source_scales[0].unwrap())
        .is_empty());
}
