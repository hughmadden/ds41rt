//! CPU tests for the bounded opt-in expert-route capture, plus an
//! env-gated serving-path test that runs only when the real native
//! library, model and `DS41RT_EXPERT_TRACE_DIR` are provided (the same
//! pattern as `mapped_tests.rs`). The serving path under test is the real
//! `ExpertExecution::execute_request_output` hook; the selection, extent
//! and file-writing helpers below are the exact same code it calls.

use super::trace::{
    ExpertTraceCapture, ExpertTraceExtents, ExpertTraceSelection, ExpertTraceWireIdentity,
    TRACE_ENV_VAR, TRACE_EXECUTOR_ID, TRACE_LAYER, TRACE_ROW_CHOICES, TRACE_SCHEMA,
    selection_from_env_var, trace_dir_name,
};
use anyhow::{Context, Result};

fn enabled_selection() -> ExpertTraceSelection {
    selection_from_env_var(Some(std::ffi::OsString::from("trace-root"))).unwrap()
}

fn capture_from<'a>(
    rows: u32,
    hidden: &'a [u8],
    route_ids: &'a [i32],
    routing: &'a [f32],
    route_partials: &'a [u8],
    compact: &'a [u8],
) -> ExpertTraceCapture<'a> {
    ExpertTraceCapture {
        wire: ExpertTraceWireIdentity {
            request_id: 2026091950,
            placement_version: 17,
        },
        layer: TRACE_LAYER,
        executor_id: TRACE_EXECUTOR_ID,
        rank: 1,
        rows,
        kernel_capacity_rows: 4096,
        input_dtype: 7,
        input_row_bytes: 5280,
        registered_destination: false,
        hidden,
        route_ids,
        routing,
        route_partials,
        compact,
    }
}

fn filled_parts(rows: u32, seed: u8) -> (Vec<u8>, Vec<i32>, Vec<f32>, Vec<u8>, Vec<u8>) {
    let extents = ExpertTraceExtents::for_rows(rows, 5280, 6).unwrap();
    let hidden: Vec<u8> = (0..extents.hidden_bytes).map(|i| seed.wrapping_add(i as u8)).collect();
    let route_ids: Vec<i32> = (0..rows * 6).map(|i| 105 + i as i32).collect();
    let routing: Vec<f32> = (0..rows * 6)
        .map(|i| f32::from_bits(0x3e80_0000u32.wrapping_add(i as u32)))
        .collect();
    let route_partials: Vec<u8> = (0..extents.route_partials_bytes)
        .map(|i| seed.wrapping_mul(3).wrapping_add(i as u8))
        .collect();
    let compact: Vec<u8> = (0..extents.compact_bytes)
        .map(|i| seed.wrapping_mul(7).wrapping_add(i as u8))
        .collect();
    (hidden, route_ids, routing, route_partials, compact)
}

#[test]
fn env_var_parsing_disables_empty_and_enables_dir() {
    assert_eq!(TRACE_ENV_VAR, "DS41RT_EXPERT_TRACE_DIR");
    assert!(selection_from_env_var(None).is_none());
    assert!(selection_from_env_var(Some("".into())).is_none());
    let selection = selection_from_env_var(Some("some-trace-dir".into())).unwrap();
    assert_eq!(selection.dir(), std::path::Path::new("some-trace-dir"));
}

#[test]
fn candidate_requires_exact_layer_executor_and_rows() {
    let selection = enabled_selection();
    assert!(selection.is_candidate(1, 2, 1));
    assert!(selection.is_candidate(1, 2, 2));
    // Neighboring layers, executors and every other row count are excluded.
    for (layer, executor, rows) in [
        (0, 2, 1),
        (2, 2, 1),
        (1, 1, 1),
        (1, 3, 1),
        (1, 0, 1),
        (1, 2, 0),
        (1, 2, 3),
        (1, 2, 16),
        (1, 2, 80),
    ] {
        assert!(!selection.is_candidate(layer, executor, rows), "({layer},{executor},{rows}) must not be a candidate");
    }
}

#[test]
fn claim_elects_exactly_one_winner_per_row_shape() {
    let selection = std::sync::Arc::new(enabled_selection());
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(32));
    let winners = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    std::thread::scope(|scope| {
        for thread in 0..32 {
            let (selection, barrier, winners) = (selection.clone(), barrier.clone(), winners.clone());
            scope.spawn(move || {
                for rows in TRACE_ROW_CHOICES {
                    barrier.wait();
                    if let Some(claim) = selection.try_claim(rows) {
                        winners.lock().unwrap().push((thread, claim.rows(), claim.ordinal()));
                    }
                }
            });
        }
    });
    let winners = winners.lock().unwrap();
    for rows in TRACE_ROW_CHOICES {
        let shape_winners: Vec<_> = winners.iter().filter(|(_, r, _)| *r == rows).collect();
        assert_eq!(shape_winners.len(), 1, "rows={rows}: exactly one winner, got {shape_winners:?}");
    }
    // Ordinals are unique and monotonically assigned by the race.
    let mut ordinals: Vec<u64> = winners.iter().map(|(_, _, ordinal)| *ordinal).collect();
    ordinals.sort_unstable();
    assert_eq!(ordinals, vec![1, 2]);
}

#[test]
fn released_claim_reopens_shape_for_later_success() {
    let selection = enabled_selection();
    let first = selection.try_claim(1).unwrap();
    let ordinal = first.ordinal();
    assert!(selection.try_claim(1).is_none());
    first.release();
    let second = selection.try_claim(1).unwrap();
    assert!(second.ordinal() > ordinal, "ordinals must stay monotonic across releases");
}

#[test]
fn extents_match_wire_contract_and_reject_overflow() {
    let one = ExpertTraceExtents::for_rows(1, 5280, 6).unwrap();
    assert_eq!(
        one,
        ExpertTraceExtents {
            rows: 1,
            input_row_bytes: 5280,
            hidden_bytes: 5280,
            route_ids_bytes: 24,
            routing_bytes: 24,
            route_partials_bytes: 1 * 6 * 5120 * 4,
            compact_bytes: 5120 * 2,
        }
    );
    let two = ExpertTraceExtents::for_rows(2, 5280, 6).unwrap();
    assert_eq!(two.hidden_bytes, 2 * 5280);
    assert_eq!(two.route_ids_bytes, 48);
    assert_eq!(two.routing_bytes, 48);
    assert_eq!(two.route_partials_bytes, 2 * 6 * 5120 * 4);
    assert_eq!(two.compact_bytes, 2 * 5120 * 2);
    // FP32 route planes must not be confused with the BF16 compact sum.
    assert_eq!(one.route_partials_bytes / one.compact_bytes, 12);
    // rows is a u32, so on 64-bit hosts the row-count axis alone cannot
    // overflow; the input_row_bytes axis can and must be rejected.
    assert!(ExpertTraceExtents::for_rows(2, usize::MAX, 6).is_err());
    assert!(ExpertTraceExtents::for_rows(0, 5280, 6).is_err());
}

#[test]
fn capacity_validation_guards_live_rows() {
    let extents = ExpertTraceExtents::for_rows(2, 5280, 6).unwrap();
    extents.validate_capacity(4096, 6, 5280).unwrap();
    assert!(extents.validate_capacity(1, 6, 5280).is_err(), "live rows exceed capacity");
    assert!(extents.validate_capacity(4096, 8, 5280).is_err(), "topk contract is six");
    assert!(extents.validate_capacity(4096, 6, 10240).is_err(), "row bytes disagree");
}

#[test]
fn write_creates_files_with_accurate_metadata() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (hidden, ids, routing, partials, compact) = filled_parts(1, 7);
    let capture = capture_from(1, &hidden, &ids, &routing, &partials, &compact);
    let ordinal = 7;
    let dir = capture.write(root.path(), ordinal)?;
    assert_eq!(dir, root.path().join(trace_dir_name(ordinal, &capture)));
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("metadata.json"))?)?;
    assert_eq!(metadata["schema"], TRACE_SCHEMA);
    assert_eq!(metadata["trace_ordinal"], ordinal);
    assert_eq!(metadata["process_id"], std::process::id());
    assert_eq!(metadata["layer"], 1);
    assert_eq!(metadata["executor_id"], 2);
    assert_eq!(metadata["rank"], 1);
    assert_eq!(metadata["wire"]["request_id"], 2026091950);
    assert_eq!(metadata["wire"]["placement_version"], 17);
    assert_eq!(metadata["rows"], 1);
    assert_eq!(metadata["kernel_capacity_rows"], 4096);
    assert_eq!(metadata["topk"], 6);
    assert_eq!(metadata["hidden_dim"], 5120);
    assert_eq!(metadata["input_dtype"], "fp8e4m3ue8m0k32");
    assert_eq!(metadata["input_row_bytes"], 5280);
    let expected = [
        ("input-payload.bin", "fp8e4m3ue8m0k32", serde_json::json!([1, 5280]), 5280usize),
        ("route-ids.bin", "int32", serde_json::json!([1, 6]), 24),
        ("route-routing-fp32.bin", "float32", serde_json::json!([1, 6]), 24),
        ("route-partials-fp32.bin", "float32", serde_json::json!([1, 6, 5120]), 6 * 5120 * 4),
        ("compact-output-bf16.bin", "bfloat16", serde_json::json!([1, 5120]), 5120 * 2),
    ];
    for (name, dtype, shape, bytes) in expected {
        let stat = std::fs::metadata(dir.join(name)).with_context(|| name.to_string())?;
        assert_eq!(stat.len() as usize, bytes, "{name} on-disk length");
        assert_eq!(metadata["files"][name]["dtype"], dtype);
        assert_eq!(metadata["files"][name]["shape"], shape);
        assert_eq!(metadata["files"][name]["bytes"], bytes);
    }
    // Contents round-trip exactly: ids and FP32 weights in original route order.
    let id_bytes = std::fs::read(dir.join("route-ids.bin"))?;
    for (index, id) in ids.iter().enumerate() {
        assert_eq!(&id_bytes[index * 4..index * 4 + 4], &id.to_le_bytes());
    }
    let routing_bytes = std::fs::read(dir.join("route-routing-fp32.bin"))?;
    for (index, weight) in routing.iter().enumerate() {
        assert_eq!(&routing_bytes[index * 4..index * 4 + 4], &weight.to_le_bytes());
    }
    assert_eq!(std::fs::read(dir.join("input-payload.bin"))?, hidden);
    assert_eq!(std::fs::read(dir.join("route-partials-fp32.bin"))?, partials);
    assert_eq!(std::fs::read(dir.join("compact-output-bf16.bin"))?, compact);
    Ok(())
}

#[test]
fn write_never_overwrites_and_rejects_duplicate_identity() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (hidden, ids, routing, partials, compact) = filled_parts(2, 9);
    let capture = capture_from(2, &hidden, &ids, &routing, &partials, &compact);
    let dir = capture.write(root.path(), 1)?;
    let metadata_before = std::fs::read(dir.join("metadata.json"))?;
    assert!(capture.write(root.path(), 1).is_err(), "same ordinal must not rewrite");
    assert_eq!(std::fs::read(dir.join("metadata.json"))?, metadata_before);
    Ok(())
}

#[test]
fn write_rejects_wrong_extents_and_leaves_no_partial_files() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (hidden, ids, routing, partials, compact) = filled_parts(1, 3);
    let capture = capture_from(1, &hidden[..hidden.len() - 1], &ids, &routing, &partials, &compact);
    assert!(capture.write(root.path(), 1).is_err(), "short input must be rejected");
    assert!(
        !root.path().join(trace_dir_name(1, &capture)).exists(),
        "failed capture must remove its partial directory"
    );
    // The same identity may succeed once the extents are corrected.
    let capture = capture_from(1, &hidden, &ids, &routing, &partials, &compact);
    assert!(capture.write(root.path(), 1).is_ok());
    Ok(())
}

#[cfg(unix)]
#[test]
fn write_surfaces_io_errors_without_panicking() -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::tempdir()?;
    let blocked = root.path().join("blocked");
    std::fs::create_dir(&blocked)?;
    std::fs::set_permissions(&blocked, std::fs::Permissions::from_mode(0o555))?;
    let (hidden, ids, routing, partials, compact) = filled_parts(1, 5);
    let capture = capture_from(1, &hidden, &ids, &routing, &partials, &compact);
    let result = capture.write(&blocked, 1);
    std::fs::set_permissions(&blocked, std::fs::Permissions::from_mode(0o755))?;
    assert!(result.is_err());
    assert!(std::fs::read_dir(&blocked)?.next().is_none());
    Ok(())
}

/// Serving-path integration: identical to production wiring, runs only when
/// the real native library, model and trace directory are provided.
#[test]
#[ignore = "requires GPU, native expert library, checkpoint and a fresh trace directory"]
fn opt_in_trace_captures_first_successful_rows1_and_rows2_on_serving_path() -> Result<()> {
    use super::super::ExpertLayer;
    use super::*;
    use ds41rt_transport::v41_expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    use ds41rt_transport::{
        ExpertProtocolV2Request, ExpertProtocolV2RowDescriptor, ExpertProtocolV2RouteEntry,
        ExpertV2Dtype, ExpertV2SourceKind, EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN,
    };
    let lib_path = std::env::var_os("DS41RT_MAPPED_EXPERT_LIBRARY").context("missing native expert library")?;
    let trace_root = std::env::var_os("DS41RT_EXPERT_TRACE_DIR").context("missing fresh trace directory")?;
    let trace_root = std::path::PathBuf::from(trace_root);
    let lib = unsafe { NativeLibrary::load(std::path::Path::new(&lib_path)) }?;
    let snapshot = std::env::var_os("DS41RT_MAPPED_EXPERT_MODEL").context("missing model")?;
    let catalog = ds41rt_loader::read_official_v41_catalog(
        ds41rt_loader::OFFICIAL_V41_MODEL_ID,
        std::path::Path::new(&snapshot),
    )?;
    let weights = ExpertWeights::load(
        &lib,
        &catalog,
        ExpertLayer::Backbone { layer: 1, rank: 1 },
        16 << 30,
    )?;
    let mut execution = weights.execution(4096, 16 << 30)?;
    let mut exchange = HostExpertExchange::new(4096)?;
    let build = |request_id: u64, rows: u32| -> Result<(ExpertProtocolV2Request, Vec<u8>)> {
        let mut hidden = Vec::new();
        for row in 0..rows {
            hidden.extend((0..5120).map(|i| (0x30 + (i + row * 11) % 97) as u8));
            hidden.extend([127u8; 160]);
        }
        let mut owned = ExpertProtocolV2Request::new(
            request_id,
            17,
            1,
            5120,
            ExpertV2Dtype::Fp8E4m3Ue8m0K32,
            (0..rows)
                .map(|r| ExpertProtocolV2RowDescriptor {
                    row_id: r as u64,
                    source_kind: ExpertV2SourceKind::Decode,
                    source_request_id: 1,
                    token_position: r as u64,
                    route_offset: r * 6,
                    route_count: 6,
                })
                .collect(),
            (0..rows * 6)
                .map(|r| ExpertProtocolV2RouteEntry {
                    row_index: r / 6,
                    expert_id: (r / 6 + r % 6 * 63) % 384,
                    gate_weight: 0.1234567,
                })
                .collect(),
            hidden.clone(),
        )?;
        owned.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        Ok((owned, hidden))
    };
    let trace_dirs = || -> Result<Vec<std::path::PathBuf>> {
        Ok(std::fs::read_dir(&trace_root)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .map_or(false, |name| name.starts_with("expert-trace-"))
            })
            .collect())
    };
    let bytes = ds41rt_transport::v41_expert::V41_PARTIAL_ROW_BYTES as usize;
    let prefix = EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN;
    let mut host = crate::v41_memory::HostAllocation::new(&lib, prefix + 2 * bytes + 64)?;
    host.bytes_mut().fill(0xa5);
    let alias = lib.cuda_host_buffer_device_alias(host.buffer)?;

    // rows=1 on the registered-destination path wins the rows=1 slot.
    let (owned, owned_hidden) = build(2026091901, 1)?;
    let frame = owned.encode()?;
    let request = ds41rt_transport::v41_expert::V41BackboneRequest::parse(&frame, 4096)?;
    let response = unsafe { execution.execute_mapped_request(&request, 2, &mut exchange, alias)? }
        .context("registered rows=1 response")?;
    assert_eq!(response.partial_output_payload.bytes, bytes);

    // A later rows=1 request must not capture again.
    let (owned_b, _hidden_b) = build(2026091902, 1)?;
    let frame_b = owned_b.encode()?;
    let request_b =
        ds41rt_transport::v41_expert::V41BackboneRequest::parse(&frame_b, 4096)?;
    let response_b = execution.execute_host_request(&request_b, 2, &mut exchange)?;
    assert_eq!(response_b.partial_output_payload.len(), bytes);

    // rows=2 on the host path wins the rows=2 slot.
    let (owned_c, owned_c_hidden) = build(2026091903, 2)?;
    let frame_c = owned_c.encode()?;
    let request_c =
        ds41rt_transport::v41_expert::V41BackboneRequest::parse(&frame_c, 4096)?;
    let response_c = execution.execute_host_request(&request_c, 2, &mut exchange)?;
    assert_eq!(response_c.partial_output_payload.len(), 2 * bytes);
    let response_c_bytes = response_c.partial_output_payload.to_vec();

    // A repeat rows=2 request must not capture again; the request itself
    // still succeeds and drains normally.
    let (owned_d, _hidden_d) = build(2026091904, 2)?;
    let frame_d = owned_d.encode()?;
    let request_d =
        ds41rt_transport::v41_expert::V41BackboneRequest::parse(&frame_d, 4096)?;
    execution.execute_host_request(&request_d, 2, &mut exchange)?;

    let dirs = trace_dirs()?;
    assert_eq!(dirs.len(), 2, "exactly one trace per row shape: {dirs:?}");
    let mut by_rows: std::collections::HashMap<u64, serde_json::Value> = Default::default();
    for dir in &dirs {
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("metadata.json"))?)?;
        assert_eq!(metadata["schema"], TRACE_SCHEMA);
        assert_eq!(metadata["layer"], 1);
        assert_eq!(metadata["executor_id"], 2);
        assert_eq!(metadata["rank"], 1);
        assert_eq!(metadata["wire"]["placement_version"], 17);
        assert_eq!(
            by_rows.insert(metadata["rows"].as_u64().unwrap(), metadata),
            None,
            "one trace per row shape"
        );
    }
    let rows1 = &by_rows[&1];
    let rows2 = &by_rows[&2];
    assert_eq!(rows1["wire"]["request_id"], 2026091901, "rows=1 winner is the registered request");
    assert_eq!(rows2["wire"]["request_id"], 2026091903, "rows=2 winner is the first rows=2 request");
    assert_eq!(rows1["compact_destination"], "registered_response_slot");
    assert_eq!(rows2["compact_destination"], "compact_output_allocation");
    assert!(rows1["trace_ordinal"].as_u64().unwrap() < rows2["trace_ordinal"].as_u64().unwrap());

    // The captured compact bytes are the actual delivered output bytes.
    let dir1 = dirs
        .iter()
        .find(|dir| dir.file_name().and_then(|n| n.to_str()).unwrap().contains("-rows1-"))
        .unwrap();
    let captured_compact = std::fs::read(dir1.join("compact-output-bf16.bin"))?;
    assert_eq!(captured_compact, &host.bytes_mut()[prefix..prefix + bytes]);
    let dir2 = dirs
        .iter()
        .find(|dir| dir.file_name().and_then(|n| n.to_str()).unwrap().contains("-rows2-"))
        .unwrap();
    assert_eq!(
        std::fs::read(dir2.join("compact-output-bf16.bin"))?,
        response_c_bytes
    );

    // Extents and route dumps match the wire contract and the request bytes.
    for (dir, rows, expected_bytes, expected_hidden) in [
        (dir1, 1u64, bytes, &owned_hidden),
        (dir2, 2, 2 * bytes, &owned_c_hidden),
    ] {
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("metadata.json"))?)?;
        for (name, file_bytes) in [
            ("input-payload.bin", rows as usize * 5280),
            ("route-ids.bin", rows as usize * 24),
            ("route-routing-fp32.bin", rows as usize * 24),
            ("route-partials-fp32.bin", rows as usize * 6 * 5120 * 4),
            ("compact-output-bf16.bin", expected_bytes),
        ] {
            assert_eq!(std::fs::metadata(dir.join(name))?.len() as usize, file_bytes, "{name}");
            assert_eq!(metadata["files"][name]["bytes"].as_u64().unwrap() as usize, file_bytes);
        }
        assert_eq!(std::fs::read(dir.join("input-payload.bin"))?, *expected_hidden);
    }
    let ids = std::fs::read(dir1.join("route-ids.bin"))?;
    for (index, route) in owned.routes.iter().enumerate() {
        assert_eq!(&ids[index * 4..index * 4 + 4], &(route.expert_id as i32).to_le_bytes());
    }
    eprintln!("PASS opt-in expert trace: bounded first-successful rows1+rows2, serving path");
    Ok(())
}
