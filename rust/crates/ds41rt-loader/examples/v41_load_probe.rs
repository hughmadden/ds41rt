//! Measure bounded real-checkpoint TP4 staging without loading a GPU or the full model.
use anyhow::{ensure, Context, Result};
use ds41rt_loader::{read_official_v41_catalog, V41ExpertSelection, OFFICIAL_V41_MODEL_ID};
use sha2::{Digest, Sha256};
use std::os::fd::AsRawFd;
use std::{collections::BTreeMap, path::PathBuf, time::Instant};
fn io_counts() -> BTreeMap<String, u64> {
    std::fs::read_to_string("/proc/self/io")
        .unwrap_or_default()
        .lines()
        .filter_map(|line| {
            let (key, value) = line.split_once(':')?;
            Some((key.to_owned(), value.trim().parse().ok()?))
        })
        .collect()
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() >= 6 && args.len() <= 8,
        "usage: v41_load_probe SNAPSHOT LAYER EXPERT_COUNT RANK SCRATCH_ROWS [--evict-read-ranges] [--prefetch]");
    ensure!(
        args[6..]
            .iter()
            .all(|a| a == "--evict-read-ranges" || a == "--prefetch"),
        "unknown probe option"
    );
    let evict = args[6..].iter().any(|a| a == "--evict-read-ranges");
    let prefetch = args[6..].iter().any(|a| a == "--prefetch");
    let snapshot = PathBuf::from(&args[1]);
    let layer = args[2].parse()?;
    let count: usize = args[3].parse()?;
    let rank = args[4].parse()?;
    let scratch_rows: usize = args[5].parse()?;
    ensure!(
        (1..=384).contains(&count) && (1..=5120).contains(&scratch_rows),
        "invalid probe extent"
    );
    let started = Instant::now();
    let catalog = read_official_v41_catalog(OFFICIAL_V41_MODEL_ID, &snapshot)?;
    let catalog_seconds = started.elapsed().as_secs_f64();
    let first = catalog.expert_staging(V41ExpertSelection::Backbone {
        layer,
        expert: 0,
        rank,
    })?;
    let mut staging = vec![0; first.staging_bytes()];
    let mut scratch = vec![
        0;
        first
            .minimum_read_scratch_bytes()
            .checked_mul(scratch_rows)
            .context("scratch overflow")?
    ];
    // Evict all selected ranges before either baseline or prefetch measurement.
    for expert in 0..count {
        let plan = catalog.expert_staging(V41ExpertSelection::Backbone {
            layer,
            expert,
            rank,
        })?;
        if evict {
            // Advisory eviction of only this expert's immutable payload ranges,
            // never global cache dropping; read_bytes reports actual disk reads.
            for name in plan.tensor_names() {
                let tensor = catalog.tensor(name)?;
                let file = std::fs::File::open(snapshot.join(&tensor.shard))?;
                let offset = i64::try_from(tensor.metadata.byte_offset)?;
                let bytes = i64::try_from(tensor.metadata.byte_length)?;
                let status = unsafe {
                    libc::posix_fadvise(file.as_raw_fd(), offset, bytes, libc::POSIX_FADV_DONTNEED)
                };
                ensure!(
                    status == 0,
                    "range eviction failed: {}",
                    std::io::Error::from_raw_os_error(status)
                );
            }
        }
    }
    let before = io_counts();
    let wall = Instant::now();
    let mut read_seconds = 0.;
    let mut useful_bytes = 0u64;
    let mut digest = Sha256::new();
    if prefetch {
        for expert in 0..4.min(count) {
            catalog
                .expert_staging(V41ExpertSelection::Backbone {
                    layer,
                    expert,
                    rank,
                })?
                .prefetch()?;
        }
    }
    for expert in 0..count {
        let plan = catalog.expert_staging(V41ExpertSelection::Backbone {
            layer,
            expert,
            rank,
        })?;
        if prefetch && expert + 4 < count {
            catalog
                .expert_staging(V41ExpertSelection::Backbone {
                    layer,
                    expert: expert + 4,
                    rank,
                })?
                .prefetch()?;
        }
        let started = Instant::now();
        plan.read_into(&mut staging, &mut scratch)?;
        read_seconds += started.elapsed().as_secs_f64();
        for range in plan.tensor_ranges() {
            useful_bytes += range.len() as u64;
            digest.update(&staging[range.clone()]);
        }
    }
    let wall_seconds = wall.elapsed().as_secs_f64();
    let after = io_counts();
    let io_delta: BTreeMap<_, _> = after
        .iter()
        .map(|(k, v)| (k, v.saturating_sub(*before.get(k).unwrap_or(&0))))
        .collect();
    println!(
        "{}",
        serde_json::json!({"snapshot":snapshot,"layer":layer,"expert_count":count,"rank":rank,"scratch_rows":scratch_rows,
        "scratch_bytes":scratch.len(),"staging_bytes":staging.len(),"catalog_seconds":catalog_seconds,
        "staging_read_seconds":read_seconds,"wall_seconds_including_sha256":wall_seconds,"useful_bytes":useful_bytes,
        "useful_gb_per_staging_second":useful_bytes as f64/read_seconds/1e9,"io_delta":io_delta,
        "staged_sha256":format!("{:x}",digest.finalize()),"gpu_loaded":false,"evict_read_ranges":evict,"prefetch":prefetch})
    );
    Ok(())
}
