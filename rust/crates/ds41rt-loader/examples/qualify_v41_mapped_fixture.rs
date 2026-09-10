//! Temporary Goal-1 check for the nonzero sparse-header fixture; see TO_DELETE_SCAFFOLDING.md.
use anyhow::{ensure, Context, Result};
use ds41rt_loader::{
    read_official_v41_catalog, EngramPrefetcher, PrefetchOutcome, OFFICIAL_V41_MODEL_ID,
};
use std::{path::PathBuf, sync::Arc};
fn main() -> Result<()> {
    let snapshot = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .context("expected immutable fixture directory")?,
    );
    let catalog = read_official_v41_catalog(OFFICIAL_V41_MODEL_ID, &snapshot)?;
    let prefetch = EngramPrefetcher::new(4, 24, 16)?;
    for &layer in &catalog.config().text().engram_layer_ids {
        // SAFETY: the qualification harness exclusively owns this immutable fixture
        // while this child process is alive and mutates it only after exit.
        let table = Arc::new(unsafe { catalog.map_engram(layer)? });
        let last = table.weights().rows() - 1;
        let rows = [last, 0, last];
        let outcome = prefetch
            .try_submit(Arc::clone(&table), &rows)?
            .context("unexpected backpressure")?
            .wait()?;
        ensure!(
            matches!(outcome, PrefetchOutcome::Advised { .. }),
            "prefetch cancelled"
        );
        let mut weights = vec![0; 3 * 256];
        let mut scales = vec![0; 3 * 8];
        table.weights().gather_into(&rows, &mut weights)?;
        table.scales().gather_into(&rows, &mut scales)?;
        ensure!(
            weights[..256].iter().all(|&b| b == 29),
            "last weight row mismatch"
        );
        ensure!(
            weights[256..512].iter().all(|&b| b == 17),
            "first weight row mismatch"
        );
        ensure!(
            weights[512..].iter().all(|&b| b == 29),
            "duplicate last weight row mismatch"
        );
        ensure!(
            scales[..8].iter().all(|&b| b == 128),
            "last scale row mismatch"
        );
        ensure!(
            scales[8..16].iter().all(|&b| b == 127),
            "first scale row mismatch"
        );
        ensure!(
            scales[16..].iter().all(|&b| b == 128),
            "duplicate last scale row mismatch"
        );
    }
    println!("qualified two mapped engram tables, nonzero first/last rows, paired prefetch and duplicate gathers");
    Ok(())
}
