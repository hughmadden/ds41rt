//! Explicit GPU smoke for the scoped target library. Never run by CPU tests.
use anyhow::{ensure, Result};
use clap::Parser;
use ds41rt_daemon::native_executor::target::{with_target, CacheInfo, SourceKind, TargetConfig, TargetInput};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{io::Write, net::SocketAddr, path::PathBuf, time::Instant};

#[derive(Parser)]
#[command(about = "Explicit native two-context GPU smoke; no HTTP or performance qualification")]
struct Args {
    #[arg(long)] snapshot: PathBuf,
    #[arg(long)] native_lib: PathBuf,
    #[arg(long, value_delimiter = ',')] peers: Vec<SocketAddr>,
    /// Fresh nonzero executor incarnation; never reuse an owner across runs.
    #[arg(long)] owner: u64,
    #[arg(long, default_value_t = 536870912)] cache_bytes: usize,
    #[arg(long, default_value_t = 80)] batch_tokens: u32,
    #[arg(long, default_value_t = 2048)] max_context_tokens: u32,
    /// Exact thinking=false tokenize result for "Reply with exactly: APPLE".
    #[arg(long, value_delimiter = ',', default_value = "0,128803,19905,418,9045,28,56684,4392,128804,128822")]
    tokens: Vec<u32>,
    /// One prefill plus optional one-token decode steps; this is a bounded smoke.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..=16))]
    steps: u32,
    /// Discard the second request's final ready batch instead of publishing it.
    #[arg(long)] cancel_ready: bool,
}

fn emit(value: Value) -> Result<()> {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    serde_json::to_writer(&mut out, &value)?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}
fn info(value: &CacheInfo) -> Value {
    json!({"owner": value.owner, "capacity_rows": value.capacity_rows,
        "source_page_capacity": value.source_page_capacity,
        "source_pages_free": value.source_pages_free, "cache_bytes": value.cache_bytes})
}
fn logits(bytes: &[u8]) -> Result<Vec<f32>> {
    ensure!(bytes.len() == 129280 * 4, "expected one full FP32 vocabulary row");
    let values = bytes.chunks_exact(4).map(|b| f32::from_ne_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
    ensure!(values.iter().all(|v| v.is_finite()), "native logits contain non-finite values");
    Ok(values)
}
fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..values.len() { if values[i] > values[best] { best = i; } }
    best
}
fn comparison(a: &[f32], b: &[f32]) -> Value {
    let squared = a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).powi(2)).sum::<f64>();
    let reference = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
    let max_abs = a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).abs()).fold(0.0f64, f64::max);
    json!({"relative_l2": (squared / reference.max(f64::MIN_POSITIVE)).sqrt(), "max_abs": max_abs})
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.peers.len() == 4, "exactly four peer endpoints required");
    ensure!(!args.tokens.is_empty() && args.tokens.len() <= 16
        && args.tokens.iter().all(|&token| token < 129280), "smoke requires 1..=16 valid token IDs");
    ensure!(args.tokens.len() as u64 + args.steps as u64 <= args.max_context_tokens as u64,
        "smoke exceeds configured context bound");
    tracing_subscriber::fmt().with_writer(std::io::stderr)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).init();
    emit(json!({"event":"start", "purpose":"native_target_gpu_smoke_not_throughput",
        "owner":args.owner, "tokens":args.tokens, "steps":args.steps,
        "cancel_ready":args.cancel_ready, "cache_budget_bytes":args.cache_bytes,
        "batch_tokens":args.batch_tokens, "warmup_performed":false}))?;
    let startup = Instant::now();
    let config = TargetConfig { owner: args.owner, snapshot: args.snapshot,
        native_lib: args.native_lib, peers: args.peers.try_into().unwrap(),
        batch_tokens: args.batch_tokens, max_context_tokens: args.max_context_tokens,
        slots: 2, cache_bytes: args.cache_bytes };
    with_target(config, |mut target| {
        let initial = target.bank().info();
        emit(json!({"event":"initialized", "startup_ms":startup.elapsed().as_secs_f64()*1000.0,
            "bank":info(&initial), "constructor_physical_banks":1, "contexts":2}))?;
        let (runtime, bank, [first, second]) = target.split();
        let a = bank.admit(0, 100)?;
        let b = bank.admit(1, 101)?;
        ensure!(a.owner() == args.owner && b.owner() == args.owner && a != b, "admission identity differs");
        let mut tokens = args.tokens.clone();
        let mut generated = Vec::new();
        for step in 0..args.steps {
            let before = [bank.committed_end(a)?, bank.committed_end(b)?];
            let selected = tokens.len()-1;
            let input = |request| TargetInput { request, tokens: tokens.clone(), selected: vec![selected],
                kind: if step == 0 { SourceKind::Prefill } else { SourceKind::Decode }, placement: 1 };
            let x = first.submit(input(a))?;
            let y = second.submit(input(b))?;
            let execute = Instant::now();
            let (left, right) = runtime.block_on(async { tokio::join!(first.execute(x), second.execute(y)) });
            left?; right?;
            let execute_ms = execute.elapsed().as_secs_f64()*1000.0;
            {
                let left = first.logits(x)?;
                let right = second.logits(y)?;
                ensure!(left.rows == 1 && right.rows == 1 && left.selected == [selected]
                    && right.selected == [selected], "compact logits row binding differs");
                ensure!(left.positions == [before[0]+selected as u64]
                    && right.positions == [before[1]+selected as u64], "logit token positions differ");
                // Only inspect descriptors here; no asynchronous external consumer
                // is launched, and neither descriptor survives its logits borrow.
                let (left, right) = unsafe { (left.device_buffer()?, right.device_buffer()?) };
                ensure!(!left.ptr.is_null() && !right.ptr.is_null() && left.ptr != right.ptr,
                    "independent target contexts alias logits storage");
                ensure!(left.bytes == 129280*4 && right.bytes == left.bytes
                    && left.device_id == right.device_id, "native logits geometry/device differs");
            }
            let download = Instant::now();
            let (left, right) = runtime.block_on(async {
                tokio::join!(first.download_logits(x, &[0]), second.download_logits(y, &[0]))
            });
            let (left, right) = (left?, right?);
            let download_ms = download.elapsed().as_secs_f64()*1000.0;
            let (a_values, b_values) = (logits(&left)?, logits(&right)?);
            let (a_token, b_token) = (argmax(&a_values), argmax(&b_values));
            let cancelled = args.cancel_ready && step+1 == args.steps;
            emit(json!({"event":"logits", "step":step, "input_rows":tokens.len(),
                "execute_ms":execute_ms, "diagnostic_d2h_ms":download_ms,
                "greedy_token_ids":[a_token,b_token], "greedy_logits":[a_values[a_token],b_values[b_token]],
                "sha256":[format!("{:x}",Sha256::digest(&left)),format!("{:x}",Sha256::digest(&right))],
                "byte_exact_between_contexts":left == right, "difference":comparison(&a_values,&b_values),
                "committed_before":before, "second_context_will_cancel":cancelled}))?;
            ensure!(a_token == b_token, "identical disjoint requests disagree on greedy next token");
            let a_end = first.commit(x, tokens.len() as u32)?;
            let b_end = if cancelled { second.cancel(y)?; bank.committed_end(b)? }
                else { second.commit(y, tokens.len() as u32)? };
            ensure!(a_end == before[0]+tokens.len() as u64
                && b_end == before[1]+if cancelled {0} else {tokens.len() as u64}, "accepted publication differs");
            emit(json!({"event":"published", "step":step, "committed_end":[a_end,b_end],
                "second_context_cancelled":cancelled, "bank":info(&bank.info())}))?;
            generated.push(a_token as u32);
            tokens = vec![a_token as u32];
        }
        bank.release(a)?; bank.release(b)?;
        ensure!(bank.committed_end(a).is_err() && bank.committed_end(b).is_err(), "released native lease remained valid");
        let final_info = bank.info();
        ensure!(final_info.source_pages_free == initial.source_pages_free, "native source page credits leaked");
        emit(json!({"event":"complete", "ok":true, "generated_token_ids":generated,
            "released_leases_rejected":true, "source_credits_restored":true, "bank":info(&final_info)}))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn host_oracle_checks_geometry_finiteness_and_lowest_tied_id() {
        assert!(logits(&[0; 4]).is_err());
        let mut bytes = vec![0; 129280*4];
        bytes[12..16].copy_from_slice(&f32::NAN.to_ne_bytes());
        assert!(logits(&bytes).is_err());
        assert_eq!(argmax(&[-1.0, 4.0, 4.0, 0.0]), 1);
        assert_eq!(comparison(&[1.0, 2.0], &[1.0, 2.0])["relative_l2"], 0.0);
    }
    #[test]
    fn exact_apple_tokens_are_default_and_probe_is_bounded() {
        let base = ["probe", "--snapshot", "/model", "--native-lib", "/native.so",
            "--peers", "127.0.0.1:1,127.0.0.1:2,127.0.0.1:3,127.0.0.1:4", "--owner", "11"];
        let args = Args::try_parse_from(base).unwrap();
        assert_eq!(args.peers.len(), 4);
        assert_eq!(args.tokens, [0,128803,19905,418,9045,28,56684,4392,128804,128822]);
        assert_eq!(args.steps, 1);
        let mut excessive = base.to_vec(); excessive.extend(["--steps", "17"]);
        assert!(Args::try_parse_from(excessive).is_err());
    }
}
