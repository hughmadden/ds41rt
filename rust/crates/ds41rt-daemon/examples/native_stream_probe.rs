//! Explicit scoped native streaming correctness probe; never a CPU-test workload.
use anyhow::{ensure, Result};
use clap::Parser;
use ds41rt_daemon::native_executor::target::{with_target, SourceKind, StreamInput, TargetConfig, TargetInput};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{cell::Cell, io::Write, net::SocketAddr, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)] snapshot: PathBuf,
    #[arg(long)] native_lib: PathBuf,
    #[arg(long, value_delimiter=',')] peers: Vec<SocketAddr>,
    #[arg(long)] owner: u64,
    /// JSON array of exact tokenizer IDs. Omit for the ten-token APPLE prompt.
    #[arg(long)] tokens_file: Option<PathBuf>,
    #[arg(long, default_value_t=256)] batch_tokens: u32,
    #[arg(long, default_value_t=80)] chunk_rows: usize,
    #[arg(long, default_value_t=536870912)] source_pool_budget_bytes: usize,
    #[arg(long, default_value_t=2, value_parser=clap::value_parser!(u32).range(1..=3))] repeats: u32,
    /// First cancel at a native dispatch/publication callback, then prove reuse.
    #[arg(long)] cancel_after_checks: Option<usize>,
    /// Cancel the final ready result instead of publishing it.
    #[arg(long)] cancel_ready: bool,
}
fn emit(v: Value) -> Result<()> {
    let mut out=std::io::stdout().lock(); serde_json::to_writer(&mut out,&v)?;
    out.write_all(b"\n")?;out.flush()?;Ok(())
}
fn summarize(bytes: &[u8]) -> Result<(u32,String)> {
    ensure!(bytes.len()==129280*4,"expected one complete vocabulary row");
    let values=bytes.chunks_exact(4).map(|b|f32::from_ne_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
    ensure!(values.iter().all(|v|v.is_finite()),"nonfinite streaming logits");
    let mut best=0;
    for i in 1..values.len() {if values[i]>values[best] {best=i;}}
    Ok((best as u32,format!("{:x}",Sha256::digest(bytes))))
}
fn main() -> Result<()> {
    let args=Args::parse();ensure!(args.peers.len()==4,"four peers required");
    let tokens:Vec<u32>=if let Some(path)=args.tokens_file {serde_json::from_slice(&std::fs::read(path)?)?}
        else {vec![0,128803,19905,418,9045,28,56684,4392,128804,128822]};
    ensure!(!tokens.is_empty() && tokens.len()<=4096,"probe input must contain1..4096 tokens");
    tracing_subscriber::fmt().with_writer(std::io::stderr)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).init();
    emit(json!({"event":"start","purpose":"native_stream_correctness_not_throughput",
        "owner":args.owner,"prompt_rows":tokens.len(),"chunk_rows":args.chunk_rows,
        "tokens_sha256":format!("{:x}",Sha256::digest(serde_json::to_vec(&tokens)?)),
        "repeats":args.repeats,"cancel_after_checks":args.cancel_after_checks,
        "cancel_ready":args.cancel_ready,"warmup_performed":false}))?;
    let config=TargetConfig{owner:args.owner,snapshot:args.snapshot,native_lib:args.native_lib,
        peers:args.peers.try_into().unwrap(),batch_tokens:args.batch_tokens,
        max_context_tokens:8192,slots:2,cache_bytes:args.source_pool_budget_bytes,dspark:None};
    with_target(config,|mut target| {
        let initial=target.bank().info();
        let input=|request|StreamInput{request,tokens:tokens.clone(),chunk_rows:args.chunk_rows,selected:vec![tokens.len()-1]};
        if let Some(limit)=args.cancel_after_checks {
            let request=target.bank().admit(0,90)?;let calls=Cell::new(0usize);
            let keep=||{let n=calls.get();calls.set(n+1);n<limit};
            let runtime=target.runtime();
            let canceled=runtime.block_on(target.stream_prefill(input(request),&keep)).is_err();
            ensure!(canceled,"probe cancellation threshold was not reached");
            ensure!(target.bank().committed_end(request).is_err(),"canceled stream lease survived");
            ensure!(target.bank().info().source_pages_free==initial.source_pages_free,"canceled stream leaked pages");
            emit(json!({"event":"canceled_inflight","checks":calls.get(),"lease_revoked":true,"source_credits_restored":true}))?;
        }
        let mut first_hash=None;
        for iteration in 0..args.repeats {
            let request=target.bank().admit(0,100+iteration as u64)?;
            let runtime=target.runtime();let start=Instant::now();
            let mut result=runtime.block_on(target.stream_prefill(input(request),&||true))?;
            let elapsed=start.elapsed().as_secs_f64()*1000.0;
            {
                let logits=result.logits()?;
                ensure!(logits.rows==1 && logits.positions==[tokens.len() as u64-1]
                    && logits.selected==[tokens.len().min(128)-1],"stream result row binding differs");
            }
            let bytes=runtime.block_on(result.download_logits())?;
            let (token,hash)=summarize(&bytes)?;
            let equal=first_hash.as_ref().map_or(true,|first|*first==hash);
            emit(json!({"event":"stream_logits","iteration":iteration,"prompt_rows":tokens.len(),
                "decoder_rows":tokens.len().min(128),"greedy_token_id":token,"sha256":hash,
                "byte_exact_to_first":equal,"execute_ms":elapsed}))?;
            ensure!(equal,"same streaming input produced differing logits");
            first_hash=Some(hash);
            if args.cancel_ready && iteration+1==args.repeats {
                result.cancel()?;
                ensure!(target.bank().committed_end(request).is_err(),"ready-cancel lease survived");
                emit(json!({"event":"canceled_ready","iteration":iteration,"lease_revoked":true}))?;
            } else {
                ensure!(result.commit()?==tokens.len() as u64,"stream published wrong end");
                let (runtime,bank,[lane,_])=target.split();
                let ticket=lane.submit(TargetInput{request,tokens:vec![token],selected:vec![0],kind:SourceKind::Decode,placement:1})?;
                runtime.block_on(lane.execute(ticket))?;
                let bytes=runtime.block_on(lane.download_logits(ticket,&[0]))?;
                let (next,hash)=summarize(&bytes)?;
                ensure!(lane.commit(ticket,1)?==tokens.len() as u64+1,"decode did not continue streamed cache");
                bank.release(request)?;
                emit(json!({"event":"decoded","iteration":iteration,"greedy_token_id":next,"sha256":hash,
                    "accepted_end":tokens.len()+1}))?;
            }
            ensure!(target.bank().info().source_pages_free==initial.source_pages_free,"stream probe leaked pages");
        }
        emit(json!({"event":"complete","ok":true,"source_credits_restored":true}))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn diagnostic_logits_reject_partial_or_nonfinite_rows() {
        assert!(summarize(&[0;4]).is_err());
        let mut values=vec![0;129280*4];values[4..8].copy_from_slice(&f32::INFINITY.to_ne_bytes());
        assert!(summarize(&values).is_err());
        values[4..8].copy_from_slice(&1.0f32.to_ne_bytes());
        assert_eq!(summarize(&values).unwrap().0,1);
    }
}
