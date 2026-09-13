//! Isolated experiment: replay private verifier proposals, discard, never commit.
use super::*;
use ds41rt_ffi::Ds41rtDeviceBuffer;
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicUsize, Ordering};
use ds41rt_transport::v41_expert::V41Tp4RocePending;
struct ResetPolling;
impl Drop for ResetPolling {
    fn drop(&mut self) { V41Tp4RocePending::set_probe_poll_quantum(250); }
}
static SNAPSHOTS: AtomicUsize = AtomicUsize::new(0);

fn hash_buffer(lib: &NativeLibrary, hash: &mut Sha256, mut buffer: Ds41rtDeviceBuffer,
    offset: usize, bytes: usize) -> Result<()> {
    ensure!(offset.checked_add(bytes).is_some_and(|n| n <= buffer.bytes), "probe hash bounds");
    buffer.ptr = unsafe { buffer.ptr.cast::<u8>().add(offset).cast() };
    buffer.bytes = bytes;
    if bytes != 0 {
        let mut host = vec![0; bytes];
        lib.copy_d2h(&mut host, buffer)?;
        hash.update(host);
    }
    Ok(())
}
fn cache_hash(lib: &NativeLibrary, requests: &Requests<'_>, batch: &RequestBatch) -> Result<String> {
    let mut hash = Sha256::new();
    let batch = batch.cache()?;
    for layer in 0..40 {
        let state = requests.cache().window(batch, layer)?;
        for chunk in batch.window_chunks(layer)? {
            let view = state.view(chunk.lease)?;
            hash.update(view.end.to_ne_bytes()); hash.update(view.begin.to_ne_bytes());
            hash_buffer(lib, &mut hash, view.device_end, 0, 8)?;
            let begin = view.begin.max(view.end.saturating_sub(128));
            // Only initialized ring rows; split around the physical wrap.
            let mut position = begin;
            while position < view.end {
                let physical = position as usize % 128;
                let count = ((view.end-position) as usize).min(128-physical);
                hash_buffer(lib, &mut hash, view.values, physical*512, count*512)?;
                hash_buffer(lib, &mut hash, view.scales, physical*16, count*16)?;
                position += count as u64;
            }
        }
    }
    for layer in [2, 8, 14, 20] {
        let state = requests.cache().source(batch, layer)?;
        for chunk in batch.source_chunks(layer)? {
            let kv = state.kv_cache(chunk.lease)?;
            let index = state.index_cache(chunk.lease)?;
            ensure!(kv.rows == index.rows && kv.pages == index.pages, "probe source correspondence");
            hash.update((kv.rows as u64).to_ne_bytes());
            hash_buffer(lib, &mut hash, kv.device_rows, 0, 8)?;
            hash_buffer(lib, &mut hash, kv.device_pages, 0, kv.rows.div_ceil(256)*4)?;
            for (logical, &page) in kv.pages.iter().take(kv.rows.div_ceil(256)).enumerate() {
                let count = (kv.rows-logical*256).min(256);
                for (buffer, width) in [(kv.values, 256), (kv.scales, 32), (index.packed, 64), (index.scales, 4)] {
                    hash_buffer(lib, &mut hash, buffer, page as usize*256*width, count*width)?;
                }
            }
        }
    }
    Ok(format!("{:x}", hash.finalize()))
}

pub(super) fn run<'w, 'a>(lib: &'a NativeLibrary, runtime: &tokio::runtime::Runtime,
    first: &mut TargetPass<'w, 'a>, second: &mut TargetPass<'w, 'a>, requests: &mut Requests<'a>,
    first_transport: &mut NativeTp4Wave<'a>, second_transport: &mut NativeTp4Wave<'a>,
    active: &[Option<Active<'a>>], members: &[Vec<usize>; 2], inputs: &[Vec<Vec<u32>>; 2],
    draft: Option<&DraftRuntime<'_, 'a>>,
) -> Result<bool> {
    let Ok(directory) = std::env::var("DS41RT_PREFIX_PROBE_DIR") else { return Ok(false); };
    if members.iter().any(|m| m.len()!=1) || inputs.iter().flatten().any(|x| x.len()!=6)
        || members.iter().flatten().any(|&slot| {
            let r=active[slot].as_ref().unwrap(); r.generated<16 || r.constraint.is_some()
        }) { return Ok(false); }
    let snapshot=SNAPSHOTS.load(Ordering::Relaxed);
    if snapshot>=6 { return Ok(false); }
    SNAPSHOTS.store(snapshot+1, Ordering::Relaxed);
    let _reset=ResetPolling;
    let ids: Vec<_>=members.iter().flatten().map(|&slot| active[slot].as_ref().unwrap().id).collect();
    let ends: Vec<_>=members.iter().flatten().map(|&slot| requests.cache().committed_end(active[slot].as_ref().unwrap().lease)).collect::<Result<_>>()?;
    let confidence: Vec<_>=ids.iter().map(|&id| draft.and_then(|d| d.confidence_trace(id)).map(|x|x.to_vec())).collect();
    let mut expected_hash: Option<Vec<String>>=None;
    let mut full_best: Option<[Vec<u32>;2]>=None;
    let mut samples=Vec::new();
    let shapes: Vec<_>=[[1,1],[3,3],[5,5],[1,5],[5,1]].into_iter()
        .flat_map(|lengths| [250,50,0].map(|interval| (interval,lengths))).collect();
    // Warm all shapes, then rotate and alternate measured order. A full-prefix
    // sentinel brackets every sweep and is excluded from timing summaries.
    for sweep in 0..4 {
        let mut order=shapes.clone(); order.rotate_left((snapshot+sweep*7)%15);
        if (snapshot+sweep)%2!=0 { order.reverse(); }
        order.insert(0,(250,[5,5])); order.push((250,[5,5]));
        for (ordinal, (poll_us,lengths)) in order.iter().enumerate() {
            V41Tp4RocePending::set_probe_poll_quantum(*poll_us);
            V41Tp4RocePending::take_probe_poll_counts();
            let trimmed: [Vec<Vec<u32>>;2]=std::array::from_fn(|lane|vec![inputs[lane][0][..lengths[lane]+1].to_vec()]);
            let start=Instant::now();
            let mut batches=[Some(prepare_decode_lane(requests,active,&members[0],&trimmed[0],true)?),
                Some(prepare_decode_lane(requests,active,&members[1],&trimmed[1],true)?)];
            let prepare_us=start.elapsed().as_micros() as u64;
            if expected_hash.is_none() {
                expected_hash=Some(batches.iter().map(|b| cache_hash(lib,requests,b.as_ref().unwrap())).collect::<Result<Vec<_>>>()?);
            }
            let [a,b]=&mut batches;
            let start=Instant::now();
            let results=runtime.block_on(async { tokio::join!(
                execute_logits(lib,first,requests,a,first_transport,true),
                execute_logits(lib,second,requests,b,second_transport,true),
            ) });
            let verify_us=start.elapsed().as_micros() as u64;
            let poll_counts=V41Tp4RocePending::take_probe_poll_counts();
            // Both futures are complete before any discard, including failure.
            let checked=(|| -> Result<_> {
                let scores=[results.0?,results.1?];
                let hashes=batches.iter().map(|b| cache_hash(lib,requests,b.as_ref().unwrap())).collect::<Result<Vec<_>>>()?;
                if let Some(expected)=&expected_hash { ensure!(expected==&hashes,"probe mutated committed cache"); }
                else { expected_hash=Some(hashes); }
                for (lane,&slot) in members.iter().flatten().enumerate() {
                    ensure!(requests.cache().committed_end(active[slot].as_ref().unwrap().lease)?==ends[lane],"probe advanced context");
                }
                let best=[scores[0].best.clone(),scores[1].best.clone()];
                if *lengths==[5,5] {
                    if let Some(expected)=&full_best { ensure!(expected==&best,"full-prefix sentinel changed"); }
                    else { full_best=Some(best.clone()); }
                }
                let work: Vec<_>=[&*first,&*second].iter().map(|pass| {
                    let routes=pass.captured_routes();
                    ensure!(routes.len()==40 && routes.iter().all(|layer| !layer.is_empty()),"probe routes incomplete");
                    let mut unique=0;let mut groups=0;
                    for layer in routes {
                        let mut counts=[0usize;384];
                        for row in layer { for &expert in row { counts[expert as usize]+=1; } }
                        unique+=counts.iter().filter(|&&n|n>0).count();
                        groups+=counts.iter().map(|n|n.div_ceil(16)).sum::<usize>();
                    }
                    Ok((unique as f64/40.,groups as f64/40.))
                }).collect::<Result<_>>()?;
                Ok((best,work))
            })();
            let cleanup_a=first.discard(batches[0].as_mut().unwrap());
            let cleanup_b=second.discard(batches[1].as_mut().unwrap());
            let (best,work)=checked?; cleanup_a?; cleanup_b?;
            samples.push(serde_json::json!({"sweep":sweep,"ordinal":ordinal,"lengths":lengths,"poll_us":poll_us,"poll_counts":poll_counts,
                "sentinel":ordinal==0 || ordinal==16,"warmup":sweep==0,"prepare_us":prepare_us,
                "verify_us":verify_us,"best":best,"expert_work":work}));
        }
    }
    std::fs::create_dir_all(&directory)?;
    let path=std::path::Path::new(&directory).join(format!("snapshot-{snapshot}.json"));
    let mut file=std::fs::OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer_pretty(&mut file,&serde_json::json!({"ids":ids,"ends":ends,
        "inputs":inputs,"confidence":confidence,"cache_sha256":expected_hash,"samples":samples}))?;
    tracing::warn!(snapshot, ?ids, ?ends, "matched prefix probe complete");
    Ok(true)
}
