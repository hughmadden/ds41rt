//! TP4 dispatch through persistent RoCE QPs; TCP is used only for bootstrap.
use super::{V41BackboneRequest, V41Tp4ChunkReceiver};
use crate::verbs::LocalTp4Client;
use crate::{ExpertProtocolV2Request, TcpTransportConfig};
use anyhow::{ensure, Result};
use std::net::SocketAddr;

pub struct V41Tp4Roce {
    clients: LocalTp4Client,
    executors: [u64; 4],
    capacity: u32,
    max_frame_bytes: usize,
}
impl V41Tp4Roce {
    pub fn new(
        peers: [SocketAddr; 4],
        executors: [u64; 4],
        capacity: u32,
        config: TcpTransportConfig,
    ) -> Result<Self> {
        ensure!(
            capacity > 0 && capacity <= 4096,
            "invalid native TP capacity"
        );
        ensure!(
            !config.timeout.is_zero(),
            "native RoCE timeout must be positive"
        );
        ensure!(
            config.max_frame_bytes >= 128 + 10240 + 40 + 6 * 12
                && config.max_frame_bytes <= 64 * 1024 * 1024,
            "invalid native RoCE frame budget"
        );
        for rank in 0..4 {
            ensure!(
                executors[rank] != 0 && !executors[..rank].contains(&executors[rank]),
                "native TP executor identities must be distinct and nonzero"
            );
            ensure!(
                !peers[..rank].contains(&peers[rank]),
                "native TP endpoints must be distinct"
            );
        }
        Ok(Self {
            clients: LocalTp4Client::new(peers, config.clone()),
            executors,
            capacity,
            max_frame_bytes: config.max_frame_bytes,
        })
    }
    pub fn capacity(&self) -> u32 {
        self.capacity
    }
    /// Reset persistent QPs before a new admission. Pending dispatches borrow this
    /// owner exclusively, so an in-flight wave cannot be reset through this API.
    pub fn reset_connections(&mut self) {
        self.clients.reset();
    }

    /// Send the same canonical request to all ranks and accept every route row.
    /// The synchronous sink must consume/copy its frame slice before returning.
    /// It runs on the polling thread, allowing CUDA owners to stay thread-local.
    /// On error or cancellation the caller must discard partial destination state;
    /// the streaming QP path does not replay partially delivered requests.
    pub async fn execute<F>(&mut self, request: &ExpertProtocolV2Request, sink: F) -> Result<()>
    where
        F: FnMut(usize, u32, &[u8]) -> Result<()>,
    {
        self.dispatch(request).await?.receive(sink).await
    }

    /// Post all four requests directly from the inference owner. Remote work
    /// overlaps the shared FFN; dispatch completion is not send completion.
    pub async fn dispatch<'c, 'r>(
        &'c mut self,
        request: &'r ExpertProtocolV2Request,
    ) -> Result<V41Tp4RocePending<'c, 'r>> {
        let frame = request.encode()?;
        ensure!(
            frame.len() <= self.max_frame_bytes,
            "native request exceeds RoCE frame budget"
        );
        let native = V41BackboneRequest::parse(&frame, self.capacity)?;
        let receiver = V41Tp4ChunkReceiver::new(&native, self.executors, self.max_frame_bytes)?;
        self.clients.dispatch(request)?;
        Ok(V41Tp4RocePending {
            receiver,
            owner: self,
            _request: std::marker::PhantomData,
            complete: false,
        })
    }
}

/// Holds exclusive admission until all four rank planes have been consumed.
/// Cancellation resets the QPs before another wave can reuse them.
pub struct V41Tp4RocePending<'c, 'r> {
    receiver: V41Tp4ChunkReceiver,
    owner: &'c mut V41Tp4Roce,
    _request: std::marker::PhantomData<&'r ExpertProtocolV2Request>,
    complete: bool,
}
impl V41Tp4RocePending<'_, '_> {
    pub async fn receive<F>(mut self, mut sink: F) -> Result<()>
    where
        F: FnMut(usize, u32, &[u8]) -> Result<()>,
    {
        // Progress all four QPs on the inference owner. Yield for cancellation
        // and other work after bounded polling; no blocking completion wait.
        let mut quantum = std::time::Instant::now();
        loop {
            let receiver = &mut self.receiver;
            if self.owner.clients.poll(|chunk| {
                receiver.push_rdma(chunk, |rank, start, bytes| {
                    ensure!(
                        rank == chunk.stream_id,
                        "native executor identity does not match its RoCE peer"
                    );
                    sink(rank, start, bytes)
                })?;
                Ok(())
            })? {
                break;
            }
            if quantum.elapsed() >= std::time::Duration::from_micros(250) {
                tokio::task::yield_now().await;
                quantum = std::time::Instant::now();
            } else {
                std::hint::spin_loop();
            }
        }
        ensure!(
            self.receiver.complete(),
            "native TP4 RoCE response coverage is incomplete"
        );
        self.complete = true;
        Ok(())
    }
}
impl Drop for V41Tp4RocePending<'_, '_> {
    fn drop(&mut self) {
        if !self.complete {
            self.owner.reset_connections();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{VerbsHostProtocolV2ResponseChunk, VerbsHostProtocolV2ResponsePayload};

    #[test]
    #[ignore = "requires four idle live V4.1 FP8 RoCE expert workers"]
    fn local_qps_replay_cancel_and_recover_live() -> Result<()> {
        let peers: [SocketAddr; 4] = std::env::var("DS41RT_LIVE_ROCE_PEERS")?
            .split(',')
            .map(str::parse)
            .collect::<std::result::Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| anyhow::anyhow!("four peers required"))?;
        let capacity: u32 = std::env::var("DS41RT_LIVE_ROCE_CAPACITY")
            .unwrap_or_else(|_| "80".into()).parse()?;
        ensure!([80, 256, 1024, 4096].contains(&capacity), "unsupported fixture capacity");
        let mut shapes = vec![1, 6, 16, 80];
        shapes.extend([256, 1024, 4096].into_iter().filter(|&rows| rows <= capacity));
        shapes.extend([6, 1]);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        runtime.block_on(async {
            let mut client = V41Tp4Roce::new(
                peers,
                [1, 2, 3, 4],
                capacity,
                TcpTransportConfig {
                    timeout: std::time::Duration::from_secs(10),
                    max_frame_bytes: 64 * 1024 * 1024,
                },
            )?;
            for rows in shapes {
                let base = super::super::tests::request(rows);
                let mut payload = Vec::new();
                for row in 0..rows {
                    payload.extend(
                        (0..5120).map(|i| {
                            0x30 + ((i + row) % 8) as u8 + if i % 2 == 0 { 0x80 } else { 0 }
                        }),
                    );
                    payload.extend([120; 160]);
                }
                let mut request = ExpertProtocolV2Request::new(
                    1000 + rows as u64,
                    base.header.placement_version,
                    39,
                    5120,
                    crate::ExpertV2Dtype::Fp8E4m3Ue8m0K32,
                    base.rows,
                    base.routes,
                    payload,
                )?;
                request.header.flags = base.header.flags;
                let mut expected = None;
                for repetition in 0..3 {
                    request.header.request_id += 1;
                    let mut planes = vec![vec![0; rows as usize * 10240]; 4];
                    client
                        .execute(&request, |rank, start, bytes| {
                            let offset = start as usize * 10240;
                            planes[rank][offset..offset + bytes.len()].copy_from_slice(bytes);
                            Ok(())
                        })
                        .await?;
                    for plane in &planes {
                        ensure!(
                            plane
                                .chunks_exact(2)
                                .any(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7fff != 0),
                            "zero expert plane"
                        );
                        ensure!(
                            plane
                                .chunks_exact(2)
                                .all(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 != 0x7f80),
                            "nonfinite expert plane"
                        );
                    }
                    if let Some(ref previous) = expected {
                        ensure!(previous == &planes, "replay changed rank planes");
                    }
                    expected = Some(planes);
                    if repetition == 0 {
                        request.header.request_id += 1;
                        // Abandon after posting all ranks, before receiving any.
                        drop(client.dispatch(&request).await?);
                    } else if repetition == 1 {
                        request.header.request_id += 1;
                        let failed = client
                            .execute(&request, |_, _, _| anyhow::bail!("injected sink failure"))
                            .await;
                        ensure!(failed.is_err(), "sink failure was swallowed");
                    }
                }
                eprintln!(
                    "local QPs rows={rows}: exact replay after abandoned dispatch and sink failure"
                );
            }
            Ok(())
        })
    }

    #[test]
    fn rdma_chunks_preserve_rank_rows_and_reject_stale_or_reordered_data() -> Result<()> {
        let request = super::super::tests::request(2);
        let frame = request.encode()?;
        let native = V41BackboneRequest::parse(&frame, 2)?;
        let mut receiver = V41Tp4ChunkReceiver::new(&native, [1, 2, 3, 4], 200_000)?;
        for row in 0..2 {
            for rank in [3, 1, 0, 2] {
                let bytes = vec![(rank + row) as u8; super::super::V41_PARTIAL_ROW_BYTES as usize];
                let mut indices = [0];
                let response = native
                    .response_chunk(rank as u64 + 1, row, &bytes, &mut indices, 200_000)?
                    .to_owned()?;
                let wire_bytes = response.encode()?.len();
                let mut chunk = VerbsHostProtocolV2ResponseChunk {
                    stream_id: rank as usize,
                    header: response.header,
                    row_indices: Some(vec![row]),
                    partial_output_payload: VerbsHostProtocolV2ResponsePayload::from_owned(
                        bytes.clone(),
                    ),
                    wire_bytes,
                };
                chunk.header.request_id += 1;
                assert!(receiver
                    .push_rdma(&chunk, |_, _, _| panic!("stale data reached sink"))
                    .is_err());
                chunk.header.request_id -= 1;
                chunk.row_indices = Some(vec![1 - row]);
                assert!(receiver
                    .push_rdma(&chunk, |_, _, _| panic!("reordered data reached sink"))
                    .is_err());
                chunk.row_indices = Some(vec![row]);
                receiver.push_rdma(&chunk, |r, start, payload| {
                    assert_eq!((r, start), (rank as usize, row));
                    assert_eq!(payload, bytes);
                    Ok(())
                })?;
                assert!(receiver
                    .push_rdma(&chunk, |_, _, _| panic!("duplicate data reached sink"))
                    .is_err());
            }
        }
        assert!(receiver.complete());
        Ok(())
    }
}
