//! TP4 dispatch through persistent RoCE QPs; TCP is used only for bootstrap.
use super::{V41BackboneRequest, V41Tp4ChunkReceiver};
use crate::{ExpertProtocolV2Request, TcpTransportConfig, VerbsHostProtocolV2PersistentClient};
use anyhow::{ensure, Result};
use std::net::SocketAddr;

pub struct V41Tp4Roce {
    clients: [VerbsHostProtocolV2PersistentClient; 4],
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
            clients: peers
                .into_iter()
                .map(|peer| VerbsHostProtocolV2PersistentClient::new(peer, config.clone()))
                .collect::<Result<Vec<_>>>()?
                .try_into()
                .map_err(|_| anyhow::anyhow!("four RoCE clients required"))?,
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
        for client in &mut self.clients {
            client.reset();
        }
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

    /// Enqueue all four requests on their QP owners. Network/GPU work overlaps
    /// the coordinator shared FFN; enqueue completion is not send completion.
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
        let (chunks_tx, chunks) = tokio::sync::mpsc::unbounded_channel();
        let mut pending = Vec::with_capacity(4);
        for (rank, client) in self.clients.iter().enumerate() {
            match client.enqueue_response_chunks(request.clone(), rank, chunks_tx.clone()) {
                Ok(done) => pending.push(done),
                Err(error) => {
                    self.reset_connections();
                    return Err(error);
                }
            }
        }
        drop(chunks_tx);
        Ok(V41Tp4RocePending {
            pending,
            chunks,
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
    pending:
        Vec<tokio::sync::oneshot::Receiver<Result<crate::VerbsHostProtocolV2ResponseStreamStats>>>,
    chunks: tokio::sync::mpsc::UnboundedReceiver<crate::VerbsHostProtocolV2ResponseChunk>,
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
        // One admitted wave and transport-side validation bound the channel to
        // four rank planes. No next wave can enqueue while this owner is borrowed.
        while let Some(chunk) = self.chunks.recv().await {
            self.receiver.push_rdma(&chunk, |rank, start, bytes| {
                ensure!(
                    rank == chunk.stream_id,
                    "native executor identity does not match its RoCE peer"
                );
                sink(rank, start, bytes)
            })?;
        }
        for done in std::mem::take(&mut self.pending) {
            done.await
                .map_err(|_| anyhow::anyhow!("RoCE owner stopped before completion"))??;
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
