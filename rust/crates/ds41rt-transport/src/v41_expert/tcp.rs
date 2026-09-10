//! Concurrent TP4 TCP dispatch with bounded per-rank response storage.
use super::{V41BackboneRequest, V41Tp4ChunkReceiver};
use crate::{ExpertProtocolV2Request, TcpProtocolV2PersistentClient, TcpTransportConfig};
use anyhow::{ensure, Result};
use std::{cell::RefCell, net::SocketAddr};

pub struct V41Tp4Tcp {
    clients: [TcpProtocolV2PersistentClient; 4],
    executors: [u64; 4],
    capacity: u32,
    max_frame_bytes: usize,
}
impl V41Tp4Tcp {
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
            "native TCP timeout must be positive"
        );
        ensure!(
            config.max_frame_bytes >= 128 + 122880 + 4
                && config.max_frame_bytes <= 64 * 1024 * 1024,
            "invalid native TCP frame budget"
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
            clients: peers.map(|peer| TcpProtocolV2PersistentClient::new(peer, config.clone())),
            executors,
            capacity,
            max_frame_bytes: config.max_frame_bytes,
        })
    }
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Send the same canonical request to all ranks and accept every route row.
    /// The synchronous sink must consume/copy its frame slice before returning.
    /// It runs on the polling thread, allowing CUDA owners to stay thread-local.
    /// On error or cancellation the caller must discard partial destination state;
    /// no partially delivered request is automatically replayed.
    pub async fn execute<F>(&mut self, request: &ExpertProtocolV2Request, mut sink: F) -> Result<()>
    where
        F: FnMut(usize, u32, &[u8]) -> Result<()>,
    {
        let frame = request.encode()?;
        ensure!(
            frame.len() <= self.max_frame_bytes,
            "native request exceeds TCP frame budget"
        );
        let native = V41BackboneRequest::parse(&frame, self.capacity)?;
        let mut receiver = V41Tp4ChunkReceiver::new(&native, self.executors, self.max_frame_bytes)?;
        // All futures are polled together on the caller's thread. Mutable assembly
        // access exists only inside a synchronous callback, never across an await.
        let state = RefCell::new((&mut receiver, &mut sink));
        let responses = self
            .clients
            .iter_mut()
            .enumerate()
            .map(|(expected_rank, client)| {
                let state = &state;
                client.roundtrip_chunks(request, native.rows() as usize, move |frame| {
                    let mut state = state.borrow_mut();
                    let (receiver, sink) = &mut *state;
                    receiver.push(frame, |rank, start, bytes| {
                        ensure!(
                            rank == expected_rank,
                            "native executor identity does not match its peer"
                        );
                        sink(rank, start, bytes)
                    })?;
                    Ok(())
                })
            });
        futures::future::try_join_all(responses).await?;
        ensure!(
            receiver.complete(),
            "native TP4 response coverage is incomplete"
        );
        Ok(())
    }
}
