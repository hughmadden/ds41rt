//! Persistent coordinator QPs progressed by the inference owner. The existing
//! response assembler/recycling code runs locally; no QP worker is spawned.
use super::*;

pub(crate) struct LocalTp4Client {
    peers: [SocketAddr; 4],
    config: TcpTransportConfig,
    sessions: [Option<VerbsHostProtocolV2PersistentClientSession>; 4],
    pending: [VecDeque<VerbsHostProtocolV2PendingChunkRoundtrip>; 4],
    chunks: Option<tokio::sync::mpsc::UnboundedReceiver<VerbsHostProtocolV2ResponseChunk>>,
    done: Vec<tokio::sync::oneshot::Receiver<Result<VerbsHostProtocolV2ResponseStreamStats>>>,
    deadline: Option<Instant>,
}
impl LocalTp4Client {
    pub(crate) fn new(peers: [SocketAddr; 4], config: TcpTransportConfig) -> Self {
        Self {
            peers,
            config,
            sessions: std::array::from_fn(|_| None),
            pending: std::array::from_fn(|_| VecDeque::with_capacity(1)),
            chunks: None,
            done: Vec::with_capacity(4),
            deadline: None,
        }
    }
    pub(crate) fn reset(&mut self) {
        // Drop queued payloads before sessions unregister their receive rings.
        self.chunks = None;
        self.done.clear();
        for pending in &mut self.pending {
            pending.clear();
        }
        self.sessions = std::array::from_fn(|_| None);
        self.deadline = None;
    }
    pub(crate) fn dispatch(&mut self, request: &ExpertProtocolV2Request) -> Result<()> {
        anyhow::ensure!(self.deadline.is_none(), "local TP4 request already pending");
        let result = self.post(request);
        if result.is_err() {
            self.reset();
        }
        result
    }
    fn post(&mut self, request: &ExpertProtocolV2Request) -> Result<()> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.chunks = Some(rx);
        for rank in 0..4 {
            if self.sessions[rank]
                .as_ref()
                .map(|s| s.fits(request))
                .transpose()?
                == Some(false)
            {
                self.sessions[rank] = None;
            }
            if self.sessions[rank].is_none() {
                self.sessions[rank] = Some(VerbsHostProtocolV2PersistentClientSession::connect_local(
                    self.peers[rank],
                    &self.config,
                    request,
                )?);
            }
            let timing = self.sessions[rank]
                .as_mut()
                .unwrap()
                .post_chunk_request(request, &self.config)?;
            let (response_tx, response_rx) = tokio::sync::oneshot::channel();
            self.pending[rank].push_back(VerbsHostProtocolV2PendingChunkRoundtrip::new(
                VerbsHostProtocolV2QueuedChunkCommand {
                    request: request.clone(),
                    stream_id: rank,
                    chunk_tx: tx.clone(),
                    response_tx,
                },
                timing,
            ));
            self.done.push(response_rx);
        }
        // The same-thread channels adapt the existing assembler; they never
        // wake or hand off work to another thread. Each sink takes ownership of its chunk; retained pinned frames
        // return to the session pool only after the owner releases them.
        self.deadline = Some(
            Instant::now()
                .checked_add(self.config.timeout)
                .context("local TP4 deadline overflow")?,
        );
        Ok(())
    }
    pub(crate) fn poll<F>(&mut self, mut sink: F) -> Result<bool>
    where
        F: FnMut(VerbsHostProtocolV2ResponseChunk) -> Result<()>,
    {
        let deadline = self.deadline.context("local TP4 has no pending request")?;
        anyhow::ensure!(
            Instant::now() < deadline,
            "local TP4 response deadline expired"
        );
        for rank in 0..4 {
            if !self.pending[rank].is_empty() {
                self.sessions[rank]
                    .as_mut()
                    .context("local TP4 session missing")?
                    .try_progress_chunk_requests(&mut self.pending[rank], &self.config)?;
            }
            let chunks = self
                .chunks
                .as_mut()
                .context("local TP4 response queue missing")?;
            while let Ok(chunk) = chunks.try_recv() {
                sink(chunk)?;
            }
        }
        if self.pending.iter().any(|p| !p.is_empty()) {
            return Ok(false);
        }
        // Final validation and completion publication happen synchronously in
        // accept_chunk_response_frame before it removes a pending request.
        for done in &mut self.done {
            done.try_recv().context("local TP4 completion missing")??;
        }
        self.done.clear();
        self.chunks = None;
        self.deadline = None;
        Ok(true)
    }
}
