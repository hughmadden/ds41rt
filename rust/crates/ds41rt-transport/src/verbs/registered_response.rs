//! Final response slots stay unposted until their GPU consumers release them.
use super::*;

pub(super) struct RegisteredResponseFrame {
    // Keep the MR, allocation and QP alive even if the session is reset.
    _endpoint: Arc<NativeRdmaEndpoint>,
    buffer: Ds41rtHostBuffer,
    slot: usize,
    recycle: mpsc::Sender<usize>,
}
// No endpoint operation is exposed through this owner. A completed, unposted
// slot is immutable after publication; drop only returns its index. Last-owner
// endpoint destruction cannot race session operations, and pinned memory is
// portable across threads. The consumer must drain asynchronous reads first.
unsafe impl Send for RegisteredResponseFrame {}
unsafe impl Sync for RegisteredResponseFrame {}
impl RegisteredResponseFrame {
    pub(super) fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.buffer.ptr.cast(), self.buffer.bytes) }
    }
    pub(super) fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.buffer.ptr.cast(), self.buffer.bytes) }
    }
    pub(super) fn payload_host_buffer(&self, start: usize, end: usize) -> Ds41rtHostBuffer {
        debug_assert!(start <= end && end <= self.buffer.bytes);
        Ds41rtHostBuffer {
            ptr: unsafe { self.buffer.ptr.cast::<u8>().add(start).cast() },
            bytes: end - start,
            flags: self.buffer.flags,
        }
    }
}
impl Drop for RegisteredResponseFrame {
    fn drop(&mut self) {
        // A closed channel means the session was reset. The Arc still owns the
        // registration through this drop; no repost occurs on an abandoned QP.
        let _ = self.recycle.send(self.slot);
    }
}

pub(super) struct RegisteredResponseRing {
    view: Ds41rtRdmaRcEndpointBufferView,
    held: Vec<bool>,
    tx: mpsc::Sender<usize>,
    rx: mpsc::Receiver<usize>,
}
impl RegisteredResponseRing {
    pub(super) fn new(endpoint: &NativeRdmaEndpoint, ring: VerbsHostRdmaRing) -> Result<Self> {
        let view = endpoint.recv_buffer_view()?;
        validate_mapped_endpoint_buffer_view(view, ring, "retained response")?;
        anyhow::ensure!(
            view.host_flags & DS41RT_HOST_BUFFER_FLAG_PINNED != 0,
            "retained response ring must be CUDA pinned"
        );
        let (tx, rx) = mpsc::channel();
        Ok(Self {
            view,
            held: vec![false; ring.depth],
            tx,
            rx,
        })
    }

    pub(super) fn reclaim_before_request(
        &mut self,
        endpoint: &NativeRdmaEndpoint,
        ring: VerbsHostRdmaRing,
    ) -> Result<()> {
        for slot in self.rx.try_iter() {
            anyhow::ensure!(
                slot < self.held.len() && self.held[slot],
                "invalid retained receive slot recycle"
            );
            endpoint.post_recv_at(
                ring.slot_offset(slot),
                ring.slot_capacity_bytes,
                VERBS_HOST_RECV_WR_ID + slot as u64,
            )?;
            self.held[slot] = false;
        }
        // Only the final frame of this single-pending-request local session is
        // retained. Repost it before the next request to preserve FIFO WR order.
        // Reject retaining an old payload across another dispatch, rather than
        // silently consuming out-of-order slots or deadlocking the receive ring.
        anyhow::ensure!(
            !self.held.iter().any(|&held| held),
            "local TP4 response slot still owned by a consumer"
        );
        Ok(())
    }

    pub(super) fn try_retain_final(
        &mut self,
        endpoint: &Arc<NativeRdmaEndpoint>,
        ring: VerbsHostRdmaRing,
        sequence: usize,
        bytes: usize,
    ) -> Result<Option<RegisteredResponseFrame>> {
        let slot = mapped_ring_slot(self.view, ring, sequence as u64)?;
        anyhow::ensure!(
            bytes <= slot.capacity_bytes && !self.held[slot.slot_index],
            "invalid retained receive slot extent or ownership"
        );
        let data = unsafe { std::slice::from_raw_parts(slot.host_ptr.cast_const(), bytes) };
        let view = ExpertProtocolV2ResponseView::parse(data)?;
        // Earlier streamed chunks must be copied and reposted immediately:
        // consumers may retain all uploads until the final reduction drains.
        if view.more_chunks() {
            return Ok(None);
        }
        self.held[slot.slot_index] = true;
        Ok(Some(RegisteredResponseFrame {
            _endpoint: Arc::clone(endpoint),
            buffer: Ds41rtHostBuffer {
                ptr: slot.host_ptr.cast(),
                bytes,
                flags: self.view.host_flags,
            },
            slot: slot.slot_index,
            recycle: self.tx.clone(),
        }))
    }
}
