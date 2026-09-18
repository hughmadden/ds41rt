use super::*;
use std::{cell::RefCell, collections::BTreeSet};

#[derive(Default)]
struct DeviceState {
    ready: BTreeSet<u64>,
    failed: BTreeSet<u64>,
    started: BTreeMap<u64, (Transaction, Vec<SourceWrites>)>,
    cancelled: BTreeSet<u64>,
    drained: BTreeSet<u64>,
    fail_start: bool,
    storage: [Vec<u64>; 4],
}
#[derive(Clone)]
struct FakeDevice(Rc<RefCell<DeviceState>>);
impl FakeDevice {
    fn new(pages: [usize; 4]) -> Self {
        Self(Rc::new(RefCell::new(DeviceState { storage: pages.map(|n| vec![0; n * 256]),
            ..Default::default() })))
    }
    fn ready(&self, tx: Transaction) { self.0.borrow_mut().ready.insert(tx.id); }
    fn fail(&self, tx: Transaction) { self.0.borrow_mut().failed.insert(tx.id); }
    fn row(&self, source: usize, physical: u64) -> u64 {
        self.0.borrow().storage[source][physical as usize]
    }
}
// SAFETY: writes occur synchronously only when poll reports ready. Cancellation
// drains by discarding queued work, and retained Rc storage survives read guards.
unsafe impl CacheDevice for FakeDevice {
    type Storage = Rc<RefCell<DeviceState>>;
    fn source_page_capacity(&self) -> [usize; 4] {
        std::array::from_fn(|s| self.0.borrow().storage[s].len() / 256)
    }
    fn retain_storage(&self) -> Self::Storage { self.0.clone() }
    fn start(&mut self, tx: Transaction, writes: Vec<SourceWrites>) -> Result<()> {
        let mut state = self.0.borrow_mut();
        assert!(state.started.insert(tx.id, (tx, writes)).is_none());
        if state.fail_start { anyhow::bail!("injected start error after enqueue"); }
        Ok(())
    }
    fn poll(&mut self, tx: Transaction) -> Completion {
        let mut state = self.0.borrow_mut();
        if state.failed.remove(&tx.id) {
            state.started.remove(&tx.id);
            state.drained.insert(tx.id);
            return Completion::Failed("injected failed writes".into());
        }
        if !state.ready.remove(&tx.id) { return Completion::Pending; }
        if let Some((_, writes)) = state.started.remove(&tx.id) {
            if !state.cancelled.contains(&tx.id) {
                for write in writes {
                    let storage = &mut state.storage[write.source];
                    for (source, destination) in write.tail_copies {
                        let rows = storage[source as usize * 256..(source as usize + 1) * 256].to_vec();
                        storage[destination as usize * 256..(destination as usize + 1) * 256].copy_from_slice(&rows);
                    }
                    for (offset, physical) in write.destinations.into_iter().enumerate() {
                        storage[physical as usize] = tx.request.slot() as u64 * 100000 +
                            (write.first_row + offset + 1) as u64;
                    }
                }
            }
        }
        state.drained.insert(tx.id);
        Completion::Complete
    }
    fn cancel(&mut self, tx: Transaction) { self.0.borrow_mut().cancelled.insert(tx.id); }
    fn drain(&mut self, tx: Transaction) {
        let mut state = self.0.borrow_mut();
        state.started.remove(&tx.id);
        state.drained.insert(tx.id);
    }
}
fn fixture(pages: [usize; 4]) -> (CacheCommands<FakeDevice>, FakeDevice) {
    let device = FakeDevice::new(pages);
    (CacheCommands::new(17, 3, pages, 1024, 16, device.clone()).unwrap(), device)
}
fn call(cache: &mut CacheCommands<FakeDevice>, command: Command) -> Outcome {
    let snapshot = cache.snapshot();
    cache.execute(Envelope { owner: snapshot.owner, epoch: snapshot.epoch,
        command_id: snapshot.epoch + 1, command }).unwrap().result
}
fn admit(cache: &mut CacheCommands<FakeDevice>, slot: usize) -> RequestHandle {
    match call(cache, Command::Admit { slot, request_id: 100 + slot as u64 }) {
        Outcome::Admitted { request } => request, _ => panic!(),
    }
}
fn begin(cache: &mut CacheCommands<FakeDevice>, request: RequestHandle, lane: usize, rows: u32) -> Transaction {
    match call(cache, Command::Begin { request, lane, rows }) {
        Outcome::Begun { transaction } => transaction, _ => panic!(),
    }
}
fn rejects(cache: &mut CacheCommands<FakeDevice>, command: Command) {
    let before = cache.snapshot();
    assert!(cache.execute(Envelope { owner: before.owner, epoch: before.epoch,
        command_id: before.epoch + 1, command }).is_err());
    assert_eq!(cache.snapshot(), before);
}
fn commit(cache: &mut CacheCommands<FakeDevice>, device: &FakeDevice, tx: Transaction, accepted: u32) {
    assert_eq!(call(cache, Command::ReservePublish { transaction: tx, accepted }), Outcome::Reserved);
    device.ready(tx);
    assert!(matches!(call(cache, Command::Publish { transaction: tx }), Outcome::Published { .. }));
}

#[test]
fn schema_and_exact_retry_are_idempotent_and_epoch_checked() {
    let (mut cache, _) = fixture([4; 4]);
    let raw = r#"{"owner":17,"epoch":0,"command_id":1,"command":{"op":"admit","slot":0,"request_id":100}}"#;
    let envelope: Envelope = serde_json::from_str(raw).unwrap();
    let ack = cache.execute(envelope.clone()).unwrap();
    assert_eq!(cache.execute(envelope.clone()).unwrap(), ack);
    let changed = Envelope { command: Command::Admit { slot: 1, request_id: 101 }, ..envelope.clone() };
    assert!(cache.execute(changed).is_err());
    assert!(cache.execute(Envelope { owner: 18, epoch: 1, command_id: 2, ..envelope.clone() }).is_err());
    assert!(cache.execute(Envelope { command_id: 2, ..envelope }).is_err());
    assert_eq!(cache.snapshot(), ack.snapshot);
    let encoded = serde_json::to_string(&ack).unwrap();
    assert_eq!(serde_json::from_str::<Acknowledgement>(&encoded).unwrap(), ack);
    assert!(serde_json::from_str::<Envelope>(raw.replace("\"slot\":0", "\"slot\":0,\"unknown\":1").as_str()).is_err());
}

#[test]
fn native_lease_generation_foreign_and_reused_slot_reject() {
    let (mut cache, _) = fixture([4; 4]);
    let old = admit(&mut cache, 0);
    let foreign = RequestHandle::new(18, old.slot(), old.generation());
    rejects(&mut cache, Command::Release { request: foreign });
    assert_eq!(call(&mut cache, Command::Release { request: old }), Outcome::Released);
    let new = admit(&mut cache, 0);
    assert_eq!(new.generation(), old.generation() + 1);
    rejects(&mut cache, Command::Begin { request: old, lane: 0, rows: 1 });
    assert!(cache.read(old).is_err());
}

#[test]
fn two_lanes_publish_in_completion_order_without_peer_alias_or_partial_ack() {
    let (mut cache, device) = fixture([8; 4]);
    let a = admit(&mut cache, 0); let b = admit(&mut cache, 1);
    let first = begin(&mut cache, a, 0, 257); let second = begin(&mut cache, b, 1, 300);
    assert_eq!(call(&mut cache, Command::ReservePublish { transaction: first, accepted: 257 }), Outcome::Reserved);
    assert_eq!(call(&mut cache, Command::ReservePublish { transaction: second, accepted: 300 }), Outcome::Reserved);
    assert_eq!(cache.snapshot().source_pages_free, [6, 6, 6, 4]);
    assert_eq!(call(&mut cache, Command::Publish { transaction: first }), Outcome::Pending);
    assert_eq!(cache.snapshot().requests[0].committed_end, 0);
    device.ready(second);
    assert_eq!(call(&mut cache, Command::Publish { transaction: second }), Outcome::Published { committed_end: 300 });
    assert_eq!(cache.snapshot().transactions, 1);
    device.ready(first);
    assert_eq!(call(&mut cache, Command::Publish { transaction: first }), Outcome::Published { committed_end: 257 });
    for (slot, rows) in [(0, 257), (1, 300)] {
        let prefix = cache.sources[3].retain_prefix(slot, rows).unwrap();
        for row in 0..rows {
            let physical = prefix.pages()[row / 256] as u64 * 256 + (row % 256) as u64;
            assert_eq!(device.row(3, physical), slot as u64 * 100000 + row as u64 + 1);
        }
    }
    rejects(&mut cache, Command::Publish { transaction: second });
    assert_eq!(cache.snapshot().transactions, 0);
}

#[test]
fn fourth_source_exhaustion_rolls_back_all_prior_native_reservations() {
    let (mut cache, _) = fixture([2, 2, 2, 1]);
    let request = admit(&mut cache, 0);
    let tx = begin(&mut cache, request, 0, 257);
    rejects(&mut cache, Command::ReservePublish { transaction: tx, accepted: 257 });
    assert_eq!(cache.snapshot().source_pages_free, [2, 2, 2, 1]);
    assert_eq!(call(&mut cache, Command::Abort { transaction: tx }), Outcome::Aborted);
}

#[test]
fn source_prefix_fork_uses_retained_native_tail_copy_on_write() {
    let (mut cache, device) = fixture([8; 4]);
    let original = admit(&mut cache, 0);
    let tx = begin(&mut cache, original, 0, 255);
    commit(&mut cache, &device, tx, 255);
    let prefix = match call(&mut cache, Command::RetainSourcePrefix { request: original }) {
        Outcome::SourcePrefix { prefix, .. } => prefix, _ => panic!(),
    };
    let fork = match call(&mut cache, Command::RestoreSourcePrefix { prefix, slot: 1, request_id: 101 }) {
        Outcome::Admitted { request } => request, _ => panic!(),
    };
    assert_eq!(cache.snapshot().source_pages_free, [7; 4]); // all three owners share the same pages
    let tx = begin(&mut cache, fork, 1, 2);
    assert_eq!(call(&mut cache, Command::ReservePublish { transaction: tx, accepted: 2 }), Outcome::Reserved);
    let queued = device.0.borrow();
    assert!(queued.started[&tx.id].1.iter().all(|w| w.tail_copies.len() == 1));
    drop(queued);
    device.ready(tx);
    assert_eq!(call(&mut cache, Command::Publish { transaction: tx }), Outcome::Published { committed_end: 257 });
    let fork_rows = cache.sources[3].retain_prefix(1, 257).unwrap();
    let original_rows = &cache.prefixes[&prefix.id].sources[3];
    assert_ne!(fork_rows.pages()[0], original_rows.pages()[0]);
    for row in 0..255 {
        assert_eq!(device.row(3, fork_rows.pages()[0] as u64 * 256 + row), row + 1);
        assert_eq!(device.row(3, original_rows.pages()[0] as u64 * 256 + row), row + 1);
    }
    assert_eq!(device.row(3, fork_rows.pages()[0] as u64 * 256 + 255), 100256);
    drop(fork_rows);
    call(&mut cache, Command::ResetSourcePrefixes);
    assert_eq!(cache.snapshot().source_prefixes, 0);
    call(&mut cache, Command::Release { request: original });
    call(&mut cache, Command::Release { request: fork });
    assert_eq!(cache.snapshot().source_pages_free, [8; 4]);
    rejects(&mut cache, Command::RestoreSourcePrefix { prefix, slot: 0, request_id: 100 });
}

#[test]
fn accepted_counts_publish_only_accepted_source_extents() {
    let (mut cache, device) = fixture([4; 4]);
    let request = admit(&mut cache, 0);
    let tx = begin(&mut cache, request, 0, 257);
    rejects(&mut cache, Command::ReservePublish { transaction: tx, accepted: 258 });
    commit(&mut cache, &device, tx, 127);
    assert_eq!(cache.snapshot().requests[0].committed_end, 127);
    assert_eq!(cache.sources[0].committed_rows(0).unwrap(), 63);
    assert_eq!(cache.sources[3].committed_rows(0).unwrap(), 127);
    let zero = begin(&mut cache, request, 0, 4);
    commit(&mut cache, &device, zero, 0);
    assert_eq!(cache.snapshot().requests[0].committed_end, 127);
}

#[test]
fn reader_blocks_publication_and_release_reuse_until_drop() {
    let (mut cache, device) = fixture([4; 4]);
    let request = admit(&mut cache, 0);
    let tx = begin(&mut cache, request, 0, 128);
    commit(&mut cache, &device, tx, 128);
    let reader = cache.read(request).unwrap();
    let next = begin(&mut cache, request, 0, 1);
    rejects(&mut cache, Command::ReservePublish { transaction: next, accepted: 1 });
    assert_eq!(call(&mut cache, Command::Release { request }), Outcome::Closing);
    assert!(cache.read(request).is_err());
    rejects(&mut cache, Command::Admit { slot: 0, request_id: 100 });
    assert_eq!(cache.snapshot().source_pages_free, [3; 4]);
    drop(reader);
    assert_eq!(call(&mut cache, Command::Drain { request }), Outcome::Released);
    assert_eq!(cache.snapshot().source_pages_free, [4; 4]);
}

#[test]
fn cancel_retains_reserved_pages_until_backend_drain_then_lane_reuses() {
    let (mut cache, device) = fixture([4; 4]);
    let request = admit(&mut cache, 0);
    let tx = begin(&mut cache, request, 0, 257);
    call(&mut cache, Command::ReservePublish { transaction: tx, accepted: 257 });
    assert_eq!(call(&mut cache, Command::Abort { transaction: tx }), Outcome::Pending);
    assert_eq!(cache.snapshot().source_pages_free, [3, 3, 3, 2]);
    rejects(&mut cache, Command::Publish { transaction: tx });
    device.ready(tx);
    assert_eq!(call(&mut cache, Command::Abort { transaction: tx }), Outcome::Aborted);
    assert_eq!(cache.snapshot().source_pages_free, [4; 4]);
    let next = begin(&mut cache, request, 0, 1);
    assert_ne!(tx, next);
    rejects(&mut cache, Command::Abort { transaction: tx });
    commit(&mut cache, &device, next, 1);
}

#[test]
fn failed_writes_and_partial_start_never_publish_or_free_early() {
    let (mut cache, device) = fixture([4; 4]);
    let request = admit(&mut cache, 0);
    let tx = begin(&mut cache, request, 0, 256);
    call(&mut cache, Command::ReservePublish { transaction: tx, accepted: 256 });
    device.fail(tx);
    assert!(matches!(call(&mut cache, Command::Publish { transaction: tx }), Outcome::Failed { .. }));
    assert_eq!(cache.snapshot().requests[0].committed_end, 0);
    assert_eq!(cache.snapshot().source_pages_free, [4; 4]);
    let tx = begin(&mut cache, request, 0, 256);
    device.0.borrow_mut().fail_start = true;
    assert!(matches!(call(&mut cache, Command::ReservePublish { transaction: tx, accepted: 256 }), Outcome::Failed { .. }));
    assert_eq!(cache.snapshot().source_pages_free, [3; 4]);
    assert_eq!(call(&mut cache, Command::Release { request }), Outcome::Closing);
    device.ready(tx);
    assert_eq!(call(&mut cache, Command::Drain { request }), Outcome::Released);
    assert_eq!(cache.snapshot().source_pages_free, [4; 4]);
}

#[test]
fn close_drains_device_and_read_guard_retains_physical_owner() {
    let (mut cache, device) = fixture([4; 4]);
    let request = admit(&mut cache, 0);
    let reader = cache.read(request).unwrap();
    let tx = begin(&mut cache, request, 0, 1);
    drop(reader);
    call(&mut cache, Command::ReservePublish { transaction: tx, accepted: 1 });
    drop(cache);
    assert!(device.0.borrow().drained.contains(&tx.id));
    assert!(device.0.borrow().started.is_empty());
    let (mut cache, device) = fixture([4; 4]);
    let request = admit(&mut cache, 0);
    let reader = cache.read(request).unwrap();
    let weak = Rc::downgrade(&device.0);
    drop(cache); drop(device);
    assert!(weak.upgrade().is_some());
    drop(reader);
    assert!(weak.upgrade().is_none());
}

#[test]
fn same_request_alias_busy_and_geometry_reject_without_state_change() {
    let (mut cache, _) = fixture([4; 4]);
    let request = admit(&mut cache, 0);
    rejects(&mut cache, Command::Begin { request, lane: 0, rows: 0 });
    rejects(&mut cache, Command::Begin { request, lane: 2, rows: 1 });
    rejects(&mut cache, Command::Begin { request, lane: 0, rows: 1025 });
    let tx = begin(&mut cache, request, 0, 1);
    rejects(&mut cache, Command::Begin { request, lane: 1, rows: 1 });
    rejects(&mut cache, Command::RetainSourcePrefix { request });
    rejects(&mut cache, Command::Publish { transaction: tx });
    call(&mut cache, Command::Abort { transaction: tx });
}

#[test]
fn repeated_reuse_restores_one_native_physical_allocation_ledger() {
    let (mut cache, device) = fixture([4; 4]);
    let bytes = cache.snapshot().cache_bytes;
    for _ in 0..30 {
        let request = admit(&mut cache, 0);
        let tx = begin(&mut cache, request, 0, 513);
        commit(&mut cache, &device, tx, 513);
        call(&mut cache, Command::Release { request });
        assert_eq!(cache.snapshot().source_pages_free, [4; 4]);
        assert_eq!(cache.snapshot().cache_bytes, bytes);
        assert!(cache.snapshot().requests.is_empty());
        assert_eq!(cache.snapshot().transactions, 0);
    }
}

#[test]
fn retained_prefix_metadata_capacity_is_bounded_and_recovers_after_eviction() {
    let (mut cache, _) = fixture([4; 4]);
    let request = admit(&mut cache, 0);
    for _ in 0..16 { call(&mut cache, Command::RetainSourcePrefix { request }); }
    rejects(&mut cache, Command::RetainSourcePrefix { request });
    call(&mut cache, Command::ResetSourcePrefixes);
    assert!(matches!(call(&mut cache, Command::RetainSourcePrefix { request }),
        Outcome::SourcePrefix { .. }));
}

#[test]
fn physical_capacity_mismatch_is_rejected_before_metadata_owner_creation() {
    let device = FakeDevice::new([2; 4]);
    assert!(CacheCommands::new(17, 3, [4; 4], 1024, 16, device).is_err());
}
