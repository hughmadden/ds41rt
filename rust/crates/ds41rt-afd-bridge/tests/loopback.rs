use ds41rt_afd_bridge::*;
use ds41rt_transport::{
    v41_expert::{V41BackboneRequest, EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16},
    ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor,
    ExpertV2Dtype, ExpertV2SourceKind, EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN,
};
use std::{
    net::SocketAddr,
    ptr,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::{oneshot, Barrier},
};

const EXECUTORS: [u64; 4] = [11, 22, 33, 44];
const FRAME: usize = 2 * 1024 * 1024;
enum Action {
    Reply,
    Delay(Duration),
    Gate(Arc<Barrier>),
    Stall,
    WrongId,
    WrongRank,
    ReverseRows,
    DuplicateRow,
    Truncate,
    Disconnect,
}
type Behavior = Arc<dyn Fn(usize, u64) -> Action + Send + Sync>;
struct Fleet {
    peers: [SocketAddr; 4],
    connections: Arc<AtomicUsize>,
    requests: Arc<AtomicUsize>,
    stop: Option<oneshot::Sender<()>>,
    worker: Option<thread::JoinHandle<()>>,
}
impl Fleet {
    fn new(behavior: impl Fn(usize, u64) -> Action + Send + Sync + 'static) -> Self {
        let sockets = (0..4)
            .map(|_| {
                let socket = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
                socket.set_nonblocking(true).unwrap();
                socket
            })
            .collect::<Vec<_>>();
        let peers = std::array::from_fn(|i| sockets[i].local_addr().unwrap());
        let connections = Arc::new(AtomicUsize::new(0));
        let requests = Arc::new(AtomicUsize::new(0));
        let worker_connections = Arc::clone(&connections);
        let worker_requests = Arc::clone(&requests);
        let behavior: Behavior = Arc::new(behavior);
        let (stop, stopped) = oneshot::channel();
        let worker = thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(async move {
                    for (rank, socket) in sockets.into_iter().enumerate() {
                        let listener = TcpListener::from_std(socket).unwrap();
                        let behavior = Arc::clone(&behavior);
                        let connections = Arc::clone(&worker_connections);
                        let requests = Arc::clone(&worker_requests);
                        tokio::spawn(async move {
                            while let Ok((stream, _)) = listener.accept().await {
                                connections.fetch_add(1, Ordering::SeqCst);
                                tokio::spawn(serve(
                                    stream,
                                    rank,
                                    Arc::clone(&behavior),
                                    Arc::clone(&requests),
                                ));
                            }
                        });
                    }
                    let _ = stopped.await;
                });
        });
        Self {
            peers,
            connections,
            requests,
            stop: Some(stop),
            worker: Some(worker),
        }
    }
}
impl Drop for Fleet {
    fn drop(&mut self) {
        let _ = self.stop.take().unwrap().send(());
        let _ = self.worker.take().unwrap().join();
    }
}
async fn serve(mut stream: TcpStream, rank: usize, behavior: Behavior, requests: Arc<AtomicUsize>) {
    stream.set_nodelay(true).unwrap();
    loop {
        let mut frame = vec![0; EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN];
        if stream.read_exact(&mut frame).await.is_err() {
            return;
        }
        let len = ExpertProtocolV2Request::wire_bytes_from_header(&frame).unwrap();
        assert!(len <= FRAME);
        frame.resize(len, 0);
        if stream
            .read_exact(&mut frame[EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN..])
            .await
            .is_err()
        {
            return;
        }
        let request = ExpertProtocolV2Request::decode(&frame).unwrap();
        let native = V41BackboneRequest::parse(&frame, 128).unwrap();
        requests.fetch_add(1, Ordering::SeqCst);
        let action = behavior(rank, request.header.request_id);
        match &action {
            Action::Delay(delay) => tokio::time::sleep(*delay).await,
            Action::Gate(gate) => {
                gate.wait().await;
            }
            Action::Stall => {
                std::future::pending::<()>().await;
            }
            Action::Disconnect => return,
            _ => {}
        }
        let rows = native.rows();
        let row_order: Vec<_> = if matches!(action, Action::ReverseRows) {
            (0..rows).rev().collect()
        } else {
            (0..rows).collect()
        };
        for row in row_order {
            let bytes = tagged_row(rank, request.header.request_id, row);
            let mut indices = [0u32];
            let mut response = native
                .response_chunk(EXECUTORS[rank], row, &bytes, &mut indices, FRAME)
                .unwrap()
                .to_owned()
                .unwrap();
            if matches!(action, Action::WrongId) {
                response.header.request_id += 999;
            }
            if matches!(action, Action::WrongRank) {
                response.header.executor_id = EXECUTORS[(rank + 1) % 4];
            }
            let encoded = response.encode().unwrap();
            if matches!(action, Action::Truncate) {
                let _ = stream.write_all(&encoded[..encoded.len() / 2]).await;
                return;
            }
            if stream.write_all(&encoded).await.is_err() {
                return;
            }
            if matches!(action, Action::DuplicateRow) && row == 0 {
                if stream.write_all(&encoded).await.is_err() {
                    return;
                }
            }
        }
    }
}
fn tagged_row(rank: usize, id: u64, row: u32) -> Vec<u8> {
    let word = (0x3e00u16 + (rank as u16) * 0x80 + row as u16).to_le_bytes();
    let mut values = word.repeat(5120);
    values[0..8].copy_from_slice(&id.to_le_bytes());
    values
}
fn request(id: u64, rows: u32) -> Vec<u8> {
    let descriptors = (0..rows)
        .map(|row| ExpertProtocolV2RowDescriptor {
            row_id: row as u64,
            source_kind: ExpertV2SourceKind::Decode,
            source_request_id: id + row as u64,
            token_position: 128 + row as u64,
            route_offset: row * 6,
            route_count: 6,
        })
        .collect();
    let routes = (0..rows)
        .flat_map(|row| {
            (0..6).map(move |expert| ExpertProtocolV2RouteEntry {
                row_index: row,
                expert_id: expert,
                gate_weight: (expert + 1) as f32 / 21.0,
            })
        })
        .collect();
    let mut request = ExpertProtocolV2Request::new(
        id,
        79,
        12,
        5120,
        ExpertV2Dtype::Bf16,
        descriptors,
        routes,
        vec![0; rows as usize * 10240],
    )
    .unwrap();
    request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    request.encode().unwrap()
}
struct Client(*mut afd_bridge_handle);
impl Client {
    fn new(fleet: &Fleet, timeout_ms: u64) -> Self {
        Self::from_config(serde_json::json!({"transport":"tcp", "peers":fleet.peers,
            "executors":EXECUTORS, "capacity_rows":128, "lanes":2,
            "timeout_ms":timeout_ms, "max_frame_bytes":FRAME}))
    }
    fn from_config(config: serde_json::Value) -> Self {
        let bytes = serde_json::to_vec(&config).unwrap();
        let mut handle = ptr::null_mut();
        assert_eq!(
            unsafe { afd_bridge_create(bytes.as_ptr(), bytes.len(), &mut handle) },
            OK
        );
        assert!(!handle.is_null());
        Self(handle)
    }
    fn submit(&self, lane: u32, id: u64, rows: u32) -> u64 {
        let frame = request(id, rows);
        let mut ticket = 0;
        assert_eq!(
            unsafe { afd_bridge_submit(self.0, lane, frame.as_ptr(), frame.len(), &mut ticket) },
            OK
        );
        ticket
    }
    fn wait(&self, lane: u32, ticket: u64) -> u32 {
        let deadline = Instant::now() + Duration::from_secs(4);
        loop {
            let mut state = 99;
            assert_eq!(
                unsafe { afd_bridge_poll(self.0, lane, ticket, &mut state) },
                OK
            );
            if state != 1 {
                return state;
            }
            assert!(Instant::now() < deadline, "bridge did not finish");
            thread::sleep(Duration::from_millis(1));
        }
    }
    fn collect(&self, lane: u32, ticket: u64, id: u64, rows: u32) {
        let mut size = 0;
        assert_eq!(
            unsafe { afd_bridge_collect(self.0, lane, ticket, ptr::null_mut(), 0, &mut size) },
            BUFFER_TOO_SMALL
        );
        assert_eq!(size, 4 * rows as usize * 10240);
        let mut bytes = vec![0; size];
        assert_eq!(
            unsafe {
                afd_bridge_collect(
                    self.0,
                    lane,
                    ticket,
                    bytes.as_mut_ptr(),
                    size - 1,
                    &mut size,
                )
            },
            BUFFER_TOO_SMALL
        );
        assert_eq!(
            unsafe {
                afd_bridge_collect(self.0, lane, ticket, bytes.as_mut_ptr(), size, &mut size)
            },
            OK
        );
        for rank in 0..4 {
            for row in 0..rows {
                let start = (rank * rows as usize + row as usize) * 10240;
                assert_eq!(&bytes[start..start + 10240], tagged_row(rank, id, row));
            }
        }
        assert_eq!(
            unsafe {
                afd_bridge_collect(self.0, lane, ticket, bytes.as_mut_ptr(), size, &mut size)
            },
            NOT_READY
        );
    }
}
impl Drop for Client {
    fn drop(&mut self) {
        assert_eq!(unsafe { afd_bridge_close(self.0) }, OK);
    }
}

#[test]
fn persistent_rank_reordering_owned_input_and_generation() {
    let fleet = Fleet::new(|rank, _| Action::Delay(Duration::from_millis((3 - rank) as u64)));
    let client = Client::new(&fleet, 1000);
    let mut previous = 0;
    for id in 1..=4 {
        let ticket = client.submit(0, id, 3);
        assert!(ticket > previous);
        if previous != 0 {
            let mut state = 0;
            assert_eq!(
                unsafe { afd_bridge_poll(client.0, 0, previous, &mut state) },
                STALE
            );
        }
        assert_eq!(client.wait(0, ticket), 2);
        client.collect(0, ticket, id, 3);
        previous = ticket;
    }
    assert_eq!(fleet.connections.load(Ordering::SeqCst), 4);
    assert_eq!(fleet.requests.load(Ordering::SeqCst), 16);
}

#[test]
fn two_lanes_dispatch_before_any_rank_replies_and_bound_admission() {
    let gate = Arc::new(Barrier::new(8));
    let fleet = Fleet::new(move |_, _| Action::Gate(Arc::clone(&gate)));
    let client = Client::new(&fleet, 2000);
    let a = client.submit(0, 100, 2);
    let b = client.submit(1, 200, 2);
    let frame = request(999, 2);
    let mut ignored = 0;
    assert_eq!(
        unsafe { afd_bridge_submit(client.0, 0, frame.as_ptr(), frame.len(), &mut ignored) },
        BUSY
    );
    assert_eq!(client.wait(0, a), 2);
    assert_eq!(client.wait(1, b), 2);
    // A ready but unconsumed plane is also a bounded lease.
    assert_eq!(
        unsafe { afd_bridge_submit(client.0, 1, frame.as_ptr(), frame.len(), &mut ignored) },
        BUSY
    );
    client.collect(0, a, 100, 2);
    client.collect(1, b, 200, 2);
    assert_eq!(fleet.connections.load(Ordering::SeqCst), 8);
}

#[test]
fn stalled_lane_does_not_block_peer_and_cancel_resets_before_reuse() {
    let fleet = Fleet::new(|_, id| {
        if id == 1 {
            Action::Stall
        } else {
            Action::Reply
        }
    });
    let client = Client::new(&fleet, 1000);
    let stalled = client.submit(0, 1, 1);
    let peer = client.submit(1, 2, 1);
    assert_eq!(client.wait(1, peer), 2);
    client.collect(1, peer, 2, 1);
    assert_eq!(unsafe { afd_bridge_cancel(client.0, 0, stalled) }, OK);
    assert_eq!(client.wait(0, stalled), 4);
    let next = client.submit(0, 3, 1);
    assert_eq!(client.wait(0, next), 2);
    client.collect(0, next, 3, 1);
    let mut state = 0;
    assert_eq!(
        unsafe { afd_bridge_poll(client.0, 0, stalled, &mut state) },
        STALE
    );
}

#[test]
fn timeout_and_malformed_responses_fail_then_reconnect() {
    for failure in 0..7 {
        let fleet = Fleet::new(move |rank, id| {
            if rank != 2 || id != 1 {
                return Action::Reply;
            }
            match failure {
                0 => Action::Stall,
                1 => Action::WrongId,
                2 => Action::WrongRank,
                3 => Action::ReverseRows,
                4 => Action::DuplicateRow,
                5 => Action::Truncate,
                _ => Action::Disconnect,
            }
        });
        let client = Client::new(&fleet, 80);
        let failed = client.submit(0, 1, 2);
        assert_eq!(client.wait(0, failed), 3, "fault case {failure}");
        let mut size = 0;
        assert_eq!(
            unsafe { afd_bridge_collect(client.0, 0, failed, ptr::null_mut(), 0, &mut size) },
            FAILED
        );
        assert_eq!(
            unsafe { afd_bridge_error(client.0, 0, failed, ptr::null_mut(), 0, &mut size) },
            BUFFER_TOO_SMALL
        );
        assert!(size > 0);
        let mut error = vec![0; size];
        assert_eq!(
            unsafe { afd_bridge_error(client.0, 0, failed, error.as_mut_ptr(), size, &mut size) },
            OK
        );
        assert!(!String::from_utf8(error).unwrap().is_empty());
        let next = client.submit(0, 2, 2);
        assert_eq!(client.wait(0, next), 2);
        client.collect(0, next, 2, 2);
        assert!(fleet.connections.load(Ordering::SeqCst) >= 8);
    }
}

#[test]
fn close_cancels_outstanding_lanes_and_ready_cancel_discards_result() {
    let fleet = Fleet::new(|_, id| {
        if id == 1 {
            Action::Stall
        } else {
            Action::Reply
        }
    });
    let client = Client::new(&fleet, 2000);
    let ready = client.submit(0, 2, 1);
    assert_eq!(client.wait(0, ready), 2);
    assert_eq!(unsafe { afd_bridge_cancel(client.0, 0, ready) }, OK);
    assert_eq!(client.wait(0, ready), 4);
    client.submit(0, 1, 1);
    client.submit(1, 1, 1);
    let started = Instant::now();
    drop(client);
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "close waited for remote deadline"
    );
}

#[test]
fn invalid_config_request_and_handle_are_rejected() {
    assert_eq!(afd_bridge_abi_version(), 1);
    assert_eq!(unsafe { afd_bridge_close(ptr::null_mut()) }, INVALID);
    let fleet = Fleet::new(|_, _| Action::Reply);
    for field in [
        "executors",
        "peers",
        "lanes",
        "capacity_rows",
        "max_frame_bytes",
        "timeout_ms",
    ] {
        let mut config = serde_json::json!({"transport":"tcp", "peers":fleet.peers,
            "executors":EXECUTORS, "capacity_rows":128});
        config[field] = match field {
            "executors" => serde_json::json!([11, 11, 33, 44]),
            "peers" => serde_json::json!(vec![fleet.peers[0]; 4]),
            "lanes" => serde_json::json!(3),
            _ => serde_json::json!(0),
        };
        let bytes = serde_json::to_vec(&config).unwrap();
        let mut handle = ptr::null_mut();
        assert_eq!(
            unsafe { afd_bridge_create(bytes.as_ptr(), bytes.len(), &mut handle) },
            INVALID,
            "{field}"
        );
        assert!(handle.is_null());
    }
    let client = Client::new(&fleet, 1000);
    let frame = request(1, 1);
    let mut ticket = 0;
    assert_eq!(
        unsafe { afd_bridge_submit(client.0, 2, frame.as_ptr(), frame.len(), &mut ticket) },
        INVALID
    );
    assert_eq!(
        unsafe { afd_bridge_submit(client.0, 0, frame.as_ptr(), frame.len() - 1, &mut ticket) },
        INVALID
    );
    let valid = client.submit(0, 2, 1);
    assert_eq!(client.wait(0, valid), 2);
    client.collect(0, valid, 2, 1);
}

#[test]
#[ignore = "diagnostic latency probe; CPU loopback is not fleet performance"]
fn cpu_loopback_latency_probe() {
    let fleet = Fleet::new(|_, _| Action::Reply);
    let client = Client::new(&fleet, 5000);
    for rows in [1, 16, 80, 128] {
        let mut samples = Vec::new();
        for id in 1..=35 {
            let started = Instant::now();
            let ticket = client.submit(0, id, rows);
            assert_eq!(client.wait(0, ticket), 2);
            client.collect(0, ticket, id, rows);
            if id > 5 {
                samples.push(started.elapsed().as_micros());
            }
        }
        samples.sort();
        eprintln!("CPU_LOOPBACK rows={rows} trials={} p50_us={} p95_us={} max_us={} (includes 1ms caller polling + result validation)",
            samples.len(), samples[samples.len()/2], samples[samples.len()*95/100], samples.last().unwrap());
    }
    assert_eq!(fleet.connections.load(Ordering::SeqCst), 4);
}

#[test]
fn rank_result_allocation_reused_across_collect_and_shape_changes() {
    let fleet = Fleet::new(|_, _| Action::Reply);
    let client = Client::new(&fleet, 1000);
    let mut original_capacity = 0;
    for (id, rows) in [8, 8, 3, 8, 16, 8].into_iter().enumerate() {
        let ticket = client.submit(0, id as u64 + 1, rows);
        assert_eq!(client.wait(0, ticket), 2);
        let (mut capacity, mut growths) = (0, 0);
        assert_eq!(
            unsafe { afd_bridge_buffer_stats(client.0, 0, &mut capacity, &mut growths) },
            OK
        );
        assert_eq!(growths, if id < 4 { 1 } else { 2 });
        if id == 0 {
            original_capacity = capacity;
        }
        if id < 4 {
            assert_eq!(capacity, original_capacity);
        }
        assert!(capacity <= 4 * 128 * 10240);
        client.collect(0, ticket, id as u64 + 1, rows);
    }
}
