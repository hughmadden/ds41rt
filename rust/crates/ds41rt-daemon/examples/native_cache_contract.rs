//! CPU-only JSON-lines contract fixture. Native CacheCommands/serde/page rules
//! are real; only device completion is controlled. Never loads a native library.
use anyhow::{bail, ensure, Context, Result};
use ds41rt_daemon::native_executor::{
    CacheCommands, CacheDevice, CacheReader, Completion, Envelope, RequestHandle, SourceWrites,
    Transaction,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    cell::RefCell,
    collections::BTreeMap,
    io::{self, BufRead, Read, Write},
    rc::Rc,
};

const MAX_LINE: usize = 131072;

#[derive(Clone, Copy)]
enum Status {
    Pending,
    Ready,
    Failed,
}

struct Queued {
    writes: Vec<SourceWrites>,
    status: Status,
    cancelled: bool,
}

struct DeviceState {
    pages: [usize; 4],
    queued: BTreeMap<u64, Queued>,
    fail_next_start: bool,
    starts: u64,
    drains: u64,
}

#[derive(Clone)]
struct FakeDevice(Rc<RefCell<DeviceState>>);

impl FakeDevice {
    fn new(pages: [usize; 4]) -> Self {
        Self(Rc::new(RefCell::new(DeviceState {
            pages,
            queued: BTreeMap::new(),
            fail_next_start: false,
            starts: 0,
            drains: 0,
        })))
    }

    fn complete(&self, id: u64, status: Status) -> Result<()> {
        self.0
            .borrow_mut()
            .queued
            .get_mut(&id)
            .context("transaction has no pending fake device work")?
            .status = status;
        Ok(())
    }

    fn snapshot(&self) -> Value {
        let state = self.0.borrow();
        let queued: BTreeMap<_, _> = state
            .queued
            .iter()
            .map(|(id, q)| {
                let copies: Vec<_> = q.writes.iter().map(|w| w.tail_copies.len()).collect();
                let rows: Vec<_> = q.writes.iter().map(|w| w.destinations.len()).collect();
                (
                    id.to_string(),
                    json!({"tail_copies": copies, "destination_rows": rows,
                "cancelled": q.cancelled}),
                )
            })
            .collect();
        json!({"queued": queued, "starts": state.starts, "drains": state.drains})
    }
}

// SAFETY: no device reads/writes exist in this fixture. Pending work is owned
// metadata only; poll removes it before reporting completion/failure. drain
// discards it synchronously and reader guards retain the Rc storage owner.
unsafe impl CacheDevice for FakeDevice {
    type Storage = Rc<RefCell<DeviceState>>;

    fn source_page_capacity(&self) -> [usize; 4] {
        self.0.borrow().pages
    }
    fn retain_storage(&self) -> Self::Storage {
        self.0.clone()
    }

    fn start(&mut self, transaction: Transaction, writes: Vec<SourceWrites>) -> Result<()> {
        let mut state = self.0.borrow_mut();
        ensure!(
            state.queued.len() < 2 && !state.queued.contains_key(&transaction.id),
            "fake device ownership bound exceeded"
        );
        state.queued.insert(
            transaction.id,
            Queued {
                writes,
                status: Status::Pending,
                cancelled: false,
            },
        );
        state.starts += 1;
        if std::mem::take(&mut state.fail_next_start) {
            bail!("controlled start failure after enqueuing metadata");
        }
        Ok(())
    }

    fn poll(&mut self, transaction: Transaction) -> Completion {
        let mut state = self.0.borrow_mut();
        let status = state.queued.get(&transaction.id).map(|q| q.status);
        match status {
            Some(Status::Pending) => Completion::Pending,
            Some(Status::Ready | Status::Failed) => {
                state.queued.remove(&transaction.id);
                state.drains += 1;
                if matches!(status, Some(Status::Ready)) {
                    Completion::Complete
                } else {
                    Completion::Failed("controlled drained write failure".into())
                }
            }
            None => Completion::Failed("fake device transaction absent".into()),
        }
    }

    fn cancel(&mut self, transaction: Transaction) {
        if let Some(queued) = self.0.borrow_mut().queued.get_mut(&transaction.id) {
            queued.cancelled = true;
        }
    }

    fn drain(&mut self, transaction: Transaction) {
        let mut state = self.0.borrow_mut();
        if state.queued.remove(&transaction.id).is_some() {
            state.drains += 1;
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "control", rename_all = "snake_case", deny_unknown_fields)]
enum Control {
    Snapshot,
    Ready {
        transaction_id: u64,
    },
    Fail {
        transaction_id: u64,
    },
    FailNextStart,
    HoldReader {
        reader_id: u64,
        request: RequestHandle,
    },
    DropReader {
        reader_id: u64,
    },
}

struct Fixture {
    cache: CacheCommands<FakeDevice>,
    device: FakeDevice,
    readers: BTreeMap<u64, CacheReader<Rc<RefCell<DeviceState>>>>,
}

impl Fixture {
    fn control(&mut self, command: Control) -> Result<Value> {
        match command {
            Control::Snapshot => {}
            Control::Ready { transaction_id } => {
                self.device.complete(transaction_id, Status::Ready)?
            }
            Control::Fail { transaction_id } => {
                self.device.complete(transaction_id, Status::Failed)?
            }
            Control::FailNextStart => self.device.0.borrow_mut().fail_next_start = true,
            Control::HoldReader { reader_id, request } => {
                ensure!(
                    self.readers.len() < 16 && !self.readers.contains_key(&reader_id),
                    "reader fixture bound/identity invalid"
                );
                self.readers.insert(reader_id, self.cache.read(request)?);
            }
            Control::DropReader { reader_id } => {
                self.readers.remove(&reader_id).context("reader absent")?;
            }
        }
        Ok(json!({"snapshot": self.cache.snapshot(), "device": self.device.snapshot()}))
    }

    fn line(&mut self, text: &str) -> Result<Value> {
        let value: Value = serde_json::from_str(text)?;
        if value.get("control").is_some() {
            self.control(serde_json::from_value(value)?)
        } else {
            let envelope: Envelope = serde_json::from_value(value)?;
            Ok(serde_json::to_value(self.cache.execute(envelope)?)?)
        }
    }
}

fn emit(output: &mut impl Write, value: Value) -> Result<()> {
    serde_json::to_writer(&mut *output, &value)?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() <= 2,
        "usage: native_cache_contract [owner [four_page_capacities_json]]"
    );
    let owner = args.first().map_or(Ok(17), |s| s.parse::<u64>())?;
    let pages: [usize; 4] = args
        .get(1)
        .map_or(Ok([4; 4]), |s| serde_json::from_str(s))?;
    ensure!(
        pages.iter().all(|&p| p <= 64),
        "fixture page capacity exceeds bound"
    );
    let device = FakeDevice::new(pages);
    let mut fixture = Fixture {
        cache: CacheCommands::new(owner, 4, pages, 1024, 8, device.clone())?,
        device,
        readers: BTreeMap::new(),
    };
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let stdout = io::stdout();
    let mut output = stdout.lock();
    emit(&mut output, fixture.control(Control::Snapshot)?)?;
    for _ in 0..10000 {
        let mut line = String::new();
        let read = Read::by_ref(&mut input)
            .take((MAX_LINE + 1) as u64)
            .read_line(&mut line)?;
        if read == 0 {
            return Ok(());
        }
        ensure!(
            read <= MAX_LINE && line.ends_with('\n'),
            "fixture line exceeds bound or lacks newline"
        );
        let value = fixture.line(&line).unwrap_or_else(|error| {
            json!({
                "error": error.to_string(), "snapshot": fixture.cache.snapshot(),
            })
        });
        emit(&mut output, value)?;
    }
    bail!("fixture command count exceeds bound")
}
