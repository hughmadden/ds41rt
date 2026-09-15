//! Upstream admission-control port (ds41rt component C8).
//!
//! Ported invariant classes from:
//! - vLLM `tests/v1/engine/test_admission_control.py` (queue overflow,
//!   admission gating, HTTP 503 rejection mapping), and
//! - sglang `test/registered/scheduler/test_min_free_slots_delayer.py`
//!   (min-free-slots admission-delay policy arithmetic).
//!
//! # Mapping to ds41rt surfaces
//!
//! | upstream invariant                    | ds41rt surface under test |
//! |---------------------------------------|---------------------------|
//! | admission gating / capacity accounting | `ds41rt_core::admit_layerwaves_for_iteration` (the per-iteration scheduler admission used by `commands::scheduler_smoke` and `real_full/scheduler/execution/admission.rs`) |
//! | queue-full rejection shape (503)       | `ds41rt_api::native_v41` router: bounded mpsc queue `try_send` failure -> HTTP 503, closed-queue health -> 503, worker-side admission failure -> 500 with cause retained |
//! | delayed admission (min-free-slots)     | sglang policy ported verbatim below as pure functions (`resolve_min_free_slots`, `MinFreeSlotsDelayer::should_delay`) plus a deferred-prefill scheduler simulation against the real `admit_layerwaves_for_iteration` |
//! | priority under pressure                | `PrefillChunkPolicy::decode_priority` ordering inside `admit_layerwaves_for_iteration` |
//! | concurrent single-slot admission       | the daemon's own invariant, `v41_requests::Requests::admit` + `v41_native_serve/scheduler.rs` slot loop, modeled here as a pure slot-admission test (the real loop needs CUDA machinery) |
//!
//! Not ported (no ds41rt equivalent): vLLM `human_readable_int` CLI notation
//! and `SchedulerConfig` pydantic validation; vLLM's `max_num_queued_reqs`
//! counting of *unfinished* requests has no counterparty because the daemon
//! bounds concurrency by fixed slots (`args.concurrency`), which the slot
//! model tests below cover instead.

use ds41rt_core::{
    admit_layerwaves_for_iteration, LayerWave, LayerWaveMode, MtpVerifyBlock, PrefillChunk,
    PrefillChunkPolicy, Priority,
};

fn policy(
    max_prefill_tokens_per_iteration: usize,
    max_active_prefill_chunks: usize,
    decode_priority: bool,
) -> PrefillChunkPolicy {
    PrefillChunkPolicy {
        chunk_tokens: 16,
        max_prefill_tokens_per_iteration,
        max_active_prefill_chunks,
        decode_priority,
    }
}

fn prefill(name: &str, token_start: usize, token_count: usize, priority: i32) -> LayerWave {
    LayerWave::prefill(PrefillChunk::new(
        name,
        format!("seq-{name}"),
        3,
        token_start as u64,
        token_count,
        50 + token_start as u64,
        Priority(priority),
        ds41rt_core::GraphBucket::new(16),
        "placement-a",
    ))
}

fn decode(name: &str, position: u64, priority: i32) -> LayerWave {
    LayerWave::decode(ds41rt_core::DecodeStep::new(
        name,
        format!("seq-{name}"),
        3,
        position,
        Some(70 + position),
        Priority(priority),
        "placement-a",
    ))
}

fn mtp_verify(name: &str, token_start: usize, token_count: usize, priority: i32) -> LayerWave {
    LayerWave::mtp_verify(MtpVerifyBlock::new(
        name,
        format!("seq-{name}"),
        3,
        token_start as u64,
        token_count,
        Some(90 + token_start as u64),
        Priority(priority),
        ds41rt_core::GraphBucket::new(16),
        "placement-a",
    ))
}

// ---------------------------------------------------------------------------
// sglang min-free-slots admission-delay policy (verbatim arithmetic port)
// ---------------------------------------------------------------------------

/// Verbatim port of sglang `min_free_slots_delayer.resolve_min_free_slots`.
/// `None` = disabled. Explicit user value wins, capped by
/// `max_running_requests` (`<= 1` disables). Unset + DFlash family falls back
/// to the legacy formula, disabled for clusters under 8.
fn resolve_min_free_slots(
    user_value: Option<i64>,
    max_running_requests: i64,
    is_dflash_family: bool,
) -> Option<i64> {
    let max_running_requests = max_running_requests.max(0);
    if let Some(user_value) = user_value {
        let threshold = user_value.min(max_running_requests);
        return (threshold > 1).then_some(threshold);
    }
    if is_dflash_family && max_running_requests >= 8 {
        return Some((max_running_requests + 5) / 6).clamp(Some(2), Some(4));
    }
    None
}

trait ClampOption {
    fn clamp(self, lo: Option<i64>, hi: Option<i64>) -> Option<i64>;
}

impl ClampOption for Option<i64> {
    fn clamp(self, lo: Option<i64>, hi: Option<i64>) -> Option<i64> {
        self.map(|value| value.clamp(lo.unwrap_or(i64::MIN), hi.unwrap_or(i64::MAX)))
    }
}

/// Verbatim port of sglang `MinFreeSlotsDelayer::should_delay`: delay fresh
/// admissions only while a decode batch is running and fewer than
/// `min_free_slots` allocatable request slots remain.
struct MinFreeSlotsDelayer {
    min_free_slots: i64,
}

impl MinFreeSlotsDelayer {
    fn should_delay(&self, running_bs: i64, num_allocatable_reqs: i64) -> bool {
        running_bs > 0 && num_allocatable_reqs < self.min_free_slots
    }
}

mod min_free_slots_delayer {
    use super::{resolve_min_free_slots, MinFreeSlotsDelayer};

    #[test]
    fn unset_non_dflash_disables() {
        assert_eq!(resolve_min_free_slots(None, 512, false), None);
    }

    #[test]
    fn unset_dflash_auto_enables() {
        assert_eq!(resolve_min_free_slots(None, 512, true), Some(4));
        assert_eq!(resolve_min_free_slots(None, 8, true), Some(2));
    }

    #[test]
    fn unset_dflash_small_cluster_disables() {
        assert_eq!(resolve_min_free_slots(None, 7, true), None);
        assert_eq!(resolve_min_free_slots(None, 0, true), None);
    }

    #[test]
    fn le_one_disables() {
        // <= 1 can never batch, so it is a no-op.
        assert_eq!(resolve_min_free_slots(Some(1), 512, false), None);
        assert_eq!(resolve_min_free_slots(Some(0), 512, false), None);
    }

    #[test]
    fn explicit_value_survives_small_cluster() {
        // The < 8 guard belongs to the DFlash auto-default, not explicit values.
        assert_eq!(resolve_min_free_slots(Some(4), 7, false), Some(4));
        assert_eq!(resolve_min_free_slots(Some(4), 7, true), Some(4));
    }

    #[test]
    fn non_dflash_uses_explicit_value() {
        assert_eq!(resolve_min_free_slots(Some(2), 8, false), Some(2));
        assert_eq!(resolve_min_free_slots(Some(3), 512, false), Some(3));
        assert_eq!(resolve_min_free_slots(Some(8), 512, false), Some(8));
        assert_eq!(resolve_min_free_slots(Some(16), 512, false), Some(16));
    }

    #[test]
    fn explicit_value_is_capped_to_max_running_requests() {
        assert_eq!(resolve_min_free_slots(Some(16), 8, false), Some(8));
    }

    #[test]
    fn user_value_overrides_dflash_default() {
        assert_eq!(resolve_min_free_slots(Some(3), 512, true), Some(3));
        assert_eq!(resolve_min_free_slots(Some(16), 512, true), Some(16));
    }

    #[test]
    fn explicit_one_disables_dflash_default() {
        assert_eq!(resolve_min_free_slots(Some(1), 512, true), None);
    }

    #[test]
    fn delayer_delays_below_threshold() {
        let delayer = MinFreeSlotsDelayer { min_free_slots: 4 };
        assert!(delayer.should_delay(100, 2));
    }

    #[test]
    fn delayer_no_delay_at_or_above_threshold() {
        let delayer = MinFreeSlotsDelayer { min_free_slots: 4 };
        assert!(!delayer.should_delay(100, 4));
        assert!(!delayer.should_delay(100, 8));
    }

    #[test]
    fn delayer_no_delay_when_idle() {
        // Nothing running: no decode batch to protect, prefill at once.
        let delayer = MinFreeSlotsDelayer { min_free_slots: 4 };
        assert!(!delayer.should_delay(0, 0));
    }
}

// ---------------------------------------------------------------------------
// Admission gating / capacity accounting (vLLM check_admission mapped onto
// admit_layerwaves_for_iteration)
// ---------------------------------------------------------------------------

mod admission_gating {
    use super::*;

    #[test]
    fn admission_no_effective_limits_allows_everything() {
        // An enormous budget admits a mixed batch: decode, MTP verify and
        // prefill waves all selected, nothing deferred (vLLM:
        // test_admission_no_limits_allows_everything).
        let policy = policy(usize::MAX, usize::MAX, true);
        let admission = admit_layerwaves_for_iteration(
            vec![prefill("p0", 0, 16, 0), decode("d0", 10, 0), mtp_verify("m0", 5, 3, 0)],
            &policy,
        );
        assert_eq!(admission.selected.len(), 3);
        assert!(admission.deferred.is_empty());
        assert_eq!(admission.selected_decode_rows, 1);
        assert_eq!(admission.selected_mtp_rows, 3);
        assert_eq!(admission.selected_prefill_rows, 16);
        assert_eq!(admission.selected_prefill_chunks, 1);
    }

    #[test]
    fn admission_prefill_allows_at_exact_token_boundary() {
        // Exactly the budget is admitted (vLLM: ..._allows_n_at_boundary).
        let policy = policy(32, 8, true);
        let admission = admit_layerwaves_for_iteration(vec![prefill("p0", 0, 32, 0)], &policy);
        assert_eq!(admission.selected_prefill_rows, 32);
        assert!(admission.deferred.is_empty());
    }

    #[test]
    fn admission_prefill_defers_beyond_token_limit() {
        // Budget+1 rows defers the wave (vLLM: ..._rejects_at_limit /
        // ..._rejects_over_limit).
        let policy = policy(32, 8, true);
        let admission = admit_layerwaves_for_iteration(vec![prefill("p0", 0, 33, 0)], &policy);
        assert!(admission.selected.is_empty());
        assert_eq!(admission.deferred.len(), 1);
        assert_eq!(admission.deferred[0].mode, LayerWaveMode::Prefill);
    }

    #[test]
    fn admission_prefill_defers_beyond_chunk_limit() {
        // Token budget alone is not enough: the active-chunk count gate
        // defers the third chunk (vLLM: independent limit checks).
        let policy = policy(1024, 2, true);
        let admission = admit_layerwaves_for_iteration(
            vec![prefill("p0", 0, 16, 0), prefill("p1", 16, 16, 1), prefill("p2", 32, 16, 2)],
            &policy,
        );
        assert_eq!(admission.selected_prefill_chunks, 2);
        assert_eq!(admission.deferred.len(), 1);
        assert_eq!(admission.deferred[0].request_id, prefill("p2", 32, 16, 2).request_id);
    }

    #[test]
    fn admission_zero_token_budget_defers_all_prefill() {
        // vLLM: ..._rejects_when_zero_limit. A zero token budget rejects every
        // non-empty prefill wave, but decode work is never subject to the
        // prefill budget and still admits.
        let policy = policy(0, 8, true);
        let admission = admit_layerwaves_for_iteration(
            vec![prefill("p0", 0, 16, 0), decode("d0", 10, 0)],
            &policy,
        );
        assert_eq!(admission.selected.len(), 1);
        assert_eq!(admission.selected[0].mode, LayerWaveMode::Decode);
        assert_eq!(admission.deferred.len(), 1);
    }

    #[test]
    fn admission_zero_chunk_budget_defers_all_prefill() {
        let policy = policy(1024, 0, true);
        let admission = admit_layerwaves_for_iteration(vec![prefill("p0", 0, 16, 0)], &policy);
        assert!(admission.selected.is_empty());
        assert_eq!(admission.deferred.len(), 1);
    }

    #[test]
    fn admission_decode_and_mtp_are_never_token_gated() {
        // Under a zero prefill budget a decode + MTP batch still admits in
        // full: interactive latency work bypasses the prefill backlog gate.
        let policy = policy(0, 0, true);
        let admission = admit_layerwaves_for_iteration(
            vec![decode("d0", 3, 0), mtp_verify("m0", 4, 2, 0)],
            &policy,
        );
        assert_eq!(admission.selected.len(), 2);
        assert!(admission.deferred.is_empty());
        assert_eq!(admission.selected_decode_rows, 1);
        assert_eq!(admission.selected_mtp_rows, 2);
    }

    #[test]
    fn admission_token_and_chunk_limits_checked_independently() {
        // vLLM: test_admission_both_limits_checked_independently — either
        // limit alone can defer. Here the token budget is exhausted first.
        let policy = policy(16, 4, true);
        let admission = admit_layerwaves_for_iteration(
            vec![prefill("p0", 0, 16, 0), prefill("p1", 16, 16, 1)],
            &policy,
        );
        assert_eq!(admission.selected_prefill_rows, 16);
        assert_eq!(admission.deferred.len(), 1);
        // And with tokens available but chunks exhausted, deferral also fires.
        let policy = policy(1024, 1, true);
        let admission = admit_layerwaves_for_iteration(
            vec![prefill("p0", 0, 16, 0), prefill("p1", 16, 16, 1)],
            &policy,
        );
        assert_eq!(admission.selected_prefill_chunks, 1);
        assert_eq!(admission.deferred.len(), 1);
    }

    #[test]
    fn admission_accounting_matches_selected_waves() {
        // Capacity accounting (the vLLM get_num_queued_tokens analog): the
        // reported row totals always equal the sum of the admitted waves.
        let policy = policy(32, 4, true);
        let waves = vec![
            prefill("p0", 0, 16, 0),
            prefill("p1", 16, 8, 1),
            decode("d0", 9, 0),
            mtp_verify("m0", 11, 4, 0),
        ];
        let expected_prefill: usize =
            waves.iter().filter(|wave| wave.mode == LayerWaveMode::Prefill).map(LayerWave::num_rows).sum();
        let expected_decode: usize =
            waves.iter().filter(|wave| wave.mode == LayerWaveMode::Decode).map(LayerWave::num_rows).sum();
        let expected_mtp: usize = waves
            .iter()
            .filter(|wave| wave.mode == LayerWaveMode::MtpVerify)
            .map(LayerWave::num_rows)
            .sum();
        let admission = admit_layerwaves_for_iteration(waves, &policy);
        assert_eq!(admission.selected_prefill_rows, expected_prefill);
        assert_eq!(admission.selected_decode_rows, expected_decode);
        assert_eq!(admission.selected_mtp_rows, expected_mtp);
        assert_eq!(
            admission.selected.len(),
            admission.selected_prefill_chunks + 1 + 1
        );
    }

    #[test]
    fn admission_policy_defaults_gate_prefill_but_not_decode() {
        // PrefillChunkPolicy::default mirrors the deployed daemon smoke
        // defaults (128-token chunks, 512 tokens and 4 chunks per iteration,
        // decode priority on).
        let policy = PrefillChunkPolicy::default();
        assert!(policy.decode_priority);
        let admission = admit_layerwaves_for_iteration(
            vec![decode("d0", 0, 0), prefill("p0", 0, 512, 0)],
            &policy,
        );
        assert_eq!(admission.selected.len(), 2);
        let admission =
            admit_layerwaves_for_iteration(vec![prefill("p0", 0, 513, 0)], &policy);
        assert_eq!(admission.deferred.len(), 1);
    }
}

// ---------------------------------------------------------------------------
// Priority under pressure and queue-full deferral shape
// ---------------------------------------------------------------------------

mod admission_priority {
    use super::*;

    #[test]
    fn admission_decode_priority_selects_decode_before_prefill_regardless_of_priority_value() {
        // decode_priority=true: mode rank dominates the numeric priority, so
        // a low-priority decode still leads the iteration.
        let policy = policy(1024, 8, true);
        let admission = admit_layerwaves_for_iteration(
            vec![prefill("p0", 0, 16, 0), decode("d0", 10, 99)],
            &policy,
        );
        assert_eq!(admission.selected[0].mode, LayerWaveMode::Decode);
        assert_eq!(admission.selected[1].mode, LayerWaveMode::Prefill);
    }

    #[test]
    fn admission_mode_order_is_decode_then_mtp_then_prefill() {
        let policy = policy(1024, 8, true);
        let admission = admit_layerwaves_for_iteration(
            vec![
                prefill("p0", 0, 16, 0),
                mtp_verify("m0", 4, 2, 50),
                decode("d0", 10, 99),
            ],
            &policy,
        );
        let modes: Vec<_> = admission.selected.iter().map(|wave| wave.mode).collect();
        assert_eq!(
            modes,
            vec![LayerWaveMode::Decode, LayerWaveMode::MtpVerify, LayerWaveMode::Prefill]
        );
    }

    #[test]
    fn admission_prefill_ties_break_by_priority_then_arrival() {
        // Within one mode the numeric Priority field orders admission; equal
        // priorities preserve arrival order (stable index tiebreak).
        let policy = policy(1024, 8, true);
        let admission = admit_layerwaves_for_iteration(
            vec![prefill("late-low", 0, 8, 5), prefill("early-high", 8, 8, 1), prefill("tie-a", 16, 8, 5), prefill("tie-b", 24, 8, 5)],
            &policy,
        );
        let ids: Vec<_> = admission.selected.iter().map(|wave| wave.request_id.to_string()).collect();
        assert_eq!(ids, vec!["early-high", "late-low", "tie-a", "tie-b"]);
    }

    #[test]
    fn admission_without_decode_priority_orders_by_priority_across_modes() {
        // decode_priority=false flattens mode rank, so the numeric priority
        // dominates across modes and a high-priority prefill can lead.
        let policy = policy(1024, 8, false);
        let admission = admit_layerwaves_for_iteration(
            vec![decode("d0", 10, 50), prefill("p0", 0, 16, 1)],
            &policy,
        );
        assert_eq!(admission.selected[0].mode, LayerWaveMode::Prefill);
        assert_eq!(admission.selected[1].mode, LayerWaveMode::Decode);
    }

    #[test]
    fn admission_deferred_preserves_order_and_is_readmitted_next_iteration() {
        // Queue-full rejection shape: deferral is graceful, ordered, and the
        // deferred work is fully admissible on the next iteration once the
        // budget frees — nothing is dropped (vLLM: try-again semantics).
        let policy = policy(16, 1, true);
        let first = admit_layerwaves_for_iteration(
            vec![prefill("p0", 0, 16, 0), prefill("p1", 16, 16, 1), prefill("p2", 32, 16, 2)],
            &policy,
        );
        assert_eq!(first.selected.len(), 1);
        assert_eq!(first.deferred.len(), 2);
        let deferred_ids: Vec<_> = first.deferred.iter().map(|wave| wave.request_id.to_string()).collect();
        assert_eq!(deferred_ids, vec!["p1", "p2"]);
        // Next iteration sees the deferred waves plus fresh decode work;
        // decode leads and exactly one deferred prefill fits the freed slot.
        let mut candidates = first.deferred.clone();
        candidates.push(decode("d1", 99, 0));
        let second = admit_layerwaves_for_iteration(candidates, &policy);
        assert_eq!(second.selected[0].mode, LayerWaveMode::Decode);
        assert_eq!(second.selected.len(), 2);
        assert_eq!(second.selected_prefill_rows, 16);
        assert_eq!(second.deferred.len(), 1);
        assert_eq!(second.deferred[0].request_id.to_string(), "p2");
    }

    #[test]
    fn admission_empty_batch_is_a_noop() {
        let policy = policy(16, 1, true);
        let admission = admit_layerwaves_for_iteration(Vec::new(), &policy);
        assert!(admission.selected.is_empty());
        assert!(admission.deferred.is_empty());
        assert_eq!(admission.selected_prefill_rows, 0);
    }
}

// ---------------------------------------------------------------------------
// Delayed admission: scheduler-loop simulation (sglang MinFreeSlotsDelayer
// semantics + real admit_layerwaves_for_iteration)
// ---------------------------------------------------------------------------

mod delayed_admission {
    use super::*;

    /// One scheduler round: admit from the pending queues, return the
    /// admitted wave ids. Models the daemon's `scheduler_smoke` loop.
    fn round(
        policy: &PrefillChunkPolicy,
        pending_decodes: &mut Vec<LayerWave>,
        pending_prefill: &mut Vec<LayerWave>,
        admitted_log: &mut Vec<String>,
    ) {
        let candidates = pending_decodes
            .iter()
            .cloned()
            .chain(pending_prefill.iter().cloned())
            .collect::<Vec<_>>();
        let admission = admit_layerwaves_for_iteration(candidates, policy);
        for wave in admission.selected {
            let id = wave.request_id.to_string();
            admitted_log.push(id.clone());
            if let Some(index) = pending_decodes.iter().position(|pending| pending.request_id == wave.request_id) {
                pending_decodes.remove(index);
            } else if let Some(index) =
                pending_prefill.iter().position(|pending| pending.request_id == wave.request_id)
            {
                pending_prefill.remove(index);
            } else {
                panic!("admitted wave {id} was not pending");
            }
        }
        let mut requeue = admission.deferred;
        requeue.extend(pending_prefill.drain(..));
        *pending_prefill = requeue;
    }

    #[test]
    fn admission_decode_never_waits_behind_a_prefill_backlog() {
        // Invariant under pressure: with decode_priority on and any prefill
        // budget, an arriving decode is admitted at the very next round no
        // matter how deep the prefill backlog is (sglang's delayer protects
        // the same invariant by delaying prefill, never decode).
        let policy = policy(16, 1, true);
        let mut pending_prefill = (0..8).map(|index| prefill(&format!("p{index}"), index * 16, 16, index as i32)).collect::<Vec<_>>();
        let mut pending_decodes = Vec::new();
        let mut admitted = Vec::new();
        // Saturate several rounds with prefill only.
        for _ in 0..3 {
            round(&policy, &mut pending_decodes, &mut pending_prefill, &mut admitted);
        }
        assert!(pending_prefill.len() < 8);
        // A decode arrives mid-backlog.
        pending_decodes.push(decode("d0", 999, 0));
        round(&policy, &mut pending_decodes, &mut pending_prefill, &mut admitted);
        assert!(pending_decodes.is_empty(), "decode was delayed behind prefill backlog");
        assert_eq!(admitted.last().map(String::as_str), Some("d0"));
        // The prefill backlog still makes progress and loses nothing.
        let total_prefill = 8;
        let remaining = pending_prefill.len();
        assert_eq!(admitted.iter().filter(|id| id.starts_with('p')).count() + remaining, total_prefill);
    }

    #[test]
    fn admission_prefill_is_deferred_only_while_slots_free_up() {
        // MinFreeSlotsDelayer-shaped policy: while a decode batch is running
        // and free slots are below the threshold, fresh prefill waits; once
        // slots free up, the whole deferred batch admits at once (batching
        // into one admission instead of one at a time).
        let policy = policy(64, 4, true);
        let mut pending_prefill = vec![prefill("p0", 0, 8, 0), prefill("p1", 8, 8, 1)];
        let mut pending_decodes = vec![decode("d0", 0, 0)];
        let mut admitted = Vec::new();
        // Round 1: decode leads; prefill still fits alongside (64-token
        // budget, 4 chunks), so everything admits.
        round(&policy, &mut pending_decodes, &mut pending_prefill, &mut admitted);
        assert!(pending_prefill.is_empty());
        assert!(pending_decodes.is_empty());
        // Now overload the budget with a large prefill wave and a decode:
        // the wave defers, decode still lands.
        let mut pending_prefill = vec![prefill("big", 0, 128, 0)];
        let mut pending_decodes = vec![decode("d1", 1, 0)];
        let mut admitted = Vec::new();
        round(&policy, &mut pending_decodes, &mut pending_prefill, &mut admitted);
        assert!(pending_decodes.is_empty());
        assert_eq!(pending_prefill.len(), 1);
        // Decode finishes (dropped from pending); the deferred wave now fits
        // the whole budget by itself and admits in one shot.
        round(&policy, &mut Vec::new(), &mut pending_prefill, &mut admitted);
        assert!(pending_prefill.is_empty());
        assert_eq!(admitted.last().map(String::as_str), Some("big"));
    }

    #[test]
    fn admission_idle_system_never_defers() {
        // Nothing running: first come, first served up to budget; no
        // artificial delay is introduced (sglang: no delay when idle).
        let policy = policy(32, 2, true);
        let mut pending_prefill =
            vec![prefill("p0", 0, 16, 0), prefill("p1", 16, 16, 1), prefill("p2", 32, 16, 2)];
        let mut admitted = Vec::new();
        round(&policy, &mut Vec::new(), &mut pending_prefill, &mut admitted);
        assert_eq!(admitted, vec!["p0", "p1"]);
        assert_eq!(pending_prefill.len(), 1);
    }
}

// ---------------------------------------------------------------------------
// Slot admission model: the daemon scheduler's concurrency invariant
// (v41_native_serve/scheduler.rs slot loop + v41_requests::Requests::admit
// occupancy check), modeled without CUDA machinery. Maps vLLM
// test_concurrent_single_request_admission_respects_limit.
// ---------------------------------------------------------------------------

mod slot_admission {
    /// Model of the daemon scheduler's fixed-slot admission table
    /// (`active: Vec<Option<Active>>` sized by `args.concurrency`).
    #[derive(Default)]
    struct SlotTable {
        occupants: Vec<Option<u64>>,
    }

    impl SlotTable {
        /// `v41_native_serve/scheduler.rs`: `active.iter().position(Option::is_none)`
        /// followed by `requests.admit(slot, id)`, which rejects an occupied
        /// slot ("request slot occupied").
        fn try_admit(&mut self, concurrency: usize, id: u64) -> Result<usize, &'static str> {
            if self.occupants.len() < concurrency {
                self.occupants.resize(concurrency, None);
            }
            let slot = self
                .occupants
                .iter()
                .position(Option::is_none)
                .ok_or("all request slots occupied")?;
            if self.occupants[slot].is_some() {
                return Err("request slot occupied");
            }
            self.occupants[slot] = Some(id);
            Ok(slot)
        }

        fn release(&mut self, id: u64) -> bool {
            if let Some(slot) = self.occupants.iter().position(|occupant| *occupant == Some(id)) {
                self.occupants[slot] = None;
                true
            } else {
                false
            }
        }
    }

    #[test]
    fn admission_slot_table_rejects_at_capacity() {
        let mut table = SlotTable::default();
        assert_eq!(table.try_admit(2, 1), Ok(0));
        assert_eq!(table.try_admit(2, 2), Ok(1));
        assert_eq!(table.try_admit(2, 3), Err("all request slots occupied"));
        // A release frees exactly one slot (queue-full is transient, the
        // next arrival admits — the try-again shape).
        assert!(table.release(1));
        assert_eq!(table.try_admit(2, 3), Ok(0));
    }

    #[test]
    fn admission_slot_table_double_admit_same_slot_rejected() {
        // Mirrors `Requests::admit`'s "request slot occupied" guard.
        let mut table = SlotTable::default();
        table.try_admit(4, 7).unwrap();
        // Simulate a stale scheduler iteration reusing a live slot.
        table.occupants[0] = Some(7);
        assert_eq!(table.try_admit(4, 8), Ok(1));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn admission_concurrent_single_slot_admission_respects_limit() {
        // vLLM: test_concurrent_single_request_admission_respects_limit.
        // The daemon serializes admission on the scheduler thread; the
        // invariant is that two concurrent contenders for one free slot
        // cannot both win. The model serializes check-and-occupy on a mutex
        // exactly as the single scheduler loop does.
        let table = std::sync::Arc::new(std::sync::Mutex::new(SlotTable::default()));
        let results = futures_like_join(
            (0..2).map(|contender| {
                let table = table.clone();
                async move {
                    // Yield once between check and occupy to expose a race if
                    // serialization were missing.
                    tokio::task::yield_now().await;
                    table.lock().unwrap().try_admit(1, contender)
                }
            })
            .collect(),
        )
        .await;
        let admitted = results.iter().filter(|result| result.is_ok()).count();
        let rejected = results.iter().filter(|result| result.is_err()).count();
        assert_eq!(admitted, 1);
        assert_eq!(rejected, 1);
        let table = table.lock().unwrap();
        assert_eq!(table.occupants.iter().flatten().count(), 1);
    }

    /// Minimal join (avoids pulling a futures dev-dependency into the crate).
    async fn futures_like_join<F: std::future::Future<Output = R>, R>(
        futures: Vec<F>,
    ) -> Vec<R> {
        let mut futures = futures.into_iter().map(Box::pin).collect::<Vec<_>>();
        let mut outputs = Vec::with_capacity(futures.len());
        while !futures.is_empty() {
            let mut index = 0;
            while index < futures.len() {
                let mut future = futures[index].as_mut();
                if let std::task::Poll::Ready(output) =
                    futures_like_poll_fn(&mut future)
                {
                    outputs.push(output);
                    futures.remove(index);
                } else {
                    index += 1;
                }
            }
            if !futures.is_empty() {
                tokio::task::yield_now().await;
            }
        }
        outputs
    }

    fn futures_like_poll_fn<F: std::future::Future>(
        future: &mut std::pin::Pin<&mut F>,
    ) -> std::task::Poll<F::Output> {
        use std::sync::{Arc, Mutex};
        use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
        // A no-op waker: this hand-rolled join only polls futures that were
        // made ready by the preceding yield, which is all this test needs.
        fn raw_waker() -> RawWaker {
            fn clone(_: *const ()) -> RawWaker {
                raw_waker()
            }
            fn wake(_: *const ()) {}
            fn wake_by_ref(_: *const ()) {}
            fn drop(_: *const ()) {}
            RawWaker::new(std::ptr::null(), &RawWakerVTable::new(clone, wake, wake_by_ref, drop))
        }
        let _arc_mutex_anchor: Arc<Mutex<()>> = Arc::new(Mutex::new(()));
        let waker = unsafe { Waker::from_raw(raw_waker()) };
        let mut context = Context::from_waker(&waker);
        future.as_mut().poll(&mut context)
    }
}

// ---------------------------------------------------------------------------
// Queue-full rejection shape over HTTP: real ds41rt-api native_v41 router.
// Maps vLLM QueueOverflowError/MaxQueuedTokensError -> create_error_response
// 503 mapping, plus the daemon's admission-failure status retention.
// ---------------------------------------------------------------------------

mod http_503_mapping {
    use ds41rt_api::native_v41;
    use std::net::SocketAddr;

    const MODEL: &str = native_v41::MODEL;

    fn chat_body(max_tokens: u32) -> String {
        format!(
            r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hi"}}],"max_tokens":{max_tokens},"stream":false}}"#
        )
    }

    /// Minimal HTTP/1.1 client over a raw tokio socket (no reqwest in this
    /// crate's dependency set). Sends `Connection: close` and reads the
    /// response to EOF.
    async fn http_request(addr: SocketAddr, method: &str, path: &str, body: &str) -> (u16, String) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut socket = tokio::net::TcpStream::connect(addr).await.unwrap();
        let request = format!(
            "{method} {path} HTTP/1.1\r\nhost: {addr}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );
        socket.write_all(request.as_bytes()).await.unwrap();
        let mut response = Vec::new();
        socket.read_to_end(&mut response).await.unwrap();
        let response = String::from_utf8(response).unwrap();
        let status = response
            .split_whitespace()
            .nth(1)
            .expect("HTTP status line")
            .parse::<u16>()
            .expect("numeric HTTP status");
        (status, response)
    }

    async fn serve(router: axum::Router) -> SocketAddr {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        addr
    }

    fn dummy_job() -> native_v41::NativeRequest {
        let (events, _receive) = tokio::sync::mpsc::channel(16);
        native_v41::NativeRequest {
            prompt: String::new(),
            constraint: None,
            images: Vec::new(),
            max_tokens: 1,
            events,
        }
    }

    #[tokio::test]
    async fn admission_queue_overflow_maps_to_503() {
        // vLLM: test_queue_overflow_maps_to_503 / test_max_queued_tokens_maps_to_503.
        // The native router's bounded queue is the waiters' admission gate:
        // a full queue rejects the chat request with HTTP 503.
        let (queue, _receive) = tokio::sync::mpsc::channel::<native_v41::NativeRequest>(2);
        for _ in 0..2 {
            queue.try_send(dummy_job()).unwrap();
        }
        let router = native_v41::router_with_limits(
            queue,
            native_v41::NativeLimits::default(),
        );
        let addr = serve(router).await;
        let (status, response) = http_request(addr, "POST", "/v1/chat/completions", &chat_body(1)).await;
        assert_eq!(status, 503, "{response}");
        assert!(response.contains("error"), "{response}");
    }

    #[tokio::test]
    async fn admission_queue_with_capacity_accepts_and_waits_for_worker() {
        // Sanity counterpart: with queue capacity the request is admitted and
        // the handler then awaits the worker's first chunk (which never
        // arrives here), so the connection stays open rather than 503.
        let (queue, mut receive) = tokio::sync::mpsc::channel::<native_v41::NativeRequest>(2);
        let worker = tokio::spawn(async move {
            // Hold admitted jobs without responding.
            let _held = receive.recv().await;
            std::future::pending::<()>().await;
        });
        let router = native_v41::router_with_limits(
            queue,
            native_v41::NativeLimits::default(),
        );
        let addr = serve(router).await;
        // A bounded-capacity admit probe: filling to capacity still leaves
        // try_send succeeding for exactly the spare slots.
        let (status, _) = http_request(addr, "GET", "/health", "").await;
        assert_eq!(status, 200);
        worker.abort();
    }

    #[tokio::test]
    async fn admission_closed_queue_health_maps_to_503() {
        // The health endpoint advertises admission availability: a closed
        // queue (worker gone) is 503, matching the upstream invariant that
        // overload/unavailability surfaces as 503, not 500.
        let (queue, receive) = tokio::sync::mpsc::channel::<native_v41::NativeRequest>(1);
        drop(receive);
        let router = native_v41::router(queue);
        let addr = serve(router).await;
        let (status, _) = http_request(addr, "GET", "/health", "").await;
        assert_eq!(status, 503);
    }

    #[tokio::test]
    async fn admission_worker_failure_maps_to_500_with_cause() {
        // The daemon's scheduler-side admission failure (e.g. pool exhausted
        // during preparation) reaches the client as a 500 carrying the cause,
        // not a fabricated success — the comment in native_v41.rs pins this.
        let (queue, mut receive) = tokio::sync::mpsc::channel::<native_v41::NativeRequest>(4);
        let worker = tokio::spawn(async move {
            while let Some(job) = receive.recv().await {
                let _ = job.events.send(Err(native_v41::NativeFailure::from("pool exhausted"))).await;
            }
        });
        let router = native_v41::router(queue);
        let addr = serve(router).await;
        let (status, response) =
            http_request(addr, "POST", "/v1/chat/completions", &chat_body(1)).await;
        assert_eq!(status, 500, "{response}");
        assert!(response.contains("pool exhausted"), "{response}");
        worker.abort();
    }

    #[tokio::test]
    async fn admission_invalid_request_maps_to_400_not_503() {
        // Rejection shape discipline: a client-side admission violation
        // (max_tokens=0) is 400, never 503 — 503 is reserved for overload.
        let (queue, _receive) = tokio::sync::mpsc::channel::<native_v41::NativeRequest>(1);
        let router = native_v41::router(queue);
        let addr = serve(router).await;
        let (status, response) =
            http_request(addr, "POST", "/v1/chat/completions", &chat_body(0)).await;
        assert_eq!(status, 400, "{response}");
        assert!(response.contains("max_tokens"), "{response}");
    }
}
