//! Additive multi-request command helpers; physical ownership stays native.
use super::*;
use actor::Bank;

pub(super) fn prepare<B: Bank, D: TargetDriver>(
    bank: &B,
    context: &mut TargetContext<D>,
    placement: u64,
    members: Vec<BatchInput>,
) -> Api<Ticket> {
    let inputs = members
        .into_iter()
        .map(|member| {
            if !matches!(member.kind, Kind::Decode) {
                return Err(fail(INVALID, "batch supports decode only"));
            }
            if bank.end(member.request).map_err(native_error)? != member.expected_committed_end {
                return Err(fail(STALE, "native batch frontier differs from grant"));
            }
            Ok(TargetInput {
                request: member.request,
                tokens: member.tokens,
                selected: member.selected,
                kind: SourceKind::Decode,
                placement,
            })
        })
        .collect::<Api<Vec<_>>>()?;
    context.submit_batch(inputs).map_err(native_error)
}
pub(super) fn frontiers<B: Bank>(
    bank: &B,
    requests: &[RequestHandle],
    cancelled: bool,
) -> Api<Vec<Value>> {
    requests
        .iter()
        .map(|&request| {
            let (end, draft) = proposal::frontiers(bank, request)?;
            let mut member =
                json!({"request":request,"committed_end":end,"draft_committed_end":draft});
            if cancelled {
                member["revoked"] = json!(false);
            }
            Ok(member)
        })
        .collect()
}
pub(super) fn prepared<B: Bank, D: TargetDriver>(
    bank: &B,
    context: &TargetContext<D>,
    ticket: Ticket,
    draft_us: u64,
) -> Api<Value> {
    let members = context
        .batch_members(ticket)
        .map_err(native_error)?
        .into_iter()
        .map(|member| {
            let (end, draft) = proposal::frontiers(bank, member.request)?;
            let mut value = serde_json::to_value(member).expect("batch manifest serializes");
            value["committed_end"] = json!(end);
            value["draft_committed_end"] = json!(draft);
            Ok(value)
        })
        .collect::<Api<Vec<Value>>>()?;
    Ok(
        json!({"state":"prepared","ticket":ticket,"members":members,"draft_us":draft_us,"bank":bank.info()}),
    )
}
