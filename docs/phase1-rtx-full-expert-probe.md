# Full-width RTX expert kernel: initial qualification

The isolated SM120 kernel now supports all 384 routed experts, full intermediate
width 2304, and six routes per row. A distinct native role (2) separates this
geometry from dSpark and the Spark TP4 shard. This is an opt-in slice export;
ordinary serving exports and the running deployment are unchanged.

Seven native cases passed exact ordered FP32 route-output comparison at live
row counts 1, 2, 5, 6, 15 and 16, using capacity-1 and capacity-16 variants.
Captured replay changes routing and input values without new device allocations.
The reference uses independently prepared CPU route groups and separately checks
its result against the existing numerical oracle. A second run relocated the
reference weights into the final 128 slots of a 384-expert arena, exercising IDs
through 383. Both runs passed. Rust ABI layout validation also passed.

The width-192 variant needs 1,614,880 scratch bytes at capacity 1 and 25,722,256
at capacity 16. These are per-execution kernel arenas, excluding weights, input,
routing, reduction, other lanes and graph/runtime memory. No throughput or
startup improvement is claimed. Synthetic expert numerics do not establish
full-model quality or equivalence to four separately quantized TP4 shards.

Reproduce the export with `python/tools/export_b12x_v41_slices_aot.py --role
rtx_backbone --rows 1,16 --width 192 --output-dir DIR`, compile the existing
native expert bridge, route reducer and packer with the exported objects for
SM120, then run `python/tools/qualify_v41_slice_native.py --native-lib LIB
--width 192 --capacities 1,16 --full-backbone` on the selected RTX.

[Machine-readable evidence](phase1-rtx-full-expert-probe.json) records artifacts,
geometry, checked cases and scope. Raw logs and artifacts are under
`/tmp/ds41-rtx-full-experts`. Remaining work includes coexistence with dSpark in
the coordinator library, weight ownership and memory planning, local dispatch,
larger capacities, official-weight checks, startup and serving qualification.
