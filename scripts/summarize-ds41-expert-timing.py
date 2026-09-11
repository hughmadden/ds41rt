#!/usr/bin/env python3
"""Summarize Spark GPU/staging, RoCE boundaries and coordinator stage timings."""
import argparse
import json
import math
import re
import statistics
from pathlib import Path


def summarize(paths):
    groups = {}
    for path in paths:
        for line in path.read_text().splitlines():
            line = re.sub(r"\x1b\[[0-9;]*m", "", line)
            if "native expert execution" in line:
                kind = "gpu_and_staging"
            elif "protocol_v2_verbs_persistent_server_roundtrip_timing " in line:
                kind = "roce_server_boundary"
            elif "protocol_v2_expert_server_roundtrip_timing request_id=" in line:
                kind = "server_boundary"
            else:
                stage = re.search(r"\btarget (attention stages|collection|experts|layer|step) ", line)
                if not stage:
                    continue
                kind = "coordinator_" + stage[1].replace(" ", "_")
            fields = {k: float(v) for k, v in re.findall(
                r"\b(\w+)=([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)", line)}
            histogram = re.search(r"expert_rows_histogram=\[([0-9, ]+)\]", line)
            if histogram:
                counts = [int(v) for v in histogram[1].split(",")]
                tail_routes = int(fields["expert_rows_tail_routes"])
                if (len(counts) != 17 or sum(counts) != fields["active_experts"]
                        or any(v and i+1 > fields["rows"] for i,v in enumerate(counts))
                        or not 17*counts[-1] <= tail_routes <= fields["rows"]*counts[-1]
                        or sum((i+1)*v for i,v in enumerate(counts[:16])) + tail_routes != fields["rows"]*6):
                    raise ValueError(f"inconsistent expert row histogram in {path}")
                fields["_expert_rows_histogram"] = counts
            rows = int(fields["rows"])
            key = (str(path), kind, rows)
            groups.setdefault(key, []).append(fields)
    if not groups:
        raise ValueError("no native expert or server-boundary timing records found")
    output = []
    for (source, kind, rows), records in sorted(groups.items()):
        metrics = {}
        for name in sorted(set.intersection(*(set(r) for r in records))):
            if not (name.endswith(("_us", "_ms", "_bytes"))
                    or name in {"active_experts", "max_expert_rows"}):
                continue
            values = sorted(r[name] for r in records)
            if not all(math.isfinite(v) and v >= 0 for v in values):
                raise ValueError(f"invalid metric {name} in {source}")
            metrics[name] = {"mean": statistics.fmean(values),
                             "median": statistics.median(values),
                             "p95": values[math.ceil(len(values) * 0.95) - 1],
                             "max": values[-1]}
        group = {"source": source, "kind": kind, "rows": rows,
                 "samples": len(records), "metrics": metrics}
        histograms = [r for r in records if "_expert_rows_histogram" in r]
        if histograms:
            counts = [sum(r["_expert_rows_histogram"][i] for r in histograms) for i in range(17)]
            total = sum(counts)
            routed = sum(r["rows"]*6 for r in histograms)
            distribution = []
            for i, count in enumerate(counts):
                routes = ((i+1)*count if i < 16 else
                          sum(r["expert_rows_tail_routes"] for r in histograms))
                distribution.append({"expert_rows": i+1 if i < 16 else "17+",
                                     "experts": count, "routes": int(routes),
                                     "expert_fraction": count/total,
                                     "route_fraction": routes/routed})
            group["expert_row_distribution"] = distribution
            group["expert_row_distribution_samples"] = len(histograms)
        output.append(group)
    return {"scope": "Instrumented development workload. GPU event intervals separate expert execution and compaction; host upload/download exclude network transfer. Unique-expert packed bytes exclude repeated reads and are not measured DRAM traffic. Server callback includes staging and, for queued workers, queueing. Coordinator expert phase includes routing, shared FFN and response collection; shared FFN overlaps the dispatched remote request. Receive time includes waiting for remote compute and client handling. Coordinator upload_us measures host copy-call duration (staging/enqueue for async uploads), and reduce_us includes the final stream drain. Dispatch measures enqueue, not NIC send completion. No interval isolates pure link latency. Component medians are not additive; nested coordinator stage groups must not be summed together.",
            "groups": output}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.logs)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Summarized {len(result['groups'])} timing groups.")
