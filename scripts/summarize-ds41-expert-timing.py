#!/usr/bin/env python3
"""Summarize opt-in Spark GPU/staging and ProtocolV2 transport timings."""
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
            elif "protocol_v2_expert_server_roundtrip_timing request_id=" in line:
                kind = "server_boundary"
            else:
                continue
            fields = {k: float(v) for k, v in re.findall(
                r"\b(\w+)=([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)", line)}
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
            metrics[name] = {"median": statistics.median(values),
                             "p95": values[math.ceil(len(values) * 0.95) - 1],
                             "max": values[-1]}
        output.append({"source": source, "kind": kind, "rows": rows,
                       "samples": len(records), "metrics": metrics})
    return {"scope": "Instrumented development workload. GPU event intervals separate expert execution and compaction; host upload/download exclude socket transfer. Unique-expert packed bytes exclude repeated reads and are not measured DRAM traffic. Server execute time also includes queueing/staging; write time is host socket handling, not pure link time.",
            "groups": output}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.logs)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Summarized {len(result['groups'])} timing groups.")
