#!/usr/bin/env python3
"""Audit each immutable EXL3 projection block as a live run closes it.

The watcher is deliberately outside the quantizer process.  It reads the
content-bound plan and commit journal through ``docker exec``, then launches
the image's independent block auditor.  Reports must live outside run state so
they cannot change resume identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import PurePosixPath


AUDITOR = "/opt/ds41rt/quantization/validate_projection_checkpoint_block.py"
PLAN_NAME = "ds41rt-gptqmodel-plan.json"
JOURNAL_NAME = ".ds41rt-exl3-error-journal.jsonl"
AUDIT_SCHEMA = "ds41rt-exl3-live-projection-block-audit-v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class Block:
    namespace: str
    logical_layer: int
    commit_threshold: int


def docker_exec(container: str, *command: str, capture: bool = True) -> str:
    result = subprocess.run(
        ["docker", "exec", container, *command],
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else ""


def container_running(container: str) -> bool:
    result = subprocess.run(
        [
            "docker",
            "inspect",
            container,
            "--format",
            "{{.State.Running}}",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def counts_for_scope(
    base_blocks: int,
    mtp_blocks: int,
    scope: object,
) -> tuple[int, int]:
    if scope is None:
        return base_blocks, mtp_blocks
    if scope == "mtp-routed-experts-only":
        return 0, mtp_blocks
    raise ValueError(f"unsupported quantization plan scope: {scope!r}")


def load_geometry(container: str, run_state: str) -> tuple[int, int, int, str]:
    plan_path = str(PurePosixPath(run_state) / PLAN_NAME)
    program = """
import hashlib, json, sys
plan = json.load(open(sys.argv[1], encoding="utf-8"))
digest = plan.get("plan_sha256")
body = {key: value for key, value in plan.items() if key != "plan_sha256"}
canonical = json.dumps(
    body,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
).encode()
if not isinstance(digest, str) or hashlib.sha256(canonical).hexdigest() != digest:
    raise SystemExit("saved plan digest is invalid")
geometry = plan["source"]["geometry"]
print(json.dumps({
    "base_blocks": geometry["num_hidden_layers"],
    "mtp_blocks": len(geometry["dspark_target_layer_ids"]),
    "scope": plan.get("scope"),
    "projections_per_block": (
        geometry["n_routed_experts"] * 3
    ),
    "plan_sha256": digest,
}))
"""
    payload = json.loads(
        docker_exec(container, "python", "-c", program, plan_path)
    )
    base_blocks, mtp_blocks = counts_for_scope(
        int(payload["base_blocks"]),
        int(payload["mtp_blocks"]),
        payload.get("scope"),
    )
    values = (
        base_blocks,
        mtp_blocks,
        int(payload["projections_per_block"]),
        payload["plan_sha256"],
    )
    if (
        values[0] < 0
        or values[1] <= 0
        or values[2] <= 0
        or not isinstance(values[3], str)
        or SHA256_RE.fullmatch(values[3]) is None
    ):
        raise ValueError(f"invalid block geometry: {values!r}")
    return values


def blocks(
    base_blocks: int, mtp_blocks: int, projections_per_block: int
) -> list[Block]:
    result: list[Block] = []
    ordinal = 0
    for namespace, count in (("base", base_blocks), ("mtp", mtp_blocks)):
        for logical_layer in range(count):
            ordinal += 1
            result.append(
                Block(
                    namespace=namespace,
                    logical_layer=logical_layer,
                    commit_threshold=ordinal * projections_per_block,
                )
            )
    return result


def committed_projections(container: str, run_state: str) -> int:
    journal = str(PurePosixPath(run_state) / JOURNAL_NAME)
    output = docker_exec(container, "wc", "-l", journal)
    return int(output.split()[0])


def journal_ready(container: str, run_state: str) -> bool:
    """Return true only after the quantizer has initialized its durable journal."""

    journal = str(PurePosixPath(run_state) / JOURNAL_NAME)
    result = subprocess.run(
        ["docker", "exec", container, "test", "-f", journal],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def complete_report_matches(
    value: object,
    block: Block,
    *,
    plan_sha256: str,
    projections_per_block: int,
) -> bool:
    if not isinstance(value, dict):
        return False
    digest = value.get("report_sha256")
    body = {key: item for key, item in value.items() if key != "report_sha256"}
    try:
        canonical = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError):
        return False
    return (
        isinstance(digest, str)
        and SHA256_RE.fullmatch(digest) is not None
        and hashlib.sha256(canonical).hexdigest() == digest
        and value.get("schema") == AUDIT_SCHEMA
        and value.get("status") == "complete"
        and value.get("block_namespace") == block.namespace
        and value.get("logical_layer") == block.logical_layer
        and value.get("plan_sha256") == plan_sha256
        and value.get("projection_count") == projections_per_block
        and value.get("expected_projection_count") == projections_per_block
        and value.get("missing_projection_count") == 0
        and value.get("complete_expert_families")
        == projections_per_block // 3
    )


def report_complete(
    container: str,
    report: str,
    block: Block,
    *,
    plan_sha256: str,
    projections_per_block: int,
) -> bool:
    result = subprocess.run(
        ["docker", "exec", container, "cat", report],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        return False
    try:
        value = json.loads(result.stdout)
    except (UnicodeError, json.JSONDecodeError):
        return False
    return complete_report_matches(
        value,
        block,
        plan_sha256=plan_sha256,
        projections_per_block=projections_per_block,
    )


def report_path(report_dir: str, block: Block) -> str:
    name = (
        f"{block.namespace}-layer-{block.logical_layer}"
        "-projection-audit.json"
    )
    return str(PurePosixPath(report_dir) / name)


def audit(container: str, run_state: str, report: str, block: Block) -> None:
    docker_exec(
        container,
        "python",
        AUDITOR,
        "--run-state",
        run_state,
        "--block-namespace",
        block.namespace,
        "--logical-layer",
        str(block.logical_layer),
        "--output",
        report,
        capture=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--run-state", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument(
        "--start-ordinal",
        type=int,
        default=0,
        help="Skip earlier block ordinals after verifying their reports",
    )
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.start_ordinal < 0:
        parser.error("--start-ordinal must be nonnegative")
    return args


def main() -> int:
    args = parse_args()
    base_count, mtp_count, projection_count, plan_sha256 = load_geometry(
        args.container, args.run_state
    )
    work = blocks(base_count, mtp_count, projection_count)
    if args.start_ordinal > len(work):
        raise ValueError(
            f"start ordinal {args.start_ordinal} exceeds {len(work)} blocks"
        )

    print(
        "watching "
        f"{base_count} base + {mtp_count} mtp blocks, "
        f"{projection_count} projections/block",
        flush=True,
    )
    while not journal_ready(args.container, args.run_state):
        if not container_running(args.container):
            raise RuntimeError(
                "container stopped before initializing the projection journal"
            )
        print("waiting for projection journal initialization", flush=True)
        time.sleep(args.poll_seconds)
    for ordinal, block in enumerate(work, start=1):
        report = report_path(args.report_dir, block)
        if ordinal <= args.start_ordinal:
            if not report_complete(
                args.container,
                report,
                block,
                plan_sha256=plan_sha256,
                projections_per_block=projection_count,
            ):
                raise RuntimeError(
                    f"missing complete report for skipped block {ordinal}: {report}"
                )
            continue
        if report_complete(
            args.container,
            report,
            block,
            plan_sha256=plan_sha256,
            projections_per_block=projection_count,
        ):
            print(f"block {ordinal}/{len(work)} already audited: {report}", flush=True)
            continue

        while True:
            committed = committed_projections(args.container, args.run_state)
            if committed >= block.commit_threshold:
                break
            if not container_running(args.container):
                raise RuntimeError(
                    f"container stopped at {committed}/{block.commit_threshold} "
                    f"commits before {block.namespace} layer {block.logical_layer}"
                )
            print(
                f"waiting for block {ordinal}/{len(work)}: "
                f"{committed}/{block.commit_threshold} commits",
                flush=True,
            )
            time.sleep(args.poll_seconds)

        print(
            f"auditing {block.namespace} layer {block.logical_layer} "
            f"at {committed} commits",
            flush=True,
        )
        audit(args.container, args.run_state, report, block)
        if not report_complete(
            args.container,
            report,
            block,
            plan_sha256=plan_sha256,
            projections_per_block=projection_count,
        ):
            raise RuntimeError(f"auditor did not publish a complete report: {report}")
        print(f"audit complete: {report}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("quantization audit watcher stopped", file=sys.stderr)
        raise SystemExit(130) from None
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"quantization audit watcher failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
