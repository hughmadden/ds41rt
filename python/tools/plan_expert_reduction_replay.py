#!/usr/bin/env python3
"""Build exact-M, full-layer replay chains from a real expert-route bank."""

from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TextIO

from bench_real_full_mtp_acceptance import WEIGHTED_CASE_IDS


DIAGNOSTIC_CASE_IDS = ("count", "repeat")


@dataclass(frozen=True)
class ReplayGeometry:
    model: str
    hidden_size: int
    top_k: int
    routed_experts: int
    quantization_recipe: str
    routed_layers: tuple[int, ...]


@dataclass(frozen=True)
class NaturalCycle:
    output_id: str
    case: str
    cycle: int
    physical_m: int
    routes_by_layer: dict[int, tuple[tuple[int, ...], ...]]


@dataclass(frozen=True)
class CyclePiece:
    cycle: NaturalCycle
    row_start: int
    row_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--physical-ms", default="8,12,16,20,24,28,32,40,48,64"
    )
    parser.add_argument("--semantic-chains", type=int, default=64)
    parser.add_argument("--diagnostic-chains", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20_260_726)
    return parser.parse_args()


def input_reader(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def parse_physical_ms(raw: str) -> list[int]:
    values = [int(value) for value in raw.split(",") if value.strip()]
    if not values or any(value < 1 for value in values):
        raise ValueError("--physical-ms must contain positive integers")
    return list(dict.fromkeys(values))


def geometry_from_manifest(manifest: dict[str, Any]) -> ReplayGeometry:
    if manifest.get("schema") != "ds41rt-expert-route-bank-v2":
        raise ValueError(
            "route bank must use ds41rt-expert-route-bank-v2 with explicit "
            "DeepSeek model geometry"
        )
    geometry = ReplayGeometry(
        model=str(manifest.get("model", "")),
        hidden_size=int(manifest.get("hidden_size", 0)),
        top_k=int(manifest.get("top_k", 0)),
        routed_experts=int(manifest.get("routed_experts", 0)),
        quantization_recipe=str(manifest.get("quantization_recipe", "")),
        routed_layers=tuple(int(layer) for layer in manifest.get("routed_layers", ())),
    )
    if not geometry.model:
        raise ValueError("route bank model must be nonempty")
    if geometry.hidden_size < 1 or geometry.hidden_size % 16:
        raise ValueError("route bank hidden_size must be a positive multiple of 16")
    if geometry.top_k < 1:
        raise ValueError("route bank top_k must be positive")
    if geometry.routed_experts < geometry.top_k:
        raise ValueError("route bank routed_experts must be at least top_k")
    if (
        not geometry.routed_layers
        or len(set(geometry.routed_layers)) != len(geometry.routed_layers)
        or any(layer < 0 for layer in geometry.routed_layers)
        or tuple(sorted(geometry.routed_layers)) != geometry.routed_layers
    ):
        raise ValueError("route bank routed_layers must be unique and increasing")
    if not geometry.quantization_recipe:
        raise ValueError("route bank quantization_recipe must be nonempty")
    return geometry


def load_cycles(
    path: Path,
) -> tuple[dict[str, Any], ReplayGeometry, list[NaturalCycle]]:
    manifest = None
    fragments: dict[
        tuple[str, str, int, int], dict[int, tuple[tuple[int, ...], ...]]
    ] = defaultdict(dict)
    with input_reader(path) as source:
        for line_number, line in enumerate(source, start=1):
            record = json.loads(line)
            kind = record.get("record")
            if kind == "manifest":
                if manifest is not None:
                    raise ValueError("route bank contains multiple manifests")
                manifest = record
            elif kind == "fragment":
                key = (
                    record["output_id"],
                    record["case"],
                    int(record["cycle"]),
                    int(record["physical_m"]),
                )
                layer_id = int(record["layer_id"])
                routes = tuple(
                    tuple(int(expert) for expert in row) for row in record["routes"]
                )
                if layer_id in fragments[key]:
                    raise ValueError(
                        f"line {line_number} duplicates layer {layer_id} for {key}"
                    )
                fragments[key][layer_id] = routes
    if manifest is None:
        raise ValueError("route bank has no manifest")
    geometry = geometry_from_manifest(manifest)

    cycles = []
    for (output_id, case, cycle, physical_m), routes_by_layer in fragments.items():
        if tuple(sorted(routes_by_layer)) != geometry.routed_layers:
            raise ValueError(
                f"{output_id} cycle {cycle} does not cover all routed layers"
            )
        for layer_id, routes in routes_by_layer.items():
            if len(routes) != physical_m:
                raise ValueError(
                    f"{output_id} cycle {cycle} layer {layer_id} has "
                    f"{len(routes)} rows, expected {physical_m}"
                )
            if any(len(row) != geometry.top_k for row in routes):
                raise ValueError(
                    f"{output_id} cycle {cycle} layer {layer_id} is not "
                    f"top-{geometry.top_k}"
                )
            if any(len(set(row)) != len(row) for row in routes):
                raise ValueError(
                    f"{output_id} cycle {cycle} layer {layer_id} repeats an "
                    "expert ID within a row"
                )
            if any(
                expert < 0 or expert >= geometry.routed_experts
                for row in routes
                for expert in row
            ):
                raise ValueError(
                    f"{output_id} cycle {cycle} layer {layer_id} contains an "
                    "out-of-range expert ID"
                )
        cycles.append(
            NaturalCycle(
                output_id=output_id,
                case=case,
                cycle=cycle,
                physical_m=physical_m,
                routes_by_layer=routes_by_layer,
            )
        )
    if not cycles:
        raise ValueError("route bank has no complete cycles")
    return manifest, geometry, cycles


def weighted_case_ids(cycles: Iterable[NaturalCycle], requested: Iterable[str]) -> list[str]:
    available = {cycle.case for cycle in cycles}
    selected = [case for case in requested if case in available]
    if not selected:
        raise ValueError(f"none of the requested cases are present: {sorted(available)}")
    return selected


def choose_cycle(
    rng: random.Random,
    cycles_by_case: dict[str, list[NaturalCycle]],
    case_weights: list[str],
    maximum_m: int | None,
    excluded: set[tuple[str, int]],
) -> NaturalCycle:
    eligible_cases = []
    for case in case_weights:
        if any(
            (maximum_m is None or cycle.physical_m <= maximum_m)
            and (cycle.output_id, cycle.cycle) not in excluded
            for cycle in cycles_by_case[case]
        ):
            eligible_cases.append(case)
    if not eligible_cases:
        excluded.clear()
        for case in case_weights:
            if any(
                maximum_m is None or cycle.physical_m <= maximum_m
                for cycle in cycles_by_case[case]
            ):
                eligible_cases.append(case)
    if not eligible_cases:
        raise ValueError(f"no natural cycle fits maximum M={maximum_m}")
    case = rng.choice(eligible_cases)
    eligible = [
        cycle
        for cycle in cycles_by_case[case]
        if (maximum_m is None or cycle.physical_m <= maximum_m)
        and (cycle.output_id, cycle.cycle) not in excluded
    ]
    return rng.choice(eligible)


def select_pieces(
    rng: random.Random,
    physical_m: int,
    cycles_by_case: dict[str, list[NaturalCycle]],
    case_weights: list[str],
) -> list[CyclePiece]:
    remaining = physical_m
    pieces = []
    used = set()
    while remaining:
        fitting_exists = any(
            cycle.physical_m <= remaining
            for case in case_weights
            for cycle in cycles_by_case[case]
        )
        cycle = choose_cycle(
            rng,
            cycles_by_case,
            case_weights,
            remaining if fitting_exists else None,
            used,
        )
        row_count = min(cycle.physical_m, remaining)
        row_start = (
            0
            if row_count == cycle.physical_m
            else rng.randrange(cycle.physical_m - row_count + 1)
        )
        pieces.append(CyclePiece(cycle, row_start, row_count))
        used.add((cycle.output_id, cycle.cycle))
        remaining -= row_count
    return pieces


def chain_record(
    geometry: ReplayGeometry,
    chain_id: str,
    cohort: str,
    physical_m: int,
    pieces: list[CyclePiece],
) -> dict[str, Any]:
    layer_routes = []
    unique_experts = []
    for layer_id in geometry.routed_layers:
        routes = tuple(
            row
            for piece in pieces
            for row in piece.cycle.routes_by_layer[layer_id][
                piece.row_start : piece.row_start + piece.row_count
            ]
        )
        if len(routes) != physical_m:
            raise AssertionError("exact-M planner produced the wrong row count")
        unique_experts.append(len({expert for row in routes for expert in row}))
        layer_routes.append({"layer_id": layer_id, "routes": routes})
    return {
        "record": "chain",
        "chain_id": chain_id,
        "cohort": cohort,
        "physical_m": physical_m,
        "pieces": [
            {
                "output_id": piece.cycle.output_id,
                "case": piece.cycle.case,
                "cycle": piece.cycle.cycle,
                "natural_m": piece.cycle.physical_m,
                "row_start": piece.row_start,
                "row_count": piece.row_count,
            }
            for piece in pieces
        ],
        "case_rows": dict(
            Counter(
                piece.cycle.case
                for piece in pieces
                for _ in range(piece.row_count)
            )
        ),
        "mean_unique_experts_per_layer": sum(unique_experts) / len(unique_experts),
        "min_unique_experts_per_layer": min(unique_experts),
        "max_unique_experts_per_layer": max(unique_experts),
        "layers": layer_routes,
    }


def write_record(output: TextIO, record: dict[str, Any]) -> None:
    output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    output.write("\n")


def main() -> None:
    args = parse_args()
    physical_ms = parse_physical_ms(args.physical_ms)
    if args.semantic_chains < 1 or args.diagnostic_chains < 0:
        raise SystemExit("chain counts must be non-negative and semantic chains positive")
    if not args.route_bank.is_file():
        raise SystemExit(f"route bank does not exist: {args.route_bank}")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing replay plan: {args.output}")

    bank_manifest, geometry, cycles = load_cycles(args.route_bank)
    cycles_by_case: dict[str, list[NaturalCycle]] = defaultdict(list)
    for cycle in cycles:
        cycles_by_case[cycle.case].append(cycle)
    semantic_weights = weighted_case_ids(cycles, WEIGHTED_CASE_IDS)
    diagnostic_weights = weighted_case_ids(cycles, DIAGNOSTIC_CASE_IDS)
    rng = random.Random(args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    records = []
    for physical_m in physical_ms:
        for index in range(args.semantic_chains):
            pieces = select_pieces(
                rng, physical_m, cycles_by_case, semantic_weights
            )
            records.append(
                chain_record(
                    geometry,
                    f"semantic-m{physical_m:03d}-{index:04d}",
                    "semantic",
                    physical_m,
                    pieces,
                )
            )
        for index in range(args.diagnostic_chains):
            pieces = select_pieces(
                rng, physical_m, cycles_by_case, diagnostic_weights
            )
            records.append(
                chain_record(
                    geometry,
                    f"diagnostic-m{physical_m:03d}-{index:04d}",
                    "diagnostic",
                    physical_m,
                    pieces,
                )
            )

    try:
        with temporary.open("w", encoding="utf-8") as output:
            write_record(
                output,
                {
                    "record": "manifest",
                    "schema": "ds41rt-expert-reduction-replay-plan-v2",
                    "route_bank_schema": bank_manifest["schema"],
                    "route_bank": str(args.route_bank.resolve()),
                    "model": geometry.model,
                    "hidden_size": geometry.hidden_size,
                    "top_k": geometry.top_k,
                    "routed_experts": geometry.routed_experts,
                    "quantization_recipe": geometry.quantization_recipe,
                    "routed_layers": geometry.routed_layers,
                    "seed": args.seed,
                    "physical_ms": physical_ms,
                    "semantic_case_weights": semantic_weights,
                    "diagnostic_case_weights": diagnostic_weights,
                    "semantic_chains_per_m": args.semantic_chains,
                    "diagnostic_chains_per_m": args.diagnostic_chains,
                },
            )
            for record in records:
                write_record(output, record)
        temporary.replace(args.output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    print(
        json.dumps(
            {
                "replay_plan": str(args.output),
                "bytes": args.output.stat().st_size,
                "chains": len(records),
                "physical_ms": physical_ms,
            }
        )
    )


if __name__ == "__main__":
    main()
