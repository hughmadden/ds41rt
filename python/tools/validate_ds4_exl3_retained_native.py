#!/usr/bin/env python3
"""Prove that every non-routed tensor in a DS4RT EXL3 artifact is byte-identical."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ds4rt_runtime.exl3_artifact_contract import (  # noqa: E402
    ARTIFACT_FILE as GPTQMODEL_ARTIFACT_FILE,
    LEDGER_MANIFEST_FILE as GPTQMODEL_LEDGER_MANIFEST_FILE,
    PLAN_FILE as GPTQMODEL_PLAN_FILE,
    RECIPE as GPTQMODEL_RECIPE,
    RECIPE_K3 as GPTQMODEL_RECIPE_K3,
    RUN_FILE as GPTQMODEL_RUN_FILE,
    is_gptqmodel_native_exl3,
    validate_gptqmodel_publication,
)
from ds4rt_runtime.exl3_quantizer import (  # noqa: E402
    EXL3_RECIPE,
    EXPERT_TENSOR_LAYOUT_CHECKPOINT_NATIVE,
    EXPERT_TENSOR_LAYOUT_GPTQMODEL,
    build_artifact_plan,
    verify_retained_native_tensors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-snapshot", type=Path, required=True)
    parser.add_argument("--exl3-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="check a resumable artifact using its unpublished index",
    )
    return parser.parse_args()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def gptqmodel_publication_identity(snapshot: Path) -> dict[str, object]:
    """Bind an external retained-native report to one immutable publication."""

    snapshot = snapshot.expanduser().resolve(strict=True)
    names = (
        "config.json",
        "model.safetensors.index.json",
        "quantize_config.json",
        GPTQMODEL_PLAN_FILE,
        GPTQMODEL_RUN_FILE,
        GPTQMODEL_ARTIFACT_FILE,
        GPTQMODEL_LEDGER_MANIFEST_FILE,
    )
    return {
        "path": str(snapshot),
        "metadata_sha256": {
            name: hash_file(snapshot / name)
            for name in names
        },
    }


def artifact_recipe_and_output(
    snapshot: Path, requested_output: Path | None
) -> tuple[str, Path]:
    snapshot = snapshot.expanduser().resolve(strict=True)
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    quant = config.get("quantization_config")
    if is_gptqmodel_native_exl3(quant):
        validate_gptqmodel_publication(
            snapshot,
            config,
            verify_all_hashes=False,
            require_canonical=True,
        )
        if requested_output is None:
            raise ValueError(
                "GPTQModel publications are immutable; --output must select an "
                "external retained-native report"
            )
        output = requested_output.expanduser().resolve()
        if output == snapshot or snapshot in output.parents:
            raise ValueError(
                "GPTQModel retained-native report must be outside the immutable artifact"
            )
        bits = quant.get("bits")
        if bits == 2 or bits == 2.0:
            recipe = GPTQMODEL_RECIPE
        elif bits == 3 or bits == 3.0:
            recipe = GPTQMODEL_RECIPE_K3
        else:
            raise ValueError(f"unsupported GPTQModel EXL3 tier {bits!r}")
        return recipe, output

    recipe = EXL3_RECIPE
    if isinstance(quant, dict):
        ds4rt = quant.get("ds4rt")
        if isinstance(ds4rt, dict) and isinstance(ds4rt.get("recipe"), str):
            recipe = ds4rt["recipe"]
    return recipe, requested_output or snapshot / "ds4rt-exl3-retained-native.json"


def retained_native_plan(native_snapshot: Path, config: dict):
    quant = config.get("quantization_config")
    gptqmodel_native = is_gptqmodel_native_exl3(quant)
    bits = int(quant["bits"]) if gptqmodel_native else 2
    return build_artifact_plan(
        native_snapshot,
        expert_tensor_layout=(
            EXPERT_TENSOR_LAYOUT_GPTQMODEL
            if gptqmodel_native
            else EXPERT_TENSOR_LAYOUT_CHECKPOINT_NATIVE
        ),
        exl3_bits=bits,
    )


def main() -> None:
    args = parse_args()
    recipe, output = artifact_recipe_and_output(args.exl3_snapshot, args.output)
    snapshot = args.exl3_snapshot.expanduser().resolve(strict=True)
    config = json.loads(
        (snapshot / "config.json").read_text(encoding="utf-8")
    )
    gptqmodel_native = is_gptqmodel_native_exl3(
        config.get("quantization_config")
    )
    plan = retained_native_plan(args.native_snapshot, config)
    report = verify_retained_native_tensors(
        plan,
        snapshot,
        allow_incomplete=args.allow_incomplete,
        recipe=recipe,
    )
    if gptqmodel_native:
        if args.allow_incomplete:
            raise ValueError(
                "GPTQModel retained-native qualification requires a complete publication"
            )
        report["gptqmodel_publication"] = gptqmodel_publication_identity(
            args.exl3_snapshot
        )
    write_json_atomic(output, report)
    summary = {key: value for key, value in report.items() if key != "tensors"}
    summary["output"] = str(output.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
