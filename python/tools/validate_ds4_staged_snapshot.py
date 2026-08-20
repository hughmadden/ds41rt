#!/usr/bin/env python3
"""Validate one immutable staged DS4RT EXL3 snapshot for serving."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from validate_ds4_flash_generation_ab import validate_staged_exl3_checkpoint


CALIBRATED_FLASH_K2_MODEL_ID = (
    "wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1"
)
CALIBRATED_FLASH_K2_REVISION = "68eaca43e99bfbfd697a5559c7796b983deb38f8"
CALIBRATED_FLASH_K2_TOTAL_SIZE = 83_963_579_128
CALIBRATED_FLASH_K2_WEIGHT_COUNT = 142_973
CALIBRATED_FLASH_K2_TENSOR_STORAGE_COUNT = 35_328
CALIBRATED_FLASH_K2_METADATA = {
    "config.json": (
        2_358,
        "6c19e963aa846683b7cf176d1beab68d0c899e4e",
    ),
    "model.safetensors.index.json": (
        12_560_799,
        "8d7d04071872384b5919698751ed9b867b2a9d557c85d312783420d1ebf2def1",
    ),
    "quantize_config.json": (
        28_374_828,
        "301a32c8887f5374f12cb0a34c13ce080abf09ae06cdd87d37db50afe5ec1a9a",
    ),
}
CALIBRATED_FLASH_K2_SHARDS = {
    "model-00001-of-00010.safetensors": (
        7_787_054_620,
        "41cb753d03ca3c64d77092ac3b8a8967bca45828b0c95c26149f2b1014e54bcf",
    ),
    "model-00002-of-00010.safetensors": (
        8_589_567_900,
        "f2c0e64f0e93bd38249e7af39853409d42eba28faf4b0507a2d4cbfc6ec1cb1d",
    ),
    "model-00003-of-00010.safetensors": (
        8_591_580_656,
        "03dcdf57f6eed221314313d7f4393162e69aed869e6ce3111a524fa30f12543e",
    ),
    "model-00004-of-00010.safetensors": (
        8_591_576_984,
        "df5a78625ec22212d5a46de92cfba7feaa3ea7eca182b6d5c669697f506192e8",
    ),
    "model-00005-of-00010.safetensors": (
        8_591_580_272,
        "133cf005668f0ccf7138909893674c4b3a6c892c5fcbae194322298d61ed803e",
    ),
    "model-00006-of-00010.safetensors": (
        8_591_577_584,
        "a8028360dd7dd5e3bc17898a7ef10207a7603eaa28f296d7803156004f953b82",
    ),
    "model-00007-of-00010.safetensors": (
        8_591_580_344,
        "709058ebde414b629793ff88ae10f4f405171011c71b789e92a3382eefc10b00",
    ),
    "model-00008-of-00010.safetensors": (
        8_591_576_920,
        "6d93ee918db07cbd3e4c1bcc6767e13f9824b481e1b04ac53bf7ff5e346f45b9",
    ),
    "model-00009-of-00010.safetensors": (
        8_591_570_240,
        "be882a3687b954c8dd2dec5427cc0e5a6063c5d669eb06d739dabd4bc7dd4310",
    ),
    "model-00010-of-00010.safetensors": (
        7_462_664_344,
        "881c0f41e4d25cadcb62a7d66293de2f18df40b836b285c1edc57ae23e8e46aa",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--allow-development-unqualified",
        action="store_true",
        help="accept an explicitly marked development-only snapshot for WIP serving",
    )
    parser.add_argument(
        "--startup-contract-only",
        action="store_true",
        help=(
            "validate the immutable staged identity and serving qualification only; "
            "do not repeat the quantization-publication provenance audit"
        ),
    )
    return parser.parse_args()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read qualified public {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"qualified public {label} is not a JSON object")
    return value


def _validate_hf_blob(
    snapshot: Path,
    model_root: Path,
    name: str,
    expected_size: int,
    expected_blob: str,
) -> None:
    path = snapshot / name
    if not path.is_symlink():
        raise ValueError(f"qualified public snapshot entry is not an HF blob link: {path}")
    try:
        target = path.resolve(strict=True)
        size = target.stat().st_size
    except OSError as error:
        raise ValueError(f"qualified public snapshot entry is unresolved: {path}") from error
    if target.parent != model_root / "blobs" or target.name != expected_blob:
        raise ValueError(f"qualified public snapshot entry targets the wrong blob: {path}")
    if size != expected_size:
        raise ValueError(f"qualified public snapshot entry has the wrong size: {path}")


def validate_qualified_public_exl3_checkpoint(
    checkpoint: Path,
    model_id: str,
    revision: str,
) -> dict[str, Any]:
    """Validate the exact public artifact without replaying publication audits.

    The immutable Hugging Face commit and its content-addressed blob names are
    the startup trust root. This checks the serving contract and shard geometry,
    but deliberately does not hash 84 GB or require private quantization files.
    """

    if (
        model_id != CALIBRATED_FLASH_K2_MODEL_ID
        or revision != CALIBRATED_FLASH_K2_REVISION
    ):
        raise ValueError("snapshot is not a qualified public EXL3 release")
    snapshot = checkpoint.expanduser().resolve(strict=True)
    if snapshot.parent.name != "snapshots" or snapshot.name != revision:
        raise ValueError(
            "qualified public EXL3 checkpoint must use snapshots/<revision>"
        )
    model_root = snapshot.parent.parent
    expected_root_name = f"models--{model_id.replace('/', '--')}"
    if model_root.name != expected_root_name:
        raise ValueError("qualified public snapshot path differs from its model ID")

    for name, (size, blob) in {
        **CALIBRATED_FLASH_K2_METADATA,
        **CALIBRATED_FLASH_K2_SHARDS,
    }.items():
        _validate_hf_blob(snapshot, model_root, name, size, blob)

    config = _read_json(snapshot / "config.json", "config.json")
    quant = config.get("quantization_config")
    if not isinstance(quant, dict):
        raise ValueError("qualified public config has no EXL3 recipe")
    expected_quant = {
        "bits": 2.0,
        "checkpoint_format": "exl3",
        "codebook": "mcg",
        "format": "exl3",
        "method": "exl3",
        "quant_method": "exl3",
    }
    if (
        config.get("model_type") != "deepseek_v4"
        or config.get("hidden_size") != 4096
        or config.get("num_hidden_layers") != 43
        or config.get("n_routed_experts") != 256
        or config.get("num_experts_per_tok") != 6
        or config.get("dspark_target_layer_ids") != [40, 41, 42]
        or any(quant.get(key) != value for key, value in expected_quant.items())
    ):
        raise ValueError("qualified public config differs from the calibrated K2 contract")

    quantize_config = _read_json(
        snapshot / "quantize_config.json", "quantize_config.json"
    )
    tensor_storage = quantize_config.get("tensor_storage")
    if (
        any(
            quantize_config.get(key) != value
            for key, value in expected_quant.items()
        )
        or not isinstance(tensor_storage, dict)
        or len(tensor_storage) != CALIBRATED_FLASH_K2_TENSOR_STORAGE_COUNT
    ):
        raise ValueError(
            "qualified public quantize_config differs from the calibrated K2 contract"
        )

    index = _read_json(
        snapshot / "model.safetensors.index.json",
        "model.safetensors.index.json",
    )
    metadata = index.get("metadata")
    weight_map = index.get("weight_map")
    if (
        not isinstance(metadata, dict)
        or metadata.get("total_size") != CALIBRATED_FLASH_K2_TOTAL_SIZE
        or not isinstance(weight_map, dict)
        or len(weight_map) != CALIBRATED_FLASH_K2_WEIGHT_COUNT
        or set(weight_map.values()) != set(CALIBRATED_FLASH_K2_SHARDS)
    ):
        raise ValueError("qualified public weight index has the wrong shard geometry")

    return {
        "schema": "ds4rt-qualified-public-exl3-snapshot-v1",
        "model_id": model_id,
        "revision": revision,
        "qualification_status": "production-qualified",
        "quantization": "exl3-k2-calibrated",
        "weight_tensors": len(weight_map),
        "weight_bytes": CALIBRATED_FLASH_K2_TOTAL_SIZE,
        "shards": len(CALIBRATED_FLASH_K2_SHARDS),
    }


def validate_public_exl3_serving_checkpoint(
    checkpoint: Path,
    model_id: str,
    revision: str,
) -> dict[str, Any]:
    """Validate only the files a standard public HF EXL3 snapshot must serve.

    Calibration ledgers, quantization-worker reports, and the local DS4RT
    staging manifest qualify an export; they are deliberately not runtime
    dependencies.  The pinned Hugging Face revision plus the loader's full
    tensor-contract validation is the serving trust boundary.
    """

    snapshot = checkpoint.expanduser().resolve(strict=True)
    if snapshot.parent.name != "snapshots" or snapshot.name != revision:
        raise ValueError("public EXL3 checkpoint must use snapshots/<revision>")
    model_root = snapshot.parent.parent
    expected_root_name = f"models--{model_id.replace('/', '--')}"
    if model_root.name != expected_root_name:
        raise ValueError("public EXL3 snapshot path differs from its model ID")

    required = (
        "config.json",
        "model.safetensors.index.json",
        "quantize_config.json",
    )
    for name in required:
        path = snapshot / name
        if not path.is_file():
            raise ValueError(f"public EXL3 snapshot is missing {name}")
        try:
            path.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"public EXL3 snapshot has unresolved {name}") from error

    config = _read_json(snapshot / "config.json", "config.json")
    quant = config.get("quantization_config")
    if config.get("model_type") != "deepseek_v4" or not isinstance(quant, dict):
        raise ValueError("public EXL3 config is not a DeepSeek V4 quantized model")
    recipe_fields = ("quant_method", "method", "format", "checkpoint_format")
    if any(str(quant.get(field, "")).lower() != "exl3" for field in recipe_fields):
        raise ValueError("public EXL3 config has an unsupported quantization recipe")
    bits = quant.get("bits")
    if isinstance(bits, bool) or not isinstance(bits, (int, float)) or bits <= 0:
        raise ValueError("public EXL3 config has an invalid bitrate")

    index = _read_json(
        snapshot / "model.safetensors.index.json",
        "model.safetensors.index.json",
    )
    metadata = index.get("metadata")
    weight_map = index.get("weight_map")
    if (
        not isinstance(metadata, dict)
        or not isinstance(metadata.get("total_size"), int)
        or metadata["total_size"] <= 0
        or not isinstance(weight_map, dict)
        or not weight_map
    ):
        raise ValueError("public EXL3 weight index has invalid geometry")
    shard_values = list(weight_map.values())
    if not shard_values or any(
        not isinstance(name, str)
        or not name.endswith(".safetensors")
        or "/" in name
        or "\\" in name
        or name in {".", ".."}
        for name in shard_values
    ):
        raise ValueError("public EXL3 weight index contains an invalid shard path")
    shards = set(shard_values)
    for name in shards:
        path = snapshot / name
        if not path.is_file():
            raise ValueError(f"public EXL3 snapshot is missing indexed shard {name}")
        try:
            path.resolve(strict=True)
        except OSError as error:
            raise ValueError(
                f"public EXL3 snapshot has unresolved indexed shard {name}"
            ) from error

    return {
        "schema": "ds4rt-public-exl3-serving-snapshot-v1",
        "model_id": model_id,
        "revision": revision,
        "checkpoint": str(snapshot),
        "weight_tensors": len(weight_map),
        "weight_bytes": metadata["total_size"],
        "shards": len(shards),
        "bits": float(bits),
    }


def validate_checkpoint(
    checkpoint: Path,
    model_id: str,
    revision: str,
    *,
    allow_development_unqualified: bool,
    startup_contract_only: bool,
) -> dict[str, Any]:
    if startup_contract_only:
        return validate_public_exl3_serving_checkpoint(
            checkpoint, model_id, revision
        )
    return validate_staged_exl3_checkpoint(
        checkpoint,
        allow_development_unqualified=allow_development_unqualified,
        audit_publication=not startup_contract_only,
    )


def main() -> int:
    args = parse_args()
    identity = validate_checkpoint(
        args.checkpoint,
        args.model_id,
        args.revision,
        allow_development_unqualified=args.allow_development_unqualified,
        startup_contract_only=args.startup_contract_only,
    )
    if identity.get("model_id") != args.model_id:
        raise ValueError(
            f"staged model ID {identity.get('model_id')!r} differs from "
            f"requested {args.model_id!r}"
        )
    if identity.get("revision") != args.revision:
        raise ValueError(
            f"staged revision {identity.get('revision')!r} differs from "
            f"requested {args.revision!r}"
        )
    print(json.dumps(identity, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
