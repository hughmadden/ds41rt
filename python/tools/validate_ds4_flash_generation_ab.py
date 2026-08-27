#!/usr/bin/env python3
"""Collect and compare native/EXL3 DeepSeek V4 generation quality."""

from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import hashlib
import json
import os
import re
import sys
import tempfile
import threading
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from uuid import UUID


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ds4rt_runtime.exl3_artifact_contract import (  # noqa: E402
    ARTIFACT_FILE as GPTQMODEL_ARTIFACT_FILE,
    LEDGER_MANIFEST_FILE as GPTQMODEL_LEDGER_MANIFEST_FILE,
    PLAN_FILE as GPTQMODEL_PLAN_FILE,
    RECIPE as GPTQMODEL_RECIPE,
    RECIPE_K3 as GPTQMODEL_RECIPE_K3,
    RECIPE_MIXED_K2_K3 as GPTQMODEL_RECIPE_MIXED_K2_K3,
    RUN_FILE as GPTQMODEL_RUN_FILE,
    is_gptqmodel_native_exl3,
    validate_inline_mixed_policy,
    validate_gptqmodel_native_exl3,
    validate_gptqmodel_publication,
)


RUN_SCHEMA = "ds4rt-flash-generation-run-v6"
REFERENCE_SCHEMA = "ds4rt-flash-generation-reference-v1"
COMPARISON_SCHEMA = "ds4rt-flash-generation-ab-v6"
MATRIX_SCHEMA = "ds4rt-flash-generation-matrix-v3"
ARM_SCHEMA = "ds4rt-flash-generation-arm-v1"
RUNTIME_GATE = "ds4rt-flash-production-runtime-v5"
STAGED_ARTIFACT_SCHEMA = "ds4rt-hf-staged-snapshot-v1"
CHECKPOINT_AUDIT_SCHEMA = "ds4rt-flash-checkpoint-audit-v1"
FLASH_LAYER_COUNT = 43
NATIVE_RECIPE = "deepseek_v4_native_fp4_fp8_mixed_v1"
EXL3_V2_RECIPE = "deepseek_v4_exl3_trellis_2bpw_v2"
EXL3_V3_RECIPE = "deepseek_v4_exl3_trellis_2bpw_v3_flash_activation_pilot"
EXL3_V4_RECIPE = "deepseek_v4_exl3_trellis_2bpw_v4_flash_natural_route"
EXL3_K3_V4_RECIPE = "deepseek_v4_exl3_trellis_3bpw_v4_flash_natural_route"
EXL3_RECIPE = EXL3_V4_RECIPE
HISTORICAL_EXL3_RECIPES = frozenset((EXL3_V2_RECIPE, EXL3_V3_RECIPE))
GPTQMODEL_RECIPES = frozenset(
    (GPTQMODEL_RECIPE, GPTQMODEL_RECIPE_K3, GPTQMODEL_RECIPE_MIXED_K2_K3)
)
PRODUCTION_EXL3_RECIPES = frozenset((EXL3_V4_RECIPE, EXL3_K3_V4_RECIPE))
STAGED_EXL3_REQUIRED_FILES = frozenset(
    (
        "config.json",
        "model.safetensors.index.json",
        "quantization_config.json",
        "ds4rt-exl3-calibration.json",
        "ds4rt-exl3-quality.json",
        "ds4rt-exl3-retained-native.json",
        "ds4rt-exl3-multigpu-production.json",
        "ds4rt-exl3-error-ledger.jsonl",
        "ds4rt-exl3-error-ledger.manifest.json",
    )
)
STAGED_GPTQMODEL_REQUIRED_FILES = frozenset(
    (
        "config.json",
        "model.safetensors.index.json",
        "quantize_config.json",
        GPTQMODEL_PLAN_FILE,
        GPTQMODEL_RUN_FILE,
        GPTQMODEL_ARTIFACT_FILE,
        "ds4rt-exl3-error-ledger.jsonl",
        GPTQMODEL_LEDGER_MANIFEST_FILE,
    )
)
GPTQMODEL_QUALIFICATION_SCHEMAS = {
    "retained-native.json": "ds4rt-exl3-retained-native-integrity-v1",
    "expert-quality.json": "ds4rt-exl3-checkpoint-quality-v1",
}
DEVELOPMENT_QUALIFICATION_STATUS = "development-unqualified"
DEVELOPMENT_BLOCKERS = (
    "complete token-aware all-layer quality evidence is absent",
    "natural MTP held-out activation quality evidence is absent",
)
GPTQMODEL_DEVELOPMENT_EVIDENCE_SCHEMAS = {
    "retained-native.json": "ds4rt-exl3-retained-native-integrity-v1",
    "diagnostic-quality.json": "ds4rt-exl3-checkpoint-quality-v1",
}
GPTQMODEL_IDENTITY_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "quantize_config.json",
    GPTQMODEL_PLAN_FILE,
    GPTQMODEL_RUN_FILE,
    GPTQMODEL_ARTIFACT_FILE,
    GPTQMODEL_LEDGER_MANIFEST_FILE,
)
GPTQMODEL_QUALITY_IDENTITY_FILES = frozenset(
    (*GPTQMODEL_IDENTITY_FILES, "ds4rt-exl3-calibration.json")
)
DEFAULT_FLASH_API_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731-full"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def is_canonical_service_instance_id(value: Any) -> bool:
    try:
        parsed = UUID(value)
    except (AttributeError, TypeError, ValueError):
        return False
    return parsed.version == 4 and str(parsed) == value


@dataclass(frozen=True)
class QualityCase:
    case_id: str
    category: str
    prompt: str
    checker: str
    expected: Any
    max_tokens: int = 64


CASES = (
    QualityCase(
        "multiply",
        "arithmetic",
        "Return only the decimal integer equal to 104729 multiplied by 37.",
        "exact",
        "3874973",
    ),
    QualityCase(
        "discount-tax",
        "arithmetic",
        "A 240 dollar item is discounted by 25%, then 8% tax is applied. "
        "Return only the final amount with exactly two decimal places and no currency symbol.",
        "exact",
        "194.40",
    ),
    QualityCase(
        "modular-power",
        "arithmetic",
        "Return only the decimal integer equal to 7^123 modulo 13.",
        "exact",
        "5",
    ),
    QualityCase(
        "reduced-fraction",
        "arithmetic",
        "Add 7/12 and 5/18. Return only the reduced fraction.",
        "exact",
        "31/36",
    ),
    QualityCase(
        "integer-sort",
        "symbolic",
        "Sort these integers in ascending order: 12, -1, 3, -8, 3, 19, 0. "
        "Return only a comma-separated list with no spaces.",
        "exact",
        "-8,-1,0,3,3,12,19",
    ),
    QualityCase(
        "reverse-string",
        "symbolic",
        'Reverse the exact character sequence "Spark-TP4". Return only the result.',
        "exact",
        "4PT-krapS",
    ),
    QualityCase(
        "hexadecimal",
        "symbolic",
        "Convert decimal 65535 to uppercase hexadecimal. Return only the digits, "
        "without a prefix.",
        "exact",
        "FFFF",
    ),
    QualityCase(
        "sequence",
        "pattern",
        "Find the next term in 2, 6, 12, 20, 30. Return only the decimal integer.",
        "exact",
        "42",
    ),
    QualityCase(
        "syllogism",
        "logic",
        "All glippets are norbs. No norb is a vex. Can any glippet be a vex? "
        "Return only yes or no in lowercase.",
        "exact",
        "no",
    ),
    QualityCase(
        "reading-count",
        "reading",
        "A box has 3 red tokens. It has twice as many blue tokens as red tokens. "
        "It has one fewer green token than blue tokens. Return only the number of "
        "green tokens.",
        "exact",
        "5",
    ),
    QualityCase(
        "set-union",
        "logic",
        "Set A is {1,2,3,5,8} and set B is {2,4,5,8,10}. Return only the number "
        "of distinct elements in their union.",
        "exact",
        "7",
    ),
    QualityCase(
        "structured-json",
        "structured-output",
        'Return only this information as a JSON object: path is "src/cache.rs", '
        'operation is "replace", line_start is 41, and line_end is 47. Use exactly '
        "those four keys.",
        "json",
        {
            "path": "src/cache.rs",
            "operation": "replace",
            "line_start": 41,
            "line_end": 47,
        },
        96,
    ),
)


def suite_identity(cases: Iterable[QualityCase] = CASES) -> str:
    encoded = json.dumps(
        [asdict(case) for case in cases],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def score_output(case: QualityCase, content: str) -> tuple[bool, str | None]:
    normalized = content.strip()
    if case.checker == "exact":
        return normalized == case.expected, None
    if case.checker == "json":
        try:
            parsed = json.loads(normalized)
        except json.JSONDecodeError as error:
            return False, f"invalid JSON: {error.msg}"
        return parsed == case.expected, None
    raise ValueError(f"unsupported checker {case.checker!r}")


def completion_payload(model: str, case: QualityCase) -> bytes:
    return json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": case.prompt}],
            "temperature": 0,
            "max_tokens": case.max_tokens,
        },
        ensure_ascii=False,
    ).encode()


def request_completion(
    url: str, model: str, case: QualityCase, timeout: float
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=completion_payload(model, case),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def request_completion_group(
    url: str,
    model: str,
    cases: tuple[QualityCase, ...],
    timeout: float,
) -> list[dict[str, Any]]:
    """Start one real concurrent request group and preserve case order."""

    if not cases:
        return []
    barrier = threading.Barrier(len(cases))

    def request_one(case: QualityCase) -> dict[str, Any]:
        barrier.wait(timeout=timeout)
        return request_completion(url, model, case, timeout)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cases)) as executor:
        futures = [executor.submit(request_one, case) for case in cases]
        return [future.result() for future in futures]


def anthropic_completion_payload(
    model: str,
    case: QualityCase,
    minimum_max_tokens: int,
    thinking_mode: str,
) -> bytes:
    if minimum_max_tokens < 1:
        raise ValueError("minimum Anthropic max_tokens must be positive")
    if thinking_mode not in {"disabled", "omit"}:
        raise ValueError(f"unsupported Anthropic thinking mode {thinking_mode!r}")
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": case.prompt}],
        "temperature": 0,
        # A reasoning-enabled endpoint charges hidden work against this same
        # output budget. The omitted-thinking diagnostic therefore keeps a
        # safety floor; explicitly disabled thinking must use the byte-matched
        # local task cap instead.
        "max_tokens": (
            case.max_tokens
            if thinking_mode == "disabled"
            else max(case.max_tokens, minimum_max_tokens)
        ),
    }
    if thinking_mode == "disabled":
        # The local generation suite deliberately exercises non-thinking mode.
        # Make the official API reference a matched behavioral control.
        payload["thinking"] = {"type": "disabled"}
    return json.dumps(payload, ensure_ascii=False).encode()


def request_anthropic_completion(
    url: str,
    model: str,
    case: QualityCase,
    auth_token: str,
    timeout: float,
    minimum_max_tokens: int,
    thinking_mode: str,
) -> dict[str, Any]:
    if not auth_token:
        raise ValueError("Anthropic API auth token is empty")
    request = urllib.request.Request(
        url,
        data=anthropic_completion_payload(
            model, case, minimum_max_tokens, thinking_mode
        ),
        headers={
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": auth_token,
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def anthropic_response_content(result: dict[str, Any]) -> tuple[str, str]:
    blocks = result.get("content")
    if not isinstance(blocks, list):
        raise ValueError("Anthropic response has no content blocks")
    visible: list[str] = []
    reasoning: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            raise ValueError("Anthropic response contains a malformed content block")
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            visible.append(block["text"])
        elif block.get("type") == "thinking" and isinstance(
            block.get("thinking"), str
        ):
            reasoning.append(block["thinking"])
    return "".join(visible), "".join(reasoning)


def checkpoint_quantization_recipe(
    checkpoint: Path, *, allow_historical_control: bool = False
) -> str:
    config_path = checkpoint / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot read checkpoint config {config_path}: {error}"
        ) from error
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        raise ValueError("checkpoint has no quantization_config")
    method = quantization.get("quant_method")
    if isinstance(method, str) and method.lower() == "fp8":
        return NATIVE_RECIPE
    if isinstance(method, str) and method.lower() == "exl3":
        if is_gptqmodel_native_exl3(quantization):
            inline_mixed = quantization.get("meta", {}).get("ds4rt_inline_mixed")
            if inline_mixed is not None:
                validate_inline_mixed_policy(inline_mixed)
                if quantization.get("bits") not in (2, 2.0):
                    raise ValueError(
                        "inline mixed GPTQModel EXL3 must keep public bits at K2"
                    )
                provenance = quantization["meta"].get("ds4rt_error_ledger")
                family = (
                    provenance.get("family_join")
                    if isinstance(provenance, dict)
                    else None
                )
                family_mixed = (
                    family.get("inline_mixed") if isinstance(family, dict) else None
                )
                if (
                    not isinstance(family_mixed, dict)
                    or family_mixed.get("base") != inline_mixed
                ):
                    raise ValueError(
                        "inline mixed GPTQModel EXL3 metadata is not provenance-bound"
                    )
                validate_inline_mixed_policy(family_mixed["base"])
                if "mtp" in family_mixed:
                    validate_inline_mixed_policy(
                        family_mixed["mtp"], namespace="mtp"
                    )
                return GPTQMODEL_RECIPE_MIXED_K2_K3
            validate_gptqmodel_native_exl3(
                quantization,
                model_config=config,
            )
            return (
                GPTQMODEL_RECIPE
                if int(quantization.get("bits", 2)) == 2
                else GPTQMODEL_RECIPE_K3
            )
        # Public GPTQModel checkpoints deliberately keep the large tensor map
        # in quantize_config.json and omit private error-ledger provenance.  A
        # compact config is valid only when it is the exact external contract
        # with tensor_storage removed; do not infer the recipe from a loose
        # collection of EXL3 fields.
        quantize_path = checkpoint / "quantize_config.json"
        if quantize_path.is_file():
            try:
                external = json.loads(quantize_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"cannot read checkpoint quantization config {quantize_path}: {error}"
                ) from error
            if not isinstance(external, dict):
                raise ValueError("checkpoint quantize_config.json is not an object")
            declaration = dict(external)
            tensor_storage = declaration.pop("tensor_storage", None)
            if declaration == quantization:
                if not isinstance(tensor_storage, dict) or not tensor_storage:
                    raise ValueError(
                        "public GPTQModel EXL3 quantize_config has no tensor storage"
                    )
                bits = external.get("bits")
                if bits not in (2, 2.0, 3, 3.0):
                    raise ValueError(
                        "public GPTQModel EXL3 config is not a uniform K2/K3 contract"
                    )
                expected = {
                    "bits": float(bits),
                    "checkpoint_format": "exl3",
                    "codebook": "mcg",
                    "desc_act": False,
                    "format": "exl3",
                    "group_size": -1,
                    "method": "exl3",
                    "module_include": [
                        r"^model\.layers\.\d+\.mlp\.experts\.\d+\."
                        r"(?:gate_proj|up_proj|down_proj)$"
                    ],
                    "out_scales": "auto",
                    "quant_method": "exl3",
                }
                if any(external.get(key) != value for key, value in expected.items()):
                    raise ValueError(
                        "public GPTQModel EXL3 config is not the routed K2/K3 MCG contract"
                    )
                meta = external.get("meta")
                inline_mixed = (
                    meta.get("ds4rt_inline_mixed") if isinstance(meta, dict) else None
                )
                if inline_mixed is not None:
                    if int(bits) != 2:
                        raise ValueError(
                            "inline mixed GPTQModel EXL3 must keep public bits at K2"
                        )
                    validate_inline_mixed_policy(inline_mixed)
                    return GPTQMODEL_RECIPE_MIXED_K2_K3
                return GPTQMODEL_RECIPE if int(bits) == 2 else GPTQMODEL_RECIPE_K3
        ds4rt = quantization.get("ds4rt")
        recipe = ds4rt.get("recipe") if isinstance(ds4rt, dict) else None
        if recipe in HISTORICAL_EXL3_RECIPES:
            if allow_historical_control:
                return str(recipe)
            raise ValueError(
                "checkpoint declares historical non-production DS4RT EXL3 recipe "
                f"{recipe!r}"
            )
        if recipe not in PRODUCTION_EXL3_RECIPES:
            raise ValueError(
                f"checkpoint declares unsupported DS4RT EXL3 recipe {recipe!r}"
            )
        return str(recipe)
    raise ValueError(f"unsupported checkpoint quantization method {method!r}")


def public_exl3_checkpoint_identity(checkpoint: Path) -> dict[str, Any]:
    """Bind an ordinary compact Hugging Face EXL3 snapshot for comparisons."""

    checkpoint = checkpoint.expanduser().resolve(strict=True)
    if checkpoint.parent.name != "snapshots":
        raise ValueError("public EXL3 checkpoint is not an immutable HF snapshot")
    model_root = checkpoint.parent.parent
    if not model_root.name.startswith("models--"):
        raise ValueError("public EXL3 checkpoint has no Hugging Face model root")
    model_id = model_root.name.removeprefix("models--").replace("--", "/", 1)
    config_path = checkpoint / "config.json"
    quantize_path = checkpoint / "quantize_config.json"
    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ValueError("public EXL3 checkpoint has no safetensors index")
    return {
        "schema": "huggingface-public-exl3-snapshot-v1",
        "model_id": model_id,
        "revision": checkpoint.name,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "quantize_config_sha256": hashlib.sha256(quantize_path.read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
    }


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _validate_staged_gptqmodel_qualification(
    checkpoint: Path,
    model_root: Path,
    revision: str,
    manifest: dict[str, Any],
    *,
    entries_key: str = "qualification",
    evidence_directory: str = "ds4rt-qualifications",
    expected_schemas: dict[str, str] = GPTQMODEL_QUALIFICATION_SCHEMAS,
) -> list[dict[str, Any]]:
    entries = manifest.get(entries_key)
    if not isinstance(entries, list) or len(entries) != 2:
        raise ValueError(
            "staged GPTQModel EXL3 snapshot lacks complete external qualification"
        )
    qualification_root = model_root / evidence_directory / revision
    if (
        not qualification_root.is_dir()
        or qualification_root.is_symlink()
        or {path.name for path in qualification_root.iterdir()}
        != set(expected_schemas)
    ):
        raise ValueError("staged GPTQModel qualification directory is invalid")

    source_snapshot = manifest.get("source_snapshot")
    if (
        not isinstance(source_snapshot, str)
        or not source_snapshot
        or not Path(source_snapshot).is_absolute()
    ):
        raise ValueError("staged GPTQModel manifest has no source publication identity")
    metadata_sha256 = {
        name: _hash_file(checkpoint / name) for name in GPTQMODEL_IDENTITY_FILES
    }
    expected_recipe = checkpoint_quantization_recipe(checkpoint)
    shard_geometry = {
        path.name: path.stat().st_size
        for path in sorted(checkpoint.glob("*.safetensors"))
    }
    if not shard_geometry:
        raise ValueError("staged GPTQModel snapshot has no safetensors shards")

    validated: list[dict[str, Any]] = []
    observed: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("staged GPTQModel qualification entry is malformed")
        name = entry.get("path")
        digest = entry.get("sha256")
        size = entry.get("size")
        schema = entry.get("schema")
        if (
            name not in expected_schemas
            or name in observed
            or schema != expected_schemas[name]
            or SHA256_RE.fullmatch(str(digest)) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise ValueError("staged GPTQModel qualification entry is invalid")
        report_path = qualification_root / str(name)
        if (
            not report_path.is_file()
            or report_path.is_symlink()
            or report_path.stat().st_size != size
            or _hash_file(report_path) != digest
        ):
            raise ValueError(f"staged GPTQModel qualification differs: {report_path}")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"cannot read staged GPTQModel qualification {report_path}: {error}"
            ) from error
        if not isinstance(report, dict) or report.get("schema") != schema:
            raise ValueError("staged GPTQModel qualification schema is invalid")

        if name == "retained-native.json":
            publication = report.get("gptqmodel_publication")
            if (
                report.get("recipe") != expected_recipe
                or report.get("quantization_scope") != "routed_experts_only"
                or report.get("exl3_snapshot") != source_snapshot
                or not isinstance(publication, dict)
                or publication.get("path") != source_snapshot
                or publication.get("metadata_sha256") != metadata_sha256
            ):
                raise ValueError(
                    "staged retained-native report is not bound to this publication"
                )
        else:
            contract = report.get("validation_contract")
            contract_sha256 = (
                contract.get("sha256") if isinstance(contract, dict) else None
            )
            contract_body = (
                {key: value for key, value in contract.items() if key != "sha256"}
                if isinstance(contract, dict)
                else None
            )
            canonical = (
                json.dumps(
                    contract_body,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                if contract_body is not None
                else b""
            )
            exl3 = contract.get("exl3") if isinstance(contract, dict) else None
            reported_metadata = (
                exl3.get("metadata_sha256") if isinstance(exl3, dict) else None
            )
            valid_metadata = (
                isinstance(reported_metadata, dict)
                and set(GPTQMODEL_IDENTITY_FILES).issubset(reported_metadata)
                and set(reported_metadata).issubset(GPTQMODEL_QUALITY_IDENTITY_FILES)
                and all(
                    SHA256_RE.fullmatch(str(digest)) is not None
                    and (checkpoint / name).is_file()
                    and _hash_file(checkpoint / name) == digest
                    for name, digest in reported_metadata.items()
                )
            )
            shards = exl3.get("shards") if isinstance(exl3, dict) else None
            valid_shards = (
                isinstance(shards, list)
                and len(shards) == len(shard_geometry)
                and all(
                    isinstance(record, dict)
                    and isinstance(record.get("name"), str)
                    and record["name"]
                    and not isinstance(record.get("size"), bool)
                    and isinstance(record.get("size"), int)
                    and record["size"] > 0
                    for record in shards
                )
                and len({record["name"] for record in shards}) == len(shards)
            )
            reported_shards = (
                {
                    record["name"]: record["size"]
                    for record in shards
                }
                if valid_shards
                else None
            )
            if (
                SHA256_RE.fullmatch(str(contract_sha256)) is None
                or hashlib.sha256(canonical).hexdigest() != contract_sha256
                or entry.get("contract_sha256") != contract_sha256
                or not isinstance(exl3, dict)
                or exl3.get("path") != source_snapshot
                or not valid_metadata
                or reported_shards != shard_geometry
                or contract.get("allow_incomplete") is not False
            ):
                raise ValueError(
                    "staged quality report is not bound to this publication"
                )
        observed.add(str(name))
        validated.append(
            {
                "path": str(name),
                "schema": str(schema),
                "sha256": str(digest),
                "size": size,
            }
        )
    if observed != set(expected_schemas):
        raise ValueError("staged GPTQModel qualification is incomplete")
    return sorted(validated, key=lambda item: item["path"])


def validate_staged_exl3_checkpoint(
    checkpoint: Path,
    *,
    allow_development_unqualified: bool = False,
    audit_publication: bool = True,
) -> dict[str, Any]:
    """Bind qualification to an immutable snapshot published by our stager.

    The staging revision is the SHA-256 of the canonical file manifest. The
    snapshot itself contains content-addressed symlinks, so validating paths,
    targets, and sizes is fast and does not reread the full model before every
    serving run.
    """

    checkpoint = checkpoint.expanduser().resolve(strict=True)
    if checkpoint.parent.name != "snapshots":
        raise ValueError(
            "production EXL3 checkpoint must be an immutable staged snapshots/<revision> path"
        )
    revision = checkpoint.name
    if SHA256_RE.fullmatch(revision) is None:
        raise ValueError("staged EXL3 snapshot revision is not a SHA-256")
    model_root = checkpoint.parent.parent
    manifest_path = model_root / "ds4rt-manifests" / f"{revision}.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot read staged EXL3 manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(manifest, dict):
        raise ValueError("staged EXL3 manifest is not a JSON object")
    if manifest.get("schema") != STAGED_ARTIFACT_SCHEMA:
        raise ValueError("staged EXL3 manifest has the wrong schema")
    if manifest.get("revision") != revision:
        raise ValueError("staged EXL3 manifest revision differs from snapshot path")
    model_id = manifest.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("staged EXL3 manifest has no model identity")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("staged EXL3 manifest has no files")

    paths: list[str] = []
    total_bytes = 0
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("staged EXL3 manifest contains a malformed file entry")
        raw_path = entry.get("path")
        sha256 = entry.get("sha256")
        size = entry.get("size")
        if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
            raise ValueError("staged EXL3 manifest contains an invalid file path")
        relative = PurePosixPath(raw_path)
        if relative.is_absolute() or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise ValueError(
                f"staged EXL3 manifest contains an unsafe file path {raw_path!r}"
            )
        if SHA256_RE.fullmatch(str(sha256)) is None:
            raise ValueError(
                f"staged EXL3 manifest has an invalid digest for {raw_path}"
            )
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"staged EXL3 manifest has an invalid size for {raw_path}")
        paths.append(raw_path)
        total_bytes += size

        snapshot_file = checkpoint.joinpath(*relative.parts)
        if not snapshot_file.is_symlink():
            raise ValueError(
                f"staged EXL3 snapshot entry is not content-addressed: {snapshot_file}"
            )
        try:
            resolved = snapshot_file.resolve(strict=True)
            actual_size = resolved.stat().st_size
        except OSError as error:
            raise ValueError(
                f"staged EXL3 snapshot entry cannot be resolved: {snapshot_file}: {error}"
            ) from error
        if resolved.name != sha256:
            raise ValueError(
                f"staged EXL3 snapshot entry targets the wrong blob: {snapshot_file}"
            )
        # Model cards are publication metadata, not serving inputs.  A public
        # Hugging Face card can legitimately be edited after the immutable
        # tensor snapshot was staged.  Permit that drift only behind the
        # explicit WIP control override; release validation remains strict and
        # every serving-relevant file still has to match its staged size.
        wip_document_drift = (
            allow_development_unqualified and raw_path == "README.md"
        )
        if actual_size != size and not wip_document_drift:
            raise ValueError(
                f"staged EXL3 snapshot entry size differs from manifest: {snapshot_file}"
            )

    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("staged EXL3 manifest paths are unsorted or duplicated")
    try:
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read staged EXL3 config: {error}") from error
    if not isinstance(config, dict):
        raise ValueError("staged EXL3 config is not a JSON object")
    gptqmodel_native = is_gptqmodel_native_exl3(config.get("quantization_config"))
    required_files = (
        STAGED_GPTQMODEL_REQUIRED_FILES
        if gptqmodel_native
        else STAGED_EXL3_REQUIRED_FILES
    )
    missing_required = required_files.difference(paths)
    development_standard_artifact_override = (
        allow_development_unqualified
        and not gptqmodel_native
        and bool(missing_required)
    )
    if missing_required and not development_standard_artifact_override:
        raise ValueError(
            "staged EXL3 manifest lacks production evidence: "
            + ", ".join(sorted(missing_required))
        )

    actual_paths: set[str] = set()
    for root, directories, names in os.walk(checkpoint, followlinks=False):
        root_path = Path(root)
        for directory in directories:
            path = root_path / directory
            if path.is_symlink():
                raise ValueError(
                    f"staged EXL3 snapshot contains a symlinked directory: {path}"
                )
        for name in names:
            actual_paths.add((root_path / name).relative_to(checkpoint).as_posix())
    if actual_paths != set(paths):
        raise ValueError("staged EXL3 snapshot contents differ from its manifest")

    canonical = json.dumps(
        {"schema": STAGED_ARTIFACT_SCHEMA, "files": files},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if hashlib.sha256(canonical).hexdigest() != revision:
        raise ValueError("staged EXL3 manifest does not derive the snapshot revision")

    qualification: list[dict[str, Any]] = []
    development_evidence: list[dict[str, Any]] = []
    qualification_status = manifest.get("qualification_status")
    if gptqmodel_native:
        if checkpoint_quantization_recipe(checkpoint) not in GPTQMODEL_RECIPES:
            raise ValueError("staged GPTQModel checkpoint is not the production recipe")
        if audit_publication:
            validate_gptqmodel_publication(
                checkpoint,
                config,
                verify_all_hashes=False,
                require_canonical=True,
            )
        if qualification_status is None:
            if manifest.get("development_evidence") not in (None, []):
                raise ValueError(
                    "production GPTQModel snapshot has unexpected development evidence"
                )
            qualification = _validate_staged_gptqmodel_qualification(
                checkpoint,
                model_root,
                revision,
                manifest,
            )
            qualification_status = "production-qualified"
        elif qualification_status == DEVELOPMENT_QUALIFICATION_STATUS:
            if not allow_development_unqualified:
                raise ValueError(
                    "staged GPTQModel snapshot is development-unqualified; "
                    "an explicit WIP-only override is required"
                )
            if (
                manifest.get("qualification") not in (None, [])
                or manifest.get("development_blockers") != list(DEVELOPMENT_BLOCKERS)
            ):
                raise ValueError(
                    "staged GPTQModel development qualification state is invalid"
                )
            development_evidence = _validate_staged_gptqmodel_qualification(
                checkpoint,
                model_root,
                revision,
                manifest,
                entries_key="development_evidence",
                evidence_directory="ds4rt-development-evidence",
                expected_schemas=GPTQMODEL_DEVELOPMENT_EVIDENCE_SCHEMAS,
            )
        else:
            raise ValueError("staged GPTQModel qualification status is invalid")
    elif manifest.get("qualification") not in (None, []):
        raise ValueError(
            "legacy staged EXL3 snapshot has unexpected external qualification"
        )

    return {
        "schema": STAGED_ARTIFACT_SCHEMA,
        "model_id": model_id,
        "revision": revision,
        "manifest": str(manifest_path),
        "source_snapshot": manifest.get("source_snapshot"),
        "files": len(files),
        "bytes": total_bytes,
        "qualification": qualification,
        "qualification_status": qualification_status,
        "development_blockers": manifest.get("development_blockers", []),
        "development_evidence": development_evidence,
        "development_override": (
            {
                "kind": "wip-standard-artifact-without-private-evidence",
                "missing_production_evidence": sorted(missing_required),
            }
            if development_standard_artifact_override
            else None
        ),
    }


def audit_checkpoint(
    checkpoint: Path,
    *,
    allow_development_unqualified: bool = False,
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    recipe = checkpoint_quantization_recipe(
        checkpoint,
        allow_historical_control=allow_development_unqualified,
    )
    manifest = (
        checkpoint.parent.parent / "ds4rt-manifests" / f"{checkpoint.name}.json"
    )
    if (
        recipe in PRODUCTION_EXL3_RECIPES or recipe in HISTORICAL_EXL3_RECIPES
    ) and manifest.is_file():
        identity = validate_staged_exl3_checkpoint(
            checkpoint,
            allow_development_unqualified=allow_development_unqualified,
        )
    elif recipe in GPTQMODEL_RECIPES:
        identity = public_exl3_checkpoint_identity(checkpoint)
    else:
        config_path = checkpoint / "config.json"
        identity = {
            "schema": "huggingface-native-snapshot-v1",
            "revision": checkpoint.name,
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        }
    return {
        "schema": CHECKPOINT_AUDIT_SCHEMA,
        "checkpoint": str(checkpoint),
        "quantization_recipe": recipe,
        "artifact_identity": identity,
        "summary": {"gate_passed": True},
    }


def resolve_collection_model(
    requested_model: str | None,
    artifact_identity: dict[str, Any],
) -> str:
    """Bind a generation request to the staged repository identity.

    Native Flash predates the dedicated staging manifest, so it retains the
    official API default unless the caller supplies an explicit model. A
    staged EXL3 checkpoint carries the exact repository ID in its immutable
    manifest; using any other request identity would test a different daemon
    configuration and is rejected before the first request.
    """

    if artifact_identity.get("schema") != STAGED_ARTIFACT_SCHEMA:
        return requested_model or DEFAULT_FLASH_API_MODEL

    staged_model_id = artifact_identity.get("model_id")
    if not isinstance(staged_model_id, str) or not staged_model_id:
        raise ValueError("staged checkpoint identity has no model ID")
    expected_model = f"{staged_model_id}-full"
    if requested_model is not None and requested_model != expected_model:
        raise ValueError(
            f"requested API model {requested_model!r} differs from staged "
            f"checkpoint identity {expected_model!r}"
        )
    return expected_model


def validate_runtime_evidence(
    result: dict[str, Any],
    *,
    model: str,
    checkpoint: Path,
    expected_quantization_recipe: str,
    expected_spark_targets: int,
    expected_dspark: str | None = None,
    expected_concurrency: int = 1,
    require_dspark_execution: bool = True,
) -> dict[str, Any]:
    """Fail closed unless a completion came from the intended production path."""

    checkpoint = checkpoint.expanduser().resolve(strict=True)
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("completion has no metrics object")
    real_full = metrics.get("real_full")
    if not isinstance(real_full, dict):
        raise ValueError("completion has no real_full runtime evidence")

    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    service_instance_id = real_full.get("service_instance_id")
    require(
        is_canonical_service_instance_id(service_instance_id),
        "runtime service_instance_id is not a canonical UUIDv4",
    )

    reported_snapshot = real_full.get("snapshot_path")
    resolved_snapshot: Path | None = None
    if isinstance(reported_snapshot, str) and reported_snapshot:
        try:
            resolved_snapshot = Path(reported_snapshot).expanduser().resolve(strict=True)
        except OSError as error:
            failures.append(f"reported snapshot cannot be resolved: {error}")
    else:
        failures.append("runtime did not report snapshot_path")

    runtime_model = real_full.get("model_id")
    require(result.get("model") == model, "response model differs from request")
    require(
        isinstance(runtime_model, str) and model == f"{runtime_model}-full",
        "API model is not the loaded full-model identity",
    )
    require(metrics.get("backend_mode") == "real-ds4-full", "backend is not real-ds4-full")
    require(
        metrics.get("transport_backend") in {"tcp", "verbs-host"},
        "transport is not a production Spark transport",
    )
    require(
        resolved_snapshot == checkpoint,
        f"runtime snapshot {resolved_snapshot} differs from checkpoint {checkpoint}",
    )
    require(bool(real_full.get("catalog_hash")), "runtime catalog hash is empty")
    require(
        real_full.get("quantization_recipe") == expected_quantization_recipe,
        "runtime quantization recipe differs from the selected checkpoint",
    )
    require(real_full.get("layer_count") == FLASH_LAYER_COUNT, "runtime is not 43-layer Flash")
    require(real_full.get("dense_layer_count") == 0, "Flash runtime has dense bootstrap layers")
    require(
        real_full.get("sparse_layer_count") == FLASH_LAYER_COUNT,
        "Flash runtime does not route every layer through MoE",
    )
    require(real_full.get("protocol") == "ExpertProtocolV2", "runtime is not using Protocol V2")
    require(real_full.get("status") == "ready", "runtime request is not ready")
    require(
        real_full.get("startup_diagnostic_mode") == "request-scheduler-execution",
        "completion did not execute the request scheduler",
    )
    require(real_full.get("blocker") in {None, ""}, "runtime reported a blocker")
    require(not real_full.get("failed_requirements"), "runtime reported failed requirements")
    for field in (
        "request_scheduler_summary_runtime_reported",
        "scheduler_numeric_progression_passed",
        "request_numeric_progression_passed",
        "scheduler_full_context_device_attention_complete",
        "scheduler_terminal_lm_head_sample_passed",
        "scheduler_terminal_lm_head_uses_final_decode_device_hidden",
        "scheduler_terminal_lm_head_covers_full_vocabulary",
        "scheduler_sparse_tcp_dispatch_passed",
        "scheduler_sparse_tcp_dispatch_all_responses_real_checkpoint_experts",
        "scheduler_sparse_tcp_dispatch_consumed_by_residual",
        "request_expert_route_entries_match_source_rows",
    ):
        require(real_full.get(field) is True, f"runtime evidence {field} is not true")
    require(
        isinstance(real_full.get("request_kv_reads"), int)
        and real_full["request_kv_reads"] > 0,
        "live request has no device-backed KV reads",
    )
    require(
        isinstance(real_full.get("request_committed_kv_writes"), int)
        and real_full["request_committed_kv_writes"] > 0,
        "live request has no committed device-backed KV writes",
    )
    require(
        isinstance(real_full.get("request_kv_reservation_bytes"), int)
        and real_full["request_kv_reservation_bytes"] > 0,
        "live request has no device-KV reservation",
    )
    require(
        real_full.get("request_coordinator_graph_captures") == 0,
        "live request performed a coordinator graph capture",
    )
    require(
        real_full.get("scheduler_sparse_tcp_dispatch_targets") == expected_spark_targets,
        f"runtime did not use exactly {expected_spark_targets} Spark targets",
    )
    require(
        isinstance(real_full.get("request_sparse_batches"), int)
        and real_full["request_sparse_batches"] > 0,
        "runtime request has no jointly issued sparse expert batch",
    )
    require(
        isinstance(real_full.get("request_expert_batch_rows"), int)
        and real_full["request_expert_batch_rows"] > 0,
        "runtime request has no expert rows",
    )
    require(
        isinstance(real_full.get("request_expert_batch_routes"), int)
        and real_full["request_expert_batch_routes"] > 0,
        "runtime request has no expert routes",
    )
    if expected_dspark not in {None, "on", "off"}:
        raise ValueError(f"unsupported expected dSpark state {expected_dspark!r}")
    if not 1 <= expected_concurrency <= 4:
        raise ValueError("expected concurrency must be in 1..4")
    if expected_dspark == "on" and require_dspark_execution:
        require(
            isinstance(real_full.get("mtp_verify_cycles"), int)
            and real_full["mtp_verify_cycles"] > 0,
            "integrated dSpark produced no verification cycles",
        )
        if expected_concurrency > 1:
            require(
                real_full.get("dspark_max_joint_batch_width")
                == expected_concurrency,
                "integrated dSpark did not jointly execute the expected concurrency",
            )
            require(
                isinstance(real_full.get("dspark_joint_cycles"), int)
                and real_full["dspark_joint_cycles"] > 0,
                "integrated dSpark produced no joint cycles",
            )
        if expected_concurrency == 4:
            require(
                real_full.get("dspark_max_joint_cohort_width") == 2
                and real_full.get("dspark_max_joint_cohort_count") == 2,
                "C4 integrated dSpark did not use two cohorts of two requests",
            )
            require(
                isinstance(real_full.get("dspark_2x2_wavefront_cycles"), int)
                and real_full["dspark_2x2_wavefront_cycles"] > 0,
                "C4 integrated dSpark produced no 2x2 wavefront cycles",
            )
    elif expected_dspark == "off":
        require(
            real_full.get("mtp_verify_cycles") == 0,
            "dSpark-off request unexpectedly produced verification cycles",
        )
        require(
            real_full.get("dspark_joint_cycles") == 0,
            "dSpark-off request unexpectedly used the joint dSpark executor",
        )
    logits = real_full.get("scheduler_terminal_lm_head_logits_evaluated")
    vocabulary = real_full.get("scheduler_terminal_lm_head_vocab_size")
    require(
        isinstance(logits, int) and logits > 0 and logits == vocabulary,
        "terminal lm_head did not evaluate the full vocabulary",
    )
    if failures:
        raise ValueError("completion failed production runtime gate: " + "; ".join(failures))

    evidence_fields = (
        "model_id",
        "service_instance_id",
        "snapshot_path",
        "catalog_hash",
        "quantization_recipe",
        "layer_count",
        "dense_layer_count",
        "sparse_layer_count",
        "protocol",
        "status",
        "startup_diagnostic_mode",
        "request_scheduler_summary_runtime_reported",
        "scheduler_numeric_progression_passed",
        "request_numeric_progression_passed",
        "scheduler_full_context_device_attention_complete",
        "scheduler_terminal_lm_head_sample_passed",
        "scheduler_terminal_lm_head_uses_final_decode_device_hidden",
        "scheduler_terminal_lm_head_covers_full_vocabulary",
        "scheduler_terminal_lm_head_logits_evaluated",
        "scheduler_terminal_lm_head_vocab_size",
        "scheduler_sparse_tcp_dispatch_targets",
        "scheduler_sparse_tcp_dispatch_passed",
        "scheduler_sparse_tcp_dispatch_all_responses_real_checkpoint_experts",
        "scheduler_sparse_tcp_dispatch_consumed_by_residual",
        "request_sparse_batches",
        "request_expert_batch_rows",
        "request_expert_batch_routes",
        "request_expert_route_entries_match_source_rows",
        "mtp_verify_cycles",
        "mtp_draft_tokens",
        "mtp_accepted_draft_tokens",
        "dspark_joint_cycles",
        "dspark_max_joint_batch_width",
        "dspark_max_joint_cohort_width",
        "dspark_max_joint_cohort_count",
        "dspark_2x2_wavefront_cycles",
        "request_kv_reads",
        "request_committed_kv_writes",
        "request_kv_reservation_bytes",
        "request_byte_backed_scheduler_trace",
        "request_coordinator_graph_captures",
    )
    return {
        "backend_mode": metrics["backend_mode"],
        "transport_backend": metrics["transport_backend"],
        **{field: real_full.get(field) for field in evidence_fields},
    }


def validate_dspark_group_evidence(
    results: list[dict[str, Any]],
    *,
    expected_dspark: str,
    expected_concurrency: int,
) -> dict[str, Any]:
    """Qualify dSpark only when every request survives the first target token.

    A request that stops after its first sampled token has no next token for a
    draft block to propose, so zero verification cycles are correct.  Joint
    C2--C4 qualification therefore belongs to the concurrent group, not every
    individual response.  Full groups whose members all emit at least two
    tokens must show the expected joint execution; the complete suite must
    still contain observed dSpark work (enforced by ``collect_run``).
    """

    if expected_dspark not in {"on", "off"}:
        raise ValueError(f"unsupported expected dSpark state {expected_dspark!r}")
    if not 1 <= expected_concurrency <= 4:
        raise ValueError("expected concurrency must be in 1..4")
    if len(results) != expected_concurrency:
        raise ValueError(
            "dSpark group result count differs from expected concurrency: "
            f"{len(results)} != {expected_concurrency}"
        )

    completion_tokens: list[int] = []
    real_full_metrics: list[dict[str, Any]] = []
    for result in results:
        count = result.get("usage", {}).get("completion_tokens")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("completion has no positive completion-token count")
        completion_tokens.append(count)
        real_full = result.get("metrics", {}).get("real_full")
        if not isinstance(real_full, dict):
            raise ValueError("completion has no real_full runtime evidence")
        real_full_metrics.append(real_full)

    execution_required = expected_dspark == "on" and all(
        count > 1 for count in completion_tokens
    )
    execution_observed = any(
        isinstance(metrics.get("mtp_verify_cycles"), int)
        and metrics["mtp_verify_cycles"] > 0
        for metrics in real_full_metrics
    )
    joint_observed = any(
        metrics.get("dspark_max_joint_batch_width") == expected_concurrency
        and isinstance(metrics.get("dspark_joint_cycles"), int)
        and metrics["dspark_joint_cycles"] > 0
        for metrics in real_full_metrics
    )
    wavefront_2x2_observed = any(
        metrics.get("dspark_max_joint_cohort_width") == 2
        and metrics.get("dspark_max_joint_cohort_count") == 2
        and isinstance(metrics.get("dspark_2x2_wavefront_cycles"), int)
        and metrics["dspark_2x2_wavefront_cycles"] > 0
        for metrics in real_full_metrics
    )

    if execution_required and not execution_observed:
        raise ValueError("eligible dSpark group produced no verification cycles")
    if execution_required and expected_concurrency > 1 and not joint_observed:
        raise ValueError(
            "eligible dSpark group did not jointly execute the expected concurrency"
        )
    if execution_required and expected_concurrency == 4 and not wavefront_2x2_observed:
        raise ValueError("eligible C4 dSpark group did not execute a 2x2 wavefront")

    return {
        "completion_tokens": completion_tokens,
        "execution_required": execution_required,
        "execution_observed": execution_observed,
        "joint_observed": joint_observed,
        "wavefront_2x2_observed": wavefront_2x2_observed,
    }


def collect_run(
    *,
    url: str,
    model: str | None,
    artifact_label: str,
    checkpoint: str,
    cases: Iterable[QualityCase],
    repeats: int,
    timeout: float,
    expected_spark_targets: int,
    expected_dspark: str,
    concurrency: int,
    allow_development_unqualified: bool = False,
) -> dict[str, Any]:
    selected = tuple(cases)
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if expected_spark_targets < 1:
        raise ValueError("expected_spark_targets must be positive")
    if expected_dspark not in {"on", "off"}:
        raise ValueError("expected_dspark must be on or off")
    if not 1 <= concurrency <= 4:
        raise ValueError("concurrency must be in 1..4")
    if not selected or len(selected) % concurrency != 0:
        raise ValueError(
            "selected case count must be a nonzero multiple of concurrency so every "
            "sample belongs to a full concurrent group"
        )
    checkpoint_path = Path(checkpoint).expanduser().resolve(strict=True)
    expected_recipe = checkpoint_quantization_recipe(
        checkpoint_path,
        allow_historical_control=allow_development_unqualified,
    )
    manifest = (
        checkpoint_path.parent.parent
        / "ds4rt-manifests"
        / f"{checkpoint_path.name}.json"
    )
    artifact_identity = (
        validate_staged_exl3_checkpoint(
            checkpoint_path,
            allow_development_unqualified=allow_development_unqualified,
        )
        if (
            expected_recipe in PRODUCTION_EXL3_RECIPES
            or expected_recipe in HISTORICAL_EXL3_RECIPES
        )
        and manifest.is_file()
        else public_exl3_checkpoint_identity(checkpoint_path)
        if expected_recipe in GPTQMODEL_RECIPES
        else {
            "schema": "huggingface-native-snapshot-v1",
            "revision": checkpoint_path.name,
            "config_sha256": hashlib.sha256(
                (checkpoint_path / "config.json").read_bytes()
            ).hexdigest(),
        }
    )
    model = resolve_collection_model(model, artifact_identity)
    records = []
    dspark_group_evidence: list[dict[str, Any]] = []
    for repeat in range(1, repeats + 1):
        for group_offset in range(0, len(selected), concurrency):
            group = selected[group_offset : group_offset + concurrency]
            results = request_completion_group(url, model, group, timeout)
            group_evidence = validate_dspark_group_evidence(
                results,
                expected_dspark=expected_dspark,
                expected_concurrency=concurrency,
            )
            dspark_group_evidence.append(group_evidence)
            for case, result in zip(group, results, strict=True):
                runtime_evidence = validate_runtime_evidence(
                    result,
                    model=model,
                    checkpoint=checkpoint_path,
                    expected_quantization_recipe=expected_recipe,
                    expected_spark_targets=expected_spark_targets,
                    expected_dspark=expected_dspark,
                    expected_concurrency=concurrency,
                    require_dspark_execution=False,
                )
                choice = result["choices"][0]
                message = choice["message"]
                content = message.get("content") or ""
                passed, score_error = score_output(case, content)
                record = {
                    "case_id": case.case_id,
                    "category": case.category,
                    "repeat": repeat,
                    "concurrent_group": group_offset // concurrency + 1,
                    "concurrency": concurrency,
                    "expected_dspark": expected_dspark,
                    "passed": passed,
                    "score_error": score_error,
                    "content": content,
                    "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "finish_reason": choice.get("finish_reason"),
                    "prompt_tokens": result.get("usage", {}).get("prompt_tokens"),
                    "completion_tokens": result.get("usage", {}).get("completion_tokens"),
                    "runtime_evidence": runtime_evidence,
                    "dspark_group_evidence": group_evidence,
                }
                records.append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
    if expected_dspark == "on":
        if not any(group["execution_observed"] for group in dspark_group_evidence):
            raise ValueError("generation suite observed no integrated dSpark execution")
        if concurrency > 1 and not any(
            group["joint_observed"] for group in dspark_group_evidence
        ):
            raise ValueError(
                "generation suite observed no joint integrated dSpark execution"
            )
        if concurrency == 4 and not any(
            group["wavefront_2x2_observed"] for group in dspark_group_evidence
        ):
            raise ValueError("generation suite observed no C4 dSpark 2x2 wavefront")
    passed = sum(record["passed"] for record in records)
    service_instance_ids = {
        record["runtime_evidence"]["service_instance_id"] for record in records
    }
    if len(service_instance_ids) != 1:
        raise ValueError(
            "generation suite crossed service lifetimes: "
            f"observed {sorted(service_instance_ids)!r}"
        )
    service_instance_id = next(iter(service_instance_ids))
    return {
        "schema": RUN_SCHEMA,
        "runtime_gate": RUNTIME_GATE,
        "service_instance_id": service_instance_id,
        "suite_sha256": suite_identity(selected),
        "case_ids": [case.case_id for case in selected],
        "artifact_label": artifact_label,
        "checkpoint": str(checkpoint_path),
        "artifact_identity": artifact_identity,
        "expected_quantization_recipe": expected_recipe,
        "expected_spark_targets": expected_spark_targets,
        "expected_dspark": expected_dspark,
        "concurrency": concurrency,
        "url": url,
        "model": model,
        "repeats": repeats,
        "records": records,
        "dspark_groups": dspark_group_evidence,
        "summary": {
            "samples": len(records),
            "passed": passed,
            "pass_rate": passed / len(records) if records else 0.0,
        },
    }


def collect_matrix_arm(
    *,
    url: str,
    model: str | None,
    artifact_label: str,
    checkpoint: str,
    cases: Iterable[QualityCase],
    repeats: int,
    timeout: float,
    expected_spark_targets: int,
    expected_dspark: str,
    allow_development_unqualified: bool = False,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    selected = tuple(cases)
    if not selected or any(len(selected) % concurrency for concurrency in range(1, 5)):
        raise ValueError(
            "matrix-arm case count must be nonzero and divisible by C1, C2, C3, and C4"
        )
    runs: dict[int, dict[str, Any]] = {}
    for concurrency in range(1, 5):
        runs[concurrency] = collect_run(
            url=url,
            model=model,
            artifact_label=f"{artifact_label}-c{concurrency}",
            checkpoint=checkpoint,
            cases=selected,
            repeats=repeats,
            timeout=timeout,
            expected_spark_targets=expected_spark_targets,
            expected_dspark=expected_dspark,
            concurrency=concurrency,
            allow_development_unqualified=allow_development_unqualified,
        )

    service_instance_ids = {
        run["service_instance_id"] for run in runs.values()
    }
    if len(service_instance_ids) != 1:
        raise ValueError(
            "C1-C4 matrix arm crossed service lifetimes: "
            f"observed {sorted(service_instance_ids)!r}"
        )
    artifact_identities = {
        json.dumps(run["artifact_identity"], sort_keys=True, separators=(",", ":"))
        for run in runs.values()
    }
    if len(artifact_identities) != 1:
        raise ValueError("C1-C4 matrix arm crossed artifact identities")
    service_instance_id = next(iter(service_instance_ids))
    artifact_identity = runs[1]["artifact_identity"]
    summary = {
        "schema": ARM_SCHEMA,
        "expected_dspark": expected_dspark,
        "service_instance_id": service_instance_id,
        "artifact_identity": artifact_identity,
        "suite_sha256": runs[1]["suite_sha256"],
        "concurrencies": [1, 2, 3, 4],
        "summary": {
            "cells": 4,
            "samples": sum(run["summary"]["samples"] for run in runs.values()),
            "passed": sum(run["summary"]["passed"] for run in runs.values()),
            "gate_passed": True,
        },
    }
    return summary, runs


def collect_reference_run(
    *,
    url: str,
    model: str,
    artifact_label: str,
    cases: Iterable[QualityCase],
    repeats: int,
    timeout: float,
    minimum_max_tokens: int,
    thinking_mode: str,
    auth_token: str,
) -> dict[str, Any]:
    selected = tuple(cases)
    if repeats < 1:
        raise ValueError("repeats must be positive")
    records = []
    for repeat in range(1, repeats + 1):
        for case in selected:
            result = request_anthropic_completion(
                url,
                model,
                case,
                auth_token,
                timeout,
                minimum_max_tokens,
                thinking_mode,
            )
            content, reasoning = anthropic_response_content(result)
            passed, score_error = score_output(case, content)
            record = {
                "case_id": case.case_id,
                "category": case.category,
                "repeat": repeat,
                "passed": passed,
                "score_error": score_error,
                "content": content,
                "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "reasoning": reasoning,
                "reasoning_sha256": hashlib.sha256(reasoning.encode()).hexdigest(),
                "stop_reason": result.get("stop_reason"),
                "response_model": result.get("model"),
                "usage": result.get("usage"),
            }
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    passed = sum(record["passed"] for record in records)
    return {
        "schema": REFERENCE_SCHEMA,
        "suite_sha256": suite_identity(selected),
        "case_ids": [case.case_id for case in selected],
        "artifact_label": artifact_label,
        "url": url,
        "model": model,
        "repeats": repeats,
        "minimum_max_tokens": minimum_max_tokens,
        "thinking_mode": thinking_mode,
        "records": records,
        "summary": {
            "samples": len(records),
            "passed": passed,
            "pass_rate": passed / len(records) if records else 0.0,
        },
    }


def compare_runs(
    native: dict[str, Any],
    candidate: dict[str, Any],
    *,
    min_candidate_pass_rate: float | None = None,
    max_quality_regressions: int = 0,
) -> dict[str, Any]:
    service_instance_ids: dict[str, str] = {}
    for label, run in (("native", native), ("candidate", candidate)):
        if run.get("schema") != RUN_SCHEMA:
            raise ValueError(f"{label} report does not use {RUN_SCHEMA}")
        if run.get("runtime_gate") != RUNTIME_GATE:
            raise ValueError(f"{label} report did not pass {RUNTIME_GATE}")
        records = run.get("records")
        if not isinstance(records, list) or not all(
            isinstance(record, dict)
            and isinstance(record.get("runtime_evidence"), dict)
            and record["runtime_evidence"].get("backend_mode") == "real-ds4-full"
            and record["runtime_evidence"].get("startup_diagnostic_mode")
            == "request-scheduler-execution"
            for record in records
        ):
            raise ValueError(f"{label} report has unqualified runtime samples")
        if run.get("expected_dspark") not in {"on", "off"}:
            raise ValueError(f"{label} report has no qualified dSpark state")
        if not isinstance(run.get("concurrency"), int) or not 1 <= run["concurrency"] <= 4:
            raise ValueError(f"{label} report has no qualified concurrency")
        service_instance_id = run.get("service_instance_id")
        if not is_canonical_service_instance_id(service_instance_id):
            raise ValueError(
                f"{label} report has no canonical service instance identity"
            )
        if any(
            record["runtime_evidence"].get("service_instance_id")
            != service_instance_id
            for record in records
        ):
            raise ValueError(
                f"{label} report crossed service lifetimes or has inconsistent identity"
            )
        service_instance_ids[label] = service_instance_id
    if native.get("expected_quantization_recipe") != NATIVE_RECIPE:
        raise ValueError("native report does not use the native Flash recipe")
    native_identity = native.get("artifact_identity")
    if not isinstance(native_identity, dict) or (
        native_identity.get("schema") != "huggingface-native-snapshot-v1"
        or not isinstance(native_identity.get("revision"), str)
        or not native_identity["revision"]
    ):
        raise ValueError("native report is not bound to an immutable snapshot")
    if candidate.get("expected_quantization_recipe") not in PRODUCTION_EXL3_RECIPES:
        raise ValueError("candidate report does not use the frozen production EXL3 recipe")
    candidate_identity = candidate.get("artifact_identity")
    if not isinstance(candidate_identity, dict) or (
        candidate_identity.get("schema") != STAGED_ARTIFACT_SCHEMA
        or SHA256_RE.fullmatch(str(candidate_identity.get("revision", ""))) is None
    ):
        raise ValueError("candidate report is not bound to an immutable staged artifact")
    if native["suite_sha256"] != candidate["suite_sha256"]:
        raise ValueError("native and candidate reports use different task suites")
    if native.get("concurrency") != candidate.get("concurrency"):
        raise ValueError("native and candidate reports use different concurrency")
    if native.get("expected_dspark") != candidate.get("expected_dspark"):
        raise ValueError("native and candidate reports use different dSpark state")

    def indexed(run: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
        records = {}
        for record in run["records"]:
            key = (record["case_id"], int(record["repeat"]))
            if key in records:
                raise ValueError(f"duplicate record {key!r}")
            records[key] = record
        return records

    native_records = indexed(native)
    candidate_records = indexed(candidate)
    if native_records.keys() != candidate_records.keys():
        raise ValueError("native and candidate reports contain different samples")

    comparisons = []
    for key in sorted(native_records, key=lambda item: (item[1], item[0])):
        reference = native_records[key]
        quantized = candidate_records[key]
        similarity = difflib.SequenceMatcher(
            None, reference["content"], quantized["content"]
        ).ratio()
        comparisons.append(
            {
                "case_id": key[0],
                "repeat": key[1],
                "native_passed": bool(reference["passed"]),
                "candidate_passed": bool(quantized["passed"]),
                "quality_regression": bool(reference["passed"])
                and not bool(quantized["passed"]),
                "quality_improvement": not bool(reference["passed"])
                and bool(quantized["passed"]),
                "content_exact": reference["content"] == quantized["content"],
                "content_similarity": similarity,
                "native_content_sha256": reference["content_sha256"],
                "candidate_content_sha256": quantized["content_sha256"],
            }
        )

    samples = len(comparisons)
    native_passed = sum(item["native_passed"] for item in comparisons)
    native_pass_rate = native_passed / samples if samples else 0.0
    candidate_passed = sum(item["candidate_passed"] for item in comparisons)
    candidate_pass_rate = candidate_passed / samples if samples else 0.0
    effective_min_pass_rate = (
        native_pass_rate
        if min_candidate_pass_rate is None
        else min_candidate_pass_rate
    )
    if not 0.0 <= effective_min_pass_rate <= 1.0:
        raise ValueError("minimum candidate pass rate must be in 0..1")
    if max_quality_regressions < 0:
        raise ValueError("maximum quality regressions must be nonnegative")
    regressions = sum(item["quality_regression"] for item in comparisons)
    exact = sum(item["content_exact"] for item in comparisons)
    mean_similarity = (
        sum(item["content_similarity"] for item in comparisons) / samples
        if samples
        else 0.0
    )
    gate_passed = (
        samples > 0
        and candidate_pass_rate >= effective_min_pass_rate
        and regressions <= max_quality_regressions
    )
    return {
        "schema": COMPARISON_SCHEMA,
        "suite_sha256": native["suite_sha256"],
        "native_artifact_label": native["artifact_label"],
        "candidate_artifact_label": candidate["artifact_label"],
        "native_service_instance_id": service_instance_ids["native"],
        "candidate_service_instance_id": service_instance_ids["candidate"],
        "native_artifact_identity": native_identity,
        "candidate_artifact_identity": candidate_identity,
        "expected_dspark": native["expected_dspark"],
        "concurrency": native["concurrency"],
        "thresholds": {
            "min_candidate_pass_rate": effective_min_pass_rate,
            "min_candidate_pass_rate_source": (
                "native_run" if min_candidate_pass_rate is None else "explicit"
            ),
            "max_quality_regressions": max_quality_regressions,
        },
        "summary": {
            "samples": samples,
            "native_pass_rate": native_pass_rate,
            "candidate_pass_rate": candidate_pass_rate,
            "quality_regressions": regressions,
            "quality_improvements": sum(
                item["quality_improvement"] for item in comparisons
            ),
            "content_exact_rate": exact / samples if samples else 0.0,
            "mean_content_similarity": mean_similarity,
            "gate_passed": gate_passed,
        },
        "comparisons": comparisons,
    }


def compare_matrix(
    native_dir: Path,
    candidate_dir: Path,
    *,
    min_candidate_pass_rate: float | None = None,
    max_quality_regressions: int = 0,
) -> dict[str, Any]:
    cells = []
    service_lifetimes: dict[str, dict[str, str]] = {
        "native": {},
        "candidate": {},
    }
    artifact_identities: dict[str, dict[str, Any]] = {}
    for expected_dspark in ("on", "off"):
        for concurrency in range(1, 5):
            name = f"dspark-{expected_dspark}-c{concurrency}.json"
            native_path = native_dir / name
            candidate_path = candidate_dir / name
            native = json.loads(native_path.read_text(encoding="utf-8"))
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            comparison = compare_runs(
                native,
                candidate,
                min_candidate_pass_rate=min_candidate_pass_rate,
                max_quality_regressions=max_quality_regressions,
            )
            for arm in ("native", "candidate"):
                service_instance_id = comparison[f"{arm}_service_instance_id"]
                prior = service_lifetimes[arm].get(expected_dspark)
                if prior is None:
                    service_lifetimes[arm][expected_dspark] = service_instance_id
                elif prior != service_instance_id:
                    raise ValueError(
                        f"{arm} dSpark-{expected_dspark} C1-C4 crossed service "
                        f"lifetimes: {prior} != {service_instance_id} at C{concurrency}"
                    )
                artifact_identity = comparison[f"{arm}_artifact_identity"]
                prior_artifact = artifact_identities.get(arm)
                if prior_artifact is None:
                    artifact_identities[arm] = artifact_identity
                elif prior_artifact != artifact_identity:
                    raise ValueError(
                        f"{arm} matrix cells use different artifact identities"
                    )
            if comparison["summary"]["samples"] == 0:
                raise ValueError(f"matrix cell {expected_dspark}/C{concurrency} is empty")
            cells.append(
                {
                    "expected_dspark": expected_dspark,
                    "concurrency": concurrency,
                    "native_report": str(native_path),
                    "candidate_report": str(candidate_path),
                    "comparison": comparison,
                }
            )
    distinct_service_lifetimes = {
        service_instance_id
        for arm in service_lifetimes.values()
        for service_instance_id in arm.values()
    }
    if len(distinct_service_lifetimes) != 4:
        raise ValueError(
            "production matrix must come from four distinct service lifetimes: "
            "native/exl3 times dSpark on/off"
        )
    return {
        "schema": MATRIX_SCHEMA,
        "native_dir": str(native_dir),
        "candidate_dir": str(candidate_dir),
        "artifact_identities": artifact_identities,
        "service_lifetimes": service_lifetimes,
        "summary": {
            "cells": len(cells),
            "passed_cells": sum(
                cell["comparison"]["summary"]["gate_passed"] for cell in cells
            ),
            "gate_passed": all(
                cell["comparison"]["summary"]["gate_passed"] for cell in cells
            ),
        },
        "cells": cells,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)


def prepare_report_output(path: Path, *, replace: bool) -> None:
    if not os.path.lexists(path):
        return
    if not replace:
        raise ValueError(
            f"report output already exists: {path}; use --replace to discard it before running"
        )
    os.unlink(path)


def selected_cases(case_ids: list[str] | None) -> tuple[QualityCase, ...]:
    if not case_ids:
        return CASES
    by_id = {case.case_id: case for case in CASES}
    return tuple(by_id[case_id] for case_id in case_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser(
        "audit-checkpoint",
        help="validate a native or immutable staged production EXL3 checkpoint",
    )
    audit.add_argument("--checkpoint", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--replace", action="store_true")
    audit.add_argument(
        "--allow-development-unqualified",
        action="store_true",
        help="accept an explicitly marked development-only GPTQModel snapshot",
    )

    collect = subparsers.add_parser("collect", help="collect one checkpoint run")
    collect.add_argument(
        "--url", default="http://127.0.0.1:8000/v1/chat/completions"
    )
    collect.add_argument(
        "--model",
        help=(
            "API model identity; defaults to <staged-model-id>-full for an "
            "immutable EXL3 checkpoint and the official Flash identity for native"
        ),
    )
    collect.add_argument("--artifact-label", required=True)
    collect.add_argument("--checkpoint", required=True)
    collect.add_argument("--expected-spark-targets", type=int, default=4)
    collect.add_argument(
        "--expected-dspark", choices=("on", "off"), default="on"
    )
    collect.add_argument("--concurrency", type=int, choices=range(1, 5), default=1)
    collect.add_argument("--repeats", type=int, default=1)
    collect.add_argument("--timeout", type=float, default=300.0)
    collect.add_argument("--case", action="append", choices=[case.case_id for case in CASES])
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--replace", action="store_true")
    collect.add_argument(
        "--allow-development-unqualified",
        action="store_true",
        help="accept an explicitly marked development-only GPTQModel snapshot",
    )

    arm = subparsers.add_parser(
        "collect-arm",
        help="collect C1 through C4 for one artifact and dSpark state",
    )
    arm.add_argument(
        "--url", default="http://127.0.0.1:8000/v1/chat/completions"
    )
    arm.add_argument(
        "--model",
        help=(
            "API model identity; defaults to <staged-model-id>-full for an "
            "immutable EXL3 checkpoint and the official Flash identity for native"
        ),
    )
    arm.add_argument("--artifact-label", required=True)
    arm.add_argument("--checkpoint", required=True)
    arm.add_argument("--expected-spark-targets", type=int, default=4)
    arm.add_argument("--expected-dspark", choices=("on", "off"), required=True)
    arm.add_argument("--repeats", type=int, default=1)
    arm.add_argument("--timeout", type=float, default=300.0)
    arm.add_argument("--case", action="append", choices=[case.case_id for case in CASES])
    arm.add_argument("--output-dir", type=Path, required=True)
    arm.add_argument("--output", type=Path, required=True)
    arm.add_argument("--replace", action="store_true")
    arm.add_argument(
        "--allow-development-unqualified",
        action="store_true",
        help="accept an explicitly marked development-only GPTQModel snapshot",
    )

    reference = subparsers.add_parser(
        "collect-reference", help="collect the suite from an Anthropic-compatible API"
    )
    reference.add_argument("--url", required=True)
    reference.add_argument("--model", default="deepseek-v4-flash")
    reference.add_argument("--artifact-label", default="official-flash-api")
    reference.add_argument("--auth-token-env", default="ANTHROPIC_AUTH_TOKEN")
    reference.add_argument("--repeats", type=int, default=1)
    reference.add_argument("--timeout", type=float, default=300.0)
    reference.add_argument("--minimum-max-tokens", type=int, default=256)
    reference.add_argument(
        "--thinking-mode",
        choices=("disabled", "omit"),
        default="disabled",
        help="disable thinking for parity with the local suite, or omit the API field",
    )
    reference.add_argument(
        "--case", action="append", choices=[case.case_id for case in CASES]
    )
    reference.add_argument("--output", type=Path, required=True)
    reference.add_argument("--replace", action="store_true")

    compare = subparsers.add_parser("compare", help="compare two collected runs")
    compare.add_argument("--native", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument(
        "--min-candidate-pass-rate",
        type=float,
        help="candidate pass-rate floor (default: the measured native pass rate)",
    )
    compare.add_argument("--max-quality-regressions", type=int, default=0)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--replace", action="store_true")

    matrix = subparsers.add_parser(
        "compare-matrix", help="compare dSpark on/off at every concurrency in C1..4"
    )
    matrix.add_argument("--native-dir", type=Path, required=True)
    matrix.add_argument("--candidate-dir", type=Path, required=True)
    matrix.add_argument(
        "--min-candidate-pass-rate",
        type=float,
        help="candidate pass-rate floor for every cell (default: each native cell)",
    )
    matrix.add_argument("--max-quality-regressions", type=int, default=0)
    matrix.add_argument("--output", type=Path, required=True)
    matrix.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arm_outputs = (
        {
            concurrency: args.output_dir
            / f"dspark-{args.expected_dspark}-c{concurrency}.json"
            for concurrency in range(1, 5)
        }
        if args.command == "collect-arm"
        else {}
    )
    if args.output.resolve() in {output.resolve() for output in arm_outputs.values()}:
        raise SystemExit("collect-arm summary output collides with a cell report")
    try:
        prepare_report_output(args.output, replace=args.replace)
        for output in arm_outputs.values():
            prepare_report_output(output, replace=args.replace)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.command == "audit-checkpoint":
        report = audit_checkpoint(
            args.checkpoint,
            allow_development_unqualified=args.allow_development_unqualified,
        )
    elif args.command == "collect":
        report = collect_run(
            url=args.url,
            model=args.model,
            artifact_label=args.artifact_label,
            checkpoint=args.checkpoint,
            cases=selected_cases(args.case),
            repeats=args.repeats,
            timeout=args.timeout,
            expected_spark_targets=args.expected_spark_targets,
            expected_dspark=args.expected_dspark,
            concurrency=args.concurrency,
            allow_development_unqualified=args.allow_development_unqualified,
        )
    elif args.command == "collect-arm":
        report, arm_reports = collect_matrix_arm(
            url=args.url,
            model=args.model,
            artifact_label=args.artifact_label,
            checkpoint=args.checkpoint,
            cases=selected_cases(args.case),
            repeats=args.repeats,
            timeout=args.timeout,
            expected_spark_targets=args.expected_spark_targets,
            expected_dspark=args.expected_dspark,
            allow_development_unqualified=args.allow_development_unqualified,
        )
        report["reports"] = {
            str(concurrency): str(arm_outputs[concurrency])
            for concurrency in range(1, 5)
        }
        for concurrency, arm_report in arm_reports.items():
            write_report(arm_outputs[concurrency], arm_report)
    elif args.command == "collect-reference":
        auth_token = os.environ.get(args.auth_token_env, "")
        if not auth_token:
            raise SystemExit(f"{args.auth_token_env} is not set")
        report = collect_reference_run(
            url=args.url,
            model=args.model,
            artifact_label=args.artifact_label,
            cases=selected_cases(args.case),
            repeats=args.repeats,
            timeout=args.timeout,
            minimum_max_tokens=args.minimum_max_tokens,
            thinking_mode=args.thinking_mode,
            auth_token=auth_token,
        )
    elif args.command == "compare":
        if args.min_candidate_pass_rate is not None and not (
            0.0 <= args.min_candidate_pass_rate <= 1.0
        ):
            raise SystemExit("--min-candidate-pass-rate must be in 0..1")
        if args.max_quality_regressions < 0:
            raise SystemExit("--max-quality-regressions must be nonnegative")
        report = compare_runs(
            json.loads(args.native.read_text(encoding="utf-8")),
            json.loads(args.candidate.read_text(encoding="utf-8")),
            min_candidate_pass_rate=args.min_candidate_pass_rate,
            max_quality_regressions=args.max_quality_regressions,
        )
    else:
        if args.min_candidate_pass_rate is not None and not (
            0.0 <= args.min_candidate_pass_rate <= 1.0
        ):
            raise SystemExit("--min-candidate-pass-rate must be in 0..1")
        if args.max_quality_regressions < 0:
            raise SystemExit("--max-quality-regressions must be nonnegative")
        report = compare_matrix(
            args.native_dir,
            args.candidate_dir,
            min_candidate_pass_rate=args.min_candidate_pass_rate,
            max_quality_regressions=args.max_quality_regressions,
        )
    write_report(args.output, report)
    print(json.dumps(report["summary"], sort_keys=True))
    if args.command in {"compare", "compare-matrix"} and not report["summary"]["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
