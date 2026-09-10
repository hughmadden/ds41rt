#!/usr/bin/env python3
"""Prepare an atomic, standard-only Hugging Face model publication tree."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ds41rt_runtime.exl3_artifact_contract import (  # noqa: E402
    validate_inline_mixed_policy,
)


CONFIG_MAX_BYTES = 1024 * 1024
SHARD_RE = re.compile(r"model-[0-9]{5}-of-[0-9]{5}\.safetensors\Z")
PUBLIC_STATIC_FILES = (
    ".gitattributes",
    "LICENSE",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "quantize_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
UNFINISHED_README_MARKERS = (
    "DS41RT_PUBLICATION_RESULTS_PENDING",
    "TODO_PUBLICATION",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--readme", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="hardlink avoids duplicating the model bytes (default: %(default)s)",
    )
    return parser.parse_args()


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def referenced_shards(snapshot: Path) -> tuple[tuple[str, ...], int]:
    index = read_json_object(snapshot / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model index has no weight map")
    shards = set()
    for tensor, raw in weight_map.items():
        if not isinstance(tensor, str) or not tensor:
            raise ValueError("model index has an invalid tensor name")
        if not isinstance(raw, str):
            raise ValueError(f"model index has an invalid shard for {tensor}")
        path = PurePosixPath(raw)
        if path.name != raw or SHARD_RE.fullmatch(raw) is None:
            raise ValueError(f"model index has a nonstandard shard path: {raw!r}")
        shards.add(raw)
    metadata = index.get("metadata")
    total_size = metadata.get("total_size") if isinstance(metadata, dict) else None
    if isinstance(total_size, bool) or not isinstance(total_size, int) or total_size <= 0:
        raise ValueError("model index has no positive metadata.total_size")
    return tuple(sorted(shards)), total_size


def public_quantization_configs(snapshot: Path) -> tuple[bytes, bytes]:
    """Return compact model and external EXL3 configs from one exact source."""

    config_path = snapshot / "config.json"
    config = read_json_object(config_path)
    embedded = config.get("quantization_config")
    if not isinstance(embedded, dict) or embedded.get("quant_method") != "exl3":
        raise ValueError("config.json does not declare EXL3 quantization")

    full_external = read_json_object(snapshot / "quantize_config.json")
    source_declaration = dict(full_external)
    storage = source_declaration.pop("tensor_storage", None)
    if not isinstance(storage, dict) or not storage:
        raise ValueError("quantize_config.json has no EXL3 tensor-storage map")
    if "tensor_storage" in embedded:
        if embedded != full_external:
            raise ValueError(
                "embedded full EXL3 config differs from quantize_config.json"
            )
    elif source_declaration != embedded:
        raise ValueError(
            "config.json quantization declaration differs from quantize_config.json"
        )

    # Internal artifact validation binds the large error ledger through this
    # metadata.  The public model has neither that ledger nor any reason to
    # expose local paths and build topology, matching the corrected v0 layout.
    public_external = dict(full_external)
    bits = public_external.get("bits")
    if (
        isinstance(bits, bool)
        or not isinstance(bits, (int, float))
        or not float(bits).is_integer()
        or int(bits) not in (2, 3)
    ):
        raise ValueError(
            "public GPTQModel EXL3 bits must name the integer K2 or K3 base tier"
        )
    # Hugging Face quantization integrations treat this as a discrete tier.
    # GPTQModel historically rendered it as 2.0/3.0, which is numerically
    # equivalent in Python but rejected by standard schema consumers. Mixed
    # average bitrate remains in the exact-rational inline-mixed metadata.
    public_external["bits"] = int(bits)
    public_meta = dict(public_external.get("meta", {}))
    private_ledger = public_meta.pop("ds41rt_error_ledger", None)
    inline_mixed = public_meta.get("ds41rt_inline_mixed")
    if inline_mixed is not None:
        if isinstance(inline_mixed, dict):
            inline_mixed = dict(inline_mixed)
            inline_mixed.pop("tier_plan_root", None)
        validate_inline_mixed_policy(inline_mixed)
        public_meta["ds41rt_inline_mixed"] = inline_mixed
    family_join = (
        private_ledger.get("family_join")
        if isinstance(private_ledger, dict)
        else None
    )
    namespace_policies = (
        family_join.get("inline_mixed")
        if isinstance(family_join, dict)
        else None
    )
    if namespace_policies is not None:
        if not isinstance(namespace_policies, dict) or not namespace_policies:
            raise ValueError("inline mixed namespace provenance is not an object")
        portable_namespaces: dict[str, dict[str, Any]] = {}
        for namespace, raw_policy in namespace_policies.items():
            if namespace not in {"base", "mtp"} or not isinstance(raw_policy, dict):
                raise ValueError("inline mixed namespace provenance is invalid")
            policy = dict(raw_policy)
            policy.pop("tier_plan_root", None)
            validate_inline_mixed_policy(policy, namespace=namespace)
            portable_namespaces[namespace] = policy
        if inline_mixed is None or portable_namespaces.get("base") != inline_mixed:
            raise ValueError(
                "inline mixed base metadata differs from namespace provenance"
            )
        # The compatibility key above continues to describe the target/base
        # experts. This additive map preserves the integrated dSpark bitrate
        # after the private error ledger is removed from the public artifact.
        public_meta["ds41rt_inline_mixed_namespaces"] = portable_namespaces
    public_external["meta"] = public_meta
    public_declaration = dict(public_external)
    public_declaration.pop("tensor_storage")
    config["quantization_config"] = public_declaration
    rendered_config = (
        json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    if len(rendered_config) > CONFIG_MAX_BYTES:
        raise ValueError(
            f"compact config.json exceeds the {CONFIG_MAX_BYTES}-byte public parser limit"
        )
    rendered_external = (
        json.dumps(public_external, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    return rendered_config, rendered_external


def compact_public_config(snapshot: Path) -> bytes:
    return public_quantization_configs(snapshot)[0]


def validate_quantization_metadata(snapshot: Path) -> None:
    public_quantization_configs(snapshot)


def publication_sources(snapshot: Path, readme: Path) -> tuple[tuple[str, Path], ...]:
    snapshot = snapshot.expanduser().resolve(strict=True)
    readme = readme.expanduser().resolve(strict=True)
    readme_text = readme.read_text(encoding="utf-8") if readme.is_file() else ""
    if not readme_text.startswith("---\n"):
        raise ValueError("README must be a UTF-8 Hugging Face model card")
    if any(marker in readme_text for marker in UNFINISHED_README_MARKERS):
        raise ValueError("README still contains an unfinished publication marker")
    for name in PUBLIC_STATIC_FILES:
        if not (snapshot / name).is_file():
            raise ValueError(f"snapshot is missing required public file {name}")
    validate_quantization_metadata(snapshot)
    shards, _ = referenced_shards(snapshot)
    source_shards = {path.name for path in snapshot.glob("*.safetensors")}
    if source_shards != set(shards):
        raise ValueError(
            "snapshot safetensor shard set differs from the model index: "
            f"extra={sorted(source_shards - set(shards))}, "
            f"missing={sorted(set(shards) - source_shards)}"
        )
    sources = [(name, snapshot / name) for name in PUBLIC_STATIC_FILES]
    sources.append(("README.md", readme))
    sources.extend((name, snapshot / name) for name in shards)
    return tuple(sorted(sources))


def prepare_publication(
    snapshot: Path,
    readme: Path,
    output: Path,
    *,
    link_mode: str = "hardlink",
) -> dict[str, Any]:
    sources = publication_sources(snapshot, readme)
    compact_config, public_quantize_config = public_quantization_configs(
        snapshot.expanduser().resolve(strict=True)
    )
    output = output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"publication output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        for name, source in sources:
            destination = temporary / name
            resolved = source.resolve(strict=True)
            if name == "config.json":
                destination.write_bytes(compact_config)
                continue
            if name == "quantize_config.json":
                destination.write_bytes(public_quantize_config)
                continue
            # Only immutable, multi-gigabyte weight shards need zero-copy
            # staging. Copy metadata and the model card so later edits to a
            # draft README or source-side metadata cannot mutate this tree.
            if link_mode == "hardlink" and name.endswith(".safetensors"):
                os.link(resolved, destination)
            elif link_mode in ("hardlink", "copy"):
                shutil.copy2(resolved, destination)
            else:
                raise ValueError(f"unsupported link mode {link_mode!r}")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    shards, indexed_tensor_bytes = referenced_shards(output)
    names = tuple(sorted(path.name for path in output.iterdir()))
    expected_names = tuple(sorted(name for name, _ in sources))
    if names != expected_names:
        raise RuntimeError("publication output contains an unexpected file surface")
    return {
        "schema": "ds41rt-hf-standard-publication-v1",
        "snapshot": str(snapshot.expanduser().resolve(strict=True)),
        "output": str(output),
        "link_mode": link_mode,
        "files": len(names),
        "shards": len(shards),
        "file_bytes": sum((output / name).stat().st_size for name in names),
        "indexed_tensor_bytes": indexed_tensor_bytes,
        "names": list(names),
    }


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            prepare_publication(
                args.snapshot,
                args.readme,
                args.output,
                link_mode=args.link_mode,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
