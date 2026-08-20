#!/usr/bin/env python3
"""Offline safetensors catalog inspection for native DeepSeek V4 snapshots."""

from __future__ import annotations

import argparse
import json
import os
import struct
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEFAULT_HOSTS = ["ostrich", "dodo", "emu", "kiwi"]


def hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def model_cache(model_id: str) -> Path:
    return hf_home() / "hub" / f"models--{model_id.replace('/', '--')}"


def resolve_snapshot(model_id: str) -> Path:
    snapshots = sorted((model_cache(model_id) / "snapshots").glob("*"))
    snapshots = [path for path in snapshots if path.is_dir()]
    if not snapshots:
        raise SystemExit(f"no local snapshot found for {model_id}")
    return snapshots[-1]


def read_config(snapshot: Path, model_id: str) -> dict[str, Any]:
    with (snapshot / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    if config.get("model_type") != "deepseek_v4":
        raise SystemExit(f"unsupported model_type for {model_id}: {config.get('model_type')!r}")
    quant = config.get("quantization_config", {})
    method = str(quant.get("quant_method", quant.get("quant_algo", "unquantized"))).lower()
    expert_dtype = str(config.get("expert_dtype", "unknown")).lower()
    if method == "exl3":
        recipe = str(
            quant.get("ds4rt", {}).get(
                "recipe", "invalid_deepseek_v4_exl3_recipe"
            )
        )
    elif method == "fp8" and expert_dtype == "fp4":
        recipe = "deepseek_v4_native_fp4_fp8_mixed_v1"
    else:
        recipe = f"deepseek_v4_{expert_dtype}_{method}_v1"
    dimensions = (
        config.get("hidden_size"),
        config.get("num_hidden_layers"),
        config.get("n_routed_experts"),
    )
    if dimensions == (4096, 43, 256):
        variant = "flash"
    elif dimensions == (7168, 61, 384):
        variant = "pro"
    else:
        variant = "custom"
    return {
        "model_id": model_id,
        "model_type": config["model_type"],
        "variant": variant,
        "hidden_size": config["hidden_size"],
        "num_hidden_layers": config["num_hidden_layers"],
        "num_hash_layers": config["num_hash_layers"],
        "first_k_dense_replace": 0,
        "routed_experts": config["n_routed_experts"],
        "top_k": config["num_experts_per_tok"],
        "moe_intermediate_size": config["moe_intermediate_size"],
        "compress_ratios": config["compress_ratios"],
        "dspark_target_layer_ids": config["dspark_target_layer_ids"],
        "dspark_markov_rank": config["dspark_markov_rank"],
        "quantization_recipe": recipe,
    }


def parse_safetensors_header(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    data_start = 8 + header_len
    out: dict[str, dict[str, Any]] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        out[name] = {
            "dtype": dtype_from_safetensors(meta["dtype"]),
            "shape": meta["shape"],
            "byte_offset": data_start + start,
            "byte_length": end - start,
        }
    return out


def dtype_from_safetensors(dtype: str) -> str:
    return {
        "BF16": "bf16",
        "F16": "f16",
        "F32": "f32",
        "F8_E4M3": "f8e4m3",
        "F8_E5M2": "f8e5m2",
        "F8_E8M0": "f8e8m0",
        "I8": "i8",
        "I16": "i16",
        "I32": "i32",
        "I64": "i64",
        "U8": "u8",
        "F4": "f4",
    }.get(dtype, f"unknown:{dtype}")


def extract_number(name: str, marker: str) -> int | None:
    if marker not in name:
        return None
    tail = name.split(marker, 1)[1]
    digits = []
    for ch in tail:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    return int("".join(digits)) if digits else None


def is_quantization_tensor(name: str) -> bool:
    return (
        name.endswith(".input_scale")
        or name.endswith(".weight_scale")
        or name.endswith(".weight_scale_2")
        or name.endswith(".scale")
        or name.endswith(".trellis_mcg")
    )


def classify(name: str, layer_id: int | None, expert_id: int | None, facts: dict[str, Any]) -> str:
    if expert_id is not None and (".ffn.experts." in name or ".mlp.experts." in name):
        return "routed-expert"
    if name.startswith("mtp."):
        return "dspark"
    if ".ffn.shared_experts." in name or ".mlp.shared_experts." in name:
        return "shared-expert"
    if ".ffn.gate." in name or ".mlp.gate." in name:
        return "router"
    if name in {"embed.weight", "model.embed_tokens.weight"}:
        return "embedding"
    if name in {"head.weight", "lm_head.weight"}:
        return "lm-head"
    if ".attn.indexer." in name:
        return "attention-indexer"
    if ".attn.compressor." in name:
        return "attention-compressor"
    if ".attn." in name or ".self_attn." in name:
        return "attention"
    if name.startswith("hc_head_") or ".hc_" in name:
        return "hyper-connection"
    if (
        "layernorm" in name
        or name == "norm.weight"
        or name.endswith("_norm.weight")
        or name.endswith(".norm.weight")
        or ".norm." in name
    ):
        return "norm"
    if ".ffn." in name or ".mlp." in name:
        return "dense-mlp"
    if layer_id is not None and layer_id >= int(facts["num_hidden_layers"]):
        return "dspark" if facts.get("model_type") == "deepseek_v4" else "mtp"
    return "other"


def build_catalog(model_id: str) -> dict[str, Any]:
    snapshot = resolve_snapshot(model_id)
    facts = read_config(snapshot, model_id)
    with (snapshot / "model.safetensors.index.json").open("r", encoding="utf-8") as f:
        index = json.load(f)["weight_map"]
    files = sorted(set(index.values()))
    headers: dict[str, tuple[str, dict[str, Any]]] = {}
    for file_name in files:
        for name, meta in parse_safetensors_header(snapshot / file_name).items():
            headers[name] = (file_name, meta)

    tensors = []
    for name, file_name in index.items():
        header_file, meta = headers[name]
        if header_file != file_name:
            raise SystemExit(f"index/header mismatch for {name}: {file_name} != {header_file}")
        layer_id = extract_number(name, "layers.")
        mtp_block_id = extract_number(name, "mtp.")
        if layer_id is None and mtp_block_id is not None:
            layer_id = int(facts["num_hidden_layers"]) + mtp_block_id
        expert_id = extract_number(name, ".ffn.experts.")
        if expert_id is None:
            expert_id = extract_number(name, ".mlp.experts.")
        quant = is_quantization_tensor(name)
        tensors.append(
            {
                "name": name,
                "file": file_name,
                "dtype": meta["dtype"],
                "shape": meta["shape"],
                "byte_offset": meta["byte_offset"],
                "byte_length": meta["byte_length"],
                "role": classify(name, layer_id, expert_id, facts),
                "layer_id": layer_id,
                "expert_id": expert_id,
                "is_quantization_metadata": quant,
            }
        )
    tensors.sort(key=lambda item: item["name"])
    return {
        "model_id": model_id,
        "snapshot_path": str(snapshot),
        "facts": facts,
        "tensors": tensors,
    }


def write_summary(catalog: dict[str, Any], path: Path) -> None:
    counts = Counter(t["role"] for t in catalog["tensors"])
    routed_quant = sum(
        1
        for t in catalog["tensors"]
        if t["role"] == "routed-expert" and t["is_quantization_metadata"]
    )
    lines = [
        "# Tensor Classification Summary",
        "",
        f"- Model: `{catalog['model_id']}`",
        f"- Snapshot: `{catalog['snapshot_path']}`",
        f"- Tensor count: `{len(catalog['tensors'])}`",
        f"- Hidden size: `{catalog['facts']['hidden_size']}`",
        f"- Hidden layers: `{catalog['facts']['num_hidden_layers']}`",
        f"- Variant: `{catalog['facts']['variant']}`",
        f"- Hash-routed MoE layers: `{catalog['facts']['num_hash_layers']}`",
        f"- Routed experts per MoE layer: `{catalog['facts']['routed_experts']}`",
        f"- Top-k experts per token: `{catalog['facts']['top_k']}`",
        f"- Quantization recipe: `{catalog['facts']['quantization_recipe']}`",
        "",
        "## Role Counts",
        "",
        "| Role | Tensors |",
        "| --- | ---: |",
    ]
    for role, count in sorted(counts.items()):
        lines.append(f"| {role} | {count} |")
    lines.extend(
        [
            "",
            "## Routed Expert Detail",
            "",
            f"- Routed expert quantization tensors: `{routed_quant}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    catalog = build_catalog(args.model_id)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(catalog, separators=(",", ":")), encoding="utf-8")
    write_summary(catalog, args.summary)
    print(f"wrote catalog tensors={len(catalog['tensors'])} out={args.out}")


if __name__ == "__main__":
    main()
