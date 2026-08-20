#!/usr/bin/env python3
"""Shared model and prompt contracts for DeepSeek V4 benchmark clients."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DEFAULT_FLASH_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEFAULT_FLASH_FULL_MODEL_ID = f"{DEFAULT_FLASH_MODEL_ID}-full"
FLASH_HF_CACHE_NAME = "models--deepseek-ai--DeepSeek-V4-Flash-0731"

DS4_BOS = "<｜begin▁of▁sentence｜>"
DS4_EOS = "<｜end▁of▁sentence｜>"
DS4_USER = "<｜User｜>"
DS4_ASSISTANT = "<｜Assistant｜>"
DS4_THINK_OPEN = "<think>"
DS4_THINK_CLOSE = "</think>"


def huggingface_hub_root() -> Path:
    explicit_hub = os.environ.get("HF_HUB_CACHE")
    if explicit_hub:
        return Path(explicit_hub).expanduser()
    hf_home = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    ).expanduser()
    return hf_home / "hub"


def default_flash_tokenizer_path() -> Path:
    model_root = huggingface_hub_root() / FLASH_HF_CACHE_NAME
    snapshots = model_root / "snapshots"
    ref_path = model_root / "refs" / "main"
    if ref_path.is_file():
        revision = ref_path.read_text(encoding="utf-8").strip()
        if (
            revision
            and revision not in {".", ".."}
            and "/" not in revision
            and "\\" not in revision
        ):
            selected = snapshots / revision / "tokenizer.json"
            if selected.is_file():
                return selected
    candidates = sorted(snapshots.glob("*/tokenizer.json"))
    if not candidates:
        raise FileNotFoundError(
            "no local DeepSeek-V4-Flash-0731 tokenizer.json; pass --tokenizer"
        )
    return candidates[-1]


def render_simple_user_prompt(content: str, *, thinking: bool = False) -> str:
    if not thinking:
        return render_nonthinking_messages(
            [{"role": "user", "content": content}]
        )
    return f"{DS4_BOS}{DS4_USER}{content}{DS4_ASSISTANT}{DS4_THINK_OPEN}"


def render_nonthinking_messages(messages: list[dict[str, Any]]) -> str:
    """Render the simple role subset used by deterministic qualification."""
    merged: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"DeepSeek benchmark renderer does not support role {role!r}")
        content = message.get("content") or ""
        if not isinstance(content, str):
            raise ValueError(f"DeepSeek benchmark {role} content must be text")
        if role == "assistant" and DS4_THINK_CLOSE in content:
            content = content.split(DS4_THINK_CLOSE, 1)[1]
        if role == "user" and merged and merged[-1]["role"] == "user":
            merged[-1]["parts"].append(content)
        elif role == "user":
            merged.append({"role": role, "parts": [content]})
        else:
            merged.append(
                {
                    "role": role,
                    "content": content,
                    "wo_eos": bool(message.get("wo_eos", False)),
                }
            )

    rendered = DS4_BOS
    for index, message in enumerate(merged):
        role = message["role"]
        if role == "system":
            rendered += message["content"]
        elif role == "user":
            rendered += DS4_USER + "\n\n".join(message["parts"])
            next_role = merged[index + 1]["role"] if index + 1 < len(merged) else None
            if next_role == "assistant" or next_role is None:
                rendered += DS4_ASSISTANT + DS4_THINK_CLOSE
        else:
            rendered += message["content"]
            if not message["wo_eos"]:
                rendered += DS4_EOS
    return rendered
