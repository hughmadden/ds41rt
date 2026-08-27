#!/usr/bin/env python3
"""Forward an OpenAI-compatible API byte-for-byte and record terminal metrics."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from aiohttp import ClientSession, ClientTimeout, web


HOP_HEADERS = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", default="127.0.0.1:8001")
    parser.add_argument("--upstream", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def response_headers(headers: Any) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_HEADERS
    }


async def main() -> None:
    args = parse_args()
    host, raw_port = args.listen.rsplit(":", 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = args.output.open("a", encoding="utf-8", buffering=1)
    lock = asyncio.Lock()
    session = ClientSession(timeout=ClientTimeout(total=None))

    async def forward(request: web.Request) -> web.StreamResponse:
        body = await request.read()
        request_json: dict[str, Any] = {}
        if request.content_type == "application/json" and body:
            try:
                parsed = json.loads(body)
                if isinstance(parsed, dict):
                    request_json = parsed
            except json.JSONDecodeError:
                pass
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in HOP_HEADERS and key.lower() != "host"
        }
        url = f"{args.upstream.rstrip('/')}/{request.match_info['tail']}"
        async with session.request(
            request.method,
            url,
            params=request.query,
            headers=headers,
            data=body,
        ) as upstream:
            response = web.StreamResponse(
                status=upstream.status,
                reason=upstream.reason,
                headers=response_headers(upstream.headers),
            )
            await response.prepare(request)
            terminal: dict[str, Any] | None = None
            pending = b""
            async for chunk in upstream.content.iter_any():
                await response.write(chunk)
                if request.path.endswith("/chat/completions"):
                    pending += chunk
                    while b"\n\n" in pending:
                        event, pending = pending.split(b"\n\n", 1)
                        for line in event.splitlines():
                            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                                continue
                            try:
                                candidate = json.loads(line[6:])
                            except json.JSONDecodeError:
                                continue
                            if isinstance(candidate, dict) and isinstance(
                                candidate.get("metrics"), dict
                            ):
                                terminal = candidate
            await response.write_eof()
            if terminal is not None:
                metrics = terminal["metrics"]
                real_full = metrics.get("real_full", {})
                record = {
                    "id": terminal.get("id"),
                    "status": upstream.status,
                    "requested_model": request_json.get("model"),
                    "requested_max_tokens": request_json.get("max_tokens"),
                    "requested_min_tokens": request_json.get("min_tokens"),
                    "requested_ignore_eos": request_json.get("ignore_eos"),
                    "requested_stream": request_json.get("stream"),
                    "requested_temperature": request_json.get("temperature"),
                    "requested_enable_thinking": request_json.get(
                        "enable_thinking"
                    ),
                    "prompt_tokens": metrics.get("prompt_tokens"),
                    "output_tokens": metrics.get("output_tokens"),
                    "mtp_verify_cycles": real_full.get("mtp_verify_cycles"),
                    "mtp_draft_tokens": real_full.get("mtp_draft_tokens"),
                    "mtp_accepted_draft_tokens": real_full.get(
                        "mtp_accepted_draft_tokens"
                    ),
                }
                async with lock:
                    output.write(json.dumps(record, sort_keys=True) + "\n")
            return response

    app = web.Application(client_max_size=64 * 1024**2)
    app.router.add_route("*", "/{tail:.*}", forward)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, int(raw_port))
    await site.start()
    print(
        json.dumps(
            {"listen": args.listen, "upstream": args.upstream, "output": str(args.output)}
        ),
        flush=True,
    )
    try:
        await asyncio.Event().wait()
    finally:
        await session.close()
        output.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
