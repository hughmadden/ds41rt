#!/usr/bin/env python3
"""Transparent OpenAI HTTP proxy that records terminal DS4RT metrics.

Use this only to capture response counters. Benchmark latency should come from
a direct server connection so the proxy cannot perturb timing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiohttp import ClientSession, web


HOP_BY_HOP_HEADERS = {
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--upstream", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def terminal_metrics(event: dict) -> dict | None:
    metrics = event.get("metrics")
    if not isinstance(metrics, dict):
        return None
    real_full = metrics.get("real_full")
    if not isinstance(real_full, dict):
        return None
    return {
        "id": event.get("id"),
        "prompt_tokens": metrics.get("prompt_tokens"),
        "output_tokens": metrics.get("output_tokens"),
        "mtp_verify_cycles": real_full.get("mtp_verify_cycles"),
        "mtp_draft_tokens": real_full.get("mtp_draft_tokens"),
        "mtp_accepted_draft_tokens": real_full.get("mtp_accepted_draft_tokens"),
        "mtp_draft_lengths": real_full.get("mtp_draft_lengths"),
        "mtp_accepted_draft_lengths": real_full.get("mtp_accepted_draft_lengths"),
    }


async def make_app(args: argparse.Namespace) -> web.Application:
    app = web.Application()
    app["session"] = ClientSession()
    app["output"] = args.output
    app["upstream"] = args.upstream.rstrip("/")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("", encoding="utf-8")

    async def proxy(request: web.Request) -> web.StreamResponse:
        body = await request.read()
        request_payload = None
        if body:
            try:
                request_payload = json.loads(body)
            except json.JSONDecodeError:
                pass

        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "host"
        }
        url = f"{app['upstream']}/{request.match_info['tail']}"
        async with app["session"].request(
            request.method,
            url,
            params=request.query,
            data=body,
            headers=headers,
        ) as upstream:
            response_headers = {
                key: value
                for key, value in upstream.headers.items()
                if key.lower() not in HOP_BY_HOP_HEADERS
            }
            response = web.StreamResponse(
                status=upstream.status,
                reason=upstream.reason,
                headers=response_headers,
            )
            await response.prepare(request)

            buffer = ""
            records: list[dict] = []
            async for chunk in upstream.content.iter_any():
                await response.write(chunk)
                buffer += chunk.decode("utf-8", errors="replace")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        record = terminal_metrics(json.loads(payload))
                    except json.JSONDecodeError:
                        record = None
                    if record is not None:
                        records.append(record)

            for record in records:
                if isinstance(request_payload, dict):
                    record["requested_max_tokens"] = request_payload.get("max_tokens")
                    record["requested_stream"] = request_payload.get("stream")
                    record["requested_temperature"] = request_payload.get("temperature")
                with app["output"].open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record, sort_keys=True) + "\n")
            await response.write_eof()
            return response

    async def close_session(_: web.Application) -> None:
        await app["session"].close()

    app.router.add_route("*", "/{tail:.*}", proxy)
    app.on_cleanup.append(close_session)
    return app


def main() -> None:
    args = parse_args()
    web.run_app(make_app(args), host=args.listen, port=args.port)


if __name__ == "__main__":
    main()
