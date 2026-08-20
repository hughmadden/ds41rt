#!/usr/bin/env python3
"""Serve authenticated, content-addressed EXL3 projection work on one Spark."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
from gptqmodel.utils.exl3_projection_checkpoint import (
    EXL3ProjectionCheckpointStore,
    canonical_json_bytes,
)
from gptqmodel.utils.exl3_remote import (
    DEFAULT_MAX_BODY_BYTES,
    REMOTE_CONTRACT,
    REMOTE_REQUEST_SCHEMA,
    REMOTE_RESULT_SCHEMA,
    decode_tensor_envelope,
    encode_tensor_envelope,
    execute_remote_projection,
)
from preflight import report_identity_sha256


class WorkerError(RuntimeError):
    """The worker cannot establish its immutable execution identity."""


def unexpected_failure_evidence(error: Exception) -> tuple[str, str]:
    """Return a bounded signed response message and a useful worker traceback."""

    trace = "".join(traceback.format_exception(error))
    if len(trace) > 8192:
        trace = "[traceback truncated to final 8192 characters]\n" + trace[-8192:]
    trace_sha256 = hashlib.sha256(trace.encode()).hexdigest()
    detail = str(error).replace("\r", " ").replace("\n", " ")[:512]
    message = (
        f"unexpected worker failure: {type(error).__name__}: {detail}; "
        f"traceback_sha256={trace_sha256}"
    )
    return message, trace


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WorkerError(f"cannot read worker preflight {path}") from error
    if not isinstance(value, dict):
        raise WorkerError("worker preflight is not a JSON object")
    return value


def build_worker_identity(name: str, preflight_path: Path) -> dict[str, Any]:
    preflight_path = preflight_path.expanduser().resolve(strict=True)
    preflight = read_json_object(preflight_path)
    image_digest = preflight.get("image_digest")
    gptqmodel = preflight.get("gptqmodel")
    gpus = preflight.get("gpus")
    if (
        not name
        or preflight.get("status") != "qualified"
        or preflight.get("role") != "expert"
        or preflight.get("target_platform") != "linux/arm64"
        or preflight.get("cuda_arch") != "121"
        or preflight.get("python", {}).get("gil_enabled") is not False
        or not isinstance(image_digest, str)
        or not image_digest.startswith("sha256:")
        or not isinstance(gptqmodel, dict)
        or not isinstance(gpus, list)
        or len(gpus) != 1
        or gpus[0].get("index") != 0
    ):
        raise WorkerError("Spark preflight does not satisfy the remote-worker contract")
    return {
        "contract": REMOTE_CONTRACT,
        "name": name,
        "preflight_sha256": report_identity_sha256(preflight),
        "image_digest": image_digest,
        "gptqmodel": gptqmodel,
        "gpu": gpus[0],
        "python": preflight["python"],
        "torch": preflight.get("torch"),
    }


class EXL3WorkerServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address,
        *,
        identity: dict[str, Any],
        token: bytes,
        checkpoint_root: Path,
        max_body_bytes: int,
    ) -> None:
        self.identity = identity
        self.token = token
        self.checkpoint_store = EXL3ProjectionCheckpointStore(checkpoint_root)
        self.max_body_bytes = max_body_bytes
        self.quantize_lock = threading.Lock()
        super().__init__(server_address, EXL3WorkerHandler)


class EXL3WorkerHandler(BaseHTTPRequestHandler):
    server: EXL3WorkerServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format_string, *args) -> None:
        print(
            f"exl3-worker[{self.server.identity['name']}]: "
            + format_string % args,
            flush=True,
        )

    def _signature_valid(self, payload: bytes) -> bool:
        claimed = self.headers.get("X-DS4RT-Signature")
        expected = hmac.new(self.server.token, payload, hashlib.sha256).hexdigest()
        return isinstance(claimed, str) and hmac.compare_digest(claimed, expected)

    def _send(self, status: int, payload: bytes, content_type: str) -> None:
        signature = hmac.new(
            self.server.token,
            payload,
            hashlib.sha256,
        ).hexdigest()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-DS4RT-Signature", signature)
        self.end_headers()
        self.wfile.write(payload)

    def _send_error(self, status: int, message: str) -> None:
        payload = canonical_json_bytes({"status": "error", "message": message})
        self._send(status, payload, "application/json")

    def do_GET(self) -> None:
        if self.path != "/v1/identity":
            self._send_error(404, "not found")
            return
        if not self._signature_valid(b"GET /v1/identity"):
            self._send_error(401, "authentication failed")
            return
        self._send(
            200,
            canonical_json_bytes(self.server.identity),
            "application/json",
        )

    def do_POST(self) -> None:
        if self.path != "/v1/exl3/quantize":
            self._send_error(404, "not found")
            return
        try:
            content_length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._send_error(400, "invalid content length")
            return
        if content_length <= 0 or content_length > self.server.max_body_bytes:
            self._send_error(413, "request is too large")
            return
        payload = self.rfile.read(content_length)
        if len(payload) != content_length:
            self._send_error(400, "request is truncated")
            return
        if not self._signature_valid(payload):
            self._send_error(401, "authentication failed")
            return
        try:
            manifest, tensors = decode_tensor_envelope(
                payload,
                max_body_bytes=self.server.max_body_bytes,
            )
            if (
                manifest.get("schema") != REMOTE_REQUEST_SCHEMA
                or manifest.get("contract") != REMOTE_CONTRACT
                or not isinstance(manifest.get("request"), dict)
            ):
                raise ValueError("remote request manifest is inconsistent")
            queue_started = time.perf_counter()
            with self.server.quantize_lock:
                queue_wait_seconds = time.perf_counter() - queue_started
                out_tensors, result, checkpoint_hit = execute_remote_projection(
                    request=manifest["request"],
                    tensors=tensors,
                    device="cuda:0",
                    worker_identity=self.server.identity,
                    checkpoint_store=self.server.checkpoint_store,
                )
            result = dict(result)
            result["worker_queue_wait_seconds"] = queue_wait_seconds
            response = encode_tensor_envelope(
                {
                    "schema": REMOTE_RESULT_SCHEMA,
                    "contract": REMOTE_CONTRACT,
                    "request_sha256": manifest["request"]["request_sha256"],
                    "checkpoint_hit": checkpoint_hit,
                    "result": result,
                },
                out_tensors,
            )
        except (ValueError, RuntimeError) as error:
            self.log_message("request rejected: %s", str(error))
            self._send_error(400, str(error))
            return
        except Exception as error:
            message, trace = unexpected_failure_evidence(error)
            self.log_message("%s\n%s", message, trace)
            self._send_error(500, message)
            return
        self._send(200, response, "application/octet-stream")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=17841)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--token-env", default="DS4RT_EXL3_WORKER_TOKEN")
    parser.add_argument("--max-body-bytes", type=int, default=DEFAULT_MAX_BODY_BYTES)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535 or args.max_body_bytes <= 0:
        parser.error("invalid port or body-size limit")
    return args


def main() -> int:
    args = parse_args()
    token_value = os.environ.get(args.token_env)
    if not token_value:
        raise WorkerError(f"worker token env `{args.token_env}` is unset")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise WorkerError("Spark worker requires exactly one visible CUDA device")
    identity = build_worker_identity(args.name, args.preflight_report)
    server = EXL3WorkerServer(
        (args.host, args.port),
        identity=identity,
        token=token_value.encode(),
        checkpoint_root=args.checkpoint_root,
        max_body_bytes=args.max_body_bytes,
    )
    print(
        json.dumps(
            {
                "status": "serving",
                "address": list(server.server_address),
                "identity": identity,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
