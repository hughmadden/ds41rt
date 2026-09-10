#!/usr/bin/env python3
"""Require RTX/Spark EXL3 parity for one real Flash expert family.

This is a hardware-parity control, not production calibration evidence. It
uses checkpoint-native w1/w2/w3 weights and the retained legacy Flash
activation capture to construct real-shape Hessians, then quantizes every
projection independently on coordinator GPU 0 and on one explicit Spark
parity reference. Promotion requires bit-repeatability
within each hardware cohort and declared numerical equivalence across SM120
and SM121.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Any, Iterable, NamedTuple

from preflight import report_identity_sha256


SCHEMA = "ds41rt.exl3-real-family-remote-parity-v1"
CALIBRATION_SCOPE = "legacy_real_activation_hardware_parity_not_production"
FAMILY_JOIN_CONTRACT = "ds41rt.exl3-real-family-parity-v1"
EXL3_SEED = 787
SIGMA_REG = 0.025
CROSS_DEVICE_MIN_COSINE = 0.995
CROSS_DEVICE_MAX_RELATIVE_L2 = 0.10
CROSS_DEVICE_MAX_PROXY_RELATIVE_DELTA = 0.005
CROSS_DEVICE_MAX_NATIVE_COSINE_DELTA = 0.001
CROSS_DEVICE_MAX_NATIVE_RELATIVE_L2_DELTA = 0.005
PROJECTIONS = (
    ("w1", "gate_proj", "shared"),
    ("w3", "up_proj", "shared"),
    ("w2", "down_proj", "down"),
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


class ValidationError(RuntimeError):
    """The real-family distributed qualification failed closed."""


class WorkerSpec(NamedTuple):
    name: str
    url: str
    preflight_path: Path
    preflight_sha256: str
    image_digest: str
    gptqmodel: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read JSON object {path}") from error
    if not isinstance(value, dict):
        raise ValidationError(f"expected one JSON object in {path}")
    return value


def _normalized_worker_url(value: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise ValidationError(f"invalid worker URL: {value!r}") from error
    if (
        parsed.scheme != "http"
        or not parsed.hostname
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValidationError(
            "worker URL must be an explicit internal http://host:port"
        )
    return value.rstrip("/")


def load_worker_specs(
    declarations: Iterable[Iterable[str]],
    *,
    lock_path: Path,
    expected_workers: int = 4,
) -> list[WorkerSpec]:
    lock_path = lock_path.expanduser().resolve(strict=True)
    lock = read_json_object(lock_path)
    revision = lock.get("revision")
    source_tree_sha256 = lock.get("source_tree_sha256")
    if (
        not isinstance(revision, str)
        or not revision
        or not isinstance(source_tree_sha256, str)
        or SHA256_RE.fullmatch(source_tree_sha256) is None
    ):
        raise ValidationError("GPTQModel lock has no immutable source identity")

    specs: list[WorkerSpec] = []
    names: set[str] = set()
    urls: set[str] = set()
    for declaration in declarations:
        values = tuple(declaration)
        if len(values) != 3:
            raise ValidationError("each worker declaration requires NAME URL PREFLIGHT")
        name, raw_url, raw_preflight = values
        if NAME_RE.fullmatch(name) is None or name in names:
            raise ValidationError("worker names must be unique stable identifiers")
        url = _normalized_worker_url(raw_url)
        if url in urls:
            raise ValidationError("worker URLs must be unique")
        preflight_path = Path(raw_preflight).expanduser().resolve(strict=True)
        preflight = read_json_object(preflight_path)
        gptqmodel = preflight.get("gptqmodel")
        gpus = preflight.get("gpus")
        image_digest = preflight.get("image_digest")
        if (
            preflight.get("status") != "qualified"
            or preflight.get("role") != "expert"
            or preflight.get("target_platform") != "linux/arm64"
            or preflight.get("cuda_arch") != "121"
            or preflight.get("python", {}).get("gil_enabled") is not False
            or not isinstance(gptqmodel, dict)
            or gptqmodel.get("revision") != revision
            or gptqmodel.get("source_tree_sha256") != source_tree_sha256
            or not isinstance(image_digest, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", image_digest) is None
            or not isinstance(gpus, list)
            or len(gpus) != 1
            or not isinstance(gpus[0], dict)
            or gpus[0].get("index") != 0
        ):
            raise ValidationError(
                f"worker {name!r} preflight does not match the Spark contract"
            )
        specs.append(
            WorkerSpec(
                name=name,
                url=url,
                preflight_path=preflight_path,
                preflight_sha256=report_identity_sha256(preflight),
                image_digest=image_digest,
                gptqmodel=gptqmodel,
            )
        )
        names.add(name)
        urls.add(url)

    if len(specs) != expected_workers:
        raise ValidationError(
            f"real-family parity requires exactly {expected_workers} Sparks, "
            f"received {len(specs)}"
        )
    if len({spec.image_digest for spec in specs}) != 1:
        raise ValidationError("all Spark workers must use one image digest")
    return sorted(specs, key=lambda spec: spec.name)


def checkpoint_projection_base(layer_id: int, expert_id: int, stem: str) -> str:
    if stem not in {item[0] for item in PROJECTIONS}:
        raise ValidationError(f"unsupported projection stem: {stem!r}")
    return f"layers.{layer_id}.ffn.experts.{expert_id}.{stem}"


def runtime_module_name(layer_id: int, expert_id: int, alias: str) -> str:
    if alias not in {item[1] for item in PROJECTIONS}:
        raise ValidationError(f"unsupported projection alias: {alias!r}")
    return f"model.layers.{layer_id}.mlp.experts.{expert_id}.{alias}"


def tensor_set_equal(
    left: dict[str, Any], right: dict[str, Any]
) -> tuple[bool, list[str]]:
    if set(left) != set(right):
        return False, sorted(set(left) ^ set(right))
    mismatches = [
        name
        for name in sorted(left)
        if not left[name].detach().cpu().equal(right[name].detach().cpu())
    ]
    return not mismatches, mismatches


def tensor_mismatch_details(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    import torch

    details: dict[str, dict[str, Any]] = {}
    for name in sorted(set(left) | set(right)):
        if name not in left or name not in right:
            details[name] = {
                "left_present": name in left,
                "right_present": name in right,
            }
            continue
        left_tensor = left[name].detach().cpu().contiguous()
        right_tensor = right[name].detach().cpu().contiguous()
        if left_tensor.equal(right_tensor):
            continue
        if (
            left_tensor.shape != right_tensor.shape
            or left_tensor.dtype != right_tensor.dtype
        ):
            details[name] = {
                "left_shape": list(left_tensor.shape),
                "right_shape": list(right_tensor.shape),
                "left_dtype": str(left_tensor.dtype),
                "right_dtype": str(right_tensor.dtype),
            }
            continue
        unequal = left_tensor != right_tensor
        delta = left_tensor.to(torch.float64) - right_tensor.to(torch.float64)
        details[name] = {
            "elements": left_tensor.numel(),
            "unequal_elements": int(unequal.sum().item()),
            "unequal_fraction": float(unequal.double().mean().item()),
            "max_abs_numeric_delta": float(delta.abs().max().item()),
        }
    return details


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as output:
        temporary = Path(output.name)
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    os.chmod(temporary, 0o644)
    os.replace(temporary, path)


def _float_value(value: Any, *, label: str) -> float:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{label} is not finite")
    return result


def _reconstruction_metrics(reference: Any, candidate: Any) -> dict[str, float]:
    import torch

    reference = reference.float()
    candidate = candidate.float()
    delta = candidate - reference
    reference_norm = torch.linalg.vector_norm(reference)
    candidate_norm = torch.linalg.vector_norm(candidate)
    denominator = float(reference_norm.item())
    if denominator <= 0.0:
        raise ValidationError("source projection has zero norm")
    cosine_denominator = reference_norm * candidate_norm
    cosine = float((reference * candidate).sum().div(cosine_denominator).item())
    return {
        "cosine": cosine,
        "relative_l2": float(torch.linalg.vector_norm(delta).item()) / denominator,
        "max_abs": float(delta.abs().max().item()),
    }


def _run_local_quantization(
    *,
    weight: Any,
    hessian: Any,
    sample_count: int,
    device: Any,
    quantize_exl3: Any,
) -> tuple[dict[str, Any], float, dict[str, Any], float]:
    import torch

    h_data = {
        "H": hessian.to(device=device),
        "count": sample_count,
        "finalized": False,
    }
    quant_args: dict[str, Any] = {
        "K": 2,
        "devices": [device],
        "apply_out_scales": None,
        "sigma_reg": SIGMA_REG,
        "seed": EXL3_SEED,
        "mcg": True,
    }
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    _weight_q, proxy_error, tensors = quantize_exl3(
        weight.to(device=device),
        h_data,
        quant_args,
        return_weight_q=False,
    )
    del _weight_q, h_data
    torch.cuda.synchronize(device)
    duration = time.perf_counter() - started
    metrics = quant_args.get("error_metrics")
    if (
        not isinstance(metrics, dict)
        or metrics.get("quantizer_path") != "hessian_ldlq"
        or metrics.get("hessian_metric_status") != "ok"
        or quant_args.get("q_fallback") is not False
        or metrics.get("hessian_sample_count") != sample_count
    ):
        raise ValidationError("local quantizer fell back or omitted Hessian metrics")
    return (
        tensors,
        _float_value(proxy_error, label="local proxy error"),
        metrics,
        duration,
    )


def run(
    *,
    snapshot: Path,
    activation_manifest: Path,
    coordinator_preflight: Path,
    worker_specs: list[WorkerSpec],
    token: bytes,
    layer_id: int,
    expert_id: int,
    down_rows: int,
    all_sparks: bool,
    timeout_seconds: float,
    max_attempts: int,
) -> dict[str, Any]:
    import torch

    from ds41rt_runtime.exl3_quantizer import (
        accumulate_activation_hessian,
        forced_expert_down_activations,
        hessian_data,
        load_activation_corpus,
        load_activation_layer_samples,
        load_native_projection,
        deterministic_expert_row_indices,
        deepseek_v4_swiglu_activation,
        read_native_model_config,
        read_source_index,
    )
    from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import (
        quantize_exl3,
        reconstruct_exl3_tensors,
    )
    from gptqmodel.utils.exl3_projection_checkpoint import (
        build_projection_request,
        tensor_identity,
    )
    from gptqmodel.utils.exl3_remote import (
        CoordinatorSlot,
        EXL3RemoteClient,
        REMOTE_REQUEST_SCHEMA,
        RemoteEndpoint,
        encode_tensor_envelope,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise ValidationError(
            "real-family parity requires coordinator GPUs 0 and 1 to be visible"
        )
    device = torch.device("cuda:0")
    snapshot = snapshot.expanduser().resolve(strict=True)
    activation_manifest = activation_manifest.expanduser().resolve(strict=True)
    coordinator_preflight = coordinator_preflight.expanduser().resolve(strict=True)
    coordinator_report = read_json_object(coordinator_preflight)
    coordinator_gpus = coordinator_report.get("gpus")
    coordinator_image = coordinator_report.get("image_digest")
    if (
        coordinator_report.get("status") != "qualified"
        or coordinator_report.get("role") != "coordinator"
        or coordinator_report.get("target_platform") != "linux/amd64"
        or coordinator_report.get("cuda_arch") != "120"
        or not isinstance(coordinator_gpus, list)
        or len(coordinator_gpus) != 2
        or [gpu.get("index") for gpu in coordinator_gpus] != [0, 1]
        or any(not isinstance(gpu.get("uuid"), str) for gpu in coordinator_gpus)
        or not isinstance(coordinator_image, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", coordinator_image) is None
    ):
        raise ValidationError("coordinator preflight does not bind both RTX GPUs")
    coordinator_preflight_sha256 = report_identity_sha256(coordinator_report)
    config, shape = read_native_model_config(snapshot)
    if not 0 <= layer_id < shape.hidden_layers:
        raise ValidationError(
            f"base layer {layer_id} is outside 0..{shape.hidden_layers - 1}"
        )
    if not 0 <= expert_id < shape.experts:
        raise ValidationError(
            f"expert {expert_id} is outside 0..{shape.experts - 1}"
        )
    if down_rows <= 0:
        raise ValidationError("down-input row count must be positive")

    endpoints = [
        RemoteEndpoint(
            name=spec.name,
            url=spec.url,
            preflight_sha256=spec.preflight_sha256,
            image_digest=spec.image_digest,
        )
        for spec in worker_specs
    ]
    client = EXL3RemoteClient(
        endpoints=endpoints,
        token=token,
        coordinator_slots=[
            CoordinatorSlot(
                device=f"cuda:{gpu['index']}",
                gpu_uuid=gpu["uuid"],
                preflight_sha256=coordinator_preflight_sha256,
                image_digest=coordinator_image,
            )
            for gpu in coordinator_gpus
        ],
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
    )
    for endpoint in client.endpoints:
        client.qualify(endpoint)
    family_key = f"base:{layer_id}:{expert_id}"
    # This utility validates hardware parity across every Spark explicitly. The
    # production work-conserving scheduler is exercised by the full-layer gate.
    assigned = client.endpoints[0]

    corpus = load_activation_corpus(
        activation_manifest,
        snapshot=snapshot,
        shape=shape,
    )
    if corpus.route_aware:
        raise ValidationError(
            "this bounded parity control expects the retained legacy route-unaware capture"
        )
    samples = load_activation_layer_samples(corpus, layer_id)
    if down_rows > int(samples.shape[0]):
        raise ValidationError(
            f"down-input rows {down_rows} exceed captured rows {samples.shape[0]}"
        )
    source_index = read_source_index(snapshot)
    shared_h_data = accumulate_activation_hessian(
        samples,
        key=f"legacy-real-layer-{layer_id}-shared",
        device=device,
    )
    shared_hessian = shared_h_data["H"].detach().cpu().contiguous()
    shared_count = int(shared_h_data["count"])
    del shared_h_data

    weights: dict[str, Any] = {}
    for stem, _alias, _hessian_kind in PROJECTIONS:
        weights[stem] = load_native_projection(
            snapshot,
            source_index,
            checkpoint_projection_base(layer_id, expert_id, stem),
            device="cuda:0",
        )
    down_samples = forced_expert_down_activations(
        samples,
        weights["w1"],
        weights["w3"],
        rows=down_rows,
        layer_id=layer_id,
        expert_id=expert_id,
        seed=EXL3_SEED,
        swiglu_limit=shape.swiglu_limit,
        device=device,
    )
    down_h_data = hessian_data(
        down_samples,
        key=f"legacy-real-layer-{layer_id}-expert-{expert_id}-down",
    )
    down_hessian = down_h_data["H"].detach().cpu().contiguous()
    down_count = int(down_h_data["count"])
    calibration_indices = deterministic_expert_row_indices(
        total_rows=int(samples.shape[0]),
        rows=down_rows,
        layer_id=layer_id,
        expert_id=expert_id,
        seed=EXL3_SEED,
        device="cpu",
    )
    evaluation_mask = torch.ones(int(samples.shape[0]), dtype=torch.bool)
    evaluation_mask[calibration_indices] = False
    evaluation_rows = min(512, int(evaluation_mask.sum().item()))
    evaluation_samples = samples[evaluation_mask][:evaluation_rows].contiguous()
    del down_h_data, down_samples, samples, calibration_indices, evaluation_mask

    config_path = snapshot / "config.json"
    index_path = snapshot / "model.safetensors.index.json"
    family_join = {
        "contract": FAMILY_JOIN_CONTRACT,
        "calibration_scope": CALIBRATION_SCOPE,
        "source_revision": snapshot.name,
        "source_config_sha256": sha256_file(config_path),
        "source_index_sha256": sha256_file(index_path),
        "activation_manifest_sha256": sha256_file(activation_manifest),
        "family_key": family_key,
    }
    execution = client.execution_contract(assigned)
    reports: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    local_family_weights: dict[str, Any] = {}
    remote_family_weights: dict[str, Any] = {}
    for stem, alias, hessian_kind in PROJECTIONS:
        weight = weights[stem].detach().cpu().contiguous().float()
        if hessian_kind == "shared":
            hessian = shared_hessian
            sample_count = shared_count
        else:
            hessian = down_hessian
            sample_count = down_count
        module_name = runtime_module_name(layer_id, expert_id, alias)
        quantizer_contract = {
            "bits": 2,
            "codebook": "mcg",
            "hessian_capture": "raw-xtx-sum-fp32-v1",
            "apply_out_scales": None,
            "sigma_reg": SIGMA_REG,
            "seed": EXL3_SEED,
            "execution": execution,
        }
        request_manifest = build_projection_request(
            module_full_name=module_name,
            layer_index=layer_id,
            input_weight=weight,
            hessian=hessian,
            sample_count=sample_count,
            quantizer_contract=quantizer_contract,
            family_join=family_join,
            route_evidence=None,
        )
        wire_manifest = {
            "schema": REMOTE_REQUEST_SCHEMA,
            "contract": "ds41rt.exl3-remote-worker-v1",
            "request": request_manifest,
        }
        request_bytes = len(
            encode_tensor_envelope(
                wire_manifest,
                {"input_weight": weight, "hessian": hessian},
            )
        )

        (
            local_tensors,
            local_proxy_value,
            local_metrics,
            local_seconds,
        ) = _run_local_quantization(
            weight=weight,
            hessian=hessian,
            sample_count=sample_count,
            device=device,
            quantize_exl3=quantize_exl3,
        )
        (
            repeat_tensors,
            repeat_proxy_value,
            repeat_metrics,
            repeat_seconds,
        ) = _run_local_quantization(
            weight=weight,
            hessian=hessian,
            sample_count=sample_count,
            device=device,
            quantize_exl3=quantize_exl3,
        )
        local_repeat_exact, local_repeat_mismatches = tensor_set_equal(
            local_tensors, repeat_tensors
        )
        local_repeat_proxy_equal = math.isclose(
            local_proxy_value,
            repeat_proxy_value,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        if not local_repeat_exact or not local_repeat_proxy_equal:
            failures.append(
                {
                    "module": module_name,
                    "gate": "local_repeatability",
                    "tensor_mismatches": local_repeat_mismatches,
                    "proxy_error_first": local_proxy_value,
                    "proxy_error_repeat": repeat_proxy_value,
                }
            )

        remote_started = time.perf_counter()
        remote_tensors, remote_result, transport = client.quantize(
            endpoint=assigned,
            request_manifest=request_manifest,
            input_weight=weight,
            hessian=hessian,
        )
        remote_seconds = time.perf_counter() - remote_started
        remote_proxy_value = _float_value(
            remote_result.get("proxy_error"), label="remote proxy error"
        )
        exact, tensor_mismatches = tensor_set_equal(local_tensors, remote_tensors)
        packed_mismatch = tensor_mismatch_details(local_tensors, remote_tensors)
        if not exact:
            observations.append(
                {
                    "module": module_name,
                    "observation": "cross_device_packed_difference",
                    "tensor_mismatches": tensor_mismatches,
                    "details": packed_mismatch,
                }
            )
        proxy_relative_delta = abs(local_proxy_value - remote_proxy_value) / max(
            abs(local_proxy_value), abs(remote_proxy_value), 1.0e-30
        )
        proxy_equivalent = (
            proxy_relative_delta <= CROSS_DEVICE_MAX_PROXY_RELATIVE_DELTA
        )
        if not proxy_equivalent:
            failures.append(
                {
                    "module": module_name,
                    "gate": "cross_device_proxy_equivalence",
                    "local": local_proxy_value,
                    "remote": remote_proxy_value,
                    "relative_delta": proxy_relative_delta,
                    "maximum": CROSS_DEVICE_MAX_PROXY_RELATIVE_DELTA,
                }
            )

        local_reconstructed = reconstruct_exl3_tensors(
            local_tensors,
            device=device,
            dtype=torch.float32,
        )
        remote_reconstructed = reconstruct_exl3_tensors(
            remote_tensors,
            device=device,
            dtype=torch.float32,
        )
        reconstructed_exact = torch.equal(local_reconstructed, remote_reconstructed)
        reconstructed_comparison = _reconstruction_metrics(
            local_reconstructed, remote_reconstructed
        )
        reconstructed_equivalent = (
            reconstructed_comparison["cosine"] >= CROSS_DEVICE_MIN_COSINE
            and reconstructed_comparison["relative_l2"]
            <= CROSS_DEVICE_MAX_RELATIVE_L2
        )
        if not reconstructed_equivalent:
            failures.append(
                {
                    "module": module_name,
                    "gate": "cross_device_reconstructed_equivalence",
                    "metrics": reconstructed_comparison,
                }
            )
        local_reconstruction = _reconstruction_metrics(
            weight.to(device=device), local_reconstructed
        )
        remote_reconstruction = _reconstruction_metrics(
            weight.to(device=device), remote_reconstructed
        )
        remote_metrics = remote_result.get("quantizer_metrics")
        if not isinstance(remote_metrics, dict):
            raise ValidationError(f"remote quantizer omitted metrics for {module_name}")
        search_contract_equal = (
            local_metrics.get("selected_global_scale")
            == remote_metrics.get("selected_global_scale")
            and local_metrics.get("scale_search_mse")
            == remote_metrics.get("scale_search_mse")
        )
        if not search_contract_equal:
            failures.append(
                {
                    "module": module_name,
                    "gate": "cross_device_scale_search",
                    "local_scale": local_metrics.get("selected_global_scale"),
                    "remote_scale": remote_metrics.get("selected_global_scale"),
                    "local_mse": local_metrics.get("scale_search_mse"),
                    "remote_mse": remote_metrics.get("scale_search_mse"),
                }
            )
        local_family_weights[stem] = local_reconstructed.detach().cpu().to(
            torch.bfloat16
        )
        remote_family_weights[stem] = remote_reconstructed.detach().cpu().to(
            torch.bfloat16
        )
        spark_controls: list[dict[str, Any]] = []
        if all_sparks:
            for control_endpoint in client.endpoints:
                if control_endpoint.name == assigned.name:
                    control_tensors = remote_tensors
                    control_result = remote_result
                    control_transport = transport
                    control_request = request_manifest
                else:
                    control_contract = dict(quantizer_contract)
                    control_contract["execution"] = client.execution_contract(
                        control_endpoint
                    )
                    control_request = build_projection_request(
                        module_full_name=module_name,
                        layer_index=layer_id,
                        input_weight=weight,
                        hessian=hessian,
                        sample_count=sample_count,
                        quantizer_contract=control_contract,
                        family_join=family_join,
                        route_evidence=None,
                    )
                    control_tensors, control_result, control_transport = client.quantize(
                        endpoint=control_endpoint,
                        request_manifest=control_request,
                        input_weight=weight,
                        hessian=hessian,
                    )
                spark_exact, spark_mismatches = tensor_set_equal(
                    remote_tensors, control_tensors
                )
                spark_mismatch = tensor_mismatch_details(
                    remote_tensors, control_tensors
                )
                if not spark_exact:
                    failures.append(
                        {
                            "module": module_name,
                            "gate": "spark_to_spark_packed_equality",
                            "reference_worker": assigned.name,
                            "worker": control_endpoint.name,
                            "tensor_mismatches": spark_mismatches,
                            "details": spark_mismatch,
                        }
                    )
                spark_controls.append(
                    {
                        "worker": control_endpoint.name,
                        "request_sha256": control_request["request_sha256"],
                        "packed_byte_equal_to_assigned": spark_exact,
                        "packed_mismatch": spark_mismatch,
                        "proxy_error": _float_value(
                            control_result.get("proxy_error"),
                            label="Spark control proxy error",
                        ),
                        "transport": control_transport,
                    }
                )
                if control_endpoint.name != assigned.name:
                    del control_tensors
        response_manifest = {
            "schema": "ds41rt.exl3-remote-result",
            "contract": "ds41rt.exl3-remote-worker-v1",
            "request_sha256": request_manifest["request_sha256"],
            "checkpoint_hit": transport["worker_checkpoint_hit"],
            "result": remote_result,
        }
        response_bytes = len(
            encode_tensor_envelope(response_manifest, remote_tensors)
        )
        reports.append(
            {
                "stem": stem,
                "module": module_name,
                "hessian_kind": hessian_kind,
                "sample_count": sample_count,
                "request_sha256": request_manifest["request_sha256"],
                "input_weight": tensor_identity(weight),
                "hessian": tensor_identity(hessian),
                "local_packed_tensors": {
                    name: tensor_identity(tensor)
                    for name, tensor in sorted(local_tensors.items())
                },
                "remote_packed_tensors": {
                    name: tensor_identity(tensor)
                    for name, tensor in sorted(remote_tensors.items())
                },
                "packed_byte_equal": exact,
                "packed_mismatch": packed_mismatch,
                "cross_device_search_contract_equal": search_contract_equal,
                "cross_device_proxy_equivalent": proxy_equivalent,
                "cross_device_proxy_relative_delta": proxy_relative_delta,
                "reconstructed_byte_equal": reconstructed_exact,
                "reconstructed_equivalent": reconstructed_equivalent,
                "reconstructed_comparison": reconstructed_comparison,
                "local_reconstruction": local_reconstruction,
                "remote_reconstruction": remote_reconstruction,
                "local": {
                    "device": "cuda:0",
                    "duration_seconds": local_seconds,
                    "proxy_error": local_proxy_value,
                    "quantizer_metrics": local_metrics,
                    "repeat": {
                        "duration_seconds": repeat_seconds,
                        "proxy_error": repeat_proxy_value,
                        "quantizer_metrics": repeat_metrics,
                        "packed_byte_equal": local_repeat_exact,
                        "tensor_mismatches": local_repeat_mismatches,
                    },
                },
                "remote": {
                    "duration_seconds": remote_result["duration_seconds"],
                    "coordinator_elapsed_seconds": remote_seconds,
                    "proxy_error": remote_proxy_value,
                    "quantizer_metrics": remote_result["quantizer_metrics"],
                    "worker": remote_result["worker"],
                    "transport": transport,
                },
                "spark_controls": spark_controls,
                "wire_bytes": {
                    "request": request_bytes,
                    "response": response_bytes,
                },
            }
        )
        del (
            local_reconstructed,
            remote_reconstructed,
            local_tensors,
            repeat_tensors,
            remote_tensors,
        )
        torch.cuda.empty_cache()

    def execute_family(projection_weights: dict[str, Any]) -> Any:
        hidden = evaluation_samples.to(device=device, dtype=torch.float32)
        gate = hidden @ projection_weights["w1"].to(device=device).float()
        up = hidden @ projection_weights["w3"].to(device=device).float()
        activated = deepseek_v4_swiglu_activation(
            gate,
            up,
            limit=shape.swiglu_limit,
        )
        return activated @ projection_weights["w2"].to(device=device).float()

    native_output = execute_family(weights)
    local_output = execute_family(local_family_weights)
    remote_output = execute_family(remote_family_weights)
    family_output_report = {
        "rows": evaluation_rows,
        "selection": "first_rows_disjoint_from_forced_down_calibration",
        "local_vs_remote": _reconstruction_metrics(local_output, remote_output),
        "native_vs_local": _reconstruction_metrics(native_output, local_output),
        "native_vs_remote": _reconstruction_metrics(native_output, remote_output),
    }
    family_cross = family_output_report["local_vs_remote"]
    family_native_local = family_output_report["native_vs_local"]
    family_native_remote = family_output_report["native_vs_remote"]
    family_output_equivalent = (
        family_cross["cosine"] >= CROSS_DEVICE_MIN_COSINE
        and family_cross["relative_l2"] <= CROSS_DEVICE_MAX_RELATIVE_L2
        and abs(family_native_local["cosine"] - family_native_remote["cosine"])
        <= CROSS_DEVICE_MAX_NATIVE_COSINE_DELTA
        and abs(
            family_native_local["relative_l2"]
            - family_native_remote["relative_l2"]
        )
        <= CROSS_DEVICE_MAX_NATIVE_RELATIVE_L2_DELTA
    )
    family_output_report["equivalent"] = family_output_equivalent
    if not family_output_equivalent:
        failures.append(
            {
                "gate": "cross_device_family_output_equivalence",
                "metrics": family_output_report,
            }
        )
    if not all_sparks:
        failures.append({"gate": "all_spark_replay_required"})

    return {
        "schema": SCHEMA,
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "observations": observations,
        "calibration_scope": CALIBRATION_SCOPE,
        "production_quality_evidence": False,
        "all_spark_replay": all_sparks,
        "source": {
            "snapshot": os.fspath(snapshot),
            "revision": snapshot.name,
            "config_sha256": sha256_file(config_path),
            "index_sha256": sha256_file(index_path),
            "geometry": {
                "hidden_size": shape.hidden_size,
                "intermediate_size": shape.intermediate_size,
                "base_layers": shape.hidden_layers,
                "dspark_layers": shape.dspark_layers,
                "experts": shape.experts,
                "swiglu_limit": shape.swiglu_limit,
            },
            "model_type": config.get("model_type"),
        },
        "activation_capture": {
            "manifest": os.fspath(activation_manifest),
            "manifest_sha256": sha256_file(activation_manifest),
            "prompt_count": corpus.prompt_count,
            "layer_rows": corpus.rows_for_layer(layer_id),
            "route_aware": corpus.route_aware,
        },
        "coordinator_preflight": {
            "path": os.fspath(coordinator_preflight),
            "sha256": coordinator_preflight_sha256,
            "image_digest": coordinator_image,
            "gpus": coordinator_gpus,
        },
        "family": {
            "key": family_key,
            "layer": layer_id,
            "expert": expert_id,
            "assigned_worker": assigned.name,
            "scheduler": "explicit-spark-parity-reference-v1",
            "slots": [
                *[slot.device for slot in client.coordinator_slots],
                *[endpoint.name for endpoint in client.endpoints],
            ],
        },
        "quantizer": {
            "bits": 2,
            "codebook": "mcg",
            "seed": EXL3_SEED,
            "sigma_reg": SIGMA_REG,
            "apply_out_scales": None,
            "gptqmodel": worker_specs[0].gptqmodel,
        },
        "cross_device_contract": {
            "packed_equality": "recorded_not_required_across_sm120_sm121",
            "same_architecture_packed_equality": "required",
            "minimum_reconstructed_and_family_output_cosine": CROSS_DEVICE_MIN_COSINE,
            "maximum_reconstructed_and_family_output_relative_l2": CROSS_DEVICE_MAX_RELATIVE_L2,
            "maximum_proxy_relative_delta": CROSS_DEVICE_MAX_PROXY_RELATIVE_DELTA,
            "maximum_native_output_cosine_delta": CROSS_DEVICE_MAX_NATIVE_COSINE_DELTA,
            "maximum_native_output_relative_l2_delta": CROSS_DEVICE_MAX_NATIVE_RELATIVE_L2_DELTA,
        },
        "workers": [
            {
                "name": spec.name,
                "url": spec.url,
                "preflight": os.fspath(spec.preflight_path),
                "preflight_sha256": spec.preflight_sha256,
                "image_digest": spec.image_digest,
            }
            for spec in worker_specs
        ],
        "projections": reports,
        "family_output": family_output_report,
        "wire_bytes": {
            "request_total": sum(item["wire_bytes"]["request"] for item in reports),
            "response_total": sum(item["wire_bytes"]["response"] for item in reports),
        },
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--activation-manifest", type=Path, required=True)
    parser.add_argument("--coordinator-preflight", type=Path, required=True)
    parser.add_argument(
        "--worker",
        action="append",
        nargs=3,
        metavar=("NAME", "URL", "PREFLIGHT"),
        required=True,
    )
    parser.add_argument(
        "--gptqmodel-lock",
        type=Path,
        default=root / "third_party" / "gptqmodel.lock.json",
    )
    parser.add_argument("--token-env", default="DS41RT_EXL3_WORKER_TOKEN")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--down-rows", type=int, default=512)
    parser.add_argument("--all-sparks", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not args.token_env
        or args.down_rows <= 0
        or args.timeout_seconds <= 0
        or not 1 <= args.max_attempts <= 10
    ):
        parser.error("token and resource limits must be positive")
    return args


def main() -> int:
    args = parse_args()
    token_value = os.environ.get(args.token_env)
    if not token_value:
        raise ValidationError(f"worker token env {args.token_env!r} is unset")
    workers = load_worker_specs(args.worker, lock_path=args.gptqmodel_lock)
    report = run(
        snapshot=args.snapshot,
        activation_manifest=args.activation_manifest,
        coordinator_preflight=args.coordinator_preflight,
        worker_specs=workers,
        token=token_value.encode(),
        layer_id=args.layer,
        expert_id=args.expert,
        down_rows=args.down_rows,
        all_sparks=args.all_sparks,
        timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
    )
    report["script_sha256"] = sha256_file(Path(__file__).resolve())
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as error:
        raise SystemExit(f"real-family-parity: {error}") from error
