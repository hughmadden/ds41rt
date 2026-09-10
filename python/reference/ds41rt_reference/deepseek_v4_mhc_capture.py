from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DS4_HC_MULT = 4
DS4_HC_MIXES = 24
DS4_HC_PARTIALS = 25
DS4_HC_BLOCK_K = 256
DS4_MAX_MHC_ROWS = 8_192
DS4_EXPERT_TP = 4
_MHC_ENTRY_PREPARED: set[tuple[int, ...]] = set()
_MHC_POST_PRE_PREPARED: set[tuple[int, ...]] = set()
_MHC_TERMINAL_PREPARED: set[tuple[int, ...]] = set()


@dataclass(frozen=True)
class DeepseekV4MHCGeometry:
    variant: str
    hidden: int
    split_k: int
    multiplier: int = DS4_HC_MULT
    mixes: int = DS4_HC_MIXES
    partials: int = DS4_HC_PARTIALS

    @property
    def function_width(self) -> int:
        return self.multiplier * self.hidden

    @property
    def checkpoint_bytes_per_layer(self) -> int:
        function = self.mixes * self.function_width * 4
        base_and_scale = (self.mixes + 3) * 4
        norm = self.hidden * 2
        return 2 * (function + base_and_scale + norm)


@dataclass(frozen=True)
class DeepseekV4MHCScratchLayout:
    partials_offset: int
    total_bytes: int


@dataclass(frozen=True)
class DeepseekV4MHCContract:
    geometry: DeepseekV4MHCGeometry
    max_rows: int
    scratch: DeepseekV4MHCScratchLayout
    stages: tuple[str, ...] = (
        "model-entry-broadcast-attention-pre-and-rmsnorm",
        "attention-delta-on-coordinator",
        "fused-attention-post-ffn-pre-and-rmsnorm",
        "coordinator-shared-plus-tp4-routed-expert-delta",
        "fused-ffn-post-next-attention-pre-and-rmsnorm",
        "terminal-ffn-post",
        "hc-head-collapse-and-final-rmsnorm",
    )
    owner: str = "coordinator-local-residual-stream"
    status: str = "qualified-full-mhc-lifecycle-not-active"

    @property
    def serving_allocates(self) -> bool:
        return False

    @property
    def residual_ping_pong_buffers(self) -> int:
        return 2

    @property
    def normalized_buffers(self) -> int:
        return 1

    @property
    def mix_state_ping_pong_buffers(self) -> int:
        return 2

    @property
    def expert_tensor_parallel(self) -> int:
        return DS4_EXPERT_TP

    @property
    def expert_parallel(self) -> bool:
        return False

    @property
    def changes_expert_tp(self) -> bool:
        return False

    @property
    def steady_state_fuses_post_pre(self) -> bool:
        return True

    @property
    def entry_pre_broadcasts_lanes(self) -> bool:
        return True

    @property
    def head_fuses_final_rmsnorm(self) -> bool:
        return True

    @property
    def head_uses_bound_normalized_output(self) -> bool:
        return True

    @property
    def head_scratch_bytes(self) -> int:
        return 0

    @property
    def expert_exchange_width(self) -> int:
        return self.geometry.hidden

    def boundary_call_counts(self, *, layers: int) -> dict[str, int]:
        layers = int(layers)
        if layers <= 0:
            raise ValueError("layers must be positive")
        return {
            "entry_pre": 1,
            "fused_post_pre": 2 * layers - 1,
            "terminal_post": 1,
            "terminal_head": 1,
        }

    def required_buffer_bytes(self) -> dict[str, int]:
        rows = self.max_rows
        hidden = self.geometry.hidden
        return {
            "shared_partials_scratch": self.scratch.total_bytes,
            "residual_lane_ping_pong": 2 * rows * DS4_HC_MULT * hidden * 2,
            "normalized_collapsed": rows * hidden * 2,
            "post_comb_ping_pong": 2
            * rows
            * (DS4_HC_MULT + DS4_HC_MULT * DS4_HC_MULT)
            * 4,
        }


def deepseek_v4_mhc_geometry(variant: str) -> DeepseekV4MHCGeometry:
    variant = str(variant).strip().lower()
    if variant == "flash":
        hidden = 4_096
    elif variant == "pro":
        hidden = 7_168
    else:
        raise ValueError(
            f"DeepSeek V4 mHC variant must be flash or pro, got {variant!r}"
        )
    return DeepseekV4MHCGeometry(
        variant=variant,
        hidden=hidden,
        split_k=DS4_HC_MULT * hidden // DS4_HC_BLOCK_K,
    )


def plan_deepseek_v4_mhc(*, variant: str, max_rows: int) -> DeepseekV4MHCContract:
    geometry = deepseek_v4_mhc_geometry(variant)
    max_rows = int(max_rows)
    if not 1 <= max_rows <= DS4_MAX_MHC_ROWS:
        raise ValueError(
            f"DeepSeek V4 mHC max_rows must be in [1, {DS4_MAX_MHC_ROWS}], "
            f"got {max_rows}"
        )
    scratch = DeepseekV4MHCScratchLayout(
        partials_offset=0,
        total_bytes=max_rows * geometry.split_k * geometry.partials * 4,
    )
    return DeepseekV4MHCContract(
        geometry=geometry,
        max_rows=max_rows,
        scratch=scratch,
    )


def qualify_deepseek_v4_mhc_contract(**kwargs: Any) -> bool:
    """Cross-check the ping-pong mHC lifecycle against pinned SparkInfer."""

    contract = plan_deepseek_v4_mhc(
        variant=str(kwargs["variant"]), max_rows=int(kwargs["max_rows"])
    )
    import torch
    from b12x.norm import mhc

    geometry = contract.geometry
    if (
        mhc.MULT != geometry.multiplier
        or mhc.MIXES != geometry.mixes
        or mhc.PARTIALS != geometry.partials
        or mhc.DEFAULT_BLOCK_K != DS4_HC_BLOCK_K
    ):
        raise RuntimeError("pinned SparkInfer mHC constants drifted")

    plan = mhc.plan(
        mhc.Caps(
            device="cpu",
            max_tokens=contract.max_rows,
            hidden_size=geometry.hidden,
            split_k=geometry.split_k,
        )
    )
    (scratch_spec,) = plan.scratch_specs()
    if int(scratch_spec.nbytes) != contract.scratch.total_bytes:
        raise RuntimeError(
            "pinned SparkInfer mHC scratch ABI drifted: "
            f"DS41RT={contract.scratch.total_bytes} "
            f"SparkInfer={scratch_spec.nbytes}"
        )

    required_surface = {
        "Caps",
        "Plan",
        "Binding",
        "plan",
        "bind",
        "run_pre",
        "run_post_pre",
        "run_post",
        "run_head",
    }
    missing_surface = required_surface.difference(mhc.META.entry_points)
    if missing_surface:
        raise RuntimeError(
            "pinned SparkInfer mHC surface is missing: "
            + ", ".join(sorted(missing_surface))
        )
    if (
        mhc.Binding.serving_allocates is not False
        or mhc.Binding.outputs_are_bound is not True
        or mhc.Binding.pre_broadcasts_residual_lanes is not True
        or mhc.Binding.post_pre_fuses_layer_boundary is not True
        or mhc.Binding.head_uses_bound_y is not True
    ):
        raise RuntimeError("pinned SparkInfer mHC serving lifecycle drifted")

    scratch = torch.empty(scratch_spec.shape, dtype=scratch_spec.dtype)
    live_rows = 1
    y = torch.empty((live_rows, geometry.hidden), dtype=torch.bfloat16)

    def bind(residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        return mhc.bind(
            plan,
            scratch=scratch,
            tokens=live_rows,
            expected_m=contract.max_rows,
            y=y,
            post=post,
            comb=comb,
            out=residual,
        )

    residual_a = torch.empty(
        (live_rows, DS4_HC_MULT, geometry.hidden), dtype=torch.bfloat16
    )
    residual_b = torch.empty_like(residual_a)
    post_a = torch.empty((live_rows, DS4_HC_MULT), dtype=torch.float32)
    post_b = torch.empty_like(post_a)
    comb_a = torch.empty((live_rows, DS4_HC_MULT, DS4_HC_MULT), dtype=torch.float32)
    comb_b = torch.empty_like(comb_a)
    binding_a = bind(residual_a, post_a, comb_a)
    binding_b = bind(residual_b, post_b, comb_b)
    if (
        binding_a.partials.untyped_storage().data_ptr()
        != binding_b.partials.untyped_storage().data_ptr()
        or binding_a.y.untyped_storage().data_ptr()
        != binding_b.y.untyped_storage().data_ptr()
        or binding_a.out.untyped_storage().data_ptr()
        == binding_b.out.untyped_storage().data_ptr()
    ):
        raise RuntimeError("pinned SparkInfer mHC ping-pong binding ABI drifted")
    return True


def deepseek_v4_mhc_scratch_nbytes(*, variant: str, max_rows: int) -> int:
    return plan_deepseek_v4_mhc(
        variant=variant, max_rows=max_rows
    ).scratch.total_bytes


def prepare_deepseek_v4_mhc_entry(ctx: dict[str, Any], **kwargs: Any) -> None:
    _run_deepseek_v4_mhc_entry(ctx, prepare_only=True, **kwargs)


def capture_deepseek_v4_mhc_entry(ctx: dict[str, Any], **kwargs: Any) -> None:
    _run_deepseek_v4_mhc_entry(ctx, prepare_only=False, **kwargs)


def _run_deepseek_v4_mhc_entry(
    ctx: dict[str, Any], *, prepare_only: bool, **kwargs: Any
) -> None:
    """Broadcast embeddings into four HC lanes and run layer-0 attention pre."""

    import torch
    from ds41rt_reference.b12x_spark_capture import _raw_tensor
    from b12x.norm import mhc

    variant = str(kwargs["variant"]).strip().lower()
    rows = int(kwargs["rows"])
    max_rows = int(kwargs["max_rows"])
    contract = plan_deepseek_v4_mhc(variant=variant, max_rows=max_rows)
    if not 1 <= rows <= max_rows:
        raise ValueError(
            f"DeepSeek V4 mHC entry rows must be in [1, {max_rows}], got {rows}"
        )
    geometry = contract.geometry
    buffers = ctx["buffers"]
    required = _mhc_boundary_required_buffers(
        rows=rows,
        hidden=geometry.hidden,
        scratch_bytes=contract.scratch.total_bytes,
        input_name="residual_input",
        hc_function_width=geometry.hidden,
    )
    device_id = _validate_mhc_capture_buffers(
        buffers, required, anchor="residual_input"
    )
    stream = torch.cuda.ExternalStream(int(ctx["cuda_stream"]), device=device_id)
    prepared_key = (
        device_id,
        rows,
        max_rows,
        *(int(buffers[name]["ptr"]) for name in required),
    )
    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        binding = _bind_mhc_capture(
            mhc=mhc,
            buffers=buffers,
            geometry=geometry,
            rows=rows,
            max_rows=max_rows,
            scratch_bytes=contract.scratch.total_bytes,
        )
        if not prepare_only and prepared_key not in _MHC_ENTRY_PREPARED:
            raise RuntimeError(
                "DeepSeek V4 mHC entry was not prepared for its fixed graph"
            )
        mhc.run_pre(
            _raw_tensor(
                buffers["residual_input"],
                (rows, geometry.hidden),
                torch.bfloat16,
                2,
                name="target model-entry embedding",
            ),
            _mhc_hc_fn(buffers, geometry.hidden),
            _mhc_hc_scale(buffers),
            _mhc_hc_base(buffers),
            rms_eps=float(kwargs.get("rms_eps", 1.0e-6)),
            hc_eps=float(kwargs.get("hc_eps", 1.0e-6)),
            sinkhorn_iters=int(kwargs.get("sinkhorn_iters", 20)),
            norm_weight=_mhc_norm_weight(buffers, geometry.hidden),
            norm_eps=float(kwargs.get("norm_eps", 1.0e-6)),
            binding=binding,
        )
        if prepare_only:
            _MHC_ENTRY_PREPARED.add(prepared_key)


def prepare_deepseek_v4_mhc_post_pre(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    _run_deepseek_v4_mhc_post_pre(ctx, prepare_only=True, **kwargs)


def capture_deepseek_v4_mhc_post_pre(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    _run_deepseek_v4_mhc_post_pre(ctx, prepare_only=False, **kwargs)


def _run_deepseek_v4_mhc_post_pre(
    ctx: dict[str, Any], *, prepare_only: bool, **kwargs: Any
) -> None:
    """Fuse an FFN delta into the next layer's attention pre-normalization."""

    import torch
    from ds41rt_reference.b12x_spark_capture import _raw_tensor
    from b12x.norm import mhc

    variant = str(kwargs["variant"]).strip().lower()
    rows = int(kwargs["rows"])
    max_rows = int(kwargs["max_rows"])
    contract = plan_deepseek_v4_mhc(variant=variant, max_rows=max_rows)
    if not 1 <= rows <= max_rows:
        raise ValueError(
            f"DeepSeek V4 mHC post/pre rows must be in [1, {max_rows}], got {rows}"
        )
    geometry = contract.geometry
    buffers = ctx["buffers"]
    required = _mhc_boundary_required_buffers(
        rows=rows,
        hidden=geometry.hidden,
        scratch_bytes=contract.scratch.total_bytes,
        input_name="delta",
        hc_function_width=DS4_HC_MULT * geometry.hidden,
    )
    required.update(
        {
            "residual": rows * DS4_HC_MULT * geometry.hidden * 2,
            "prev_post": rows * DS4_HC_MULT * 4,
            "prev_comb": rows * DS4_HC_MULT * DS4_HC_MULT * 4,
            "aux_hidden_output": rows * geometry.hidden * 2,
        }
    )
    device_id = _validate_mhc_capture_buffers(buffers, required, anchor="delta")
    stream = torch.cuda.ExternalStream(int(ctx["cuda_stream"]), device=device_id)
    prepared_key = (
        device_id,
        int(ctx["cuda_stream"]),
        *map(ord, variant),
        rows,
        max_rows,
        int(bool(kwargs.get("capture_aux_hidden", False))),
    )
    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        binding = _bind_mhc_capture(
            mhc=mhc,
            buffers=buffers,
            geometry=geometry,
            rows=rows,
            max_rows=max_rows,
            scratch_bytes=contract.scratch.total_bytes,
        )
        if not prepare_only and prepared_key not in _MHC_POST_PRE_PREPARED:
            raise RuntimeError(
                "DeepSeek V4 mHC post/pre was not prepared for its fixed graph"
            )
        residual_out, _, _, _ = mhc.run_post_pre(
            _raw_tensor(
                buffers["delta"],
                (rows, geometry.hidden),
                torch.bfloat16,
                2,
                name="target FFN delta",
            ),
            _raw_tensor(
                buffers["residual"],
                (rows, DS4_HC_MULT, geometry.hidden),
                torch.bfloat16,
                2,
                name="target FFN HC residual",
            ),
            _raw_tensor(
                buffers["prev_post"],
                (rows, DS4_HC_MULT),
                torch.float32,
                4,
                name="target FFN previous post mix",
            ),
            _raw_tensor(
                buffers["prev_comb"],
                (rows, DS4_HC_MULT, DS4_HC_MULT),
                torch.float32,
                4,
                name="target FFN previous combination mix",
            ),
            _mhc_hc_fn(buffers, DS4_HC_MULT * geometry.hidden),
            _mhc_hc_scale(buffers),
            _mhc_hc_base(buffers),
            rms_eps=float(kwargs.get("rms_eps", 1.0e-6)),
            hc_eps=float(kwargs.get("hc_eps", 1.0e-6)),
            sinkhorn_iters=int(kwargs.get("sinkhorn_iters", 20)),
            norm_weight=_mhc_norm_weight(buffers, geometry.hidden),
            norm_eps=float(kwargs.get("norm_eps", 1.0e-6)),
            binding=binding,
        )
        # DeepSeek dSpark is trained on the raw post-layer mHC state collapsed
        # across its four residual lanes.  ``binding.y`` is instead the
        # RMS-normalized input to the next attention layer and is not a valid
        # substitute for this auxiliary hidden state.
        if bool(kwargs.get("capture_aux_hidden", False)):
            torch.mean(
                residual_out,
                dim=1,
                out=_raw_tensor(
                    buffers["aux_hidden_output"],
                    (rows, geometry.hidden),
                    torch.bfloat16,
                    2,
                    name="target dSpark raw post-layer auxiliary hidden",
                ),
            )
        if prepare_only:
            _MHC_POST_PRE_PREPARED.add(prepared_key)


def prepare_deepseek_v4_mhc_terminal(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    _run_deepseek_v4_mhc_terminal(ctx, prepare_only=True, **kwargs)


def capture_deepseek_v4_mhc_terminal(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    _run_deepseek_v4_mhc_terminal(ctx, prepare_only=False, **kwargs)


def _run_deepseek_v4_mhc_terminal(
    ctx: dict[str, Any], *, prepare_only: bool, **kwargs: Any
) -> None:
    """Finish the final FFN and retain raw plus RMS-normalized head hidden."""

    import torch
    from ds41rt_reference.b12x_spark_capture import _raw_tensor
    from b12x.norm import mhc

    variant = str(kwargs["variant"]).strip().lower()
    rows = int(kwargs["rows"])
    max_rows = int(kwargs["max_rows"])
    contract = plan_deepseek_v4_mhc(variant=variant, max_rows=max_rows)
    if not 1 <= rows <= max_rows:
        raise ValueError(
            f"DeepSeek V4 mHC terminal rows must be in [1, {max_rows}], got {rows}"
        )
    geometry = contract.geometry
    hidden = geometry.hidden
    buffers = ctx["buffers"]
    required = {
        "scratch": contract.scratch.total_bytes,
        "delta": rows * hidden * 2,
        "residual": rows * DS4_HC_MULT * hidden * 2,
        "prev_post": rows * DS4_HC_MULT * 4,
        "prev_comb": rows * DS4_HC_MULT * DS4_HC_MULT * 4,
        "terminal_residual": rows * DS4_HC_MULT * hidden * 2,
        "collapsed_output": rows * hidden * 2,
        "normalized_output": rows * hidden * 2,
        "aux_hidden_output": rows * hidden * 2,
        "hc_head_fn": DS4_HC_MULT * DS4_HC_MULT * hidden * 4,
        "hc_head_scale": 4,
        "hc_head_base": DS4_HC_MULT * 4,
        "norm_weight": hidden * 2,
    }
    device_id = _validate_mhc_capture_buffers(buffers, required, anchor="delta")
    if int(buffers["collapsed_output"]["ptr"]) == int(
        buffers["normalized_output"]["ptr"]
    ):
        raise ValueError(
            "DeepSeek V4 mHC terminal raw and normalized hidden must not alias"
        )
    stream = torch.cuda.ExternalStream(int(ctx["cuda_stream"]), device=device_id)
    prepared_key = (
        device_id,
        rows,
        max_rows,
        *(int(buffers[name]["ptr"]) for name in required),
    )
    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        binding = mhc.bind(
            mhc.plan(
                mhc.Caps(
                    device=torch.device("cuda", device_id),
                    max_tokens=max_rows,
                    hidden_size=hidden,
                    split_k=geometry.split_k,
                    dtype=torch.bfloat16,
                )
            ),
            scratch=_raw_tensor(
                buffers["scratch"],
                (contract.scratch.total_bytes,),
                torch.uint8,
                1,
                name="target terminal mHC scratch",
            ),
            tokens=rows,
            # This binding belongs to an exact-row CUDA graph.  max_rows is
            # scratch capacity, not a dynamic-shape policy hint.
            expected_m=rows,
            y=_raw_tensor(
                buffers["normalized_output"],
                (rows, hidden),
                torch.bfloat16,
                2,
                name="target terminal normalized hidden",
            ),
        )
        if not prepare_only and prepared_key not in _MHC_TERMINAL_PREPARED:
            raise RuntimeError(
                "DeepSeek V4 mHC terminal was not prepared for its fixed graph"
            )
        terminal_residual = _raw_tensor(
            buffers["terminal_residual"],
            (rows, DS4_HC_MULT, hidden),
            torch.bfloat16,
            2,
            name="target completed terminal HC residual",
        )
        mhc.run_post(
            _raw_tensor(
                buffers["delta"],
                (rows, hidden),
                torch.bfloat16,
                2,
                name="target terminal FFN delta",
            ),
            _raw_tensor(
                buffers["residual"],
                (rows, DS4_HC_MULT, hidden),
                torch.bfloat16,
                2,
                name="target terminal post-attention residual",
            ),
            _raw_tensor(
                buffers["prev_post"],
                (rows, DS4_HC_MULT),
                torch.float32,
                4,
                name="target terminal FFN post mix",
            ),
            _raw_tensor(
                buffers["prev_comb"],
                (rows, DS4_HC_MULT, DS4_HC_MULT),
                torch.float32,
                4,
                name="target terminal FFN combination mix",
            ),
            out=terminal_residual,
        )
        mhc.run_head(
            terminal_residual,
            _raw_tensor(
                buffers["hc_head_fn"],
                (DS4_HC_MULT, DS4_HC_MULT * hidden),
                torch.float32,
                4,
                name="target terminal HC head function",
            ),
            _raw_tensor(
                buffers["hc_head_scale"],
                (1,),
                torch.float32,
                4,
                name="target terminal HC head scale",
            ),
            _raw_tensor(
                buffers["hc_head_base"],
                (DS4_HC_MULT,),
                torch.float32,
                4,
                name="target terminal HC head base",
            ),
            _raw_tensor(
                buffers["norm_weight"],
                (hidden,),
                torch.bfloat16,
                2,
                name="target terminal RMSNorm weight",
            ),
            rms_eps=float(kwargs.get("rms_eps", 1.0e-6)),
            hc_eps=float(kwargs.get("hc_eps", 1.0e-6)),
            norm_eps=float(kwargs.get("norm_eps", 1.0e-6)),
            collapsed_out=_raw_tensor(
                buffers["collapsed_output"],
                (rows, hidden),
                torch.bfloat16,
                2,
                name="target terminal unnormalized collapsed hidden",
            ),
            binding=binding,
        )
        torch.mean(
            terminal_residual,
            dim=1,
            out=_raw_tensor(
                buffers["aux_hidden_output"],
                (rows, hidden),
                torch.bfloat16,
                2,
                name="target terminal dSpark raw post-layer auxiliary hidden",
            ),
        )
        if prepare_only:
            _MHC_TERMINAL_PREPARED.add(prepared_key)


def _mhc_boundary_required_buffers(
    *,
    rows: int,
    hidden: int,
    scratch_bytes: int,
    input_name: str,
    hc_function_width: int,
) -> dict[str, int]:
    # Model entry uses the checkpoint function after summing its four input-lane
    # blocks; steady-state boundaries retain the complete four-lane function.
    return {
        "scratch": scratch_bytes,
        input_name: rows * hidden * 2,
        "normalized_output": rows * hidden * 2,
        "residual_out": rows * DS4_HC_MULT * hidden * 2,
        "post_out": rows * DS4_HC_MULT * 4,
        "comb_out": rows * DS4_HC_MULT * DS4_HC_MULT * 4,
        "hc_fn": DS4_HC_MIXES * hc_function_width * 4,
        "hc_scale": 3 * 4,
        "hc_base": DS4_HC_MIXES * 4,
        "norm_weight": hidden * 2,
    }


def _bind_mhc_capture(
    *,
    mhc: Any,
    buffers: dict[str, dict[str, Any]],
    geometry: DeepseekV4MHCGeometry,
    rows: int,
    max_rows: int,
    scratch_bytes: int,
) -> Any:
    import torch
    from ds41rt_reference.b12x_spark_capture import _raw_tensor

    plan = mhc.plan(
        mhc.Caps(
            device=torch.device("cuda", int(buffers["scratch"]["device_id"])),
            max_tokens=max_rows,
            hidden_size=geometry.hidden,
            split_k=geometry.split_k,
            dtype=torch.bfloat16,
        )
    )
    return mhc.bind(
        plan,
        scratch=_raw_tensor(
            buffers["scratch"],
            (scratch_bytes,),
            torch.uint8,
            1,
            name="target mHC boundary scratch",
        ),
        tokens=rows,
        # DS41RT captures a distinct graph for every exact row shape.  Keep
        # max_rows as the allocation bound and select the kernel from live M.
        expected_m=rows,
        y=_raw_tensor(
            buffers["normalized_output"],
            (rows, geometry.hidden),
            torch.bfloat16,
            2,
            name="target mHC normalized output",
        ),
        post=_raw_tensor(
            buffers["post_out"],
            (rows, DS4_HC_MULT),
            torch.float32,
            4,
            name="target mHC post output",
        ),
        comb=_raw_tensor(
            buffers["comb_out"],
            (rows, DS4_HC_MULT, DS4_HC_MULT),
            torch.float32,
            4,
            name="target mHC combination output",
        ),
        out=_raw_tensor(
            buffers["residual_out"],
            (rows, DS4_HC_MULT, geometry.hidden),
            torch.bfloat16,
            2,
            name="target mHC residual output",
        ),
    )


def _mhc_hc_fn(
    buffers: dict[str, dict[str, Any]], function_width: int
) -> Any:
    import torch
    from ds41rt_reference.b12x_spark_capture import _raw_tensor

    return _raw_tensor(
        buffers["hc_fn"],
        (DS4_HC_MIXES, function_width),
        torch.float32,
        4,
        name="target mHC function",
    )


def _mhc_hc_scale(buffers: dict[str, dict[str, Any]]) -> Any:
    import torch
    from ds41rt_reference.b12x_spark_capture import _raw_tensor

    return _raw_tensor(
        buffers["hc_scale"],
        (3,),
        torch.float32,
        4,
        name="target mHC scale",
    )


def _mhc_hc_base(buffers: dict[str, dict[str, Any]]) -> Any:
    import torch
    from ds41rt_reference.b12x_spark_capture import _raw_tensor

    return _raw_tensor(
        buffers["hc_base"],
        (DS4_HC_MIXES,),
        torch.float32,
        4,
        name="target mHC base",
    )


def _mhc_norm_weight(
    buffers: dict[str, dict[str, Any]], hidden: int
) -> Any:
    import torch
    from ds41rt_reference.b12x_spark_capture import _raw_tensor

    return _raw_tensor(
        buffers["norm_weight"],
        (hidden,),
        torch.bfloat16,
        2,
        name="target mHC RMSNorm weight",
    )


def _validate_mhc_capture_buffers(
    buffers: dict[str, dict[str, Any]],
    required: dict[str, int],
    *,
    anchor: str,
) -> int:
    missing = sorted(set(required) - set(buffers))
    if missing:
        raise ValueError(
            "DeepSeek V4 mHC capture is missing buffers: " + ", ".join(missing)
        )
    device_id = int(buffers[anchor]["device_id"])
    for name, required_bytes in required.items():
        buffer = buffers[name]
        if int(buffer["device_id"]) != device_id:
            raise ValueError(
                "DeepSeek V4 mHC capture buffers must share one device; "
                f"{anchor} is cuda:{device_id}, {name} is cuda:{buffer['device_id']}"
            )
        if int(buffer["bytes"]) < required_bytes:
            raise ValueError(
                f"DeepSeek V4 mHC capture {name} needs {required_bytes} bytes, "
                f"got {buffer['bytes']}"
            )
    return device_id


__all__ = [
    "DeepseekV4MHCContract",
    "DeepseekV4MHCGeometry",
    "DeepseekV4MHCScratchLayout",
    "capture_deepseek_v4_mhc_entry",
    "capture_deepseek_v4_mhc_post_pre",
    "capture_deepseek_v4_mhc_terminal",
    "deepseek_v4_mhc_geometry",
    "deepseek_v4_mhc_scratch_nbytes",
    "plan_deepseek_v4_mhc",
    "prepare_deepseek_v4_mhc_entry",
    "prepare_deepseek_v4_mhc_post_pre",
    "prepare_deepseek_v4_mhc_terminal",
    "qualify_deepseek_v4_mhc_contract",
]
