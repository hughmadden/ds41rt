from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from ds4rt_reference.deepseek_v4_attention_layer_capture import (
    bind_deepseek_v4_sliding_attention_layer,
    run_deepseek_v4_sliding_attention_layer,
)
from ds4rt_reference.deepseek_v4_mhc_capture import DS4_EXPERT_TP, DS4_HC_MULT
from ds4rt_reference.deepseek_v4_kv_format import (
    DS4_SOURCE_PAGE_TOKENS,
    FP8_UE8M0,
    deepseek_v4_kv_format,
)
from ds4rt_reference.deepseek_v4_sparse_block_capture import (
    DeepseekV4SparseBlockContract,
    bind_deepseek_v4_sparse_block,
    bind_deepseek_v4_sparse_block_arena,
    plan_deepseek_v4_sparse_block,
    qualify_deepseek_v4_sparse_block_contract,
    run_deepseek_v4_sparse_block_attention,
    run_deepseek_v4_sparse_block_post_dispatch,
)

DS4_DSPARK_BLOCKS = 3
DS4_DSPARK_PROPOSAL_TOKENS = 5
DS4_DSPARK_SLIDING_WINDOW = 128
DS4_DSPARK_HEAD_DIM = 512
DS4_DSPARK_NOISE_TOKEN_ID = 128_799
DS4_DSPARK_VOCAB_SIZE = 129_280
DS4_DSPARK_CACHE_PAGE_TOKENS = DS4_SOURCE_PAGE_TOKENS
DS4_DSPARK_CACHE_PAGE_BYTES = FP8_UE8M0.main_page_bytes
DS4_DSPARK_SELECTION_WIDTH = (
    DS4_DSPARK_SLIDING_WINDOW + DS4_DSPARK_PROPOSAL_TOKENS
)
DS4_DSPARK_ARENA_ALIGNMENT = 256
DS4_DSPARK_PRODUCER_ALIGNMENT = 1_024
DS4_DSPARK_PROPOSAL_ALIGNMENT = 16
DS4_DSPARK_ENTRY_ACTIVATION_BLOCK_SIZE = 128

_ENTRY_PROJECTION_STATE: dict[tuple[int, int, int, int, int], tuple[Any, Any]] = {}
_ENTRY_PROJECTION_PREPARED: set[tuple[int, int, int, int, int, int]] = set()
_PROMPT_PRIME_STATE: dict[tuple[int, int, int, int, int], tuple[Any, Any]] = {}
_PROMPT_PRIME_PREPARED: set[tuple[int, int, int, int, int, int]] = set()
_PROPOSAL_ENTRY_PREPARED: set[tuple[int, ...]] = set()
_BLOCK_ATTENTION_STATE: dict[tuple[int, ...], tuple[Any, Any]] = {}
_BLOCK_ATTENTION_PREPARED: set[tuple[int, ...]] = set()
_BLOCK_POST_DISPATCH_PREPARED: set[tuple[int, ...]] = set()
_TERMINAL_COLLAPSE_PREPARED: set[tuple[int, ...]] = set()


@dataclass(frozen=True)
class DeepseekV4DsparkGeometry:
    variant: str
    hidden: int
    target_layers: int
    target_taps: tuple[int, int, int]
    routed_experts: int
    expert_intermediate: int
    attention_heads: int
    markov_rank: int
    blocks: int = DS4_DSPARK_BLOCKS
    proposal_tokens: int = DS4_DSPARK_PROPOSAL_TOKENS
    sliding_window: int = DS4_DSPARK_SLIDING_WINDOW
    head_dim: int = DS4_DSPARK_HEAD_DIM
    noise_token_id: int = DS4_DSPARK_NOISE_TOKEN_ID
    vocab_size: int = DS4_DSPARK_VOCAB_SIZE

    @property
    def target_projection_input_width(self) -> int:
        return len(self.target_taps) * self.hidden

    @property
    def proposal_hc_width(self) -> int:
        return DS4_HC_MULT * self.hidden

    @property
    def physical_block_ids(self) -> tuple[int, int, int]:
        return tuple(
            self.target_layers + block_index for block_index in range(self.blocks)
        )

    @property
    def storage_prefixes(self) -> tuple[str, str, str]:
        # This is checkpoint serialization only. It does not select an MTP
        # algorithm or import any recurrent GLM MTP state.
        return tuple(f"mtp.{block_index}" for block_index in range(self.blocks))


@dataclass(frozen=True)
class DeepseekV4DsparkBlockContract:
    block_index: int
    physical_layer_id: int
    target_tap_id: int
    storage_prefix: str
    expert_handoff: DeepseekV4SparseBlockContract
    attention_semantics: str = "dual-source-dspark-sliding-attention"
    expert_topology: str = "strict-tensor-parallel-4"

    @property
    def uses_target_main_kv(self) -> bool:
        return True

    @property
    def proposal_queries_include_proposal_kv(self) -> bool:
        return True

    @property
    def checkpoint_namespace_is_algorithm_name(self) -> bool:
        return False


@dataclass(frozen=True)
class DeepseekV4DsparkArenaRegion:
    name: str
    offset: int
    nbytes: int


@dataclass(frozen=True)
class DeepseekV4DsparkArenaLayout:
    regions: tuple[DeepseekV4DsparkArenaRegion, ...]
    total_bytes: int
    persistent_kv_bytes: int

    def region(self, name: str) -> DeepseekV4DsparkArenaRegion:
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(name)


@dataclass(frozen=True)
class DeepseekV4DsparkKVProducerScratchLayout:
    kv_linear_offset: int
    kv_linear_bytes: int
    kv_output_offset: int
    kv_output_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class DeepseekV4DsparkEntryScratchLayout:
    target_tap_concat_offset: int
    target_tap_concat_bytes: int
    projection_scratch_offset: int
    projection_scratch_bytes: int
    prompt_positions_offset: int
    prompt_positions_bytes: int
    prompt_main_slots_offset: int
    prompt_main_slots_bytes: int
    prompt_cos_sin_offset: int
    prompt_cos_sin_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class DeepseekV4DsparkProposalScratchLayout:
    draft_input_token_ids_offset: int
    draft_input_token_ids_bytes: int
    residual_ping_offset: int
    residual_ping_bytes: int
    residual_pong_offset: int
    residual_pong_bytes: int
    collapsed_hidden_offset: int
    collapsed_hidden_bytes: int
    positions_offset: int
    positions_bytes: int
    main_slots_offset: int
    main_slots_bytes: int
    cos_sin_offset: int
    cos_sin_bytes: int
    post_ping_offset: int
    post_ping_bytes: int
    comb_ping_offset: int
    comb_ping_bytes: int
    post_pong_offset: int
    post_pong_bytes: int
    comb_pong_offset: int
    comb_pong_bytes: int
    selected_indices_offset: int
    selected_indices_bytes: int
    selected_lengths_offset: int
    selected_lengths_bytes: int

    def buffer(self, name: str) -> tuple[int, int]:
        buffers = {
            "draft_input_token_ids": (
                self.draft_input_token_ids_offset,
                self.draft_input_token_ids_bytes,
            ),
            "residual_ping": (self.residual_ping_offset, self.residual_ping_bytes),
            "residual_pong": (self.residual_pong_offset, self.residual_pong_bytes),
            "collapsed_hidden": (
                self.collapsed_hidden_offset,
                self.collapsed_hidden_bytes,
            ),
            "positions": (self.positions_offset, self.positions_bytes),
            "main_slots": (self.main_slots_offset, self.main_slots_bytes),
            "cos_sin_cache": (self.cos_sin_offset, self.cos_sin_bytes),
            "post_ping": (self.post_ping_offset, self.post_ping_bytes),
            "comb_ping": (self.comb_ping_offset, self.comb_ping_bytes),
            "post_pong": (self.post_pong_offset, self.post_pong_bytes),
            "comb_pong": (self.comb_pong_offset, self.comb_pong_bytes),
            "selected_indices": (
                self.selected_indices_offset,
                self.selected_indices_bytes,
            ),
            "selected_lengths": (
                self.selected_lengths_offset,
                self.selected_lengths_bytes,
            ),
        }
        return buffers[str(name)]


@dataclass(frozen=True)
class DeepseekV4DsparkAttentionContract:
    variant: str
    max_batch: int
    rows: int
    heads: int
    max_chunks_per_row: int
    scratch_bytes: int
    base_attention_scratch_bytes: int
    cache_format: str
    cache_page_bytes: int

    @property
    def target_ring_slots(self) -> int:
        return DS4_DSPARK_SLIDING_WINDOW

    @property
    def proposal_slots(self) -> tuple[int, int, int, int, int]:
        return tuple(
            range(
                DS4_DSPARK_SLIDING_WINDOW,
                DS4_DSPARK_SELECTION_WIDTH,
            )
        )

    @property
    def selection_width(self) -> int:
        return DS4_DSPARK_SELECTION_WIDTH

    @property
    def cache_page_tokens(self) -> int:
        return DS4_DSPARK_CACHE_PAGE_TOKENS

    @property
    def selected_indices_bytes(self) -> int:
        return self.rows * self.selection_width * 4

    @property
    def selected_lengths_bytes(self) -> int:
        return self.rows * 4

    @property
    def workspace_growth_bytes(self) -> int:
        return max(0, self.scratch_bytes - self.base_attention_scratch_bytes)

    @property
    def all_proposal_kv_visible_to_every_query(self) -> bool:
        return True

    @property
    def proposal_attention_is_causal(self) -> bool:
        return False

    def physical_selection(
        self,
        *,
        request_index: int,
        cache_window_start: int,
        main_context_end: int,
    ) -> tuple[int, ...]:
        request_index = int(request_index)
        cache_window_start = int(cache_window_start)
        main_context_end = int(main_context_end)
        if not 0 <= request_index < self.max_batch:
            raise ValueError(
                f"dSpark request index must be in [0, {self.max_batch}), got {request_index}"
            )
        if not 0 <= cache_window_start < main_context_end:
            raise ValueError(
                "dSpark active target window must be non-empty and ordered, got "
                f"{cache_window_start}..{main_context_end}"
            )
        if main_context_end - cache_window_start > self.target_ring_slots:
            raise ValueError(
                "dSpark active target window exceeds its 128-slot ring: "
                f"{cache_window_start}..{main_context_end}"
            )
        page_base = request_index * self.cache_page_tokens
        main_slots = sorted(
            position % self.target_ring_slots
            for position in range(cache_window_start, main_context_end)
        )
        return tuple(page_base + slot for slot in main_slots) + tuple(
            page_base + slot for slot in self.proposal_slots
        )


@dataclass(frozen=True)
class DeepseekV4DsparkContract:
    geometry: DeepseekV4DsparkGeometry
    max_batch: int
    max_main_rows: int
    blocks: tuple[
        DeepseekV4DsparkBlockContract,
        DeepseekV4DsparkBlockContract,
        DeepseekV4DsparkBlockContract,
    ]
    arena: DeepseekV4DsparkArenaLayout
    entry_scratch: DeepseekV4DsparkEntryScratchLayout
    proposal_scratch: DeepseekV4DsparkProposalScratchLayout
    target_main_kv_producer_scratch: DeepseekV4DsparkKVProducerScratchLayout
    decode_attention: DeepseekV4DsparkAttentionContract
    prefill_stages: tuple[str, ...] = (
        "concatenate-three-target-taps-and-project-once",
        "prime-each-dspark-block-target-main-kv-ring",
        "return-without-proposal-head",
    )
    decode_stages: tuple[str, ...] = (
        "concatenate-three-target-taps-and-project-once",
        "build-anchor-plus-four-noise-token-proposal-state",
        "run-three-dual-source-dspark-blocks-with-tp4-dispatch-barriers",
        "collapse-hyper-connections-and-run-shared-lm-head",
        "sequentially-add-markov-bias-and-sample-five-tokens",
        "score-five-token-confidence-from-hidden-plus-markov-embedding",
    )
    owner: str = "coordinator-gpu0-request-local-integrated-dspark"
    status: str = "qualified-integrated-dspark-composite-not-active"

    @property
    def serving_allocates(self) -> bool:
        return False

    @property
    def target_taps_are_caller_owned(self) -> bool:
        return True

    @property
    def target_projection_is_shared_by_all_blocks(self) -> bool:
        return True

    @property
    def cache_is_request_local(self) -> bool:
        return True

    @property
    def cache_device_id(self) -> int:
        return 0

    @property
    def proposal_input_token_pattern(self) -> tuple[str, ...]:
        return ("anchor", "noise", "noise", "noise", "noise")

    @property
    def dispatch_barriers_per_decode(self) -> int:
        return self.geometry.blocks

    @property
    def cuda_graph_segments_per_decode(self) -> int:
        # Entry and head surround two GLMRT-style graph segments per sparse
        # block. Spark dispatch is the barrier between each pair.
        return 2 + 2 * self.geometry.blocks

    @property
    def sparse_workspace_reused_across_blocks(self) -> bool:
        return True

    @property
    def entry_scratch_reuses_sparse_workspace(self) -> bool:
        return (
            self.entry_scratch.total_bytes
            <= self.arena.region("reused_sparse_block_workspace").nbytes
        )

    @property
    def entry_activation_block_size(self) -> int:
        return DS4_DSPARK_ENTRY_ACTIVATION_BLOCK_SIZE

    @property
    def entry_capture_surface(self) -> tuple[str, str]:
        return (
            "prepare_deepseek_v4_dspark_entry_projection",
            "capture_deepseek_v4_dspark_entry_projection",
        )

    @property
    def target_main_kv_producer_scratch_reused_across_blocks(self) -> bool:
        return True

    @property
    def prompt_prime_computes_query(self) -> bool:
        return False

    @property
    def prompt_prime_producer_surface(self) -> tuple[str, str, str]:
        return ("plan_kv", "bind_kv", "run_kv")

    @property
    def prompt_prime_capture_surface(self) -> tuple[str, str]:
        return (
            "prepare_deepseek_v4_dspark_prompt_prime",
            "capture_deepseek_v4_dspark_prompt_prime",
        )

    @property
    def proposal_entry_capture_surface(self) -> tuple[str, str]:
        return (
            "prepare_deepseek_v4_dspark_proposal_entry",
            "capture_deepseek_v4_dspark_proposal_entry",
        )

    @property
    def block_attention_capture_surface(self) -> tuple[str, str]:
        return (
            "prepare_deepseek_v4_dspark_block_attention",
            "capture_deepseek_v4_dspark_block_attention",
        )

    @property
    def block_post_dispatch_capture_surface(self) -> tuple[str, str]:
        return (
            "prepare_deepseek_v4_dspark_block_post_dispatch",
            "capture_deepseek_v4_dspark_block_post_dispatch",
        )

    @property
    def terminal_collapse_capture_surface(self) -> tuple[str, str]:
        return (
            "prepare_deepseek_v4_dspark_terminal_collapse",
            "capture_deepseek_v4_dspark_terminal_collapse",
        )

    @property
    def terminal_batch_axis(self) -> str:
        return "active-requests-times-five-proposal-rows"

    @property
    def terminal_serial_dependency_axis(self) -> str:
        return "five-markov-positions-only"

    @property
    def terminal_preserves_unnormalized_hidden(self) -> bool:
        # The shared LM head consumes RMS-normalized collapsed hidden, while
        # the confidence head consumes the unnormalized collapsed hidden.
        return True

    @property
    def terminal_jointly_issues_active_requests(self) -> bool:
        return True

    @property
    def proposal_scratch_reuses_markov_storage(self) -> bool:
        return True

    @property
    def expert_tensor_parallel(self) -> int:
        return DS4_EXPERT_TP

    @property
    def expert_parallel(self) -> bool:
        return False

    @property
    def recurrent_mtp_state(self) -> bool:
        return False


@dataclass(frozen=True)
class DeepseekV4DsparkPromptPrimeBlockBinding:
    contract: DeepseekV4DsparkContract
    block: DeepseekV4DsparkBlockContract
    producer_binding: Any
    serving_allocates: ClassVar[bool] = False
    cuda_graph_safe: ClassVar[bool] = True
    computes_query: ClassVar[bool] = False


@dataclass(frozen=True)
class DeepseekV4DsparkProposalEntryBinding:
    """One request's embeddings through block-0 attention pre-mix.

    DeepSeek serializes the owning weights under ``mtp.0.*``, but this is the
    entry to the integrated dSpark stack.  It carries no recurrent GLM MTP
    state and always produces the normal mHC state consumed by block zero. The
    reference repeats each embedding into four identical lanes, so the entry
    fast path uses the exactly equivalent lane-summed HC function.
    """

    contract: DeepseekV4DsparkContract
    mhc_binding: Any
    residual_input: Any
    hc_fn: Any
    hc_scale: Any
    hc_base: Any
    norm_weight: Any
    rms_eps: float
    hc_eps: float
    sinkhorn_iters: int
    norm_eps: float
    serving_allocates: ClassVar[bool] = False
    cuda_graph_safe: ClassVar[bool] = True
    recurrent_mtp_state: ClassVar[bool] = False


@dataclass(frozen=True)
class DeepseekV4DsparkBlockBinding:
    contract: DeepseekV4DsparkContract
    block: DeepseekV4DsparkBlockContract
    sparse_binding: Any
    serving_allocates: ClassVar[bool] = False
    cuda_graph_safe: ClassVar[bool] = True
    cuda_graph_segments: ClassVar[int] = 2
    dispatch_barrier_between_graphs: ClassVar[bool] = True
    expert_tensor_parallel: ClassVar[int] = DS4_EXPERT_TP
    expert_parallel: ClassVar[bool] = False

    @property
    def dispatch_hidden(self) -> Any:
        return self.sparse_binding.dispatch_hidden

    @property
    def route_indices(self) -> Any:
        return self.sparse_binding.arena_binding.route_indices

    @property
    def route_weights(self) -> Any:
        return self.sparse_binding.arena_binding.route_weights


@dataclass(frozen=True)
class DeepseekV4DsparkTerminalReferenceOutput:
    """Exact checkpoint math for the coordinator-owned dSpark terminal.

    This is an allocation-tolerant oracle, not a serving path. Serving binds
    the same tensors to the preplanned arena and issues the shared head for
    every active request together. Only the five Markov positions are
    sequential because position ``i + 1`` depends on the token sampled at
    position ``i``.
    """

    output_token_ids: Any
    logits: Any
    collapsed_hidden: Any
    normalized_hidden: Any
    markov_embeddings: Any
    confidence_logits: Any
    conditional_confidence: Any


def run_deepseek_v4_dspark_terminal_reference(
    terminal_residual: Any,
    anchor_token_ids: Any,
    *,
    hc_head_fn: Any,
    hc_head_scale: Any,
    hc_head_base: Any,
    norm_weight: Any,
    shared_head_weight: Any,
    markov_w1: Any,
    markov_w2: Any,
    confidence_weight: Any,
    temperature: float = 0.0,
    exponential_noise: Any | None = None,
    rms_eps: float = 1.0e-6,
    hc_eps: float = 1.0e-6,
    norm_eps: float = 1.0e-6,
) -> DeepseekV4DsparkTerminalReferenceOutput:
    """Run DeepSeek's dSpark head with a joint request batch.

    ``terminal_residual`` is ``[requests, 5, 4, hidden]`` after block two's
    FFN HC-post. The storage weights are named ``mtp.2.*`` by the checkpoint,
    but this is the integrated dSpark terminal and carries no recurrent MTP
    state.
    """

    import math

    import torch
    import torch.nn.functional as F

    if not isinstance(terminal_residual, torch.Tensor) or terminal_residual.ndim != 4:
        raise ValueError(
            "dSpark terminal residual must be rank-4 [requests, 5, 4, hidden]"
        )
    requests, proposal_tokens, hc_mult, hidden = map(
        int, terminal_residual.shape
    )
    if requests <= 0 or proposal_tokens != DS4_DSPARK_PROPOSAL_TOKENS:
        raise ValueError(
            "dSpark terminal requires at least one request and exactly five "
            f"proposal rows, got {tuple(terminal_residual.shape)}"
        )
    if hc_mult != DS4_HC_MULT:
        raise ValueError(
            f"dSpark terminal HC multiplier must be {DS4_HC_MULT}, got {hc_mult}"
        )
    if tuple(anchor_token_ids.shape) != (requests,):
        raise ValueError(
            f"dSpark terminal anchors must have shape {(requests,)}, got "
            f"{tuple(anchor_token_ids.shape)}"
        )
    vocab, head_hidden = map(int, shared_head_weight.shape)
    markov_vocab, markov_rank = map(int, markov_w1.shape)
    expected_shapes = {
        "hc_head_fn": (DS4_HC_MULT, DS4_HC_MULT * hidden),
        "hc_head_scale": (1,),
        "hc_head_base": (DS4_HC_MULT,),
        "norm_weight": (hidden,),
        "shared_head_weight": (vocab, hidden),
        "markov_w1": (vocab, markov_rank),
        "markov_w2": (vocab, markov_rank),
        "confidence_weight": (1, hidden + markov_rank),
    }
    tensors = {
        "hc_head_fn": hc_head_fn,
        "hc_head_scale": hc_head_scale,
        "hc_head_base": hc_head_base,
        "norm_weight": norm_weight,
        "shared_head_weight": shared_head_weight,
        "markov_w1": markov_w1,
        "markov_w2": markov_w2,
        "confidence_weight": confidence_weight,
    }
    if head_hidden != hidden or markov_vocab != vocab:
        raise ValueError("dSpark shared and Markov heads disagree on geometry")
    for name, expected in expected_shapes.items():
        value = tensors[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
            raise ValueError(
                f"dSpark terminal {name} must have shape {expected}, got "
                f"{getattr(value, 'shape', None)}"
            )
        if value.device != terminal_residual.device:
            raise ValueError(f"dSpark terminal {name} must share the residual device")
    if anchor_token_ids.device != terminal_residual.device:
        raise ValueError("dSpark terminal anchors must share the residual device")
    if not all(
        math.isfinite(float(value)) and float(value) > 0.0
        for value in (rms_eps, hc_eps, norm_eps)
    ):
        raise ValueError("dSpark terminal epsilons must be finite and positive")
    if not math.isfinite(float(temperature)) or float(temperature) < 0.0:
        raise ValueError("dSpark terminal temperature must be finite and nonnegative")

    rows = requests * proposal_tokens
    residual = terminal_residual.reshape(rows, DS4_HC_MULT, hidden)
    flat = residual.flatten(1).float()
    mixes = F.linear(flat, hc_head_fn.float()) * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + float(rms_eps)
    )
    pre = (
        torch.sigmoid(
            mixes * hc_head_scale.float() + hc_head_base.float()
        )
        + float(hc_eps)
    )
    collapsed = (pre.unsqueeze(-1) * residual.float()).sum(dim=1).to(
        terminal_residual.dtype
    )
    collapsed_f32 = collapsed.float()
    normalized = (
        collapsed_f32
        * torch.rsqrt(
            collapsed_f32.square().mean(dim=-1, keepdim=True)
            + float(norm_eps)
        )
        * norm_weight.float()
    ).to(terminal_residual.dtype)
    logits = F.linear(normalized.float(), shared_head_weight.float()).reshape(
        requests, proposal_tokens, vocab
    )

    output_ids = torch.empty(
        (requests, proposal_tokens + 1),
        dtype=anchor_token_ids.dtype,
        device=anchor_token_ids.device,
    )
    output_ids[:, 0].copy_(anchor_token_ids)
    markov_embeddings = []
    if exponential_noise is not None and tuple(exponential_noise.shape) != (
        requests,
        proposal_tokens,
        vocab,
    ):
        raise ValueError(
            "dSpark terminal exponential noise must have shape "
            f"{(requests, proposal_tokens, vocab)}"
        )
    for position in range(proposal_tokens):
        markov_embed = F.embedding(output_ids[:, position], markov_w1)
        logits[:, position].add_(
            F.linear(markov_embed.float(), markov_w2.float())
        )
        markov_embeddings.append(markov_embed)
        if temperature == 0.0:
            sampled = logits[:, position].argmax(dim=-1)
        else:
            probs = torch.softmax(
                logits[:, position] / max(float(temperature), 1.0e-5),
                dim=-1,
                dtype=torch.float32,
            )
            noise = (
                torch.empty_like(probs).exponential_()
                if exponential_noise is None
                else exponential_noise[:, position]
            )
            if bool(torch.any(noise <= 0.0)):
                raise ValueError("dSpark terminal exponential noise must be positive")
            sampled = probs.div(noise).argmax(dim=-1)
        output_ids[:, position + 1].copy_(sampled)
    markov = torch.stack(markov_embeddings, dim=1)
    collapsed_by_request = collapsed.reshape(requests, proposal_tokens, hidden)
    confidence_logits = F.linear(
        torch.cat([collapsed_by_request.float(), markov.float()], dim=-1),
        confidence_weight.float(),
    ).squeeze(-1)
    return DeepseekV4DsparkTerminalReferenceOutput(
        output_token_ids=output_ids,
        logits=logits,
        collapsed_hidden=collapsed_by_request,
        normalized_hidden=normalized.reshape(requests, proposal_tokens, hidden),
        markov_embeddings=markov,
        confidence_logits=confidence_logits,
        conditional_confidence=torch.sigmoid(confidence_logits),
    )


def bind_deepseek_v4_dspark_proposal_entry(
    contract: DeepseekV4DsparkContract,
    *,
    scratch: Any,
    residual_input: Any,
    residual_output: Any,
    normalized_output: Any,
    post_output: Any,
    comb_output: Any,
    hc_fn: Any,
    hc_scale: Any,
    hc_base: Any,
    norm_weight: Any,
    rms_eps: float = 1.0e-6,
    hc_eps: float = 1.0e-6,
    sinkhorn_iters: int = 20,
    norm_eps: float = 1.0e-6,
) -> DeepseekV4DsparkProposalEntryBinding:
    """Bind block zero's attention pre-mix for one five-token proposal."""

    import torch
    from b12x.norm import mhc

    if not isinstance(contract, DeepseekV4DsparkContract):
        raise TypeError("dSpark proposal entry requires a planned contract")
    rows = contract.geometry.proposal_tokens
    hidden = contract.geometry.hidden
    expected = (rows, hidden)
    if (
        not isinstance(residual_input, torch.Tensor)
        or residual_input.shape != expected
        or residual_input.dtype != torch.bfloat16
        or not residual_input.is_contiguous()
    ):
        raise ValueError(
            f"dSpark proposal residual input must be contiguous BF16 {expected}"
        )
    device = residual_input.device
    _require_tensor(
        "residual_output",
        residual_output,
        (rows, DS4_HC_MULT, hidden),
        torch.bfloat16,
        device,
    )
    _require_tensor(
        "normalized_output", normalized_output, (rows, hidden), torch.bfloat16, device
    )
    _require_tensor("post_output", post_output, (rows, DS4_HC_MULT), torch.float32, device)
    _require_tensor(
        "comb_output",
        comb_output,
        (rows, DS4_HC_MULT, DS4_HC_MULT),
        torch.float32,
        device,
    )
    _require_tensor(
        "hc_fn",
        hc_fn,
        ((2 + DS4_HC_MULT) * DS4_HC_MULT, hidden),
        torch.float32,
        device,
    )
    _require_tensor("hc_scale", hc_scale, (3,), torch.float32, device)
    _require_tensor(
        "hc_base",
        hc_base,
        ((2 + DS4_HC_MULT) * DS4_HC_MULT,),
        torch.float32,
        device,
    )
    _require_tensor("norm_weight", norm_weight, (hidden,), torch.bfloat16, device)
    plan = mhc.plan(
        mhc.Caps(
            device=device,
            max_tokens=rows,
            hidden_size=hidden,
            split_k=DS4_HC_MULT * hidden // 256,
            dtype=torch.bfloat16,
        )
    )
    mhc_binding = mhc.bind(
        plan,
        scratch=scratch,
        tokens=rows,
        expected_m=rows,
        y=normalized_output,
        post=post_output,
        comb=comb_output,
        out=residual_output,
    )
    return DeepseekV4DsparkProposalEntryBinding(
        contract=contract,
        mhc_binding=mhc_binding,
        residual_input=residual_input,
        hc_fn=hc_fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        norm_weight=norm_weight,
        rms_eps=float(rms_eps),
        hc_eps=float(hc_eps),
        sinkhorn_iters=int(sinkhorn_iters),
        norm_eps=float(norm_eps),
    )


def run_deepseek_v4_dspark_proposal_entry(
    binding: DeepseekV4DsparkProposalEntryBinding,
) -> tuple[Any, Any, Any, Any]:
    from b12x.norm import mhc

    if not isinstance(binding, DeepseekV4DsparkProposalEntryBinding):
        raise TypeError("dSpark proposal entry requires its fixed binding")
    return mhc.run_pre(
        binding.residual_input,
        binding.hc_fn,
        binding.hc_scale,
        binding.hc_base,
        rms_eps=binding.rms_eps,
        hc_eps=binding.hc_eps,
        sinkhorn_iters=binding.sinkhorn_iters,
        norm_weight=binding.norm_weight,
        norm_eps=binding.norm_eps,
        binding=binding.mhc_binding,
    )


def bind_deepseek_v4_dspark_prompt_prime_block(
    contract: DeepseekV4DsparkContract,
    *,
    block_index: int,
    scratch: Any,
    projected_target_main: Any,
    positions: Any,
    main_slots: Any,
    cos_sin_cache: Any,
    main_kv_cache: Any,
    producer_weights: Any,
    producer_eps: float = 1.0e-6,
) -> DeepseekV4DsparkPromptPrimeBlockBinding:
    """Bind one block's KV-only prompt prime with the shared scratch arena."""

    import torch
    from b12x.attention import dsv4_producer

    block = _dspark_block(contract, block_index)
    if (
        not isinstance(projected_target_main, torch.Tensor)
        or projected_target_main.ndim != 2
        or projected_target_main.dtype != torch.bfloat16
        or int(projected_target_main.shape[1]) != contract.geometry.hidden
    ):
        raise ValueError(
            "dSpark projected target main must be a BF16 matrix at model hidden width"
        )
    rows = int(projected_target_main.shape[0])
    if not 1 <= rows <= contract.max_main_rows:
        raise ValueError(
            "dSpark prompt-prime rows must be in "
            f"[1, {contract.max_main_rows}], got {rows}"
        )
    producer_plan = dsv4_producer.plan_kv(
        dsv4_producer.Caps(
            device=projected_target_main.device,
            max_tokens=contract.max_main_rows,
            hidden=contract.geometry.hidden,
            q_lora_rank=(1_024 if contract.geometry.variant == "flash" else 1_536),
            heads=contract.geometry.attention_heads,
            head_dim=contract.geometry.head_dim,
            nope_dim=448,
            rope_dim=64,
            page_size=contract.decode_attention.cache_page_tokens,
            dtype=torch.bfloat16,
            cache_format=contract.decode_attention.cache_format,
        )
    )
    producer_binding = dsv4_producer.bind_kv(
        producer_plan,
        scratch=scratch,
        hidden_states=projected_target_main,
        positions=positions,
        main_slots=main_slots,
        cos_sin_cache=cos_sin_cache,
        main_kv_cache=main_kv_cache,
        weights=producer_weights,
        eps=float(producer_eps),
        expected_m=contract.max_main_rows,
    )
    return DeepseekV4DsparkPromptPrimeBlockBinding(
        contract=contract,
        block=block,
        producer_binding=producer_binding,
    )


def run_deepseek_v4_dspark_prompt_prime_block(
    binding: DeepseekV4DsparkPromptPrimeBlockBinding,
) -> Any:
    from b12x.attention import dsv4_producer

    if not isinstance(binding, DeepseekV4DsparkPromptPrimeBlockBinding):
        raise TypeError("dSpark prompt prime requires its fixed block binding")
    return dsv4_producer.run_kv(binding=binding.producer_binding)


def bind_deepseek_v4_dspark_block(
    contract: DeepseekV4DsparkContract,
    *,
    block_index: int,
    arena: Any,
    attention_kwargs: dict[str, Any],
    sparse_kwargs: dict[str, Any],
) -> DeepseekV4DsparkBlockBinding:
    """Bind one dual-source dSpark block around the normal split TP4 graph.

    The two kwargs maps are consumed only while binding at startup. Replay uses
    the returned fixed object and performs no Python-side tensor allocation.
    """

    block = _dspark_block(contract, block_index)
    forbidden_attention = {"contract", "arena"}.intersection(attention_kwargs)
    if forbidden_attention:
        raise ValueError(
            "dSpark owns attention binding fields "
            f"{sorted(forbidden_attention)}"
        )
    forbidden_sparse = {
        "contract",
        "arena_binding",
        "attention_binding",
        "attention_runner",
    }.intersection(sparse_kwargs)
    if forbidden_sparse:
        raise ValueError(
            f"dSpark owns sparse binding fields {sorted(forbidden_sparse)}"
        )
    hidden_states = attention_kwargs.get("hidden_states")
    tokens = int(getattr(hidden_states, "shape", (0,))[0])
    if (
        tokens <= 0
        or tokens > block.expert_handoff.max_rows
        or tokens % contract.geometry.proposal_tokens
    ):
        raise ValueError(
            "dSpark block hidden rows must contain one five-row proposal group "
            f"per active request, got {tokens}"
        )
    arena_binding = bind_deepseek_v4_sparse_block_arena(
        block.expert_handoff,
        arena=arena,
        tokens=tokens,
    )
    attention_binding = bind_deepseek_v4_sliding_attention_layer(
        block.expert_handoff.attention,
        arena=arena_binding.attention_arena,
        **attention_kwargs,
    )
    sparse_binding = bind_deepseek_v4_sparse_block(
        block.expert_handoff,
        arena_binding=arena_binding,
        attention_binding=attention_binding,
        attention_runner=run_deepseek_v4_sliding_attention_layer,
        **sparse_kwargs,
    )
    return DeepseekV4DsparkBlockBinding(
        contract=contract,
        block=block,
        sparse_binding=sparse_binding,
    )


def run_deepseek_v4_dspark_block_pre_dispatch(
    binding: DeepseekV4DsparkBlockBinding,
) -> tuple[Any, Any, Any, Any]:
    if not isinstance(binding, DeepseekV4DsparkBlockBinding):
        raise TypeError("dSpark pre-dispatch requires its fixed block binding")
    return run_deepseek_v4_sparse_block_attention(binding.sparse_binding)


def run_deepseek_v4_dspark_block_post_dispatch(
    binding: DeepseekV4DsparkBlockBinding,
) -> tuple[Any, Any, Any, Any]:
    if not isinstance(binding, DeepseekV4DsparkBlockBinding):
        raise TypeError("dSpark post-dispatch requires its fixed block binding")
    return run_deepseek_v4_sparse_block_post_dispatch(binding.sparse_binding)


def _dspark_block(
    contract: DeepseekV4DsparkContract, block_index: int
) -> DeepseekV4DsparkBlockContract:
    if not isinstance(contract, DeepseekV4DsparkContract):
        raise TypeError("dSpark binding requires a planned contract")
    block_index = int(block_index)
    if not 0 <= block_index < contract.geometry.blocks:
        raise ValueError(
            f"dSpark block index must be in [0, {contract.geometry.blocks}), got {block_index}"
        )
    return contract.blocks[block_index]


def deepseek_v4_dspark_geometry(variant: str) -> DeepseekV4DsparkGeometry:
    variant = str(variant).strip().lower()
    if variant == "flash":
        return DeepseekV4DsparkGeometry(
            variant=variant,
            hidden=4_096,
            target_layers=43,
            target_taps=(40, 41, 42),
            routed_experts=256,
            expert_intermediate=2_048,
            attention_heads=64,
            markov_rank=256,
        )
    if variant == "pro":
        return DeepseekV4DsparkGeometry(
            variant=variant,
            hidden=7_168,
            target_layers=61,
            target_taps=(58, 59, 60),
            routed_experts=384,
            expert_intermediate=3_072,
            attention_heads=128,
            markov_rank=512,
        )
    raise ValueError(
        f"DeepSeek V4 dSpark variant must be flash or pro, got {variant!r}"
    )


def plan_deepseek_v4_dspark(
    *, variant: str, max_batch: int, max_main_rows: int, cache_format: str = "fp8"
) -> DeepseekV4DsparkContract:
    geometry = deepseek_v4_dspark_geometry(variant)
    format_plan = deepseek_v4_kv_format(cache_format)
    max_batch = int(max_batch)
    max_main_rows = int(max_main_rows)
    if not 1 <= max_batch <= 128:
        raise ValueError(f"dSpark max_batch must be in [1, 128], got {max_batch}")
    if not 1 <= max_main_rows <= 1_048_576:
        raise ValueError(
            "dSpark max_main_rows must be in [1, 1048576], "
            f"got {max_main_rows}"
        )
    proposal_rows = max_batch * geometry.proposal_tokens
    entry_scratch = _plan_dspark_entry_scratch(
        geometry=geometry,
        max_rows=max_main_rows,
    )
    target_main_kv_producer_scratch = _plan_target_main_kv_producer_scratch(
        geometry=geometry,
        max_rows=max_main_rows,
    )
    expert_handoff = plan_deepseek_v4_sparse_block(
        variant=geometry.variant,
        mode="decode",
        compression=0,
        max_rows=proposal_rows,
        source_pages=max_batch,
        max_positions=1_048_576,
        swa_width=DS4_DSPARK_SELECTION_WIDTH,
        cache_format=format_plan.name,
    )
    decode_attention = _plan_dspark_decode_attention(
        geometry=geometry,
        max_batch=max_batch,
        base_attention_scratch_bytes=expert_handoff.attention.arena.attention_scratch_bytes,
        cache_format=format_plan.name,
        cache_page_bytes=format_plan.main_page_bytes,
    )
    blocks = tuple(
        DeepseekV4DsparkBlockContract(
            block_index=block_index,
            physical_layer_id=geometry.physical_block_ids[block_index],
            target_tap_id=geometry.target_taps[block_index],
            storage_prefix=geometry.storage_prefixes[block_index],
            expert_handoff=expert_handoff,
        )
        for block_index in range(geometry.blocks)
    )

    regions: list[DeepseekV4DsparkArenaRegion] = []
    offset = 0

    def reserve(name: str, nbytes: int) -> None:
        nonlocal offset
        offset = _align_up(offset)
        regions.append(DeepseekV4DsparkArenaRegion(name, offset, int(nbytes)))
        offset += int(nbytes)

    reserve(
        "projected_target_main",
        max_main_rows * geometry.hidden * 2,
    )
    reserve(
        "reused_target_main_kv_producer_scratch",
        target_main_kv_producer_scratch.total_bytes,
    )
    reserve("draft_input_token_ids", proposal_rows * 8)
    reserve(
        "proposal_residual_ping_pong",
        2 * proposal_rows * DS4_HC_MULT * geometry.hidden * 2,
    )
    reserve("proposal_collapsed_hidden", proposal_rows * geometry.hidden * 2)
    # Terminal collapse is still produced into stable request-slot storage.
    # The joint head gathers the active slot set into position-major compact
    # rows so non-contiguous live slots never force request-serial GEMMs.
    reserve("proposal_normalized_hidden", proposal_rows * geometry.hidden * 2)
    reserve("terminal_compact_normalized_hidden", proposal_rows * geometry.hidden * 2)
    reserve("shared_lm_logits", proposal_rows * geometry.vocab_size * 4)
    reserve("terminal_markov_logits", max_batch * geometry.vocab_size * 4)
    reserve("markov_embeddings", proposal_rows * geometry.markov_rank * 2)
    reserve("confidence", proposal_rows * 4)
    reserve("terminal_active_slot_ids", max_batch * 4)
    reserve("terminal_anchor_token_ids", max_batch * 4)
    reserve(
        "terminal_output_token_ids",
        max_batch * (geometry.proposal_tokens + 1) * 4,
    )
    reserve(
        "reused_dspark_attention_selected_indices",
        decode_attention.selected_indices_bytes,
    )
    reserve(
        "reused_dspark_attention_selected_lengths",
        decode_attention.selected_lengths_bytes,
    )
    reserve(
        "reused_sparse_block_workspace",
        max(
            entry_scratch.total_bytes,
            expert_handoff.arena.total_bytes
            + decode_attention.workspace_growth_bytes,
        ),
    )

    persistent_kv_bytes = (
        geometry.blocks * max_batch * decode_attention.cache_page_bytes
    )
    arena = DeepseekV4DsparkArenaLayout(
        regions=tuple(regions),
        total_bytes=_align_up(offset),
        persistent_kv_bytes=persistent_kv_bytes,
    )
    return DeepseekV4DsparkContract(
        geometry=geometry,
        max_batch=max_batch,
        max_main_rows=max_main_rows,
        blocks=blocks,
        arena=arena,
        proposal_scratch=_plan_dspark_proposal_scratch(
            geometry=geometry,
            max_batch=max_batch,
            arena=arena,
        ),
        entry_scratch=entry_scratch,
        target_main_kv_producer_scratch=target_main_kv_producer_scratch,
        decode_attention=decode_attention,
    )


def deepseek_v4_dspark_arena_nbytes(
    *, variant: str, max_batch: int, max_main_rows: int
) -> int:
    return plan_deepseek_v4_dspark(
        variant=variant,
        max_batch=max_batch,
        max_main_rows=max_main_rows,
    ).arena.total_bytes


def deepseek_v4_dspark_persistent_kv_nbytes(
    *, variant: str, max_batch: int, max_main_rows: int, cache_format: str = "fp8"
) -> int:
    return plan_deepseek_v4_dspark(
        variant=variant,
        max_batch=max_batch,
        max_main_rows=max_main_rows,
        cache_format=cache_format,
    ).arena.persistent_kv_bytes


def deepseek_v4_dspark_arena_region_offset(
    *, variant: str, max_batch: int, max_main_rows: int, region: str
) -> int:
    return plan_deepseek_v4_dspark(
        variant=variant,
        max_batch=max_batch,
        max_main_rows=max_main_rows,
    ).arena.region(str(region)).offset


def deepseek_v4_dspark_arena_region_nbytes(
    *, variant: str, max_batch: int, max_main_rows: int, region: str
) -> int:
    return plan_deepseek_v4_dspark(
        variant=variant,
        max_batch=max_batch,
        max_main_rows=max_main_rows,
    ).arena.region(str(region)).nbytes


def deepseek_v4_dspark_entry_buffer_offset(
    *, variant: str, max_batch: int, max_main_rows: int, buffer: str
) -> int:
    entry = plan_deepseek_v4_dspark(
        variant=variant,
        max_batch=max_batch,
        max_main_rows=max_main_rows,
    ).entry_scratch
    offsets = {
        "target_tap_concat": entry.target_tap_concat_offset,
        "projection_scratch": entry.projection_scratch_offset,
    }
    return offsets[str(buffer)]


def deepseek_v4_dspark_entry_buffer_nbytes(
    *, variant: str, max_batch: int, max_main_rows: int, buffer: str
) -> int:
    entry = plan_deepseek_v4_dspark(
        variant=variant,
        max_batch=max_batch,
        max_main_rows=max_main_rows,
    ).entry_scratch
    sizes = {
        "target_tap_concat": entry.target_tap_concat_bytes,
        "projection_scratch": entry.projection_scratch_bytes,
    }
    return sizes[str(buffer)]


def deepseek_v4_dspark_prompt_buffer_offset(
    *, variant: str, max_batch: int, max_main_rows: int, buffer: str
) -> int:
    entry = plan_deepseek_v4_dspark(
        variant=variant,
        max_batch=max_batch,
        max_main_rows=max_main_rows,
    ).entry_scratch
    offsets = {
        "positions": entry.prompt_positions_offset,
        "main_slots": entry.prompt_main_slots_offset,
        "cos_sin_cache": entry.prompt_cos_sin_offset,
    }
    return offsets[str(buffer)]


def deepseek_v4_dspark_prompt_buffer_nbytes(
    *, variant: str, max_batch: int, max_main_rows: int, buffer: str
) -> int:
    entry = plan_deepseek_v4_dspark(
        variant=variant,
        max_batch=max_batch,
        max_main_rows=max_main_rows,
    ).entry_scratch
    sizes = {
        "positions": entry.prompt_positions_bytes,
        "main_slots": entry.prompt_main_slots_bytes,
        "cos_sin_cache": entry.prompt_cos_sin_bytes,
    }
    return sizes[str(buffer)]


def deepseek_v4_dspark_proposal_buffer_offset(
    *, variant: str, max_batch: int, max_main_rows: int, buffer: str
) -> int:
    return plan_deepseek_v4_dspark(
        variant=variant,
        max_batch=max_batch,
        max_main_rows=max_main_rows,
    ).proposal_scratch.buffer(buffer)[0]


def deepseek_v4_dspark_proposal_buffer_nbytes(
    *, variant: str, max_batch: int, max_main_rows: int, buffer: str
) -> int:
    return plan_deepseek_v4_dspark(
        variant=variant,
        max_batch=max_batch,
        max_main_rows=max_main_rows,
    ).proposal_scratch.buffer(buffer)[1]


def deepseek_v4_dspark_block_buffer_offset(
    *, variant: str, max_batch: int, max_main_rows: int, buffer: str
) -> int:
    contract = plan_deepseek_v4_dspark(
        variant=variant,
        max_batch=max_batch,
        max_main_rows=max_main_rows,
    )
    regions = {
        name: (offset, nbytes)
        for name, offset, nbytes in contract.blocks[0].expert_handoff.arena.regions()
    }
    return regions[str(buffer)][0]


def deepseek_v4_dspark_block_buffer_nbytes(
    *, variant: str, max_batch: int, max_main_rows: int, buffer: str
) -> int:
    contract = plan_deepseek_v4_dspark(
        variant=variant,
        max_batch=max_batch,
        max_main_rows=max_main_rows,
    )
    regions = {
        name: (offset, nbytes)
        for name, offset, nbytes in contract.blocks[0].expert_handoff.arena.regions()
    }
    return regions[str(buffer)][1]


def prepare_deepseek_v4_dspark_entry_projection(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    _run_deepseek_v4_dspark_entry_projection(ctx, prepare_only=True, **kwargs)


def capture_deepseek_v4_dspark_entry_projection(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    _run_deepseek_v4_dspark_entry_projection(ctx, prepare_only=False, **kwargs)


def _run_deepseek_v4_dspark_entry_projection(
    ctx: dict[str, Any], *, prepare_only: bool, **kwargs: Any
) -> None:
    import torch
    from ds4rt_reference.b12x_spark_capture import _raw_tensor
    from b12x.gemm import block_fp8_linear

    variant = str(kwargs["variant"])
    rows = int(kwargs["rows"])
    max_rows = int(kwargs["max_rows"])
    contract = plan_deepseek_v4_dspark(
        variant=variant,
        max_batch=int(kwargs.get("max_batch", 16)),
        max_main_rows=max_rows,
    )
    geometry = contract.geometry
    if not 1 <= rows <= max_rows:
        raise ValueError(
            f"dSpark entry projection rows must be in [1, {max_rows}], got {rows}"
        )
    buffers = ctx["buffers"]
    required = {
        "target_tap_concat": rows
        * geometry.target_projection_input_width
        * 2,
        "main_proj_weight": geometry.hidden
        * geometry.target_projection_input_width,
        "main_proj_scale": _ceil_div(geometry.hidden, 128)
        * _ceil_div(geometry.target_projection_input_width, 128),
        "projection_scratch": contract.entry_scratch.projection_scratch_bytes,
        "projected_target_main": rows * geometry.hidden * 2,
    }
    device_id = _validate_device_buffers(
        buffers, required, anchor="target_tap_concat"
    )
    stream = torch.cuda.ExternalStream(int(ctx["cuda_stream"]), device=device_id)
    state_key = (
        device_id,
        int(buffers["main_proj_weight"]["ptr"]),
        int(buffers["main_proj_scale"]["ptr"]),
        geometry.hidden,
        max_rows,
    )
    prepared_key = state_key + (rows,)
    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        state = _ENTRY_PROJECTION_STATE.get(state_key)
        if state is None:
            weight = _raw_tensor(
                buffers["main_proj_weight"],
                (geometry.hidden, geometry.target_projection_input_width),
                torch.float8_e4m3fn,
                1,
                name="dSpark main_proj weight",
            )
            scale = _raw_tensor(
                buffers["main_proj_scale"],
                (
                    _ceil_div(geometry.hidden, 128),
                    _ceil_div(geometry.target_projection_input_width, 128),
                ),
                torch.float8_e8m0fnu,
                1,
                name="dSpark main_proj scale",
            )
            packed_weight = block_fp8_linear.pack_weight(weight, scale)
            projection_plan = block_fp8_linear.plan(
                block_fp8_linear.Caps(
                    device=torch.device("cuda", device_id),
                    max_tokens=max_rows,
                    in_features=geometry.target_projection_input_width,
                    out_features=geometry.hidden,
                    output_dtype=torch.bfloat16,
                )
            )
            (scratch_spec,) = projection_plan.scratch_specs()
            if int(scratch_spec.nbytes) != contract.entry_scratch.projection_scratch_bytes:
                raise RuntimeError(
                    "dSpark entry projection scratch changed after startup qualification"
                )
            state = (projection_plan, packed_weight)
            _ENTRY_PROJECTION_STATE[state_key] = state
        projection_plan, packed_weight = state
        source = _raw_tensor(
            buffers["target_tap_concat"],
            (rows, geometry.target_projection_input_width),
            torch.bfloat16,
            2,
            name="dSpark target-tap concat",
        )
        scratch = _raw_tensor(
            buffers["projection_scratch"],
            (contract.entry_scratch.projection_scratch_bytes,),
            torch.uint8,
            1,
            name="dSpark entry projection scratch",
        )
        output = _raw_tensor(
            buffers["projected_target_main"],
            (rows, geometry.hidden, 1),
            torch.bfloat16,
            2,
            name="dSpark projected target main",
        )
        binding = block_fp8_linear.bind(
            projection_plan,
            scratch=scratch,
            source=source,
            packed_weight=packed_weight,
            output=output,
            expected_m=rows,
            activation_block_size=DS4_DSPARK_ENTRY_ACTIVATION_BLOCK_SIZE,
        )
        if not prepare_only and prepared_key not in _ENTRY_PROJECTION_PREPARED:
            raise RuntimeError(
                "dSpark entry projection capture was not prepared for its row bucket"
            )
        block_fp8_linear.run(binding=binding, stream=stream)
        if prepare_only:
            _ENTRY_PROJECTION_PREPARED.add(prepared_key)


def _validate_device_buffers(
    buffers: dict[str, dict[str, Any]],
    required: dict[str, int],
    *,
    anchor: str,
) -> int:
    missing = sorted(set(required) - set(buffers))
    if missing:
        raise ValueError(
            f"dSpark entry projection is missing buffers: {', '.join(missing)}"
        )
    device_id = int(buffers[anchor]["device_id"])
    for name, required_bytes in required.items():
        buffer = buffers[name]
        if int(buffer["device_id"]) != device_id:
            raise ValueError(
                "dSpark entry projection buffers must share one device; "
                f"{anchor} is cuda:{device_id}, "
                f"{name} is cuda:{buffer['device_id']}"
            )
        if int(buffer["bytes"]) < required_bytes:
            raise ValueError(
                f"dSpark entry projection {name} needs {required_bytes} bytes, "
                f"got {buffer['bytes']}"
            )
    return device_id


def prepare_deepseek_v4_dspark_prompt_prime(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    _run_deepseek_v4_dspark_prompt_prime(ctx, prepare_only=True, **kwargs)


def capture_deepseek_v4_dspark_prompt_prime(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    _run_deepseek_v4_dspark_prompt_prime(ctx, prepare_only=False, **kwargs)


def _run_deepseek_v4_dspark_prompt_prime(
    ctx: dict[str, Any], *, prepare_only: bool, **kwargs: Any
) -> None:
    import torch
    from ds4rt_reference.b12x_spark_capture import _raw_tensor
    from b12x.attention import dsv4_producer

    variant = str(kwargs["variant"])
    block_index = int(kwargs["block_index"])
    rows = int(kwargs["rows"])
    max_rows = int(kwargs["max_rows"])
    max_batch = int(kwargs.get("max_batch", 16))
    cache_format = str(kwargs.get("cache_format", "fp8"))
    contract = plan_deepseek_v4_dspark(
        variant=variant,
        max_batch=max_batch,
        max_main_rows=max_rows,
        cache_format=cache_format,
    )
    _dspark_block(contract, block_index)
    geometry = contract.geometry
    if not 1 <= rows <= max_rows:
        raise ValueError(
            f"dSpark prompt-prime rows must be in [1, {max_rows}], got {rows}"
        )
    buffers = ctx["buffers"]
    required = {
        "projected_target_main": rows * geometry.hidden * 2,
        "positions": rows * 4,
        "main_slots": rows * 4,
        "cos_sin_cache": rows * DS4_DSPARK_HEAD_DIM // 8 * 4,
        "main_kv_cache": max_batch * contract.decode_attention.cache_page_bytes,
        "producer_scratch": contract.target_main_kv_producer_scratch.total_bytes,
        "wkv_weight": DS4_DSPARK_HEAD_DIM * geometry.hidden,
        "wkv_scale": (DS4_DSPARK_HEAD_DIM // 128)
        * _ceil_div(geometry.hidden, 128),
        "kv_norm_weight": DS4_DSPARK_HEAD_DIM * 2,
    }
    device_id = _validate_device_buffers(
        buffers, required, anchor="projected_target_main"
    )
    stream = torch.cuda.ExternalStream(int(ctx["cuda_stream"]), device=device_id)
    state_key = (
        device_id,
        int(buffers["wkv_weight"]["ptr"]),
        int(buffers["wkv_scale"]["ptr"]),
        int(buffers["kv_norm_weight"]["ptr"]),
        max_rows,
        *map(ord, contract.decode_attention.cache_format),
    )
    prepared_key = state_key + (rows,)
    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        state = _PROMPT_PRIME_STATE.get(state_key)
        if state is None:
            wkv = _raw_tensor(
                buffers["wkv_weight"],
                (DS4_DSPARK_HEAD_DIM, geometry.hidden),
                torch.float8_e4m3fn,
                1,
                name="dSpark prompt wkv weight",
            )
            wkv_scale = _raw_tensor(
                buffers["wkv_scale"],
                (DS4_DSPARK_HEAD_DIM // 128, _ceil_div(geometry.hidden, 128)),
                torch.float8_e8m0fnu,
                1,
                name="dSpark prompt wkv scale",
            )
            kv_norm = _raw_tensor(
                buffers["kv_norm_weight"],
                (DS4_DSPARK_HEAD_DIM,),
                torch.bfloat16,
                2,
                name="dSpark prompt kv norm",
            )
            weights = dsv4_producer.pack_kv_weights(wkv, wkv_scale, kv_norm)
            producer_plan = dsv4_producer.plan_kv(
                dsv4_producer.Caps(
                    device=torch.device("cuda", device_id),
                    max_tokens=max_rows,
                    hidden=geometry.hidden,
                    q_lora_rank=(1_024 if variant == "flash" else 1_536),
                    heads=geometry.attention_heads,
                    head_dim=DS4_DSPARK_HEAD_DIM,
                    nope_dim=448,
                    rope_dim=64,
                    page_size=DS4_DSPARK_CACHE_PAGE_TOKENS,
                    dtype=torch.bfloat16,
                    cache_format=contract.decode_attention.cache_format,
                )
            )
            state = (producer_plan, weights)
            _PROMPT_PRIME_STATE[state_key] = state
        producer_plan, weights = state
        binding = dsv4_producer.bind_kv(
            producer_plan,
            scratch=_raw_tensor(
                buffers["producer_scratch"],
                (contract.target_main_kv_producer_scratch.total_bytes,),
                torch.uint8,
                1,
                name="dSpark prompt producer scratch",
            ),
            hidden_states=_raw_tensor(
                buffers["projected_target_main"],
                (rows, geometry.hidden),
                torch.bfloat16,
                2,
                name="dSpark projected target main",
            ),
            positions=_raw_tensor(
                buffers["positions"],
                (rows,),
                torch.int32,
                4,
                name="dSpark prompt positions",
            ),
            main_slots=_raw_tensor(
                buffers["main_slots"],
                (rows,),
                torch.int32,
                4,
                name="dSpark prompt main slots",
            ),
            cos_sin_cache=_raw_tensor(
                buffers["cos_sin_cache"],
                (rows, 64),
                torch.float32,
                4,
                name="dSpark prompt local cos/sin cache",
            ),
            main_kv_cache=_raw_tensor(
                buffers["main_kv_cache"],
                (max_batch, contract.decode_attention.cache_page_bytes),
                torch.uint8,
                1,
                name="dSpark prompt main KV cache",
            ),
            weights=weights,
            eps=float(kwargs.get("producer_eps", 1.0e-6)),
            expected_m=rows,
        )
        if not prepare_only and prepared_key not in _PROMPT_PRIME_PREPARED:
            raise RuntimeError(
                "dSpark prompt-prime capture was not prepared for its row bucket"
            )
        dsv4_producer.run_kv(binding=binding)
        if prepare_only:
            _PROMPT_PRIME_PREPARED.add(prepared_key)


def prepare_deepseek_v4_dspark_proposal_entry(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    _run_deepseek_v4_dspark_proposal_entry(ctx, prepare_only=True, **kwargs)


def capture_deepseek_v4_dspark_proposal_entry(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    _run_deepseek_v4_dspark_proposal_entry(ctx, prepare_only=False, **kwargs)


def _run_deepseek_v4_dspark_proposal_entry(
    ctx: dict[str, Any], *, prepare_only: bool, **kwargs: Any
) -> None:
    import torch
    from ds4rt_reference.b12x_spark_capture import _raw_tensor

    variant = str(kwargs["variant"])
    max_batch = int(kwargs["max_batch"])
    max_main_rows = int(kwargs["max_main_rows"])
    cache_format = str(kwargs.get("cache_format", "fp8"))
    contract = plan_deepseek_v4_dspark(
        variant=variant,
        max_batch=max_batch,
        max_main_rows=max_main_rows,
        cache_format=cache_format,
    )
    geometry = contract.geometry
    rows = geometry.proposal_tokens
    hidden = geometry.hidden
    hc_mixes = (2 + DS4_HC_MULT) * DS4_HC_MULT
    workspace_bytes = contract.arena.region("reused_sparse_block_workspace").nbytes
    buffers = ctx["buffers"]
    required = {
        "scratch": workspace_bytes,
        "residual_input": rows * hidden * 2,
        "residual_output": rows * DS4_HC_MULT * hidden * 2,
        "normalized_output": rows * hidden * 2,
        "post_output": rows * DS4_HC_MULT * 4,
        "comb_output": rows * DS4_HC_MULT * DS4_HC_MULT * 4,
        "hc_fn": hc_mixes * hidden * 4,
        "hc_scale": 3 * 4,
        "hc_base": hc_mixes * 4,
        "norm_weight": hidden * 2,
    }
    device_id = _validate_device_buffers(
        buffers, required, anchor="residual_input"
    )
    stream = torch.cuda.ExternalStream(int(ctx["cuda_stream"]), device=device_id)
    prepared_key = (
        device_id,
        *(int(buffers[name]["ptr"]) for name in required),
        hidden,
    )
    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        binding = bind_deepseek_v4_dspark_proposal_entry(
            contract,
            scratch=_raw_tensor(
                buffers["scratch"],
                (workspace_bytes,),
                torch.uint8,
                1,
                name="dSpark proposal-entry mHC scratch",
            ),
            residual_input=_raw_tensor(
                buffers["residual_input"],
                (rows, hidden),
                torch.bfloat16,
                2,
                name="dSpark proposal-entry residual input",
            ),
            residual_output=_raw_tensor(
                buffers["residual_output"],
                (rows, DS4_HC_MULT, hidden),
                torch.bfloat16,
                2,
                name="dSpark proposal-entry residual output",
            ),
            normalized_output=_raw_tensor(
                buffers["normalized_output"],
                (rows, hidden),
                torch.bfloat16,
                2,
                name="dSpark proposal-entry normalized output",
            ),
            post_output=_raw_tensor(
                buffers["post_output"],
                (rows, DS4_HC_MULT),
                torch.float32,
                4,
                name="dSpark proposal-entry post mix",
            ),
            comb_output=_raw_tensor(
                buffers["comb_output"],
                (rows, DS4_HC_MULT, DS4_HC_MULT),
                torch.float32,
                4,
                name="dSpark proposal-entry combination mix",
            ),
            hc_fn=_raw_tensor(
                buffers["hc_fn"],
                (hc_mixes, hidden),
                torch.float32,
                4,
                name="dSpark block-0 lane-summed attention HC function",
            ),
            hc_scale=_raw_tensor(
                buffers["hc_scale"],
                (3,),
                torch.float32,
                4,
                name="dSpark block-0 attention HC scale",
            ),
            hc_base=_raw_tensor(
                buffers["hc_base"],
                (hc_mixes,),
                torch.float32,
                4,
                name="dSpark block-0 attention HC base",
            ),
            norm_weight=_raw_tensor(
                buffers["norm_weight"],
                (hidden,),
                torch.bfloat16,
                2,
                name="dSpark block-0 attention norm",
            ),
            rms_eps=float(kwargs.get("rms_eps", 1.0e-6)),
            hc_eps=float(kwargs.get("hc_eps", 1.0e-6)),
            sinkhorn_iters=int(kwargs.get("sinkhorn_iters", 20)),
            norm_eps=float(kwargs.get("norm_eps", 1.0e-6)),
        )
        if not prepare_only and prepared_key not in _PROPOSAL_ENTRY_PREPARED:
            raise RuntimeError(
                "dSpark proposal-entry capture was not prepared for its request slot"
            )
        run_deepseek_v4_dspark_proposal_entry(binding)
        if prepare_only:
            _PROPOSAL_ENTRY_PREPARED.add(prepared_key)


def prepare_deepseek_v4_dspark_block_attention(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    _run_deepseek_v4_dspark_block_attention(ctx, prepare_only=True, **kwargs)


def capture_deepseek_v4_dspark_block_attention(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    _run_deepseek_v4_dspark_block_attention(ctx, prepare_only=False, **kwargs)


def _run_deepseek_v4_dspark_block_attention(
    ctx: dict[str, Any], *, prepare_only: bool, **kwargs: Any
) -> None:
    """Run one dSpark attention half-block through normalized TP4 dispatch input.

    ``mtp.<block>`` is only the checkpoint prefix. This graph is the ordinary
    dSpark transformer half-block and deliberately stops before the strict-TP4
    expert transport barrier.
    """

    import torch
    from ds4rt_reference.b12x_spark_capture import _raw_tensor
    from b12x.attention import dsv4_producer
    from b12x.gemm import wo_projection

    variant = str(kwargs["variant"])
    block_index = int(kwargs["block_index"])
    max_batch = int(kwargs["max_batch"])
    max_main_rows = int(kwargs["max_main_rows"])
    cache_format = str(kwargs.get("cache_format", "fp8"))
    contract = plan_deepseek_v4_dspark(
        variant=variant,
        max_batch=max_batch,
        max_main_rows=max_main_rows,
        cache_format=cache_format,
    )
    _dspark_block(contract, block_index)
    geometry = contract.geometry
    rows = geometry.proposal_tokens
    hidden = geometry.hidden
    heads = geometry.attention_heads
    q_rank = 1_024 if variant == "flash" else 1_536
    output_groups = 8 if variant == "flash" else 16
    output_rank = 1_024
    output_group_width = heads // output_groups * geometry.head_dim
    query_width = heads * geometry.head_dim
    output_projected_width = output_groups * output_rank
    workspace_bytes = contract.arena.region("reused_sparse_block_workspace").nbytes
    buffers = ctx["buffers"]
    required = {
        "workspace": workspace_bytes,
        "hidden_states": rows * hidden * 2,
        "positions": rows * 4,
        "main_slots": rows * 4,
        "cos_sin_cache": rows * 64 * 4,
        "main_kv_cache": max_batch * contract.decode_attention.cache_page_bytes,
        "selected_indices": rows * DS4_DSPARK_SELECTION_WIDTH * 4,
        "selected_lengths": rows * 4,
        "residual": rows * DS4_HC_MULT * hidden * 2,
        "prev_post": rows * DS4_HC_MULT * 4,
        "prev_comb": rows * DS4_HC_MULT * DS4_HC_MULT * 4,
        "residual_out": rows * DS4_HC_MULT * hidden * 2,
        "post_out": rows * DS4_HC_MULT * 4,
        "comb_out": rows * DS4_HC_MULT * DS4_HC_MULT * 4,
        "wq_a_weight": q_rank * hidden,
        "wq_a_scale": _ceil_div(q_rank, 128) * _ceil_div(hidden, 128),
        "wq_b_weight": query_width * q_rank,
        "wq_b_scale": _ceil_div(query_width, 128) * _ceil_div(q_rank, 128),
        "wkv_weight": geometry.head_dim * hidden,
        "wkv_scale": _ceil_div(geometry.head_dim, 128)
        * _ceil_div(hidden, 128),
        "q_norm_weight": q_rank * 2,
        "kv_norm_weight": geometry.head_dim * 2,
        "wo_a_weight": output_projected_width * output_group_width,
        "wo_a_scale": _ceil_div(output_projected_width, 128)
        * _ceil_div(output_group_width, 128),
        "wo_b_weight": hidden * output_projected_width,
        "wo_b_scale": _ceil_div(hidden, 128)
        * _ceil_div(output_projected_width, 128),
        "attn_sink": heads * 4,
        "hc_fn": (2 + DS4_HC_MULT)
        * DS4_HC_MULT
        * DS4_HC_MULT
        * hidden
        * 4,
        "hc_scale": 3 * 4,
        "hc_base": (2 + DS4_HC_MULT) * DS4_HC_MULT * 4,
        "norm_weight": hidden * 2,
    }
    device_id = _validate_device_buffers(buffers, required, anchor="hidden_states")
    stream = torch.cuda.ExternalStream(int(ctx["cuda_stream"]), device=device_id)
    state_key = (
        device_id,
        block_index,
        *map(ord, contract.decode_attention.cache_format),
        *(int(buffers[name]["ptr"]) for name in (
            "wq_a_weight",
            "wq_a_scale",
            "wq_b_weight",
            "wq_b_scale",
            "wkv_weight",
            "wkv_scale",
            "q_norm_weight",
            "kv_norm_weight",
            "wo_a_weight",
            "wo_a_scale",
            "wo_b_weight",
            "wo_b_scale",
        )),
    )
    prepared_key = (
        device_id,
        block_index,
        max_batch,
        *map(ord, contract.decode_attention.cache_format),
        *(int(buffers[name]["ptr"]) for name in required),
    )
    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        state = _BLOCK_ATTENTION_STATE.get(state_key)
        if state is None:
            fp8 = torch.float8_e4m3fn
            e8m0 = torch.float8_e8m0fnu
            producer_weights = dsv4_producer.pack_weights(
                _raw_tensor(
                    buffers["wq_a_weight"],
                    (q_rank, hidden),
                    fp8,
                    1,
                    name="dSpark attention wq_a weight",
                ),
                _raw_tensor(
                    buffers["wq_a_scale"],
                    (_ceil_div(q_rank, 128), _ceil_div(hidden, 128)),
                    e8m0,
                    1,
                    name="dSpark attention wq_a scale",
                ),
                _raw_tensor(
                    buffers["wq_b_weight"],
                    (query_width, q_rank),
                    fp8,
                    1,
                    name="dSpark attention wq_b weight",
                ),
                _raw_tensor(
                    buffers["wq_b_scale"],
                    (_ceil_div(query_width, 128), _ceil_div(q_rank, 128)),
                    e8m0,
                    1,
                    name="dSpark attention wq_b scale",
                ),
                _raw_tensor(
                    buffers["wkv_weight"],
                    (geometry.head_dim, hidden),
                    fp8,
                    1,
                    name="dSpark attention wkv weight",
                ),
                _raw_tensor(
                    buffers["wkv_scale"],
                    (
                        _ceil_div(geometry.head_dim, 128),
                        _ceil_div(hidden, 128),
                    ),
                    e8m0,
                    1,
                    name="dSpark attention wkv scale",
                ),
                _raw_tensor(
                    buffers["q_norm_weight"],
                    (q_rank,),
                    torch.bfloat16,
                    2,
                    name="dSpark attention q norm",
                ),
                _raw_tensor(
                    buffers["kv_norm_weight"],
                    (geometry.head_dim,),
                    torch.bfloat16,
                    2,
                    name="dSpark attention kv norm",
                ),
            )
            output_weights = wo_projection.pack_weights(
                _raw_tensor(
                    buffers["wo_a_weight"],
                    (output_projected_width, output_group_width),
                    fp8,
                    1,
                    name="dSpark attention wo_a weight",
                ),
                _raw_tensor(
                    buffers["wo_a_scale"],
                    (
                        _ceil_div(output_projected_width, 128),
                        _ceil_div(output_group_width, 128),
                    ),
                    e8m0,
                    1,
                    name="dSpark attention wo_a scale",
                ),
                _raw_tensor(
                    buffers["wo_b_weight"],
                    (hidden, output_projected_width),
                    fp8,
                    1,
                    name="dSpark attention wo_b weight",
                ),
                _raw_tensor(
                    buffers["wo_b_scale"],
                    (
                        _ceil_div(hidden, 128),
                        _ceil_div(output_projected_width, 128),
                    ),
                    e8m0,
                    1,
                    name="dSpark attention wo_b scale",
                ),
                groups=output_groups,
                group_width=output_group_width,
                rank=output_rank,
                hidden=hidden,
            )
            state = (producer_weights, output_weights)
            _BLOCK_ATTENTION_STATE[state_key] = state
        producer_weights, output_weights = state
        binding = bind_deepseek_v4_sliding_attention_layer(
            contract.blocks[block_index].expert_handoff.attention,
            arena=_raw_tensor(
                buffers["workspace"],
                (workspace_bytes,),
                torch.uint8,
                1,
                name="dSpark reused sparse-block workspace",
            ),
            hidden_states=_raw_tensor(
                buffers["hidden_states"],
                (rows, hidden),
                torch.bfloat16,
                2,
                name="dSpark attention normalized hidden",
            ),
            positions=_raw_tensor(
                buffers["positions"],
                (rows,),
                torch.int32,
                4,
                name="dSpark proposal positions",
            ),
            main_slots=_raw_tensor(
                buffers["main_slots"],
                (rows,),
                torch.int32,
                4,
                name="dSpark proposal KV slots",
            ),
            cos_sin_cache=_raw_tensor(
                buffers["cos_sin_cache"],
                (rows, 64),
                torch.float32,
                4,
                name="dSpark proposal local cos/sin",
            ),
            main_kv_cache=_raw_tensor(
                buffers["main_kv_cache"],
                (max_batch, contract.decode_attention.cache_page_bytes),
                torch.uint8,
                1,
                name="dSpark block request-page KV cache",
            ),
            swa_indices=_raw_tensor(
                buffers["selected_indices"],
                (rows, DS4_DSPARK_SELECTION_WIDTH),
                torch.int32,
                4,
                name="dSpark dual-source physical selection",
            ),
            swa_lengths=_raw_tensor(
                buffers["selected_lengths"],
                (rows,),
                torch.int32,
                4,
                name="dSpark dual-source selection lengths",
            ),
            producer_weights=producer_weights,
            output_weights=output_weights,
            residual=_raw_tensor(
                buffers["residual"],
                (rows, DS4_HC_MULT, hidden),
                torch.bfloat16,
                2,
                name="dSpark attention HC residual",
            ),
            prev_post=_raw_tensor(
                buffers["prev_post"],
                (rows, DS4_HC_MULT),
                torch.float32,
                4,
                name="dSpark attention previous post mix",
            ),
            prev_comb=_raw_tensor(
                buffers["prev_comb"],
                (rows, DS4_HC_MULT, DS4_HC_MULT),
                torch.float32,
                4,
                name="dSpark attention previous combination mix",
            ),
            fn=_raw_tensor(
                buffers["hc_fn"],
                ((2 + DS4_HC_MULT) * DS4_HC_MULT, DS4_HC_MULT * hidden),
                torch.float32,
                4,
                name="dSpark FFN HC function",
            ),
            hc_scale=_raw_tensor(
                buffers["hc_scale"],
                (3,),
                torch.float32,
                4,
                name="dSpark FFN HC scale",
            ),
            hc_base=_raw_tensor(
                buffers["hc_base"],
                ((2 + DS4_HC_MULT) * DS4_HC_MULT,),
                torch.float32,
                4,
                name="dSpark FFN HC base",
            ),
            norm_weight=_raw_tensor(
                buffers["norm_weight"],
                (hidden,),
                torch.bfloat16,
                2,
                name="dSpark FFN norm",
            ),
            residual_out=_raw_tensor(
                buffers["residual_out"],
                (rows, DS4_HC_MULT, hidden),
                torch.bfloat16,
                2,
                name="dSpark post-attention residual",
            ),
            y_out=_raw_tensor(
                buffers["hidden_states"],
                (rows, hidden),
                torch.bfloat16,
                2,
                name="dSpark normalized TP4 dispatch hidden",
            ),
            post_out=_raw_tensor(
                buffers["post_out"],
                (rows, DS4_HC_MULT),
                torch.float32,
                4,
                name="dSpark FFN post mix",
            ),
            comb_out=_raw_tensor(
                buffers["comb_out"],
                (rows, DS4_HC_MULT, DS4_HC_MULT),
                torch.float32,
                4,
                name="dSpark FFN combination mix",
            ),
            attn_sink=_raw_tensor(
                buffers["attn_sink"],
                (heads,),
                torch.float32,
                4,
                name="dSpark attention sink",
            ),
            producer_eps=float(kwargs.get("producer_eps", 1.0e-6)),
            rms_eps=float(kwargs.get("rms_eps", 1.0e-6)),
            hc_eps=float(kwargs.get("hc_eps", 1.0e-6)),
            sinkhorn_iters=int(kwargs.get("sinkhorn_iters", 20)),
            norm_eps=float(kwargs.get("norm_eps", 1.0e-6)),
        )
        if not prepare_only and prepared_key not in _BLOCK_ATTENTION_PREPARED:
            raise RuntimeError(
                "dSpark block-attention capture was not prepared for its fixed slot"
            )
        run_deepseek_v4_sliding_attention_layer(binding)
        if prepare_only:
            _BLOCK_ATTENTION_PREPARED.add(prepared_key)


def prepare_deepseek_v4_dspark_block_post_dispatch(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    _run_deepseek_v4_dspark_block_post_dispatch(
        ctx, prepare_only=True, **kwargs
    )


def capture_deepseek_v4_dspark_block_post_dispatch(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    _run_deepseek_v4_dspark_block_post_dispatch(
        ctx, prepare_only=False, **kwargs
    )


def _run_deepseek_v4_dspark_block_post_dispatch(
    ctx: dict[str, Any], *, prepare_only: bool, **kwargs: Any
) -> None:
    """Cross one dSpark expert barrier into the next attention pre-mix.

    The coordinator has already reduced the shared expert and four strict-TP4
    rank partials into ``ffn_delta``. This capture performs the current block's
    FFN HC-post and the next block's attention HC-pre/RMSNorm as one stable
    GLMRT-style graph segment.
    """

    import torch
    from ds4rt_reference.b12x_spark_capture import _raw_tensor
    from b12x.norm import mhc

    variant = str(kwargs["variant"])
    block_index = int(kwargs["block_index"])
    max_batch = int(kwargs["max_batch"])
    max_main_rows = int(kwargs["max_main_rows"])
    contract = plan_deepseek_v4_dspark(
        variant=variant,
        max_batch=max_batch,
        max_main_rows=max_main_rows,
    )
    if not 0 <= block_index < contract.geometry.blocks - 1:
        raise ValueError(
            "dSpark inter-block post-dispatch capture requires block 0 or 1, "
            f"got {block_index}"
        )
    geometry = contract.geometry
    rows = geometry.proposal_tokens
    hidden = geometry.hidden
    hc_mixes = (2 + DS4_HC_MULT) * DS4_HC_MULT
    workspace_bytes = contract.arena.region(
        "reused_sparse_block_workspace"
    ).nbytes
    buffers = ctx["buffers"]
    required = {
        "workspace": workspace_bytes,
        "ffn_delta": rows * hidden * 2,
        "residual": rows * DS4_HC_MULT * hidden * 2,
        "prev_post": rows * DS4_HC_MULT * 4,
        "prev_comb": rows * DS4_HC_MULT * DS4_HC_MULT * 4,
        "residual_out": rows * DS4_HC_MULT * hidden * 2,
        "hidden_states": rows * hidden * 2,
        "post_out": rows * DS4_HC_MULT * 4,
        "comb_out": rows * DS4_HC_MULT * DS4_HC_MULT * 4,
        "next_hc_fn": hc_mixes * DS4_HC_MULT * hidden * 4,
        "next_hc_scale": 3 * 4,
        "next_hc_base": hc_mixes * 4,
        "next_norm_weight": hidden * 2,
    }
    device_id = _validate_device_buffers(
        buffers, required, anchor="ffn_delta"
    )
    stream = torch.cuda.ExternalStream(int(ctx["cuda_stream"]), device=device_id)
    prepared_key = (
        device_id,
        block_index,
        max_batch,
        *(int(buffers[name]["ptr"]) for name in required),
    )
    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        plan = mhc.plan(
            mhc.Caps(
                device=torch.device("cuda", device_id),
                max_tokens=contract.blocks[
                    block_index
                ].expert_handoff.max_rows,
                hidden_size=hidden,
                split_k=contract.blocks[
                    block_index
                ].expert_handoff.ffn_mhc.geometry.split_k,
                dtype=torch.bfloat16,
            )
        )
        binding = mhc.bind(
            plan,
            scratch=_raw_tensor(
                buffers["workspace"],
                (workspace_bytes,),
                torch.uint8,
                1,
                name="dSpark inter-block mHC scratch",
            ),
            tokens=rows,
            expected_m=contract.blocks[block_index].expert_handoff.max_rows,
            y=_raw_tensor(
                buffers["hidden_states"],
                (rows, hidden),
                torch.bfloat16,
                2,
                name="dSpark next-block normalized attention input",
            ),
            post=_raw_tensor(
                buffers["post_out"],
                (rows, DS4_HC_MULT),
                torch.float32,
                4,
                name="dSpark next-block attention post mix",
            ),
            comb=_raw_tensor(
                buffers["comb_out"],
                (rows, DS4_HC_MULT, DS4_HC_MULT),
                torch.float32,
                4,
                name="dSpark next-block attention combination mix",
            ),
            out=_raw_tensor(
                buffers["residual_out"],
                (rows, DS4_HC_MULT, hidden),
                torch.bfloat16,
                2,
                name="dSpark next-block attention residual",
            ),
        )
        if not prepare_only and prepared_key not in _BLOCK_POST_DISPATCH_PREPARED:
            raise RuntimeError(
                "dSpark block post-dispatch capture was not prepared for its fixed slot"
            )
        mhc.run_post_pre(
            _raw_tensor(
                buffers["ffn_delta"],
                (rows, hidden),
                torch.bfloat16,
                2,
                name="dSpark coordinator-reduced FFN delta",
            ),
            _raw_tensor(
                buffers["residual"],
                (rows, DS4_HC_MULT, hidden),
                torch.bfloat16,
                2,
                name="dSpark post-attention FFN residual",
            ),
            _raw_tensor(
                buffers["prev_post"],
                (rows, DS4_HC_MULT),
                torch.float32,
                4,
                name="dSpark FFN previous post mix",
            ),
            _raw_tensor(
                buffers["prev_comb"],
                (rows, DS4_HC_MULT, DS4_HC_MULT),
                torch.float32,
                4,
                name="dSpark FFN previous combination mix",
            ),
            _raw_tensor(
                buffers["next_hc_fn"],
                (hc_mixes, DS4_HC_MULT * hidden),
                torch.float32,
                4,
                name="dSpark next attention HC function",
            ),
            _raw_tensor(
                buffers["next_hc_scale"],
                (3,),
                torch.float32,
                4,
                name="dSpark next attention HC scale",
            ),
            _raw_tensor(
                buffers["next_hc_base"],
                (hc_mixes,),
                torch.float32,
                4,
                name="dSpark next attention HC base",
            ),
            rms_eps=float(kwargs.get("rms_eps", 1.0e-6)),
            hc_eps=float(kwargs.get("hc_eps", 1.0e-6)),
            sinkhorn_iters=int(kwargs.get("sinkhorn_iters", 20)),
            norm_weight=_raw_tensor(
                buffers["next_norm_weight"],
                (hidden,),
                torch.bfloat16,
                2,
                name="dSpark next attention norm",
            ),
            norm_eps=float(kwargs.get("norm_eps", 1.0e-6)),
            binding=binding,
        )
        if prepare_only:
            _BLOCK_POST_DISPATCH_PREPARED.add(prepared_key)


def prepare_deepseek_v4_dspark_terminal_collapse(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    _run_deepseek_v4_dspark_terminal_collapse(
        ctx, prepare_only=True, **kwargs
    )


def capture_deepseek_v4_dspark_terminal_collapse(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    _run_deepseek_v4_dspark_terminal_collapse(
        ctx, prepare_only=False, **kwargs
    )


def _run_deepseek_v4_dspark_terminal_collapse(
    ctx: dict[str, Any], *, prepare_only: bool, **kwargs: Any
) -> None:
    """Finish block two and retain both inputs required by dSpark's heads.

    The final FFN HC-post first produces the completed four-lane dSpark state.
    ``run_head`` then writes the unnormalized collapse for the confidence head
    and its RMS-normalized sibling for the shared LM head in one launch.
    """

    import torch
    from ds4rt_reference.b12x_spark_capture import _raw_tensor
    from b12x.norm import mhc

    variant = str(kwargs["variant"])
    max_batch = int(kwargs["max_batch"])
    max_main_rows = int(kwargs["max_main_rows"])
    contract = plan_deepseek_v4_dspark(
        variant=variant,
        max_batch=max_batch,
        max_main_rows=max_main_rows,
    )
    geometry = contract.geometry
    rows = geometry.proposal_tokens
    hidden = geometry.hidden
    workspace_bytes = contract.arena.region(
        "reused_sparse_block_workspace"
    ).nbytes
    buffers = ctx["buffers"]
    required = {
        "workspace": workspace_bytes,
        "ffn_delta": rows * hidden * 2,
        "residual": rows * DS4_HC_MULT * hidden * 2,
        "prev_post": rows * DS4_HC_MULT * 4,
        "prev_comb": rows * DS4_HC_MULT * DS4_HC_MULT * 4,
        "terminal_residual": rows * DS4_HC_MULT * hidden * 2,
        "collapsed_hidden": rows * hidden * 2,
        "normalized_hidden": rows * hidden * 2,
        "hc_head_fn": DS4_HC_MULT * DS4_HC_MULT * hidden * 4,
        "hc_head_scale": 4,
        "hc_head_base": DS4_HC_MULT * 4,
        "norm_weight": hidden * 2,
    }
    device_id = _validate_device_buffers(
        buffers, required, anchor="ffn_delta"
    )
    if int(buffers["collapsed_hidden"]["ptr"]) == int(
        buffers["normalized_hidden"]["ptr"]
    ):
        raise ValueError(
            "dSpark terminal raw and normalized collapsed hidden must not alias"
        )
    stream = torch.cuda.ExternalStream(int(ctx["cuda_stream"]), device=device_id)
    prepared_key = (
        device_id,
        max_batch,
        *(int(buffers[name]["ptr"]) for name in required),
    )
    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        scratch = _raw_tensor(
            buffers["workspace"],
            (workspace_bytes,),
            torch.uint8,
            1,
            name="dSpark terminal mHC scratch",
        )
        ffn_delta = _raw_tensor(
            buffers["ffn_delta"],
            (rows, hidden),
            torch.bfloat16,
            2,
            name="dSpark terminal coordinator-reduced FFN delta",
        )
        terminal_residual = _raw_tensor(
            buffers["terminal_residual"],
            (rows, DS4_HC_MULT, hidden),
            torch.bfloat16,
            2,
            name="dSpark completed four-lane terminal residual",
        )
        normalized_hidden = _raw_tensor(
            buffers["normalized_hidden"],
            (rows, hidden),
            torch.bfloat16,
            2,
            name="dSpark slot-resident normalized shared-head hidden",
        )
        binding = mhc.bind(
            mhc.plan(
                mhc.Caps(
                    device=torch.device("cuda", device_id),
                    max_tokens=contract.blocks[2].expert_handoff.max_rows,
                    hidden_size=hidden,
                    split_k=contract.blocks[
                        2
                    ].expert_handoff.ffn_mhc.geometry.split_k,
                    dtype=torch.bfloat16,
                )
            ),
            scratch=scratch,
            tokens=rows,
            expected_m=contract.blocks[2].expert_handoff.max_rows,
            y=normalized_hidden,
        )
        if not prepare_only and prepared_key not in _TERMINAL_COLLAPSE_PREPARED:
            raise RuntimeError(
                "dSpark terminal collapse capture was not prepared for its fixed slot"
            )
        mhc.run_post(
            ffn_delta,
            _raw_tensor(
                buffers["residual"],
                (rows, DS4_HC_MULT, hidden),
                torch.bfloat16,
                2,
                name="dSpark block-2 post-attention residual",
            ),
            _raw_tensor(
                buffers["prev_post"],
                (rows, DS4_HC_MULT),
                torch.float32,
                4,
                name="dSpark block-2 FFN post mix",
            ),
            _raw_tensor(
                buffers["prev_comb"],
                (rows, DS4_HC_MULT, DS4_HC_MULT),
                torch.float32,
                4,
                name="dSpark block-2 FFN combination mix",
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
                name="dSpark terminal HC function",
            ),
            _raw_tensor(
                buffers["hc_head_scale"],
                (1,),
                torch.float32,
                4,
                name="dSpark terminal HC scale",
            ),
            _raw_tensor(
                buffers["hc_head_base"],
                (DS4_HC_MULT,),
                torch.float32,
                4,
                name="dSpark terminal HC base",
            ),
            _raw_tensor(
                buffers["norm_weight"],
                (hidden,),
                torch.bfloat16,
                2,
                name="dSpark terminal RMS norm",
            ),
            rms_eps=float(kwargs.get("rms_eps", 1.0e-6)),
            hc_eps=float(kwargs.get("hc_eps", 1.0e-6)),
            norm_eps=float(kwargs.get("norm_eps", 1.0e-6)),
            collapsed_out=_raw_tensor(
                buffers["collapsed_hidden"],
                (rows, hidden),
                torch.bfloat16,
                2,
                name="dSpark unnormalized confidence-head hidden",
            ),
            binding=binding,
        )
        if prepare_only:
            _TERMINAL_COLLAPSE_PREPARED.add(prepared_key)


def qualify_deepseek_v4_dspark_contract(**kwargs: Any) -> bool:
    contract = plan_deepseek_v4_dspark(
        variant=str(kwargs["variant"]),
        max_batch=int(kwargs["max_batch"]),
        max_main_rows=int(kwargs["max_main_rows"]),
        cache_format=str(kwargs.get("cache_format", "fp8")),
    )
    geometry = contract.geometry
    validate_sparkinfer = bool(kwargs.get("validate_sparkinfer", True))
    expected_physical = tuple(
        range(geometry.target_layers, geometry.target_layers + DS4_DSPARK_BLOCKS)
    )
    if (
        geometry.blocks != DS4_DSPARK_BLOCKS
        or geometry.proposal_tokens != DS4_DSPARK_PROPOSAL_TOKENS
        or geometry.sliding_window != DS4_DSPARK_SLIDING_WINDOW
        or geometry.head_dim != DS4_DSPARK_HEAD_DIM
        or geometry.noise_token_id != DS4_DSPARK_NOISE_TOKEN_ID
        or geometry.vocab_size != DS4_DSPARK_VOCAB_SIZE
        or geometry.physical_block_ids != expected_physical
        or geometry.storage_prefixes != ("mtp.0", "mtp.1", "mtp.2")
        or contract.serving_allocates
        or not contract.target_taps_are_caller_owned
        or not contract.target_projection_is_shared_by_all_blocks
        or not contract.cache_is_request_local
        or contract.cache_device_id != 0
        or contract.proposal_input_token_pattern
        != ("anchor", "noise", "noise", "noise", "noise")
        or contract.dispatch_barriers_per_decode != DS4_DSPARK_BLOCKS
        or contract.cuda_graph_segments_per_decode != 8
        or not contract.sparse_workspace_reused_across_blocks
        or not contract.entry_scratch_reuses_sparse_workspace
        or contract.entry_activation_block_size
        != DS4_DSPARK_ENTRY_ACTIVATION_BLOCK_SIZE
        or contract.entry_capture_surface
        != (
            "prepare_deepseek_v4_dspark_entry_projection",
            "capture_deepseek_v4_dspark_entry_projection",
        )
        or not contract.target_main_kv_producer_scratch_reused_across_blocks
        or contract.prompt_prime_computes_query
        or contract.prompt_prime_producer_surface
        != ("plan_kv", "bind_kv", "run_kv")
        or contract.prompt_prime_capture_surface
        != (
            "prepare_deepseek_v4_dspark_prompt_prime",
            "capture_deepseek_v4_dspark_prompt_prime",
        )
        or contract.proposal_entry_capture_surface
        != (
            "prepare_deepseek_v4_dspark_proposal_entry",
            "capture_deepseek_v4_dspark_proposal_entry",
        )
        or contract.block_attention_capture_surface
        != (
            "prepare_deepseek_v4_dspark_block_attention",
            "capture_deepseek_v4_dspark_block_attention",
        )
        or contract.block_post_dispatch_capture_surface
        != (
            "prepare_deepseek_v4_dspark_block_post_dispatch",
            "capture_deepseek_v4_dspark_block_post_dispatch",
        )
        or contract.terminal_collapse_capture_surface
        != (
            "prepare_deepseek_v4_dspark_terminal_collapse",
            "capture_deepseek_v4_dspark_terminal_collapse",
        )
        or contract.terminal_batch_axis
        != "active-requests-times-five-proposal-rows"
        or contract.terminal_serial_dependency_axis
        != "five-markov-positions-only"
        or not contract.terminal_preserves_unnormalized_hidden
        or not contract.terminal_jointly_issues_active_requests
        or not contract.proposal_scratch_reuses_markov_storage
        or contract.decode_attention.target_ring_slots
        != DS4_DSPARK_SLIDING_WINDOW
        or contract.decode_attention.proposal_slots != (128, 129, 130, 131, 132)
        or contract.decode_attention.selection_width != DS4_DSPARK_SELECTION_WIDTH
        or contract.decode_attention.cache_page_tokens
        != DS4_DSPARK_CACHE_PAGE_TOKENS
        or contract.decode_attention.cache_page_bytes
        != deepseek_v4_kv_format(contract.decode_attention.cache_format).main_page_bytes
        or not contract.decode_attention.all_proposal_kv_visible_to_every_query
        or contract.decode_attention.proposal_attention_is_causal
        or contract.expert_tensor_parallel != DS4_EXPERT_TP
        or contract.expert_parallel
        or contract.recurrent_mtp_state
        or DeepseekV4DsparkPromptPrimeBlockBinding.serving_allocates
        or not DeepseekV4DsparkPromptPrimeBlockBinding.cuda_graph_safe
        or DeepseekV4DsparkPromptPrimeBlockBinding.computes_query
        or DeepseekV4DsparkProposalEntryBinding.serving_allocates
        or not DeepseekV4DsparkProposalEntryBinding.cuda_graph_safe
        or DeepseekV4DsparkProposalEntryBinding.recurrent_mtp_state
        or DeepseekV4DsparkBlockBinding.serving_allocates
        or not DeepseekV4DsparkBlockBinding.cuda_graph_safe
        or DeepseekV4DsparkBlockBinding.cuda_graph_segments != 2
        or not DeepseekV4DsparkBlockBinding.dispatch_barrier_between_graphs
        or DeepseekV4DsparkBlockBinding.expert_tensor_parallel != DS4_EXPERT_TP
        or DeepseekV4DsparkBlockBinding.expert_parallel
    ):
        raise RuntimeError("integrated DeepSeek dSpark lifecycle drifted")
    if tuple(block.physical_layer_id for block in contract.blocks) != expected_physical:
        raise RuntimeError("dSpark physical block IDs drifted")
    if validate_sparkinfer:
        if not qualify_deepseek_v4_sparse_block_contract(
            variant=geometry.variant,
            mode="decode",
            compression=0,
            max_rows=contract.max_batch * geometry.proposal_tokens,
            source_pages=contract.max_batch,
            max_positions=1_048_576,
            swa_width=DS4_DSPARK_SELECTION_WIDTH,
            cache_format=contract.decode_attention.cache_format,
        ):
            raise RuntimeError("integrated dSpark sparse-block graph failed qualification")
        _qualify_dspark_entry(contract)
        _qualify_target_main_kv_producer(contract)
        _qualify_dspark_decode_attention(contract)
    for block_index, block in enumerate(contract.blocks):
        handoff = block.expert_handoff
        if (
            block.block_index != block_index
            or block.target_tap_id != geometry.target_taps[block_index]
            or block.storage_prefix != f"mtp.{block_index}"
            or block.checkpoint_namespace_is_algorithm_name
            or block.attention_semantics != "dual-source-dspark-sliding-attention"
            or not block.uses_target_main_kv
            or not block.proposal_queries_include_proposal_kv
            or handoff.expert_tensor_parallel != DS4_EXPERT_TP
            or handoff.expert_parallel
            or handoff.attention.swa_width != DS4_DSPARK_SELECTION_WIDTH
            or contract.decode_attention.workspace_growth_bytes != 0
            or not handoff.dispatch_barrier_between_graphs
            or not handoff.one_route_buffer_fans_out_to_all_ranks
        ):
            raise RuntimeError(f"dSpark block {block_index} contract drifted")
    previous_end = 0
    for region in contract.arena.regions:
        if region.offset % DS4_DSPARK_ARENA_ALIGNMENT:
            raise RuntimeError(f"dSpark arena region {region.name} lost alignment")
        if region.offset < previous_end or region.nbytes <= 0:
            raise RuntimeError(f"dSpark arena region {region.name} overlaps or is empty")
        previous_end = region.offset + region.nbytes
    if _align_up(previous_end) != contract.arena.total_bytes:
        raise RuntimeError("dSpark arena total drifted")
    expected_persistent_kv_bytes = (
        geometry.blocks
        * contract.max_batch
        * contract.decode_attention.cache_page_bytes
    )
    if contract.arena.persistent_kv_bytes != expected_persistent_kv_bytes:
        raise RuntimeError("dSpark packed KV cache residency drifted")
    return True


def _qualify_target_main_kv_producer(contract: DeepseekV4DsparkContract) -> None:
    import torch
    from b12x.attention import dsv4_producer

    required_surface = {"KVPlan", "KVBinding", "plan_kv", "bind_kv", "run_kv"}
    if not required_surface.issubset(dsv4_producer.META.entry_points):
        raise RuntimeError("pinned SparkInfer DSV4 KV-only producer surface drifted")
    geometry = contract.geometry
    producer_plan = dsv4_producer.plan_kv(
        dsv4_producer.Caps(
            device="cpu",
            max_tokens=contract.max_main_rows,
            hidden=geometry.hidden,
            q_lora_rank=1_024 if geometry.variant == "flash" else 1_536,
            heads=geometry.attention_heads,
            head_dim=geometry.head_dim,
            nope_dim=448,
            rope_dim=64,
            page_size=256,
            dtype=torch.bfloat16,
            cache_format=contract.decode_attention.cache_format,
        )
    )
    (producer_scratch_spec,) = producer_plan.scratch_specs()
    producer_scratch = contract.target_main_kv_producer_scratch
    if int(producer_scratch_spec.nbytes) != producer_scratch.total_bytes:
        raise RuntimeError(
            "pinned SparkInfer DSV4 KV-only producer scratch ABI drifted: "
            f"DS4RT={producer_scratch.total_bytes} "
            f"SparkInfer={producer_scratch_spec.nbytes}"
        )
    expected_layout = (
        producer_scratch.kv_linear_offset,
        producer_scratch.kv_linear_bytes,
        producer_scratch.kv_output_offset,
        producer_scratch.kv_output_bytes,
        producer_scratch.total_bytes,
    )
    actual_layout = (
        int(producer_plan.layout.kv_linear_offset),
        int(producer_plan.layout.kv_linear_bytes),
        int(producer_plan.layout.kv_output_offset),
        int(producer_plan.layout.kv_output_bytes),
        int(producer_plan.layout.nbytes),
    )
    if actual_layout != expected_layout:
        raise RuntimeError(
            "pinned SparkInfer DSV4 KV-only producer arena offsets drifted: "
            f"DS4RT={expected_layout} SparkInfer={actual_layout}"
        )


def _qualify_dspark_decode_attention(contract: DeepseekV4DsparkContract) -> None:
    import torch
    from b12x.attention import compressed_sparse_mla as compressed_mla

    required_surface = {"Caps", "Plan", "Binding", "plan", "bind", "run"}
    if not required_surface.issubset(compressed_mla.META.entry_points):
        raise RuntimeError("pinned SparkInfer compressed-MLA surface drifted")
    attention = contract.decode_attention
    chunks = compressed_mla.split_chunks_for_contract(
        rows=attention.rows,
        width=attention.selection_width,
    )
    if int(chunks) != attention.max_chunks_per_row:
        raise RuntimeError(
            "pinned SparkInfer dSpark attention split policy drifted: "
            f"DS4RT={attention.max_chunks_per_row} SparkInfer={chunks}"
        )
    plan = compressed_mla.plan(
        compressed_mla.Caps(
            device="cpu",
            dtype=torch.bfloat16,
            kv_dtype=torch.uint8,
            num_q_heads=attention.heads,
            head_dim=DS4_DSPARK_HEAD_DIM,
            v_head_dim=DS4_DSPARK_HEAD_DIM,
            max_width=attention.selection_width,
            max_page_table_width=attention.max_batch,
            max_q_rows=attention.rows,
            max_batch=attention.rows,
            max_kv_rows=0,
            max_chunks_per_row=attention.max_chunks_per_row,
            page_size=attention.cache_page_tokens,
        )
    )
    (scratch_spec,) = plan.scratch_specs()
    if int(scratch_spec.nbytes) != attention.scratch_bytes:
        raise RuntimeError(
            "pinned SparkInfer dSpark attention scratch ABI drifted: "
            f"DS4RT={attention.scratch_bytes} SparkInfer={scratch_spec.nbytes}"
        )


def _align_up(value: int) -> int:
    return (
        (int(value) + DS4_DSPARK_ARENA_ALIGNMENT - 1)
        // DS4_DSPARK_ARENA_ALIGNMENT
        * DS4_DSPARK_ARENA_ALIGNMENT
    )


def _plan_target_main_kv_producer_scratch(
    *, geometry: DeepseekV4DsparkGeometry, max_rows: int
) -> DeepseekV4DsparkKVProducerScratchLayout:
    cursor = 0
    kv_linear_offset = _align_producer(cursor)
    kv_linear_bytes = _block_fp8_linear_scratch_bytes(
        rows=max_rows, input_width=geometry.hidden
    )
    cursor = kv_linear_offset + kv_linear_bytes
    kv_output_offset = _align_producer(cursor)
    kv_output_bytes = max_rows * geometry.head_dim * 2
    cursor = kv_output_offset + kv_output_bytes
    return DeepseekV4DsparkKVProducerScratchLayout(
        kv_linear_offset=kv_linear_offset,
        kv_linear_bytes=kv_linear_bytes,
        kv_output_offset=kv_output_offset,
        kv_output_bytes=kv_output_bytes,
        total_bytes=_align_producer(cursor),
    )


def _plan_dspark_entry_scratch(
    *, geometry: DeepseekV4DsparkGeometry, max_rows: int
) -> DeepseekV4DsparkEntryScratchLayout:
    cursor = 0
    target_tap_concat_offset = _align_producer(cursor)
    target_tap_concat_bytes = (
        max_rows * geometry.target_projection_input_width * 2
    )
    prompt_positions_offset = 0
    prompt_positions_bytes = max_rows * 4
    prompt_main_slots_offset = _align_producer(
        prompt_positions_offset + prompt_positions_bytes
    )
    prompt_main_slots_bytes = max_rows * 4
    prompt_cos_sin_offset = _align_producer(
        prompt_main_slots_offset + prompt_main_slots_bytes
    )
    prompt_cos_sin_bytes = max_rows * 64 * 4
    if prompt_cos_sin_offset + prompt_cos_sin_bytes > target_tap_concat_bytes:
        raise RuntimeError("dSpark prompt metadata no longer fits target-tap concat reuse")
    cursor = target_tap_concat_offset + target_tap_concat_bytes
    projection_scratch_offset = _align_producer(cursor)
    projection_scratch_bytes = _block_fp8_linear_scratch_bytes(
        rows=max_rows,
        input_width=geometry.target_projection_input_width,
    )
    cursor = projection_scratch_offset + projection_scratch_bytes
    return DeepseekV4DsparkEntryScratchLayout(
        target_tap_concat_offset=target_tap_concat_offset,
        target_tap_concat_bytes=target_tap_concat_bytes,
        projection_scratch_offset=projection_scratch_offset,
        projection_scratch_bytes=projection_scratch_bytes,
        prompt_positions_offset=prompt_positions_offset,
        prompt_positions_bytes=prompt_positions_bytes,
        prompt_main_slots_offset=prompt_main_slots_offset,
        prompt_main_slots_bytes=prompt_main_slots_bytes,
        prompt_cos_sin_offset=prompt_cos_sin_offset,
        prompt_cos_sin_bytes=prompt_cos_sin_bytes,
        total_bytes=_align_producer(cursor),
    )


def _plan_dspark_proposal_scratch(
    *,
    geometry: DeepseekV4DsparkGeometry,
    max_batch: int,
    arena: DeepseekV4DsparkArenaLayout,
) -> DeepseekV4DsparkProposalScratchLayout:
    rows = max_batch * geometry.proposal_tokens
    token_ids = arena.region("draft_input_token_ids")
    residual = arena.region("proposal_residual_ping_pong")
    collapsed = arena.region("proposal_collapsed_hidden")
    markov = arena.region("markov_embeddings")
    selected_indices = arena.region("reused_dspark_attention_selected_indices")
    selected_lengths = arena.region("reused_dspark_attention_selected_lengths")

    token_bytes = rows * 4
    residual_bytes = rows * DS4_HC_MULT * geometry.hidden * 2
    collapsed_bytes = rows * geometry.hidden * 2
    if token_bytes > token_ids.nbytes or 2 * residual_bytes != residual.nbytes:
        raise RuntimeError("dSpark proposal token/residual storage contract drifted")
    if collapsed_bytes != collapsed.nbytes:
        raise RuntimeError("dSpark proposal collapsed-hidden storage contract drifted")

    cursor = 0
    positions_relative = _align_proposal(cursor)
    positions_bytes = rows * 4
    cursor = positions_relative + positions_bytes
    main_slots_relative = _align_proposal(cursor)
    main_slots_bytes = rows * 4
    cursor = main_slots_relative + main_slots_bytes
    cos_sin_relative = _align_proposal(cursor)
    cos_sin_bytes = rows * 64 * 4
    cursor = cos_sin_relative + cos_sin_bytes
    post_ping_relative = _align_proposal(cursor)
    post_bytes = rows * DS4_HC_MULT * 4
    cursor = post_ping_relative + post_bytes
    comb_ping_relative = _align_proposal(cursor)
    comb_bytes = rows * DS4_HC_MULT * DS4_HC_MULT * 4
    cursor = comb_ping_relative + comb_bytes
    post_pong_relative = _align_proposal(cursor)
    cursor = post_pong_relative + post_bytes
    comb_pong_relative = _align_proposal(cursor)
    cursor = comb_pong_relative + comb_bytes
    if cursor > markov.nbytes:
        raise RuntimeError(
            "dSpark proposal metadata and mHC mix state no longer fit the pre-head "
            "markov embedding overlay"
        )

    return DeepseekV4DsparkProposalScratchLayout(
        draft_input_token_ids_offset=token_ids.offset,
        draft_input_token_ids_bytes=token_bytes,
        residual_ping_offset=residual.offset,
        residual_ping_bytes=residual_bytes,
        residual_pong_offset=residual.offset + residual_bytes,
        residual_pong_bytes=residual_bytes,
        collapsed_hidden_offset=collapsed.offset,
        collapsed_hidden_bytes=collapsed_bytes,
        positions_offset=markov.offset + positions_relative,
        positions_bytes=positions_bytes,
        main_slots_offset=markov.offset + main_slots_relative,
        main_slots_bytes=main_slots_bytes,
        cos_sin_offset=markov.offset + cos_sin_relative,
        cos_sin_bytes=cos_sin_bytes,
        post_ping_offset=markov.offset + post_ping_relative,
        post_ping_bytes=post_bytes,
        comb_ping_offset=markov.offset + comb_ping_relative,
        comb_ping_bytes=comb_bytes,
        post_pong_offset=markov.offset + post_pong_relative,
        post_pong_bytes=post_bytes,
        comb_pong_offset=markov.offset + comb_pong_relative,
        comb_pong_bytes=comb_bytes,
        selected_indices_offset=selected_indices.offset,
        selected_indices_bytes=selected_indices.nbytes,
        selected_lengths_offset=selected_lengths.offset,
        selected_lengths_bytes=selected_lengths.nbytes,
    )


def _qualify_dspark_entry(contract: DeepseekV4DsparkContract) -> None:
    import torch
    from b12x.gemm import block_fp8_linear

    geometry = contract.geometry
    plan = block_fp8_linear.plan(
        block_fp8_linear.Caps(
            device="cpu",
            max_tokens=contract.max_main_rows,
            in_features=geometry.target_projection_input_width,
            out_features=geometry.hidden,
            output_dtype=torch.bfloat16,
        )
    )
    (scratch_spec,) = plan.scratch_specs()
    if int(scratch_spec.nbytes) != contract.entry_scratch.projection_scratch_bytes:
        raise RuntimeError(
            "pinned SparkInfer dSpark entry projection scratch ABI drifted: "
            f"DS4RT={contract.entry_scratch.projection_scratch_bytes} "
            f"SparkInfer={scratch_spec.nbytes}"
        )


def _plan_dspark_decode_attention(
    *,
    geometry: DeepseekV4DsparkGeometry,
    max_batch: int,
    base_attention_scratch_bytes: int,
    cache_format: str,
    cache_page_bytes: int,
) -> DeepseekV4DsparkAttentionContract:
    rows = max_batch * geometry.proposal_tokens
    chunks = _dspark_attention_split_chunks(
        rows=rows,
        width=DS4_DSPARK_SELECTION_WIDTH,
    )
    scratch_bytes = _compressed_mla_scratch_bytes(
        rows=rows,
        heads=geometry.attention_heads,
        chunks=chunks,
    )
    return DeepseekV4DsparkAttentionContract(
        variant=geometry.variant,
        max_batch=max_batch,
        rows=rows,
        heads=geometry.attention_heads,
        max_chunks_per_row=chunks,
        scratch_bytes=scratch_bytes,
        base_attention_scratch_bytes=int(base_attention_scratch_bytes),
        cache_format=str(cache_format),
        cache_page_bytes=int(cache_page_bytes),
    )


def _dspark_attention_split_chunks(*, rows: int, width: int) -> int:
    if rows <= 256:
        decode_chunks = _ceil_div(width, 12)
        if decode_chunks <= 256:
            return decode_chunks
        wide_chunks = _ceil_div(width, 64)
        if wide_chunks <= 256:
            return wide_chunks
    return min(_ceil_div(width, 1_024), 256)


def _compressed_mla_scratch_bytes(*, rows: int, heads: int, chunks: int) -> int:
    q_chunks = rows * chunks
    cursor = 0
    cursor = _align_producer(cursor) + q_chunks * heads * DS4_DSPARK_HEAD_DIM * 2
    cursor = _align_producer(cursor) + q_chunks * heads * 4
    cursor = _align_producer(cursor) + rows * heads * 4
    cursor = _align_producer(cursor) + 4
    cursor = _align_producer(cursor) + 4
    cursor = _align_producer(cursor) + 4
    return _align_producer(cursor)


def _require_tensor(
    name: str,
    tensor: Any,
    shape: tuple[int, ...],
    dtype: Any,
    device: Any,
) -> None:
    import torch

    if (
        not isinstance(tensor, torch.Tensor)
        or tuple(tensor.shape) != shape
        or tensor.dtype != dtype
        or tensor.device != device
        or not tensor.is_contiguous()
    ):
        raise ValueError(
            f"{name} must be contiguous {dtype} {shape} on {device}"
        )


def _block_fp8_linear_scratch_bytes(*, rows: int, input_width: int) -> int:
    cursor = 0
    cursor = _align_producer(cursor) + rows * input_width
    cursor = _align_producer(cursor) + rows * (input_width // 32)
    scale_mma_bytes = (
        _ceil_div(rows, 128) * _ceil_div(input_width, 128) * 32 * 4 * 4
    )
    return _align_producer(cursor) + scale_mma_bytes


def _align_producer(value: int) -> int:
    return (
        (int(value) + DS4_DSPARK_PRODUCER_ALIGNMENT - 1)
        // DS4_DSPARK_PRODUCER_ALIGNMENT
        * DS4_DSPARK_PRODUCER_ALIGNMENT
    )


def _align_proposal(value: int) -> int:
    return (
        (int(value) + DS4_DSPARK_PROPOSAL_ALIGNMENT - 1)
        // DS4_DSPARK_PROPOSAL_ALIGNMENT
        * DS4_DSPARK_PROPOSAL_ALIGNMENT
    )


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


__all__ = [
    "DS4_DSPARK_BLOCKS",
    "DS4_DSPARK_PROPOSAL_ALIGNMENT",
    "DS4_DSPARK_CACHE_PAGE_BYTES",
    "DS4_DSPARK_CACHE_PAGE_TOKENS",
    "DS4_DSPARK_ENTRY_ACTIVATION_BLOCK_SIZE",
    "DS4_DSPARK_NOISE_TOKEN_ID",
    "DS4_DSPARK_PROPOSAL_TOKENS",
    "DS4_DSPARK_SLIDING_WINDOW",
    "DeepseekV4DsparkArenaLayout",
    "DeepseekV4DsparkArenaRegion",
    "DeepseekV4DsparkAttentionContract",
    "DeepseekV4DsparkBlockBinding",
    "DeepseekV4DsparkBlockContract",
    "DeepseekV4DsparkContract",
    "DeepseekV4DsparkEntryScratchLayout",
    "DeepseekV4DsparkGeometry",
    "DeepseekV4DsparkKVProducerScratchLayout",
    "DeepseekV4DsparkPromptPrimeBlockBinding",
    "DeepseekV4DsparkProposalEntryBinding",
    "DeepseekV4DsparkProposalScratchLayout",
    "bind_deepseek_v4_dspark_block",
    "bind_deepseek_v4_dspark_prompt_prime_block",
    "bind_deepseek_v4_dspark_proposal_entry",
    "capture_deepseek_v4_dspark_block_attention",
    "capture_deepseek_v4_dspark_block_post_dispatch",
    "capture_deepseek_v4_dspark_terminal_collapse",
    "capture_deepseek_v4_dspark_entry_projection",
    "capture_deepseek_v4_dspark_prompt_prime",
    "capture_deepseek_v4_dspark_proposal_entry",
    "deepseek_v4_dspark_arena_nbytes",
    "deepseek_v4_dspark_arena_region_nbytes",
    "deepseek_v4_dspark_arena_region_offset",
    "deepseek_v4_dspark_block_buffer_nbytes",
    "deepseek_v4_dspark_block_buffer_offset",
    "deepseek_v4_dspark_entry_buffer_nbytes",
    "deepseek_v4_dspark_entry_buffer_offset",
    "deepseek_v4_dspark_geometry",
    "deepseek_v4_dspark_persistent_kv_nbytes",
    "deepseek_v4_dspark_prompt_buffer_nbytes",
    "deepseek_v4_dspark_prompt_buffer_offset",
    "deepseek_v4_dspark_proposal_buffer_nbytes",
    "deepseek_v4_dspark_proposal_buffer_offset",
    "plan_deepseek_v4_dspark",
    "prepare_deepseek_v4_dspark_entry_projection",
    "prepare_deepseek_v4_dspark_block_attention",
    "prepare_deepseek_v4_dspark_block_post_dispatch",
    "prepare_deepseek_v4_dspark_terminal_collapse",
    "prepare_deepseek_v4_dspark_prompt_prime",
    "prepare_deepseek_v4_dspark_proposal_entry",
    "qualify_deepseek_v4_dspark_contract",
    "run_deepseek_v4_dspark_block_post_dispatch",
    "run_deepseek_v4_dspark_block_pre_dispatch",
    "run_deepseek_v4_dspark_prompt_prime_block",
    "run_deepseek_v4_dspark_proposal_entry",
]
