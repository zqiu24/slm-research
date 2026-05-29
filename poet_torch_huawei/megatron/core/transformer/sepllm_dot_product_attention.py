# Copyright (c) 2024, SepLLM Authors. CUDA-optimized SepLLM attention module.
# Drop-in replacement for DotProductAttention that uses FlexAttention / Triton
# kernels to skip masked-out blocks entirely.

import math
from typing import Optional

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide

from megatron.core.transformer.sepllm_kernels import (
    get_sepllm_context,
    get_sepllm_block_mask,
    set_sepllm_block_mask,
    build_sepllm_block_mask,
    sepllm_attention,
    _FLEX_AVAILABLE,
)


class SepLLMDotProductAttention(MegatronModule):
    """SepLLM-accelerated attention.

    Same interface as DotProductAttention, but internally dispatches to
    FlexAttention or custom Triton kernels that exploit SepLLM's block-sparse
    structure (init tokens + separator tokens + local window).

    QKV tensors arrive as [sq, b, np, hn] (Megatron layout).
    Sparse kernels expect [B, H, S, D], so we transpose in/out.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
        softmax_scale: float = None,
        cp_comm_type: str = None,
        model_comm_pgs: ModelCommProcessGroups = None,
    ):
        super().__init__(config=config)

        self.config = config
        assert config.context_parallel_size == 1, (
            "SepLLMDotProductAttention does not support context parallelism"
        )
        assert config.pipeline_model_parallel_size <= 1, (
            "SepLLMDotProductAttention requires PP=1 "
            "(sep_mask context is thread-local and only available on the first stage)"
        )

        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

        projection_size = config.kv_channels * config.num_attention_heads
        if model_comm_pgs is None:
            model_comm_pgs = ModelCommProcessGroups.use_mpu_process_groups(required_pgs=['tp'])

        world_size = model_comm_pgs.tp.size()
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(config.num_query_groups, world_size)

        if softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head)
        else:
            self.softmax_scale = softmax_scale

        if config.apply_query_key_layer_scaling:
            self.softmax_scale /= self.layer_number

        self.attention_dropout = torch.nn.Dropout(
            config.attention_dropout if attention_dropout is None else attention_dropout
        )

        kernel = getattr(config, 'sepllm_kernel', 'auto')
        if kernel == 'auto' and self.hidden_size_per_attention_head > 384:
            # The D-chunked Triton kernel supports HEAD_DIM up to 3*D_CHUNK=384.
            # Beyond that, fall back to dense SDPA.
            import warnings
            warnings.warn(
                f"SepLLM: head_dim={self.hidden_size_per_attention_head} > 384; "
                f"chunked sparse kernel unsupported at this head_dim. "
                f"Falling back to dense SDPA (no speedup).",
                stacklevel=2,
            )
            kernel = 'dense'
        # At head_dim in (128, 384], _SepLLMTritonAttnFn automatically uses the
        # D-chunked kernels (see sepllm_kernels.py). Measured speedup at D=384:
        # ~3-5x over dense SDPA on L20X, ~3x over FlashAttention causal baseline.
        self.kernel = kernel
        self.init_token_count = config.sepllm_init_token_count
        self.local_window_size = config.sepllm_local_window_size

    def forward(
        self,
        query: Tensor,       # [sq, b, np, hn]
        key: Tensor,         # [sk, b, ng, hn]
        value: Tensor,       # [sk, b, ng, hn]
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: Tensor = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        assert packed_seq_params is None, (
            "Packed sequence not supported by SepLLMDotProductAttention."
        )
        assert attention_bias is None, (
            "Attention bias not supported by SepLLMDotProductAttention."
        )

        _, sep_mask = get_sepllm_context()
        if sep_mask is None:
            raise RuntimeError(
                "SepLLM context not set. "
                "Ensure set_sepllm_context() is called before model.forward()."
            )

        kernel = self.kernel
        if kernel == 'auto':
            from megatron.core.transformer.sepllm_kernels import _TRITON_AVAILABLE
            if _TRITON_AVAILABLE:
                kernel = 'triton'
            elif _FLEX_AVAILABLE:
                kernel = 'flex_attention'

        gqa_ratio = self.num_attention_heads_per_partition // self.num_query_groups_per_partition

        # GQA expansion: FlexAttention and the D-chunked Triton kernel (D>128) handle
        # GQA natively. Single-tile Triton (D<=128) and dense-SDPA need explicit
        # expansion since they require K/V heads to match Q heads.
        chunked_triton_native_gqa = (
            kernel == 'triton' and self.hidden_size_per_attention_head > 128
        )
        need_gqa_expand = (
            gqa_ratio > 1
            and kernel != 'flex_attention'
            and not chunked_triton_native_gqa
        )
        if need_gqa_expand:
            key = key.unsqueeze(3).expand(-1, -1, -1, gqa_ratio, -1).reshape(
                key.shape[0], key.shape[1], -1, key.shape[3]
            )
            value = value.unsqueeze(3).expand(-1, -1, -1, gqa_ratio, -1).reshape(
                value.shape[0], value.shape[1], -1, value.shape[3]
            )

        # Reshape from Megatron [S, B, H, D] -> kernel [B, H, S, D]
        sq, b, np_, hn = query.shape
        sk = key.shape[0]

        q = query.permute(1, 2, 0, 3).contiguous()  # [B, H, Sq, D]
        k = key.permute(1, 2, 0, 3).contiguous()    # [B, H, Sk, D]
        v = value.permute(1, 2, 0, 3).contiguous()   # [B, H, Sk, D]

        # For FlexAttention: build block mask once, cache for all layers
        if kernel == 'flex_attention' and _FLEX_AVAILABLE:
            blk_mask = get_sepllm_block_mask()
            if blk_mask is None:
                blk_mask = build_sepllm_block_mask(
                    sep_mask, 1, sq,
                    self.init_token_count,
                    self.local_window_size,
                )
                set_sepllm_block_mask(blk_mask)

            from megatron.core.transformer.sepllm_kernels import flex_attention_forward
            out = flex_attention_forward(
                q, k, v,
                block_mask=blk_mask,
                scale=self.softmax_scale,
                enable_gqa=(gqa_ratio > 1),
            )
        else:
            out = sepllm_attention(
                q, k, v,
                sep_mask=sep_mask,
                init_token_count=self.init_token_count,
                local_window_size=self.local_window_size,
                softmax_scale=self.softmax_scale,
                kernel=kernel,
            )

        # Back to Megatron layout [Sq, B, H, D] -> [Sq, B, Hp]
        context = out.permute(2, 0, 1, 3).contiguous()  # [Sq, B, H, D]
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.view(*new_context_shape)

        return context
