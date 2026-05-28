# Copyright (c) 2024, SepLLM Authors. Integration into Megatron-LM.
# Implements the SepLLM sparse attention mask construction for training.
# Reference: "SepLLM: Accelerate Large Language Models by Compressing One Segment
#             into One Separator" (ICML 2025, arXiv:2412.12094)

import torch
from torch import Tensor
from typing import List, Optional


class SepLLMAttentionMaskBuilder:
    """Builds the SepLLM sparse attention mask from input_ids.

    SepLLM keeps attention only to:
      1) Initial tokens (attention sinks) — first `init_token_count` tokens
      2) Separator tokens (punctuation, whitespace, etc.)
      3) Local window (last `local_window_size` tokens for each query position)

    The mask is applied on top of the standard causal mask.
    """

    def __init__(
        self,
        separator_token_ids: List[int],
        init_token_count: int = 3,
        local_window_size: int = 64,
        padding_token_id: int = 0,
    ):
        self.separator_token_ids = separator_token_ids
        self.init_token_count = init_token_count
        self.local_window_size = local_window_size
        self.padding_token_id = padding_token_id

    def build_sepllm_mask(
        self,
        input_ids: Tensor,
        causal_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Build SepLLM attention mask.

        Args:
            input_ids: [batch_size, seq_len]
            causal_mask: Optional existing causal mask [b, 1, sq, sk] where True = masked out.
                         If None, a standard causal mask is created.

        Returns:
            attention_mask: [b, 1, sq, sk] where True = masked out (Megatron convention)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # --- Separator token positions: [b, seq_len] ---
        sep_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        for sid in self.separator_token_ids:
            sep_mask = sep_mask | (input_ids == sid)

        # --- Initial tokens: first `init_token_count` positions always visible ---
        init_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        init_count = min(self.init_token_count, seq_len)
        init_mask[:, :init_count] = True

        # --- Combine: tokens to keep (attend to) = sep OR initial ---
        keep_kv = sep_mask | init_mask  # [b, seq_len]

        # Expand to [b, 1, 1, seq_len] for broadcasting across query dimension
        keep_kv_expanded = keep_kv[:, None, None, :]  # [b, 1, 1, sk]
        keep_kv_expanded = keep_kv_expanded.expand(-1, -1, seq_len, -1)  # [b, 1, sq, sk]

        # --- Local window: for each query position q, attend to keys in [q-w+1, q] ---
        q_pos = torch.arange(seq_len, device=device).unsqueeze(1)  # [sq, 1]
        k_pos = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, sk]
        local_window_mask = (q_pos - k_pos) < self.local_window_size  # [sq, sk]
        local_window_mask = local_window_mask & (k_pos <= q_pos)  # also causal
        local_window_mask = local_window_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, sq, sk]

        # --- Standard causal mask: lower triangular ---
        causal = (k_pos <= q_pos).unsqueeze(0).unsqueeze(0)  # [1, 1, sq, sk]

        # --- Final attend mask: (sep/init OR local_window) AND causal ---
        attend_mask = (keep_kv_expanded | local_window_mask) & causal  # [b, 1, sq, sk]

        # Megatron convention: True = masked out (don't attend)
        sepllm_mask = ~attend_mask

        # If an existing causal mask was provided, combine with it
        if causal_mask is not None:
            sepllm_mask = sepllm_mask | causal_mask

        return sepllm_mask


def sepllm_attention_mask_sparsity_causal(attention_mask: Tensor) -> Tensor:
    """Fraction of **blocked** (masked) entries within the causal lower triangle.

    Megatron convention: ``True`` = do not attend.

    Only positions with ``k <= q`` are counted (the upper triangle is excluded).

    **Why values like ~0.9 are normal for SepLLM:** for large ``q``, each valid key
    row has ``q+1`` causal slots but only about ``local_window_size + init_token_count``
    plus separator keys to the left (~ a fraction of ``q``) may attend. The row-wise
    blocked fraction approaches ``1 - sep_rate`` as ``q`` grows; pair-count averages
    are dominated by large ``q``, so the global metric is often **0.85–0.95**, not
    a small number. This is **pair-blocked rate in the causal block**, not FLOPs
    saved vs fused dense kernel.

    Returns:
        0-dim float tensor on the same device as ``attention_mask``.
    """
    if attention_mask.dtype == torch.bool:
        m = attention_mask
    else:
        m = attention_mask > 0
    sq, sk = m.shape[-2], m.shape[-1]
    row = torch.arange(sq, device=m.device, dtype=torch.long).unsqueeze(1)
    col = torch.arange(sk, device=m.device, dtype=torch.long).unsqueeze(0)
    causal = col <= row
    for _ in range(m.dim() - 2):
        causal = causal.unsqueeze(0)
    causal = causal.expand_as(m)
    denom = causal.sum().to(torch.float32).clamp_min(1.0)
    num = (m & causal).sum().to(torch.float32)
    return num / denom


def build_sepllm_attention_mask(
    input_ids: Tensor,
    attention_mask: Optional[Tensor],
    config,
) -> Tensor:
    """Top-level function to build SepLLM mask, called from forward_step.

    Args:
        input_ids: [b, seq_len] token ids
        attention_mask: existing causal mask (True=masked) or None
        config: object with sepllm_* attributes

    Returns:
        Modified attention_mask with SepLLM sparsity applied
    """
    if not getattr(config, 'use_sepllm_attention', False):
        return attention_mask

    builder = SepLLMAttentionMaskBuilder(
        separator_token_ids=config.sepllm_separator_token_ids,
        init_token_count=config.sepllm_init_token_count,
        local_window_size=config.sepllm_local_window_size,
        padding_token_id=config.sepllm_padding_token_id,
    )

    sepllm_mask = builder.build_sepllm_mask(
        input_ids=input_ids,
        causal_mask=attention_mask,
    )

    return sepllm_mask
