
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide, get_pg_size

try:
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TELinear

    HAVE_TE = True
except ImportError:
    TELinear = None
    TEColumnParallelLinear = None
    HAVE_TE = False


@dataclass
class NSASelfAttentionSubmodules:
    """Submodules for NSA self-attention (compress-only).

    The four ``compress_linear_*`` fields should be **replicated** linears
    (e.g. ``backend.linear()`` → ``TELinear`` with
    ``parallel_mode='duplicated'``).  They form two 2-layer MLPs::

        K MLP:  compress_linear_k_1 → ReLU → compress_linear_k_2
        V MLP:  compress_linear_v_1 → ReLU → compress_linear_v_2
    """

    linear_qkv: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    compress_layernorm: Union[ModuleSpec, type] = None    # IdentityOp → no layernorm
    compress_mid_norm: Union[ModuleSpec, type] = None     # norm between down/up in compress MLP
    compress_linear_k_1: Union[ModuleSpec, type] = None   # K MLP first linear
    compress_linear_k_2: Union[ModuleSpec, type] = None   # K MLP second linear
    compress_linear_v_1: Union[ModuleSpec, type] = None   # V MLP first linear
    compress_linear_v_2: Union[ModuleSpec, type] = None   # V MLP second linear


class NSASelfAttention(MegatronModule):
    """Compressed-only Native Sparse Attention.

    TP strategy
    -----------
    * QKV projection  : ColumnParallelLinear  (standard Megatron)
    * Compress MLP     : replicated linear (``TELinear`` with
      ``parallel_mode='duplicated'``, or ``ColumnParallelLinear`` with
      ``gather_output``).  Each TP rank processes its own KV head partition;
      weight gradients are synchronized internally by the backend.
    * Output projection: RowParallelLinear (standard Megatron)
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: NSASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        cp_comm_type: str = None,
        model_comm_pgs=None,
    ):
        super().__init__(config=config)
        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type

        # ---- process groups ------------------------------------------------
        if model_comm_pgs is None:
            from megatron.core.process_groups_config import ModelCommProcessGroups

            model_comm_pgs = ModelCommProcessGroups.use_mpu_process_groups(
                required_pgs=['tp', 'cp']
            )
        else:
            assert hasattr(model_comm_pgs, 'tp'), (
                "NSASelfAttention model_comm_pgs must have tp process group"
            )
            assert hasattr(model_comm_pgs, 'cp'), (
                "NSASelfAttention model_comm_pgs must have cp process group"
            )
        self.model_comm_pgs = model_comm_pgs
        tp_group = self.model_comm_pgs.tp
        tp_world_size = get_pg_size(tp_group) if tp_group is not None else 1

        cp_group = self.model_comm_pgs.cp
        cp_world_size = get_pg_size(cp_group) if cp_group is not None else 1
        if cp_world_size > 1:
            raise NotImplementedError(
                "NSA compress-only attention does not support context "
                "parallelism (CP > 1) yet."
            )

        # ---- config validation ---------------------------------------------
        assert getattr(config, "use_native_sparse_attention", False), (
            "NSASelfAttention requires config.use_native_sparse_attention=True"
        )

        # ---- head setting -------------------------------------------------
        num_heads = config.num_attention_heads
        num_kv_heads = (
            config.num_query_groups if config.num_query_groups is not None else num_heads
        )
        head_dim = (
            config.kv_channels if config.kv_channels is not None
            else config.hidden_size // num_heads
        )

        self.num_attention_heads_per_partition = divide(num_heads, tp_world_size)
        self.num_query_groups_per_partition = divide(num_kv_heads, tp_world_size)
        self.hidden_size_per_attention_head = head_dim
        self.head_dim = head_dim
        self.num_grouped_queries = num_heads // num_kv_heads

        query_projection_size = num_heads * head_dim
        kv_projection_size = num_kv_heads * head_dim
        self.query_projection_size = query_projection_size

        # ---- scale ---------------------------------------------------------
        self.softmax_scale = head_dim ** -0.5
        if config.softmax_scale is not None:
            self.softmax_scale = config.softmax_scale

        # ---- NSA hyper-parameters ------------------------------------------
        self.swa_only = config.nsa_swa_only
        self.swa_window_size = config.nsa_swa_window_size

        self.compress_block_size = config.nsa_compress_block_size
        self.compress_block_sliding_stride = config.nsa_compress_block_sliding_stride
        self.num_compressed_mem_kv = config.nsa_num_compressed_mem_kv
        compress_mlp_expand_factor = config.nsa_compress_mlp_expand_factor

        if not self.swa_only:
            assert self.compress_block_size >= self.compress_block_sliding_stride > 0, (
                f"compress_block_size ({self.compress_block_size}) must be >= "
                f"compress_block_sliding_stride ({self.compress_block_sliding_stride}) > 0"
            )
            assert self.num_compressed_mem_kv > 0, "num_compressed_mem_kv must be > 0"

        self.causal = True  # NSA is for autoregressive models

        init_method = config.init_method

        # ==================================================================
        # 1. QKV projection  (ColumnParallelLinear – standard Megatron)
        # ==================================================================
        self.linear_qkv = build_module(
            submodules.linear_qkv,
            config.hidden_size,
            query_projection_size + 2 * kv_projection_size,
            config=config,
            init_method=init_method,
            gather_output=False,
            bias=config.add_bias_linear or getattr(config, "add_qkv_bias", False),
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="qkv",
            tp_group=tp_group,
        )

        # ==================================================================
        # 2. Compress path  (skipped when swa_only=True)
        # ==================================================================
        if not self.swa_only:
            compress_dim = self.compress_block_size * head_dim

            if submodules.compress_layernorm is not None:
                self.compress_layernorm = build_module(
                    submodules.compress_layernorm,
                    config=config,
                    hidden_size=compress_dim,
                    eps=config.layernorm_epsilon,
                )
            else:
                self.compress_layernorm = None

            compress_hidden = int(compress_mlp_expand_factor * compress_dim)
            self._compress_hidden = compress_hidden

            if submodules.compress_mid_norm is not None and submodules.compress_mid_norm is not IdentityOp:
                self.compress_mid_norm = build_module(
                    submodules.compress_mid_norm,
                    config=config,
                    hidden_size=compress_hidden,
                    eps=config.layernorm_epsilon,
                )
            else:
                self.compress_mid_norm = None

            self.compress_k_down = self._build_compress_linear(
                submodules.compress_linear_k_1, compress_dim, compress_hidden,
                config, init_method, 'compress_k_down',
            )
            self.compress_k_up = self._build_compress_linear(
                submodules.compress_linear_k_2, compress_hidden, head_dim,
                config, init_method, 'compress_k_up',
            )
            self.compress_v_down = self._build_compress_linear(
                submodules.compress_linear_v_1, compress_dim, compress_hidden,
                config, init_method, 'compress_v_down',
            )
            self.compress_v_up = self._build_compress_linear(
                submodules.compress_linear_v_2, compress_hidden, head_dim,
                config, init_method, 'compress_v_up',
            )
            self.compress_act = nn.ReLU()

            self.compress_mem_kv = nn.Parameter(
                torch.zeros(
                    2,
                    self.num_query_groups_per_partition,
                    self.num_compressed_mem_kv,
                    head_dim,
                )
            )
            self.k_intrablock_positions = nn.Parameter(
                torch.zeros(
                    self.num_query_groups_per_partition,
                    self.compress_block_size,
                    head_dim,
                )
            )
            self.v_intrablock_positions = nn.Parameter(
                torch.zeros(
                    self.num_query_groups_per_partition,
                    self.compress_block_size,
                    head_dim,
                )
            )

        # ==================================================================
        # 3. Output projection (RowParallelLinear – standard Megatron)
        # ==================================================================
        self.linear_proj = build_module(
            submodules.linear_proj,
            query_projection_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="proj",
            tp_group=tp_group,
        )

        # ---- attention dropout ---------------------------------------------
        self.attn_drop = (
            nn.Dropout(config.attention_dropout)
            if config.attention_dropout > 0
            else nn.Identity()
        )

    # ------------------------------------------------------------------
    # Compress-MLP helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_compress_linear(
        linear_spec, in_features, out_features, config, init_method, buffer_name,
    ):
        """Build a single replicated linear for the compress MLP.

        Detects the module type (following the MLA pattern) and passes the
        correct kwargs so that weight gradients are synchronized across TP.
        """
        extra_kwargs = {}
        if TELinear is not None and linear_spec in [TELinear]:
            extra_kwargs['parallel_mode'] = 'duplicated'
        elif linear_spec in (
            [x for x in [ColumnParallelLinear, TEColumnParallelLinear] if x is not None]
        ):
            extra_kwargs['gather_output'] = False

        return build_module(
            linear_spec,
            in_features,
            out_features,
            config=config,
            init_method=init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name=buffer_name,
            skip_weight_param_allocation=False,
            **extra_kwargs,
        )

    def _apply_compress_mlp(self, x: Tensor, down, up, expected_hidden: int) -> Tensor:
        """Run a 2-layer compress MLP:  down → ReLU → [mid_norm] → up.

        Handles the (output, bias) tuple returned by Megatron/TE linear layers,
        and gathers output when using ColumnParallelLinear (detected by
        comparing actual output size against the expected full dimension).
        """
        h, _ = down(x)
        if h.size(-1) != expected_hidden:
            h = gather_from_tensor_model_parallel_region(h)
        h = self.compress_act(h)
        if self.compress_mid_norm is not None:
            orig_shape = h.shape
            h = self.compress_mid_norm(h.reshape(-1, orig_shape[-1])).reshape(orig_shape)
        out, _ = up(h)
        if out.size(-1) != self.head_dim:
            out = gather_from_tensor_model_parallel_region(out)
        return out

    # ------------------------------------------------------------------
    # RoPE helper for (B, H, S, D) layout
    # ------------------------------------------------------------------

    def _apply_rope_bhsd(self, t: Tensor, freqs: Tensor, positions: Tensor) -> Tensor:
        """Apply RoPE to a tensor in ``(B, H, S, D)`` layout at arbitrary positions.

        Args:
            t: ``(B, H, S, D)``
            freqs: ``(max_seq_len, 1, 1, rot_dim)`` — raw angle frequencies
                from the rotary embedding module.
            positions: ``(S,)`` ``long`` — absolute position index for each
                token along the S dimension.

        Returns:
            ``(B, H, S, D)`` with RoPE applied to the first ``rot_dim`` dims.
        """
        pos_freqs = freqs[positions][:, 0, 0, :]          # (S, rot_dim)
        pos_freqs = pos_freqs[None, None, :, :]            # (1, 1, S, rot_dim)

        rot_dim = pos_freqs.shape[-1]
        t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]

        cos_ = torch.cos(pos_freqs).to(t.dtype)
        sin_ = torch.sin(pos_freqs).to(t.dtype)

        if not self.config.rotary_interleaved:
            t1, t2 = torch.chunk(t_rot, 2, dim=-1)
            t_rotated = torch.cat((-t2, t1), dim=-1)
        else:
            t1 = t_rot[..., ::2]
            t2 = t_rot[..., 1::2]
            t_rotated = torch.stack((-t2, t1), dim=-1).reshape_as(t_rot)

        t_rot = t_rot * cos_ + t_rotated * sin_
        return torch.cat((t_rot, t_pass), dim=-1)

    # ------------------------------------------------------------------
    # QKV extraction
    # ------------------------------------------------------------------

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """Derive Q, K, V from hidden_states via QKV projection.

        Args:
            hidden_states: ``(sq, b, hidden_size)``

        Returns:
            query:  ``(sq, b, num_q_heads_per_partition, head_dim)``
            key:    ``(sq, b, num_kv_groups_per_partition, head_dim)``
            value:  ``(sq, b, num_kv_groups_per_partition, head_dim)``
        """
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        q_size = self.num_attention_heads_per_partition * self.head_dim
        kv_size = self.num_query_groups_per_partition * self.head_dim
        query, key, value = torch.split(mixed_qkv, [q_size, kv_size, kv_size], dim=-1)

        sq, b = query.size(0), query.size(1)
        query = query.view(sq, b, self.num_attention_heads_per_partition, self.head_dim)
        key = key.view(sq, b, self.num_query_groups_per_partition, self.head_dim)
        value = value.view(sq, b, self.num_query_groups_per_partition, self.head_dim)
        return query, key, value


    def _split_compress_windows(self, x: Tensor) -> Tensor:
        """Split a KV tensor into overlapping sliding windows for compression.

        Left-pads the sequence so that sliding windows cover all original
        tokens (consistent with the reference implementation).

        Args:
            x: ``(batch, num_kv_groups_per_partition, seq_len, head_dim)``

        Returns:
            ``(batch, num_kv_groups_per_partition, num_windows,
              compress_block_size, head_dim)``
        """
        batch, heads, seq_len, d = x.shape
        stride = self.compress_block_sliding_stride
        block_size = self.compress_block_size

        if seq_len == 0:
            return x.new_zeros(batch, heads, 0, block_size, d)

        # Left-pad so windows cover the beginning of the sequence
        left_pad = block_size - stride
        x_padded = F.pad(x, (0, 0, left_pad, 0))  # pad seq-dim on the left
        padded_len = x_padded.size(2)

        num_windows = (padded_len - block_size) // stride + 1
        if num_windows <= 0:
            return x.new_zeros(batch, heads, 0, block_size, d)

        # Efficient window extraction via torch.Tensor.unfold
        windows = x_padded.unfold(2, block_size, stride)
        # (B, H, num_windows, D, block_size) → (B, H, num_windows, block_size, D)
        windows = windows.permute(0, 1, 2, 4, 3).contiguous()
        return windows

    def _compress_kv(
        self, k: Tensor, v: Tensor, attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compress K and V using learned MLP with intra-block positions.

        Args:
            k, v: ``(batch, num_kv_groups_per_partition, seq_len, head_dim)``
            attention_mask: ``(B_mask, 1, sq, sq)`` bool, ``True`` = blocked.
                When provided, tokens inside a window that belong to a
                different document (determined by the mask) are zeroed out
                before the compress MLP, preventing cross-document leakage.

        Returns:
            ck, cv: ``(batch, num_kv_groups_per_partition, num_compressed, head_dim)``
        """
        batch, heads, seq_len, d = k.shape
        stride = self.compress_block_sliding_stride
        block_sz = self.compress_block_size

        # Split into overlapping windows
        windows_k = self._split_compress_windows(k)  # (B, H, W, block_size, D)
        windows_v = self._split_compress_windows(v)

        num_windows = windows_k.size(2)
        if num_windows == 0:
            return (
                k.new_zeros(batch, heads, 0, d),
                v.new_zeros(batch, heads, 0, d),
            )

        # Add learnable intra-block position embeddings
        windows_k = windows_k + self.k_intrablock_positions[None, :, None, :, :]
        windows_v = windows_v + self.v_intrablock_positions[None, :, None, :, :]

        # Zero out tokens that cross document boundaries within each window.
        # For window w ending at position end_w, a slot j at original position
        # p is zeroed if attention_mask[b, 0, end_w, p] says "blocked" (i.e.
        # p belongs to a different document than end_w) or if p < 0 (left pad).
        if attention_mask is not None and attention_mask.size(0) > 1:
            device = k.device
            w_idx = torch.arange(num_windows, device=device)
            j_idx = torch.arange(block_sz, device=device)

            orig_pos = (
                (w_idx[:, None] + 1) * stride - block_sz + j_idx[None, :]
            )                                                       # (W, block_sz)
            end_pos = (w_idx + 1) * stride - 1                     # (W,)

            valid = orig_pos >= 0                                   # (W, block_sz)
            orig_safe = orig_pos.clamp(0, seq_len - 1)
            end_safe = end_pos.clamp(0, seq_len - 1)

            # attention_mask: (B_mask, 1, sq, sq)
            mask_2d = attention_mask[:, 0]                          # (B_mask, sq, sq)
            rows = mask_2d[:, end_safe, :]                          # (B_mask, W, sq)
            is_blocked = rows.gather(
                2, orig_safe.unsqueeze(0).expand(rows.size(0), -1, -1)
            )                                                       # (B_mask, W, block_sz)

            keep = valid.unsqueeze(0) & ~is_blocked                 # (B_mask, W, block_sz)
            keep = keep[:, None, :, :, None].to(windows_k.dtype)    # (B_mask,1,W,block_sz,1)

            windows_k = windows_k * keep
            windows_v = windows_v * keep

        # Flatten each window: (B, H, W, block_size × D)
        flat_k = windows_k.reshape(batch, heads, num_windows, -1)
        flat_v = windows_v.reshape(batch, heads, num_windows, -1)

        # Optional layernorm before MLP
        if self.compress_layernorm is not None:
            flat_k = self.compress_layernorm(flat_k)
            flat_v = self.compress_layernorm(flat_v)

        # Apply compress MLP (shared across heads; operates on last dim)
        ck = self._apply_compress_mlp(
            flat_k, self.compress_k_down, self.compress_k_up, self._compress_hidden,
        )
        cv = self._apply_compress_mlp(
            flat_v, self.compress_v_down, self.compress_v_up, self._compress_hidden,
        )

        return ck, cv

    def _gather_swa_kv(
        self,
        k_seq: Tensor,
        v_seq: Tensor,
        sq: int,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Gather last ``swa_window_size`` keys/values per query position.

        Uses ``F.pad`` + ``unfold`` to return strided *views* instead of
        copies, so memory cost is O(sq) rather than O(sq * W).

        Args:
            k_seq, v_seq: ``(B, H_kv, sq, D)`` (RoPE already applied to ``k_seq``).
        Returns:
            k_win, v_win: ``(B, H_kv, sq, W, D)`` — strided views, not copies.
            valid: ``(B, 1, sq, W)`` — ``True`` where the slot is a real token.
        """
        w = self.swa_window_size
        b = k_seq.size(0)

        k_padded = F.pad(k_seq, (0, 0, w - 1, 0))  # (B, H, sq+w-1, D)
        v_padded = F.pad(v_seq, (0, 0, w - 1, 0))

        k_win = k_padded.unfold(2, w, 1).permute(0, 1, 2, 4, 3)  # (B, H, sq, W, D)
        v_win = v_padded.unfold(2, w, 1).permute(0, 1, 2, 4, 3)

        t_idx = torch.arange(sq, device=device, dtype=torch.long).unsqueeze(1)
        w_idx = torch.arange(w, device=device, dtype=torch.long).unsqueeze(0)
        valid = (t_idx - (w - 1) + w_idx) >= 0  # (sq, W)
        valid_b = valid.unsqueeze(0).unsqueeze(0).expand(b, 1, -1, -1)

        return k_win, v_win, valid_b

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------

    def _build_compress_mask(
        self,
        num_compress: int,
        seq_len: int,
        batch: int,
        device: torch.device,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Causal + document mask for compressed KV (including memory tokens).

        Returns:
            ``(B_out, 1, sq, mem + num_compress)`` bool, ``True`` = attend.
            ``B_out`` is ``batch`` when a per-sample document mask is present,
            otherwise 1 (broadcastable).
        """
        q_pos = torch.arange(seq_len, device=device, dtype=torch.long)

        if num_compress > 0:
            k_pos = (
                (torch.arange(num_compress, device=device, dtype=torch.long) + 1)
                * self.compress_block_sliding_stride - 1
            )
            k_pos = F.pad(k_pos, (self.num_compressed_mem_kv, 0), value=-1)
        else:
            k_pos = torch.full(
                (self.num_compressed_mem_kv,), -1,
                device=device, dtype=torch.long,
            )

        mask = (k_pos.unsqueeze(0) <= q_pos.unsqueeze(1))[None, None, :, :]

        if attention_mask is not None and attention_mask.size(0) > 1 and num_compress > 0:
            compress_end = (
                (torch.arange(num_compress, device=device, dtype=torch.long) + 1)
                * self.compress_block_sliding_stride - 1
            ).clamp(0, seq_len - 1)
            doc_blocked = attention_mask[:, :, :, compress_end]
            mem_ok = torch.zeros(
                attention_mask.size(0), 1, seq_len,
                self.num_compressed_mem_kv,
                dtype=torch.bool, device=device,
            )
            mask = mask & ~torch.cat([mem_ok, doc_blocked], dim=-1)

        if mask.size(0) == 1 and batch > 1:
            mask = mask.expand(batch, -1, -1, -1)
        return mask

    def _build_swa_mask(
        self,
        swa_valid: Tensor,
        seq_len: int,
        batch: int,
        device: torch.device,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Sliding-window validity mask with optional document boundary masking.

        Args:
            swa_valid: ``(B, 1, sq, W)`` from ``_gather_swa_kv``.

        Returns:
            ``(B, 1, sq, W)`` bool, ``True`` = attend.
        """
        if attention_mask is None or attention_mask.size(0) <= 1:
            return swa_valid

        w = self.swa_window_size
        t_idx = torch.arange(seq_len, device=device).view(seq_len, 1)
        w_idx = torch.arange(w, device=device).view(1, w)
        j_clamped = (t_idx - (w - 1) + w_idx).clamp(0, seq_len - 1).long()

        m2d = attention_mask[:, 0]                                      # (B, sq, sq)
        j_idx = j_clamped.view(1, seq_len, w).expand(batch, -1, -1)
        blocked = torch.gather(m2d, dim=2, index=j_idx).unsqueeze(1)    # (B, 1, sq, W)
        return swa_valid & ~blocked

    # ------------------------------------------------------------------
    # Compressed (+ optional SWA) attention
    # ------------------------------------------------------------------

    def _prepend_mem_kv(
        self,
        ck: Tensor,
        cv: Tensor,
        batch: int,
        k_freqs: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Prepend learnable memory tokens to compressed KV.

        When ``k_freqs`` is provided, RoPE at position 0 is applied to the
        memory K tokens before concatenation (compress-only path).
        """
        mem_ck = self.compress_mem_kv[0].unsqueeze(0).expand(batch, -1, -1, -1)
        mem_cv = self.compress_mem_kv[1].unsqueeze(0).expand(batch, -1, -1, -1)
        if k_freqs is not None:
            mem_pos = torch.zeros(
                self.num_compressed_mem_kv, device=device, dtype=torch.long,
            )
            mem_ck = self._apply_rope_bhsd(mem_ck, k_freqs, mem_pos)
        num_compress = ck.size(2)
        ck = torch.cat([mem_ck, ck], dim=2) if num_compress > 0 else mem_ck
        cv = torch.cat([mem_cv, cv], dim=2) if num_compress > 0 else mem_cv
        return ck, cv

    def _compress_only_attention(
        self,
        q: Tensor,
        ck: Tensor,
        cv: Tensor,
        seq_len: int,
        device: torch.device,
        attention_mask: Optional[Tensor] = None,
        k_freqs: Optional[Tensor] = None,
    ) -> Tensor:
        """Attention over compressed KV only — uses SDPA fused kernel."""
        batch = q.size(0)
        num_compress = ck.size(2)

        if num_compress == 0:
            return torch.zeros_like(q)

        if k_freqs is not None:
            virtual_pos = (
                (torch.arange(num_compress, device=device, dtype=torch.long) + 1)
                * self.compress_block_sliding_stride - 1
            ).clamp(0, k_freqs.size(0) - 1)
            ck = self._apply_rope_bhsd(ck, k_freqs, virtual_pos)

        ck, cv = self._prepend_mem_kv(ck, cv, batch, k_freqs=k_freqs, device=device)

        # Additive mask for SDPA: 0 = attend, -inf = block
        sdpa_mask: Optional[Tensor] = None
        if self.causal:
            bool_mask = self._build_compress_mask(
                num_compress, seq_len, batch, device, attention_mask,
            )
            sdpa_mask = torch.zeros_like(bool_mask, dtype=q.dtype)
            sdpa_mask.masked_fill_(~bool_mask, float('-inf'))

        # GQA: expand KV heads to match Q (MQA with Hkv=1 broadcasts naturally)
        if self.num_grouped_queries > 1 and ck.size(1) > 1:
            ck = ck.repeat_interleave(self.num_grouped_queries, dim=1)
            cv = cv.repeat_interleave(self.num_grouped_queries, dim=1)

        return F.scaled_dot_product_attention(
            q, ck, cv,
            attn_mask=sdpa_mask,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            scale=self.softmax_scale,
        )

    def _compress_swa_attention(
        self,
        q: Tensor,
        ck: Tensor,
        cv: Tensor,
        k_rope_seq: Tensor,
        v_seq: Tensor,
        seq_len: int,
        device: torch.device,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compress branch (NoPE) + sliding-window branch (RoPE).

        Scores are ``[compress | swa]`` along the key axis with a joint softmax
        over ``mem + num_compress + W`` positions.
        """
        batch = q.size(0)
        num_compress = ck.size(2)
        w = self.swa_window_size

        ck_cat, cv_cat = self._prepend_mem_kv(ck, cv, batch)
        c_kv = ck_cat.size(2)

        k_win, v_win, swa_valid = self._gather_swa_kv(
            k_rope_seq, v_seq, seq_len, device,
        )

        group_size = self.num_grouped_queries
        assert q.size(1) == self.num_query_groups_per_partition * group_size, (
            "Unexpected query head layout for NSA GQA."
        )
        q_grouped = q.reshape(
            batch, self.num_query_groups_per_partition, group_size,
            seq_len, self.head_dim,
        )

        attn_mask: Optional[Tensor] = None
        if self.causal:
            mask_c = self._build_compress_mask(
                num_compress, seq_len, batch, device, attention_mask,
            )
            mask_s = self._build_swa_mask(
                swa_valid, seq_len, batch, device, attention_mask,
            )
            attn_mask = torch.cat([mask_c, mask_s], dim=-1)

        softmax_scale = self.softmax_scale
        attn_drop = self.attn_drop

        def _attn_core_hybrid(qg_, ck_cat_, cv_cat_, k_win_, v_win_):
            logits_c = torch.einsum('bhgtd,bhkd->bhgtk', qg_, ck_cat_) * softmax_scale
            logits_s = torch.einsum('bhgtd,bhtwd->bhgtw', qg_, k_win_) * softmax_scale
            aw = torch.cat([logits_c, logits_s], dim=-1)
            if attn_mask is not None:
                aw = aw.masked_fill(
                    ~attn_mask.unsqueeze(2), torch.finfo(aw.dtype).min,
                )
            ap = F.softmax(aw, dim=-1, dtype=torch.float32).to(qg_.dtype)
            ap = attn_drop(ap)
            ap_c, ap_s = ap.split([c_kv, w], dim=-1)
            out = torch.einsum('bhgtk,bhkd->bhgtd', ap_c, cv_cat_)
            out = out + torch.einsum('bhgtw,bhtwd->bhgtd', ap_s, v_win_)
            return out.reshape(batch, q.size(1), seq_len, self.head_dim)

        if self.training and self.config.recompute_granularity is not None:
            return torch.utils.checkpoint.checkpoint(
                _attn_core_hybrid,
                q_grouped, ck_cat, cv_cat, k_win, v_win,
                use_reentrant=False,
            )
        return _attn_core_hybrid(q_grouped, ck_cat, cv_cat, k_win, v_win)

    def _swa_only_attention(
        self,
        q: Tensor,
        k_rope_seq: Tensor,
        v_seq: Tensor,
        seq_len: int,
        device: torch.device,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Sliding-window attention only (no compression branch)."""
        batch = q.size(0)

        k_win, v_win, swa_valid = self._gather_swa_kv(
            k_rope_seq, v_seq, seq_len, device,
        )

        group_size = self.num_grouped_queries
        q_grouped = q.reshape(
            batch, self.num_query_groups_per_partition, group_size,
            seq_len, self.head_dim,
        )

        attn_mask: Optional[Tensor] = None
        if self.causal:
            attn_mask = self._build_swa_mask(
                swa_valid, seq_len, batch, device, attention_mask,
            )

        softmax_scale = self.softmax_scale
        attn_drop = self.attn_drop

        def _attn_core_swa(qg_, k_win_, v_win_):
            logits = torch.einsum('bhgtd,bhtwd->bhgtw', qg_, k_win_) * softmax_scale
            if attn_mask is not None:
                logits = logits.masked_fill(
                    ~attn_mask.unsqueeze(2), torch.finfo(logits.dtype).min,
                )
            ap = F.softmax(logits, dim=-1, dtype=torch.float32).to(qg_.dtype)
            ap = attn_drop(ap)
            out = torch.einsum('bhgtw,bhtwd->bhgtd', ap, v_win_)
            return out.reshape(batch, q.size(1), seq_len, self.head_dim)

        if self.training and self.config.recompute_granularity is not None:
            return torch.utils.checkpoint.checkpoint(
                _attn_core_swa, q_grouped, k_win, v_win,
                use_reentrant=False,
            )
        return _attn_core_swa(q_grouped, k_win, v_win)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        key_value_states: Optional[Tensor] = None,
        inference_context=None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params=None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params=None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass — compress-only NSA.

        Args:
            hidden_states: ``(sq, b, hidden_size)``

        Returns:
            ``(output, bias)`` where ``output`` is ``(sq, b, hidden_size)``.
        """        
        if packed_seq_params is not None:
            raise NotImplementedError("NSA does not support packed_seq_params yet.")

        # ===== 1. Get Q, K, V ==============================================
        query, key, value = self.get_query_key_value_tensors(
            hidden_states, key_value_states
        )
        # query, key, value: (sq, b, num_q_heads_per_partition, D)

        sq, batch = query.size(0), query.size(1)
        device = query.device

        # ===== 1b. Optional RoPE on Q =====================================
        # The original NSA paper §3.2 uses intra-block position embeddings
        # instead of RoPE.  When rotary_pos_emb is provided we additionally
        # apply RoPE to Q (real positions) and compressed K (virtual positions)
        # to give the coarse attention soft distance awareness.
        k_freqs: Optional[Tensor] = None
        k_rope_bhsd: Optional[Tensor] = None
        if self.swa_window_size > 0 and rotary_pos_emb is None:
            raise RuntimeError(
                "NSA SWA requires rotary_pos_emb (for RoPE on SWA keys). "
                "Disable SWA or ensure this layer has RoPE enabled."
            )
        if rotary_pos_emb is not None:
            if not isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = (rotary_pos_emb, rotary_pos_emb)
            q_pos_emb, k_pos_emb = rotary_pos_emb
            query = apply_rotary_pos_emb(
                query, q_pos_emb, config=self.config)
            k_freqs = k_pos_emb                     # (sq, 1, 1, rot_dim)
            if self.swa_window_size > 0:
                key_rope = apply_rotary_pos_emb(
                    key, k_pos_emb, config=self.config)
                k_rope_bhsd = key_rope.permute(1, 2, 0, 3)

        # Convert to (batch, heads, seq, D) for attention maths
        q = query.permute(1, 2, 0, 3)  # (B, Hq, sq, D)  — RoPE already applied
        k = key.permute(1, 2, 0, 3)    # (B, Hkv, sq, D) — raw, goes to compression
        v = value.permute(1, 2, 0, 3)  # (B, Hkv, sq, D)

        # ===== 2. Attend =====================================================
        if self.swa_only:
            out = self._swa_only_attention(
                q, k_rope_bhsd, v, sq, device,
                attention_mask=attention_mask,
            )
        else:
            ck, cv = self._compress_kv(k, v, attention_mask=attention_mask)
            if self.swa_window_size > 0:
                out = self._compress_swa_attention(
                    q, ck, cv, k_rope_bhsd, v, sq, device,
                    attention_mask=attention_mask,
                )
            else:
                out = self._compress_only_attention(
                    q, ck, cv, sq, device,
                    attention_mask=attention_mask, k_freqs=k_freqs,
                )
        # out: (B, Hq, sq, D)

        # ===== 3. Output projection =========================================
        out = out.permute(2, 0, 1, 3).contiguous()  # (sq, B, Hq, D)
        out = out.view(sq, batch, -1)                 # (sq, B, Hq×D)

        output, bias = self.linear_proj(out)
        return output, bias

    def set_for_recompute_input_layernorm(self):
        """Placeholder (only needed for fp8)."""
        pass
