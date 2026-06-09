"""Single-layer parity: Megatron NGPTTransformerLayer vs reference Block.

Runs on 1 GPU. We build ONE nGPT layer from the production spec, transfer a
reference Block's weights into it, feed an identical hidden state, and compare
outputs. Two tests:
  (A) RoPE OFF on both sides — isolates the nGPT math (residual blend, sqk/suv
      scaling, fused-qkv interleaving, sqrt(head_dim) softmax scale). The correct
      transfer matches to ~5e-5, so this asserts a tight 1e-3 bound — tight enough
      to catch a wrong fused-qkv interleaving (~2.5e-2 here), which a loose 5e-2
      bound would miss because the lr~0.05 residual blend heavily damps branch errors.
  (B) RoPE matched — Megatron's interleaved RoPE (base 10000) does NOT reproduce
      the reference's bespoke sinusoidal RoPE (~1.5e-2 residual, vs ~5e-5 with RoPE
      off), so this is an `xfail`: a *documented deviation*, not a fake pass. The
      nGPT math is validated by (A); production nGPT simply uses Megatron's standard
      RoPE. See docs/experiments/ngpt.md.

The reference (NVIDIA nGPT `Block`) runs attention through `flash_attn_func`
with q/k/v cast to bf16; the Megatron layer runs `DotProductAttention` in fp32.
That backend/precision gap (damped by the residual blend) is the ~5e-5 floor.
"""

import math

import pytest
import torch

pytestmark = [pytest.mark.gpu]

import tests._fixtures.ngpt_reference.model as refmod  # noqa: E402
from tests._fixtures.ngpt_reference.model import Block as RefBlock  # noqa: E402
from tests._fixtures.ngpt_reference.model import GPTConfig  # noqa: E402

_HIDDEN, _HEADS, _FFN = 64, 4, 256
_HEAD_DIM = _HIDDEN // _HEADS
_BASE_SCALE = 1.0 / math.sqrt(_HIDDEN)


@pytest.fixture(scope="module")
def _megatron_dist():
    """Single-process Megatron model-parallel context (tp=1), like test_poet_layers."""
    import os

    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29577")
    created_pg = not dist.is_initialized()
    if created_pg:
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    ps.initialize_model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(0)
    try:
        yield
    finally:
        ps.destroy_model_parallel()
        if created_pg:
            dist.destroy_process_group()


def _build_megatron_ngpt_layer():
    from megatron.core.transformer.spec_utils import build_module
    from megatron.core.transformer.transformer_config import TransformerConfig

    from src.specs.ngpt_layer_spec import build_ngpt_layer_spec

    config = TransformerConfig(
        num_layers=1,
        hidden_size=_HIDDEN,
        num_attention_heads=_HEADS,
        ffn_hidden_size=_FFN,
        kv_channels=_HEAD_DIM,
        num_query_groups=_HEADS,  # MHA, matches nGPT
        add_bias_linear=False,
        gated_linear_unit=True,  # SwiGLU
        activation_func=torch.nn.functional.silu,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        bf16=False,
        params_dtype=torch.float32,
        recompute_granularity=None,
    )
    # Fields the nGPT layer/spec read off config (normally stamped by the
    # ngpt_apply_spec patch). Set explicitly for the standalone test.
    config.ngpt_base_scale = _BASE_SCALE
    config.ngpt_alpha_init = 0.05
    config.ngpt_sqk_init = 1.0
    config.ngpt_suv_init = 1.0
    config.softmax_scale = math.sqrt(_HEAD_DIM)  # nGPT: sqrt(head_dim)

    spec = build_ngpt_layer_spec(config)
    layer = build_module(spec, config=config, layer_number=1).cuda().float()
    return layer


def _fused_qkv_from_ref(ref):
    """Interleave reference q/k/v into Megatron's fused linear_qkv layout.

    linear_qkv.weight is (3*hidden, hidden); for MHA (num_query_groups == heads)
    the rows are grouped per head as [q(head_dim), k(head_dim), v(head_dim)].
    """
    q = ref.query.weight.float()
    k = ref.key.weight.float()
    v = ref.value.weight.float()
    rows = []
    for i in range(_HEADS):
        sl = slice(i * _HEAD_DIM, (i + 1) * _HEAD_DIM)
        rows.extend((q[sl], k[sl], v[sl]))
    return torch.cat(rows, dim=0)


def _transfer(ref, layer):
    """Copy reference Block weights into the Megatron nGPT layer (fp32)."""
    sa = layer.self_attention
    with torch.no_grad():
        sa.linear_qkv.weight.copy_(_fused_qkv_from_ref(ref))
        sa.linear_proj.weight.copy_(ref.att_c_proj.weight.float())
        sa.q_layernorm.sqk.param.copy_(ref.sqk.float())
        sa.k_layernorm.sqk.param.copy_(ref.sqk.float())
        layer.mlp.linear_fc1.weight.copy_(ref.c_fc.weight.float())
        layer.mlp.linear_fc2.weight.copy_(ref.mlp_c_proj.weight.float())
        layer.mlp.suv.param.copy_(ref.suv.float())
        layer.attn_alpha.param.copy_(ref.attn_alpha.float())
        layer.mlp_alpha.param.copy_(ref.mlp_alpha.float())


def _ref_cfg():
    return GPTConfig(
        block_size=16,
        vocab_size=37,
        n_layer=1,
        n_head=_HEADS,
        n_embd=_HIDDEN,
        base_scale=_BASE_SCALE,
        use_nGPT=1,
        dropout=0.0,
        bias=False,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs 1 GPU")
def test_megatron_layer_matches_reference_block_no_rope(_megatron_dist, monkeypatch):
    """RoPE OFF on both sides: isolates the nGPT math from position encoding."""
    torch.manual_seed(0)
    # Disable RoPE in the reference by making its rotary application the identity.
    monkeypatch.setattr(refmod, "apply_rotary_position_embeddings", lambda pos, q, k: (q, k))

    ref = RefBlock(_ref_cfg(), iblock=0).float().cuda()
    layer = _build_megatron_ngpt_layer()
    _transfer(ref, layer)

    s, b = 8, 1
    h_sbh = torch.randn(s, b, _HIDDEN, device="cuda")
    h_bsh = h_sbh.transpose(0, 1).contiguous()

    out_m, _ = layer(h_sbh, attention_mask=None, rotary_pos_emb=None)
    out_r = ref(h_bsh).transpose(0, 1)
    diff = (out_m.float() - out_r.float()).abs().max().item()
    # Correct transfer matches to ~5e-5; a wrong fused-qkv interleaving measures
    # ~2.5e-2. 1e-3 clears the bf16 attention floor (~15x margin) yet still fails
    # on a layout regression that a loose 5e-2 bound would silently accept.
    assert diff < 1e-3, f"single-layer (no-RoPE) parity diff = {diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs 1 GPU")
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Megatron's interleaved RoPE (base 10000) does not reproduce the reference's "
        "bespoke get_sinusoidal_embeddings/apply_rotary_position_embeddings convention "
        "(~1.5e-2 single-layer residual, vs ~5e-5 with RoPE off; interleaved=False is no "
        "better at ~1.2e-2). The nGPT math is validated by the no-RoPE test; production "
        "nGPT uses Megatron's standard RoPE. Documented in docs/experiments/ngpt.md. "
        "strict=True so that if a future change aligns the conventions this flips to a "
        "failure prompting a tolerance/marker update."
    ),
)
def test_megatron_layer_matches_reference_block_rope_matched(_megatron_dist):
    """Documents the RoPE convention deviation: Megatron interleaved RoPE vs reference sinusoidal.

    Asserted at the SAME tight 1e-3 bound as the no-RoPE test, so the xfail records a
    genuine non-match (not a loose-tolerance fake pass).
    """
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

    torch.manual_seed(0)
    ref = RefBlock(_ref_cfg(), iblock=0).float().cuda()
    layer = _build_megatron_ngpt_layer()
    _transfer(ref, layer)

    s, b = 8, 1
    h_sbh = torch.randn(s, b, _HIDDEN, device="cuda")
    h_bsh = h_sbh.transpose(0, 1).contiguous()

    rotary = RotaryEmbedding(
        kv_channels=_HEAD_DIM, rotary_percent=1.0, rotary_interleaved=True, rotary_base=10000
    )
    rotary_pos_emb = rotary(s)

    out_m, _ = layer(h_sbh, attention_mask=None, rotary_pos_emb=rotary_pos_emb)
    out_r = ref(h_bsh).transpose(0, 1)  # reference applies its own RoPE
    diff = (out_m.float() - out_r.float()).abs().max().item()
    assert diff < 1e-3, f"single-layer (RoPE-matched) parity diff = {diff}"
