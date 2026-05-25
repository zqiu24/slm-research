"""Pure-PyTorch NGPTBlock parity vs the vendored reference Block (use_nGPT=1)."""

import torch

from src.model.ngpt.layer import NGPTBlock
from tests._fixtures.ngpt_reference.model import Block as RefBlock
from tests._fixtures.ngpt_reference.model import GPTConfig

# The reference Block uses flash_attn when it is installed; flash_attn
# requires a CUDA device.  Fall back to CPU only when flash_attn is absent
# (i.e., the pure-SDPA fallback path in the fixture is active).
try:
    import flash_attn  # noqa: F401

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _DEVICE = "cpu"


def _ref_config(n_embd=64, n_head=4, vocab_size=100):
    return GPTConfig(
        block_size=32,
        vocab_size=vocab_size,
        n_layer=2,
        n_head=n_head,
        n_embd=n_embd,
        base_scale=1.0 / (n_embd**0.5),
        use_nGPT=1,
        dropout=0.0,
        bias=False,
    )


def test_ngpt_block_matches_reference_at_init():
    torch.manual_seed(123)
    cfg = _ref_config()
    ref = RefBlock(cfg, iblock=0).float().to(_DEVICE)
    ours = NGPTBlock(
        hidden_size=cfg.n_embd,
        num_heads=cfg.n_head,
        ffn_hidden_size=4 * cfg.n_embd,
        base_scale=cfg.base_scale,
        dtype=torch.float32,
    ).to(_DEVICE)
    # Copy reference weights into ours (same shapes / convention).
    with torch.no_grad():
        ours.query.weight.copy_(ref.query.weight)
        ours.key.weight.copy_(ref.key.weight)
        ours.value.weight.copy_(ref.value.weight)
        ours.att_c_proj.weight.copy_(ref.att_c_proj.weight)
        ours.c_fc.weight.copy_(ref.c_fc.weight)
        ours.mlp_c_proj.weight.copy_(ref.mlp_c_proj.weight)
        ours.sqk.param.copy_(ref.sqk)
        ours.suv.param.copy_(ref.suv)
        ours.attn_alpha.param.copy_(ref.attn_alpha)
        ours.mlp_alpha.param.copy_(ref.mlp_alpha)

    x = torch.randn(1, 8, cfg.n_embd, device=_DEVICE)  # (B, T, C) matches reference
    ours.eval()
    ref.eval()
    with torch.no_grad():
        y_ours = ours(x)
        y_ref = ref(x).float()
    # Both blocks now do attention in bf16 internally (see NGPTBlock._attn),
    # so the only remaining gap is upstream SDPA vs flash_attn rounding.
    assert torch.allclose(
        y_ours, y_ref, atol=2e-3, rtol=2e-3
    ), f"max abs diff = {(y_ours - y_ref).abs().max().item()}"


def test_ngpt_block_residual_is_unit_norm_per_token():
    """After the second hypersphere blend the residual lies on S^{C-1}."""
    cfg = _ref_config(n_embd=32, n_head=4)
    blk = NGPTBlock(
        hidden_size=cfg.n_embd,
        num_heads=cfg.n_head,
        ffn_hidden_size=4 * cfg.n_embd,
        base_scale=cfg.base_scale,
        dtype=torch.float32,
    )
    x = torch.randn(2, 4, cfg.n_embd)
    y = blk(x)
    norms = y.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
