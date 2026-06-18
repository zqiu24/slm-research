"""Pure-PyTorch PGPTBlock parity vs the vendored reference Block (use_nGPT=1).

pgpt's forward is byte-identical to nGPT's, so the same reference oracle applies.
"""

import torch

from src.model.pgpt.block import PGPTBlock
from tests._fixtures.pgpt_reference.model import Block as RefBlock
from tests._fixtures.pgpt_reference.model import GPTConfig

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


def test_pgpt_block_matches_reference_at_init():
    torch.manual_seed(123)
    cfg = _ref_config()
    ref = RefBlock(cfg, iblock=0).float().to(_DEVICE)
    ours = PGPTBlock(
        hidden_size=cfg.n_embd,
        num_heads=cfg.n_head,
        ffn_hidden_size=4 * cfg.n_embd,
        base_scale=cfg.base_scale,
        dtype=torch.float32,
    ).to(_DEVICE)
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

    x = torch.randn(1, 8, cfg.n_embd, device=_DEVICE)
    ours.eval()
    ref.eval()
    with torch.no_grad():
        y_ours = ours(x)
        y_ref = ref(x).float()
    assert torch.allclose(
        y_ours, y_ref, atol=2e-3, rtol=2e-3
    ), f"max abs diff = {(y_ours - y_ref).abs().max().item()}"


def test_pgpt_block_residual_is_unit_norm_per_token():
    cfg = _ref_config(n_embd=32, n_head=4)
    blk = PGPTBlock(
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
