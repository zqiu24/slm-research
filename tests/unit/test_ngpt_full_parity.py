"""Full-model parity vs the vendored NVIDIA reference at a toy config.

We assemble our own minimal nGPT model from primitives (token
embedding -> N x NGPTBlock -> lm_head -> sz) and compare to the
reference GPT(use_nGPT=1). The reference uses flash_attn (CUDA-only),
so this runs on CUDA when flash_attn is installed (matching
test_ngpt_layer_block_forward.py).
"""

import math

import torch
import torch.nn as nn

from src.model.ngpt.block import NGPTBlock
from src.model.ngpt.normalize import normalize_module_matrices
from src.model.ngpt.scaling_params import LearnedScaling
from tests._fixtures.ngpt_reference.model import GPT as RefGPT  # noqa: N811
from tests._fixtures.ngpt_reference.model import GPTConfig

try:
    import flash_attn  # noqa: F401

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _DEVICE = "cpu"


def _build_role_map(model, n_layer):
    role_map = {}
    role_map[model.wte.weight] = "rows"
    role_map[model.lm_head.weight] = "rows"
    for i in range(n_layer):
        b = model.blocks[i]
        role_map[b.query.weight] = "rows"
        role_map[b.key.weight] = "rows"
        role_map[b.value.weight] = "rows"
        role_map[b.att_c_proj.weight] = "cols"
        role_map[b.c_fc.weight] = "rows"
        role_map[b.mlp_c_proj.weight] = "cols"
    return role_map


class _OurNGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd, dtype=torch.float32)
        self.blocks = nn.ModuleList(
            [
                NGPTBlock(
                    hidden_size=cfg.n_embd,
                    num_heads=cfg.n_head,
                    ffn_hidden_size=4 * cfg.n_embd,
                    base_scale=cfg.base_scale,
                    dtype=torch.float32,
                )
                for _ in range(cfg.n_layer)
            ]
        )
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False, dtype=torch.float32)
        self.sz = LearnedScaling((cfg.vocab_size,), init_value=1.0, init_scaling=cfg.base_scale)
        # Initialize all 2D weights as in reference: normal_(0, base_scale)
        with torch.no_grad():
            nn.init.normal_(self.wte.weight, mean=0.0, std=cfg.base_scale)
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=cfg.base_scale)
            for b in self.blocks:
                for lin in (b.query, b.key, b.value, b.att_c_proj, b.c_fc, b.mlp_c_proj):
                    nn.init.normal_(lin.weight, mean=0.0, std=cfg.base_scale)

    def forward(self, idx):
        x = self.wte(idx)
        for b in self.blocks:
            x = b(x)
        logits = self.lm_head(x)
        sz_eff = self.sz.scaled_value()
        return sz_eff * logits


def _copy_ref_to_ours(ref: RefGPT, ours: _OurNGPT, cfg: GPTConfig):
    with torch.no_grad():
        ours.wte.weight.copy_(ref.transformer.wte.weight.float())
        ours.lm_head.weight.copy_(ref.lm_head.weight.float())
        ours.sz.param.copy_(ref.sz)
        for i in range(cfg.n_layer):
            rb = ref.transformer.h[i]
            ob = ours.blocks[i]
            ob.query.weight.copy_(rb.query.weight.float())
            ob.key.weight.copy_(rb.key.weight.float())
            ob.value.weight.copy_(rb.value.weight.float())
            ob.att_c_proj.weight.copy_(rb.att_c_proj.weight.float())
            ob.c_fc.weight.copy_(rb.c_fc.weight.float())
            ob.mlp_c_proj.weight.copy_(rb.mlp_c_proj.weight.float())
            ob.sqk.param.copy_(rb.sqk)
            ob.suv.param.copy_(rb.suv)
            ob.attn_alpha.param.copy_(rb.attn_alpha)
            ob.mlp_alpha.param.copy_(rb.mlp_alpha)


def test_full_model_logit_parity_at_init():
    torch.manual_seed(7)
    cfg = GPTConfig(
        block_size=16,
        vocab_size=37,
        n_layer=2,
        n_head=4,
        n_embd=32,
        base_scale=1.0 / math.sqrt(32),
        use_nGPT=1,
        dropout=0.0,
        bias=False,
    )
    ref = RefGPT(cfg).float().to(_DEVICE)
    ours = _OurNGPT(cfg).to(_DEVICE)
    _copy_ref_to_ours(ref, ours, cfg)

    # Normalize like the reference does at init (train.py:411).
    normalize_module_matrices(_build_role_map(ours, cfg.n_layer))

    # Also re-copy normalized weights back to ref so both start in the
    # same state.
    with torch.no_grad():
        ref.transformer.wte.weight.copy_(ours.wte.weight)
        ref.lm_head.weight.copy_(ours.lm_head.weight)
        for i in range(cfg.n_layer):
            rb = ref.transformer.h[i]
            ob = ours.blocks[i]
            rb.query.weight.copy_(ob.query.weight.to(rb.query.weight.dtype))
            rb.key.weight.copy_(ob.key.weight.to(rb.key.weight.dtype))
            rb.value.weight.copy_(ob.value.weight.to(rb.value.weight.dtype))
            rb.att_c_proj.weight.copy_(ob.att_c_proj.weight.to(rb.att_c_proj.weight.dtype))
            rb.c_fc.weight.copy_(ob.c_fc.weight.to(rb.c_fc.weight.dtype))
            rb.mlp_c_proj.weight.copy_(ob.mlp_c_proj.weight.to(rb.mlp_c_proj.weight.dtype))

    idx = torch.randint(0, cfg.vocab_size, (1, 8), device=_DEVICE)
    ours.eval()
    ref.eval()
    with torch.no_grad():
        ours_logits = ours(idx)
        # reference returns (logits, loss); request loss by passing targets so it
        # runs the full lm_head path (matches our forward).
        ref_logits, _ = ref(idx, targets=idx)
    assert ours_logits.shape == ref_logits.shape
    # bf16 weights inside ref dominate; loose tolerance but functionally equivalent.
    abs_diff = (ours_logits - ref_logits.float()).abs().max().item()
    rel = (ours_logits - ref_logits.float()).abs().max().item() / max(
        1e-6, ref_logits.float().abs().max().item()
    )
    assert (
        abs_diff < 5e-2 or rel < 1e-2
    ), f"logit parity failed: max abs diff = {abs_diff}, rel = {rel}"
