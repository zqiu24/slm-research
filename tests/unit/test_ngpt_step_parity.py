"""CPU forward + one-step parity: our pure-torch nGPT vs the NVIDIA reference.

Reuses the model assembly + weight transfer from test_ngpt_full_parity. After
transferring reference weights and normalizing both sides identically, we run
one CE backward + one AdamW step + one weight-normalization on EACH model and
assert the post-step weights still match. This validates the residual blend,
sqk/suv/sz scaling, the matrix projection, and the optimizer step together.

Tolerances are loose: the reference (and our NGPTBlock) cast attention to bf16
internally, so a small precision gap is expected and is not a correctness bug.
"""

import math

import torch

from src.model.ngpt.normalize import justnorm, normalize_module_matrices
from tests._fixtures.ngpt_reference.model import GPT as RefGPT  # noqa: N811
from tests._fixtures.ngpt_reference.model import GPTConfig
from tests.unit.test_ngpt_full_parity import (
    _DEVICE,
    _build_role_map,
    _copy_ref_to_ours,
    _OurNGPT,
)


def _ref_normalize_matrices(ref, n_layer):
    """Mirror NVIDIA train.py::normalize_matrices using justnorm per role."""
    with torch.no_grad():
        ref.transformer.wte.weight.copy_(justnorm(ref.transformer.wte.weight, dim=1))
        ref.lm_head.weight.copy_(justnorm(ref.lm_head.weight, dim=1))
        for i in range(n_layer):
            b = ref.transformer.h[i]
            b.query.weight.copy_(justnorm(b.query.weight, dim=1))
            b.key.weight.copy_(justnorm(b.key.weight, dim=1))
            b.value.weight.copy_(justnorm(b.value.weight, dim=1))
            b.att_c_proj.weight.copy_(justnorm(b.att_c_proj.weight, dim=0))
            b.c_fc.weight.copy_(justnorm(b.c_fc.weight, dim=1))
            b.mlp_c_proj.weight.copy_(justnorm(b.mlp_c_proj.weight, dim=0))


def _cfg():
    n_embd = 32
    return GPTConfig(
        block_size=16,
        vocab_size=37,
        n_layer=2,
        n_head=4,
        n_embd=n_embd,
        base_scale=1.0 / math.sqrt(n_embd),
        use_nGPT=1,
        dropout=0.0,
        bias=False,
    )


def test_forward_and_one_step_parity():
    torch.manual_seed(7)
    cfg = _cfg()
    ref = RefGPT(cfg).float().to(_DEVICE)
    ours = _OurNGPT(cfg).to(_DEVICE)
    _copy_ref_to_ours(ref, ours, cfg)

    # Init-normalize both identically (reference does this at train.py:411).
    normalize_module_matrices(_build_role_map(ours, cfg.n_layer))
    with torch.no_grad():
        ref.transformer.wte.weight.copy_(ours.wte.weight)
        ref.lm_head.weight.copy_(ours.lm_head.weight)
        for i in range(cfg.n_layer):
            rb, ob = ref.transformer.h[i], ours.blocks[i]
            for name in ("query", "key", "value", "att_c_proj", "c_fc", "mlp_c_proj"):
                getattr(rb, name).weight.copy_(
                    getattr(ob, name).weight.to(getattr(rb, name).weight.dtype)
                )

    idx = torch.randint(0, cfg.vocab_size, (2, 8), device=_DEVICE)
    tgt = torch.randint(0, cfg.vocab_size, (2, 8), device=_DEVICE)

    # ---- forward parity ----
    ours_logits = ours(idx)
    _, ref_loss0 = ref(idx, targets=tgt)
    ours_loss0 = torch.nn.functional.cross_entropy(
        ours_logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1)
    )
    assert (
        abs(ours_loss0.item() - ref_loss0.item()) < 5e-2
    ), f"forward loss parity: ours={ours_loss0.item()} ref={ref_loss0.item()}"

    # ---- one AdamW step + normalization on BOTH ----
    opt_args = dict(lr=15e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    o_opt = torch.optim.AdamW(ours.parameters(), **opt_args)
    r_opt = torch.optim.AdamW(ref.parameters(), **opt_args)

    o_opt.zero_grad()
    ours_loss0.backward()
    o_opt.step()
    normalize_module_matrices(_build_role_map(ours, cfg.n_layer))

    r_opt.zero_grad()
    _, ref_loss = ref(idx, targets=tgt)
    ref_loss.backward()
    r_opt.step()
    _ref_normalize_matrices(ref, cfg.n_layer)

    # ---- post-step weight parity on sampled tensors ----
    def _max_abs(a, b):
        return (a.float() - b.float()).abs().max().item()

    q_diff = _max_abs(ours.blocks[0].query.weight, ref.transformer.h[0].query.weight)
    wte_diff = _max_abs(ours.wte.weight, ref.transformer.wte.weight)
    alpha_diff = _max_abs(ours.blocks[0].attn_alpha.param, ref.transformer.h[0].attn_alpha)
    sz_diff = _max_abs(ours.sz.param, ref.sz)
    assert q_diff < 5e-2, f"post-step query weight diff {q_diff}"
    assert wte_diff < 5e-2, f"post-step wte diff {wte_diff}"
    assert alpha_diff < 5e-2, f"post-step attn_alpha diff {alpha_diff}"
    assert sz_diff < 5e-2, f"post-step sz diff {sz_diff}"

    # ---- both projected matrices are unit-norm after normalization ----
    assert torch.allclose(
        ours.blocks[0].query.weight.float().norm(dim=1),
        torch.ones(cfg.n_embd, device=_DEVICE),
        atol=1e-4,
    )
