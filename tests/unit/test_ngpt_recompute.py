"""nGPT forward is recompute-safe: wrapping it in activation checkpointing
yields identical output AND gradients.

This is the property Megatron's `recompute_granularity='full'` relies on —
it re-runs the whole layer forward in the backward pass. The nGPT forward is a
pure function of its inputs (no dropout, no in-place state), so checkpointing
must be a no-op on the math. The Megatron-specific `tensor_parallel.checkpoint`
path is GPU-only; this proves the underlying invariant on CPU via
`torch.utils.checkpoint`.
"""

import torch
import torch.utils.checkpoint as ckpt

from src.model.ngpt.block import NGPTBlock


def test_ngpt_block_checkpoint_parity_output_and_grads():
    torch.manual_seed(0)
    block = NGPTBlock(
        hidden_size=16,
        num_heads=2,
        ffn_hidden_size=32,
        base_scale=0.25,
        dtype=torch.float32,
    )
    params = list(block.parameters())

    x = torch.randn(2, 4, 16, requires_grad=True)
    out_direct = block(x)
    g_direct = torch.autograd.grad(out_direct.square().sum(), [x, *params])

    x2 = x.detach().clone().requires_grad_(True)
    out_ck = ckpt.checkpoint(block, x2, use_reentrant=False)
    g_ck = torch.autograd.grad(out_ck.square().sum(), [x2, *params])

    assert torch.allclose(out_direct, out_ck, atol=1e-6)
    assert len(g_direct) == len(g_ck)
    for gd, gc in zip(g_direct, g_ck, strict=False):
        assert torch.allclose(gd, gc, atol=1e-5)
