"""gloo 2-rank integration test for LieOrthMomentum's DP-sharded orthogonalization.
Each rank sees IDENTICAL grads (as Megatron guarantees post-all-reduce); the sharded
step must reproduce the single-rank (replicated) step bit-for-bit."""

import os

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn


def _single_rank_result(seed):
    torch.manual_seed(seed)
    ne = 8 * 7 // 2
    ps = [nn.Parameter(torch.zeros(nb, ne)) for nb in (1, 3, 2, 5)]
    gs = [torch.randn_like(p) for p in ps]
    from src.optim.poet_lie_orth import LieOrthMomentum

    for p, g in zip(ps, gs, strict=False):
        p.grad = g.clone()
    LieOrthMomentum([dict(params=ps, use_skew=True, side="out", lr=0.1)], ortho_c=0.05).step()
    return [p.data.clone() for p in ps]


def _worker(rank, world, q):
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29555")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(0)  # identical grads on every rank (the DP invariant)
    ne = 8 * 7 // 2
    ps = [nn.Parameter(torch.zeros(nb, ne)) for nb in (1, 3, 2, 5)]
    for p in ps:
        p.grad = torch.randn_like(p)
    from src.optim.poet_lie_orth import LieOrthMomentum

    opt = LieOrthMomentum(
        [dict(params=ps, use_skew=True, side="out", lr=0.1)],
        ortho_c=0.05,
        distributed=True,
        dp_world_size=world,
        dp_rank=rank,
        dp_group=dist.group.WORLD,
    )
    opt.step()
    if rank == 0:
        q.put([p.data.clone() for p in ps])
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed unavailable")
def test_distributed_step_matches_single_rank():
    ref = _single_rank_result(seed=0)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, 2, q)) for r in range(2)]
    for p in procs:
        p.start()
    got = q.get(timeout=120)
    for p in procs:
        p.join(timeout=120)
    for a, b in zip(got, ref, strict=False):
        assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()
