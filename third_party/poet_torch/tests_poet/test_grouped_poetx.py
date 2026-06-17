import torch

from poet_torch.poetx_ops import _conj
from poet_torch.single_step import _blockdiag_skew_vec
from poet_torch.grouped_poetx_ops import _grouped_blockdiag_skew_vecs


def _ref_one(G, Wx, pin, pout, bs_in, bs_out, ri, ci, ro, co):
    M_in = _conj(G @ Wx, pin)
    M_out = _conj(Wx @ G, pout)
    return (_blockdiag_skew_vec(M_in, bs_in, ri, ci),
            _blockdiag_skew_vec(M_out, bs_out, ro, co))


def test_grouped_blockdiag_matches_per_expert():
    torch.manual_seed(0)
    E, in_f, out_f, b = 4, 8, 8, 4
    nb_in, nb_out = in_f // b, out_f // b
    ri, ci = torch.triu_indices(b, b, 1)
    ro, co = torch.triu_indices(b, b, 1)
    G = torch.randn(E, in_f, out_f, dtype=torch.float64)
    Wx = torch.randn(E, out_f, in_f, dtype=torch.float64)
    pin = torch.stack([torch.randperm(in_f) for _ in range(E)])
    pout = torch.stack([torch.randperm(out_f) for _ in range(E)])

    g_in, g_out = _grouped_blockdiag_skew_vecs(G, Wx, pin, pout, b, b, ri, ci, ro, co)
    for e in range(E):
        r_in, r_out = _ref_one(G[e], Wx[e], pin[e], pout[e], b, b, ri, ci, ro, co)
        assert torch.allclose(g_in[e], r_in, atol=1e-10)
        assert torch.allclose(g_out[e], r_out, atol=1e-10)


def test_grouped_blockdiag_matches_per_expert_asymmetric():
    torch.manual_seed(0)
    E, in_f, out_f = 3, 12, 8
    bs_in, bs_out = 4, 4
    nb_in, nb_out = in_f // bs_in, out_f // bs_out
    ri, ci = torch.triu_indices(bs_in, bs_in, 1)
    ro, co = torch.triu_indices(bs_out, bs_out, 1)
    G = torch.randn(E, in_f, out_f, dtype=torch.float64)
    Wx = torch.randn(E, out_f, in_f, dtype=torch.float64)
    pin = torch.stack([torch.randperm(in_f) for _ in range(E)])
    pout = torch.stack([torch.randperm(out_f) for _ in range(E)])

    g_in, g_out = _grouped_blockdiag_skew_vecs(G, Wx, pin, pout, bs_in, bs_out, ri, ci, ro, co)
    for e in range(E):
        r_in, r_out = _ref_one(G[e], Wx[e], pin[e], pout[e], bs_in, bs_out, ri, ci, ro, co)
        assert torch.allclose(g_in[e], r_in, atol=1e-10)
        assert torch.allclose(g_out[e], r_out, atol=1e-10)


def test_grouped_function_matches_per_expert_poetx():
    import torch
    from poet_torch.poetx_ops import POETXSingleStepFunction
    from poet_torch.grouped_poetx_ops import GroupedPOETXFunction

    torch.manual_seed(0)
    E, in_f, out_f, b = 3, 8, 8, 4
    ri, ci = torch.triu_indices(b, b, 1).to(torch.int32)
    ro, co = torch.triu_indices(b, b, 1).to(torch.int32)
    sizes = (2, 3, 4)
    Wx = torch.randn(E, out_f, in_f, dtype=torch.float64)
    pin = torch.stack([torch.randperm(in_f) for _ in range(E)]).to(torch.int32)
    pout = torch.stack([torch.randperm(out_f) for _ in range(E)]).to(torch.int32)

    # per-expert reference
    ref_y, ref_gin, ref_gout, xs = [], [], [], []
    for e in range(E):
        x = torch.randn(sizes[e], in_f, dtype=torch.float64, requires_grad=True)
        oin = torch.zeros(in_f // b, b * (b - 1) // 2, dtype=torch.float64, requires_grad=True)
        oout = torch.zeros(out_f // b, b * (b - 1) // 2, dtype=torch.float64, requires_grad=True)
        y = POETXSingleStepFunction.apply(x, oin, oout, Wx[e], None, pin[e], pout[e],
                                          ri, ci, ro, co, b, b)
        y.sum().backward()
        ref_y.append(y.detach()); ref_gin.append(oin.grad); ref_gout.append(oout.grad)
        xs.append(x.detach())

    # grouped
    cx = torch.cat(xs, 0).requires_grad_(True)
    oin = torch.zeros(E, in_f // b, b * (b - 1) // 2, dtype=torch.float64, requires_grad=True)
    oout = torch.zeros(E, out_f // b, b * (b - 1) // 2, dtype=torch.float64, requires_grad=True)
    gy = GroupedPOETXFunction.apply(cx, oin, oout, Wx, pin, pout, ri, ci, ro, co, b, b, sizes)
    gy.sum().backward()

    assert torch.allclose(gy.detach(), torch.cat(ref_y, 0), atol=1e-10)
    for e in range(E):
        assert torch.allclose(oin.grad[e], ref_gin[e], atol=1e-10)
        assert torch.allclose(oout.grad[e], ref_gout[e], atol=1e-10)


def test_grouped_module_matches_independent_poetx_linears():
    import torch
    from poet_torch import POETXLinear
    from poet_torch.grouped_poetx_layer import GroupedPOETXLinear

    torch.manual_seed(0)
    E, in_f, out_f, bc = 3, 8, 8, 2          # block_count=2 -> block_size 4
    sizes = (2, 3, 4)

    # Build E reference POETXLinears with known weights + perms.
    refs = []
    for e in range(E):
        pl = POETXLinear(in_features=in_f, out_features=out_f, block_count=bc,
                         bias=False, dtype=torch.float64, alternating=True)
        pl.weight.data.copy_(torch.randn(out_f, in_f, dtype=torch.float64))
        pl.bake_perms_into_weight()
        refs.append(pl)

    g = GroupedPOETXLinear(E, in_f, out_f, block_count=bc, alternating=True,
                           alternate_every=1, dtype=torch.float64)
    # mirror each ref into the grouped module's experts, then bind the buffer.
    for e in range(E):
        g.experts[e].weight.data.copy_(refs[e].weight)
        for buf in ("perm_in", "perm_in_inv", "perm_out", "perm_out_inv"):
            getattr(g.experts[e], buf).copy_(getattr(refs[e], buf))
    g.bind_weights()

    # set nonzero oft_R identically on both sides
    for e in range(E):
        gi = torch.randn_like(g.experts[e].oft_R_in) * 0.1
        go = torch.randn_like(g.experts[e].oft_R_out) * 0.1
        g.experts[e].oft_R_in.data.copy_(gi); g.experts[e].oft_R_out.data.copy_(go)
        refs[e].oft_R_in.data.copy_(gi); refs[e].oft_R_out.data.copy_(go)

    # forward + backward parity
    xs = [torch.randn(sizes[e], in_f, dtype=torch.float64) for e in range(E)]
    ref_y = [refs[e](xs[e].clone().requires_grad_(True)) for e in range(E)]
    cx = torch.cat([x.clone() for x in xs], 0).requires_grad_(True)
    gy = g(cx, torch.tensor(sizes))
    assert torch.allclose(gy, torch.cat([y.detach() for y in ref_y], 0), atol=1e-9)

    gy.sum().backward()
    for e in range(E):
        x = xs[e].clone().requires_grad_(True)
        refs[e].oft_R_in.grad = None; refs[e].oft_R_out.grad = None
        refs[e](x).sum().backward()
        assert torch.allclose(g.experts[e].oft_R_in.grad, refs[e].oft_R_in.grad, atol=1e-9)
        assert torch.allclose(g.experts[e].oft_R_out.grad, refs[e].oft_R_out.grad, atol=1e-9)

    # merge parity (active-only fold; alternating=True). Inject pure-torch cayley so
    # the fold runs on the CPU dev box (the default Triton op raises "0 active drivers").
    from poet_torch.alt_state import active_side
    from poet_torch.poet_layer import cayley_batch
    active = active_side(1)
    w_before = g.weight.clone()
    g._fold_active_side(active, reinit_perm=False, cayley_fn=cayley_batch)
    for e in range(E):
        refs[e]._fold_active_side(active, reinit_perm=False, cayley_fn=cayley_batch)
        assert torch.allclose(g.weight[e], refs[e].weight, atol=1e-9)
    assert not torch.allclose(g.weight, w_before)        # something actually folded
