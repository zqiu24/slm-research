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
