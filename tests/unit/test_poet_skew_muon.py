import torch

from src.diag.skew_conditioning import block_spectral_stats, vec_to_skew
from src.optim.poet_skew_muon import SkewMuon, orthogonalize_skew_blocks


def test_ns_flattens_the_skew_spectrum():
    """Newton-Schulz democratizes the skew gradient's spectrum: a heavy-tailed
    (high condition number) input becomes near-uniform (condition ~1). NS
    *preserves rank* (it equalizes the nonzero singular values but cannot
    resurrect near-zero ones in a few steps), so the robust signal is the
    condition number collapsing — assert that, on a FULL-RANK heavy-tailed input.
    (stable rank also rises here, but it would NOT for a genuinely low-rank input,
    which is why condition number is the primary assertion.)"""
    torch.manual_seed(0)
    b, num_blocks = 16, 2
    ne = b * (b - 1) // 2
    v = torch.randn(num_blocks, ne)
    v[:, :3] *= 6.0  # full-rank but heavy-tailed (a few dominant directions)
    Q = vec_to_skew(v, b)
    s_in = block_spectral_stats(Q)
    X = orthogonalize_skew_blocks(Q.float(), ns_steps=5)
    X = (X - X.transpose(-2, -1)) / 2  # re-skew
    s_out = block_spectral_stats(X)
    cond_in = s_in["condition_number"].mean().item()
    cond_out = s_out["condition_number"].mean().item()
    assert cond_in > 10.0  # heavy-tailed input
    assert cond_out < 5.0  # NS democratized -> near-uniform spectrum
    assert cond_out < cond_in / 5.0  # a large drop
    assert s_out["stable_rank"].mean() > s_in["stable_rank"].mean()  # energy spread


def test_constant_angle_scaling_hits_theta():
    b, ne = 8, 8 * 7 // 2
    p = torch.nn.Parameter(torch.zeros(1, ne))
    p.grad = torch.randn(1, ne)
    opt = SkewMuon(skew_params=[p], adamw_params=[], theta=0.3, ns_steps=5, momentum=0.0)
    opt.step()
    # the realized skew step ||step||_F per block == theta (oft_R moved from 0 by -step)
    from src.diag.skew_conditioning import vec_to_skew as _v

    step_fro = torch.linalg.matrix_norm(_v(-p.data, b), ord="fro", dim=(-2, -1))
    assert torch.allclose(step_fro, torch.full_like(step_fro, 0.3), atol=1e-4)


def test_adamw_branch_steps_non_skew_params():
    w = torch.nn.Parameter(torch.randn(4, 4))
    w.grad = torch.randn(4, 4)
    w0 = w.data.clone()
    opt = SkewMuon(skew_params=[], adamw_params=[w], theta=0.3, adamw_lr=1e-2)
    opt.step()
    assert not torch.allclose(w.data, w0)  # moved
    assert opt.state[w]["use_skew"] is False


def test_skew_param_stays_a_valid_skew_vector():
    ne = 8 * 7 // 2
    p = torch.nn.Parameter(torch.randn(3, ne) * 0.1)
    p.grad = torch.randn(3, ne)
    opt = SkewMuon(skew_params=[p], adamw_params=[], theta=0.2, ns_steps=5, momentum=0.95)
    opt.step()
    assert p.data.shape == (3, ne)  # still the (n_blocks, n_elems) skew-vector layout
    assert torch.isfinite(p.data).all()
    assert opt.state[p]["use_skew"] is True
    assert "momentum_buffer" in opt.state[p]


def test_split_skew_vs_adamw_by_name():
    """oft_R params -> skew branch, everything else -> adamw (pure split logic)."""
    import torch.nn as nn

    from src.optim.poet import _split_poet_muon_params

    class FakeChunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Module()
            self.layer.oft_R_in = nn.Parameter(torch.zeros(1, 6))
            self.layer.oft_R_out = nn.Parameter(torch.zeros(1, 6))
            self.embedding = nn.Parameter(torch.zeros(8, 8))  # non-oft_R

    chunk = FakeChunk()
    skew, adamw = _split_poet_muon_params([chunk])
    assert len(skew) == 2  # oft_R_in, oft_R_out
    assert len(adamw) == 1  # embedding


def test_muon_update_spectral_stats_flattens_condition_number():
    """The SkewMuon UPDATE spectrum (NS-orthogonalize -> re-skew, exactly what
    SkewMuon.step applies) is well-conditioned (cond ~1) even when the raw skew
    gradient is heavy-tailed. This is the metric that makes Muon's preconditioning
    visible: poet_cond reads the RAW gradient (stays heavy-tailed), so it can never
    show what Muon does to the update. Mirrors test_ns_flattens_the_skew_spectrum
    but for the packaged helper + its dict return."""
    from src.diag.skew_conditioning import block_spectral_stats, vec_to_skew
    from src.optim.poet_skew_muon import muon_update_spectral_stats

    torch.manual_seed(0)
    b, ne = 16, 16 * 15 // 2
    v = torch.randn(2, ne)
    v[:, :3] *= 6.0  # full-rank but heavy-tailed
    Q = vec_to_skew(v, b)
    cond_raw = block_spectral_stats(Q)["condition_number"].mean().item()
    stats = muon_update_spectral_stats(Q, ns_steps=5)
    cond_update = stats["condition_number"].mean().item()
    assert cond_raw > 10.0  # raw gradient heavy-tailed (the conditioning problem)
    assert cond_update < 5.0  # Muon update spectrum flattened (~1)
    assert set(stats) >= {"condition_number", "stable_rank", "sigma_max_over_median"}
