"""pgpt renorm hook: embedding+lm_head rows go unit-norm; other matrices untouched."""

import torch
import torch.nn as nn

from src.patches.pgpt_optimizer_setup import _install_renorm_step


class _FakeOpt:
    def __init__(self):
        self.stepped = 0

    def step(self):
        self.stepped += 1


def test_renorm_step_projects_only_role_map_params():
    emb = nn.Parameter(torch.randn(10, 8) * 3.0)  # (vocab, hidden) rows
    head = nn.Parameter(torch.randn(10, 8) * 5.0)
    other = nn.Parameter(torch.randn(8, 8) * 7.0)  # a POET-wrapped matrix: untouched
    other_before = other.detach().clone()

    role_map = {emb: "rows", head: "rows"}
    opt = _FakeOpt()
    _install_renorm_step(opt, [role_map])

    opt.step()

    assert opt.stepped == 1
    assert torch.allclose(emb.data.norm(dim=1), torch.ones(10), atol=1e-5)
    assert torch.allclose(head.data.norm(dim=1), torch.ones(10), atol=1e-5)
    assert torch.equal(other.data, other_before)  # not in the role map -> unchanged


def test_install_is_idempotent():
    p = nn.Parameter(torch.randn(4, 4))
    opt = _FakeOpt()
    _install_renorm_step(opt, [{p: "rows"}])
    _install_renorm_step(opt, [{p: "rows"}])  # second call is a no-op
    opt.step()
    assert opt.stepped == 1
