"""pgpt renorm hook: embedding+lm_head rows go unit-norm; other matrices untouched."""

import types

import torch
import torch.nn as nn

from src.patches.pgpt_optimizer_setup import _install_renorm_step, _unwrapped_chunks


class _Wrap:
    """Mimics Float16Module/DDP: holds ``.module``, no ``__getattr__`` delegation."""

    def __init__(self, inner):
        self.module = inner


def _peel(m):
    """Pure stand-in for ``megatron.core.utils.unwrap_model``."""
    while hasattr(m, "module"):
        m = m.module
    return m


def test_unwrapped_chunks_reaches_inner_role_map():
    # Regression (review HIGH finding): the role map lives on the inner GPTModel,
    # but setup_model_and_optimizer hands back DDP(Float16Module(GPTModel)) and the
    # wrappers do not delegate attribute access — so a direct read silently misses
    # it and the renorm hook never installs. _unwrapped_chunks must peel first.
    inner = types.SimpleNamespace(_pgpt_post_step_norm_role_map={"p": "rows"})
    wrapped = _Wrap(_Wrap(inner))
    assert getattr(wrapped, "_pgpt_post_step_norm_role_map", None) is None  # the bug
    cores = _unwrapped_chunks([wrapped], unwrap=_peel)  # the fix
    assert getattr(cores[0], "_pgpt_post_step_norm_role_map", None) == {"p": "rows"}


def test_unwrapped_chunks_accepts_single_or_list():
    inner = types.SimpleNamespace(_pgpt_post_step_norm_role_map={"p": "rows"})
    assert _unwrapped_chunks(_Wrap(inner), unwrap=_peel)[0] is inner
    assert _unwrapped_chunks([_Wrap(inner)], unwrap=_peel)[0] is inner


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
