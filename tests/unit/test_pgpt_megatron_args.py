"""_pgpt_arch_args emits the nGPT arch flags when experiment.kind == 'pgpt'."""

from omegaconf import OmegaConf

from src.utils.megatron_args import _pgpt_arch_args


def _cfg(kind, **ngpt):
    return OmegaConf.create({"experiment": {"kind": kind}, "optim": {"ngpt": ngpt}})


def test_emits_ngpt_flags_for_pgpt_kind():
    out = _pgpt_arch_args(_cfg("pgpt", alpha_init=0.05, sqk_init=1.0, suv_init=1.0, sz_init=1.0))
    assert "--ngpt" in out
    assert "--ngpt-alpha-init" in out
    assert "--ngpt-no-warmup" in out  # default no_warmup=True


def test_noop_for_non_pgpt_kind():
    assert _pgpt_arch_args(_cfg("ngpt")) == []
    assert _pgpt_arch_args(_cfg("adamw")) == []
