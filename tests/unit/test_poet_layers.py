"""Test the POET layer-replacement walk on a toy module tree.

Megatron's ColumnParallelLinear / TEColumnParallelLinear aren't importable
without a CUDA build, so we drive the walk via ``extra_linear_types`` with
plain ``torch.nn.Linear`` instead.
"""

import torch
import torch.nn as nn

from src.optim.poet_layers import POETMegatronLinear, replace_linears_with_poet


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16, bias=False)  # both dims divisible by 8 → replace
        self.fc2 = nn.Linear(16, 13, bias=False)  # 13 % 8 != 0 → skip
        self.output_layer = nn.Linear(16, 16, bias=False)  # name-skipped


def test_replace_skips_indivisible_dims():
    m = ToyModel()
    n_replaced = replace_linears_with_poet(
        m,
        block_size=8,
        init_type="none",
        extra_linear_types=(nn.Linear,),
    )
    assert n_replaced == 1
    assert isinstance(m.fc1, POETMegatronLinear)
    assert isinstance(m.fc2, nn.Linear)
    assert isinstance(m.output_layer, nn.Linear)


def test_init_type_none_preserves_weight_norm():
    m = ToyModel()
    orig = m.fc1.weight.detach().clone()
    replace_linears_with_poet(
        m,
        block_size=8,
        init_type="none",
        extra_linear_types=(nn.Linear,),
    )
    new = m.fc1.poet_linear.weight.detach()
    assert torch.allclose(new, orig.to(new.dtype), atol=1e-6)


def test_wrapper_shape_and_tuple_convention():
    """Inspect ``POETMegatronLinear`` structurally — its ``forward`` returns a
    2-tuple via :class:`POETMegatronLinear.forward` definition. We can't
    actually execute the forward on CPU because POETLinear's kernel runs
    through torch.compile / inductor which has no CPU backend.
    """
    import inspect

    sig = inspect.signature(POETMegatronLinear.forward)
    # ``self``, ``input_``, ``weight``, **kw.
    assert "input_" in sig.parameters
    src = inspect.getsource(POETMegatronLinear.forward)
    assert "return output, None" in src


def test_replace_uses_cached_poet_linear_when_cache_mode_set():
    from src.optim import poet_cache as pc

    pc.reset_for_testing()
    m = ToyModel()
    n = replace_linears_with_poet(
        m,
        block_size=8,
        init_type="none",
        extra_linear_types=(nn.Linear,),
        cache_mode="cached_fwd_bwd",
    )
    assert n == 1
    assert isinstance(m.fc1.poet_linear, pc.CachedPOETLinear)
    live = list(pc.iter_live_layers())
    assert len(live) == 1
    assert live[0] is m.fc1.poet_linear


def test_replace_uses_upstream_poet_linear_when_cache_mode_none():
    from poet_torch import POETLinear

    from src.optim import poet_cache as pc

    pc.reset_for_testing()
    m = ToyModel()
    n = replace_linears_with_poet(
        m,
        block_size=8,
        init_type="none",
        extra_linear_types=(nn.Linear,),
        cache_mode="none",
    )
    assert n == 1
    # In `none` mode we use upstream POETLinear and do NOT register.
    assert type(m.fc1.poet_linear) is POETLinear
    assert list(pc.iter_live_layers()) == []
