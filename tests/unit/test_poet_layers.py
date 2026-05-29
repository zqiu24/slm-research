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


class DecoupledToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # in=8, out=16 → block_count=4 ⇒ bs_in=2, bs_out=4 (decoupled).
        self.fc1 = nn.Linear(8, 16, bias=False)
        # in=8, out=12 → block_count=4 divides both; bs_in=2, bs_out=3.
        self.fc2 = nn.Linear(8, 12, bias=False)


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


def test_replace_with_block_count_builds_decoupled_layers():
    """block_count plumbs through to POETLinear(block_count=...), giving each
    side `n` blocks with potentially different block sizes."""
    m = DecoupledToyModel()
    n = replace_linears_with_poet(
        m,
        block_count=4,
        init_type="none",
        extra_linear_types=(nn.Linear,),
    )
    assert n == 2
    fc1 = m.fc1.poet_linear  # in=8, out=16
    assert fc1.block_size_in == 2 and fc1.block_size_out == 4
    assert fc1.r_in == 4 and fc1.r_out == 4
    fc2 = m.fc2.poet_linear  # in=8, out=12
    assert fc2.block_size_in == 2 and fc2.block_size_out == 3


def test_replace_with_block_count_skips_indivisible():
    """A layer whose dims aren't divisible by block_count is skipped, not raised."""
    m = ToyModel()  # fc1 8x16 ok for bc=4; fc2 16x13 (13%4!=0) skipped; output skipped
    n = replace_linears_with_poet(
        m,
        block_count=4,
        init_type="none",
        extra_linear_types=(nn.Linear,),
    )
    assert n == 1
    assert isinstance(m.fc1, POETMegatronLinear)
    assert isinstance(m.fc2, nn.Linear)  # skipped (13 not divisible by 4)


def test_replace_hard_errors_on_indivisible_unfused_segment():
    """An unfused sub-projection (e.g. linear_k) that POET can't wrap due to
    block-size divisibility is a hard error, not a silent skip."""
    import pytest

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_k = nn.Linear(8, 12, bias=False)  # out 12 % block 8 != 0

    m = M()
    with pytest.raises(ValueError, match="linear_k"):
        replace_linears_with_poet(
            m, block_size=8, init_type="none", extra_linear_types=(nn.Linear,)
        )


def test_replace_still_skips_indivisible_non_unfused_layer():
    """A non-unfused layer that isn't divisible is still skipped (not raised)."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.some_proj = nn.Linear(8, 12, bias=False)  # 12 % 8 != 0, not unfused

    m = M()
    n = replace_linears_with_poet(
        m, block_size=8, init_type="none", extra_linear_types=(nn.Linear,)
    )
    assert n == 0
    assert isinstance(m.some_proj, nn.Linear)


def test_block_count_validation_raises_at_layer_construction():
    """Task 8.6: a block_count that doesn't divide the layer dims raises a
    clear ValueError when the POETLinear is constructed directly."""
    import pytest
    from poet_torch import POETLinear

    with pytest.raises(ValueError, match="block_count 7 doesn't divide"):
        POETLinear(in_features=1536, out_features=1536, block_count=7, dtype=torch.float32)
