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


def test_merge_fold_only_is_forward_invariant_and_keeps_perm():
    import torch
    from poet_torch import POETLinear

    torch.manual_seed(0)
    layer = POETLinear(
        in_features=8,
        out_features=8,
        block_count=1,
        dtype=torch.float32,
        parameterization="exp",
    )
    layer.random_init_parameters()  # nonzero oft_R so the fold actually changes W
    x = torch.randn(4, 8, dtype=torch.float32)

    out_before = layer(x).detach().clone()
    perm_in_before = layer.perm_in.clone()
    perm_out_before = layer.perm_out.clone()
    weight_before = layer.weight.clone()

    layer.merge_then_reinitialize(reinit_perm=False)

    # Forward output unchanged (rotation moved into W, not lost) ...
    assert torch.allclose(out_before, layer(x), atol=1e-4)
    # ... Ψ unchanged (fold-only) ... oft_R reset ... weight absorbed the rotation.
    assert torch.equal(layer.perm_in, perm_in_before)
    assert torch.equal(layer.perm_out, perm_out_before)
    assert torch.count_nonzero(layer.oft_R_in) == 0
    assert torch.count_nonzero(layer.oft_R_out) == 0
    assert not torch.allclose(layer.weight, weight_before)


def test_merge_reinit_perm_true_is_forward_invariant_and_resamples_perm():
    import torch
    from poet_torch import POETLinear

    torch.manual_seed(0)
    layer = POETLinear(
        in_features=8,
        out_features=8,
        block_count=1,
        dtype=torch.float32,
        parameterization="exp",
    )
    layer.random_init_parameters()
    x = torch.randn(4, 8, dtype=torch.float32)

    out_before = layer(x).detach().clone()
    perm_in_before = layer.perm_in.clone()

    layer.merge_then_reinitialize(reinit_perm=True)

    # Still forward-invariant (weight re-permuted to match the new Ψ) ...
    assert torch.allclose(out_before, layer(x), atol=1e-4)
    # ... but Ψ WAS resampled (collision prob for 8! is negligible).
    assert not torch.equal(layer.perm_in, perm_in_before)
    assert torch.count_nonzero(layer.oft_R_in) == 0


def test_merge_fold_only_forward_invariant_nonsquare_ffn_shape():
    # FFN layers are non-square (e.g. hidden->ffn); the decoupled fold path is
    # where a row/col-perm swap would hide. Verified: diff ~3.6e-7.
    import torch
    from poet_torch import POETLinear

    torch.manual_seed(3)
    layer = POETLinear(
        in_features=8,
        out_features=16,
        block_count=1,
        dtype=torch.float32,
        parameterization="exp",
    )
    layer.random_init_parameters()
    x = torch.randn(5, 8, dtype=torch.float32)

    out_before = layer(x).detach().clone()
    layer.merge_then_reinitialize(reinit_perm=False)
    assert torch.allclose(out_before, layer(x), atol=1e-4)


def test_merge_then_reinitialize_defaults_to_reinit():
    import inspect

    from poet_torch import POETLinear

    sig = inspect.signature(POETLinear.merge_then_reinitialize)
    assert sig.parameters["reinit_perm"].default is True


def test_head_aligned_routing_and_gqa():
    import torch.nn as nn
    from poet_torch import HeadAlignedPOETLinear, POETLinear

    from src.optim.poet_layers import POETMegatronLinear, replace_linears_with_poet

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_q = nn.Linear(512, 512, bias=False)  # 8 q heads
            self.linear_k = nn.Linear(512, 256, bias=False)  # 4 kv heads (GQA)
            self.linear_v = nn.Linear(512, 256, bias=False)
            self.linear_proj = nn.Linear(512, 512, bias=False)  # o: head side = in

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attention = Attn()
            self.linear_fc1_gate = nn.Linear(512, 1536, bias=False)
            self.linear_fc2 = nn.Linear(1536, 512, bias=False)

    m = Block()
    n = replace_linears_with_poet(
        m,
        block_count=1,
        head_aligned_attn=True,
        head_dim=64,
        extra_linear_types=(nn.Linear,),
    )
    assert n == 6

    def inner(mod):
        assert isinstance(mod, POETMegatronLinear)
        return mod.poet_linear

    q = inner(m.self_attention.linear_q)
    assert isinstance(q, HeadAlignedPOETLinear) and q.head_side == "out" and q.head_count == 8
    k = inner(m.self_attention.linear_k)
    assert (
        isinstance(k, HeadAlignedPOETLinear) and k.head_side == "out" and k.head_count == 4
    )  # GQA
    o = inner(m.self_attention.linear_proj)
    assert isinstance(o, HeadAlignedPOETLinear) and o.head_side == "in" and o.head_count == 8
    # MLP stays stock POETLinear.
    assert isinstance(inner(m.linear_fc1_gate), POETLinear)
    assert not isinstance(inner(m.linear_fc1_gate), HeadAlignedPOETLinear)


def test_sharded_state_dict_is_deduped_replicated_and_complete():
    """Regression: POET-wrapped linears MUST expose ``sharded_state_dict`` or
    Megatron's ``torch_dist`` save aborts at the first checkpoint with
    ``AttributeError: 'POETMegatronLinear' object has no attribute
    'sharded_state_dict'`` (the save walks every submodule). Verify the wrapper
    emits fully-replicated ShardedTensors (tp=1) covering every param + buffer,
    with the aliased base weight/bias serialized exactly once.
    """
    import pytest

    pytest.importorskip("megatron.core")
    import os

    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.dist_checkpointing.mapping import ShardedTensor

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29577")
    created_pg = not dist.is_initialized()
    if created_pg:
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        ps.initialize_model_parallel(tensor_model_parallel_size=1)
        try:

            class Toy(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(16, 16, bias=True)

            m = Toy()
            assert (
                replace_linears_with_poet(
                    m, block_size=8, init_type="none", extra_linear_types=(nn.Linear,)
                )
                == 1
            )
            ssd = m.fc1.sharded_state_dict(prefix="fc1.")

            # Aliased base weight/bias deduped -> serialized once under poet_linear.*.
            assert "fc1.weight" not in ssd and "fc1.bias" not in ssd
            assert sum(1 for k in ssd if k.endswith("weight")) == 1
            # Trainable rotations + persistent permutation/skew buffers all present.
            for k in (
                "fc1.poet_linear.weight",
                "fc1.poet_linear.bias",
                "fc1.poet_linear.oft_R_in",
                "fc1.poet_linear.oft_R_out",
                "fc1.poet_linear.perm_in",
                "fc1.poet_linear.rows_in",
            ):
                assert k in ssd, k
            # Replicated at tp=1: every entry is a ShardedTensor with local==global.
            for v in ssd.values():
                assert isinstance(v, ShardedTensor)
                assert tuple(v.local_shape) == tuple(v.global_shape)
        finally:
            ps.destroy_model_parallel()
    finally:
        if created_pg:
            dist.destroy_process_group()


def test_head_aligned_requires_unfused_qkv():
    import torch.nn as nn

    from src.optim.poet_layers import replace_linears_with_poet

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_qkv = nn.Linear(512, 1024, bias=False)  # still fused

    import pytest

    with pytest.raises(ValueError, match="unfused"):
        replace_linears_with_poet(
            Attn(),
            block_count=1,
            head_aligned_attn=True,
            head_dim=64,
            extra_linear_types=(nn.Linear,),
        )


def test_single_step_fast_flag_set_on_wrapped_layers():
    import torch.nn as nn

    from src.optim.poet_layers import POETMegatronLinear, replace_linears_with_poet

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16, bias=False)

    m = M()
    replace_linears_with_poet(
        m,
        block_count=1,
        init_type="none",
        extra_linear_types=(nn.Linear,),
        single_step_fast=True,
    )
    assert isinstance(m.fc1, POETMegatronLinear)
    assert m.fc1.poet_linear.single_step_fast is True


def test_single_step_native_uses_new_class():
    import torch.nn as nn
    from poet_torch import SingleStepPOETLinear

    from src.optim.poet_layers import POETMegatronLinear, replace_linears_with_poet

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16, bias=False)

    m = M()
    replace_linears_with_poet(
        m,
        block_count=1,
        init_type="none",
        extra_linear_types=(nn.Linear,),
        single_step_native=True,
    )
    assert isinstance(m.fc1, POETMegatronLinear)
    assert isinstance(m.fc1.poet_linear, SingleStepPOETLinear)


def test_single_step_x_uses_poetx_class():
    import torch.nn as nn
    from poet_torch import POETXLinear

    from src.optim.poet_layers import POETMegatronLinear, replace_linears_with_poet

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16, bias=False)

    m = M()
    orig = m.fc1.weight.detach().clone()
    replace_linears_with_poet(
        m,
        block_count=1,
        init_type="none",
        extra_linear_types=(nn.Linear,),
        single_step_x=True,
    )
    assert isinstance(m.fc1, POETMegatronLinear)
    pl = m.fc1.poet_linear
    assert isinstance(pl, POETXLinear)
    # The stored weight is the FORWARD frame: Wx = orig[perm_out][:,perm_in].
    eff = orig.index_select(0, pl.perm_out.long()).index_select(1, pl.perm_in.long())
    import torch

    assert torch.allclose(pl.weight, eff.to(pl.weight.dtype), atol=1e-6)


def test_single_step_x_alternating_uses_alternating_poetx_class():
    import torch.nn as nn
    from poet_torch import AlternatingPOETXLinear

    from src.optim.poet_layers import POETMegatronLinear, replace_linears_with_poet

    # Mirror test_single_step_x_uses_poetx_class: a plain nn.Linear is NOT in the
    # default linear_types, so pass extra_linear_types=(nn.Linear,) + init_type="none".
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 8, bias=False)

    m = M()
    replace_linears_with_poet(
        m,
        block_count=1,
        init_type="none",
        extra_linear_types=(nn.Linear,),
        single_step_x=True,
        single_step_x_alternating=True,
        alternate_every=2,
    )
    assert isinstance(m.fc1, POETMegatronLinear)
    pl = m.fc1.poet_linear
    assert isinstance(pl, AlternatingPOETXLinear)
    assert pl.alternate_every == 2


def test_single_step_x_with_lie_alternating_builds_alternating_poetx():
    import torch.nn as nn
    from poet_torch import AlternatingPOETXLinear, POETXLinear

    from src.optim.poet_layers import POETMegatronLinear, replace_linears_with_poet

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 8, bias=False)

    m = M()
    replace_linears_with_poet(
        m,
        block_count=1,
        init_type="none",
        extra_linear_types=(nn.Linear,),
        single_step_x=True,
        lie_alternating=True,
        alternate_every=2,
    )
    assert isinstance(m.fc1, POETMegatronLinear)
    pl = m.fc1.poet_linear
    # Integrated path: a PLAIN POETXLinear with the alternating flag set -- NOT the
    # true-single-side AlternatingPOETXLinear subclass (both momenta stay fed).
    assert isinstance(pl, POETXLinear)
    assert not isinstance(pl, AlternatingPOETXLinear)
    assert pl.alternating is True
    assert pl.alternate_every == 2


def test_single_step_x_without_lie_alternating_builds_plain_poetx():
    import torch.nn as nn
    from poet_torch import POETXLinear

    from src.optim.poet_layers import replace_linears_with_poet

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 8, bias=False)

    m = M()
    replace_linears_with_poet(
        m,
        block_count=1,
        init_type="none",
        extra_linear_types=(nn.Linear,),
        single_step_x=True,
    )
    pl = m.fc1.poet_linear
    assert isinstance(pl, POETXLinear)
    assert pl.alternating is False
