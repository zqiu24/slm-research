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


def test_normalized_init_unit_row_norm():
    m = ToyModel()
    replace_linears_with_poet(
        m,
        block_size=8,
        init_type="normalized",
        extra_linear_types=(nn.Linear,),
    )
    w = m.fc1.poet_linear.weight.detach().float()
    row_norms = torch.norm(w, dim=1)
    assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-5)


def test_init_scale_scales_norm_preserving_shape():
    """init_scale is a uniform multiply: it scales the operating norm linearly
    but leaves the spectrum *shape* (condition number) untouched."""
    m1, m2 = ToyModel(), ToyModel()
    # Same RNG so the underlying child weights match before init.
    torch.manual_seed(0)
    m1 = ToyModel()
    torch.manual_seed(0)
    m2 = ToyModel()
    replace_linears_with_poet(
        m1,
        block_size=8,
        init_type="normalized",
        init_scale=1.0,
        extra_linear_types=(nn.Linear,),
    )
    replace_linears_with_poet(
        m2,
        block_size=8,
        init_type="normalized",
        init_scale=3.0,
        extra_linear_types=(nn.Linear,),
    )
    w1 = m1.fc1.poet_linear.weight.detach().float()
    w2 = m2.fc1.poet_linear.weight.detach().float()
    # Norm scales by exactly init_scale.
    assert torch.allclose(w2, 3.0 * w1, atol=1e-5)
    # Condition number (spectrum shape) is invariant under a scalar multiply.
    sv1 = torch.linalg.svdvals(w1)
    sv2 = torch.linalg.svdvals(w2)
    assert abs((sv1.max() / sv1.min()) - (sv2.max() / sv2.min())) < 1e-3


def test_orthogonal_init_is_well_conditioned_and_matched_norm():
    """orthogonal init → all singular values equal (condition number ≈ 1), and at
    init_scale=1 its per-element RMS matches `normalized` (1/√in)."""
    m = ToyModel()
    replace_linears_with_poet(
        m,
        block_size=8,
        init_type="orthogonal",
        extra_linear_types=(nn.Linear,),
    )
    w = m.fc1.poet_linear.weight.detach().float()  # (16, 8)
    sv = torch.linalg.svdvals(w)
    # All singular values equal ⇒ condition number ≈ 1.
    assert (sv.max() / sv.min()) < 1.05
    # Per-element RMS anchored to normalized's 1/√in (in=8).
    rms = w.pow(2).mean().sqrt().item()
    assert abs(rms - 1.0 / (8**0.5)) < 1e-4


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


def test_apply_patch_threads_lie_alternating_into_walk():
    import types

    import torch.nn as nn
    from poet_torch import AlternatingPOETXLinear, POETXLinear

    import src.patches.poet_apply_to_model as ap

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 8, bias=False)

    # Minimal args carrying just the POET knobs _apply_poet_to_chunk reads.
    args = types.SimpleNamespace(
        poet_block_size=256,
        poet_block_count=1,
        poet_init_type="none",
        poet_mup_alpha=1.0,
        poet_cache_mode="none",
        poet_parameterization="cayley",
        poet_freeze_output_rotation=False,
        poet_head_aligned_attn=False,
        poet_no_head_resid_perm=False,
        poet_single_step_fast=True,
        poet_single_step_native=False,
        poet_single_step_x=True,
        poet_single_step_x_alternating=False,
        poet_lie_alternating=True,
        poet_lie_alternate_every=3,
        kv_channels=None,
        hidden_size=8,
        num_attention_heads=1,
    )
    # _apply_poet_to_chunk discovers Megatron linear types lazily; on a CPU node that
    # returns () and the walk falls back to extra_linear_types. We can't pass
    # extra_linear_types through the patch, so monkeypatch the walk to assert the
    # flag is forwarded, then build for real.
    seen = {}
    orig = ap.replace_linears_with_poet

    def _spy(model, **kw):
        seen.update(kw)
        return orig(model, extra_linear_types=(nn.Linear,), **kw)

    ap.replace_linears_with_poet = _spy
    try:
        m = M()
        ap._apply_poet_to_chunk(m, args)
    finally:
        ap.replace_linears_with_poet = orig

    assert seen["lie_alternating"] is True
    assert seen["alternate_every"] == 3
    pl = m.fc1.poet_linear
    assert isinstance(pl, POETXLinear) and not isinstance(pl, AlternatingPOETXLinear)
    assert pl.alternating is True


def test_head_aligned_poetx_built_for_attention_under_single_step_x():
    import torch.nn as nn
    from poet_torch import HeadAlignedPOETLinear, HeadAlignedPOETXLinear

    from src.optim.poet_layers import POETMegatronLinear, replace_linears_with_poet

    # linear_q is in _HEAD_ALIGNED_SIDES (head_side="out"). hidden=16 -> heads*head_dim=32.
    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_q = nn.Linear(16, 32, bias=False)

    m = Attn()
    replace_linears_with_poet(
        m,
        block_count=1,
        init_type="none",
        extra_linear_types=(nn.Linear,),
        single_step_x=True,
        head_aligned_attn=True,
        head_dim=8,
        head_resid_block_count=2,
    )
    assert isinstance(m.linear_q, POETMegatronLinear)
    pl = m.linear_q.poet_linear
    assert isinstance(pl, HeadAlignedPOETXLinear)
    assert not isinstance(pl, HeadAlignedPOETLinear)  # the POETX port, not legacy
    assert pl.head_side == "out"
    assert pl.block_size_out == 8 and pl.r_out == 4  # per-head blocks
    assert pl.block_size_in == 8 and pl.r_in == 2  # permuted multi-block residual


def test_head_aligned_legacy_built_without_single_step_x():
    import torch.nn as nn
    from poet_torch import HeadAlignedPOETLinear

    from src.optim.poet_layers import replace_linears_with_poet

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_q = nn.Linear(16, 32, bias=False)

    m = Attn()
    replace_linears_with_poet(
        m,
        block_count=1,
        init_type="none",
        extra_linear_types=(nn.Linear,),
        head_aligned_attn=True,
        head_dim=8,
    )
    assert isinstance(m.linear_q.poet_linear, HeadAlignedPOETLinear)


def test_replace_with_one_sided_in_builds_in_only_layers():
    import torch.nn as nn
    from poet_torch import InOnlyPOETXLinear, OutOnlyPOETXLinear

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
        single_step_x_one_sided="in",
    )
    assert isinstance(m.fc1, POETMegatronLinear)
    pl = m.fc1.poet_linear
    assert isinstance(pl, InOnlyPOETXLinear)
    assert not isinstance(pl, OutOnlyPOETXLinear)
    assert pl.side == "in"


def test_replace_with_one_sided_out_builds_out_only_layers():
    import torch.nn as nn
    from poet_torch import OutOnlyPOETXLinear

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
        single_step_x_one_sided="out",
    )
    assert isinstance(m.fc1, POETMegatronLinear)
    pl = m.fc1.poet_linear
    assert isinstance(pl, OutOnlyPOETXLinear)
    assert pl.side == "out"
