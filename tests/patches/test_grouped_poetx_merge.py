import torch
from poet_torch.grouped_poetx_layer import GroupedPOETXLinear

from src.patches.poet_merge_step import _merge_grouped


def test_merge_grouped_folds_and_zeros_active_side():
    torch.manual_seed(0)
    g = GroupedPOETXLinear(
        3, 8, 8, block_count=2, alternating=True, alternate_every=1, dtype=torch.float64
    )
    for e in range(3):
        g.experts[e].weight.data.copy_(torch.randn(8, 8, dtype=torch.float64))
        g.experts[e].bake_perms_into_weight()
    g.bind_weights()
    for ex in g.experts:
        ex.oft_R_in.data.normal_(std=0.1)
        ex.oft_R_out.data.normal_(std=0.1)

    # effective weight via a forward at the current oft_R, captured before merge.
    # Inject pure-torch cayley so the fold runs on CPU (default Triton op needs a GPU).
    from poet_torch.poet_layer import cayley_batch

    w_before = g.weight.clone()
    _merge_grouped([g], reinit_perm=False, cayley_fn=cayley_batch)
    # active side folded into weight; folded side's oft_R zeroed; weight changed.
    assert not torch.allclose(g.weight, w_before)
    from poet_torch.alt_state import active_side

    active = active_side(1)
    folded = "oft_R_in" if active == "in" else "oft_R_out"
    for ex in g.experts:
        assert getattr(ex, folded).abs().max() == 0
