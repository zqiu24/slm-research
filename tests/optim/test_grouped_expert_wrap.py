import torch
import torch.nn as nn

from src.optim.poet_layers import replace_linears_with_poet


class _ColLinear(nn.Module):  # stands in for ColumnParallelLinear
    def __init__(self, i, o):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(o, i, dtype=torch.float64))
        self.bias = None
        self.skip_bias_add = True

    def forward(self, x):
        return x @ self.weight.t(), None


class _Expert(nn.Module):
    def __init__(self, h=8, f=8):
        super().__init__()
        self.linear_fc1 = _ColLinear(h, f)
        self.linear_fc2 = _ColLinear(f, h)

    def forward(self, x, probs=None):
        h, _ = self.linear_fc1(x)
        h = torch.relu(h)
        o, _ = self.linear_fc2(h)
        return o, None


class _FakeSequentialMLP(nn.Module):
    """Mirrors the SequentialMLP contract the grouped install targets."""

    def __init__(self, num_experts=3, h=8, f=8):
        super().__init__()
        self.num_local_experts = num_experts
        self.local_experts = nn.ModuleList([_Expert(h, f) for _ in range(num_experts)])

    def forward(self, permuted, tokens_per_expert, probs):
        outs = []
        for ex, t in zip(
            self.local_experts, torch.split(permuted, tokens_per_expert.tolist()), strict=False
        ):
            o, _ = ex(t)
            outs.append(o)
        return torch.cat(outs, 0), None


def test_walk_installs_grouped_poetx_and_matches_per_expert_subinstances():
    from poet_torch.grouped_poetx_layer import GroupedPOETXLinear

    torch.manual_seed(0)
    m = _FakeSequentialMLP().to(torch.float64)

    tokens = torch.randn(9, 8, dtype=torch.float64)
    tpe = torch.tensor([2, 3, 4])

    n = replace_linears_with_poet(
        m,
        block_count=2,
        single_step_x=True,
        single_step_fast=True,
        lie_alternating=True,
        alternate_every=1,
        group_experts=True,
        extra_grouped_types=(_FakeSequentialMLP,),
    )
    assert n >= 1
    assert any(isinstance(mod, GroupedPOETXLinear) for mod in m.modules())

    # POET re-inits the base weight (normalize + perm bake), so it does NOT preserve
    # the ORIGINAL expert forward. The meaningful invariant is that the grouped
    # BATCHED swapped forward equals the per-expert path run through the grouped
    # module's OWN POETX sub-instances (fc1 -> relu -> fc2).
    g1 = m._poet_grouped["linear_fc1"]
    g2 = m._poet_grouped["linear_fc2"]
    outs = []
    for e, t in zip(range(len(tpe)), torch.split(tokens, tpe.tolist()), strict=False):
        h = g1.experts[e](t)
        h = torch.relu(h)
        o = g2.experts[e](h)
        outs.append(o)
    ref = torch.cat(outs, 0)

    out, _ = m(tokens, tpe, None)
    assert torch.allclose(out, ref, atol=1e-9)
