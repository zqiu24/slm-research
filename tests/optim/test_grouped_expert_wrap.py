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


def test_walk_installs_grouped_poetx_and_forward_matches():
    from poet_torch.grouped_poetx_layer import GroupedPOETXLinear

    torch.manual_seed(0)
    m = _FakeSequentialMLP().to(torch.float64)
    ref = _FakeSequentialMLP().to(torch.float64)
    ref.load_state_dict(m.state_dict())

    tokens = torch.randn(9, 8, dtype=torch.float64)
    tpe = torch.tensor([2, 3, 4])
    ref_out, _ = ref(tokens, tpe, None)

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
    # oft_R==0 at init -> grouped forward equals the original expert forward.
    out, _ = m(tokens, tpe, None)
    assert torch.allclose(out, ref_out, atol=1e-9)
