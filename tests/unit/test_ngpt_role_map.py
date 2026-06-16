"""Role-map suffix matching for nGPT weight normalization (fused + unfused)."""

import torch

from src.patches.ngpt_apply_spec import _match_role, _register_ngpt_norm_roles


def test_fused_names_map_to_roles():
    assert _match_role("decoder.layers.0.self_attention.linear_qkv.weight") == "rows"
    assert _match_role("decoder.layers.0.self_attention.linear_proj.weight") == "cols"
    assert _match_role("decoder.layers.0.mlp.linear_fc1.weight") == "rows"
    assert _match_role("decoder.layers.0.mlp.linear_fc2.weight") == "cols"


def test_unfused_qkv_names_map_to_rows():
    base = "decoder.layers.3.self_attention."
    assert _match_role(base + "linear_q.weight") == "rows"
    assert _match_role(base + "linear_k.weight") == "rows"
    assert _match_role(base + "linear_v.weight") == "rows"


def test_unfused_mlp_uv_names_map_to_rows():
    base = "decoder.layers.3.mlp."
    assert _match_role(base + "linear_fc1_u.weight") == "rows"
    assert _match_role(base + "linear_fc1_v.weight") == "rows"


def test_layer_norm_weight_and_unrelated_params_do_not_match():
    # TE LayerNorm-fused weight must not be mistaken for a matrix to normalize.
    assert _match_role("decoder.layers.0.self_attention.linear_qkv.layer_norm_weight") is None
    assert _match_role("decoder.final_layernorm.weight") is None


def test_generic_unfuse_gate_up_names_are_intentionally_unmapped():
    # nGPT's MLP unfuses to u/v (native), NOT to the generic gate/up that
    # model_unfuse_linears would produce. Asserting these stay unmapped guards
    # against a future accidental addition and documents the intentional gap.
    base = "decoder.layers.0.mlp."
    assert _match_role(base + "linear_fc1_gate.weight") is None
    assert _match_role(base + "linear_fc1_up.weight") is None


class _FakeUnfusedModel:
    """Plain stand-in: `_register_ngpt_norm_roles` only calls
    `named_parameters()` and assigns `_ngpt_norm_role_map`. Param names mirror a
    fully-unfused nGPT model; distinct tensors so they key the role dict
    uniquely."""

    def __init__(self, layers=2, hidden=8, ffn=16):
        self._params = []
        for i in range(layers):
            for sub, rows in [
                (f"decoder.layers.{i}.self_attention.linear_q", hidden),
                (f"decoder.layers.{i}.self_attention.linear_k", hidden),
                (f"decoder.layers.{i}.self_attention.linear_v", hidden),
                (f"decoder.layers.{i}.self_attention.linear_proj", hidden),
                (f"decoder.layers.{i}.mlp.linear_fc1_u", ffn),
                (f"decoder.layers.{i}.mlp.linear_fc1_v", ffn),
                (f"decoder.layers.{i}.mlp.linear_fc2", hidden),
            ]:
                self._params.append(
                    (sub + ".weight", torch.nn.Parameter(torch.randn(rows, hidden)))
                )
        self._params.append(
            ("embedding.word_embeddings.weight", torch.nn.Parameter(torch.randn(10, hidden)))
        )

    def named_parameters(self, *a, **k):
        return list(self._params)


def test_register_roles_matches_all_unfused_matrices():
    model = _FakeUnfusedModel(layers=2, hidden=8, ffn=16)
    _register_ngpt_norm_roles(model, expected_layers=2)
    roles = model._ngpt_norm_role_map
    # 7 matrices/layer * 2 + 1 embedding = 15 normalizable params.
    assert len(roles) == 15
    n_rows = sum(1 for r in roles.values() if r == "rows")
    n_cols = sum(1 for r in roles.values() if r == "cols")
    # rows: q,k,v,fc1_u,fc1_v (5/layer) + embedding = 11 ; cols: proj,fc2 (2/layer) = 4
    assert (n_rows, n_cols) == (11, 4)
