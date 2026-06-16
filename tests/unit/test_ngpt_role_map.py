"""Role-map suffix matching for nGPT weight normalization (fused + unfused)."""

from src.patches.ngpt_apply_spec import _match_role


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
