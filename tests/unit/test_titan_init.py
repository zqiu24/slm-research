"""titan_init re-initializes a Megatron-named model to torchtitan's llama3 scheme.

CPU-only: builds a fake module tree with the same parameter names Megatron emits
(after unfuse), runs the re-init, and checks the empirical std of each weight
against torchtitan's recipe — plus determinism, RNG non-perturbation, and the
fused-fc1 fallback. Mirrors the stds in
third_party/torchtitan/torchtitan/models/llama3/model/model.py.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.model.titan_init import apply_titan_init

DIM = 256
KV = 64
FFN = 688
VOCAB = 4096
N_LAYERS = 4
SEED = 7


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_q = nn.Linear(DIM, DIM, bias=False)
        self.linear_k = nn.Linear(DIM, KV, bias=False)
        self.linear_v = nn.Linear(DIM, KV, bias=False)
        self.linear_proj = nn.Linear(DIM, DIM, bias=False)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_fc1_gate = nn.Linear(DIM, FFN, bias=False)
        self.linear_fc1_up = nn.Linear(DIM, FFN, bias=False)
        self.linear_fc2 = nn.Linear(FFN, DIM, bias=False)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(DIM)
        self.self_attention = _Attn()
        self.pre_mlp_layernorm = nn.LayerNorm(DIM)
        self.mlp = _MLP()


class _Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(VOCAB, DIM)


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(_Layer() for _ in range(N_LAYERS))
        self.final_layernorm = nn.LayerNorm(DIM)


class FakeGPT(nn.Module):
    """Untied (separate output_layer), unfused (q/k/v + gate/up) — the 300m path."""

    def __init__(self):
        super().__init__()
        self.embedding = _Embedding()
        self.decoder = _Decoder()
        self.output_layer = nn.Linear(DIM, VOCAB, bias=False)


def _depth_std(layer: int) -> float:
    return 0.02 / math.sqrt(2.0 * (layer + 1))


def _named(model, suffix):
    return {n: p for n, p in model.named_parameters() if n.endswith(suffix)}


def test_per_category_stds_match_torchtitan():
    model = FakeGPT()
    # Mark layernorms so we can prove they're left untouched.
    for n, p in model.named_parameters():
        if n.endswith("layernorm.weight"):
            nn.init.constant_(p, 0.5)

    apply_titan_init(model, hidden_size=DIM, num_layers=N_LAYERS, seed=SEED)

    # Embeddings: normal(0, 1.0).
    emb = next(iter(_named(model, "word_embeddings.weight").values()))
    assert abs(emb.std().item() - 1.0) < 0.05

    # Fan-in projections (q/k/v, gate): fixed 0.02.
    for suffix in (
        "linear_q.weight",
        "linear_k.weight",
        "linear_v.weight",
        "linear_fc1_gate.weight",
    ):
        for w in _named(model, suffix).values():
            assert abs(w.std().item() - 0.02) < 0.0025, suffix

    # Depth-scaled (attn out / up / down): 0.02 / sqrt(2*(layer+1)) per layer.
    for layer in range(N_LAYERS):
        prefix = f"decoder.layers.{layer}."
        exp = _depth_std(layer)
        for n, p in model.named_parameters():
            if not n.startswith(prefix):
                continue
            if n.endswith(("linear_proj.weight", "linear_fc1_up.weight", "linear_fc2.weight")):
                assert abs(p.std().item() - exp) < 0.15 * exp, n

    # LM head: trunc_normal(0, dim**-0.5) truncated at +/-3 std.
    head = model.output_layer.weight
    std = DIM**-0.5
    assert abs(head.std().item() - std) < 0.05 * std
    assert head.abs().max().item() <= 3.0 * std + 1e-6

    # Layernorms untouched.
    for n, p in model.named_parameters():
        if n.endswith("layernorm.weight"):
            assert torch.allclose(p, torch.full_like(p, 0.5)), n


def test_deterministic_across_replicas():
    a, b = FakeGPT(), FakeGPT()
    apply_titan_init(a, hidden_size=DIM, num_layers=N_LAYERS, seed=SEED)
    apply_titan_init(b, hidden_size=DIM, num_layers=N_LAYERS, seed=SEED)
    da = dict(a.named_parameters())
    for n, p in b.named_parameters():
        assert torch.equal(p, da[n]), n


def test_restores_ambient_rng():
    model = FakeGPT()
    torch.manual_seed(123)
    state = torch.get_rng_state()
    apply_titan_init(model, hidden_size=DIM, num_layers=N_LAYERS, seed=SEED)
    assert torch.equal(state, torch.get_rng_state())


def test_fused_fc1_fallback_splits_gate_and_up():
    """Without --unfuse-fc1 the model has a fused linear_fc1 [gate; up]; the gate
    half must get 0.02 and the up half the depth-scaled std."""

    class FusedMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_fc1 = nn.Linear(DIM, 2 * FFN, bias=False)
            self.linear_fc2 = nn.Linear(FFN, DIM, bias=False)

    class FusedLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = FusedMLP()

    class FusedDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([FusedLayer()])

    class FusedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = FusedDecoder()

    model = FusedModel()
    apply_titan_init(model, hidden_size=DIM, num_layers=1, seed=SEED)
    fc1 = model.decoder.layers[0].mlp.linear_fc1.weight
    gate, up = fc1[:FFN], fc1[FFN:]
    assert abs(gate.std().item() - 0.02) < 0.0025
    assert abs(up.std().item() - _depth_std(0)) < 0.15 * _depth_std(0)
