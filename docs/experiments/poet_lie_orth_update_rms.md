# poet_lie_orth_update_rms - Lie-Orth update-RMS optimizer

`poet_lie_orth_update_rms` is a separate optimizer class,
`LieOrthUpdateRMSMomentum`, selected with `q_optimizer=lie_ortho_update_rms`.
It does not change the fixed-angle `LieOrthMomentum` path.

The optimizer keeps alternating Lie-Orth mechanics, but replaces
`lr * poet.scale * lie_ortho_c` with a Muon/Kimi-style target:

```text
theta = min(lr * rho / RMS(W), lie_ortho_max_angle)
```

The default `rho=0.2` is the pre-LR update-RMS target. `poet.scale` must be
`1.0` and `lie_ortho_c` is not used by this mode. Alternating plus
`merge_period=1` is required so each step writes exactly one side against the
current folded/effective weight.

The first reproduction target is the current best POET run near `lr=5e-3`
(`lrsc_mup_lr5_ps50`, val/loss 3.4766), using `init_type=mup_normalized`,
`mup_alpha=4`, Nesterov Lie-Orth, and `max_angle=0.024`.

Suggested first run:

```bash
codexlog poet_urms_r020_lr5 bash scripts/train_poet_lie_orth_update_rms.sh llama3 \
  scheduler=cosine_poet training_regime=ablation_40x \
  optim.lr=0.005 optim.poet.lie_ortho_update_rms=0.2 \
  optim.poet.lie_ortho_max_angle=0.024 optim.poet.lie_ortho_rms_mode=weight \
  optim.poet.init_type=mup_normalized optim.poet.mup_alpha=4.0 \
  optim.poet.head_aligned_attn=false optim.poet.lie_alternating=true \
  optim.poet.lie_alternate_every=1 optim.poet.lie_ortho_distributed=true
```
