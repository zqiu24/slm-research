# poet_torch pin

`poet_torch` is vendored under [third_party/poet_torch/](../third_party/poet_torch/)
as plain files (not a submodule). It provides `POETLinear` and friends
(POET orthogonal-block linear layer with `merge_then_reinitialize()`), and
is consumed by [src/optim/poet_layers.py](../src/optim/poet_layers.py) and
the patches `poet_apply_to_model` / `poet_merge_step`.

## Origin

Original source: `/lustre/fast/fast/zqiu/tmp/GaLore/poet_torch/` (a research
checkout of `GaLore`). The directory was copied into `third_party/poet_torch/`
on 2026-05-13 alongside a slim `pyproject.toml` so it can be `pip install -e`'d
directly without bringing the rest of `GaLore` along.

When `GaLore` upstream stabilises (or a thin `poet_torch` repo is carved out
of it), bump this directory by replacing its contents and recording the new
upstream SHA below.

## Current pin

- Source: local copy from `/lustre/fast/fast/zqiu/tmp/GaLore/poet_torch/`
- Vendored on: 2026-05-13
- Upstream SHA: not yet pinned — local working tree of GaLore had no
  initialised git remote at copy time. To create a stable pin, push GaLore
  to GitHub, then update this doc with the SHA.

## Install

`install_slm_env.sh` runs the editable install automatically:

```bash
uv pip install --no-build-isolation -e "${SLM_REPO}/third_party/poet_torch"
```

If you build a fresh env, the line above is what ensures `import poet_torch`
succeeds inside `.venv`.

## Bump procedure

Same shape as [Megatron-LM pin §Bump procedure](megatron_pin.md):

1. Branch `poet-torch-bump-<date>`.
2. Replace contents of `third_party/poet_torch/` with the new upstream tree
   (keep the local `pyproject.toml` if upstream still lacks one).
3. Re-run `pytest tests/unit/test_poet_torch_import.py` and the layer
   replacement tests under `tests/unit/test_poet_layers.py`.
4. If `POETLinear` API surface changed, fix `src/optim/poet_layers.py` and
   `src/patches/poet_merge_step.py` (the only two slm-research call sites).
5. Update this doc with the new SHA / date / reason.
6. Merge to `main`.
