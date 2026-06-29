# Reproduce the slm-research env on another account

A concrete, account-specific companion to
[INSTALL.md → "Reproducing a known-good env"](../INSTALL.md). Captured from the
known-good working env `/lustre/fast/fast/zqiu/slm_env` (2026-06-29, same
cluster). The version pins ship with this repo in
[`docs/known_good_constraint.txt`](known_good_constraint.txt), so they reach you
via `git clone` — nothing to copy by hand.

## Why a from-scratch rebuild (you can't copy it)

`/lustre/fast/fast/zqiu` (= `/fast/zqiu`) is mode **700**. For any other account
ALL of these are untraversable — not just unreadable:

- the clone        `/lustre/fast/fast/zqiu/slm-research`
- the env          `/lustre/fast/fast/zqiu/slm_env/.venv`
- the uv cache     `/lustre/home/zqiu/.cache/uv_cu13_*`  (installer default)
- the build TMPDIR `/lustre/fast/fast/zqiu/tmp`          (installer default)
- prebuilt apex    `/lustre/fast/fast/zqiu/software/cu13/apex`

So you rebuild from your own clone with your own cache/TMPDIR. apex falls back
to a source build automatically (the prebuilt is unreachable). Everything else
is public — GitHub + the shared `/is/software/nvidia/cuda-13.2`.

## Known-good provenance (the target)

| item | value |
|------|-------|
| built by | `install_slm_env.sh <name>` (sibling-of-clone uv project) |
| slm-research commit | `aba10c6` |
| python | 3.12.13 |
| uv | 0.10.11 |
| torch | **2.11.0** (exact; every CUDA ext is ABI-bound to it) |
| CUDA toolchain | `module load cuda/13.2` → `/is/software/nvidia/cuda-13.2`, nvcc V13.2.78 |
| gcc | 11.4.0 |
| total resolved pkgs | 247 (`uv pip freeze`) |

Submodule commits (must match — Megatron/TE pins are aligned to these):

```
third_party/Megatron-Bridge  c810129341a84e58f4cbed3093f70668a088c028  (v0.4.2)
third_party/Megatron-LM      9539a12e1b04a68423f57b3eb41d6125161dca24  (core_v0.17.0)
third_party/pion             e5c3e8bd432091add0f26312807534363d385e1b  (main)
third_party/torchtitan       73a0e6979dd10b6b1904098eb3c8f62c18ab87ce  (v0.2.2)
```

Source-built / git-pinned deps — the installer rebuilds these explicitly, so
they are **not** in `known_good_constraint.txt`:

```
torch==2.11.0
flash-attn==2.8.3
transformer-engine    @ git ...TransformerEngine@71bbefbf153418f943640df0f7373625dc93fa46
deep-ep               @ git ...DeepEP@567632dd59810d77b3cc05553df953cc0f779799
nvidia-resiliency-ext @ git ...@15a851565a4ce846c04431ecb0cf09903ab4837e
emerging-optimizers   @ git ...Emerging-Optimizers@d5363b4a418128cd8111983b191c4b8869a9766b
apex   -> prebuilt for zqiu; YOU get a source build at commit becbb77... (auto fallback)
```

`known_good_constraint.txt` is the remaining 238 `name==version` pins — the
floating transitive deps, locked back to the known-good resolve.

## Reproduce — run as the OTHER user

```bash
# pick your own writable root (NOT under zqiu)
export ROOT=/lustre/fast/fast/<other>
cd "$ROOT"

# 1. fresh clone + submodules at the SAME commits
git clone --recurse-submodules https://github.com/zqiu24/slm-research.git
cd slm-research
git checkout aba10c6                       # match the known-good clone (or stay on main)
git submodule update --init --recursive
git -C third_party/Megatron-LM     checkout 9539a12e1b04a68423f57b3eb41d6125161dca24
git -C third_party/torchtitan      checkout 73a0e6979dd10b6b1904098eb3c8f62c18ab87ce
git -C third_party/pion            checkout e5c3e8bd432091add0f26312807534363d385e1b
git -C third_party/Megatron-Bridge checkout c810129341a84e58f4cbed3093f70668a088c028
cd "$ROOT"

# 2. uv (skip if `uv --version` already works)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. cache + scratch on LOCAL flock-capable disk (Lustre has no flock; zqiu's
#    default cache/TMPDIR are unreachable for you anyway)
export UV_CACHE_DIR=/tmp/uv_cu13_<other>
export TMPDIR=/tmp/uv_tmp_<other>
export UV_LINK_MODE=symlink
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"

# 4. EXACT-repro pin: uv reads UV_CONSTRAINT globally, so every `uv pip install`
#    in the installer honors these versions — no edits to install_slm_env.sh.
export UV_CONSTRAINT="$ROOT/slm-research/docs/known_good_constraint.txt"

# 5. build the env (sibling of the clone, ~15-25 min first time)
bash slm-research/install_slm_env.sh slm_env
cd slm_env
source .venv/bin/activate
```

Notes:
- The installer auto-appends a guarded `source load_cuda13_2_nccl_env.sh` to
  `activate`, so `import torch` / `transformer_engine` get the LD_PRELOAD
  libcublasLt + CUDA_HOME after `source .venv/bin/activate`.
- apex prebuilt is unreachable → installer prints "not readable; building apex
  from source" and continues. apex is optional (Megatron has fallbacks).
- W&B: the code loads a per-user `.env` at runtime — put your own
  `WANDB_API_KEY=...` in `$ROOT/slm-research/.env` (not zqiu's key).

## Verify the rebuild

```bash
source "$ROOT/slm_env/.venv/bin/activate"

# a. imports clean (the 4 source-built CUDA exts) — needs a GPU node
python -c "import torch, transformer_engine, flash_attn, apex, deep_ep; \
print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
#   expect torch 2.11.0, cuda 13.x; arch list must include your GPU
#   (B200 => '10.0'). compiled exts are arch-bound.

# b. resolved versions match the pins (no output = perfect)
uv pip freeze | grep -E '^[A-Za-z0-9_.-]+==' | sort > /tmp/new_pins.txt
comm -23 <(sort docs/known_good_constraint.txt) /tmp/new_pins.txt
#   any line printed is a pin that resolved differently — investigate / add to
#   the constraint and re-run the relevant install step.

# c. CPU test suite
pytest -m "not gpu"
```

For a byte-exact diff including the git/editable lines, ask zqiu for the full
`uv pip freeze` (247 lines) from the working env; only the 238 `==` pins are
tracked here.
