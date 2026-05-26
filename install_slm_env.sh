#!/usr/bin/env bash
# Bootstrap a uv env for slm-research on a CUDA host.
#
# Creates a sibling uv project that has slm-research + Megatron-LM installed
# editable, pointing back at this repo. Many sibling envs can share one
# slm-research clone.
#
# Usage (run from a GPU node, from a directory OUTSIDE the slm-research repo):
#
#   cd /lustre/fast/fast/zqiu/
#   export UV_LINK_MODE=symlink
#   bash /lustre/fast/fast/zqiu/slm-research/install_slm_env.sh <env_name>
#   cd <env_name>
#   source .venv/bin/activate
#
# Pins are aligned with third_party/Megatron-LM at core_v0.17.0
# (SHA 9539a12e1b04..., see docs/megatron_pin.md). The whole stack is CUDA
# 13: torch is pinned to 2.11.0 and TransformerEngine + flash-attn + apex +
# DeepEP are built from source under the cuda/13.2 module so they link the
# cu13 runtime. Reproduces the known-good clthegoat-cu13 env.

set -euo pipefail

# --- args ------------------------------------------------------------------
if [ $# -lt 1 ] || [ -z "${1:-}" ]; then
    echo "Usage: bash install_slm_env.sh <env_name>" >&2
    echo "  Creates ./<env_name>/ in the current directory." >&2
    exit 1
fi
ENV_NAME="$1"
ENV_PARENT="$(pwd)"
ENV_DIR="$ENV_PARENT/$ENV_NAME"

# --- locate slm-research repo (this script lives in the repo root) ---------
SLM_REPO="$(cd "$(dirname "$0")" && pwd)"
if [ ! -f "$SLM_REPO/pyproject.toml" ] || [ ! -d "$SLM_REPO/third_party/Megatron-LM" ]; then
    echo "ERROR: $SLM_REPO doesn't look like the slm-research repo root." >&2
    exit 1
fi
if [ ! -f "$SLM_REPO/third_party/Megatron-LM/pyproject.toml" ]; then
    echo "ERROR: $SLM_REPO/third_party/Megatron-LM is empty. Run:" >&2
    echo "  git -C $SLM_REPO submodule update --init --recursive" >&2
    exit 1
fi

# --- refuse to create the env folder INSIDE the slm-research clone --------
case "$ENV_PARENT/" in
    "$SLM_REPO"/*)
        echo "ERROR: $ENV_PARENT is inside the slm-research repo." >&2
        echo "       cd to a sibling directory (e.g. $(dirname "$SLM_REPO")) first." >&2
        exit 1 ;;
esac

if [ -e "$ENV_DIR" ]; then
    echo "ERROR: $ENV_DIR already exists. Pick a different name or rm -rf it." >&2
    exit 1
fi

# --- cluster env -----------------------------------------------------------
# CUDA 13 is required: torch 2.12+ ships with cu13 wheels, Megatron 0.17.0
# explicitly pins transformer-engine[pytorch,core_cu13], and any extension
# built from source (causal-conv1d, mamba-ssm, apex, DeepEP) checks that
# nvcc's CUDA major version matches torch.version.cuda.
export TMPDIR="${TMPDIR:-/lustre/fast/fast/zqiu/tmp}"
export UV_LINK_MODE="${UV_LINK_MODE:-symlink}"
# Per-CUDA build cache. Default to a clean cache root (NOT the older
# ~/.cache/uv_cu13 which still hosts the broken-symbol TE artifact that
# clthegoat-cu13 and software/nk-env-cu13 symlink into). The :- form lets
# you override per-invocation when building parallel envs:
#
#   UV_CACHE_DIR=~/.cache/uv_cu13_slm_env2 bash install_slm_env.sh slm_env2
export UV_CACHE_DIR="${UV_CACHE_DIR:-/lustre/home/zqiu/.cache/uv_cu13_slm_rebuild}"
# Source the same loader the user sources for runs, so we get both the
# `module load cuda/13.2` AND the explicit CUDA_HOME / PATH / LD_LIBRARY_PATH
# exports. In a non-interactive shell (tmux child, nohup, etc.) `module`
# may be undefined or a silent no-op, which leaves nvcc off PATH and makes
# TE's CMake configure fail with "Failed to find nvcc". The loader's
# explicit `export CUDA_HOME=...` + `export PATH=$CUDA_HOME/bin:$PATH`
# survives that case.
source "$SLM_REPO/load_cuda13_2_nccl_env.sh"
# CUDA 13 split libcudacxx out into <cuda>/targets/<arch>/include/cccl/.
# Add it to CPATH so `#include "cuda/std/tuple"` resolves during builds
# of extensions that touch nvshmem (DeepEP, etc.). torch's BuildExtension
# auto-adds <cuda>/include but does not yet know about the new cccl path.
export CPATH="/is/software/nvidia/cuda-13.2/targets/x86_64-linux/include/cccl${CPATH:+:$CPATH}"

if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then conda deactivate || true; fi

# --- create the sibling uv project -----------------------------------------
echo "==> Creating uv project at $ENV_DIR (editable installs will point at $SLM_REPO)"
cd "$ENV_PARENT"
uv python pin 3.12
uv init "$ENV_NAME" --no-readme --no-pin-python --bare 2>/dev/null || uv init "$ENV_NAME"
cd "$ENV_DIR"
uv python pin 3.12
uv venv --python 3.12
# shellcheck disable=SC1091
source .venv/bin/activate

# --- build prereqs + torch (torch first; --no-build-isolation needs it) ----
# pybind11 + Cython + wheel are needed by Megatron-LM's editable build under
# --no-build-isolation (uv won't create a build venv from [build-system].requires
# when isolation is disabled, so we install the build-deps into the runtime env).
uv pip install ninja packaging psutil pybind11 Cython wheel
# torch is pinned EXACT: every compiled/prebuilt CUDA extension below
# (TransformerEngine, flash-attn, apex, DeepEP, mamba-ssm) is ABI-bound to
# it. Floating to a newer torch (e.g. 2.12) breaks flash-attn's c10 symbols.
uv pip install "torch==2.11.0"
uv pip install nvidia-mathdx==25.6.0          # TE build dep

# --- TransformerEngine at Megatron's pinned SHA ---------------------------
# Matches third_party/Megatron-LM core_v0.17.0's [tool.uv.sources] entry
# so what mcore later resolves to is the same artifact that's already
# installed here — no re-clone, no rebuild.
#
# Known runtime quirk: this source build links libtransformer_engine.so
# against the SYSTEM cuda/13.2 libcublasLt (which exports a newer
# `cublasLtGroupedMatrixLayoutInit_internal@libcublasLt.so.13`) while
# torch at runtime eagerly RTLD_GLOBAL-loads the OLDER libcublasLt
# bundled inside the venv (from nvidia-cublas==13.1.0.3, a torch dep)
# — that lacks the symbol, so first `import transformer_engine` would
# crash with "undefined symbol: cublasLtGroupedMatrixLayoutInit_internal,
# version libcublasLt.so.13".
#
# The runtime fix lives in load_cuda13_2_nccl_env.sh — it sets
# LD_PRELOAD=/is/software/nvidia/cuda-13.2/lib64/libcublasLt.so.13 so
# the system cuBLAS is loaded BEFORE torch can pin the older venv copy,
# satisfying TE's symbol reference. Anyone using this env at runtime
# MUST source that loader first.
export NVTE_FRAMEWORK=pytorch
MAX_JOBS=16 NVTE_BUILD_THREADS_PER_JOB=2 \
  uv pip install --no-build-isolation \
  "transformer_engine @ git+https://github.com/NVIDIA/TransformerEngine.git@71bbefbf153418f943640df0f7373625dc93fa46"

# --- slm-research itself (editable, pointing at $SLM_REPO) ----------------
uv pip install -e "${SLM_REPO}[dev,gpu]"

# --- poet_torch (vendored under third_party/, editable) -------------------
# Provides POETLinear used by src/optim/poet_layers.py.
# Pin tracked in docs/poet_torch_pin.md.
uv pip install --no-build-isolation -e "${SLM_REPO}/third_party/poet_torch"

# --- Megatron-LM submodule, editable --------------------------------------
# [mlm] = LM training helpers (sentencepiece, tiktoken, transformers, ...)
# [dev] = mamba-ssm, causal-conv1d, flash-linear-attention, flashinfer,
#         modelopt, nv-resiliency-ext, tensorstore, einops, datasets, ...
# Use per-package no-build-isolation matching Megatron's own [tool.uv]
# no-build-isolation-package list. Packages NOT in this list (e.g.
# nvidia-resiliency-ext, which needs poetry-dynamic-versioning) get a
# normal build venv from their declared [build-system].requires.
MAX_JOBS=4 uv pip install \
  --no-build-isolation-package megatron-core \
  --no-build-isolation-package causal-conv1d \
  --no-build-isolation-package mamba-ssm \
  --no-build-isolation-package flash_mla \
  --no-build-isolation-package transformer-engine \
  --no-build-isolation-package transformer-engine-torch \
  -e "${SLM_REPO}/third_party/Megatron-LM[mlm,dev]"

# --- variant kernels not in mcore extras ----------------------------------
uv pip install liger-kernel

# Flash-Attention 2: built from the PyPI sdist against the pinned torch and
# cuda/13.2 (there is no prebuilt cu130 wheel matching torch 2.11.0).
# --no-build-isolation so the build sees the torch installed above.
MAX_JOBS=16 uv pip install --no-build-isolation flash-attn==2.8.3

# DeepEP at the commit the clthegoat-cu13 env was built against.
TORCH_CUDA_ARCH_LIST='9.0;10.0;10.3' MAX_JOBS=8 \
  uv pip install --no-build-isolation \
  "git+https://github.com/deepseek-ai/DeepEP.git@567632dd59810d77b3cc05553df953cc0f779799"

# --- Apex (fused norm/optimizer kernels) ----------------------------------
# cu13 apex tree (built for CUDA 13 / torch 2.11); the older software/apex
# was compiled for cuda/12.9.
if [ -d /lustre/fast/fast/zqiu/software/cu13/apex ]; then
  NVCC_APPEND_FLAGS="--threads 4" APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 \
    uv pip install -v --no-build-isolation /lustre/fast/fast/zqiu/software/cu13/apex
else
  echo "WARN: /lustre/fast/fast/zqiu/software/cu13/apex not found; skipping apex." >&2
fi

# --- git hooks (installed into the slm-research clone, not the env) -------
( cd "$SLM_REPO" && pre-commit install ) || \
  echo "WARN: pre-commit install failed (run later from $SLM_REPO)."

cat <<EOF

--------------------------------------------------------------------------
slm-research env install complete.

  Env folder:  $ENV_DIR
  Repo (editable):  $SLM_REPO

To use:
  cd $ENV_DIR
  source .venv/bin/activate

Sanity (run from the slm-research repo):
  cd $SLM_REPO
  pytest -m "not gpu"
--------------------------------------------------------------------------
EOF
