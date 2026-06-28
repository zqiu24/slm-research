#!/usr/bin/env bash

module load cuda/13.2
module load nccl
# cuda/13.2 auto-loads cudnn/9.10.2 as a "required" module, which prepends
# /is/software/nvidia/cudnn-9.10.2/lib to LD_LIBRARY_PATH. torch 2.11.0 was
# compiled against nvidia-cudnn-cu13==9.19.0.56 (venv-bundled) and refuses
# to start if the older 9.10.2 wins resolution. Unload it so torch's
# bundled cudnn is what gets loaded at `import torch` time.
module unload cudnn || true

export CUDA_HOME=/is/software/nvidia/cuda-13.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
# Intentionally NOT exporting /is/software/nvidia/cudnn-9.10.2/lib here:
# torch 2.11.0 was compiled against nvidia-cudnn-cu13==9.19.0.56 (bundled in
# the venv) and refuses to start if it finds the older 9.10.2 at runtime
# ("cuDNN version incompatibility: PyTorch was compiled against (9, 19, 0)
# but found runtime version (9, 10, 2)"). Even when torch starts, TE's
# fused-attention kernels were compiled against the newer cuDNN and fail
# with "No valid engine configs" when the older runtime is loaded. Let
# torch's _load_global_deps() pull the venv-bundled libcudnn.so.9 instead.

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}

# Force the system cuda/13.2 libcublasLt to be loaded BEFORE torch's
# `import torch` can RTLD_GLOBAL-load the older venv-bundled
# nvidia/cu13/lib/libcublasLt.so.13 (from nvidia-cublas==13.1.0.3). TE
# (built against the system cuBLAS headers) references the symbol
# `cublasLtGroupedMatrixLayoutInit_internal@libcublasLt.so.13`, which
# only the system lib exports; without this preload the venv lib wins
# the soname race and TE crashes at first import.
export LD_PRELOAD=/is/software/nvidia/cuda-13.2/lib64/libcublasLt.so.13${LD_PRELOAD:+:$LD_PRELOAD}

# --- per-user secrets (W&B API key, etc.) ---------------------------------
# Source a gitignored .env at the repo root if present, so secrets like
# WANDB_API_KEY reach runs without committing them or relying on ~/.netrc.
# Each user creates their own .env (see .env.example). `set -a` auto-exports
# every KEY=VALUE the file defines; existing env vars are overridden by .env,
# so unset a var first if you want a one-off override to win.
__slm_env_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
if [ -f "${__slm_env_root}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "${__slm_env_root}/.env"
  set +a
fi
unset __slm_env_root
