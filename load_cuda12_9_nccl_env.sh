#!/usr/bin/env bash

module load cuda/12.9
module load nccl

export CUDA_HOME=/is/software/nvidia/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/is/software/nvidia/cudnn-9.10.2/lib:${LD_LIBRARY_PATH:-}

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
