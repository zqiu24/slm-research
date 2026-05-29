# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

import os

import torch

from megatron.core import mpu
from megatron.training.global_vars import get_args
from megatron.training.utils import print_rank_0


def ensure_directory_exists(filename):
    """Build filename's path if it does not already exists."""
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)



def get_save_name(save_path, iteration, release=False,
                        pipeline_parallel=None,
                        tensor_rank=None, pipeline_rank=None,
                        expert_parallel=None, expert_rank=None, data_rank=None, file_name="default.txt"):
    """Determine the directory name for this rank's file."""
    if release:
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(iteration)

    # Use both the tensor and pipeline MP rank.
    if pipeline_parallel is None:
        pipeline_parallel = (mpu.get_pipeline_model_parallel_world_size() > 1)
    if tensor_rank is None:
        tensor_rank = mpu.get_tensor_model_parallel_rank()
    if pipeline_rank is None:
        pipeline_rank = mpu.get_pipeline_model_parallel_rank()
    if expert_parallel is None:
        expert_parallel = (mpu.get_expert_model_parallel_world_size() > 1)
    if expert_rank is None:
        expert_rank = mpu.get_expert_model_parallel_rank()
    if data_rank is None:
        data_rank = mpu.get_data_parallel_rank()  # TODO: incorporate into save path for dist-optimizer logging

    # Use both the tensor and pipeline MP rank. If using the distributed
    # optimizer, then the optimizer's path must additionally include the
    # data parallel rank.
    if not pipeline_parallel:
        common_path = os.path.join(save_path, directory,
                            f'mp_rank_{tensor_rank:02d}')
    else:
        common_path = os.path.join(save_path, directory,
                f'mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}')

    if expert_parallel:
        common_path = common_path + f'_{expert_rank:03d}'

    return os.path.join(common_path, file_name)


def save_value(iteration, save_dict, label=""):
    """Write log for this rank."""
    args = get_args()

    # Log name.
    save_name = get_save_name(os.path.join(args.save, "log"), iteration, file_name=f"{label}.pkl")

    # Save.
    ensure_directory_exists(save_name)

    # Barrier before save so all ranks synchronize before any I/O begins.
    # This prevents rank 0 from hanging other ranks at a post-save barrier
    # if torch.save fails (e.g. disk full).
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if not torch.distributed.is_initialized() or mpu.get_expert_data_parallel_rank() == 0:
        torch.save(save_dict, save_name)


def monitor_param_grad(model):
    from megatron.training.nethook import compute_stat
    args = get_args()
    grad_stats = [{} for i in range(len(model))]
    param_stats = [{} for i in range(len(model))]
    for i, model_chunk in enumerate(model):
        for name, param in model_chunk.named_parameters():
            grad = param.main_grad if hasattr(param, 'main_grad') else param.grad
            grad_stats[i][name] = compute_stat(grad)
            param_stats[i][name] = compute_stat(param)
    if args.monitor_log:
        print_rank_0('grad stats: {}'.format(grad_stats))
        print_rank_0('param stats: {}'.format(param_stats))
    save_value(args.curr_iteration, grad_stats, label="grad")
    save_value(args.curr_iteration, param_stats, label="param")


def set_router_monitor(model, enabled):
    """Toggle _monitor_enabled flag on all Router modules."""
    from megatron.core.transformer.moe.router import Router
    for model_chunk in model:
        for module in model_chunk.modules():
            if isinstance(module, Router):
                module._monitor_enabled = enabled


def monitor_router_logits(model):
    """Collect and save routing logits statistics from all Router modules."""
    from megatron.core.transformer.moe.router import Router
    from megatron.training.nethook import compute_stat
    args = get_args()

    logits_stats = [{} for _ in range(len(model))]
    for i, model_chunk in enumerate(model):
        for name, module in model_chunk.named_modules():
            if isinstance(module, Router) and hasattr(module, '_routing_logits'):
                logits_stats[i][name] = compute_stat(module._routing_logits)
                del module._routing_logits

    if args.monitor_log:
        print_rank_0('router logits stats: {}'.format(logits_stats))
    save_value(args.curr_iteration, logits_stats, label="router_logits")
