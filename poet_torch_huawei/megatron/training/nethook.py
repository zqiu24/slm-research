
"""
Trace will hook one module at a time.
TraceDict will hook all modules at once.
"""

import contextlib
import logging
import os
from collections import OrderedDict

import torch

from megatron.training.global_vars import get_args

logger = logging.getLogger(__name__)
_SMALL_TENSOR_SKIP_WARNED = False


def _get_monitor_min_elements():
    """Minimum number of elements required for compute_stat."""
    # Keep backward-compatible behavior by default (skip <=2 elements).
    raw_value = os.getenv("MEGATRON_MONITOR_MIN_ELEMENTS", "3")
    try:
        min_elements = int(raw_value)
    except ValueError:
        logger.warning(
            "Invalid MEGATRON_MONITOR_MIN_ELEMENTS=%r, fallback to 3.",
            raw_value,
        )
        min_elements = 3
    return max(min_elements, 1)

def compute_stat(param):
    global _SMALL_TENSOR_SKIP_WARNED
    args = get_args()
    monitor_values = set(args.monitor_values.split(','))
    if isinstance(param, tuple):
        param = param[0]
    if isinstance(param, torch.Tensor) or isinstance(param, torch.nn.Parameter):
        param = param.detach()
        param = param.reshape(-1)
        min_elements = _get_monitor_min_elements()
        if param.shape[0] < min_elements:
            if not _SMALL_TENSOR_SKIP_WARNED:
                logger.warning(
                    "Skipping tensors with fewer than %d elements in compute_stat. "
                    "Set MEGATRON_MONITOR_MIN_ELEMENTS to adjust this threshold.",
                    min_elements,
                )
                _SMALL_TENSOR_SKIP_WARNED = True
            logger.debug(
                "Skipping parameter with %d element(s) in compute_stat (threshold=%d).",
                param.shape[0],
                min_elements,
            )
            return None
        num_elements = param.shape[0]
        result = {}
        if "mean" in monitor_values:
            mean_param = torch.mean(param)
            result["mean"] = mean_param.item()
        if "std" in monitor_values:
            std_param = torch.std(param)
            result["std"] = std_param.item()
        param = torch.abs(param)
        if "top1" in monitor_values:
            max_param = torch.max(param)
            result["top1"] = max_param.item()
        if "top10%" in monitor_values:
            if num_elements > 10:
                k = num_elements // 10 + 1
                topk_value = torch.topk(param, k)[0]
                top10percent = topk_value[-1]
            else:
                top10percent = torch.max(param)
            result["top10%"] = top10percent.item()
        return result
    else:
        return None


class Trace:
    """
    To get the statistics of the output of a module during the computation of
    the given network.
    """

    def __init__(
        self,
        module,
        retain_input=False,
        retain_output=True
    ):
        """
        Method to replace a forward method with a closure that
        intercepts the call, and tracks the hook so that it can be reverted.
        """
        self.module = module
        self.used = False
        self.input_stat = None
        self.output_stat = None

        def retain_hook(m, inputs, output):
            if self.used:
                return
            self.used = True

            if retain_input:
                self.input_stat = compute_stat(inputs)
            if retain_output:
                self.output_stat = compute_stat(output)

        self.registered_hook = module.register_forward_hook(retain_hook)

    def close(self):
        self.registered_hook.remove()


class TraceTrainingStats(contextlib.AbstractContextManager):
    """
    To get the statistics of all modules during the computation
    of the given network.
    """

    def __init__(
        self,
        model,
        retain_input=False,
        retain_output=True
    ):
        self.traces = [OrderedDict() for i in range(len(model))]
        for i, model_chunk in enumerate(model):
            for name, module in model_chunk.named_modules():
                self.traces[i][name] = Trace(
                    module=module,
                    retain_input=retain_input,
                    retain_output=retain_output,
                )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

        if exc_type is not None:
            return False

        from megatron.training.utils import print_rank_0
        from megatron.training import monitor

        args = get_args()

        activation_stat = [{key: value.output_stat
            for key, value in self.traces[i].items() if value.output_stat is not None}
            for i in range(len(self.traces))]
        monitor.save_value(args.curr_iteration, activation_stat, label="activation")

        if args.monitor_log:
            print_rank_0(activation_stat)

    def close(self):
        for model_chunk_trace in self.traces:
            for _, trace in reversed(model_chunk_trace.items()):
                trace.close()

