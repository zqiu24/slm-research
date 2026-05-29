# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Base classes for stability monitoring.

Provides abstract base class and common utilities for creating
layer-level monitors with distributed training support.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Type
from collections import defaultdict
import torch
import torch.nn as nn
import torch.distributed


@dataclass
class MonitorConfig:
    """Configuration for stability monitors.

    Attributes:
        enabled: Whether the monitor is enabled.
        sample_freq: Frequency of metric computation (in training steps).
        log_per_module: Whether to log metrics for each module separately.
        log_aggregated: Whether to log aggregated (across-module averaged) metrics.
        sample_tokens: Number of tokens to sample for metric computation.
    """
    enabled: bool = False
    sample_freq: int = 100
    log_per_module: bool = False
    log_aggregated: bool = True
    sample_tokens: int = 256


class BaseLayerMonitor(ABC):
    """
    Abstract base class for layer-level stability monitors.

    Provides:
    - Hook registration/removal on target module types
    - Distributed aggregation utilities (TP-aware)
    - Metric storage and export

    Subclasses must implement:
    - target_module_class: The module type to hook
    - compute_metrics(): The metric computation logic

    Example:
        class MyMonitor(BaseLayerMonitor):
            target_module_class = TransformerLayer

            def compute_metrics(self, name, module, x_in, x_out):
                return {"my_metric": x_out.norm().item()}
    """

    # Subclass must define: the module type to hook
    target_module_class: Type[nn.Module] = None

    def __init__(self, config: MonitorConfig):
        """Initialize the monitor.

        Args:
            config: Monitor configuration.
        """
        self.config = config
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._input_cache: Dict[str, torch.Tensor] = {}
        self._metrics: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self._enabled = False

        # Distributed state (lazy initialized)
        self._tp_group = None
        self._tp_size = 1
        self._dp_group = None
        self._pp_group = None
        self._distributed_initialized = False

    def _init_distributed(self):
        """Lazily initialize distributed communication groups."""
        if self._distributed_initialized:
            return
        try:
            from megatron.core import parallel_state
            if parallel_state.is_initialized():
                self._tp_group = parallel_state.get_tensor_model_parallel_group()
                self._tp_size = parallel_state.get_tensor_model_parallel_world_size()
                self._dp_group = parallel_state.get_data_parallel_group()
                self._pp_group = parallel_state.get_pipeline_model_parallel_group()
        except (ImportError, RuntimeError):
            pass  # Not in distributed mode
        self._distributed_initialized = True

    # ========== Hook Management (Reusable) ==========

    def register(self, model: nn.Module):
        """Register hooks on all target modules in the model.

        Args:
            model: The model to register hooks on.
        """
        self._init_distributed()
        if self.target_module_class is None:
            raise ValueError(
                f"{self.__class__.__name__} must define target_module_class"
            )

        for name, module in model.named_modules():
            if isinstance(module, self.target_module_class):
                pre_hook = module.register_forward_pre_hook(
                    self._make_pre_hook(name)
                )
                post_hook = module.register_forward_hook(
                    self._make_post_hook(name)
                )
                self._hooks.extend([pre_hook, post_hook])

    def unregister(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _make_pre_hook(self, name: str):
        """Create a pre-forward hook for caching input."""
        def hook(module, inputs):
            if self._enabled:
                captured = self._capture_input(inputs)
                if captured is not None:
                    self._input_cache[name] = captured
        return hook

    def _make_post_hook(self, name: str):
        """Create a post-forward hook for computing metrics."""
        def hook(module, inputs, outputs):
            if self._enabled and name in self._input_cache:
                input_tensor = self._input_cache.pop(name)
                output_tensor = self._capture_output(outputs)
                if output_tensor is not None:
                    metrics = self.compute_metrics(
                        name, module, input_tensor, output_tensor
                    )
                    for k, v in metrics.items():
                        self._metrics[f"{name}/{k}"].append(v)
        return hook

    # ========== Overridable Methods ==========

    def _capture_input(self, inputs) -> Optional[torch.Tensor]:
        """Extract the tensor to monitor from hook inputs.

        Override this method for custom input extraction.

        Args:
            inputs: The inputs passed to the forward hook.

        Returns:
            The tensor to monitor, or None to skip.
        """
        if inputs and len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
            return inputs[0].detach()
        return None

    def _capture_output(self, outputs) -> Optional[torch.Tensor]:
        """Extract the tensor to monitor from hook outputs.

        Override this method for custom output extraction.

        Args:
            outputs: The outputs from the forward hook.

        Returns:
            The tensor to monitor, or None to skip.
        """
        if isinstance(outputs, tuple) and len(outputs) > 0:
            out = outputs[0]
        else:
            out = outputs

        if isinstance(out, torch.Tensor):
            return out.detach()
        return None

    @abstractmethod
    def compute_metrics(
        self,
        name: str,
        module: nn.Module,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute and return metrics for a single forward pass.

        Subclasses must implement this method.

        Args:
            name: The module name (from named_modules).
            module: The module instance.
            input_tensor: The input tensor (from _capture_input).
            output_tensor: The output tensor (from _capture_output).

        Returns:
            A dictionary of metric names to scalar tensors.
            Example: {"velocity": tensor(0.05), "cos_sim": tensor(0.98)}
        """
        pass

    # ========== Distributed Utilities (Reusable) ==========

    def reduce_scalar_tp(
        self,
        value: torch.Tensor,
        op=torch.distributed.ReduceOp.SUM,
    ) -> torch.Tensor:
        """Reduce a scalar across the TP group.

        Args:
            value: A scalar tensor.
            op: The reduction operation.

        Returns:
            The reduced scalar.
        """
        if self._tp_size > 1 and self._tp_group is not None:
            torch.distributed.all_reduce(value, op=op, group=self._tp_group)
        return value

    def compute_global_norm_sq(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the global L2 norm squared (TP-aware).

        For TP-sharded tensors, this correctly computes the global norm
        by summing local squared values across TP ranks.

        Args:
            x: The input tensor.

        Returns:
            The global L2 norm squared as a scalar tensor.
        """
        local_sq = (x.float() ** 2).sum()
        return self.reduce_scalar_tp(local_sq)

    def compute_global_dot(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Compute the global dot product (TP-aware).

        For TP-sharded tensors, this correctly computes the global dot product
        by summing local dot products across TP ranks.

        Args:
            x: First input tensor.
            y: Second input tensor.

        Returns:
            The global dot product as a scalar tensor.
        """
        local_dot = (x.float() * y.float()).sum()
        return self.reduce_scalar_tp(local_dot)

    # ========== Enable/Disable ==========

    def enable(self):
        """Enable metric collection."""
        self._enabled = True

    def disable(self):
        """Disable metric collection and clear input cache."""
        self._enabled = False
        self._input_cache.clear()

    @property
    def is_enabled(self) -> bool:
        """Check if the monitor is currently enabled."""
        return self._enabled

    # ========== Metric Export ==========

    def get_and_clear_metrics(self) -> Dict[str, float]:
        """Get aggregated metrics and clear the buffer.

        For each metric key, returns the mean across all collected values
        (e.g., across microbatches).
        Warning: This triggers a CPU-GPU synchronization.

        Returns:
            A dictionary of metric names to averaged values.
        """
        result = {}
        # Collect all tensors to move to CPU in one go if possible
        # However, since they might be scalars from different steps,
        # let's stack them first.

        for key, values in self._metrics.items():
            if not values:
                continue

            # Stack tensors on GPU
            # values is List[torch.Tensor]
            if len(values) > 0:
                stacked = torch.stack(values)
                # Helper to move to cpu and get mean
                # Synchronize here
                mean_val = stacked.float().mean().item()
                result[key] = mean_val

        self._metrics.clear()
        return result

    def get_aggregated_metrics(self) -> Dict[str, float]:
        """Get cross-module aggregated metrics.

        Returns metrics averaged across all monitored modules.
        Keys are prefixed with 'avg_'.

        Returns:
            A dictionary of aggregated metric names to values.
        """
        all_metrics = self.get_and_clear_metrics()
        aggregated = defaultdict(list)
        for key, value in all_metrics.items():
            # key format: "module_name/metric_name"
            metric_name = key.split("/")[-1]
            aggregated[f"avg_{metric_name}"].append(value)

        return {k: sum(v) / len(v) for k, v in aggregated.items() if v}
