# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Stability monitoring utilities for integration with Megatron-LM training.

This module provides utility functions to set up, manage, and log stability
monitoring metrics during training.
"""

from typing import Dict, Optional, Any
import torch.nn as nn

# Global registry for stability monitors
_STABILITY_REGISTRY = None



def setup_stability_monitors(model: nn.Module, args) -> Optional[Any]:
    """Set up stability monitors based on command-line arguments.

    This should be called after model creation in setup_model_and_optimizer().

    Args:
        model: The model to attach monitors to.
        args: Command-line arguments from get_args().

    Returns:
        The MonitorRegistry instance, or None if no monitors are enabled.
    """
    global _STABILITY_REGISTRY

    # Check if any monitors are enabled
    if not (
        getattr(args, 'enable_transformer_stats_monitor', False) or
        getattr(args, 'enable_attention_stats_monitor', False)
    ):
        return None

    # Import here to avoid circular imports
    from megatron.core.stability import (
        MonitorRegistry,
        TransformerStatsMonitor,
        AttentionStatsMonitor,
    )
    from megatron.core.stability.transformer_stats_monitor import TransformerStatsConfig
    from megatron.core.stability.attention_stats_monitor import AttentionStatsConfig
    from megatron.training.utils import print_rank_0

    registry = MonitorRegistry()

    # Setup Transformer Stats Monitor
    if getattr(args, 'enable_transformer_stats_monitor', False):
        config = TransformerStatsConfig(
            enabled=True,
            sample_freq=getattr(args, 'stability_monitor_freq', 100),
            sample_tokens=getattr(args, 'stability_monitor_sample_tokens', 256),
            log_per_module=getattr(args, 'stability_log_per_layer', False),
            velocity_threshold=getattr(args, 'stability_velocity_threshold', 1.5),
            cos_sim_threshold=getattr(args, 'stability_cos_sim_threshold', 0.5),
            var_growth_threshold=getattr(args, 'stability_var_growth_threshold', 2.0),
        )
        registry.register_monitor("transformer_stats", TransformerStatsMonitor(config))
        print_rank_0("> Enabled Transformer Stats Monitor")

    # Setup Attention Stats Monitor
    if getattr(args, 'enable_attention_stats_monitor', False):
        config = AttentionStatsConfig(
            enabled=True,
            sample_freq=getattr(args, 'stability_monitor_freq', 100),
            sample_tokens=getattr(args, 'stability_monitor_sample_tokens', 256),
            log_per_module=getattr(args, 'stability_log_per_layer', False),
        )
        registry.register_monitor("attention_stats", AttentionStatsMonitor(config))
        print_rank_0("> Enabled Attention Stats Monitor")

    # Register hooks on all model chunks
    if isinstance(model, list):
        for model_chunk in model:
            registry.register_all(model_chunk)
    else:
        registry.register_all(model)

    # Debug: Print hook counts
    for name, monitor in registry._monitors.items():
        num_hooks = len(monitor._hooks)
        print_rank_0(f"> Monitor '{name}': registered {num_hooks} hooks")


    _STABILITY_REGISTRY = registry
    return registry


def get_stability_registry() -> Optional[Any]:
    """Get the global stability monitor registry.

    Returns:
        The MonitorRegistry instance, or None if not set up.
    """
    return _STABILITY_REGISTRY


def update_stability_monitors(iteration: int, args) -> None:
    """Enable/disable stability monitors based on sampling frequency.

    Call this at the start of each training step.

    Args:
        iteration: Current training iteration.
        args: Command-line arguments from get_args().
    """
    registry = get_stability_registry()
    if registry is None:
        return

    sample_freq = getattr(args, 'stability_monitor_freq', 100)
    if iteration % sample_freq == 0:
        registry.enable_all()
    else:
        registry.disable_all()


def log_stability_metrics(
    iteration: int,
    args,
    writer: Optional[Any] = None,
    wandb_writer: Optional[Any] = None,
) -> Dict[str, float]:
    """Collect and log stability metrics.

    Call this during training_log().

    Args:
        iteration: Current training iteration.
        args: Command-line arguments from get_args().
        writer: TensorBoard SummaryWriter instance.
        wandb_writer: WandB writer instance.

    Returns:
        Dictionary of aggregated metrics.
    """
    registry = get_stability_registry()
    if registry is None:
        return {}

    sample_freq = getattr(args, 'stability_monitor_freq', 100)
    if iteration % sample_freq != 0:
        return {}

    # Debug: Print what metrics we got from each monitor (without clearing)
    if getattr(args, 'stability_monitor_debug', False):
        from megatron.training.utils import print_rank_0
        for monitor_name, monitor in registry._monitors.items():
            num_metrics = len(monitor._metrics)  # Check buffer size without clearing
            print_rank_0(
                f"[DEBUG] Monitor '{monitor_name}': has {num_metrics} metric keys in buffer"
            )
            sample_keys = list(monitor._metrics.keys())[:3]
            print_rank_0(f"[DEBUG]   Sample keys: {sample_keys}")

    # IMPORTANT: Only call registry get methods once (they clear the buffer!)
    log_per_layer = getattr(args, 'stability_log_per_layer', False)
    aggregated_metrics, per_layer_metrics = registry.get_metrics(log_per_layer)
    tb_wandb_metrics = per_layer_metrics if log_per_layer else aggregated_metrics

    if not aggregated_metrics and not tb_wandb_metrics:
        return {}


    # Log to TensorBoard
    if writer is not None:
        # Log aggregated metrics at TOP LEVEL (alongside grad_norm, loss, etc.)
        for key, value in aggregated_metrics.items():
            # Remove 'avg_' prefix for cleaner names
            metric_name = key.replace('avg_', '')
            writer.add_scalar(metric_name, value, iteration)
            writer.add_scalar(f'{metric_name} vs samples', value,
                            getattr(args, 'consumed_train_samples', 0))

        # If per-layer flag is set, log detailed metrics under stability/ namespace
        if log_per_layer and tb_wandb_metrics != aggregated_metrics:
            for key, value in tb_wandb_metrics.items():
                writer.add_scalar(f'stability/{key}', value, iteration)

    # Log to WandB
    if wandb_writer is not None:
        # Log aggregated at TOP LEVEL
        wandb_metrics = {}
        for key, value in aggregated_metrics.items():
            metric_name = key.replace('avg_', '')
            wandb_metrics[metric_name] = value

        # If per-layer flag is set, log detailed metrics under stability/ namespace
        if log_per_layer and tb_wandb_metrics != aggregated_metrics:
            for key, value in tb_wandb_metrics.items():
                wandb_metrics[f'stability/{key}'] = value

        wandb_writer.log(wandb_metrics, iteration)

    # Check for anomalies and print warnings (using monitor-defined checks)
    if aggregated_metrics and getattr(args, 'check_stability_anomalies', False):
        from megatron.core.stability.registry import check_all_anomalies
        from megatron.training.utils import print_rank_0
        check_all_anomalies(registry, aggregated_metrics, printer=print_rank_0)

    # Return aggregated metrics for console display
    return aggregated_metrics


def get_stability_log_string(metrics: Dict[str, float]) -> str:
    """Generate a log string for stability metrics.

    Args:
        metrics: Dictionary of aggregated metrics.

    Returns:
        Formatted string for console logging.
    """
    if not metrics:
        return ""

    # Extract key metrics for console display
    parts = []
    for key, value in sorted(metrics.items()):
        # For per-layer metrics, keep the layer identifier
        # For aggregated metrics, simplify the name
        parts_list = key.split('/')

        if len(parts_list) >= 3:
            # Per-layer format: "monitor_name/layer_name/metric_name"
            # Extract layer number if present
            layer_part = parts_list[-2]  # e.g., "module.module.decoder.layers.0.self_attention"
            metric_name = parts_list[-1]  # e.g., "concentration"

            # Try to extract layer number from anywhere in the path
            import re
            layer_match = re.search(r'layers\.(\d+)', layer_part)
            if layer_match:
                layer_num = layer_match.group(1)

                short_key = f"layer{layer_num}_{metric_name}"
            else:
                # No layer number found, use abbreviated form
                short_key = f"{metric_name}"
        else:
            # Aggregated format: "monitor_name/avg_metric_name"
            short_key = parts_list[-1]
            if short_key.startswith('avg_'):
                short_key = short_key[4:]

        parts.append(f"{short_key}: {value:.4f}")

    return " | ".join(parts)
