# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Monitor Registry for unified management of multiple stability monitors.

Provides a centralized interface for:
- Registering multiple monitors
- Enabling/disabling monitors by name
- Collecting metrics from all monitors
- Logging metrics to TensorBoard/WandB
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
import torch
import torch.nn as nn

from megatron.core.stability.base_monitor import BaseLayerMonitor


class MonitorRegistry:
    """
    Unified registry for managing multiple stability monitors.

    Example:
        registry = MonitorRegistry()
        registry.register_monitor("transformer_stats", TransformerStatsMonitor(config))
        registry.register_monitor("attention_stats", AttentionStatsMonitor(config))
        registry.register_all(model)

        # In training loop
        if iteration % sample_freq == 0:
            registry.enable_all()
        # ... forward pass ...
        metrics = registry.get_aggregated_metrics()
    """

    def __init__(self):
        """Initialize the monitor registry."""
        self._monitors: Dict[str, BaseLayerMonitor] = {}

    def register_monitor(self, name: str, monitor: BaseLayerMonitor):
        """Register a monitor with a unique name.

        Args:
            name: Unique identifier for the monitor.
            monitor: The monitor instance to register.
        """
        if name in self._monitors:
            raise ValueError(f"Monitor '{name}' already registered")
        self._monitors[name] = monitor

    def unregister_monitor(self, name: str):
        """Unregister a monitor by name.

        Args:
            name: The name of the monitor to remove.
        """
        if name in self._monitors:
            self._monitors[name].unregister()
            del self._monitors[name]

    def register_all(self, model: nn.Module):
        """Register hooks from all monitors onto the model.

        Args:
            model: The model to attach hooks to.
        """
        for monitor in self._monitors.values():
            monitor.register(model)

    def unregister_all(self):
        """Remove all hooks from all monitors."""
        for monitor in self._monitors.values():
            monitor.unregister()

    def enable_all(self):
        """Enable metric collection for all monitors."""
        for monitor in self._monitors.values():
            monitor.enable()

    def disable_all(self):
        """Disable metric collection for all monitors."""
        for monitor in self._monitors.values():
            monitor.disable()

    def enable(self, *names: str):
        """Enable specific monitors by name.

        Args:
            *names: Names of monitors to enable.
        """
        for name in names:
            if name in self._monitors:
                self._monitors[name].enable()

    def disable(self, *names: str):
        """Disable specific monitors by name.

        Args:
            *names: Names of monitors to disable.
        """
        for name in names:
            if name in self._monitors:
                self._monitors[name].disable()

    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Collect metrics from all monitors (per-module metrics).

        Returns:
            Nested dictionary: {monitor_name: {metric_key: value}}.
        """
        return {
            name: monitor.get_and_clear_metrics()
            for name, monitor in self._monitors.items()
        }

    def get_aggregated_metrics(self) -> Dict[str, float]:
        """Collect aggregated metrics from all monitors (flattened).

        Returns:
            Flat dictionary with keys like "monitor_name/avg_metric_name".
        """
        result = {}
        for name, monitor in self._monitors.items():
            for metric_key, value in monitor.get_aggregated_metrics().items():
                result[f"{name}/{metric_key}"] = value
        return result

    def get_per_layer_metrics(self) -> Dict[str, float]:
        """Collect per-layer metrics from all monitors.

        Returns:
            Flat dictionary with keys like "monitor_name/layer_x/metric_name".
        """
        result = {}
        for name, monitor in self._monitors.items():
            for metric_key, value in monitor.get_and_clear_metrics().items():
                result[f"{name}/{metric_key}"] = value
        return result

    def _aggregate_per_layer_metrics(
        self, per_layer_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Aggregate per-layer metrics into per-monitor averages.

        Args:
            per_layer_metrics: Flat dict with keys like
                "monitor_name/layer_x/metric_name".

        Returns:
            Flat dict with keys like "monitor_name/avg_metric_name".
        """
        from collections import defaultdict

        sums = defaultdict(float)
        counts = defaultdict(int)
        for key, value in per_layer_metrics.items():
            parts = key.split('/')
            if len(parts) < 3:
                continue
            monitor_name = parts[0]
            metric_name = parts[-1]
            agg_key = f"{monitor_name}/avg_{metric_name}"
            sums[agg_key] += value
            counts[agg_key] += 1

        return {k: sums[k] / counts[k] for k in sums if counts[k]}

    def get_metrics(
        self, log_per_layer: bool = False
    ) -> Tuple[Dict[str, float], Optional[Dict[str, float]]]:
        """Collect aggregated metrics and (optionally) per-layer metrics.

        Args:
            log_per_layer: Whether to return per-layer metrics.

        Returns:
            Tuple of (aggregated_metrics, per_layer_metrics or None).
        """
        if log_per_layer:
            per_layer_metrics = self.get_per_layer_metrics()
            aggregated_metrics = self._aggregate_per_layer_metrics(per_layer_metrics)
            return aggregated_metrics, per_layer_metrics

        return self.get_aggregated_metrics(), None

    @property
    def monitor_names(self) -> List[str]:
        """Get list of registered monitor names."""
        return list(self._monitors.keys())

    def __len__(self) -> int:
        """Return number of registered monitors."""
        return len(self._monitors)

    def __contains__(self, name: str) -> bool:
        """Check if a monitor is registered."""
        return name in self._monitors


def log_stability_metrics(
    metrics: Dict[str, float],
    writer: Optional[Any] = None,
    wandb_writer: Optional[Any] = None,
    iteration: int = 0,
    samples: int = 0,
    prefix: str = "stability",
):
    """Log stability metrics to TensorBoard and/or WandB.

    Args:
        metrics: Dictionary of metric names to values.
        writer: TensorBoard SummaryWriter instance.
        wandb_writer: WandB run instance.
        iteration: Current training iteration.
        samples: Current number of consumed samples.
        prefix: Prefix for metric names in logs.
    """
    for key, value in metrics.items():
        full_key = f"{prefix}/{key}"

        # TensorBoard
        if writer is not None:
            writer.add_scalar(full_key, value, iteration)
            if samples > 0:
                writer.add_scalar(f"{full_key} vs samples", value, samples)

        # WandB
        if wandb_writer is not None:
            wandb_writer.log({full_key: value}, iteration)


def check_all_anomalies(
    monitors: MonitorRegistry,
    metrics: Dict[str, float],
    printer: Optional[Callable[[str], None]] = None,
) -> List[str]:
    """Check for anomalies across all monitors.

    Args:
        monitors: The monitor registry.
        metrics: Aggregated metrics from get_aggregated_metrics().
        printer: Optional function to print warnings.

    Returns:
        List of warning messages.
    """
    alerts = []
    for name, monitor in monitors._monitors.items():
        if hasattr(monitor, 'check_anomalies'):
            monitor_metrics = {
                k.replace(f"{name}/", ""): v
                for k, v in metrics.items()
                if k.startswith(f"{name}/")
            }
            for alert in monitor.check_anomalies(monitor_metrics):
                alerts.append(alert)
                if printer is not None:
                    printer(alert)
    return alerts
