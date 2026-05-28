# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Transformer Stats Monitor for tracking training stability.

Computes proxy statistics of transformer layer input/output:
- Residual Velocity: ||f(x)|| / ||x|| (residual update magnitude)
- Cosine Similarity: cos(x, x + f(x)) (trajectory smoothness)
- Variance Growth: Var(x_out) / Var(x_in) (signal explosion indicator)
"""

from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn

from megatron.core.stability.base_monitor import BaseLayerMonitor, MonitorConfig


@dataclass
class TransformerStatsConfig(MonitorConfig):
    """Configuration for Transformer Stats Monitor.

    Attributes:
        velocity_threshold: Warning threshold for feature velocity.
        cos_sim_threshold: Warning threshold for cosine similarity (below this).
        var_growth_threshold: Warning threshold for variance growth.
    """
    velocity_threshold: float = 1.5
    cos_sim_threshold: float = 0.5
    var_growth_threshold: float = 2.0


class TransformerStatsMonitor(BaseLayerMonitor):
    """
    Transformer Stats Monitor for transformer layer stability.

    Tracks three key stability metrics for each transformer layer:
    - velocity: ||f(x)|| / ||x|| where f(x) = x_out - x
    - cos_sim: cosine similarity between x and x_out
    - var_growth: variance ratio between output and input

    For MoE layers, monitors the overall layer input/output, which are
    replicated across EP ranks, avoiding any EP communication overhead.

    Example:
        config = TransformerStatsConfig(enabled=True, sample_freq=100)
        monitor = TransformerStatsMonitor(config)
        monitor.register(model)

        # In training loop
        if iteration % config.sample_freq == 0:
            monitor.enable()
        # ... forward pass ...
        metrics = monitor.get_aggregated_metrics()
    """

    # Import here to avoid circular imports
    @property
    def target_module_class(self):
        from megatron.core.transformer.transformer_layer import TransformerLayer
        return TransformerLayer

    def __init__(self, config: Optional[TransformerStatsConfig] = None):
        """Initialize the Transformer Stats Monitor.

        Args:
            config: Monitor configuration. If None, uses default config.
        """
        if config is None:
            config = TransformerStatsConfig()
        super().__init__(config)
        self._config: TransformerStatsConfig = config

    def register(self, model: nn.Module):
        """Register hooks on TransformerLayer modules.

        Args:
            model: The model to register hooks on.
        """
        self._init_distributed()
        from megatron.core.transformer.transformer_layer import TransformerLayer

        count = 0
        for name, module in model.named_modules():
            if isinstance(module, TransformerLayer):
                pre_hook = module.register_forward_pre_hook(
                    self._make_pre_hook(name),
                    with_kwargs=True  # Enable kwargs access in hook
                )
                post_hook = module.register_forward_hook(
                    self._make_post_hook(name)
                )
                self._hooks.extend([pre_hook, post_hook])
                count += 1

        # Debug: confirm registration
        from megatron.training.utils import print_rank_0
        print_rank_0(
            f"[TransformerStats] Registered hooks on {count} TransformerLayer modules"
        )

    def _make_pre_hook(self, name: str):
        """Create a pre-forward hook for caching input."""
        def hook(module, args, kwargs):
            if self._enabled:
                # TransformerLayer uses kwargs, not positional args
                # The first argument is 'hidden_states'
                if 'hidden_states' in kwargs:
                    captured = kwargs['hidden_states']
                elif args and len(args) > 0:
                    captured = args[0]
                else:
                    return

                if isinstance(captured, torch.Tensor):
                    self._input_cache[name] = captured.detach()
        return hook

    def _make_post_hook(self, name: str):
        """Create a post-forward hook for computing metrics."""
        def hook(module, inputs, outputs):
            if self._enabled and name in self._input_cache:
                input_tensor = self._input_cache.pop(name)
                output_tensor = self._capture_output(outputs)
                if output_tensor is not None:
                    try:
                        metrics = self.compute_metrics(
                            name, module, input_tensor, output_tensor
                        )
                        for k, v in metrics.items():
                            self._metrics[f"{name}/{k}"].append(v)
                    except Exception as e:
                        # Silently ignore errors to avoid disrupting training
                        pass
        return hook


    def compute_metrics(
        self,
        name: str,
        module: nn.Module,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute feature evolution metrics.

        Args:
            name: Module name.
            module: The TransformerLayer module.
            input_tensor: Input hidden states [seq, batch, hidden].
            output_tensor: Output hidden states [seq, batch, hidden].

        Returns:
            Dictionary with keys: velocity, cos_sim, var_growth.
        """
        # Sample tokens to reduce compute overhead
        x = self._sample_tokens(input_tensor)
        x_out = self._sample_tokens(output_tensor)

        # Compute residual f(x) = x_out - x
        f_x = x_out - x

        # Feature Velocity (token-wise): mean over tokens of ||f(x)|| / ||x||
        # This is more stable than a global norm ratio for layerwise comparisons.
        x_norm = x.float().norm(dim=-1)
        fx_norm = f_x.float().norm(dim=-1)
        velocity = (fx_norm / (x_norm + 1e-8)).mean()

        # Cosine Similarity (token-wise): mean over tokens of cosine similarity
        # This is more stable than a global cosine for layerwise comparisons.
        x_out_norm = x_out.float().norm(dim=-1)
        dot_token = (x.float() * x_out.float()).sum(dim=-1)
        cos_token = dot_token / (x_norm * x_out_norm + 1e-8)
        cos_sim = cos_token.mean()

        # Variance Growth (token-wise): mean over tokens of hidden-dim variance
        # This is more stable for post-norm detection than global variance.
        # Note: For TP, variance is approximate (computed on local shard)
        var_x_token = x.var(dim=-1, unbiased=False).mean()
        var_xout_token = x_out.var(dim=-1, unbiased=False).mean()
        var_growth = var_xout_token / (var_x_token + 1e-8)

        return {
            "velocity": velocity,
            "cos_sim": cos_sim,
            "var_growth": var_growth,
        }

    def _sample_tokens(
        self, tensor: torch.Tensor, max_tokens: Optional[int] = None
    ) -> torch.Tensor:
        """Subsample tokens to reduce compute overhead.

        Args:
            tensor: Input tensor of shape [seq, batch, hidden] or [batch, seq, hidden].
            max_tokens: Maximum number of tokens to sample.
                        Defaults to config.sample_tokens.

        Returns:
            Flattened tensor of shape [min(num_tokens, max_tokens), hidden].
        """
        if max_tokens is None:
            max_tokens = self.config.sample_tokens

        # Flatten to [num_tokens, hidden]
        flat = tensor.reshape(-1, tensor.shape[-1])
        num_tokens = flat.shape[0]

        if num_tokens <= max_tokens:
            return flat

        # Random sampling
        indices = torch.randperm(num_tokens, device=tensor.device)[:max_tokens]
        return flat[indices]

    def check_anomalies(self, metrics: Dict[str, float]) -> list:
        """Check for anomalous metric values.

        Args:
            metrics: Dictionary of metric names to values.

        Returns:
            List of warning messages for anomalous values.
        """
        alerts = []
        for key, value in metrics.items():
            if "velocity" in key and value > self._config.velocity_threshold:
                alerts.append(
                    f"[STABILITY WARNING] {key}={value:.4f} > {self._config.velocity_threshold} "
                    f"(feature velocity too high)"
                )
            if "cos_sim" in key and value < self._config.cos_sim_threshold:
                alerts.append(
                    f"[STABILITY WARNING] {key}={value:.4f} < {self._config.cos_sim_threshold} "
                    f"(trajectory disruption)"
                )
            if "var_growth" in key and value > self._config.var_growth_threshold:
                alerts.append(
                    f"[STABILITY WARNING] {key}={value:.4f} > {self._config.var_growth_threshold} "
                    f"(variance explosion)"
                )
        return alerts
