# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Attention Stats Monitor for tracking attention distribution characteristics.

Monitors proxy statistics of attention outputs to detect:
- Attention collapse (overly concentrated outputs)
- Attention diffusion (overly uniform outputs)

Note: This monitor requires access to attention weights to compute true entropy,
which is not available with some attention implementations (e.g., flash attention).
When attention weights are not available, it falls back to monitoring
output statistics as a proxy.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn

from megatron.core.stability.base_monitor import BaseLayerMonitor, MonitorConfig


@dataclass
class AttentionStatsConfig(MonitorConfig):
    """Configuration for Attention Stats Monitor.

    Attributes:
        entropy_low_threshold: Warning threshold for low entropy (attention collapse).
        entropy_high_threshold: Warning threshold for high entropy (diffuse attention).
        use_qkv_proxy: If True, use QKV statistics when attention weights unavailable.
    """
    entropy_low_threshold: float = 0.1
    entropy_high_threshold: float = 0.95
    use_qkv_proxy: bool = True


class AttentionStatsMonitor(BaseLayerMonitor):
    """
    Attention Stats Monitor for tracking attention distribution characteristics.

    This monitor hooks onto SelfAttention modules and computes:
    - output concentration: Std/mean ratio of attention outputs
    - residual ratio: Relative change introduced by attention
    - sparsity proxy: L1/L2 ratio of attention outputs

    Since flash attention doesn't provide attention weights, this monitor
    computes output-based proxy metrics by default.

    Example:
        config = AttentionStatsConfig(enabled=True, sample_freq=100)
        monitor = AttentionStatsMonitor(config)
        monitor.register(model)
    """



    def __init__(self, config: Optional[AttentionStatsConfig] = None):
        """Initialize the Attention Stats Monitor.

        Args:
            config: Monitor configuration. If None, uses default config.
        """
        if config is None:
            config = AttentionStatsConfig()
        super().__init__(config)
        self._config: AttentionStatsConfig = config

    def register(self, model: nn.Module):
        """Register hooks on self-attention modules (both SelfAttention and MLASelfAttention).

        Args:
            model: The model to register hooks on.
        """
        self._init_distributed()
        from megatron.core.transformer.attention import Attention

        count = 0
        for name, module in model.named_modules():
            # Hook the base Attention class and filter for self-attention type
            # This supports both SelfAttention and MLASelfAttention
            if isinstance(module, Attention) and getattr(module, 'attention_type', None) == 'self':
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
            f"[AttentionStats] Registered hooks on {count} Attention modules"
        )

    def _make_pre_hook(self, name: str):
        """Create a pre-forward hook for caching input."""
        def hook(module, args, kwargs):
            if self._enabled:
                # SelfAttention can receive hidden_states as positional or keyword arg
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
                    except Exception:
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
        """Compute attention-related metrics.

        Since flash attention doesn't expose attention weights, we compute
        proxy metrics based on input/output statistics.

        Args:
            name: Module name.
            module: The SelfAttention module.
            input_tensor: Input hidden states [seq, batch, hidden].
            output_tensor: Output from attention [seq, batch, hidden].

        Returns:
            Dictionary with proxy metrics for attention characteristics.
        """
        # Sample tokens to reduce compute overhead
        x = self._sample_tokens(input_tensor)
        x_out = self._sample_tokens(output_tensor)

        # Compute attention output characteristics
        # High variation in output suggests sharp attention
        # Low variation suggests diffuse attention

        # 1. Output concentration (token-wise): std/mean ratio per token
        # Higher values indicate sharper attention patterns
        out_std = x_out.std(dim=-1, unbiased=False)
        out_mean = x_out.abs().mean(dim=-1)
        concentration_ratio = (out_std / (out_mean + 1e-8)).mean()

        # 2. Residual ratio (token-wise): how much does attention change the input
        residual = x_out - x
        residual_norm = residual.float().norm(dim=-1)
        input_norm = x.float().norm(dim=-1)
        residual_ratio = (residual_norm / (input_norm + 1e-8)).mean()

        # 3. Sparsity proxy (token-wise): L1/L2 norm ratio per token
        # Higher ratio suggests more diffuse (less sparse) patterns
        l1_norm = x_out.abs().sum(dim=-1)
        l2_norm = x_out.norm(dim=-1)
        hidden_dim = x_out.shape[-1]
        # Normalized: 1.0 means uniform, lower means more concentrated
        sparsity_proxy = (l1_norm / (l2_norm * hidden_dim**0.5 + 1e-8)).mean()

        return {
            "concentration": concentration_ratio,
            "residual_ratio": residual_ratio,
            "sparsity_proxy": sparsity_proxy,
        }

    def _sample_tokens(
        self, tensor: torch.Tensor, max_tokens: Optional[int] = None
    ) -> torch.Tensor:
        """Subsample tokens to reduce compute overhead.

        Args:
            tensor: Input tensor of shape [seq, batch, hidden] or [batch, seq, hidden].
            max_tokens: Maximum number of tokens to sample.

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

        indices = torch.randperm(num_tokens, device=tensor.device)[:max_tokens]
        return flat[indices]

    def check_anomalies(self, metrics: Dict[str, float]) -> list:
        """Check for anomalous attention patterns.

        Args:
            metrics: Dictionary of metric names to values.

        Returns:
            List of warning messages for anomalous values.
        """
        alerts = []
        for key, value in metrics.items():
            # Very low concentration might indicate attention collapse
            if "concentration" in key and value < 0.1:
                alerts.append(
                    f"[STABILITY WARNING] {key}={value:.4f} < 0.1 (possible attention collapse)"
                )
            # Very high concentration might indicate instability
            if "concentration" in key and value > 5.0:
                alerts.append(
                    f"[STABILITY WARNING] {key}={value:.4f} > 5.0 (attention output highly variable)"
                )
        return alerts
