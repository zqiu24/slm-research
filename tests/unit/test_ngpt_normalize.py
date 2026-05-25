"""CPU tests for nGPT hypersphere normalization primitives."""

import torch
import torch.nn as nn

from src.model.ngpt.normalize import justnorm, normalize_module_matrices


def test_justnorm_unit_norm_last_dim():
    x = torch.randn(3, 5, 7)
    y = justnorm(x)
    assert torch.allclose(y.norm(dim=-1), torch.ones(3, 5), atol=1e-5)


def test_justnorm_unit_norm_explicit_dim():
    x = torch.randn(3, 5, 7)
    y = justnorm(x, dim=1)
    assert torch.allclose(y.norm(dim=1), torch.ones(3, 7), atol=1e-5)


def test_justnorm_preserves_dtype_bf16():
    x = torch.randn(2, 4, dtype=torch.bfloat16)
    y = justnorm(x)
    assert y.dtype == torch.bfloat16
    # cast to fp32 before checking the norm: bf16 mantissa is too short
    assert torch.allclose(y.float().norm(dim=-1), torch.ones(2), atol=1e-2)


def test_normalize_module_matrices_rows_unit_norm():
    lin = nn.Linear(8, 4, bias=False)
    normalize_module_matrices({lin.weight: "rows"})
    # rows: shape (out, in)=(4,8); each of the 4 rows must be unit-norm.
    assert torch.allclose(lin.weight.data.norm(dim=1), torch.ones(4), atol=1e-5)


def test_normalize_module_matrices_cols_unit_norm():
    lin = nn.Linear(8, 4, bias=False)
    normalize_module_matrices({lin.weight: "cols"})
    # cols: each of the 8 columns must be unit-norm.
    assert torch.allclose(lin.weight.data.norm(dim=0), torch.ones(8), atol=1e-5)


def test_normalize_module_matrices_rejects_bad_role():
    lin = nn.Linear(2, 2, bias=False)
    import pytest

    with pytest.raises(ValueError, match="role"):
        normalize_module_matrices({lin.weight: "diag"})
