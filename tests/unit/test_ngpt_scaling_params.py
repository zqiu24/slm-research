"""CPU tests for the _LearnedScaling helper."""

import torch

from src.model.ngpt.scaling_params import LearnedScaling


def test_storage_matches_init_scaling():
    ls = LearnedScaling(shape=(4,), init_value=1.0, init_scaling=0.1)
    assert ls.param.shape == (4,)
    assert ls.param.dtype == torch.float32
    assert torch.allclose(ls.param.data, 0.1 * torch.ones(4))


def test_scaled_value_matches_init_value_at_init():
    ls = LearnedScaling(shape=(4,), init_value=0.05, init_scaling=1.0 / 8.0)
    expected = (1.0 / 8.0) * torch.ones(4) * (0.05 / (1.0 / 8.0))
    assert torch.allclose(ls.scaled_value(), expected)
    # i.e. uniform 0.05
    assert torch.allclose(ls.scaled_value(), 0.05 * torch.ones(4))


def test_scaled_value_scales_with_param_data():
    ls = LearnedScaling(shape=(3,), init_value=2.0, init_scaling=0.5)
    ls.param.data.copy_(torch.tensor([0.5, 1.0, 1.5]))
    # multiplier is init_value/init_scaling = 4.0
    expected = torch.tensor([2.0, 4.0, 6.0])
    assert torch.allclose(ls.scaled_value(), expected)


def test_is_registered_as_nn_module_with_one_param():
    ls = LearnedScaling(shape=(2,), init_value=1.0, init_scaling=1.0)
    params = list(ls.parameters())
    assert len(params) == 1
    assert params[0] is ls.param
