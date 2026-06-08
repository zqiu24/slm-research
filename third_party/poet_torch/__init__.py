from .adamw import POETAdamW as POETAdamW
from .poet_layer import prepare_model_for_int8_training_poet, QPOETLinear
from .poet_layer import POETLinear as POETLinear
from .poet_layer import replace_linear_with_poet as replace_linear_with_poet
from .poet_layer import check_and_merge as check_and_merge
from .poet_layer import get_grad_clipping_value as get_grad_clipping_value
from .poet_layer import estimate_poet_delta_weff_spec as estimate_poet_delta_weff_spec

# Cayley-manifold variant (no Cayley-Neumann in forward; Adam on Stiefel manifold)
from .cayley_adam import CayleyAdam as CayleyAdamMulti
from .cayley_adam_single import CayleyAdamSingle as CayleyAdam
from .poet_cayley_layer import POETCayleyLinear as POETCayleyLinear
from .poet_cayley_layer import replace_linear_with_poet_cayley as replace_linear_with_poet_cayley
from .poet_cayley_layer import check_and_merge_cayley as check_and_merge_cayley

from .head_aligned_layer import HeadAlignedPOETLinear as HeadAlignedPOETLinear

from .single_step import SingleStepPOETFunction as SingleStepPOETFunction
from .single_step import HeadAlignedSingleStepFunction as HeadAlignedSingleStepFunction

from .single_step_native import NativeSingleStepFunction as NativeSingleStepFunction
from .single_step_native import SingleStepPOETLinear as SingleStepPOETLinear

from .poetx_ops import POETXSingleStepFunction as POETXSingleStepFunction
from .poetx_layer import POETXLinear as POETXLinear
