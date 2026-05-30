"""Size an slm flavor by cloning a torchtitan NATIVE model-args template.

We do not construct any family-specific model-args class ourselves. Instead we
take an existing flavor of the same family from the native TrainSpec's model_args
registry (the "template") and dataclasses.replace() the dimension fields slm
sets, ignoring any slm field the template doesn't model (e.g. ffn_hidden_size —
torchtitan sizes the FFN from `dim` natively for llama3). This keeps torchtitan's
own model class + FFN/MoE conventions, so the flavor "qualifies as" that family.
No torchtitan import here, so the helper is unit-testable on CPU with a fake args
dataclass.
"""

from __future__ import annotations

import dataclasses


def build_slm_flavor(template, dims: dict):
    """Return a copy of native `template` with slm's dims applied.

    Only fields the template actually has are overridden — across families the
    args classes differ, and slm sets fields (ffn_hidden_size) torchtitan derives
    rather than stores. Per the slm-side goal we honor layer/hidden/head/vocab
    counts (so a "300m" is ~300m) and let everything else follow native defaults.
    """
    overrides = {k: v for k, v in dims.items() if hasattr(template, k)}
    return dataclasses.replace(template, **overrides)


def pick_template(model_args: dict):
    """Pick a deterministic template flavor from a native model_args registry."""
    for key in ("debugmodel", "debug", "1B", "8B"):
        if key in model_args:
            return model_args[key]
    return next(iter(model_args.values()))  # any registered flavor of this family
