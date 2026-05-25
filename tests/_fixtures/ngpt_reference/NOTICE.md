# nGPT reference fixture

`model.py` is a verbatim copy of `model.py` from
https://github.com/NVIDIA/ngpt (MIT License, Copyright (c) 2024 NVIDIA
CORPORATION & AFFILIATES). One surgical change: the top-level
`flash_attn` import is wrapped in a try/except so the file imports on
CPU, falling back to `torch.nn.functional.scaled_dot_product_attention`.

This file exists **solely as the numerical oracle for the parity test
in tests/unit/test_ngpt_full_parity.py**. It is never imported from
`src/` and never used at training time.
