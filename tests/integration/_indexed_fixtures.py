"""Helpers to write a tiny Megatron .bin/.idx dataset for CPU parity tests.

Uses the vendored mcore IndexedDatasetBuilder (verified API: __init__(bin_path,
dtype=numpy.int32), add_item(torch.Tensor), end_document(), finalize(idx_path);
files are `<prefix>.bin` / `<prefix>.idx`).
"""

from __future__ import annotations

import numpy as np


def make_tiny_indexed_dataset(tmp_path, *, num_docs=64, doc_len=128, vocab=256, seed=0) -> str:
    """Write `num_docs` random documents of `doc_len` int32 tokens and return the
    `text_document` prefix (the path WITHOUT .bin/.idx, which GPTDataset's blend
    expects). int32 matches the >65536-vocab token_dtype_code GPTDataset computes,
    so the reader and writer dtypes agree."""
    import torch  # local: torch import is heavy; keep module import light
    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

    prefix = str(tmp_path / "text_document")
    rng = np.random.default_rng(seed)
    builder = IndexedDatasetBuilder(prefix + ".bin", dtype=np.int32)
    for _ in range(num_docs):
        toks = rng.integers(0, vocab, size=doc_len, dtype=np.int32)
        builder.add_item(torch.from_numpy(toks))
        builder.end_document()
    builder.finalize(prefix + ".idx")
    return prefix
