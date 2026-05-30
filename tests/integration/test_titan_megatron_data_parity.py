"""First N global batches from the torchtitan dataloader must byte-match the
Megatron GPTDataset on the same (path, seq_len, gbs, seed).

M2 gate (CPU). Skips cleanly when Megatron can't be imported — in the bare dev
env `import megatron.core` raises OSError (transformer_engine/cuBLAS symbol)
rather than ImportError, so a plain pytest.importorskip would error instead of
skip; we catch broadly and skip at module level. With the CUDA env sourced
(`source load_cuda13_2_nccl_env.sh`) the import succeeds and the test runs.
"""

from __future__ import annotations

import pytest

try:  # - need the bound exception for the skip message
    import megatron.core.datasets.gpt_dataset  # noqa: F401
except Exception as exc:  # OSError (TE/cuBLAS) in bare env, or ImportError in CI
    pytest.skip(f"megatron datasets unavailable: {exc}", allow_module_level=True)

from src.titan_ext.dataloader import build_megatron_indexed_batches
from tests.integration._indexed_fixtures import make_tiny_indexed_dataset

pytestmark = pytest.mark.integration


def test_first_batches_match_megatron(tmp_path):
    prefix = make_tiny_indexed_dataset(tmp_path, num_docs=64, doc_len=128, vocab=256, seed=0)
    seq_len, gbs, seed, n = 32, 8, 1234, 4

    mg_batches = build_megatron_indexed_batches(
        path=prefix,
        seq_len=seq_len,
        global_batch_size=gbs,
        seed=seed,
        n_batches=n,
        source="megatron",
    )
    tt_batches = build_megatron_indexed_batches(
        path=prefix,
        seq_len=seq_len,
        global_batch_size=gbs,
        seed=seed,
        n_batches=n,
        source="torchtitan",
    )

    assert len(mg_batches) == len(tt_batches) == n
    for a, b in zip(mg_batches, tt_batches, strict=True):
        assert a.tolist() == b.tolist()
