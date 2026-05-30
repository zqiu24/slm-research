"""torchtitan dataloader backed by Megatron's GPTDataset.

Both backends build the SAME megatron.core.datasets.GPTDataset (same indexing,
same shuffle seed), so token order is identical by construction. The
`source="torchtitan"` path additionally wraps it in torchtitan's
ParallelAwareDataloader (API per docs/torchtitan_api_notes.md §4); the
`source="megatron"` path iterates the GPTDataset directly for the parity test.

No torch / torchtitan / megatron import at module load — every heavy import is
lazy inside a function, so this module imports cleanly in the CPU unit-test env.
"""

from __future__ import annotations

import numpy as np

# Default vocab for the parity path (no cfg available there). >65536 so Megatron's
# token_dtype_code resolves to int32, matching the int32 .bin the fixture writes
# and the real llama3 corpus (vocab 128256). build_dataloader passes the real
# nominal vocab from the resolved config instead.
_DEFAULT_VOCAB = 131072


class _FixedVocabTokenizer:
    """Minimal stand-in tokenizer.

    GPTDatasetConfig.__post_init__ asserts `tokenizer is not None` and reads
    `tokenizer.vocab_size` (verified against the vendored mcore source).
    MegatronDataset.__init__ also reads `tokenizer.unique_identifiers` and
    json-dumps it into the dataset's cache key. The corpus is pre-tokenized
    .bin/.idx, so no encode/decode is needed on the train path — only vocab_size
    (dtype sizing), eod (unused while eod_mask_loss is False), and
    unique_identifiers (cache key) are touched.
    """

    def __init__(self, vocab_size: int):
        self._vocab_size = int(vocab_size)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eod(self) -> int:
        return 0

    @property
    def unique_identifiers(self) -> dict:
        # JSON-serializable: MegatronDataset hashes this into its cache key.
        return {"class": "FixedVocabTokenizer", "vocab_size": self._vocab_size}


def _build_gpt_dataset(
    path: str, seq_len: int, num_samples: int, seed: int, vocab_size: int, path_to_cache=None
):
    from megatron.core.datasets.blended_megatron_dataset_builder import (
        BlendedMegatronDatasetBuilder,
    )
    from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig

    # Reseed numpy's global RNG so the (uncached) shuffle is reproducible even when
    # each call uses a private cache dir — the parity gate needs two independent
    # fresh builds to be bit-identical, not merely equal-via-shared-cache.
    np.random.seed(int(seed))
    # Build a single-split GPTDataset; field names track the vendored mcore pin
    # (docs/torchtitan_api_notes.md cross-checks these against the source). With
    # path_to_cache=None Megatron caches next to the data prefix (production); the
    # parity path passes a private dir so the two builds never race on one file.
    config = GPTDatasetConfig(
        random_seed=seed,
        sequence_length=seq_len,
        blend=([path], None),
        split="100,0,0",
        path_to_cache=path_to_cache,
        tokenizer=_FixedVocabTokenizer(vocab_size),
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
    )
    builder = BlendedMegatronDatasetBuilder(GPTDataset, [num_samples, 0, 0], lambda: True, config)
    train, _, _ = builder.build()
    return train


def build_megatron_indexed_batches(
    *, path, seq_len, global_batch_size, seed, n_batches, source, vocab_size=_DEFAULT_VOCAB
):
    """Return the first `n_batches` global batches as np arrays of token ids.

    `source` selects the construction path; both build the same GPTDataset (so the
    batches are byte-identical by construction — that is exactly what the M2 parity
    test asserts).
    """
    import tempfile

    num_samples = global_batch_size * n_batches
    # Private cache dir per call: each build computes fresh (no shared cache file
    # to race on), and the numpy reseed makes the fresh shuffle deterministic, so
    # the two sources are bit-identical by construction.
    cache_dir = tempfile.mkdtemp(prefix=f"slm_titan_parity_{source}_")
    ds = _build_gpt_dataset(path, seq_len, num_samples, seed, vocab_size, path_to_cache=cache_dir)
    batches = []
    for b in range(n_batches):
        rows = [
            np.asarray(ds[b * global_batch_size + i]["tokens"]) for i in range(global_batch_size)
        ]
        batches.append(np.stack(rows, axis=0))
    return batches


def build_dataloader(*, dp_world_size, dp_rank, tokenizer, job_config):
    """TrainSpec build_dataloader_fn. VERIFIED signature (api notes §2/§4):
    (dp_world_size, dp_rank, tokenizer, job_config). `tokenizer` is unused — the
    corpus is pre-tokenized .bin/.idx. Data `path`/`seed`/`vocab` come from
    SLM_RESOLVED_CONFIG (single source of truth); seq_len/steps/batch from
    job_config."""
    import os

    from omegaconf import OmegaConf
    from torchtitan.components.dataloader import ParallelAwareDataloader  # verified §4

    slm = OmegaConf.load(os.environ["SLM_RESOLVED_CONFIG"])
    path, seed = str(slm.data.path), int(slm.seed)
    vocab_size = int(slm.base.tokenizer.nominal_vocab_size)
    seq_len = job_config.training.seq_len
    local_bs = job_config.training.local_batch_size  # per-DP-rank batch size
    num_samples = job_config.training.steps * job_config.training.global_batch_size
    ds = _build_gpt_dataset(path, seq_len, num_samples, seed, vocab_size)
    # ParallelAwareDataloader(dataset, dp_rank, dp_world_size, **kwargs) — verified §4.
    return ParallelAwareDataloader(ds, dp_rank, dp_world_size, batch_size=local_bs)
