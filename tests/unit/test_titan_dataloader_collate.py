"""The Megatron→torchtitan batch collate contract.

torchtitan's training loop unpacks every batch as `input_dict, labels = batch`
(third_party/torchtitan/torchtitan/train.py `batch_generator`) and then reads
`input_dict["input"]` as the model input (`post_dataloading_process`). Megatron's
GPTDataset yields a per-sample DICT (`tokens`/`labels`/`attention_mask`/
`loss_mask`/`position_ids`); without a collate that reshapes a batch into
torchtitan's `(input_dict, labels)` 2-tuple, the default collate produces a
5-key dict and the unpack raises `ValueError: too many values to unpack
(expected 2)`. These tests lock the collate to the 2-tuple contract.
"""

from __future__ import annotations

import torch

from src.titan_ext.dataloader import _collate_megatron_to_titan


def _sample(seq_len: int, fill: int) -> dict:
    # Mirrors GPTDataset.__getitem__: tokens/labels are already torch.long and
    # pre-shifted (tokens = text[:-1], labels = text[1:]). The extra keys must be
    # DROPPED by the collate — torchtitan's llama3 builds its own causal mask + RoPE.
    return {
        "tokens": torch.arange(fill, fill + seq_len, dtype=torch.long),
        "labels": torch.arange(fill + 1, fill + 1 + seq_len, dtype=torch.long),
        "attention_mask": torch.ones(1, seq_len, seq_len, dtype=torch.bool),
        "loss_mask": torch.ones(seq_len, dtype=torch.float),
        "position_ids": torch.arange(seq_len, dtype=torch.long),
    }


def test_collate_returns_input_dict_labels_2tuple():
    bs, seq = 3, 8
    batch = _collate_megatron_to_titan([_sample(seq, i * 100) for i in range(bs)])
    # The exact unpack torchtitan's batch_generator performs — must not raise.
    input_dict, labels = batch
    assert set(input_dict) == {"input"}  # extras dropped; only "input" survives
    assert input_dict["input"].shape == (bs, seq)
    assert labels.shape == (bs, seq)
    # Embedding lookup + cross-entropy targets both require int64.
    assert input_dict["input"].dtype == torch.long
    assert labels.dtype == torch.long


def test_collate_preserves_tokens_and_megatron_shift():
    # tokens flow to input unchanged; labels stay Megatron's pre-shifted next tokens.
    s = _sample(8, 0)
    input_dict, labels = _collate_megatron_to_titan([s])
    assert torch.equal(input_dict["input"][0], s["tokens"])
    assert torch.equal(labels[0], s["labels"])
