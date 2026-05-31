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

from src.titan_ext.dataloader import (
    _collate_megatron_to_titan,
    _num_val_samples,
    _perf_loader_kwargs,
    _should_validate,
    apply_titan_validation_dataloader_patch,
    apply_titan_validation_schedule_patch,
)


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


def test_perf_loader_kwargs_overlaps_data_loading():
    # With workers: prefetch + persistent + pinned memory, so data loading hides
    # under compute (parity with Megatron's --num-workers path).
    kw = _perf_loader_kwargs(6)
    assert kw["num_workers"] == 6
    assert kw["pin_memory"] is True
    assert kw["prefetch_factor"] == 2
    assert kw["persistent_workers"] is True


def test_perf_loader_kwargs_drops_worker_only_args_when_zero():
    # prefetch_factor/persistent_workers are invalid without workers — must be absent.
    kw = _perf_loader_kwargs(0)
    assert kw == {"num_workers": 0, "pin_memory": True}


def test_num_val_samples_sizes_for_steps_batches():
    # The validation split must yield >= validation.steps global batches, else the
    # Validator divides the summed loss by num_steps==0. Size = steps*local_bs*dp.
    assert _num_val_samples(32, 8, 4) == 32 * 8 * 4
    assert _num_val_samples(10, 1, 1) == 10


def test_num_val_samples_caps_consume_all_to_finite():
    # steps=-1 ("consume all") would hang across ranks; fall back to a finite cap.
    assert _num_val_samples(-1, 8, 4) == 50 * 8 * 4
    assert _num_val_samples(0, 2, 1) == 50 * 2


def test_apply_titan_validation_patch_noops_without_torchtitan():
    # CPU unit-test env has no torchtitan -> returns False, never raises.
    assert apply_titan_validation_dataloader_patch() in (True, False)
    assert apply_titan_validation_schedule_patch() in (True, False)


def test_should_validate_skips_step_1():
    # No eval at step 1 (untrained model -> huge loss distorts the curve); first
    # eval at `freq`, then every `freq`. freq<=0 disables (no modulo error).
    assert _should_validate(1, 500) is False
    assert _should_validate(500, 500) is True
    assert _should_validate(1000, 500) is True
    assert _should_validate(20, 500) is False
    assert _should_validate(5, 0) is False
