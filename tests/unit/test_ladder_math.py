"""Unit tests for ladder math (SPEC.md §1.3, §1.4, §10.1)."""

from __future__ import annotations

import pytest

from src.utils.ladder_math import (
    embedding_params,
    parse_token_count,
    steps_from_tokens,
    total_tokens,
)


def test_total_tokens_basic():
    # 1.2B * 20x = 24B
    assert total_tokens(20, 1_200_000_000) == 24_000_000_000


def test_total_tokens_float_factor():
    assert total_tokens(2.5, 1_000_000) == 2_500_000


@pytest.mark.parametrize("bad", [0, -1, -0.5])
def test_total_tokens_rejects_nonpositive(bad):
    with pytest.raises(ValueError):
        total_tokens(bad, 10)


def test_embedding_tied():
    pc = embedding_params(vocab_size=128_000, hidden_size=1536, tie_embeddings=True)
    assert pc.embedding == 128_000 * 1536
    assert pc.lm_head == 0


def test_embedding_untied():
    pc = embedding_params(vocab_size=128_000, hidden_size=2304, tie_embeddings=False)
    assert pc.lm_head == pc.embedding == 128_000 * 2304


def test_steps_from_tokens_at_frozen_batch():
    # 24B tokens, seq 4096, 1024-seq global batch -> ceil(samples / gbs).
    # Equivalent to the old 4M-token batch (4096 * 1024 = 4_194_304).
    tokens = 24_000_000_000
    seq_length = 4096
    gbs = 1024
    samples = tokens // seq_length
    expected = -(-samples // gbs)  # ceil
    assert steps_from_tokens(tokens, seq_length=seq_length, global_batch_size=gbs) == expected


def test_steps_rejects_zero_batch():
    with pytest.raises(ValueError):
        steps_from_tokens(1, seq_length=256, global_batch_size=0)


def test_steps_rejects_zero_seq_length():
    with pytest.raises(ValueError):
        steps_from_tokens(1, seq_length=0, global_batch_size=512)


def test_parse_token_count_int_passthrough():
    assert parse_token_count(500_000_000) == 500_000_000


def test_parse_token_count_suffixes():
    assert parse_token_count("500M") == 500_000_000
    assert parse_token_count("1B") == 1_000_000_000
    assert parse_token_count("10B") == 10_000_000_000
    assert parse_token_count("50b") == 50_000_000_000
    assert parse_token_count("100B") == 100_000_000_000
    assert parse_token_count("2.5B") == 2_500_000_000
    assert parse_token_count("10K") == 10_000
    assert parse_token_count("1T") == 1_000_000_000_000


def test_parse_token_count_plain_strings():
    assert parse_token_count("1_000_000") == 1_000_000
    assert parse_token_count("5e9") == 5_000_000_000


@pytest.mark.parametrize("bad", ["", "abc", "10Q", "-1B", "B", 0, -5, 0.0])
def test_parse_token_count_rejects_bad(bad):
    with pytest.raises(ValueError):
        parse_token_count(bad)
