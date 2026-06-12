"""Ladder math: tokens-per-param budgets, embedding accounting.

Parameter-counting convention (SPEC.md §1.4): the ladder is defined in
**non-embedding parameters**. Token budgets use non-embedding params; total
params are derived (for logging) from the frozen tokenizer's vocab size.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParamCounts:
    non_embedding: int
    embedding: int
    lm_head: int

    @property
    def total(self) -> int:
        return self.non_embedding + self.embedding + self.lm_head


def embedding_params(vocab_size: int, hidden_size: int, *, tie_embeddings: bool) -> ParamCounts:
    """Compute embedding + LM-head param counts from vocab, hidden, and tying.

    ``non_embedding`` is left at 0 here; the caller fills it from the scale
    config. Embedding and LM-head are shape-fixed once the tokenizer and
    scale are fixed (see SPEC.md §5.1.1).
    """
    emb = vocab_size * hidden_size
    head = 0 if tie_embeddings else emb
    return ParamCounts(non_embedding=0, embedding=emb, lm_head=head)


def total_tokens(tokens_per_param: float | int, non_embedding_params: int) -> int:
    """training.total_tokens = tokens_per_param * base.non_embedding_params."""
    if tokens_per_param <= 0:
        raise ValueError(f"tokens_per_param must be positive, got {tokens_per_param!r}")
    if non_embedding_params <= 0:
        raise ValueError(f"non_embedding_params must be positive, got {non_embedding_params!r}")
    return int(round(float(tokens_per_param) * non_embedding_params))


_TOKEN_SUFFIXES = {"K": 10**3, "M": 10**6, "B": 10**9, "T": 10**12}


def parse_token_count(value: int | float | str) -> int:
    """Parse an explicit token budget into an int token count.

    Accepts plain numbers (``500_000_000``, ``5e9``) and human-friendly
    suffixed strings (``"500M"``, ``"1B"``, ``"2.5b"``; K/M/B/T,
    case-insensitive). Underscores and commas in strings are ignored.
    Rejects non-positive results.
    """
    if isinstance(value, bool):
        raise ValueError(f"token count must be a number or suffixed string, got {value!r}")
    if isinstance(value, int | float):
        tokens = int(round(value))
    else:
        text = str(value).strip().replace("_", "").replace(",", "")
        if not text:
            raise ValueError(f"token count is empty: {value!r}")
        multiplier = _TOKEN_SUFFIXES.get(text[-1].upper())
        if multiplier is not None:
            text = text[:-1]
        else:
            multiplier = 1
        try:
            tokens = int(round(float(text) * multiplier))
        except ValueError as exc:
            raise ValueError(f"cannot parse token count {value!r}") from exc
    if tokens <= 0:
        raise ValueError(f"token count must be positive, got {value!r}")
    return tokens


def steps_from_tokens(
    total_tokens_: int,
    *,
    seq_length: int,
    global_batch_size: int,
) -> int:
    """Training steps implied by a token budget at a fixed global batch size.

    The token budget is converted to samples via ``seq_length`` (matching
    ``--train-samples = total_tokens // seq_length``), then divided by the
    per-step ``global_batch_size`` (in sequences), which is frozen across the
    ladder (SPEC.md §1.3).
    """
    if seq_length <= 0:
        raise ValueError("seq_length must be positive")
    if global_batch_size <= 0:
        raise ValueError("global_batch_size must be positive")
    samples = total_tokens_ // seq_length
    return (samples + global_batch_size - 1) // global_batch_size
