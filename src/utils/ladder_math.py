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


def steps_from_tokens(
    total_tokens_: int,
    *,
    global_batch_size_tokens: int,
) -> int:
    """Training steps implied by a token budget and a fixed global batch size.

    Global batch size is frozen across the ladder (SPEC.md §1.3).
    """
    if global_batch_size_tokens <= 0:
        raise ValueError("global_batch_size_tokens must be positive")
    return (total_tokens_ + global_batch_size_tokens - 1) // global_batch_size_tokens
