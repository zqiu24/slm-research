"""Derive TP/PP/DP from ``base.non_embedding_params`` and cluster rules.

Rules are declared per cluster in ``configs/clusters/<cluster>.yaml`` under
``parallelism.tp_size_rules`` — a list of ``{model_params_lt, tp, pp}``
entries matched in order. The first rule whose threshold exceeds the model
size wins.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParallelismSpec:
    tp: int
    pp: int
    dp: int


def resolve(
    non_embedding_params: int,
    *,
    cluster_nodes: int,
    gpus_per_node: int,
    tp_size_rules: list[dict],
) -> ParallelismSpec:
    """Pick TP/PP from the first matching rule; derive DP from the allocation."""
    tp = pp = None
    for rule in tp_size_rules:
        threshold = float(rule["model_params_lt"])
        if non_embedding_params < threshold:
            tp, pp = int(rule["tp"]), int(rule["pp"])
            break
    if tp is None or pp is None:
        # No rule matched -> use the last rule as a ceiling
        rule = tp_size_rules[-1]
        tp, pp = int(rule["tp"]), int(rule["pp"])

    world_size = cluster_nodes * gpus_per_node
    model_parallel = tp * pp
    if world_size % model_parallel != 0:
        raise ValueError(
            f"World size {world_size} not divisible by TP*PP={model_parallel}; "
            f"adjust cluster_nodes or the tp_size_rules."
        )
    dp = world_size // model_parallel
    return ParallelismSpec(tp=tp, pp=pp, dp=dp)
