"""Print total/active non-embedding params for a family+scale pair.

Usage (from repo root):
  python tools/size_check.py base/family=deepseek_v3 base/scale=600m_deepseek_v3
"""

from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.arch_params import (  # noqa: E402
    active_non_embedding_params,
    non_embedding_params,
)


def main() -> None:
    kv = dict(item.split("=", 1) for item in sys.argv[1:])
    family = kv["base/family"]
    scale = kv["base/scale"]
    fam = OmegaConf.load(REPO_ROOT / "configs/base/family" / f"{family}.yaml")
    sc = OmegaConf.load(REPO_ROOT / "configs/base/scale" / f"{scale}.yaml")
    merged = OmegaConf.merge(fam, sc)
    model = OmegaConf.to_container(merged.base.model, resolve=True)
    budget = int(merged.base.non_embedding_params)
    total = non_embedding_params(model)
    active = active_non_embedding_params(model)
    print(f"family={family} scale={scale}")
    print(f"budget {budget:>15,}")
    print(f"total  {total:>15,}  ({(total - budget) / budget:+.2%} vs budget)")
    print(f"active {active:>15,}  ({active / total:.1%} of total)")


if __name__ == "__main__":
    main()
