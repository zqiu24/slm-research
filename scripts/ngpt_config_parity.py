"""Diff two resolved_config.yaml archives (nGPT vs matched baseline).

Usage:
    python scripts/ngpt_config_parity.py <ngpt_run_dir> <baseline_run_dir>

Prints every differing leaf key. Keys whose dotted path contains any of the
EXPECTED_DELTA substrings are intended method differences; anything else is
flagged UNEXPECTED and exits non-zero so drift fails loudly.
"""

import sys

from omegaconf import OmegaConf

EXPECTED_DELTA = (
    "optim",
    "experiment.name",
    "experiment.kind",
    "experiment.family",
    "experiment.patches",
    "experiment.description",
    "experiment.references",
    "scheduler",  # nGPT zeroes warmup; baseline keeps it
    "_derived",  # run_name / hashes / archive paths differ by construction
    "wandb.name",
    "wandb.run",
)


def flat(cfg):
    out = {}

    def rec(node, prefix):
        if hasattr(node, "items"):
            for k, v in node.items():
                rec(v, f"{prefix}.{k}" if prefix else str(k))
        elif isinstance(node, list | tuple):
            out[prefix] = list(node)
        else:
            out[prefix] = node

    rec(cfg, "")
    return out


def main():
    ngpt_dir, base_dir = sys.argv[1], sys.argv[2]
    a = flat(OmegaConf.load(f"{ngpt_dir}/resolved_config.yaml"))
    b = flat(OmegaConf.load(f"{base_dir}/resolved_config.yaml"))
    unexpected = []
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            tag = "EXPECTED" if any(s in k for s in EXPECTED_DELTA) else "UNEXPECTED"
            print(f"[{tag}] {k}: ngpt={a.get(k)!r}  baseline={b.get(k)!r}")
            if tag == "UNEXPECTED":
                unexpected.append(k)
    if unexpected:
        print(f"\nFAIL: {len(unexpected)} unexpected config delta(s): {unexpected}")
        sys.exit(1)
    print("\nOK: arms differ only by the intended method/recipe deltas.")


if __name__ == "__main__":
    main()
