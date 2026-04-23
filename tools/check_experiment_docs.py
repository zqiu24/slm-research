"""Pre-commit / CI hook: every experiment YAML has a corresponding doc.

Enforces SPEC.md §8.6 — every ``configs/experiments/<family>/<name>.yaml``
has a ``docs/experiments/<name>.md`` and a non-empty ``experiment.description``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_ROOT = REPO_ROOT / "configs" / "experiments"
DOC_ROOT = REPO_ROOT / "docs" / "experiments"

# These YAMLs are metadata, not experiments, so they're allowed to skip the doc.
_ALLOWLIST = {"_template.yaml", "champion.yaml"}


def _iter_experiment_yamls():
    for path in sorted(EXP_ROOT.rglob("*.yaml")):
        if path.name in _ALLOWLIST:
            continue
        yield path


def check_one(yaml_path: Path) -> list[str]:
    problems: list[str] = []
    data = yaml.safe_load(yaml_path.read_text()) or {}
    exp = data.get("experiment", {}) or {}
    name = exp.get("name")
    desc = (exp.get("description") or "").strip()

    if not name or name == "TEMPLATE_REPLACE_ME":
        problems.append(f"{yaml_path}: experiment.name is missing or still TEMPLATE_REPLACE_ME")
    if not desc or "TEMPLATE_REPLACE_ME" in desc:
        problems.append(
            f"{yaml_path}: experiment.description is missing or still contains TEMPLATE_REPLACE_ME"
        )
    if name:
        doc = DOC_ROOT / f"{name}.md"
        if not doc.exists():
            problems.append(f"{yaml_path}: no matching doc at {doc.relative_to(REPO_ROOT)}")
    return problems


def main() -> int:
    problems: list[str] = []
    for path in _iter_experiment_yamls():
        problems.extend(check_one(path))
    if problems:
        print("Experiment doc check failed:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
