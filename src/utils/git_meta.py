"""Git metadata: repo SHA, submodule SHA, dirty-detection.

Every run records ``(git_sha, megatron_sha)``; in shared W&B projects, a
dirty working tree aborts the launch (see SPEC.md §5.2, §9.3).
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitDirty(RuntimeError):  # noqa: N818
    """Raised when the working tree has uncommitted changes in a non-scratch run."""


def _run(args: list[str], cwd: Path | str | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def git_sha(cwd: Path | str | None = None, *, allow_dirty: bool = False) -> str:
    """Return the current HEAD SHA, with ``+dirty`` suffix if appropriate.

    When ``allow_dirty=False`` (the default — shared W&B projects), a dirty
    tree raises ``GitDirty``.
    """
    sha = _run(["git", "rev-parse", "HEAD"], cwd=cwd)
    dirty = _run(["git", "status", "--porcelain"], cwd=cwd)
    if dirty:
        if not allow_dirty:
            raise GitDirty(
                "Working tree is dirty. Commit or stash before submitting to a shared project, "
                "or submit to wandb.project=slm-<you> which permits dirty trees."
            )
        return f"{sha}+dirty"
    return sha


def submodule_sha(submodule_path: str, cwd: Path | str | None = None) -> str:
    """Return the SHA currently checked out in a submodule."""
    return _run(["git", "rev-parse", "HEAD"], cwd=Path(cwd or ".") / submodule_path)
