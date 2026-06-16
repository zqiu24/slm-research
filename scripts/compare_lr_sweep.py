#!/usr/bin/env python3
"""Compare the nGPT vs adam learning-rate sweeps.

Scrapes the final validation loss from each sweep run's codexlog and prints a
side-by-side `lr -> val/loss` table plus the best-of-sweep winner. Sources the
logs written by scripts/sweep_ngpt_lr.sh and scripts/sweep_adam_lr.sh
(`$CODEX_LOG_DIR/<prefix>_lr<tag>.log`, default /lustre/home/zqiu/log).

Runs still in progress are shown with their latest val and iteration so you can
compare partial curves; '-' means no validation logged yet.

Usage:
    python scripts/compare_lr_sweep.py
    python scripts/compare_lr_sweep.py --logdir /lustre/home/zqiu/log
    python scripts/compare_lr_sweep.py --total-iters 9155   # to flag partial runs
"""

from __future__ import annotations

import argparse
import glob
import os
import re

# "validation loss at iteration 1000 | lm loss value: 3.993223E+00 | lm loss PPL: ..."
_VAL_RE = re.compile(r"validation loss at iteration\s+(\d+)\s*\|\s*lm loss value:\s*([0-9.eE+-]+)")
# filename tag: ngpt_lr30.log -> 30 ; lr = tag / 1e4 (30 -> 0.003)
_NAME_RE = re.compile(r"^(?P<prefix>[a-z0-9_]+?)_lr(?P<tag>\d+)\.log$")


def _last_val(path: str) -> tuple[int, float] | None:
    """Return (iteration, lm_loss) of the LAST validation in the log, or None."""
    last = None
    with open(path, errors="ignore") as fh:
        for line in fh:
            m = _VAL_RE.search(line)
            if m:
                last = (int(m.group(1)), float(m.group(2)))
    return last


def _collect(logdir: str, prefix: str) -> dict[int, tuple[int, int, float | None]]:
    """tag -> (lr_milli, last_iter, val_or_None) for one sweep prefix."""
    out: dict[int, tuple[int, int, float | None]] = {}
    for path in glob.glob(os.path.join(logdir, f"{prefix}_lr*.log")):
        m = _NAME_RE.match(os.path.basename(path))
        if not m or m.group("prefix") != prefix:
            continue
        tag = int(m.group("tag"))
        v = _last_val(path)
        out[tag] = (tag, v[0] if v else -1, v[1] if v else None)
    return out


def _fmt(val: float | None, it: int, total: int | None) -> str:
    if val is None:
        return "       -"
    partial = " (run)" if (total and it < total) else ""
    return f"{val:8.4f}{partial}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default=os.environ.get("CODEX_LOG_DIR", "/lustre/home/zqiu/log"))
    ap.add_argument("--ngpt-prefix", default="ngpt")
    ap.add_argument("--adam-prefix", default="adam")
    ap.add_argument("--total-iters", type=int, default=None, help="flag runs below this as partial")
    args = ap.parse_args()

    ngpt = _collect(args.logdir, args.ngpt_prefix)
    adam = _collect(args.logdir, args.adam_prefix)
    tags = sorted(set(ngpt) | set(adam))
    if not tags:
        print(f"No {args.ngpt_prefix}_lr*.log / {args.adam_prefix}_lr*.log in {args.logdir}")
        return

    print(f"\nLR sweep: nGPT vs adam  (final val/loss; lower is better)  [{args.logdir}]\n")
    print(f"  {'lr':>8}  {'adam':>14}  {'ngpt':>14}   {'Δ(ngpt-adam)':>13}")
    print(f"  {'-' * 8}  {'-' * 14}  {'-' * 14}   {'-' * 13}")
    best = {"adam": (None, None), "ngpt": (None, None)}  # (val, lr)
    for tag in tags:
        lr = tag / 1e4
        av = adam.get(tag, (tag, -1, None))[2]
        nv = ngpt.get(tag, (tag, -1, None))[2]
        ai = adam.get(tag, (tag, -1, None))[1]
        ni = ngpt.get(tag, (tag, -1, None))[1]
        delta = f"{nv - av:+.4f}" if (av is not None and nv is not None) else "      -"
        print(
            f"  {lr:8.4f}  {_fmt(av, ai, args.total_iters):>14}  "
            f"{_fmt(nv, ni, args.total_iters):>14}   {delta:>13}"
        )
        for key, val in (("adam", av), ("ngpt", nv)):
            if val is not None and (best[key][0] is None or val < best[key][0]):
                best[key] = (val, lr)

    print()
    bv_a, blr_a = best["adam"]
    bv_n, blr_n = best["ngpt"]
    if bv_a is not None:
        print(f"  best adam:  {bv_a:.4f}  @ lr {blr_a}")
    if bv_n is not None:
        print(f"  best ngpt:  {bv_n:.4f}  @ lr {blr_n}")
    if bv_a is not None and bv_n is not None:
        win = "nGPT" if bv_n < bv_a else "adam"
        print(f"  ==> {win} wins by {abs(bv_n - bv_a):.4f} val/loss")
    print()


if __name__ == "__main__":
    main()
