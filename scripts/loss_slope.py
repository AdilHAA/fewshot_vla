"""Is a finished run still learning, or sitting on the lr floor? Gate G0.

The SmolVLA lr preset floors at 2.5e-6 after scheduler_decay_steps (30k by default)
and does NOT scale with --steps, so a flat loss late in a run can be an lr-floor
artifact rather than convergence. This reads the TensorBoard scalars of a finished
run and compares the loss slope before/after the decay horizon.

  python scripts/loss_slope.py outputs/dino_cls
  python scripts/loss_slope.py outputs/dino_cls --decay 30000
  python scripts/loss_slope.py outputs/*          # several runs at once

Reads <run>/tensorboard (written when train.sh runs with TB=1).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys


def read_scalars(tb_dir: str, tag: str):
    """(steps, values) for one scalar tag, or None if the tag/event files are absent."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(tb_dir)
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return None
    ev = acc.Scalars(tag)
    return [e.step for e in ev], [e.value for e in ev]


def slope(steps, values, lo: float, hi: float):
    """Least-squares slope of values against steps, within [lo, hi). None if empty."""
    pts = [(s, v) for s, v in zip(steps, values) if lo <= s < hi]
    if len(pts) < 2:
        return None, 0
    n = len(pts)
    sx = sum(s for s, _ in pts)
    sy = sum(v for _, v in pts)
    sxx = sum(s * s for s, _ in pts)
    sxy = sum(s * v for s, v in pts)
    denom = n * sxx - sx * sx
    if denom == 0:
        return None, n
    return (n * sxy - sx * sy) / denom, n


def analyse(run_dir: str, decay: int, windows=((10_000, 30_000), (30_000, 1e9))):
    tb = os.path.join(run_dir, "tensorboard")
    if not os.path.isdir(tb):
        return f"{run_dir}: no tensorboard/ (arm trained without TB=1?)", False
    loss = read_scalars(tb, "train/loss")
    if loss is None:
        return f"{run_dir}: no train/loss scalar", False
    steps, values = loss
    lr = read_scalars(tb, "train/lr")
    lr_last = f"{lr[1][-1]:.2e}" if lr else "?"

    parts = [f"{os.path.basename(run_dir)}: "
             f"loss {steps[0]}..{steps[-1]} ({len(steps)} pts), last={values[-1]:.4f}, lr@last={lr_last}"]

    def wnd(hi):
        return "end" if hi >= 1e9 else f"{hi/1e3:.0f}k"

    prev = None
    verdict = "?"
    for lo, hi in windows:
        m, n = slope(steps, values, lo, min(hi, steps[-1] + 1))
        if m is None:
            parts.append(f"  [{lo/1e3:.0f}k,{wnd(hi)}): n={n} (too few points)")
            continue
        parts.append(f"  [{lo/1e3:.0f}k,{wnd(hi)}): slope={m*1e6:+.2f}e-6/step (n={n})")
        if prev is not None and m is not None:
            # slope collapsed by >=10x while lr sits at the floor -> plateau is the
            # lr artifact, not convergence; extending steps is only useful WITH a
            # longer decay horizon.
            if prev < 0 and m > prev / 10:
                verdict = "plateau (slope collapsed >=10x) — check lr floor; extend WITH SCHED_DECAY"
            elif m < prev / 2:
                verdict = "still improving"
        prev = m
    if verdict == "?":
        verdict = "no clear change between windows"
    parts.append(f"  => {verdict}")
    return "\n".join(parts), True


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="run dirs (contain tensorboard/)")
    p.add_argument("--decay", type=float, default=30_000,
                   help="scheduler decay horizon (default 30000, the SmolVLA preset)")
    args = p.parse_args(argv)

    dirs = []
    for r in args.runs:
        dirs.extend(sorted(glob.glob(r)) or [r])
    ok = False
    for d in dirs:
        text, found = analyse(d, args.decay)
        print(text, file=sys.stdout if found else sys.stderr)
        ok = ok or found
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
