"""Block until the GPU is quiet enough to run a wall-clock-timed experiment.

scripts/wait_gpu_idle.ps1 requires N *consecutive* instantaneous samples below a threshold. That
works on a quiet machine but not on a desktop in use: measured here, utilisation is p50 8%, p90 13%,
but spikes to ~42% a couple of percent of the time (editor, browser, a running slide deck). A single
spike resets the counter, so the consecutive test can fail indefinitely and then "proceed anyway"
at its deadline -- unguarded, which is the outcome it exists to prevent.

This gate looks at a trailing window instead:

  pass when  median(window) <= --median  and  max(window) <= --max

The median answers "is the GPU broadly free", so brief desktop spikes are tolerated. The max still
rejects anything sustained: a 3DGS episode holds the card at 70-100%, far above both bars. On
timeout this exits non-zero rather than proceeding, so a caller never silently produces timings
taken against a co-tenant.

Exit codes:  0 quiet, 1 still busy at the deadline, 2 nvidia-smi unavailable.
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time


def sample() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    line = (out.stdout or "").strip().splitlines()
    try:
        return int(line[0].strip())
    except (IndexError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--median", type=float, default=20.0, help="max median utilisation (%%) over the window")
    ap.add_argument("--max", type=float, default=60.0, help="max single sample (%%) allowed in the window")
    ap.add_argument("--window", type=int, default=12, help="samples in the trailing window")
    ap.add_argument("--interval", type=float, default=5.0, help="seconds between samples")
    ap.add_argument("--max-wait-min", type=float, default=120.0)
    args = ap.parse_args()

    deadline = time.time() + 60.0 * args.max_wait_min
    window: list[int] = []
    last_report = 0.0
    while True:
        v = sample()
        if v is None:
            print("wait_gpu_quiet: nvidia-smi unavailable", flush=True)
            return 2
        window.append(v)
        if len(window) > args.window:
            window.pop(0)
        if len(window) == args.window:
            med, hi = statistics.median(window), max(window)
            if med <= args.median and hi <= args.max:
                print(f"GPU quiet (median {med:.0f}%, max {hi}% over {args.window} samples) - proceeding", flush=True)
                return 0
            if time.time() - last_report > 60:
                print(f"GPU busy (median {med:.0f}%, max {hi}%); waiting...", flush=True)
                last_report = time.time()
        if time.time() > deadline:
            med = statistics.median(window) if window else -1
            print(f"GPU still busy after {args.max_wait_min:.0f} min (median {med:.0f}%) - NOT proceeding", flush=True)
            return 1
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
