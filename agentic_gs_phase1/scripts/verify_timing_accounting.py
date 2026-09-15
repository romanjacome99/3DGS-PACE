"""Independent reconstruction of the deployment-inclusive time-to-target accounting.

`make_timing_tables.py` builds the paper's timing tables; this script rebuilds the same quantities by a
different route, from the raw trainer-side block logs, as a cross-check and to run the sensitivity sweep
quoted in the appendix. CPU only, no GPU time: every quantity it needs was logged during the original
30k rollouts.

For every scene x backend pair, the per-block clocks are

    t_train(k)  = sum_{j<=k} block_seconds_j                                   (the historical clock)
    t_deploy(k) = sum_{j<=k} [ block_seconds_j + validation_seconds_j + c ]

where validation_seconds_j is the feedback render of block j as logged during that rollout (so the growth
of the render cost with model size is exact) and c is the per-block cost of the population statistics,
observation assembly, opacity reset and actor forward from the measured breakdown in
outputs/timing_audit/<backend>_<method>/controller_overhead.json (model-size-insensitive; ~12 ms).

Time-to-target is re-interpolated on the run's own test-PSNR curve under three accountings:
  A train-loop : both methods on t_train        (the clock of Table 1)
  B symmetric  : both methods on t_deploy
  C deployment : PACE on t_deploy, fixed schedule on t_train -- a deployed fixed schedule renders no
                 feedback views and runs no policy, so this is the least favourable accounting for PACE.

Usage:
  python agentic_gs_phase1/scripts/verify_timing_accounting.py
  python agentic_gs_phase1/scripts/verify_timing_accounting.py --const-scale 5   # sensitivity to c
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RL = ROOT / "outputs" / "agentic_rl_real"
AUDIT = ROOT / "outputs" / "timing_audit"

RUNS = {
    ("train", "3dgs"): "protocol_nonaug_3dgs_train_fr",
    ("train", "fgs"): "protocol_aug_base_fastergs_train_rerun",
    ("train", "dash"): "protocol_nonaug_dash_train",
    ("barn", "3dgs"): "protocol_3dgs_barn_30k",
    ("barn", "fgs"): "protocol_aug_base_fastergs_barn",
    ("barn", "dash"): "protocol_dash_barn_30k",
    ("caterpillar", "3dgs"): "protocol_3dgs_caterpillar_30k",
    ("caterpillar", "fgs"): "protocol_aug_base_fastergs_caterpillar",
    ("caterpillar", "dash"): "protocol_dash_caterpillar_30k",
    ("ignatius", "3dgs"): "protocol_3dgs_ignatius_30k",
    ("ignatius", "fgs"): "protocol_aug_base_fastergs_ignatius",
    ("ignatius", "dash"): "protocol_dash_ignatius_30k",
    ("bicycle", "3dgs"): "protocol_3dgs_bicycle_30k",
    ("bicycle", "fgs"): "protocol_aug_base_fastergs_bicycle",
    ("bicycle", "dash"): "protocol_dash_bicycle_30k",
    ("stump", "3dgs"): "protocol_3dgs_stump_30k",
    ("stump", "fgs"): "protocol_aug_base_fastergs_stump",
    ("stump", "dash"): "protocol_dash_stump_30k",
}
PRETTY = {"3dgs": "3DGS", "fgs": "Faster-GS", "dash": "DashGaussian"}
# the PSNR ladder each scene is evaluated on (as in the paper's per-scene tables)
LADDER = {"train": range(16, 22), "barn": range(16, 27), "caterpillar": range(16, 27),
          "ignatius": range(16, 27), "bicycle": range(16, 25), "stump": range(19, 25)}


def num(x, default=0.0):
    try:
        v = float(x)
        return default if math.isnan(v) else v
    except (TypeError, ValueError):
        return default


def const_seconds(backend: str, method: str, scale: float) -> float:
    """Per-block controller cost other than the feedback render, from the measured breakdown."""
    p = AUDIT / f"{backend}_{method}" / "controller_overhead.json"
    if not p.exists():
        return 0.0
    ms = json.loads(p.read_text())["per_block_ms"]
    return scale * sum(ms.get(k, 0.0) for k in ("stats", "obs", "policy", "reset")) / 1000.0


def clocks(run: str, method: str, backend: str, scale: float):
    """[(iteration, t_train, t_deploy)] at the end of every block."""
    logs = list((RL / run / f"_run_{method}").glob("*/episode_0000/agentic_blocks.csv"))
    if not logs:
        return None
    c = const_seconds(backend, method, scale)
    out, tt, td = [], 0.0, 0.0
    with open(logs[0], newline="") as f:
        for r in csv.DictReader(f):
            b = num(r.get("block_seconds"))
            tt += b
            td += b + num(r.get("validation_seconds")) + c
            out.append((int(num(r.get("iteration"))), tt, td))
    return out


def at_iter(cl, it):
    """Both clocks at an iteration, linear inside the block that contains it."""
    if not cl or it <= 0:
        return 0.0, 0.0
    if it >= cl[-1][0]:
        return cl[-1][1], cl[-1][2]
    prev = (0, 0.0, 0.0)
    for cur in cl:
        if cur[0] >= it:
            span = cur[0] - prev[0]
            f = (it - prev[0]) / span if span else 1.0
            return prev[1] + f * (cur[1] - prev[1]), prev[2] + f * (cur[2] - prev[2])
        prev = cur
    return cl[-1][1], cl[-1][2]


def samples(curve, method, cl):
    return sorted((at_iter(cl, int(num(r["iter"]))) + (num(r["test_psnr"]),))
                  for r in curve if r["method"] == method)


def time_to(pts, target, idx):
    for (t0, d0, p0), (t1, d1, p1) in zip(pts, pts[1:]):
        if p0 <= target <= p1 and p1 > p0:
            f = (target - p0) / (p1 - p0)
            lo, hi = ((t0, d0)[idx], (t1, d1)[idx])
            return lo + f * (hi - lo)
    return None


def gmean(values):
    v = [x for x in values if x and x > 0]
    return math.exp(sum(map(math.log, v)) / len(v)) if v else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--const-scale", type=float, default=1.0,
                    help="multiply the non-render per-block constant (sensitivity analysis)")
    ap.add_argument("--out", default=str(AUDIT / "accounting_measured.csv"))
    args = ap.parse_args()

    rows, missing = [], []
    print(f"{'scene':12s} {'backend':12s} {'n':>2s} | {'A train':>8s} {'B symm':>7s} {'C deploy':>8s} | "
          f"{'ovh PACE':>8s} {'ovh fixed':>9s}")
    for (scene, bk), run in RUNS.items():
        ca = clocks(run, "agentic", bk, args.const_scale)
        cb = clocks(run, "baseline", bk, args.const_scale)
        cf = RL / run / "curve.csv"
        if not (ca and cb and cf.exists()):
            missing.append(run)
            continue
        with open(cf, newline="") as f:
            curve = list(csv.DictReader(f))
        pa, pb = samples(curve, "agentic", ca), samples(curve, "baseline", cb)
        A, B, C = [], [], []
        for tgt in LADDER[scene]:
            a_t, a_d = time_to(pa, tgt, 0), time_to(pa, tgt, 1)
            b_t, b_d = time_to(pb, tgt, 0), time_to(pb, tgt, 1)
            if a_t and b_t:
                A.append(b_t / a_t)
            if a_d and b_d:
                B.append(b_d / a_d)
            if a_d and b_t:
                C.append(b_t / a_d)
        ga, gb, gc = gmean(A), gmean(B), gmean(C)
        oa = 100 * (ca[-1][2] - ca[-1][1]) / ca[-1][1]
        ob = 100 * (cb[-1][2] - cb[-1][1]) / cb[-1][1]
        print(f"{scene:12s} {PRETTY[bk]:12s} {len(A):2d} | {ga or float('nan'):8.3f} {gb or float('nan'):7.3f} "
              f"{gc or float('nan'):8.3f} | {oa:7.2f}% {ob:8.2f}%")
        rows.append({"scene": scene, "backend": bk, "pretty": PRETTY[bk], "n_targets": len(A),
                     "blocks_pace": len(ca), "blocks_fixed": len(cb),
                     "overhead_pace_pct": round(oa, 3), "overhead_fixed_pct": round(ob, 3),
                     "gmean_A": round(ga, 4) if ga else "", "gmean_B": round(gb, 4) if gb else "",
                     "gmean_C": round(gc, 4) if gc else ""})

    if missing:
        print("missing runs:", missing)
    if not rows:
        print("nothing to do")
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print()
    for k, label in (("gmean_A", "train-loop"), ("gmean_B", "symmetric"), ("gmean_C", "deployment")):
        print(f"overall {label:11s}: {gmean([r[k] for r in rows if r[k] != '']):.3f}")
    below = [(r["scene"], r["pretty"], r["gmean_C"]) for r in rows if r["gmean_C"] != "" and r["gmean_C"] < 1]
    print(f"below 1x under the deployment clock ({len(below)}):", below)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
