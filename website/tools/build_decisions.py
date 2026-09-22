"""Rebuild website/data/decisions.json, the only data file the landing page needs.

build_landing.py injects Tables 1 and 2 from it, so the page's numbers can be regenerated from the
experiment outputs without hand-editing HTML. CPU only; reads
outputs/agentic_rl_real/protocol_*/{summary.json,curve.csv,time_to_target.csv,*_blocks.csv}.

This is what remains of build_site_data.py after the interactive demo was removed: the splat
quantiser, the cloud thumbnails and the per-scene 3-D manifest went with the viewer.

Usage:  python website/tools/build_decisions.py
"""
from __future__ import annotations
import csv, json, math, time
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Roman\3DGS_PROPOSAL")
RL = ROOT / "outputs" / "agentic_rl_real"
DATA = ROOT / "website" / "data"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


POLICIES = {
    "3dgs": {"label": "3DGS controller", "checkpoint": "outputs/agentic_rl_real/final_accel_3dgs/checkpoints/selected_accel.pth",
             "note": "PPO, acceleration reward, trained on the real-scene pool, acceleration-aware checkpoint selection."},
    "fastergs": {"label": "Faster-GS controller", "checkpoint": "outputs/agentic_rl_real/real_fastergs_accel_aug_base/checkpoints/best.pth",
                 "note": "PPO, acceleration reward, photometric-augmented real-scene pool, same 45-dim state."},
    "dash": {"label": "DashGaussian controller", "checkpoint": "outputs/agentic_rl_real/final_accel_dash_fixed_horizon/checkpoints/policy_update_0109.pth",
             "note": "PPO, acceleration reward, trained on the real-scene pool with Dash's coarse-to-fine horizon pinned to the 30k evaluation cap."},
}
BACKEND_LABEL = {"3dgs": "3DGS", "fastergs": "Faster-GS", "dash": "DashGaussian", "legs": "LeGS"}


# protocol runs for the decisions explorer: (scene, target backend, policy backend, run dir, kind)
# Every Dash row is the fixed-horizon policy (run final_accel_dash_fixed_horizon, checkpoint
# policy_update_0109, dash.schedule_horizon_iterations=30000) -- the same runs the paper's
# Table 1 and cross-backend table use. The earlier protocol_dash_*_30k / protocol_cross_dashPolicy_*
# runs came from the policy whose curriculum was keyed to the episode cap and never densified.
PROTOCOL = [
    ("train", "3dgs", "3dgs", "protocol_nonaug_3dgs_train_fr", "native"),
    ("train", "fastergs", "fastergs", "protocol_aug_base_fastergs_train_rerun", "native"),
    ("train", "dash", "dash", "protocol_dashfixed_train", "native"),
    ("stump", "3dgs", "3dgs", "protocol_3dgs_stump_30k", "native"),
    ("stump", "fastergs", "fastergs", "protocol_aug_base_fastergs_stump", "native"),
    ("stump", "dash", "dash", "protocol_dashfixed_stump", "native"),
    ("bicycle", "3dgs", "3dgs", "protocol_3dgs_bicycle_30k", "native"),
    ("bicycle", "fastergs", "fastergs", "protocol_aug_base_fastergs_bicycle", "native"),
    ("bicycle", "dash", "dash", "protocol_dashfixed_bicycle", "native"),
    ("barn", "3dgs", "3dgs", "protocol_3dgs_barn_30k", "native"),
    ("barn", "fastergs", "fastergs", "protocol_aug_base_fastergs_barn", "native"),
    ("barn", "dash", "dash", "protocol_dashfixed_barn", "native"),
    ("caterpillar", "3dgs", "3dgs", "protocol_3dgs_caterpillar_30k", "native"),
    ("caterpillar", "fastergs", "fastergs", "protocol_aug_base_fastergs_caterpillar", "native"),
    ("caterpillar", "dash", "dash", "protocol_dashfixed_caterpillar", "native"),
    ("ignatius", "3dgs", "3dgs", "protocol_3dgs_ignatius_30k", "native"),
    ("ignatius", "fastergs", "fastergs", "protocol_aug_base_fastergs_ignatius", "native"),
    ("ignatius", "dash", "dash", "protocol_dashfixed_ignatius", "native"),
    ("train", "dash", "3dgs", "protocol_cross_3dgsPolicy_on_dash", "transfer"),
    ("train", "fastergs", "3dgs", "protocol_cross_3dgsPolicy_on_fastergs", "transfer"),
    ("train", "3dgs", "dash", "protocol_dashfixed_on_3dgs_train", "transfer"),
    ("train", "3dgs", "fastergs", "protocol_cross_fgsPolicy_on_3dgs", "transfer"),
    ("train", "dash", "fastergs", "protocol_cross_fgsPolicy_on_dash", "transfer"),
    ("train", "legs", "fastergs", "protocol_cross_fgsPolicy_on_legs", "transfer"),
    ("train", "legs", "3dgs", "protocol_cross_3dgsPolicy_on_legs", "transfer"),
    ("train", "legs", "dash", "protocol_dashfixed_on_legs_train", "transfer"),
    ("train", "fastergs", "dash", "protocol_dashfixed_on_fgs_train", "transfer"),
]
SCENE_META = {
    "train": ("Tanks & Temples", "held-out"), "bicycle": ("Mip-NeRF 360", "zero-shot"), "stump": ("Mip-NeRF 360", "zero-shot"),
    "barn": ("Tanks & Temples", "held-out"), "caterpillar": ("Tanks & Temples", "held-out"),
    "ignatius": ("Tanks & Temples", "held-out"),
}

ACTION_FIELDS = ["block_steps", "densify_mode", "densification_interval", "prune_mode", "opacity_reset", "stop",
                 "densify_threshold_mult", "prune_opacity_threshold", "position_lr_mult", "feature_lr_mult",
                 "opacity_lr_mult", "scaling_lr_mult", "rotation_lr_mult"]


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(x):
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except ValueError:
        return x
    if math.isnan(v):
        return None
    if v.is_integer() and "." not in str(x):
        return int(v)
    return round(v, 4)


def compact_blocks(rows: list[dict], keep_actions: bool) -> list[dict]:
    out = []
    for r in rows:
        d = {k: num(r.get(k)) for k in ["block", "iter", "t", "gaussians", "added", "pruned", "block_seconds",
                                        "val_psnr", "val_ssim", "reward", "stop_intent"] if k in r}
        if keep_actions:
            for k in ACTION_FIELDS:
                if k in r:
                    d[k] = num(r[k])
        out.append(d)
    return out


# Per-scene PSNR ladder for the headline (native) rows, identical to the paper's Table 1
# (agentic_gs_phase1/scripts/make_heldout_tables.py). The ladder matters: each run's own
# time_to_target.csv was written with whatever targets that evaluation used, and on stump only
# 21 dB is inside the 16-21 range both methods cross, so a gmean taken from the CSV rests on a
# single target. Recomputing from curve.csv with the paper's ladder keeps the site and the paper
# reporting the same number. Transfer runs keep the CSV (the paper's cross-backend table does too).
LADDER = {"train": range(16, 22), "barn": range(16, 27), "caterpillar": range(16, 27),
          "ignatius": range(16, 27), "bicycle": range(16, 25), "stump": range(19, 25)}


def time_to(curve: list[tuple[float, float]], target: float):
    """First crossing of `target`, linearly interpolated -- same rule as scripts/eval_protocol.py."""
    for (t0, p0), (t1, p1) in zip(curve, curve[1:]):
        if p0 <= target <= p1 and p1 > p0:
            return t0 + (t1 - t0) * (target - p0) / (p1 - p0)
    return None


def ladder_ttt(curve: list[dict], ladder) -> tuple[list[dict], float | None]:
    tracks = {m: sorted((r["t"], r["test_psnr"]) for r in curve if r["method"] == m)
              for m in ("baseline", "agentic")}
    rows = []
    for tg in ladder:
        tb, ta = time_to(tracks["baseline"], tg), time_to(tracks["agentic"], tg)
        rows.append({"target_psnr": float(tg), "baseline_s": tb and round(tb, 2),
                     "agentic_s": ta and round(ta, 2), "speedup": round(tb / ta, 3) if tb and ta else None})
    sp = [r["speedup"] for r in rows if r["speedup"]]
    return rows, (round(float(np.exp(np.mean(np.log(sp)))), 3) if sp else None)


def build_decisions() -> dict:
    runs = []
    for scene, be, pol, run, kind in PROTOCOL:
        rd = RL / run
        if not (rd / "summary.json").exists():
            log(f"[decisions] skip {run} (no summary)")
            continue
        summ = json.loads((rd / "summary.json").read_text())
        agent = compact_blocks(read_csv(rd / "agentic_blocks.csv"), keep_actions=True)
        base = compact_blocks(read_csv(rd / "baseline_blocks.csv"), keep_actions=False)
        curve = [{k: num(v) for k, v in r.items()} for r in read_csv(rd / "curve.csv")]
        ttt = [{k: num(v) for k, v in r.items()} for r in read_csv(rd / "time_to_target.csv")]
        sp = [r["speedup"] for r in ttt if isinstance(r.get("speedup"), (int, float)) and r["speedup"]]
        gmean = round(float(np.exp(np.mean(np.log(sp)))), 3) if sp else None
        if kind == "native" and scene in LADDER:
            ttt, gmean = ladder_ttt(curve, LADDER[scene])
        ds, role = SCENE_META.get(scene, ("", ""))
        runs.append({
            "id": run, "scene": scene, "dataset": ds, "scene_role": role, "backend": be, "backend_label": BACKEND_LABEL[be],
            "policy": pol, "policy_label": POLICIES[pol]["label"], "kind": kind, "max_iter": summ.get("max_iter"),
            "checkpoint": summ.get("checkpoint"),
            "final": {m: summ["methods"][m]["final"] for m in summ["methods"]},
            "natural_stop_iter": summ["methods"].get("agentic", {}).get("natural_stop_iter"),
            "time_to_target": ttt, "gmean_speedup": gmean,
            "agent_blocks": agent, "baseline_blocks": base, "curve": curve,
        })
    return {"runs": runs, "action_fields": ACTION_FIELDS,
            "discrete": {"block_steps": [50, 100, 200], "densify_mode": ["off", "conservative", "default", "aggressive"],
                         "densification_interval": [50, 100, 200, 400], "prune_mode": ["off", "opacity_only", "opacity_and_size"],
                         "opacity_reset": ["no_reset", "reset_if_plateau", "force_reset"], "stop": ["continue", "stop"]},
            "continuous": {"densify_threshold_mult": [0.25, 4.0, "log", 1.0], "prune_opacity_threshold": [0.001, 0.02, "linear", 0.005],
                           "position_lr_mult": [0.25, 4.0, "log", 1.0], "feature_lr_mult": [0.25, 2.0, "log", 1.0],
                           "opacity_lr_mult": [0.25, 2.0, "log", 1.0], "scaling_lr_mult": [0.25, 2.0, "log", 1.0],
                           "rotation_lr_mult": [0.25, 2.0, "log", 1.0]},
            "baseline_defaults": {"block_steps": 100, "densify_mode": "default", "densification_interval": 100,
                                  "prune_mode": "opacity_and_size", "opacity_reset": "reset_if_plateau", "stop": "continue",
                                  "densify_threshold_mult": 1.0, "prune_opacity_threshold": 0.005, "position_lr_mult": 1.0,
                                  "feature_lr_mult": 1.0, "opacity_lr_mult": 1.0, "scaling_lr_mult": 1.0, "rotation_lr_mult": 1.0}}


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "decisions.json"
    out.write_text(json.dumps(build_decisions(), separators=(",", ":")))
    log(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
