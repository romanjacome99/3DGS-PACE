"""Aggregate the training-view ablation (scripts/run_ablation_views.sh) into
  Paper/figures/ablation_views.pdf|png      speed-up / PSNR / Gaussians vs number of training views
  Paper/tables/ablation_views.tex           per scene x views table
The full-view reference of each scene is its existing time-to-target protocol run.
CPU only; run with the env_pytorch_3dgs python (matplotlib) or any python with numpy+matplotlib.
"""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(r"C:\Roman\3DGS_PROPOSAL")
RL = ROOT / "outputs" / "agentic_rl_real"
ABL = RL / "ablation_views_3dgs"
FULL = {  # scene -> (existing full-view protocol run, number of training cameras incl. the validation split)
    "train": ("protocol_nonaug_3dgs_train_fr", 263),
    "ignatius": ("protocol_3dgs_ignatius_30k", 230),
    "caterpillar": ("protocol_3dgs_caterpillar_30k", 335),
    "barn": ("protocol_3dgs_barn_30k", 358),
}
VIEWS = [25, 50, 100]
TARGETS = list(range(12, 29))


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def time_to(curve, method, tgt):
    c = sorted((num(r["t"]), num(r["test_psnr"])) for r in curve if r["method"] == method)
    for (t0, p0), (t1, p1) in zip(c, c[1:]):
        if p0 <= tgt <= p1 and p1 > p0:
            return t0 + (t1 - t0) * (tgt - p0) / (p1 - p0)
    return None


def load_run(d: Path, scene: str, views: int) -> dict | None:
    if not (d / "summary.json").exists():
        return None
    summ = json.loads((d / "summary.json").read_text())
    curve = read_csv(d / "curve.csv")
    ttt = []
    for tg in TARGETS:                       # recompute on a common target grid
        tb, ta = time_to(curve, "baseline", tg), time_to(curve, "agentic", tg)
        ttt.append({"target": tg, "baseline_s": tb, "agentic_s": ta, "speedup": (tb / ta) if (tb and ta) else None})
    sp = [r["speedup"] for r in ttt if r["speedup"] and r["target"] >= 16]      # paper ladder (>= 16 dB)
    sp_all = [r["speedup"] for r in ttt if r["speedup"]]
    fa, fb = summ["methods"]["agentic"]["final"], summ["methods"]["baseline"]["final"]
    ab = read_csv(d / "agentic_blocks.csv")
    meta = None
    mp = list(d.glob("_run_agentic/*/episode_0000/agentic_episode_metadata.json"))
    if mp:
        meta = json.loads(mp[0].read_text())
    return {
        "scene": scene, "views": views, "run": d.name,
        "train_cams": meta.get("train_camera_count") if meta else None,
        "val_cams": meta.get("validation_camera_count") if meta else None,
        "gmean_speedup": float(np.exp(np.mean(np.log(sp)))) if sp else None, "n_targets": len(sp),
        "gmean_all_targets": float(np.exp(np.mean(np.log(sp_all)))) if sp_all else None, "n_targets_all": len(sp_all),
        "max_common_target": max([r["target"] for r in ttt if r["speedup"]], default=None),
        "best_psnr": {"agent": max(num(r["test_psnr"]) for r in curve if r["method"] == "agentic"),
                      "baseline": max(num(r["test_psnr"]) for r in curve if r["method"] == "baseline")},
        "final": {"agent": fa, "baseline": fb},
        "time_to_target": ttt,
        "curve": [{"method": r["method"], "iter": int(float(r["iter"])), "t": num(r["t"]), "psnr": num(r["test_psnr"]), "N": int(float(r["N"]))} for r in curve],
        "agent_stats": {
            "densify_on_frac": sum(1 for b in ab if b["densify_mode"] != "off") / len(ab),
            "interval_mean": float(np.average([float(b["densification_interval"]) for b in ab], weights=[float(b["block_steps"]) for b in ab])),
            "no_reset_frac": sum(1 for b in ab if b["opacity_reset"] == "no_reset") / len(ab),
            "position_lr_gmean": float(np.exp(np.mean([math.log(float(b["position_lr_mult"])) for b in ab]))),
            "feature_lr_gmean": float(np.exp(np.mean([math.log(float(b["feature_lr_mult"])) for b in ab]))),
        },
    }


def main():
    scenes = {}
    for scene, (full_run, n_full) in FULL.items():
        rows = []
        r = load_run(RL / full_run, scene, n_full)
        if r: rows.append(r)
        for v in VIEWS:
            r = load_run(ABL / f"{scene}_v{v}", scene, v)
            if r: rows.append(r)
        rows.sort(key=lambda x: x["views"])
        scenes[scene] = rows
    out = {"backend": "3dgs", "policy": "3DGS controller (final_accel_3dgs/selected_accel.pth)", "views": VIEWS,
           "full_views": {s: n for s, (_, n) in FULL.items()}, "targets": TARGETS, "scenes": scenes}
    done = {s: [r["views"] for r in rows] for s, rows in scenes.items()}
    print("runs available:", done)

    # ---------------- paper table + figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing: skipped figure"); return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
    colors = {"train": "#0072B2", "ignatius": "#E69F00", "caterpillar": "#009E73", "barn": "#CC79A7"}
    for scene, rows in scenes.items():
        if not rows: continue
        xs = [r["views"] for r in rows]
        axes[0].plot(xs, [r["gmean_speedup"] or np.nan for r in rows], "o-", color=colors[scene], label=scene)
        axes[1].plot(xs, [r["best_psnr"]["agent"] for r in rows], "o-", color=colors[scene], label=f"{scene} agent")
        axes[1].plot(xs, [r["best_psnr"]["baseline"] for r in rows], "s--", color=colors[scene], alpha=0.6, label=f"{scene} fixed")
        axes[2].plot(xs, [r["final"]["agent"]["N"] / 1e6 for r in rows], "o-", color=colors[scene])
        axes[2].plot(xs, [r["final"]["baseline"]["N"] / 1e6 for r in rows], "s--", color=colors[scene], alpha=0.6)
    axes[0].axhline(1.0, color="#999", lw=0.8, ls=":")
    for ax, yl in zip(axes, ["time-to-target speed-up (gmean)", "best test PSNR (dB)", "Gaussians at end (M)"]):
        ax.set_xscale("log"); ax.set_xticks([25, 50, 100, 250]); ax.set_xticklabels(["25", "50", "100", "all"])
        ax.set_xlabel("training views"); ax.set_ylabel(yl); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].legend(fontsize=6, frameon=False, ncol=2)
    axes[1].set_title("circles: agent · squares: fixed schedule", fontsize=8)
    fig.tight_layout()
    fd = ROOT / "Paper" / "figures"; fd.mkdir(parents=True, exist_ok=True)
    fig.savefig(fd / "ablation_views.pdf"); fig.savefig(fd / "ablation_views.png", dpi=160)
    td = ROOT / "Paper" / "tables"; td.mkdir(parents=True, exist_ok=True)
    lines = [r"\begin{tabular}{llccccc}", r"\toprule",
             r"Scene & Views & Speed-up (gmean) & Targets & PSNR agent / fixed (dB) & \#G agent / fixed & $t_{\to\max}$ agent / fixed (s) \\", r"\midrule"]
    for scene, rows in scenes.items():
        for i, r in enumerate(rows):
            g = f"{r['gmean_speedup']:.2f}$\\times$" if r["gmean_speedup"] else "--"
            if r["gmean_speedup"] and r["gmean_speedup"] >= 1.5: g = r"\textbf{" + g + "}"
            mt = r["max_common_target"]
            ta = next((x["agentic_s"] for x in r["time_to_target"] if x["target"] == mt), None) if mt else None
            tb = next((x["baseline_s"] for x in r["time_to_target"] if x["target"] == mt), None) if mt else None
            lines.append(f"{scene if i == 0 else ''} & {'all (' + str(r['views']) + ')' if r['views'] > 100 else r['views']} & {g} & {r['n_targets']} & "
                         f"{r['best_psnr']['agent']:.2f} / {r['best_psnr']['baseline']:.2f} & {r['final']['agent']['N'] / 1e6:.2f}M / {r['final']['baseline']['N'] / 1e6:.2f}M & "
                         f"{(f'{ta:.0f} / {tb:.0f} (to {mt} dB)') if ta and tb else '--'} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"; lines.append(r"\end{tabular}")
    (td / "ablation_views.tex").write_text("\n".join(lines))
    print("wrote figure + table")


if __name__ == "__main__":
    main()
