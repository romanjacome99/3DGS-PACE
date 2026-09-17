"""First-page teaser, on held-out `train` over the 50k-iteration horizon:
  (left)  one held-out test view at four early wall-clock instants for two backends (3DGS and
          Faster-GS), PACE above the fixed schedule, zoomed to the window PACE improves most,
          with the ground truth and the full view;
  (right) per backend, test PSNR at 50k against the wall-clock needed for those 50k iterations,
          with the marker area proportional to the number of Gaussians, fixed schedule -> PACE.

Inputs: outputs/agentic_rl_real/protocol_*_train_50k/summary.json        (time-to-target, sizes)
        outputs/gaussian_evolution/study_train50k_<backend>_{agent,baseline}/
          snapshots.csv, render/view<k>_<tag>.png, view<k>_gt.png        (wall-clock renders)
Output: Paper/figures/teaser.pdf (+ .png).  Run with the env_pytorch_3dgs python.

`--left bars` restores the older left panel (time-to-PSNR bars on held-out `train` at 30k).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_visual_zoom import imread, pick_crop, psnr          # noqa: E402  (shared crop selection)

ROOT = r"C:\Roman\3DGS_PROPOSAL"
RUNS = os.path.join(ROOT, "outputs", "agentic_rl_real")
EVO = os.path.join(ROOT, "outputs", "gaussian_evolution")
FIG = os.path.join(ROOT, "Paper", "figures", "teaser.pdf")
LINEWIDTH_IN = 5.5                           # \linewidth of the ICLR template

# right panel: the 50k time-to-target protocol runs on held-out `train`
PROTOCOL = [("3DGS", "protocol_nonaug_3dgs_train_50k", "#0072B2"),
            ("Faster-GS", "protocol_aug_base_fastergs_train_50k", "#E69F00"),
            ("DashGaussian", "protocol_nonaug_dash_train_50k", "#009E73")]
LADDER = [16, 17, 18, 19, 20, 21]            # PSNR targets summarized by the geometric mean
# right panel: (offset in points, vertical alignment) of each backend's multiplier, kept apart by hand
LBL = {"3DGS": (-4, "top"), "Faster-GS": (4, "bottom"), "DashGaussian": (-13, "top")}
# left panel: the 50k rollouts logged with wall-clock renders
SCENE = "train50k"
VIEW_BLOCKS = [("3DGS", "3dgs", "#0072B2"), ("Faster-GS", "fastergs", "#E69F00")]
VIEW = 6                                     # of the four rendered views, the one where the schedules differ most
VIEW_TAGS = ["t15s", "t30s", "t60s", "t120s"]
REF_TAG = "t60s"
# fixed zoom window (x, y, w, h) on the locomotive lettering: a legible region whose PACE-vs-fixed
# gain matches the automatically selected one (+1.5/+1.3 dB vs +1.2/+1.5 dB on 3DGS/Faster-GS)
CROP = (30, 190, 280, 207)
CROP_FROM = "3dgs"                           # used only when CROP is None
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
BASE_C, BOX_C = "#5a5a5a", "#d62728"
YMAX = 2.35e6                                # y limit of the right panel (Gaussians)
FS = 1.0                                     # font pre-scale; set in main() from the figure width

# older teaser (left panel only): held-out `train` at 30k
TRAIN_RUNS = [("3DGS", "protocol_nonaug_3dgs_train_fr", "#0072B2"),
              ("Faster-GS", "protocol_aug_base_fastergs_train_rerun", "#E69F00"),
              ("DashGaussian", "protocol_nonaug_dash_train", "#009E73")]
TRAIN_LADDER = [16, 17, 18, 19, 20, 21]


def set_fonts(scale):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.5 * scale, "axes.labelsize": 9 * scale, "axes.titlesize": 9.5 * scale,
        "xtick.labelsize": 8 * scale, "ytick.labelsize": 8 * scale,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.7, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42,
    })


def study_dir(backend, method):
    return os.path.join(EVO, "study_%s_%s_%s" % (SCENE, backend, method))


def snapshots(backend, method):
    with open(os.path.join(study_dir(backend, method), "snapshots.csv")) as f:
        return {r["tag"]: r for r in csv.DictReader(f)}


def protocol_summary(run):
    """(gmean wall-clock over the ladder, Gaussians at 50k) for the fixed schedule and PACE."""
    s = json.load(open(os.path.join(RUNS, run, "summary.json")))
    tb, ta = [], []
    for t in s["time_to_target"]:
        if t["target_psnr"] in LADDER and t["baseline_s"] and t["agentic_s"]:
            tb.append(float(t["baseline_s"]))
            ta.append(float(t["agentic_s"]))
    g = lambda v: float(np.exp(np.mean(np.log(v))))                      # noqa: E731
    return (g(tb), s["methods"]["baseline"]["final"]["N"],
            g(ta), s["methods"]["agentic"]["final"]["N"], len(tb))


def first_cross(t, p, tgt):
    for (t0, p0), (t1, p1) in zip(zip(t, p), zip(t[1:], p[1:])):
        if p0 <= tgt <= p1 and p1 > p0:
            return t0 + (t1 - t0) * (tgt - p0) / (p1 - p0)
    return None


def kfmt(x, _=None):
    return "%.1fM" % (x / 1e6) if x >= 1e6 else "%dk" % round(x / 1e3)


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def show(ax, img):
    ax.imshow(np.clip(img, 0, 1), aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def draw_view_blocks(fig, x0, y0, h, gx, gy, gblock, ac):
    """Left panel: one block per backend, PACE over the fixed schedule, plus a reference column."""
    ref = {}
    for _, bk, _ in VIEW_BLOCKS:
        ref[bk] = {m: study_dir(bk, m) for m in ("agent", "baseline")}
    gt = imread(os.path.join(ref[CROP_FROM]["agent"], "render", "view%d_gt.png" % VIEW))
    cx, cy, cw, ch = CROP if CROP else pick_crop(
        gt, imread(os.path.join(ref[CROP_FROM]["agent"], "render", "view%d_%s.png" % (VIEW, REF_TAG))),
        imread(os.path.join(ref[CROP_FROM]["baseline"], "render", "view%d_%s.png" % (VIEW, REF_TAG))),
        0.28, ac)
    pw, fw, fh = ac * h, fig.get_figwidth(), fig.get_figheight()
    bh = 2 * h + gy                                            # height of one backend block

    def rect(x, y, w, hh):
        return [x / fw, 1.0 - (y + hh) / fh, w / fw, hh / fh]

    for bi, (name, bk, col) in enumerate(VIEW_BLOCKS):
        yb = y0 + bi * (bh + gblock)
        snaps = {m: snapshots(bk, m) for m in ("agent", "baseline")}
        for ri, method in enumerate(("agent", "baseline")):
            for ci, t in enumerate(VIEW_TAGS):
                ax = fig.add_axes(rect(x0 + ci * (pw + gx), yb + ri * (h + gy), pw, h))
                img = imread(os.path.join(ref[bk][method], "render", "view%d_%s.png" % (VIEW, t)))
                show(ax, img[cy:cy + ch, cx:cx + cw])
                if ci == 0:
                    ax.text(0.04, 0.95, "PACE" if method == "agent" else "Fixed", transform=ax.transAxes,
                            fontsize=5.4 * FS, color="white", ha="left", va="top", fontweight="bold",
                            bbox=dict(facecolor=col if method == "agent" else BASE_C, alpha=0.92, pad=0.9,
                                      edgecolor="none"))
                if bi == 0 and ri == 0:
                    ax.set_title("%g s" % float(snaps[method][t]["trigger_s"]), fontsize=7.2 * FS, pad=2.5)
        fig.text((x0 - 0.045) / fw, 1.0 - (yb + bh / 2) / fh, name, rotation=90, va="center", ha="center",
                 fontsize=7.0 * FS, color=col, fontweight="bold")

    xr = x0 + len(VIEW_TAGS) * (pw + gx)                       # reference column: target, then context
    axg = fig.add_axes(rect(xr, y0, ac * bh, bh))
    show(axg, gt[cy:cy + ch, cx:cx + cw])
    for sp in axg.spines.values():
        sp.set_visible(True)
        sp.set_color(BOX_C)
        sp.set_linewidth(0.9)
    axg.set_title("Ground truth", fontsize=7.2 * FS, pad=2.5, color="#444444")
    axv = fig.add_axes(rect(xr, y0 + bh + gblock, ac * bh, bh))
    vh, vw = gt.shape[:2]                                      # window of the panel's aspect, no stretching
    ww = min(vw, int(round(vh * ac)))
    wx = int(np.clip(cx + cw // 2 - ww // 2, 0, vw - ww))
    show(axv, gt[:, wx:wx + ww])
    axv.add_patch(Rectangle((cx - wx, cy), cw, ch, fill=False, ec=BOX_C, lw=0.9))
    return ac * bh


def scene_speedups():
    """Per scene and backend: (scene, geometric-mean speed-up, final Gaussians PACE, final Gaussians
    fixed, seconds saved, seconds for the fixed schedule, seconds for PACE). The times are measured
    at the hardest PSNR target both methods reach, so they are the wall-clock each method needs to
    train that scene to its plateau, and their difference is what PACE removes. Taken from the
    held-out table's own generator (same runs, same per-scene PSNR ladders, same first-crossing
    interpolation), so the teaser and \\cref{tab:heldout} can never disagree."""
    import make_heldout_tables as H
    key = {"3dgs": "3DGS", "fgs": "Faster-GS", "dash": "DashGaussian"}
    out = {v: [] for v in key.values()}
    for scene, _, ladder, runs in H.SCENES:
        for bk, _ in H.BACKENDS:
            cv = H.load_curve(runs[bk])
            if cv is None:
                continue
            rows, g = H.ttt(cv, ladder)
            both = [(tb, ta) for _, tb, ta, su in rows if tb and ta]
            if not (g and both):
                continue
            out[key[bk]].append((scene, g, cv["agentic"][-1][2], cv["baseline"][-1][2],
                                 both[-1][0] - both[-1][1], both[-1][0], both[-1][1]))
    return out


def draw_scenes(axR):
    """Right panel: the wall-clock each method needs to train a scene to its plateau, over six
    held-out scenes -- the fixed schedule (hollow) against PACE (solid). The average is geometric so
    that the ratio of the two bars is exactly the plateau speed-up printed beside them; an arithmetic
    mean would print a factor the bars do not support. Note this is the speed-up at the hardest
    target both methods reach, so it is not \\cref{tab:heldout}'s geometric mean over the whole
    PSNR ladder, which is the more conservative number quoted in the caption."""
    sp = scene_speedups()
    fig = axR.figure                                   # split the panel in two, sharing the rows
    pos = axR.get_position()
    ws, gap = pos.width * 0.615, pos.width * 0.105
    axR.set_position([pos.x0, pos.y0, ws, pos.height])
    axG = fig.add_axes([pos.x0 + ws + gap, pos.y0, pos.width - ws - gap, pos.height])
    tr = matplotlib.transforms.blended_transform_factory(axR.transAxes, axR.transData)
    for bi, (name, _, col) in enumerate(PROTOCOL):
        v = sp[name]
        na = np.mean([x[2] for x in v]) / 1e6          # mean model size over the same scenes
        nb = np.mean([x[3] for x in v]) / 1e6
        gm = float(np.exp(np.mean(np.log([x[1] for x in v]))))
        tb = float(np.exp(np.mean(np.log([x[5] for x in v]))))   # gmean wall-clock, fixed schedule
        ta = float(np.exp(np.mean(np.log([x[6] for x in v]))))   # gmean wall-clock, PACE
        ms = tb / ta                                             # == gmean of the per-scene ratios
        y0 = 2.0 - bi
        axR.text(0.012, y0 + 0.40, name, transform=tr, ha="left", va="center",
                 fontsize=7.8 * FS, color=col, fontweight="bold")
        for val, dy, face in ((tb, 0.18, "white"), (ta, -0.18, col)):
            axR.barh([y0 + dy], [val], height=0.31, facecolor=face, edgecolor=col, linewidth=1.1,
                     zorder=3)
        axR.annotate("%.2f\u00d7" % ms, xy=(tb, y0), xytext=(8, 0), textcoords="offset points",
                     ha="left", va="center", fontsize=7.8 * FS, color=col, fontweight="bold")
        for k, (val, face) in enumerate([(nb, "white"), (na, col)]):
            axG.barh([y0 + (0.18 if k == 0 else -0.18)], [val], height=0.31, facecolor=face,
                     edgecolor=col, linewidth=1.1, zorder=3)
        print("%-12s gmean wall-clock %.0f s -> %.0f s over %d scenes (plateau %.2fx; ladder gmean "
              "%.2fx)  mean G %.2fM -> %.2fM" % (name, tb, ta, len(v), ms, gm, nb, na))
    axR.set_yticks([])
    axR.set_ylim(-1.62, 3.0)
    axR.set_xlim(0, 520)
    axR.set_xticks([0, 200, 400])
    axR.set_xlabel("Wall-clock to plateau (s)", labelpad=1.5)
    style(axR)
    axR.grid(False, axis="y")
    axR.legend(handles=[Patch(facecolor="white", edgecolor="#555555", lw=1.1, label="Fixed schedule"),
                        Patch(facecolor="#555555", edgecolor="#555555", label="PACE (ours)")],
               loc="lower right", bbox_to_anchor=(1.03, -0.015), frameon=False, fontsize=7.2 * FS,
               handlelength=1.3, handleheight=0.85, borderpad=0.2, labelspacing=0.3,
               handletextpad=0.45)
    axG.set_yticks([])
    axG.set_ylim(axR.get_ylim())
    axG.set_xlim(0, 2.45)
    axG.set_xticks([0, 1, 2])
    axG.set_xlabel("Gaussians (M)", labelpad=1.5)
    style(axG)
    axG.grid(False, axis="y")


def best_so_far(run, method):
    """(t, best test PSNR up to t, N) for one method: the eval curve is noisy and non-monotone, so
    quality at time t is the best reached by t -- the form the time-to-target metric is defined on."""
    v = []
    with open(os.path.join(RUNS, run, "curve.csv")) as f:
        for r in csv.DictReader(f):
            if r["method"] == method:
                v.append((float(r["t"]), float(r["test_psnr"]), float(r["N"])))
    v = np.array(sorted(v))
    v[:, 1] = np.maximum.accumulate(v[:, 1])
    return v


def reach(c, tgt):
    """First wall-clock at which the (monotone) curve c reaches tgt, or None."""
    t, p = c[:, 0], c[:, 1]
    if p[-1] < tgt - 1e-9:
        return None
    for (t0, p0), (t1, p1) in zip(zip(t, p), zip(t[1:], p[1:])):
        if p0 < tgt <= p1:
            return t0 + (t1 - t0) * (tgt - p0) / (p1 - p0)
    return float(t[0])


def draw_curves(axR):
    """Right panel: held-out quality against wall-clock. PACE (solid) reaches the quality the fixed
    schedule (dashed) needs the whole 50k run for, in a fraction of the time; the endpoint marker
    area is the number of Gaussians, so the compaction shows on the same plot."""
    area = lambda n: 18.0 + 140.0 * n / 2.0e6          # noqa: E731  (marker area from Gaussian count)
    tags = []
    for name, run, col in PROTOCOL:
        a, b = best_so_far(run, "agentic"), best_so_far(run, "baseline")
        gi = np.geomspace(max(a[0, 0], b[0, 0]), min(a[-1, 0], b[-1, 0]), 200)
        ya, yb = np.interp(gi, a[:, 0], a[:, 1]), np.interp(gi, b[:, 0], b[:, 1])
        axR.fill_between(gi, yb, ya, where=ya > yb, color=col, alpha=0.13, lw=0, zorder=1)
        axR.plot(b[:, 0], b[:, 1], ls=(0, (3, 2)), color=col, lw=1.15, zorder=2, alpha=0.85)
        axR.plot(a[:, 0], a[:, 1], ls="-", color=col, lw=1.6, zorder=3)
        axR.scatter([b[-1, 0]], [b[-1, 1]], s=area(b[-1, 2]), marker="o", facecolor="white",
                    edgecolor=col, linewidth=1.2, zorder=4)
        axR.scatter([a[-1, 0]], [a[-1, 1]], s=area(a[-1, 2]) * 1.7, marker="*", facecolor=col,
                    edgecolor="white", linewidth=0.5, zorder=5)
        # one uniform rule, readable straight off the axes: the wall-clock each schedule needs for
        # the best quality the fixed schedule reaches anywhere in its 50k run
        tgt = float(b[-1, 1])
        ta, tb = reach(a, tgt), reach(b, tgt)
        if ta is not None:
            axR.annotate("", xy=(ta, tgt), xytext=(tb, tgt),
                         arrowprops=dict(arrowstyle="<|-", color=col, lw=1.0, mutation_scale=7,
                                         shrinkA=1, shrinkB=1))
            tags.append((name, col, "%.1f× faster" % (tb / ta)))
            print("%-12s to %.2f dB: PACE %.0f s vs fixed %.0f s -> %.2fx ; N %s vs %s"
                  % (name, tgt, ta, tb, tb / ta, kfmt(a[-1, 2]), kfmt(b[-1, 2])))
        else:
            # never reaches the fixed schedule's best quality; its win is compaction, labelled as such
            tags.append((name, col, "%.1f× smaller" % (b[-1, 2] / a[-1, 2])))
            print("%-12s never reaches the fixed schedule's %.2f dB; model %.1fx smaller (%s vs %s)"
                  % (name, tgt, b[-1, 2] / a[-1, 2], kfmt(a[-1, 2]), kfmt(b[-1, 2])))
    axR.set_xscale("log")
    axR.set_xlim(7, 2300)
    axR.set_xticks([10, 30, 100, 300, 1000])
    axR.set_xticklabels(["10", "30", "100", "300", "1000"])
    axR.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axR.set_ylim(12.9, 22.9)
    axR.set_yticks([14, 16, 18, 20, 22])
    axR.set_xlabel("Wall-clock (s, log)", labelpad=1.5)
    axR.set_ylabel("Test PSNR reached (dB)", labelpad=1.5)
    style(axR)
    # the multiplier rides in the legend entry: on a log axis every backend ends in the same corner,
    # so per-arrow labels collide
    leg = axR.legend(handles=[plt.Line2D([], [], ls="-", color=c, lw=1.7,
                                         label="%s   %s" % (n, m)) for n, c, m in tags],
                     loc="lower right", bbox_to_anchor=(1.025, -0.03), frameon=False,
                     fontsize=7.4 * FS, handlelength=1.3, borderpad=0.2, labelspacing=0.35,
                     handletextpad=0.45)
    for t, (_, c, _) in zip(leg.get_texts(), tags):
        t.set_color(c)
        t.set_fontweight("bold")
    axR.add_artist(leg)
    axR.legend(handles=[plt.Line2D([], [], ls="-", color="#555555", lw=1.6, label="PACE (ours)"),
                        plt.Line2D([], [], ls=(0, (3, 2)), color="#555555", lw=1.15,
                                   label="Fixed schedule")],
               loc="upper left", bbox_to_anchor=(-0.015, 1.04), frameon=False, fontsize=7.0 * FS,
               handlelength=1.6, borderpad=0.15, labelspacing=0.25, handletextpad=0.4)


def draw_scatter(axR):
    """Right panel: quality against wall-clock at the 50k endpoint, marker area = model size."""
    area = lambda n: 26.0 + 200.0 * n / 2.0e6          # noqa: E731  (marker area from Gaussian count)
    for name, run, col in PROTOCOL:
        s = json.load(open(os.path.join(RUNS, run, "summary.json")))
        b, a = s["methods"]["baseline"]["final"], s["methods"]["agentic"]["final"]
        sa, sb = area(a["N"]), area(b["N"])
        axR.annotate("", xy=(a["t"], a["test_psnr"]), xytext=(b["t"], b["test_psnr"]),
                     arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2, mutation_scale=8,
                                     shrinkA=np.sqrt(sb) / 2 + 3, shrinkB=np.sqrt(sa) / 2 + 4))
        axR.scatter([b["t"]], [b["test_psnr"]], s=sb, marker="o", facecolor="white", edgecolor=col,
                    linewidth=1.3, zorder=3)
        axR.scatter([a["t"]], [a["test_psnr"]], s=sa * 1.6, marker="*", facecolor=col, edgecolor="white",
                    linewidth=0.5, zorder=4)
        axR.plot([], [], "*", ms=9, mfc=col, mec="white", label=name)     # backends named in the legend
        # the acceleration factor is the geometric-mean speed-up to reach the LADDER targets (the
        # headline metric), not the ratio of the two plotted end-times; the caption says so
        gb, _, ga, _, ntg = protocol_summary(run)
        dx, dy, ha, va = {"3DGS": (0, -10, "center", "top"),
                          "Faster-GS": (0, 10, "center", "bottom"),
                          "DashGaussian": (8, -8, "left", "top")}[name]
        axR.annotate("%.2f×" % (gb / ga), xy=(a["t"], a["test_psnr"]), xytext=(dx, dy),
                     textcoords="offset points", fontsize=7.4 * FS, color=col, fontweight="bold",
                     ha=ha, va=va)
        print("%-12s 50k: fixed %.0f s / %.2f dB / %s  ->  PACE %.0f s / %.2f dB / %s   "
              "gmean speed-up to %d targets %.2fx"
              % (name, b["t"], b["test_psnr"], kfmt(b["N"]), a["t"], a["test_psnr"], kfmt(a["N"]),
                 ntg, gb / ga))
    axR.set_xlim(520, 1980)
    axR.set_xticks([600, 900, 1200, 1500, 1800])
    axR.set_ylim(20.9, 22.62)
    axR.set_yticks([21.0, 21.5, 22.0])
    axR.set_xlabel("Wall-clock for 50k iterations (s)", labelpad=1.5)
    axR.set_ylabel("Test PSNR at 50k (dB)", labelpad=1.5)
    style(axR)
    # circle = fixed schedule and star = PACE are stated in the caption
    leg = axR.legend(loc="upper right", bbox_to_anchor=(1.01, 1.12), frameon=False, fontsize=7.0 * FS,
                     handlelength=1.1, borderpad=0.2, labelspacing=0.28, handletextpad=0.4)
    axR.add_artist(leg)
    sizes = [(200_000, "0.2M"), (1_000_000, "1M"), (2_000_000, "2M")]     # marker area = Gaussian count
    axR.legend(handles=[axR.scatter([], [], s=area(n), marker="o", facecolor="none", edgecolor="#777777",
                                    linewidth=1.0, label=lab) for n, lab in sizes],
               loc="upper left", bbox_to_anchor=(-0.02, 1.15), frameon=False, fontsize=6.6 * FS, ncol=3,
               handlelength=1.0, borderpad=0.1, labelspacing=0.2, handletextpad=0.25, columnspacing=0.5,
               title="Gaussians at 50k", title_fontsize=6.6 * FS)


def draw_bars(ax):
    """Older left panel: wall-clock to reach each PSNR level on held-out `train` (30k)."""
    def load_curve(run):
        out = {"baseline": [], "agentic": []}
        with open(os.path.join(RUNS, run, "curve.csv")) as f:
            for r in csv.DictReader(f):
                out[r["method"]].append((float(r["t"]), float(r["test_psnr"])))
        return {m: np.array(sorted(v)) for m, v in out.items()}

    def time_to(summary, tgt, method):
        for t in summary["time_to_target"]:
            if abs(t["target_psnr"] - tgt) < 1e-6:
                return t["baseline_s"] if method == "baseline" else t["agentic_s"]
        return None

    nb = len(TRAIN_RUNS)
    pair_w = 0.26
    bar_w = pair_w / 2
    xg = np.arange(len(TRAIN_LADDER))
    for bi, (name, run, col) in enumerate(TRAIN_RUNS):
        c = load_curve(run)
        s = json.load(open(os.path.join(RUNS, run, "summary.json")))
        x0 = xg - pair_w * nb / 2 + bi * pair_w
        tb = [first_cross(c["baseline"][:, 0], c["baseline"][:, 1], tg) for tg in TRAIN_LADDER]
        ta = [first_cross(c["agentic"][:, 0], c["agentic"][:, 1], tg) for tg in TRAIN_LADDER]
        tb[-1], ta[-1] = time_to(s, 21.0, "baseline"), time_to(s, 21.0, "agentic")
        ax.bar(x0 + bar_w / 2, tb, bar_w, facecolor="white", edgecolor=col, lw=1.1, zorder=3)
        ax.bar(x0 + 1.5 * bar_w, ta, bar_w, facecolor=col, edgecolor=col, lw=0.6, zorder=3)
        ax.annotate("%.1f\u00d7" % (tb[-1] / ta[-1]), xy=(x0[-1] + bar_w, max(tb[-1], ta[-1])), xytext=(0, 2.5),
                    textcoords="offset points", ha="center", va="bottom", fontsize=6.8, color=col, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(8, 1900)
    ax.set_yticks([10, 30, 100, 300, 1000])
    ax.set_yticklabels(["10", "30", "100", "300", "1000"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(xg)
    ax.set_xticklabels(["%d" % tg for tg in TRAIN_LADDER])
    ax.set_xlim(-0.55, len(TRAIN_LADDER) - 0.45)
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("Test PSNR reached (dB)")
    ax.set_ylabel("Wall-clock needed (s)")
    style(ax)
    ax.grid(False, axis="x")
    handles = [Patch(facecolor=col, edgecolor=col, label=name) for name, _, col in TRAIN_RUNS]
    handles += [Patch(facecolor="white", edgecolor="#444444", lw=1.1, label="Fixed schedule"),
                Patch(facecolor="#444444", edgecolor="#444444", label="PACE (ours)")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7.2, handlelength=1.1, handleheight=0.85,
              borderpad=0.2, labelspacing=0.25, ncol=2, columnspacing=0.9)


RIGHT = {}          # right-panel variants, filled below (draw_* are defined above main)


def main():
    global FS
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", choices=["views", "bars"], default="views")
    ap.add_argument("--right", choices=["scenes", "curves", "scatter"], default="scenes")
    a = ap.parse_args()

    if a.left == "bars":
        set_fonts(1.0)
        fig = plt.figure(figsize=(7.1, 2.45))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.28, left=0.07, right=0.985,
                              top=0.97, bottom=0.18)
        draw_bars(fig.add_subplot(gs[0, 0]))
        RIGHT[a.right](fig.add_subplot(gs[0, 1]))
    else:
        ac, h, gx, gy, gblock = 1.35, 0.48, 0.025, 0.025, 0.11   # crop aspect, panel height (in)
        lab, top, right_w, gap, bottom = 0.17, 0.22, 3.10, 0.60, 0.46
        bh = 2 * h + gy
        cols_w = len(VIEW_TAGS) * (ac * h + gx)
        left_w = lab + cols_w + ac * bh
        fig_w = left_w + gap + right_w + 0.04
        fig_h = top + 2 * bh + gblock + bottom
        FS = fig_w / LINEWIDTH_IN                                # the figure is scaled to \linewidth
        set_fonts(FS)
        fig = plt.figure(figsize=(fig_w, fig_h))
        draw_view_blocks(fig, lab, top, h, gx, gy, gblock, ac)
        axR = fig.add_axes([(left_w + gap) / fig_w, (bottom - 0.06) / fig_h,
                            right_w / fig_w, (2 * bh + gblock + top - 0.30) / fig_h])
        RIGHT[a.right](axR)

    fig.savefig(FIG, bbox_inches="tight")
    fig.savefig(FIG.replace(".pdf", ".png"), dpi=220, bbox_inches="tight")
    print("wrote", FIG)


RIGHT.update(scenes=draw_scenes, curves=draw_curves, scatter=draw_scatter)

if __name__ == "__main__":
    main()
