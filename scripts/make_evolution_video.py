"""Side-by-side flight through the saved Gaussian states: PACE vs the fixed schedule.

One video per (scene, backend). The camera flies a continuous path through the scene while the
two panels step through the snapshots saved along training, both panels at the SAME wall-clock
instant, with the Gaussian-count / ms-per-iteration / held-out-PSNR curves drawn underneath and a
marker sweeping along them.

Nothing is trained and nothing is re-evaluated: this reads the .ply states that
scripts/run_web_snapshots.sh already wrote under outputs/gaussian_evolution_web/ and the per-block
stats.csv logged beside them.

Two stages, because the Faster-GS conda env has no matplotlib:

  render   GPU, runs in the backend's own env, writes JPEG frames + frames.json
  compose  CPU, runs in env_pytorch_3dgs, draws the curve strip and encodes the .mp4

  python scripts/make_evolution_video.py render  --scene train --backend 3dgs --frames-dir DIR
  python scripts/make_evolution_video.py compose --frames-dir DIR --out outputs/videos/train_3dgs.mp4

scripts/run_evolution_videos.sh drives both stages for all nine (scene, backend) pairs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Roman\3DGS_PROPOSAL")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WEB = ROOT / "outputs" / "gaussian_evolution_web"

# Must stay in step with cfg_for() in scripts/run_web_snapshots.sh -- the video has to replay the
# same protocol the snapshots came from.
CFG = {
    "3dgs": {"train": "configs/final_accel_3dgs.json",
             "stump": "configs/final_accel_3dgs_stump.json",
             "*": "configs/final_accel_3dgs_tandt.json"},
    "fastergs": {"train": "outputs/agentic_rl_real/real_fastergs_accel_aug_base/resolved_config.json",
                 "stump": "agentic_gs_phase1/configs/real_fastergs_accel_aug_base_stump.json",
                 "*": "agentic_gs_phase1/configs/real_fastergs_accel_aug_base_tandt.json"},
    "dash": {"train": "configs/final_accel_dash_fixed_horizon.json",
             "stump": "configs/final_accel_dash_stump.json",
             "*": "configs/final_accel_dash_fixed_tandt.json"},
}
BACKEND_LABEL = {"3dgs": "3DGS", "fastergs": "Faster-GS", "dash": "DashGaussian"}
# Okabe-Ito, the paper's per-backend hues (figures/pace_mechanism.pdf)
BACKEND_COLOUR = {"3dgs": "#0072B2", "fastergs": "#009E73", "dash": "#D55E00"}
BASELINE_COLOUR = "#555555"

# How long each snapshot is held, in units of the per-stage dwell. The early instants are where the
# two schedules diverge, the last one is where the model-size gap is the point, so both get room.
DWELL = {"init": 0.7, "t5s": 0.9, "t15s": 1.0, "t30s": 1.0, "t60s": 1.0,
         "t120s": 1.1, "t300s": 1.3, "t600s": 1.3, "final": 2.2}


def cfg_for(scene: str, backend: str) -> str:
    table = CFG[backend]
    return table.get(scene, table["*"])


def read_snapshots(run: Path) -> list[dict]:
    rows = []
    for r in csv.DictReader(open(run / "snapshots.csv")):
        rows.append({"tag": r["tag"], "trigger_s": float(r["trigger_s"]), "t": float(r["time_s"]),
                     "iter": int(r["iter"]), "N": int(r["gaussians"]), "psnr": float(r["psnr"]),
                     "ply": r["ply"]})
    return rows


# --------------------------------------------------------------------------------------- camera

def quat_from_matrix(m: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z), Shepperd's branchless-enough variant."""
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    if np.dot(q0, q1) < 0.0:          # same rotation, opposite hemisphere: take the short way
        q1 = -q1
    d = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if d > 0.9995:
        return (q0 + u * (q1 - q0)) / np.linalg.norm(q0 + u * (q1 - q0))
    th = math.acos(d)
    return (math.sin((1 - u) * th) * q0 + math.sin(u * th) * q1) / math.sin(th)


def smooth_positions(C: np.ndarray, win: int) -> np.ndarray:
    """Moving average along the capture order; the raw COLMAP poses of a hand-held walk jitter."""
    if win < 3:
        return C
    k = win // 2
    pad = np.concatenate([C[:1].repeat(k, 0), C, C[-1:].repeat(k, 0)], 0)
    ker = np.ones(win) / win
    return np.stack([np.convolve(pad[:, i], ker, mode="valid") for i in range(3)], 1)


def build_path(cams, n_frames: int, smooth_win: int, span: float):
    """Interpolate the capture's own poses into `n_frames` (position, rotation) pairs.

    Flying the poses COLMAP solved for keeps every frame inside the region the capture covered, so
    both panels are judged on geometry the reconstruction actually saw. A synthetic orbit would
    swing through unobserved angles where both models show holes.
    """
    order = sorted(range(len(cams)), key=lambda i: cams[i].image_name)
    cams = [cams[i] for i in order]
    C = np.stack([c.camera_center.detach().cpu().numpy().astype(np.float64) for c in cams])
    Q = np.stack([quat_from_matrix(np.asarray(c.R, dtype=np.float64)) for c in cams])
    C = smooth_positions(C, smooth_win)

    n = len(cams)
    # `span` < 1 uses the middle of the capture, where the trajectory is usually best conditioned.
    lo = (1.0 - span) * 0.5 * (n - 1)
    hi = (n - 1) - (1.0 - span) * 0.5 * (n - 1)
    us = np.linspace(lo, hi, n_frames)
    out = []
    for u in us:
        i0 = int(math.floor(u))
        i0 = min(max(i0, 0), n - 2)
        f = float(u - i0)
        pos = C[i0] * (1 - f) + C[i0 + 1] * f
        rot = matrix_from_quat(slerp(Q[i0], Q[i0 + 1], f))
        out.append((pos, rot))
    return out


def make_camera(base, R_c2w: np.ndarray, pos: np.ndarray, width: int):
    """A render-only clone of `base` at a new pose. Only the matrices the rasterizer reads change."""
    import copy
    import torch
    from utils.graphics_utils import getWorld2View2

    cam = copy.copy(base)
    T = -R_c2w.T @ pos                       # world->cam translation for this centre
    cam.R = R_c2w
    cam.T = T
    h = int(round(width * base.image_height / base.image_width))
    cam.image_width, cam.image_height = int(width), h
    w2c = torch.tensor(getWorld2View2(R_c2w, T, base.trans, base.scale)).transpose(0, 1).cuda()
    cam.world_view_transform = w2c
    cam.full_proj_transform = w2c.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0)).squeeze(0)
    cam.camera_center = w2c.inverse()[3, :3]
    return cam


# --------------------------------------------------------------------------------------- stages

def plan_stages(agent: list[dict], baseline: list[dict], n_frames: int):
    """Frame ranges per snapshot, and which state each side shows in each.

    A run that stopped early (Faster-GS on ignatius hits the 3M safety rail at 11.6k) simply has no
    snapshot at the later instants; it holds its last state and the overlay says so.
    """
    # Union of both sides' instants, in time order: whichever run stopped early (Faster-GS on
    # ignatius hits the 3M rail at 11.6k) must not delete the other's later snapshots from the film.
    seen = {}
    for r in agent + baseline:
        seen.setdefault(r["tag"], r["trigger_s"])
    order = sorted(seen.items(), key=lambda kv: (kv[1], kv[0] == "final"))
    tags = [t for t, _ in order if t != "final"] + (["final"] if "final" in seen else [])
    weights = np.array([DWELL.get(t, 1.0) for t in tags], dtype=float)
    edges = np.concatenate([[0.0], np.cumsum(weights / weights.sum())]) * n_frames
    edges = np.round(edges).astype(int)

    def state_at(rows, tag, trigger):
        exact = next((r for r in rows if r["tag"] == tag), None)
        if exact is not None:
            return exact, False
        earlier = [r for r in rows if r["trigger_s"] <= trigger]
        return (earlier[-1] if earlier else rows[0]), True

    stages = []
    for i, tag in enumerate(tags):
        trigger = seen[tag]
        a, a_held = state_at(agent, tag, trigger)
        b, b_held = state_at(baseline, tag, trigger)
        stages.append({"tag": tag, "trigger_s": trigger,
                       "frames": [int(edges[i]), int(edges[i + 1])],
                       "agent": a, "baseline": b,
                       "agent_held": a_held, "baseline_held": b_held})
    return stages


# --------------------------------------------------------------------------------------- render

def cmd_render(args) -> int:
    import torch
    from PIL import Image

    from agentic_gs_phase1.envs import AgenticGSEnv

    runs = {m: WEB / f"{args.scene}_{args.backend}_{m}" for m in ("agent", "baseline")}
    for m, d in runs.items():
        if not (d / "snapshots.csv").exists():
            print(f"[error] missing {d}", file=sys.stderr)
            return 2

    out = Path(args.frames_dir)
    (out / "agent").mkdir(parents=True, exist_ok=True)
    (out / "baseline").mkdir(parents=True, exist_ok=True)

    cfg = json.loads((ROOT / cfg_for(args.scene, args.backend)).read_text())
    cfg["fastergs_acknowledge_grad_bug"] = True
    cfg["save_episode_models"] = False
    env = AgenticGSEnv(cfg, run_dir=ROOT / "outputs" / "gaussian_evolution" / "_video_run", seed=0)
    env.reset(args.scene, episode_id=0)

    base = list(env.scene.getTestCameras())[0]
    path_cams = list(env.scene.getTrainCameras())
    print(f"[{args.scene}/{args.backend}] {len(path_cams)} capture poses, "
          f"render {args.width}x{int(round(args.width * base.image_height / base.image_width))}")

    path = build_path(path_cams, args.frames, args.smooth, args.span)
    snaps = {m: read_snapshots(d) for m, d in runs.items()}
    stages = plan_stages(snaps["agent"], snaps["baseline"], args.frames)

    sh_deg, opt_type = env.dataset.sh_degree, env.opt.optimizer_type
    meta = {"scene": args.scene, "backend": args.backend, "frames": args.frames,
            "width": args.width, "stages": [], "runs": {m: str(d) for m, d in runs.items()}}

    for st in stages:
        f0, f1 = st["frames"]
        if f1 <= f0:
            continue
        for method in ("agent", "baseline"):
            rec = st[method]
            ply = runs[method] / "ply" / rec["ply"]
            g = env.backend.make_gaussians(sh_deg, opt_type)
            g.load_ply(str(ply))
            with torch.no_grad():
                for f in range(f0, f1):
                    pos, rot = path[f]
                    cam = make_camera(base, rot, pos, args.width)
                    img = env.backend.render_image(cam, g, env.pipe, env.background,
                                                   use_trained_exp=False)
                    arr = (img.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(arr).save(out / method / f"f{f:05d}.jpg", quality=92)
            del g
            torch.cuda.empty_cache()
            print(f"  [{st['tag']:>6}] {method:<8} frames {f0}-{f1 - 1}  N={rec['N'] / 1e3:.0f}k  "
                  f"it={rec['iter']}  psnr={rec['psnr']:.2f}", flush=True)
        meta["stages"].append({k: st[k] for k in ("tag", "trigger_s", "frames", "agent",
                                                  "baseline", "agent_held", "baseline_held")})

    (out / "frames.json").write_text(json.dumps(meta, indent=1))
    env.close()
    print("render done ->", out)
    return 0


# -------------------------------------------------------------------------------------- compose

def load_stats(run: Path) -> dict:
    t, n, ms, psnr = [], [], [], []
    for r in csv.DictReader(open(run / "stats.csv")):
        if int(r["block"]) == 0:
            continue                      # block 0 is the pre-training state: t=0, ms/iter=0
        t.append(float(r["t"]))
        n.append(float(r["N"]))
        ms.append(float(r["ms_per_iter"]))
        psnr.append(float(r["val_psnr"]))
    return {"t": np.array(t), "N": np.array(n), "ms": np.array(ms), "psnr": np.array(psnr)}


def build_curve_strip(meta: dict, size: tuple[int, int]):
    """Static curve panels + the data->pixel maps needed to stamp a marker on them each frame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w, h = size
    colour = BACKEND_COLOUR[meta["backend"]]
    stats = {m: load_stats(Path(p)) for m, p in meta["runs"].items()}

    dpi = 100.0
    fig, axes = plt.subplots(1, 3, figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")
    panels = [("N", "Gaussians", lambda v: f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.0f}k"),
              ("ms", "ms / iteration", None),
              ("psnr", "held-out PSNR (dB)", None)]
    maps = []
    for ax, (key, label, fmt) in zip(axes, panels):
        ax.plot(stats["baseline"]["t"], stats["baseline"][key], color=BASELINE_COLOUR,
                lw=2.2, ls="--", label="fixed schedule", zorder=2)
        ax.plot(stats["agent"]["t"], stats["agent"][key], color=colour, lw=2.8,
                label="PACE", zorder=3)
        ax.set_xlabel("wall-clock training time (s)", fontsize=12)
        ax.set_title(label, fontsize=14, loc="left")
        ax.tick_params(labelsize=11)
        ax.grid(alpha=0.25, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if fmt is not None:
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _p, _f=fmt: _f(v)))
    axes[0].legend(fontsize=12, loc="upper left", frameon=False)
    fig.tight_layout(pad=1.4)
    fig.canvas.draw()

    for ax in axes:
        # Linear axes, so two probes give the exact data->pixel map. Matplotlib's display origin is
        # bottom-left and the image's is top-left, hence the flip on y.
        (x0, y0), (x1, y1) = ax.transData.transform([(0.0, 0.0), (1.0, 1.0)])
        maps.append({"xs": x1 - x0, "x0": x0, "ys": y1 - y0, "y0": y0, "h": h,
                     "clip": [ax.get_xlim(), ax.get_ylim()]})

    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf[:, :, ::-1].copy(), maps, stats      # BGR for OpenCV


def draw_marker(strip, maps, stats, t_agent, t_base, colour_bgr, base_bgr):
    import cv2
    img = strip.copy()
    keys = ["N", "ms", "psnr"]
    for mp, key in zip(maps, keys):
        (xlo, xhi), (ylo, yhi) = mp["clip"]

        def to_px(tv, vv):
            x = mp["x0"] + mp["xs"] * tv
            y = mp["h"] - (mp["y0"] + mp["ys"] * vv)
            return int(round(x)), int(round(y))

        for method, tt, col in (("agent", t_agent, colour_bgr), ("baseline", t_base, base_bgr)):
            s = stats[method]
            if len(s["t"]) == 0:
                continue
            tv = float(min(tt, s["t"][-1]))
            vv = float(np.interp(tv, s["t"], s[key]))
            if not (xlo <= tv <= xhi and ylo <= vv <= yhi):
                continue
            x, y = to_px(tv, vv)
            if method == "agent":
                top = mp["h"] - (mp["y0"] + mp["ys"] * yhi)
                bot = mp["h"] - (mp["y0"] + mp["ys"] * ylo)
                cv2.line(img, (x, int(round(top))), (x, int(round(bot))), (200, 200, 200), 1)
            cv2.circle(img, (x, y), 7, col, -1)
            cv2.circle(img, (x, y), 7, (255, 255, 255), 2)
    return img


def hex_rgb_light(h: str, amount: float = 0.45) -> tuple[int, int, int]:
    """The backend hue, lifted toward white: it is drawn on a darkened plate over the render, where
    the print colour (3DGS blue especially) is too dark to read once the page scales the video down."""
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return tuple(int(c + (255 - c) * amount) for c in (r, g, b))


def hex_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def cmd_compose(args) -> int:
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    frames_dir = Path(args.frames_dir)
    meta = json.loads((frames_dir / "frames.json").read_text())
    scene, backend = meta["scene"], meta["backend"]
    colour_bgr, base_bgr = hex_bgr(BACKEND_COLOUR[backend]), hex_bgr(BASELINE_COLOUR)

    probe = cv2.imread(str(frames_dir / "agent" / "f00000.jpg"))
    if probe is None:
        print("[error] no frames in", frames_dir, file=sys.stderr)
        return 2
    ph, pw = probe.shape[:2]

    W, gutter, margin, header = args.width, 16, 24, 90
    panel_w = (W - 2 * margin - gutter) // 2
    panel_h = int(round(panel_w * ph / pw))
    strip_h = args.height - header - panel_h - 2 * margin
    if strip_h < 220:
        print(f"[error] strip height {strip_h}px too small; raise --height", file=sys.stderr)
        return 2
    strip, maps, stats = build_curve_strip(meta, (W, strip_h))

    import matplotlib
    fpath = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    font_b = ImageFont.truetype(str(fpath / "DejaVuSans-Bold.ttf"), 28)
    font_m = ImageFont.truetype(str(fpath / "DejaVuSans.ttf"), 25)
    font_t = ImageFont.truetype(str(fpath / "DejaVuSans-Bold.ttf"), 34)
    font_s = ImageFont.truetype(str(fpath / "DejaVuSans.ttf"), 21)

    # Size each label plate to its own widest line: "fixed schedule - DashGaussian" overruns a
    # fixed-width plate and the tail of the text lands unreadably on whatever the render shows.
    scratch = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    plate_w = {}
    for method, name in (("agent", "PACE"), ("baseline", "fixed schedule")):
        widest = scratch.textlength(f"{name} - {BACKEND_LABEL[backend]}", font=font_b)
        for st in meta["stages"]:
            r = st[method]
            for line in (f"{r['t']:.0f} s  -  iter {r['iter']:,}",
                         f"{r['N'] / 1e6:.2f}M Gaussians  -  {r['psnr']:.2f} dB"):
                widest = max(widest, scratch.textlength(line, font=font_m))
        plate_w[method] = int(widest) + 30

    per_frame = {}
    for st in meta["stages"]:
        for f in range(st["frames"][0], st["frames"][1]):
            per_frame[f] = st

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # H.264 through ffmpeg when it is available: OpenCV's default mp4v is MPEG-4 Part 2, which no
    # browser will play, so a page embedding those files shows a dead player.
    writer = _open_writer(out, W, args.height, args.fps, args.crf)
    if writer is None:
        return 2

    title = f"{scene}  -  {BACKEND_LABEL[backend]}"
    for f in range(meta["frames"]):
        st = per_frame.get(f)
        if st is None:
            continue
        canvas = np.full((args.height, W, 3), 255, np.uint8)
        for method, x0 in (("agent", margin), ("baseline", margin + panel_w + gutter)):
            im = cv2.imread(str(frames_dir / method / f"f{f:05d}.jpg"))
            if im is None:
                continue
            canvas[header:header + panel_h, x0:x0 + panel_w] = cv2.resize(im, (panel_w, panel_h))

        t_agent = st["agent"]["t"]
        t_base = st["baseline"]["t"]
        canvas[args.height - margin - strip_h:args.height - margin] = draw_marker(
            strip, maps, stats, t_agent, t_base, colour_bgr, base_bgr)

        # Darken the label plates on the array: an RGB PIL image takes no alpha in `fill`.
        for method, x0 in (("agent", margin), ("baseline", margin + panel_w + gutter)):
            held = st.get(f"{method}_held", False) and st["tag"] != "final"
            y0, y1 = header + 12, header + 12 + (142 if held else 112)
            pw = min(plate_w[method], panel_w - 24)
            patch = canvas[y0:y1, x0 + 12:x0 + 12 + pw]
            canvas[y0:y1, x0 + 12:x0 + 12 + pw] = (patch * 0.32).astype(np.uint8)

        pil = Image.fromarray(canvas[:, :, ::-1])
        d = ImageDraw.Draw(pil)
        d.text((margin, 24), title, font=font_t, fill=(20, 20, 20))
        same = st["tag"] != "final"
        note = (f"same wall-clock for both:  {st['trigger_s']:.0f} s"
                if same else "end of each run")
        d.text((W - margin - d.textlength(note, font=font_s), 34), note, font=font_s, fill=(90, 90, 90))

        for method, x0, name, col in (("agent", margin, "PACE", BACKEND_COLOUR[backend]),
                                      ("baseline", margin + panel_w + gutter, "fixed schedule",
                                       "#E8E8E8")):
            r = st[method]
            held = st.get(f"{method}_held", False) and st["tag"] != "final"
            d.text((x0 + 22, header + 20), f"{name} - {BACKEND_LABEL[backend]}", font=font_b,
                   fill=hex_rgb_light(col) if method == "agent" else (236, 236, 236))
            d.text((x0 + 22, header + 52), f"{r['t']:.0f} s  -  iter {r['iter']:,}",
                   font=font_m, fill=(235, 235, 235))
            d.text((x0 + 22, header + 84),
                   f"{r['N'] / 1e6:.2f}M Gaussians  -  {r['psnr']:.2f} dB",
                   font=font_m, fill=(235, 235, 235))
            if held:
                d.text((x0 + 22, header + 116), "run already ended", font=font_s, fill=(255, 190, 120))
        frame = np.asarray(pil)[:, :, ::-1].copy()
        writer.write(frame)
        if args.poster and f == args.poster_frame:
            cv2.imwrite(str(Path(args.poster)), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if f % 100 == 0:
            print(f"  composed {f}/{meta['frames']}", flush=True)

    writer.close()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


class _FFmpegWriter:
    def __init__(self, proc):
        self.proc = proc

    def write(self, frame):
        self.proc.stdin.write(frame.tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


class _CV2Writer:
    def __init__(self, vw):
        self.vw = vw

    def write(self, frame):
        self.vw.write(frame)

    def close(self):
        self.vw.release()


def _open_writer(out: Path, w: int, h: int, fps: int, crf: int):
    import subprocess
    import cv2
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = None
    if exe:
        cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
               "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
        return _FFmpegWriter(subprocess.Popen(cmd, stdin=subprocess.PIPE))
    print("[warn] no ffmpeg: falling back to mp4v, which browsers cannot play", file=sys.stderr)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        print("[error] could not open the video writer", file=sys.stderr)
        return None
    return _CV2Writer(vw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="GPU: render the two flights (run in the backend's env)")
    r.add_argument("--scene", required=True)
    r.add_argument("--backend", required=True, choices=["3dgs", "fastergs", "dash"])
    r.add_argument("--frames-dir", required=True)
    r.add_argument("--frames", type=int, default=1080)
    r.add_argument("--width", type=int, default=944, help="rendered panel width in px")
    r.add_argument("--smooth", type=int, default=9, help="moving-average window over capture poses")
    r.add_argument("--span", type=float, default=0.9, help="fraction of the capture to fly")
    r.set_defaults(fn=cmd_render)

    c = sub.add_parser("compose", help="CPU: curve strip + mp4 (run in env_pytorch_3dgs)")
    c.add_argument("--frames-dir", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--fps", type=int, default=30)
    c.add_argument("--width", type=int, default=1920)
    c.add_argument("--height", type=int, default=1080)
    c.add_argument("--crf", type=int, default=23, help="libx264 quality; lower is bigger and better")
    c.add_argument("--poster", default=None, help="write this frame as a JPEG poster for <video>")
    c.add_argument("--poster-frame", type=int, default=250)
    c.set_defaults(fn=cmd_compose)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
