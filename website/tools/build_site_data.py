"""Build the static data bundle for the Agentic3DGS interactive website.

CPU-only. Reads EXISTING experiment artifacts (no training, no GPU):
  outputs/gaussian_evolution/<run>/{ply,render,snapshots.csv,blocks.csv,meta.json,_run/*/episode_0000/cameras.json}
  outputs/agentic_rl_real/protocol_*/{agentic_blocks.csv,baseline_blocks.csv,curve.csv,time_to_target.csv,summary.json}
  archive/real_scenes/... (ground-truth photos for the preset test views)

Writes website/data/:
  manifest.json                                  everything the viewer needs (scenes, runs, snapshots, cameras, blocks)
  splats/<scene>/init.agsp                       shared initial point cloud
  splats/<scene>/<backend>/<method>/<tag>.agsp   quantized Gaussian snapshots (see encode_splats)
  renders/<scene>/<backend>/<method>/<tag>.jpg   fixed-test-view renders (from render_snapshots.py output)
  clouds/<scene>/<backend>/<method>/<tag>.png    top-view density thumbnails of the Gaussian centres (opacity-weighted)
  gt/<scene>/<idx>.jpg                           ground-truth photos of the preset test cameras
  decisions.json                                 per-block action logs + curves + time-to-target of the protocol runs

AGSP binary format (little-endian), planar:
  header 48 B (UNCOMPRESSED): magic 'AGSP', u32 version=3, u32 count, u32 true_count,
               u32 flags(bit0=subsampled), f32 bbox_lo[3], f32 bbox_hi[3], u32 n_core
  ---- everything below is one gzip stream, inflated in the browser with DecompressionStream ----
  pos   u16 x 3*n_core            core positions, quantized in bbox, MORTON ordered and stored as
                                  deltas mod 2^16 along that curve (element 0 is absolute)
        f32 x 3*(count-n_core)    far/outlier positions (raw, unordered)
  scale u8  x 3*count   log-scale quantized, ls = q/255*16 - 12
  rot   u8  x 4*count   unit quaternion (w,x,y,z), q = v/255*2 - 1
  rgba  u8  x 4*count   SH-DC colour + sigmoid(opacity)

The Morton ordering is what makes this worth doing: in training order consecutive records are
spatially unrelated and the stream is nearly incompressible (gzip reached only 90% of raw). Sorted
along a Z-curve, neighbouring splats sit close together, their position deltas are small, and their
scales, rotations and colours are locally correlated too.

Usage:  python website/tools/build_site_data.py [cap_per_snapshot=300000] [all|splats|meta]
"""
from __future__ import annotations
import csv, gzip, json, math, struct, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(r"C:\Roman\3DGS_PROPOSAL")
EVO = ROOT / "outputs" / "gaussian_evolution"
EVO_WEB = ROOT / "outputs" / "gaussian_evolution_web"
RL = ROOT / "outputs" / "agentic_rl_real"
SITE = ROOT / "website"
DATA = SITE / "data"
CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000    # per-snapshot Gaussian cap; use a
ONLY = sys.argv[2] if len(sys.argv) > 2 else "all"          # huge value for "keep everything"

# Which scenes the site ships. stump is configured below but excluded: its six runs cost ~330 MB
# of the bundle, and dropping them is what lets every remaining model be shipped UNCAPPED inside
# the 1 GB GitHub Pages limit. Add it back here (and rebuild) if the budget ever allows.
SITE_SCENES = ["train", "ignatius", "caterpillar"]

# Instants kept as 3-D snapshots, reduced from the full set (5, 15, 30, 60, 120, 300, 600 s) to
# spend the budget on Gaussians per frame rather than on frame count; the film strip and the
# curves still cover every instant, because those are JPEGs and CSV rows rather than splats.
KEEP_TAGS = {"t15s", "t30s", "t120s", "t300s"}

POLICIES = {
    "3dgs": {"label": "3DGS controller", "checkpoint": "outputs/agentic_rl_real/final_accel_3dgs/checkpoints/selected_accel.pth",
             "note": "PPO, acceleration reward, trained on the real-scene pool, acceleration-aware checkpoint selection."},
    "fastergs": {"label": "Faster-GS controller", "checkpoint": "outputs/agentic_rl_real/real_fastergs_accel_aug_base/checkpoints/best.pth",
                 "note": "PPO, acceleration reward, photometric-augmented real-scene pool, same 45-dim state."},
    "dash": {"label": "DashGaussian controller", "checkpoint": "outputs/agentic_rl_real/final_accel_dash_fixed_horizon/checkpoints/policy_update_0109.pth",
             "note": "PPO, acceleration reward, trained on the real-scene pool with Dash's coarse-to-fine horizon pinned to the 30k evaluation cap."},
}
BACKEND_LABEL = {"3dgs": "3DGS", "fastergs": "Faster-GS", "dash": "DashGaussian", "legs": "LeGS"}


def web_runs(scene: str) -> dict:
    """The six run directories scripts/run_web_snapshots.sh writes for a scene, per backend."""
    return {be: (f"{scene}_{be}_agent", f"{scene}_{be}_baseline") for be in ("3dgs", "fastergs", "dash")}


SCENES = {
    "train": {
        "label": "train", "dataset_label": "Tanks & Temples", "role": "held-out real scene",
        "dataset": ROOT / "archive/real_scenes/rl/train", "images": "images",
        "runs": web_runs("train"),
    },
    "ignatius": {
        "label": "ignatius", "dataset_label": "Tanks & Temples", "role": "held-out real scene",
        "dataset": ROOT / "archive/real_scenes/tandt/ignatius", "images": "images",
        "runs": web_runs("ignatius"),
    },
    "caterpillar": {
        "label": "caterpillar", "dataset_label": "Tanks & Temples", "role": "held-out real scene",
        "dataset": ROOT / "archive/real_scenes/tandt/caterpillar", "images": "images",
        "runs": web_runs("caterpillar"),
    },
    "stump": {
        "label": "stump", "dataset_label": "Mip-NeRF 360", "role": "zero-shot unbounded scene",
        "dataset": ROOT / "archive/real_scenes/mip360/stump", "images": "images_4",
        "runs": web_runs("stump"),
    },
}

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


def run_dir(name: str) -> Path:
    """Snapshot run directory for `name`.

    The site is built from outputs/gaussian_evolution_web: the 30k protocol replays written by
    scripts/run_web_snapshots.sh, which carry the converged models and the same schedule the paper
    reports. The older outputs/gaussian_evolution runs stopped at a 120 s horizon under a
    7000-iteration cap -- a different protocol -- so they are only used if a run has not been
    regenerated yet, and main() skips any scene that is not complete under one root.
    """
    p = EVO_WEB / name
    return p if (p / "snapshots.csv").exists() else EVO / name


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- PLY
def read_ply(path: Path):
    with open(path, "rb") as f:
        props, n = [], 0
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                _, typ, name = line.split()
                assert typ == "float", typ
                props.append(name)
            elif line == "end_header":
                break
        dt = np.dtype([(p, "<f4") for p in props])
        return np.fromfile(f, dtype=dt, count=n)


def population_stats(alpha, ls) -> dict:
    """Distribution summary of a Gaussian population (opacity, scale, anisotropy)."""
    s = np.exp(ls.astype(np.float64))
    gm = np.exp(ls.astype(np.float64).mean(1))               # geometric-mean radius
    aniso = s.max(1) / np.maximum(s.min(1), 1e-9)
    q = lambda a, qs: [round(float(x), 5) for x in np.percentile(a, qs)]
    return {"opacity_q": q(alpha, [10, 50, 90]), "near_transparent_frac": round(float((alpha < 0.01).mean()), 4),
            "opaque_frac": round(float((alpha > 0.9).mean()), 4), "scale_q": q(gm, [10, 50, 90]),
            "aniso_q": q(aniso, [50, 90]), "mean_opacity": round(float(alpha.mean()), 4)}


def cloud_thumbnail(xyz, alpha, basis: dict, out: Path, size=(288, 192)):
    """Top-view density image of the Gaussian centres (weighted by opacity) in a scene-fixed frame.
    Fixed dynamic range per scene (basis['norm']) so panels of one scene are comparable."""
    c = np.asarray(basis["center"], np.float64); U = np.asarray(basis["axes"], np.float64)   # 2 x 3
    P = (xyz.astype(np.float64) - c) @ U.T
    (x0, x1), (y0, y1) = basis["xlim"], basis["ylim"]
    W, H = size
    inside = (P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)
    img, _, _ = np.histogram2d(P[inside, 1], P[inside, 0], bins=[H, W], range=[[y0, y1], [x0, x1]], weights=alpha[inside])
    # 3x3 blur so isolated centres remain visible
    pad = np.pad(img, 1, mode="edge")
    img = sum(pad[i:i + H, j:j + W] for i in range(3) for j in range(3)) / 9.0
    v = np.log1p(img / basis["norm"]) / np.log1p(150.0)
    v = np.clip(v, 0, 1)
    ramp = np.array([[10, 12, 16], [28, 48, 110], [40, 130, 200], [120, 220, 200], [250, 235, 120], [255, 255, 255]], np.float64) / 255
    t = v[..., None] * (len(ramp) - 1)
    i0 = np.clip(np.floor(t).astype(int), 0, len(ramp) - 2); f = t - i0
    rgb = ramp[i0[..., 0]] * (1 - f) + ramp[i0[..., 0] + 1] * f
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.flipud(rgb) * 255).astype(np.uint8)).save(out, "PNG", optimize=True)


def cloud_basis(ply: Path) -> dict:
    """Scene-fixed 2-D frame for the thumbnails: the two principal axes of the (clipped) initial cloud."""
    v = read_ply(ply)
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    lo, hi = np.percentile(xyz, 2, axis=0), np.percentile(xyz, 98, axis=0)
    m = np.all((xyz >= lo) & (xyz <= hi), 1); X = xyz[m]
    c = X.mean(0); _, _, Vt = np.linalg.svd(X - c, full_matrices=False)
    P = (X - c) @ Vt[:2].T
    pad = 0.08
    xl = np.percentile(P[:, 0], [0.5, 99.5]); yl = np.percentile(P[:, 1], [0.5, 99.5])
    xl = [float(xl[0] - pad * (xl[1] - xl[0])), float(xl[1] + pad * (xl[1] - xl[0]))]
    yl = [float(yl[0] - pad * (yl[1] - yl[0])), float(yl[1] + pad * (yl[1] - yl[0]))]
    W, H = 288, 192
    img, _, _ = np.histogram2d(P[:, 1], P[:, 0], bins=[H, W], range=[yl, xl])
    padd = np.pad(img, 1, mode="edge")
    img = sum(padd[i:i + H, j:j + W] for i in range(3) for j in range(3)) / 9.0
    nz = img[img > 0]
    ref = float(np.percentile(nz, 60)) * 0.1 if nz.size else 1.0     # init opacities are 0.1
    return {"center": c.tolist(), "axes": Vt[:2].tolist(), "xlim": xl, "ylim": yl, "norm": max(ref, 1e-4)}


def morton_codes(p: np.ndarray) -> np.ndarray:
    """Z-order key from the top 10 bits of each quantized axis (1024^3 cells is ample here)."""
    q = (p >> 6).astype(np.uint64)
    out = np.zeros(len(q), dtype=np.uint64)
    for b in range(10):
        for a in range(3):
            out |= ((q[:, a] >> np.uint64(b)) & np.uint64(1)) << np.uint64(3 * b + a)
    return out


def encode_splats(ply: Path, out: Path, cap: int, true_count: int, basis: dict | None = None, thumb: Path | None = None) -> dict:
    v = read_ply(ply)
    n0 = len(v)
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    rgb = np.clip(0.5 + 0.28209479177387814 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1), 0, 1)
    alpha = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))
    ls = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float32)
    q = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float64)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    q[q[:, 0] < 0] *= -1.0

    finite = np.isfinite(xyz).all(1) & np.isfinite(ls).all(1) & np.isfinite(q).all(1)
    pop = population_stats(alpha[finite], ls[finite])
    if basis is not None and thumb is not None:
        cloud_thumbnail(xyz[finite], alpha[finite], basis, thumb)
    visible = alpha >= (0.5 / 255.0)
    keep = finite & visible
    dropped_transparent = int((finite & ~visible).sum())

    idx = np.nonzero(keep)[0]
    subsampled = False
    if len(idx) > cap:
        s = np.exp(ls[idx])
        s_sorted = np.sort(s, axis=1)
        area = s_sorted[:, 1] * s_sorted[:, 2]          # two largest axes -> projected area
        imp = alpha[idx] * area
        order = np.argsort(-imp, kind="stable")[:cap]
        idx = np.sort(idx[order])
        subsampled = True
    n = len(idx)

    # core box: robust percentile box of the kept splats, expanded; splats outside are stored with f32 positions
    P = xyz[idx]
    p_lo, p_hi = np.percentile(P, 2.0, axis=0), np.percentile(P, 98.0, axis=0)
    half = 0.5 * (p_hi - p_lo) * 1.6 + 1e-3
    mid = 0.5 * (p_hi + p_lo)
    lo, hi = (mid - half).astype(np.float32), (mid + half).astype(np.float32)
    core = np.all((P >= lo) & (P <= hi), 1)
    order = np.concatenate([np.nonzero(core)[0], np.nonzero(~core)[0]])
    idx = idx[order]
    n_core = int(core.sum()); n_far = n - n_core
    P = xyz[idx]

    pos_core = np.clip((P[:n_core] - lo) / np.maximum(hi - lo, 1e-9) * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    if n_core > 1:
        # reorder the core group along a Z-curve, carrying the attribute indices with it
        z = np.argsort(morton_codes(pos_core), kind="stable")
        idx[:n_core] = idx[:n_core][z]
        pos_core = pos_core[z]
    pos_far = xyz[idx[n_core:]].astype("<f4")
    # delta along the curve, wrapping mod 2^16 so a jump between Z-curve cells stays 2 bytes
    d_core = np.empty_like(pos_core)
    if n_core:
        d_core[0] = pos_core[0]
        if n_core > 1:
            d_core[1:] = ((pos_core[1:].astype(np.int32) - pos_core[:-1].astype(np.int32)) & 0xFFFF).astype(np.uint16)
    sc_q = np.clip((np.clip(ls[idx], -12.0, 4.0) + 12.0) / 16.0 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    rot_q = np.clip((q[idx] + 1.0) * 0.5 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    rgba = np.empty((n, 4), np.uint8)
    rgba[:, :3] = np.clip(rgb[idx] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    rgba[:, 3] = np.clip(alpha[idx] * 255.0 + 0.5, 1, 255).astype(np.uint8)

    flags = (1 if subsampled else 0)
    header = struct.pack("<4sIIII3f3fI", b"AGSP", 3, n, int(true_count), flags, *lo.tolist(), *hi.tolist(), n_core)
    assert len(header) == 48
    payload = b"".join((d_core.astype("<u2").tobytes(), pos_far.tobytes(),
                        sc_q.tobytes(), rot_q.tobytes(), rgba.tobytes()))
    raw_bytes = len(payload)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(header)
        f.write(gzip.compress(payload, 6, mtime=0))     # mtime=0 keeps rebuilds byte-identical
    dropped_outside = n_far
    return {"kept": n, "ply_count": n0, "far": dropped_outside, "dropped_transparent": dropped_transparent,
            "subsampled": subsampled, "bytes": out.stat().st_size, "raw_bytes": raw_bytes + 48, "pop": pop}


# ----------------------------------------------------------------------------- helpers
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


def save_jpeg(src: Path, dst: Path, longest: int, quality: int = 86):
    im = Image.open(src).convert("RGB")
    w, h = im.size
    s = min(1.0, longest / max(w, h))
    if s < 1.0:
        im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, "JPEG", quality=quality, optimize=True, progressive=True)
    return im.size


def camera_json(run_dir: Path, scene: str) -> list[dict]:
    p = run_dir / "_run" / scene / "episode_0000" / "cameras.json"
    return json.loads(p.read_text())


def orbit_center(cams: list[dict]) -> tuple[list[float], float]:
    """Point closest (least squares) to all optical axes; radius = mean camera distance to it."""
    A = np.zeros((3, 3)); b = np.zeros(3)
    P = []
    for c in cams:
        R = np.array(c["rotation"], dtype=np.float64)   # camera-to-world rotation
        p = np.array(c["position"], dtype=np.float64)
        d = R[:, 2]; d /= np.linalg.norm(d)              # optical axis (+z of the OpenCV camera) in world
        M = np.eye(3) - np.outer(d, d)
        A += M; b += M @ p
        P.append(p)
    ctr = np.linalg.lstsq(A, b, rcond=None)[0]
    P = np.array(P)
    rad = float(np.mean(np.linalg.norm(P - ctr, axis=1)))
    return ctr.tolist(), rad


def prev_stats(scene: str, be: str | None, method: str | None, tag: str) -> dict:
    """Re-use stats from an existing manifest when only metadata is rebuilt."""
    p = DATA / "manifest.json"
    if not p.exists():
        return {}
    try:
        m = json.loads(p.read_text())["scenes"][scene]
        if be is None:
            return m["init"]["stats"]
        for e in m["backends"][be]["methods"][method]["snapshots"]:
            if e["tag"] == tag:
                return e["stats"]
    except Exception:
        pass
    return {}


# ----------------------------------------------------------------------------- build
def build_scene(scene: str, spec: dict, do_splats: bool) -> dict:
    out = {"label": spec["label"], "dataset": spec["dataset_label"], "role": spec["role"], "backends": {}}
    first_run = run_dir(spec["runs"]["3dgs"][0])
    cams = camera_json(first_run, scene)
    n_test = len([i for i in range(len(cams)) if i % 8 == 0])       # llffhold = 8, test cams are dumped first
    test = cams[:n_test]
    ctr, rad = orbit_center(cams)
    out["orbit"] = {"center": [round(x, 4) for x in ctr], "radius": round(rad, 4)}
    out["camera_positions"] = [[round(x, 3) for x in c["position"]] for c in cams]
    step = max(1, n_test // 12)
    presets = []
    img_dir = spec["dataset"] / spec["images"]
    for j, c in enumerate(test[::step][:12]):
        gt_rel = f"gt/{scene}/{j:02d}.jpg"
        size = None
        src = img_dir / c["img_name"]
        if src.exists():
            size = save_jpeg(src, DATA / gt_rel, 720)
        presets.append({"name": f"test view {j + 1}", "img_name": c["img_name"], "width": c["width"], "height": c["height"],
                        "fx": c["fx"], "fy": c["fy"], "position": c["position"], "rotation": c["rotation"],
                        "gt": gt_rel if size else None, "test_index": test.index(c)})
    out["presets"] = presets
    out["n_test_cameras"] = n_test
    out["n_cameras"] = len(cams)

    init_src = first_run / "ply" / "snap_init.ply"
    init_rel = f"splats/{scene}/init.agsp"
    init_rows = read_csv(first_run / "snapshots.csv")
    init_row = [r for r in init_rows if r["tag"] == "init"][0]
    basis = cloud_basis(init_src)
    out["cloud_frame"] = {"xlim": basis["xlim"], "ylim": basis["ylim"]}
    if do_splats:
        log(f"[{scene}] init {init_src.name}")
        st = encode_splats(init_src, DATA / init_rel, CAP, int(init_row["gaussians"]), basis, DATA / f"clouds/{scene}/init.png")
    else:
        st = prev_stats(scene, None, None, "init")
    out["init"] = {"file": init_rel, "N": int(init_row["gaussians"]), "stats": st, "cloud": f"clouds/{scene}/init.png"}

    for be, (agent_run, base_run) in spec["runs"].items():
        bo = {"label": BACKEND_LABEL[be], "policy": POLICIES[be], "methods": {}}
        for method, run in (("agent", agent_run), ("baseline", base_run)):
            rd = run_dir(run)
            meta = json.loads((rd / "meta.json").read_text())
            snaps = read_csv(rd / "snapshots.csv")
            # the 120 s runs logged blocks.csv; the 30k protocol replays log stats.csv, which
            # carries the same action columns plus per-block validation quality and cost
            bf = rd / "blocks.csv"
            blocks = read_csv(bf if bf.exists() else rd / "stats.csv")
            for b in blocks:
                b.setdefault("gaussians", b.get("N"))
            timed = [r for r in snaps if r["tag"] in KEEP_TAGS]
            final = [r for r in snaps if r["tag"] == "final"]
            if final and (not timed or int(final[0]["iter"]) != int(timed[-1]["iter"])):
                timed.append(final[0])
            entries = []
            gt_png = rd / "render" / "gt.png"
            render_gt_rel = f"renders/{scene}/{be}/{method}/gt.jpg"
            if gt_png.exists():
                save_jpeg(gt_png, DATA / render_gt_rel, 512)
            init_png = rd / "render" / "snap_init.png"
            render_init_rel = f"renders/{scene}/{be}/{method}/init.jpg"
            if init_png.exists():
                save_jpeg(init_png, DATA / render_init_rel, 512)
            for r in timed:
                tag = r["tag"]
                ply = rd / "ply" / r["ply"]
                rel = f"splats/{scene}/{be}/{method}/{tag}.agsp"
                render_rel = None
                png = rd / "render" / (Path(r["ply"]).stem + ".png")
                if png.exists():
                    render_rel = f"renders/{scene}/{be}/{method}/{tag}.jpg"
                    save_jpeg(png, DATA / render_rel, 512)
                cloud_rel = f"clouds/{scene}/{be}/{method}/{tag}.png"
                if do_splats:
                    log(f"[{scene}/{be}/{method}] {tag}  N={r['gaussians']}")
                    st = encode_splats(ply, DATA / rel, CAP, int(r["gaussians"]), basis, DATA / cloud_rel)
                else:
                    st = prev_stats(scene, be, method, tag)
                entries.append({"tag": tag, "trigger_s": float(r["trigger_s"]), "t": float(r["time_s"]), "iter": int(r["iter"]),
                                "N": int(r["gaussians"]), "psnr": float(r["psnr"]), "ssim": float(r["ssim"]),
                                "file": rel, "render": render_rel, "cloud": cloud_rel, "stats": st})
            init_here = [x for x in snaps if x["tag"] == "init"][0]
            bo["methods"][method] = {
                "run": run, "mode": meta.get("mode", "acceleration"), "horizon_s": meta.get("horizon_s"),
                "max_iter": meta.get("max_iter"), "checkpoint": meta.get("checkpoint"),
                "init_psnr": float(init_here["psnr"]), "init_ssim": float(init_here["ssim"]),
                "snapshots": entries,
                "blocks": compact_blocks(blocks, keep_actions=True),
                "render_gt": render_gt_rel if gt_png.exists() else None,
                "render_init": render_init_rel if init_png.exists() else None,
            }
        out["backends"][be] = bo
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
    do_splats = ONLY in ("all", "splats")
    t0 = time.time()
    manifest = {"generated": time.strftime("%Y-%m-%d"), "cap_per_snapshot": CAP, "scenes": {}, "policies": POLICIES,
                "backend_labels": BACKEND_LABEL}
    for scene in SITE_SCENES:
        spec = SCENES[scene]
        runs = [r for pair in spec["runs"].values() for r in pair]
        missing = [r for r in runs if not (run_dir(r) / "snapshots.csv").exists()]
        if missing:
            log(f"[{scene}] skipped, missing runs: {missing}")
            continue
        # A scene must come entirely from one protocol. Several old run directories happen to share
        # the new naming, so a single failed replay would otherwise pair a 30k backend against a
        # 120 s one and the wall-clock instants would not be comparable across the panels.
        roots = {run_dir(r).parent.name for r in runs}
        if len(roots) > 1:
            log(f"[{scene}] SKIPPED, runs span two protocols: {sorted(roots)}")
            continue
        manifest["scenes"][scene] = build_scene(scene, spec, do_splats)
    if ONLY in ("all", "meta"):
        log("[decisions]")
        (DATA / "decisions.json").write_text(json.dumps(build_decisions(), separators=(",", ":")))
    (DATA / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))
    tot = sum(p.stat().st_size for p in DATA.rglob("*") if p.is_file())
    log(f"done in {time.time() - t0:.0f}s; data size {tot / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
