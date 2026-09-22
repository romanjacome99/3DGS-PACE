"""Gaussian-evolution study: HOW the policy shapes the Gaussian population vs. the
fixed schedule, over a full protocol episode (same overrides as eval_protocol.py).

For one (backend config, method) it drives an episode block-by-block and records, at
every block boundary, population statistics computed directly from the Gaussian
tensors (count, opacity distribution, scale distribution, anisotropy), the block's
compute cost (ms / iteration), the held-out validation quality, and the decoded
action.  At requested wall-clock thresholds (and at the end) it dumps a compact
snapshot of the population (xyz, opacity, scale, base colour) as .npz and measures
multi-view TEST PSNR/SSIM.

Outputs under --out:
  stats.csv        one row per block (see STAT_FIELDS below)
  snapshots.csv    one row per snapshot: tag, trigger_s, iter, time_s, gaussians, psnr, ssim, file
  snap/<tag>.npz   xyz (N,3) f32, opacity (N,) f32, scale (N,3) f32, rgb (N,3) u8
  ply/snap_<tag>.ply   full Gaussian state (adds rotation and the SH bands) -- only with --save-ply
  meta.json

Run in the conda env matching the backend (env_pytorch_3dgs for 3dgs/dash, fgs_cu128 for fastergs).
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(r"C:\Roman\3DGS_PROPOSAL")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from agentic_gs_phase1.envs import AgenticGSEnv  # noqa: E402
from agentic_gs_phase1.envs.spaces import DISCRETE_ACTIONS, default_action  # noqa: E402
from agentic_gs_phase1.policies import ActorCritic  # noqa: E402

ACTION_FIELDS = ["block_steps", "densify_mode", "densification_interval", "prune_mode", "opacity_reset",
                 "densify_threshold_mult", "prune_opacity_threshold", "position_lr_mult", "feature_lr_mult",
                 "opacity_lr_mult", "scaling_lr_mult", "rotation_lr_mult"]
CONT = DISCRETE_ACTIONS["stop"].index("continue")
SH_C0 = 0.28209479177387814


def q(t: torch.Tensor, v: float) -> float:
    if t.numel() == 0:
        return 0.0
    if t.numel() > 2_000_000:  # torch.quantile input-size limit
        idx = torch.randperm(t.numel(), device=t.device)[:2_000_000]
        t = t[idx]
    return float(torch.quantile(t, v).item())


@torch.no_grad()
def population_stats(gaussians, extent: float) -> dict:
    op = gaussians.get_opacity.detach().flatten().float()
    sc = gaussians.get_scaling.detach().float()
    smax = sc.max(dim=1).values
    smin = torch.clamp(sc.min(dim=1).values, min=1e-8)
    smean = sc.mean(dim=1)
    n = op.numel()
    return {
        "N": n,
        "op_mean": float(op.mean()), "op_q10": q(op, .10), "op_q50": q(op, .50), "op_q90": q(op, .90),
        "frac_opaque": float((op > 0.5).float().mean()),          # confident splats
        "frac_transparent": float((op < 0.05).float().mean()),    # near-invisible splats
        "scale_q10": q(smean, .10), "scale_q50": q(smean, .50), "scale_q90": q(smean, .90),
        "scale_max_q50": q(smax, .50),
        "scale_rel_q50": q(smean, .50) / max(extent, 1e-6),      # median scale relative to scene extent
        "aniso_q50": q(smax / smin, .50),
    }


@torch.no_grad()
def save_snapshot_npz(gaussians, path: Path):
    xyz = gaussians.get_xyz.detach().float().cpu().numpy().astype(np.float32)
    op = gaussians.get_opacity.detach().flatten().float().cpu().numpy().astype(np.float32)
    sc = gaussians.get_scaling.detach().float().cpu().numpy().astype(np.float32)
    try:
        dc = gaussians.get_features[:, 0, :].detach().float()      # (N,3) SH DC
        rgb = (SH_C0 * dc + 0.5).clamp(0, 1).cpu().numpy()
    except Exception:  # noqa: BLE001 - backend without SH features
        rgb = np.full((xyz.shape[0], 3), 0.5, dtype=np.float32)
    np.savez_compressed(path, xyz=xyz, opacity=op, scale=sc, rgb=(rgb * 255).astype(np.uint8))


@torch.no_grad()
def save_png(t: torch.Tensor, path: Path):
    arr = (t.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True, choices=["agentic", "baseline"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--max-iter", type=int, default=30000)
    ap.add_argument("--snap-times", type=float, nargs="+", default=[30, 60, 120, 300, 600],
                    help="wall-clock thresholds (s) at which to dump a compact snapshot + test metrics")
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--render-views", type=int, nargs="*", default=[],
                    help="test-camera indices to render (full resolution) at every snapshot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-ply", action="store_true",
                    help="also dump the FULL Gaussian state (incl. rotation and SH) as ply/snap_<tag>.ply; "
                         "needed by the website viewer, which cannot splat the reduced .npz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = torch.device("cuda")
    cfg = json.loads(Path(args.config).read_text())
    # identical overrides to scripts/eval_protocol.py (force-full protocol)
    cfg["max_episode_iterations"] = args.max_iter
    cfg["fastergs_acknowledge_grad_bug"] = True
    cfg["save_episode_models"] = False
    cfg.setdefault("safety", {})["min_iterations_before_stop"] = args.max_iter
    cfg.setdefault("optimization", {})["position_lr_max_steps"] = args.max_iter

    out = Path(args.out)
    (out / "snap").mkdir(parents=True, exist_ok=True)
    if args.render_views:
        (out / "render").mkdir(parents=True, exist_ok=True)

    policy = None
    if args.method == "agentic":
        if not args.checkpoint:
            raise SystemExit("--checkpoint required for --method agentic")
        ckpt = torch.load(args.checkpoint, map_location=dev)
        p = cfg.get("policy", {})
        policy = ActorCritic(int(ckpt.get("obs_dim", len(AgenticGSEnv.observation_names))),
                             int(p.get("hidden_width", 256)), int(p.get("hidden_layers", 3)),
                             str(p.get("activation", "gelu")), config=cfg).to(dev)
        policy.load_state_dict(ckpt["policy_state_dict"])
        policy.eval()

    env = AgenticGSEnv(cfg, run_dir=out / "_run", seed=args.seed)
    obs = env.reset(args.scene, episode_id=0)
    extent = float(env.scene.cameras_extent)

    render_cams = []
    if args.render_views:
        all_cams = list(env.scene.getTestCameras())
        for k in args.render_views:
            render_cams.append((k, all_cams[k % len(all_cams)]))
        for k, c in render_cams:
            save_png(c.original_image.cuda(), out / "render" / f"view{k}_gt.png")

    @torch.no_grad()
    def render_snapshot(tag):
        for k, c in render_cams:
            im = env.backend.render_image(c, env.gaussians, env.pipe, env.background,
                                          use_trained_exp=env.dataset.train_test_exp)
            save_png(im, out / "render" / f"view{k}_{tag}.png")

    def test_metrics():
        cams = list(env.scene.getTestCameras())
        cams = cams[::max(1, len(cams) // args.views)][:args.views]
        ps, ss = [], []
        with torch.no_grad():
            for c in cams:
                im = env.backend.render_image(c, env.gaussians, env.pipe, env.background,
                                              use_trained_exp=env.dataset.train_test_exp).clamp(0, 1)
                gt = c.original_image.cuda().clamp(0, 1)
                ps.append(float(env.backend.psnr(im.unsqueeze(0), gt.unsqueeze(0)).mean()))
                ss.append(float(env.backend.ssim(im, gt)))
        return sum(ps) / len(ps), sum(ss) / len(ss)

    snaps, rows, saved = [], [], set()

    def snapshot(tag, trigger_s):
        f = out / "snap" / f"{tag}.npz"
        save_snapshot_npz(env.gaussians, f)
        if args.save_ply:
            # written outside the block timer, so it cannot perturb the wall-clock snapshot triggers
            (out / "ply").mkdir(parents=True, exist_ok=True)
            env.gaussians.save_ply(str(out / "ply" / f"snap_{tag}.ply"))
        render_snapshot(tag)
        p_, s_ = test_metrics()
        st = population_stats(env.gaussians, extent)
        rec = {"tag": tag, "trigger_s": trigger_s, "iter": env.iteration, "time_s": round(env.training_seconds, 2),
               "gaussians": st["N"], "psnr": round(p_, 3), "ssim": round(s_, 4),
               "op_mean": round(st["op_mean"], 4), "frac_opaque": round(st["frac_opaque"], 4),
               "scale_q50": round(st["scale_q50"], 5), "file": f.name}
        if args.save_ply:
            # column name expected by scripts/render_snapshots.py and scripts/make_evolution_video.py
            rec["ply"] = f"snap_{tag}.ply"
        snaps.append(rec)
        print(f"[snap {tag:>7}] it={rec['iter']:6d} t={rec['time_s']:7.1f}s N={st['N']:8d} "
              f"psnr={p_:5.2f} op={st['op_mean']:.3f} opaque={st['frac_opaque']:.2f} "
              f"scale50={st['scale_q50']:.4f}", flush=True)

    # initial population (before any optimisation)
    st0 = population_stats(env.gaussians, extent)
    rows.append({"block": 0, "iter": 0, "t": 0.0, "ms_per_iter": 0.0, "added": 0, "pruned": 0,
                 "block_seconds": 0.0, "val_psnr": 0.0, "val_ssim": 0.0, "densify_events": 0,
                 **st0, **{f: None for f in ACTION_FIELDS}})
    snapshot("init", 0.0)

    done, aborted = False, False
    snap_times = sorted(set(float(t) for t in args.snap_times))
    while not done and env.iteration < args.max_iter:
        if args.method == "agentic":
            a, _, _ = policy.act(torch.as_tensor(obs, dtype=torch.float32, device=dev), deterministic=True)
            a["discrete"]["stop"] = CONT   # force-full, as in eval_protocol.py
        else:
            a = default_action()
        try:
            obs, reward, done, info = env.step(a)
        except torch.cuda.OutOfMemoryError as e:   # large models can exhaust the GPU mid-episode
            print(f"[abort] CUDA OOM at iteration {env.iteration}: {e}", flush=True)
            aborted = True
            break
        bs, act, val = info["block_stats"], info["action"], info["validation"]
        it_ran = max(1, int(bs.get("iterations_ran", act.get("block_steps", 1) or 1)))
        st = population_stats(env.gaussians, extent)
        row = {"block": info["block_index"], "iter": info["iteration"], "t": round(env.training_seconds, 3),
               "ms_per_iter": round(1000.0 * float(bs.get("block_seconds", 0.0)) / it_ran, 3),
               "added": int(bs.get("gaussians_added", 0)), "pruned": int(bs.get("gaussians_pruned", 0)),
               "block_seconds": round(float(bs.get("block_seconds", 0.0)), 4),
               "val_psnr": round(float(val.get("psnr", 0.0)), 3), "val_ssim": round(float(val.get("ssim", 0.0)), 4),
               "densify_events": int(bs.get("densify_events", 0)), **st}
        for f in ACTION_FIELDS:
            v = act.get(f)
            row[f] = round(v, 4) if isinstance(v, float) else v
        rows.append(row)
        if info["block_index"] % 20 == 0:
            print(f"[{args.method}] blk={row['block']:4d} it={row['iter']:6d} t={row['t']:7.1f}s N={st['N']:8d} "
                  f"val={row['val_psnr']:5.2f} op={st['op_mean']:.3f} opaque={st['frac_opaque']:.2f} "
                  f"scale50={st['scale_q50']:.4f} ms/it={row['ms_per_iter']:.1f} dens={row['densify_mode']}",
                  flush=True)
        for t in snap_times:
            if t not in saved and env.training_seconds >= t:
                saved.add(t)
                snapshot(f"t{int(t)}s", t)

    if not aborted:
        snapshot("final", round(env.training_seconds, 2))
    env.close()

    with open(out / "stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[-1].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(out / "snapshots.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(snaps[0].keys()))
        w.writeheader()
        for r in snaps:
            w.writerow(r)
    (out / "meta.json").write_text(json.dumps({
        "method": args.method, "scene": args.scene, "backend": cfg.get("trainer_backend", "3dgs"),
        "config": args.config, "checkpoint": args.checkpoint, "max_iter": args.max_iter,
        "snap_times": snap_times, "seed": args.seed, "cameras_extent": extent,
        "render_views": args.render_views, "aborted_oom": aborted,
        "last_iteration": int(env.iteration)}, indent=1))
    print(f"[done] {len(rows)} blocks, {len(snaps)} snapshots -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
