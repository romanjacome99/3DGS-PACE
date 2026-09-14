# Agentic 3DGS — interactive demo site

Static, dependency-free web page (plain HTML/CSS/ES modules, WebGL2) that shows how the trained
controllers steer Gaussian-Splatting training: a live 3-D viewer of the saved training states
(agent vs. fixed schedule, same camera, same wall-clock), the fixed-view image strip, per-block
decision logs for every scene × backend, how the Gaussian population evolves along the iterations
(top-view density thumbnails, per-block added/pruned, opacity/size statistics), and the cross-backend transfer matrix.

Nothing here trains anything. All content is derived from existing experiment outputs.

## Layout

```
website/
  index.html            static landing page: summary, teaser, workflow, key results (tables built by tools/build_landing.py)
  demo.html             the interactive demo (viewer, decisions, transfer, ablation)
  css/site.css
  js/main.js            bootstrap (loads data/manifest.json + data/decisions.json)
  js/viewer.js          3-D viewer UI, camera/orbit, HUD, decision strip, image strip
  js/splat-renderer.js  WebGL2 Gaussian-splat renderer (EWA splatting, front-to-back compositing)
  js/sort-worker.js     depth sort in a Web Worker (16-bit counting sort)
  js/agsp.js            decoder of the quantised snapshot format (.agsp)
  js/charts.js          SVG line charts, categorical strips, small multiples
  js/decisions.js       decisions explorer (per-scene overview) + transfer matrix
  js/population.js      Gaussian population evolution section
  tools/build_site_data.py   rebuilds data/ from outputs/ (CPU only, numpy + Pillow)
  tools/build_landing.py     injects the landing-page result tables from data/decisions.json
  figures/              paper figures exported with pdftocairo (SVG)
  data/                 ~460 MB bundle: splats/ (quantised Gaussians), renders/, clouds/, gt/, manifest.json, decisions.json
```

## Run locally

Modules and `fetch` need HTTP (not `file://`):

```
cd website
python -m http.server 8765
# open http://127.0.0.1:8765/
```

## Deploy to GitHub Pages

`.github/workflows/pages.yml` publishes `website/` on every push to `main` that touches it.
One-time setup: repository **Settings → Pages → Build and deployment → Source: GitHub Actions**.
The site then lives at `https://<user>.github.io/<repo>/`.

The data bundle is ~460 MB (mostly splats, each file ≤ 6 MB), well inside GitHub's per-file
(100 MB) and Pages (1 GB) limits, but it does make the repository heavier. If you prefer to keep the
splats out of git, host `website/data/` anywhere that serves files with CORS (a GitHub release,
Hugging Face dataset, S3/R2 bucket…) and set, before `js/main.js` loads:

```html
<script>window.AGS_DATA_BASE = "https://your-host/path/data/";</script>
```

## Rebuild the data bundle

```
python website/tools/build_site_data.py            # cap 300k splats per snapshot
python website/tools/build_site_data.py 500000     # larger cap (bigger files)
python website/tools/build_site_data.py 300000 meta   # only manifest/decisions/renders, keep splats
```

Inputs (all pre-existing):

- `outputs/gaussian_evolution/<run>/` — `ply/snap_*.ply`, `render/*.png`, `snapshots.csv`, `blocks.csv`,
  `meta.json`, `_run/<scene>/episode_0000/cameras.json` (from `scripts/save_gaussian_snapshots.py` and
  `scripts/render_snapshots.py`). Runs used: `{3dgs,fastergs_base,dash}_accel_{agent,baseline}` (train),
  `{ignatius,caterpillar,stump}_{3dgs,fastergs_base,dash}_{agent,baseline}`. The ignatius/caterpillar rollouts were
  produced with the same scripts and the frozen checkpoints (`configs/final_accel_{3dgs,dash}_tandt.json`,
  `agentic_gs_phase1/configs/real_fastergs_accel_aug_base_tandt.json`, 120 s horizon, 7000-iteration cap).
- `outputs/agentic_rl_real/protocol_*/` — `agentic_blocks.csv`, `baseline_blocks.csv`, `curve.csv`,
  `time_to_target.csv`, `summary.json` (from `scripts/eval_protocol.py`). The list is `PROTOCOL` in the build script.
- `archive/real_scenes/…/images*` — ground-truth photos for the preset test cameras.

### .agsp format (little-endian, planar)

```
header 48 B : 'AGSP', u32 version=2, u32 count, u32 true_count, u32 flags (bit0 = subsampled),
              f32 bbox_lo[3], f32 bbox_hi[3], u32 n_core
pos   u16 × 3·n_core      quantised inside bbox        f32 × 3·(count−n_core)  far outliers, raw
scale u8  × 3·count       log-scale, ls = q/255·16 − 12
rot   u8  × 4·count       unit quaternion (w,x,y,z), q = v/255·2 − 1
rgba  u8  × 4·count       SH-DC colour, sigmoid(opacity)
```

Snapshots above the cap keep their most visible splats (opacity × projected area); the HUD always
shows the true Gaussian count. Only the view-independent colour band is kept, so the web render is an
approximation of the CUDA rasteriser; all PSNR numbers come from the reference renderer.
