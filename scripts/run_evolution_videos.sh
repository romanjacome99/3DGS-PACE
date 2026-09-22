#!/usr/bin/env bash
# Build the PACE-vs-fixed-schedule flight videos for every (scene, backend) pair.
#
# Inference only: scripts/make_evolution_video.py replays the .ply states saved under
# outputs/gaussian_evolution_web/ (written by scripts/run_web_snapshots.sh) and the stats.csv
# beside them. Nothing is trained and no protocol number is recomputed.
#
# Each pair runs in two stages because the Faster-GS env has no matplotlib: the GPU render runs in
# the backend's own env, the curve strip and the mp4 are built in env_pytorch_3dgs.
#
# Usage:  bash scripts/run_evolution_videos.sh [scene ...]
#         BACKENDS="3dgs dash" bash scripts/run_evolution_videos.sh train
#         KEEP_FRAMES=1 ...    keep the intermediate JPEGs (~250 MB per video)
set -u

ROOT=/c/Roman/3DGS_PROPOSAL
cd "$ROOT" || exit 1

PY_3DGS=/c/Users/User/anaconda3/envs/env_pytorch_3dgs/python.exe
PY_FGS=/c/Users/User/anaconda3/envs/fgs_cu128/python.exe

OUT=outputs/videos
FRAMES_ROOT=$OUT/_frames
FRAMES=${FRAMES:-1080}          # 36 s at 30 fps
FPS=${FPS:-30}
WIDTH=${WIDTH:-944}             # rendered panel width; the canvas is 1920x1080
mkdir -p "$OUT" "$FRAMES_ROOT"
LOG=$OUT/videos.log

py_for() { if [ "$1" = fastergs ]; then echo "$PY_FGS"; else echo "$PY_3DGS"; fi; }

# Same reasoning as scripts/run_web_snapshots.sh: a co-tenant on the GPU would stretch the render
# and, more to the point, the other session's training gates reset when we spike the card.
gate() {
  "$PY_3DGS" scripts/wait_gpu_quiet.py --median 20 --max 60 --window 12 --interval 5 \
    --max-wait-min 120 >>"$LOG" 2>&1
}

SCENES=${*:-"train ignatius caterpillar"}
BACKENDS=${BACKENDS:-"3dgs fastergs dash"}
echo "=== VIDEOS START $(date '+%F %T') scenes: $SCENES backends: $BACKENDS ===" | tee -a "$LOG"

for scene in $SCENES; do
  for be in $BACKENDS; do
    mp4="$OUT/${scene}_${be}.mp4"
    fdir="$FRAMES_ROOT/${scene}_${be}"
    if [ -f "$mp4" ]; then
      echo "=== SKIP $scene $be (already built) ===" | tee -a "$LOG"
      continue
    fi
    for m in agent baseline; do
      if [ ! -f "outputs/gaussian_evolution_web/${scene}_${be}_${m}/snapshots.csv" ]; then
        echo "=== MISSING $scene $be $m snapshots ===" | tee -a "$LOG"
        continue 2
      fi
    done

    echo "=== RENDER $scene $be $(date '+%F %T') ===" | tee -a "$LOG"
    if ! gate; then
      echo "=== GATETIMEOUT $scene $be (GPU busy; re-run to pick it up) ===" | tee -a "$LOG"
      continue
    fi
    if ! "$(py_for "$be")" -u scripts/make_evolution_video.py render \
        --scene "$scene" --backend "$be" --frames-dir "$fdir" \
        --frames "$FRAMES" --width "$WIDTH" >>"$LOG" 2>&1; then
      echo "=== RFAILED $scene $be ===" | tee -a "$LOG"
      continue
    fi

    echo "=== COMPOSE $scene $be $(date '+%F %T') ===" | tee -a "$LOG"
    if ! "$PY_3DGS" -u scripts/make_evolution_video.py compose \
        --frames-dir "$fdir" --out "$mp4" --fps "$FPS" >>"$LOG" 2>&1; then
      echo "=== CFAILED $scene $be ===" | tee -a "$LOG"
      continue
    fi
    [ "${KEEP_FRAMES:-0}" = "1" ] || rm -rf "$fdir"
    echo "=== OK $scene $be $(date '+%F %T') -> $mp4 ===" | tee -a "$LOG"
  done
done
echo "=== VIDEOS DONE $(date '+%F %T') ===" | tee -a "$LOG"
ls -l "$OUT"/*.mp4 2>/dev/null | tee -a "$LOG"
