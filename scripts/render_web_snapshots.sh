#!/usr/bin/env bash
# Render the fixed test view from every snapshot PLY produced by scripts/run_web_snapshots.sh.
#
# Fills outputs/gaussian_evolution_web/<run>/render/{gt.png,snap_<tag>.png}, which is what the
# website's film strip shows next to the interactive 3-D panels. Separate from the sweep because it
# only reads saved Gaussian states -- it does no training, so it needs no wall-clock gate and can be
# re-run at will.
#
# Usage:  bash scripts/render_web_snapshots.sh [scene ...]      (default: all four site scenes)
set -u

ROOT=/c/Roman/3DGS_PROPOSAL
cd "$ROOT" || exit 1

PY_3DGS=/c/Users/User/anaconda3/envs/env_pytorch_3dgs/python.exe
PY_FGS=/c/Users/User/anaconda3/envs/fgs_cu128/python.exe
OUTROOT=outputs/gaussian_evolution_web
LOG="$OUTROOT/render.log"

# must stay in step with cfg_for() in scripts/run_web_snapshots.sh
cfg_for() {
  case "$2" in
    3dgs)     case "$1" in
                train) echo configs/final_accel_3dgs.json ;;
                stump) echo configs/final_accel_3dgs_stump.json ;;
                *)     echo configs/final_accel_3dgs_tandt.json ;;
              esac ;;
    fastergs) case "$1" in
                train) echo outputs/agentic_rl_real/real_fastergs_accel_aug_base/resolved_config.json ;;
                stump) echo agentic_gs_phase1/configs/real_fastergs_accel_aug_base_stump.json ;;
                *)     echo agentic_gs_phase1/configs/real_fastergs_accel_aug_base_tandt.json ;;
              esac ;;
    dash)     case "$1" in
                train) echo configs/final_accel_dash_fixed_horizon.json ;;
                stump) echo configs/final_accel_dash_stump.json ;;
                *)     echo configs/final_accel_dash_fixed_tandt.json ;;
              esac ;;
  esac
}
py_for() { if [ "$1" = fastergs ]; then echo "$PY_FGS"; else echo "$PY_3DGS"; fi; }

SCENES=${*:-"train ignatius caterpillar stump"}
echo "=== RENDER START $(date '+%F %T') scenes: $SCENES ===" | tee -a "$LOG"

for scene in $SCENES; do
  for be in 3dgs fastergs dash; do
    dirs=""
    for method in agent baseline; do
      d="$OUTROOT/${scene}_${be}_${method}"
      [ -f "$d/snapshots.csv" ] && dirs="$dirs $d"
    done
    if [ -z "$dirs" ]; then
      echo "=== RSKIP $scene $be (no runs) ===" | tee -a "$LOG"
      continue
    fi
    echo "=== RENDER $scene $be $(date '+%F %T') ===" | tee -a "$LOG"
    # shellcheck disable=SC2086
    "$(py_for "$be")" -u scripts/render_snapshots.py --config "$(cfg_for "$scene" "$be")" \
      --scene "$scene" --dirs $dirs >>"$LOG" 2>&1
    if [ $? -ne 0 ]; then
      echo "=== RFAILED $scene $be ===" | tee -a "$LOG"
    else
      echo "=== ROK $scene $be $(date '+%F %T') ===" | tee -a "$LOG"
    fi
  done
done
echo "=== RENDER DONE $(date '+%F %T') ===" | tee -a "$LOG"
