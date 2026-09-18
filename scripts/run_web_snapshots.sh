#!/usr/bin/env bash
# Regenerate the website's 3-D assets from the SAME protocol the paper reports.
#
# Why this exists: the site's splat snapshots came from scripts/save_gaussian_snapshots.py, which
# leaves max_episode_iterations at the config value (7000) and stops at a 120 s wall-clock horizon.
# The paper's numbers come from the 30k protocol (scripts/eval_protocol.py and
# scripts/gaussian_evolution_study.py both override the cap to --max-iter, which also stretches the
# position-LR decay and DashGaussian's resolution curriculum). Those are different schedules, so the
# viewer was not showing an early frame of the paper's run -- it was showing a different run that
# topped out around 17-20 dB. This sweep replays the protocol episodes at 30k with --save-ply, so the
# viewer can show the converged models and the timeline the paper actually talks about.
#
# Inference only: the trained policies are replayed deterministically, nothing is retrained.
# Writes to outputs/gaussian_evolution_web/ so the study_* dirs backing Figure 1 and appendices
# A.5/A.6 are left untouched.
#
# Usage:  bash scripts/run_web_snapshots.sh [scene ...]      (default: all four site scenes)
set -u

ROOT=/c/Roman/3DGS_PROPOSAL
cd "$ROOT" || exit 1

PY_3DGS=/c/Users/User/anaconda3/envs/env_pytorch_3dgs/python.exe
PY_FGS=/c/Users/User/anaconda3/envs/fgs_cu128/python.exe

OUTROOT=outputs/gaussian_evolution_web
SNAPS="5 15 30 60 120 300 600"
ITERS=30000
mkdir -p "$OUTROOT"
LOG="$OUTROOT/run.log"

CK_3dgs=outputs/agentic_rl_real/final_accel_3dgs/checkpoints/selected_accel.pth
CK_fastergs=outputs/agentic_rl_real/real_fastergs_accel_aug_base/checkpoints/best.pth
# Dash: the fixed-horizon policy (dash.schedule_horizon_iterations=30000), the one Table 1 and the
# cross-backend table report. Its configs pin the same horizon, so the viewer replays the paper's run.
CK_dash=outputs/agentic_rl_real/final_accel_dash_fixed_horizon/checkpoints/policy_update_0109.pth

# config per (scene, backend) -- taken from the summary.json of the corresponding protocol_* run,
# so the website replays exactly what Table 1 and Table 2 were computed from.
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

# Every timed run in this project is GPU-gated; a co-tenant would move the wall-clock snapshot
# triggers to different iteration counts and break comparability with the rest of the site.
#
# wait_gpu_idle.ps1 is not usable here: it needs 20 CONSECUTIVE samples under a threshold, and this
# desktop measures p50 8% / p90 13% but spikes to ~42% on about 2% of samples, so the counter keeps
# resetting -- observed in practice, the gate ran 8 minutes without passing. wait_gpu_quiet.py tests
# the median and max of a trailing window instead, which tolerates spikes while still rejecting
# anything sustained, and it exits non-zero rather than proceeding unguarded at its deadline.
gate() {
  "$PY_3DGS" scripts/wait_gpu_quiet.py --median 20 --max 60 --window 12 --interval 5 \
    --max-wait-min 120 >>"$LOG" 2>&1
}

SCENES=${*:-"train ignatius caterpillar stump"}
echo "=== SWEEP START $(date '+%F %T') scenes: $SCENES ===" | tee -a "$LOG"

for scene in $SCENES; do
  for be in 3dgs fastergs dash; do
    for method in agent baseline; do
      out="$OUTROOT/${scene}_${be}_${method}"
      if [ -f "$out/meta.json" ] && [ -f "$out/ply/snap_final.ply" ]; then
        echo "=== SKIP $scene $be $method (already complete) ===" | tee -a "$LOG"
        continue
      fi
      cfg=$(cfg_for "$scene" "$be")
      py=$(py_for "$be")
      eval "ck=\$CK_$be"
      if [ "$method" = agent ]; then
        ckarg="--method agentic --checkpoint $ck"
      else
        ckarg="--method baseline"
      fi
      echo "=== JOB $scene $be $method $(date '+%F %T') ===" | tee -a "$LOG"
      if ! gate; then
        echo "=== GATETIMEOUT $scene $be $method (GPU busy; re-run the sweep to pick it up) ===" | tee -a "$LOG"
        continue
      fi
      # shellcheck disable=SC2086
      "$py" -u scripts/gaussian_evolution_study.py --config "$cfg" $ckarg --scene "$scene" \
        --max-iter $ITERS --snap-times $SNAPS --save-ply --out "$out" >>"$LOG" 2>&1
      rc=$?
      if [ $rc -ne 0 ]; then
        echo "=== FAILED $scene $be $method rc=$rc ===" | tee -a "$LOG"
      else
        echo "=== OK $scene $be $method $(date '+%F %T') ===" | tee -a "$LOG"
      fi
    done
  done
done
echo "=== SWEEP DONE $(date '+%F %T') ===" | tee -a "$LOG"
