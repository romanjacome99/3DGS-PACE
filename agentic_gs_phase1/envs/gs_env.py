from __future__ import annotations

import copy
import csv
import gc
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from .spaces import (
    OBSERVATION_NAMES,
    action_to_observation_values,
    decode_action,
    default_action,
    fastergs_aa_enabled,
    fastergs_ext_enabled,
    observation_keep_indices,
    observation_names_for,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The underlying Gaussian-Splatting codebase (3DGS official vs Faster-GS fork) is
# selected at env construction via a pluggable backend. The render/loss/model
# symbols are provided by self.backend rather than imported at module load, so a
# run can choose its trainer (3DGS-Agent, FasterGS-Agent, ...) without eagerly
# importing — and conflicting with — the other codebase.
from .backends import load_backend  # noqa: E402
from .backends.dash_schedule import DashScheduler  # noqa: E402

# Backward-compat module-level flag: some eval scripts import this from here.
# It is independent of the active backend (purely whether the sparse-Adam
# rasterizer extension is importable).
try:  # noqa: E402
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401

    SPARSE_ADAM_AVAILABLE = True
except Exception:  # noqa: E402
    SPARSE_ADAM_AVAILABLE = False


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(max(value, 0.0), 1.0))


def _memory_gb() -> dict[str, float]:
    return {
        "allocated": torch.cuda.memory_allocated() / (1024**3),
        "reserved": torch.cuda.memory_reserved() / (1024**3),
        "peak": torch.cuda.max_memory_allocated() / (1024**3),
    }


class AgenticGSEnv:
    """Block-wise RL environment around the stock 3DGS optimizer.

    Rewards and observations use only a held-out subset of training cameras.
    Dataset test cameras are loaded by the upstream scene object but are never
    used for policy state or reward construction in this environment.
    """

    observation_names = OBSERVATION_NAMES

    def __init__(self, config: dict[str, Any], run_dir: str | Path | None = None, seed: int = 0):
        self.config = config
        # Pluggable trainer backend (3DGS-Agent / FasterGS-Agent / ...).
        self.backend = load_backend(config)
        # Budget-conditioning: sample a wall-clock training budget per episode,
        # expose it to the policy, terminate on it, and make the reward's
        # time-value discount budget-relative. When off, behaviour is unchanged.
        bcfg = config.get("budget", {})
        self.budget_conditioned = bool(config.get("budget_conditioned", False))
        self.budget_min_seconds = float(bcfg.get("min_seconds", 20.0))
        self.budget_max_seconds = float(bcfg.get("max_seconds", 300.0))
        self.budget_tau_fraction = float(bcfg.get("tau_fraction", 0.6))
        self.time_budget = None
        # If set, reset() uses this budget instead of sampling (used by the probe
        # for a consistent, reachable accel-checkpoint evaluation).
        self.budget_override = None
        self.observation_names = observation_names_for(config)
        # State-vector ablation (opt-in via config["ablation"]["dropped_observations"]):
        # the env still computes every channel, then keeps only these indices.
        self._full_observation_names = observation_names_for(config, include_dropped=True)
        self._obs_keep_indices = observation_keep_indices(config)
        # FasterGS-specific policy extension (opt-in). When on, the env applies the
        # extra actions (antialiasing / SH-unlock gate) and emits the extra
        # observations (per-stage GPU-time fractions + coverage stats). Off => base.
        self.fastergs_ext = fastergs_ext_enabled(config)
        self.fastergs_aa = fastergs_aa_enabled(config)   # antialiasing action/obs present?
        self._fgs_ev: list = []            # per-iteration CUDA event tuples for stage timing
        self._fgs_last_radii = None        # last iteration's per-Gaussian visibility mask (di[0])
        self._fgs_last_grad = None         # last iteration's densification gradient (di[1])
        # DashGaussian schedule (opt-in): auto-on for trainer_backend="dash", or force
        # via a dash.enabled flag (e.g. to run the schedule on another backend). When on,
        # the env renders training views at the scheduled resolution and caps densification
        # by the momentum primitive budget; the policy still controls everything else.
        dash_cfg = dict(config.get("dash", {}))
        self.dash_enabled = bool(dash_cfg.get("enabled", getattr(self.backend, "is_dash", False)))
        self.dash_cfg = dash_cfg
        self._dash = None
        # Iteration horizon the Dash curriculum is built against; set in reset(). None until then.
        self.dash_schedule_horizon = None
        self._current_res_scale = 1.0 / max(1.0, float(config.get("resolution", 1)))
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)
        self.archive_root = _resolve_path(config.get("archive_root", "archive/nerf_synthetic"))
        self.output_root = _resolve_path(config.get("output_root", "outputs/agentic_gs_phase1"))
        self.run_dir = _resolve_path(run_dir) if run_dir is not None else self.output_root / time.strftime("%Y%m%d_%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.scene_id = ""
        self.model_path: Path | None = None
        self.scene: Any = None
        self.gaussians: Any = None
        self.dataset = None
        self.opt = None
        self.pipe = None
        self.background = None
        self.train_cameras = []
        self.validation_cameras = []
        self.train_stack = []

        self.iteration = 0
        self.block_index = 0
        self.episode_start_time = 0.0
        self.elapsed_seconds = 0.0
        self.training_seconds = 0.0
        self.last_validation = {"psnr": 0.0, "ssim": 0.0, "l1": 1.0, "quality": 0.0}
        self.initial_validation_quality = 0.0
        self.prev_validation_quality = 0.0
        self.last_quality_gain = 0.0
        self.prev_gaussian_count = 0
        self.prev_action = decode_action(default_action(self.config), self.config)
        self.prev_action_improved = False
        self.last_quality_gain = 0.0
        self.opacity_resets = 0
        self.last_opacity_reset_block = -10_000
        self.last_opacity_reset_iteration = -10_000
        self.last_block_stats = self._empty_block_stats()
        self.loss_history: list[float] = []
        self.block_log_path: Path | None = None
        self.initial_gaussian_count = 0

    @property
    def obs_dim(self) -> int:
        return len(self.observation_names)

    def reset(self, scene_id: str, episode_id: int = 0) -> np.ndarray:
        self.close()
        self.scene_id = scene_id
        scene_path = self.archive_root / scene_id
        if not scene_path.exists():
            raise FileNotFoundError(f"Scene '{scene_id}' was not found under {self.archive_root}")

        self.model_path = self.run_dir / scene_id / f"episode_{episode_id:04d}"
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.block_log_path = self.model_path / "agentic_blocks.csv"
        self._init_block_log()

        self.iteration = 0
        self.block_index = 0
        self.elapsed_seconds = 0.0
        self.training_seconds = 0.0
        # Sample this episode's wall-clock training budget (budget-conditioning).
        if self.budget_conditioned:
            if self.budget_override is not None:
                self.time_budget = float(self.budget_override)
            else:
                self.time_budget = float(self.rng.uniform(self.budget_min_seconds, self.budget_max_seconds))
        else:
            self.time_budget = None
        self.loss_history = []
        self.last_block_stats = self._empty_block_stats()
        self.prev_action = decode_action(default_action(self.config), self.config)
        self.prev_action_improved = False
        self.opacity_resets = 0
        self.last_opacity_reset_block = -10_000
        self.last_opacity_reset_iteration = -10_000

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        self.dataset = self._dataset_args(scene_path, self.model_path)
        self.opt = self._optimization_args()
        self.pipe = self._pipeline_args()
        if getattr(self.backend, "is_legs", False):
            # LeGS: inject its argument defaults onto opt and bind it so the
            # model can build its inner per-Gaussian RL controller.
            self.backend.attach_opt(self.opt, self.config)
        self.gaussians = self.backend.make_gaussians(self.dataset.sh_degree, self.opt.optimizer_type)
        self.scene = self.backend.make_scene(self.dataset, self.gaussians)
        self.gaussians.training_setup(self.opt)

        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self._split_train_validation()

        self.prev_gaussian_count = self.gaussians.get_xyz.shape[0]
        self.initial_gaussian_count = int(self.prev_gaussian_count)
        # Build the DashGaussian schedule for this episode from the training views.
        self._dash = None
        self._current_res_scale = 1.0 / max(1.0, float(self.config.get("resolution", 1)))
        if self.dash_enabled:
            dcfg = self.dash_cfg
            # DashGaussian's resolution curriculum and its resolution-coupled primitive budget
            # are a property of the TRAINING RUN being scheduled, not of how long one RL episode
            # happens to be. Keying them to self.opt.iterations (= max_episode_iterations) makes
            # the substrate itself change shape between training and evaluation: with a 7k episode
            # cap the resolution ramp completes at 6.3k, so a 3k training episode ends at 48% of
            # the ramp, while at the 30k evaluation cap the ramp completes at 27k and iteration 3k
            # is only 11% of the way up. The policy is then trained on one curriculum and evaluated
            # on another, and since the budget's allowed growth scales with scale^2, densification
            # is suppressed to a different degree in each. Set dash.schedule_horizon_iterations to
            # the deployment horizon (e.g. 30000) to pin the curriculum to a fixed horizon in both.
            # Absent/0 keeps the historical behaviour, so existing configs and every published Dash
            # result are unchanged.
            dash_horizon = int(dcfg.get("schedule_horizon_iterations", 0) or 0)
            if dash_horizon <= 0:
                dash_horizon = int(self.opt.iterations)
            self.dash_schedule_horizon = dash_horizon
            self._dash = DashScheduler(
                [c.original_image for c in self.train_cameras],
                max_steps=dash_horizon,
                densify_until_iter=int(self.opt.densify_until_iter),
                mode=str(dcfg.get("mode", "freq")),
                max_reso_scale=float(dcfg.get("max_reso_scale", 8.0)),
                reso_sample_num=int(dcfg.get("reso_sample_num", 32)),
                start_significance_factor=float(dcfg.get("start_significance_factor", 4.0)),
                res_full_fraction=float(dcfg.get("res_full_fraction", 0.9)),
                max_densify_rate_per_step=float(dcfg.get("max_densify_rate_per_step", 0.2)),
                momentum_decay=float(dcfg.get("momentum_decay", 0.98)),
                initial_count=self.initial_gaussian_count,
            )
        self.last_validation = self._evaluate_validation_subset()
        self.initial_validation_quality = self.last_validation["quality"]
        self.prev_validation_quality = self.last_validation["quality"]
        self.episode_start_time = time.perf_counter()

        self._write_episode_metadata()
        return self._build_observation()

    def step(self, action: dict[str, Any]) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self.scene is None or self.gaussians is None:
            raise RuntimeError("Call reset(scene_id) before step(action).")

        controls = decode_action(action, self.config)
        self._maybe_reset_opacity(controls)
        block_stats = self._empty_block_stats()
        block_stats["start_iteration"] = self.iteration + 1
        block_stats["start_gaussians"] = int(self.gaussians.get_xyz.shape[0])
        if self.fastergs_ext:
            self._fgs_ev = []
            self._fgs_last_radii = None
            self._fgs_last_grad = None
        block_start = time.perf_counter()

        for _ in range(controls.block_steps):
            if self.iteration >= int(self.opt.iterations):
                break
            self.iteration += 1
            iter_stats = self._run_training_iteration(controls)
            for key in ["l1", "dssim", "total_loss", "visible_gaussians"]:
                block_stats[key] += iter_stats[key]
            block_stats["iterations_ran"] += 1
            block_stats["last_visible_fraction"] = iter_stats["visible_fraction"]
            block_stats["densify_events"] += iter_stats["densify_event"]
            block_stats["gaussians_added"] += iter_stats["gaussians_added"]
            block_stats["gaussians_pruned"] += iter_stats["gaussians_pruned"]

            if not np.isfinite(iter_stats["total_loss"]):
                block_stats["numerical_failure"] = 1
                break

            if self._hard_budget_exceeded():
                block_stats["hard_budget_exceeded"] = 1
                break

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        block_stats["block_seconds"] = time.perf_counter() - block_start
        self.elapsed_seconds = time.perf_counter() - self.episode_start_time
        self.training_seconds += block_stats["block_seconds"]
        if self.fastergs_ext:
            self._finalize_fastergs_telemetry(block_stats, controls)

        iterations_ran = max(1, int(block_stats["iterations_ran"]))
        block_stats["l1"] /= iterations_ran
        block_stats["dssim"] /= iterations_ran
        block_stats["total_loss"] /= iterations_ran
        block_stats["visible_gaussians"] /= iterations_ran
        block_stats["time_per_iteration"] = block_stats["block_seconds"] / iterations_ran
        block_stats["end_gaussians"] = int(self.gaussians.get_xyz.shape[0])
        block_stats["gaussian_growth"] = block_stats["end_gaussians"] - block_stats["start_gaussians"]

        validation = self._evaluate_validation_subset()
        reward, reward_terms = self._compute_reward(validation, block_stats)
        quality_gain = validation["quality"] - self.prev_validation_quality
        self.last_quality_gain = quality_gain
        self.prev_action_improved = quality_gain > 0.0
        self.prev_validation_quality = validation["quality"]
        self.last_validation = validation
        self.prev_action = controls
        self.last_block_stats = block_stats
        self.loss_history.append(block_stats["total_loss"])

        self.block_index += 1
        done = self._is_done(controls, block_stats)
        if done and block_stats["numerical_failure"]:
            reward -= float(self.config.get("reward", {}).get("failure_penalty", 2.0))
            reward_terms["failure_penalty"] = -float(self.config.get("reward", {}).get("failure_penalty", 2.0))

        info = {
            "scene": self.scene_id,
            "iteration": self.iteration,
            "block_index": self.block_index,
            "training_seconds": self.training_seconds,
            "time_budget": self.time_budget,
            "action": controls.as_dict(),
            "validation": validation,
            "reward_terms": reward_terms,
            "block_stats": block_stats,
            "test_camera_accessed": False,
        }
        self._append_block_log(info, reward)
        return self._build_observation(), float(reward), done, info

    def close(self) -> None:
        self.scene = None
        self.gaussians = None
        self.dataset = None
        self.opt = None
        self.pipe = None
        self.background = None
        self.train_cameras = []
        self.validation_cameras = []
        self.train_stack = []
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def save_model(self) -> None:
        if self.scene is not None:
            self.scene.save(self.iteration)

    def _dataset_args(self, scene_path: Path, model_path: Path) -> SimpleNamespace:
        return SimpleNamespace(
            sh_degree=int(self.config.get("sh_degree", 3)),
            source_path=str(scene_path),
            model_path=str(model_path),
            images=str(self.config.get("images", "images")),
            depths="",
            resolution=int(self.config.get("resolution", 1)),
            white_background=bool(self.config.get("white_background", True)),
            data_device=str(self.config.get("data_device", "cuda")),
            eval=True,
            train_test_exp=False,
        )

    def _optimization_args(self) -> SimpleNamespace:
        opt_cfg = self.config.get("optimization", {})
        return SimpleNamespace(
            iterations=int(self.config.get("max_episode_iterations", 7000)),
            position_lr_init=float(opt_cfg.get("position_lr_init", 0.00016)),
            position_lr_final=float(opt_cfg.get("position_lr_final", 0.0000016)),
            position_lr_delay_mult=float(opt_cfg.get("position_lr_delay_mult", 0.01)),
            position_lr_max_steps=int(opt_cfg.get("position_lr_max_steps", self.config.get("max_episode_iterations", 7000))),
            feature_lr=float(opt_cfg.get("feature_lr", 0.0025)),
            opacity_lr=float(opt_cfg.get("opacity_lr", 0.025)),
            scaling_lr=float(opt_cfg.get("scaling_lr", 0.005)),
            rotation_lr=float(opt_cfg.get("rotation_lr", 0.001)),
            exposure_lr_init=float(opt_cfg.get("exposure_lr_init", 0.01)),
            exposure_lr_final=float(opt_cfg.get("exposure_lr_final", 0.001)),
            exposure_lr_delay_steps=int(opt_cfg.get("exposure_lr_delay_steps", 0)),
            exposure_lr_delay_mult=float(opt_cfg.get("exposure_lr_delay_mult", 0.0)),
            percent_dense=float(opt_cfg.get("percent_dense", 0.01)),
            lambda_dssim=float(opt_cfg.get("lambda_dssim", 0.2)),
            densification_interval=int(opt_cfg.get("densification_interval", 100)),
            opacity_reset_interval=int(opt_cfg.get("opacity_reset_interval", 3000)),
            densify_from_iter=int(opt_cfg.get("densify_from_iter", 500)),
            densify_until_iter=int(opt_cfg.get("densify_until_iter", 15000)),
            densify_grad_threshold=float(opt_cfg.get("densify_grad_threshold", 0.0002)),
            depth_l1_weight_init=float(opt_cfg.get("depth_l1_weight_init", 0.0)),
            depth_l1_weight_final=float(opt_cfg.get("depth_l1_weight_final", 0.0)),
            random_background=bool(opt_cfg.get("random_background", False)),
            optimizer_type=str(opt_cfg.get("optimizer_type", "default")),
        )

    def _pipeline_args(self) -> SimpleNamespace:
        pipe_cfg = self.config.get("pipeline", {})
        return SimpleNamespace(
            convert_SHs_python=bool(pipe_cfg.get("convert_SHs_python", False)),
            compute_cov3D_python=bool(pipe_cfg.get("compute_cov3D_python", False)),
            debug=bool(pipe_cfg.get("debug", False)),
            antialiasing=bool(pipe_cfg.get("antialiasing", False)),
        )

    def _split_train_validation(self) -> None:
        cameras = list(self.scene.getTrainCameras())
        limit = self.config.get("train_camera_limit")
        if limit is not None and 0 < int(limit) < len(cameras):
            # Training-view ablation: keep only `limit` of the training views, chosen by farthest-point
            # sampling with a fixed seed so every method (agent, fixed schedule) sees the identical subset.
            # The validation split below is then taken from this reduced set; test cameras are untouched.
            sub_rng = random.Random(int(self.config.get("train_camera_limit_seed", 1234)))
            cameras = self._coverage_subset(cameras, int(limit), rng=sub_rng)
        if len(cameras) < 2:
            raise RuntimeError("Need at least two training cameras for train/validation split.")
        requested = int(self.config.get("validation_cameras", 8))
        validation_count = min(max(1, requested), max(1, len(cameras) // 5), len(cameras) - 1)
        selection = str(self.config.get("validation_selection", "coverage"))
        if selection == "coverage":
            self.validation_cameras = self._coverage_subset(cameras, validation_count)
        else:
            self.rng.shuffle(cameras)
            self.validation_cameras = cameras[:validation_count]
        validation_ids = {id(camera) for camera in self.validation_cameras}
        self.train_cameras = [camera for camera in cameras if id(camera) not in validation_ids]
        self.train_stack = []

    def _coverage_subset(self, cameras: list, count: int, rng=None) -> list:
        """Farthest-point sampling over camera centers (random seed start; `rng` overrides the episode RNG).

        A random subset can cluster on one side of the hemisphere, letting the
        policy optimize those views while unobserved directions degrade — which
        the test cameras then expose. FPS spreads the reward cameras over the
        capture sphere so validation quality tracks novel-view quality better.
        """
        centers = np.stack(
            [
                np.asarray(camera.camera_center.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
                for camera in cameras
            ]
        )
        start = (rng or self.rng).randrange(len(cameras))
        chosen = [start]
        distances = np.linalg.norm(centers - centers[start], axis=1)
        while len(chosen) < min(count, len(cameras)):
            next_index = int(distances.argmax())
            chosen.append(next_index)
            distances = np.minimum(distances, np.linalg.norm(centers - centers[next_index], axis=1))
        return [cameras[i] for i in chosen]

    def _sample_training_camera(self):
        if not self.train_stack:
            self.train_stack = list(self.train_cameras)
            self.rng.shuffle(self.train_stack)
        return self.train_stack.pop()

    def _dash_scaled_inputs(self, cam, scale: float, gt_image):
        """Return (render_cam, gt_image, alpha_mask) at the DashGaussian scheduled scale.

        The render camera is a shallow copy with scaled image dimensions (the rasterizer
        reads only image_height/width from the camera; FoV/transforms are resolution
        independent), and the GT / alpha mask are bilinearly downsized to match so the
        loss is computed at the scheduled resolution.
        """
        if scale >= 0.999:
            return cam, gt_image, cam.alpha_mask
        H, W = int(gt_image.shape[-2]), int(gt_image.shape[-1])
        h = max(1, int(round(H * scale)))
        w = max(1, int(round(W * scale)))
        if h == H and w == W:
            return cam, gt_image, cam.alpha_mask
        interp = torch.nn.functional.interpolate
        gt_s = interp(gt_image.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
        mask_s = None
        if cam.alpha_mask is not None:
            mask_s = interp(cam.alpha_mask.cuda().unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
        render_cam = copy.copy(cam)
        render_cam.image_width = w
        render_cam.image_height = h
        return render_cam, gt_s, mask_s

    def _run_training_iteration(self, controls) -> dict[str, float]:
        assert self.gaussians is not None
        self.gaussians.update_learning_rate(self.iteration)
        self._apply_lr_multipliers(controls)

        # FasterGS extension actions: antialiasing toggle (v1 only) + SH-band unlock gate.
        if self.fastergs_aa:
            self.pipe.antialiasing = controls.fastergs_antialiasing == "on"
        allow_sh = (not self.fastergs_ext) or (controls.fastergs_sh_unlock == "allow")
        if self.iteration % 1000 == 0 and allow_sh:
            self.gaussians.oneupSHdegree()

        # FasterGS extension: per-stage GPU timing via async CUDA events (no per-iter
        # host sync, so the wall-clock reward is undistorted; read after the block sync).
        ev = None
        if self.fastergs_ext and torch.cuda.is_available():
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
            ev[0].record()

        viewpoint_cam = self._sample_training_camera()
        # DashGaussian resolution schedule: render (and compute loss) at the scheduled
        # resolution for this iteration; validation/eval stay full-res elsewhere.
        render_cam = viewpoint_cam
        gt_image = viewpoint_cam.original_image.cuda()
        # LeGS cameras pre-multiply the alpha mask into the image (no attribute).
        alpha_mask = getattr(viewpoint_cam, "alpha_mask", None)
        if self._dash is not None:
            scale = self._dash.res_scale(self.iteration)
            self._current_res_scale = scale
            render_cam, gt_image, alpha_mask = self._dash_scaled_inputs(viewpoint_cam, scale, gt_image)
        bg = torch.rand((3), device="cuda") if self.opt.random_background else self.background
        render_out = self.backend.render_training(
            render_cam,
            self.gaussians,
            self.pipe,
            bg,
            use_trained_exp=self.dataset.train_test_exp,
        )
        if ev is not None:
            ev[1].record()
        image = render_out["image"]
        visibility_filter = render_out["visibility_filter"]
        radii = render_out["radii"]

        if alpha_mask is not None:
            image *= alpha_mask.cuda()
        l1_value = self.backend.l1_loss(image, gt_image)
        if self.backend.FUSED_SSIM_AVAILABLE:
            ssim_value = self.backend.fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = self.backend.ssim(image, gt_image)
        dssim_value = 1.0 - ssim_value
        loss = (1.0 - self.opt.lambda_dssim) * l1_value + self.opt.lambda_dssim * dssim_value
        reg_value = self._compactness_regularizer(controls)
        loss = loss + reg_value
        loss.backward()
        if ev is not None:
            ev[2].record()

        densify_stats = {"added": 0, "pruned": 0}
        densify_event = 0
        is_legs = getattr(self.backend, "is_legs", False)
        legs_densify_due = False
        with torch.no_grad():
            if self.iteration < self.opt.densify_until_iter:
                self.backend.accumulate_densification_stats(self.gaussians, render_out)
                # DashGaussian momentum primitive budget: advance the momentum every
                # step in the densify window and cap growth once the count reaches the
                # budget (pruning still runs). Suppresses over-densification at low res.
                dash_allows_growth = True
                if self._dash is not None:
                    cur = int(self.gaussians.get_xyz.shape[0])
                    dash_allows_growth = cur < self._dash.primitive_budget(self.iteration, cur)
                if self._should_apply_densification(controls):
                    if is_legs:
                        # LeGS ADC needs autograd (its state features are per-
                        # Gaussian gradients) and must run after the optimizer
                        # step; deferred below.
                        legs_densify_due = controls.densify_mode != "off"
                    else:
                        densify_event = 1
                        size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
                        threshold = self._controlled_densify_threshold(controls)
                        prune_mode = "opacity_only" if controls.prune_mode == "opacity_only" else controls.prune_mode
                        densify_stats = self.gaussians.densify_and_prune_controlled(
                            threshold,
                            self._effective_prune_opacity_threshold(controls),
                            self.scene.cameras_extent,
                            size_threshold,
                            radii,
                            densify_enabled=controls.densify_mode != "off" and dash_allows_growth,
                            prune_mode=prune_mode,
                            min_remaining=self._min_remaining_gaussians(),
                        )
            if ev is not None:
                ev[3].record()

            if self.iteration < self.opt.iterations:
                exposure_opt = getattr(self.gaussians, "exposure_optimizer", None)
                if exposure_opt is not None:
                    exposure_opt.step()
                    exposure_opt.zero_grad(set_to_none=True)
                if is_legs:
                    # LeGS's lazy dual-optimizer schedule (SH band stepped less
                    # often) is part of its speed; use it verbatim.
                    self.gaussians.optimizer_step(self.iteration)
                elif self.opt.optimizer_type == "sparse_adam" and self.backend.SPARSE_ADAM_AVAILABLE:
                    visible = radii > 0
                    self.gaussians.optimizer.step(visible, radii.shape[0])
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
            if ev is not None:
                ev[4].record()

        if is_legs:
            # LeGS ADC + efficient delayed reward, outside no_grad (the state
            # features need backward passes of their own; grads are consumed
            # after the optimizer step so the main update is unaffected).
            if legs_densify_due:
                allow_growth = int(self.gaussians.get_xyz.shape[0]) < self._hard_stop_gaussians()
                densify_stats = self.backend.legs_adc(
                    self.gaussians, self.train_cameras, self.pipe, self.background,
                    self.iteration,
                    grad_scale=float(controls.densify_threshold_mult),
                    allow_growth=allow_growth,
                )
                densify_event = 1
            self.backend.legs_reward_tick(
                self.gaussians, self.pipe, self.background, self.iteration
            )

        if ev is not None:
            self._fgs_ev.append(ev)
            self._fgs_last_radii = radii.detach()
            di = render_out.get("densification_info")
            self._fgs_last_grad = di[1].detach() if di is not None else None

        visible_count = int((radii > 0).sum().item())
        total_count = max(1, int(self.gaussians.get_xyz.shape[0]))
        return {
            "l1": float(l1_value.detach().item()),
            "dssim": float(dssim_value.detach().item()),
            "total_loss": float(loss.detach().item()),
            "visible_gaussians": float(visible_count),
            "visible_fraction": float(visible_count / total_count),
            "densify_event": float(densify_event),
            "gaussians_added": float(densify_stats.get("added", 0)),
            "gaussians_pruned": float(densify_stats.get("pruned", 0)),
        }

    def _apply_lr_multipliers(self, controls) -> None:
        mults = {
            "xyz": controls.position_lr_mult,
            "f_dc": controls.feature_lr_mult,
            "f_rest": controls.feature_lr_mult,
            "opacity": controls.opacity_lr_mult,
            "scaling": controls.scaling_lr_mult,
            "rotation": controls.rotation_lr_mult,
        }
        base_lrs = {
            "f_dc": self.opt.feature_lr,
            "f_rest": self.opt.feature_lr / 20.0,
            "opacity": self.opt.opacity_lr,
            "scaling": self.opt.scaling_lr,
            "rotation": self.opt.rotation_lr,
        }
        if getattr(self.backend, "is_legs", False):
            # LeGS splits SH into two optimizers with its own base lrs.
            base_lrs["f_dc"] = float(self.opt.lowfeature_lr)
        for group in self.gaussians.optimizer.param_groups:
            name = group["name"]
            if name == "xyz":
                group["lr"] *= mults[name]
            elif name in base_lrs:
                group["lr"] = base_lrs[name] * mults[name]
        sh_optimizer = getattr(self.gaussians, "shoptimizer", None)
        if sh_optimizer is not None:
            for group in sh_optimizer.param_groups:
                if group["name"] == "f_rest":
                    group["lr"] = (float(self.opt.highfeature_lr) / 20.0) * mults["f_rest"]

    def _compactness_regularizer(self, controls) -> torch.Tensor:
        """Opacity / volume regularization added to the photometric loss.

        Pushes low-contribution Gaussians toward transparency and discourages
        oversized primitives so that pruning actually reduces the model size.
        Gated to start after ``reg_from_iter`` so early densification is healthy.
        Strength can be scaled by an optional policy action ``compaction_mult``.
        """
        loss_cfg = self.config.get("loss", {})
        reg = torch.zeros((), device="cuda")
        if self.gaussians is None:
            return reg
        if self.iteration < int(loss_cfg.get("reg_from_iter", 0)):
            return reg
        mult = float(getattr(controls, "compaction_mult", 1.0))
        lambda_opacity = float(loss_cfg.get("lambda_opacity", 0.0)) * mult
        lambda_volume = float(loss_cfg.get("lambda_volume", 0.0)) * mult
        if lambda_opacity > 0.0:
            reg = reg + lambda_opacity * self.gaussians.get_opacity.mean()
        if lambda_volume > 0.0:
            extent = max(1e-6, float(self.scene.cameras_extent))
            reg = reg + lambda_volume * (self.gaussians.get_scaling / extent).prod(dim=1).mean()
        return reg

    def _reference_gaussian_count(self) -> float:
        """Budget the policy is expected to stay under (defaults to a multiple
        of the initial count, overridable per scene to the measured baseline)."""
        reward_cfg = self.config.get("reward", {})
        budgets = reward_cfg.get("reference_gaussian_budget", {})
        if isinstance(budgets, dict) and self.scene_id in budgets:
            return max(1.0, float(budgets[self.scene_id]))
        target = float(reward_cfg.get("reference_gaussian_target", 0.0))
        if target > 0.0:
            return target
        mult = float(reward_cfg.get("reference_gaussian_mult", 3.0))
        return max(1.0, mult * max(1, self.initial_gaussian_count))

    def _controlled_densify_threshold(self, controls) -> float:
        mode_mult = {
            "off": 1.0,
            "conservative": 2.0,
            "default": 1.0,
            "aggressive": 0.5,
        }[controls.densify_mode]
        return float(self.opt.densify_grad_threshold * mode_mult * controls.densify_threshold_mult)

    def _effective_prune_opacity_threshold(self, controls) -> float:
        """Post-reset prune guard.

        Stock 3DGS resets opacity to 0.01 and prunes at 0.005, so survivors can
        recover. Our action range allows prune thresholds above the reset value,
        which lets a reset followed by one prune event wipe the model down to the
        safety floor in a single block (observed: 104k -> 50k at iter ~800). Cap
        the threshold to the stock value for a recovery window after each reset.
        """
        safety = self.config.get("safety", {})
        guard_iters = int(safety.get("post_reset_prune_guard_iters", 300))
        if self.iteration - self.last_opacity_reset_iteration < guard_iters:
            cap = float(safety.get("post_reset_prune_cap", 0.005))
            return min(float(controls.prune_opacity_threshold), cap)
        return float(controls.prune_opacity_threshold)

    def _min_remaining_gaussians(self) -> int:
        safety = self.config.get("safety", {})
        absolute_min = int(safety.get("min_gaussians", 1000))
        initial_fraction = float(safety.get("min_gaussian_fraction_of_initial", 0.5))
        initial_min = int(round(max(0.0, initial_fraction) * max(1, self.initial_gaussian_count)))
        floor = max(absolute_min, initial_min)
        # Cap the loss from any single prune event so a collapse to the floor
        # takes sustained pruning across blocks (visible to the reward) instead
        # of one cliff-edge action.
        max_prune_fraction = float(safety.get("max_prune_fraction_per_event", 0.25))
        if 0.0 < max_prune_fraction < 1.0 and self.gaussians is not None:
            current = int(self.gaussians.get_xyz.shape[0])
            floor = max(floor, int(round((1.0 - max_prune_fraction) * current)))
        return floor

    def _should_apply_densification(self, controls) -> bool:
        if self.iteration <= self.opt.densify_from_iter:
            return False
        return self.iteration % int(controls.densification_interval) == 0

    def _maybe_reset_opacity(self, controls) -> None:
        if self.gaussians is None:
            return
        safety = self.config.get("safety", {})
        min_iter = int(safety.get("min_opacity_reset_iter", 500))
        cooldown = int(safety.get("opacity_reset_cooldown_blocks", 3))
        max_resets = int(safety.get("max_opacity_resets", 5))
        if self.iteration < min_iter:
            return
        if self.opacity_resets >= max_resets:
            return
        if self.block_index - self.last_opacity_reset_block < cooldown:
            return

        should_reset = False
        if controls.opacity_reset == "force_reset":
            should_reset = True
        elif controls.opacity_reset == "reset_if_plateau":
            plateau_eps = float(safety.get("plateau_psnr_epsilon", 0.02))
            should_reset = self.last_quality_gain <= plateau_eps

        if should_reset:
            self.gaussians.reset_opacity()
            self.opacity_resets += 1
            self.last_opacity_reset_block = self.block_index
            self.last_opacity_reset_iteration = self.iteration

    @torch.no_grad()
    def _render_camera_metrics(self, cameras: list) -> dict[str, list[float]]:
        per_view: dict[str, list[float]] = {"psnr": [], "ssim": [], "l1": []}
        for viewpoint in cameras:
            image = torch.clamp(
                self.backend.render_image(
                    viewpoint,
                    self.gaussians,
                    self.pipe,
                    self.background,
                    use_trained_exp=self.dataset.train_test_exp,
                ),
                0.0,
                1.0,
            )
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            per_view["l1"].append(float(self.backend.l1_loss(image, gt_image).item()))
            per_view["psnr"].append(float(self.backend.psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().item()))
            per_view["ssim"].append(float(self.backend.ssim(image, gt_image).item()))
        return per_view

    @torch.no_grad()
    def _evaluate_validation_subset(self) -> dict[str, float]:
        if not self.validation_cameras:
            return {"psnr": 0.0, "ssim": 0.0, "l1": 1.0, "psnr_min": 0.0, "quality": 0.0, "seconds": 0.0, "views": 0}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        per_view = self._render_camera_metrics(self.validation_cameras)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start

        reward_cfg = self.config.get("reward", {})
        ssim_weight = float(reward_cfg.get("ssim_quality_weight", 10.0))
        per_view_quality = [p + ssim_weight * s for p, s in zip(per_view["psnr"], per_view["ssim"])]
        mean_quality = float(np.mean(per_view_quality))
        min_quality = float(np.min(per_view_quality))
        # Mean quality lets the policy trade away poorly covered views, which the
        # (full-hemisphere) test cameras then expose. Blending in the worst view
        # makes the reward proxy robust to that view-selective optimization.
        min_weight = float(reward_cfg.get("min_view_weight", 0.0))
        quality = mean_quality + min_weight * (min_quality - mean_quality)
        return {
            "psnr": float(np.mean(per_view["psnr"])),
            "ssim": float(np.mean(per_view["ssim"])),
            "l1": float(np.mean(per_view["l1"])),
            "psnr_min": float(np.min(per_view["psnr"])),
            "quality": quality,
            "seconds": seconds,
            "views": len(per_view["psnr"]),
        }

    @torch.no_grad()
    def evaluate_test_cameras(self, max_views: int = 0) -> dict[str, float]:
        """Mean metrics over the dataset test cameras.

        For checkpoint selection and reporting ONLY. This is never called from
        reset()/step(), so test cameras still never influence policy state or
        reward construction. With max_views > 0 an evenly spaced deterministic
        subset is used (fast mid-episode checkpoints for acceleration scoring).
        """
        cameras = list(self.scene.getTestCameras()) if self.scene is not None else []
        if max_views > 0 and len(cameras) > max_views:
            step = max(1, len(cameras) // max_views)
            cameras = cameras[::step][:max_views]
        if not cameras:
            return {"psnr": 0.0, "ssim": 0.0, "l1": 1.0, "views": 0}
        per_view = self._render_camera_metrics(cameras)
        return {
            "psnr": float(np.mean(per_view["psnr"])),
            "ssim": float(np.mean(per_view["ssim"])),
            "l1": float(np.mean(per_view["l1"])),
            "views": len(per_view["psnr"]),
        }

    def _compute_reward(self, validation: dict[str, float], block_stats: dict[str, float]) -> tuple[float, dict[str, float]]:
        reward_cfg = self.config.get("reward", {})
        quality_gain = validation["quality"] - self.prev_validation_quality
        psnr_gain = validation["psnr"] - self.last_validation["psnr"]
        ssim_gain = validation["ssim"] - self.last_validation["ssim"]

        # --- model-complexity reference and discount ---------------------------
        n_now = float(block_stats["end_gaussians"])
        n_ref = self._reference_gaussian_count()
        excess = max(0.0, n_now / max(1.0, n_ref) - 1.0)
        kappa = float(reward_cfg.get("complexity_discount_kappa", 1.0))
        complexity_discount = 1.0 / (1.0 + kappa * excess)

        # --- quality credit: positive gains are discounted when over budget,
        #     drops keep full magnitude (handled again by instability term) -----
        pos_gain = max(0.0, quality_gain)
        neg_gain = min(0.0, quality_gain)
        quality_term = float(reward_cfg.get("quality_gain_weight", 1.0)) * (
            pos_gain * complexity_discount + neg_gain
        )

        # --- time value of progress (acceleration objective) --------------------
        # Plain gain telescopes to (final - initial) quality, which is indifferent
        # to WHEN quality is earned and so cannot prefer a faster-converging
        # schedule. Weighting each gain by exp(-train_seconds/tau) makes early
        # progress worth more — the policy maximizes the early area under the
        # quality-vs-time curve, which is what time-to-target measures. Stopping
        # emerges naturally once discounted marginal gains drop below time cost.
        # Budget-conditioned: make the time-value discount budget-relative
        # (tau = fraction * B) so the "front-load quality" incentive keeps the
        # same shape across sampled budgets rather than a fixed absolute tau.
        if self.budget_conditioned and self.time_budget:
            tau = max(1e-6, self.budget_tau_fraction * self.time_budget)
        else:
            tau = float(reward_cfg.get("time_value_tau_seconds", 0.0))
        time_value = 1.0
        if tau > 0.0:
            time_value = float(np.exp(-self.training_seconds / tau))
            quality_term *= time_value

        # --- convergence / compute cost (quality-per-time) ---------------------
        t_ref = max(1e-6, float(reward_cfg.get("block_time_ref_seconds", 1.0)))
        time_penalty = float(reward_cfg.get("time_penalty_weight", 0.05)) * (
            block_stats["block_seconds"] / t_ref
        )

        # --- explicit over-baseline count penalty ------------------------------
        count_penalty = float(reward_cfg.get("count_excess_penalty_weight", 0.3)) * excess

        # --- growth penalty / compaction bonus, gated to the over-budget range --
        # Below N_ref both terms are neutral: the objective is complexity <=
        # budget, not minimal count. Paying for pruning at any level collapses
        # the policy to the safety floor (v1/v3 runs); penalizing growth at any
        # level blocks recovery from that floor. Only the over-budget portion
        # of a change is priced.
        start_n = float(block_stats["start_gaussians"])
        growth_above_ref = max(0.0, n_now - max(start_n, n_ref))
        growth_penalty = float(reward_cfg.get("gaussian_growth_penalty_weight", 1.0)) * (
            growth_above_ref / max(1.0, n_ref)
        )
        tol = float(reward_cfg.get("compaction_quality_tolerance", 0.0))
        compaction_bonus = 0.0
        if quality_gain >= -tol:
            reduction_above_ref = max(0.0, start_n - max(n_now, n_ref))
            compaction_bonus = float(reward_cfg.get("compaction_bonus_weight", 1.0)) * (
                reduction_above_ref / max(1.0, n_ref)
            )

        # --- backend-aware quality/compute term --------------------------------
        # On a cheap backend (Faster-GS) each optimization iteration costs little, so the
        # policy should spend that compute reaching higher quality rather than
        # over-compacting. We credit positive quality gains amplified by how cheap the
        # measured per-iteration compute is (cost_ref / time_per_iteration, capped). On a
        # slower backend the factor stays ~1. Config-gated (default weight 0 => no effect,
        # so the base 3DGS / plain reward is unchanged).
        qpc_weight = float(reward_cfg.get("quality_per_compute_weight", 0.0))
        qpc_term = 0.0
        if qpc_weight > 0.0:
            cost_ref = max(1e-6, float(reward_cfg.get("compute_cost_ref_seconds", 0.01)))
            per_iter = max(1e-6, float(block_stats.get("time_per_iteration", cost_ref)))
            speed_factor = min(float(reward_cfg.get("compute_speed_cap", 3.0)), cost_ref / per_iter)
            qpc_term = qpc_weight * max(0.0, quality_gain) * speed_factor

        mem = _memory_gb()
        target_vram = float(self.config.get("safety", {}).get("target_vram_gb", 20.0))
        vram_penalty = float(reward_cfg.get("vram_penalty_weight", 0.01)) * max(0.0, mem["peak"] - target_vram)
        instability_penalty = float(reward_cfg.get("validation_drop_penalty_weight", 0.1)) * max(0.0, -quality_gain)

        # Terminal quality bonus: reward the final quality reached by the episode
        # end. Fire it at the iteration cap OR — for budget-conditioning — at
        # budget exhaustion (otherwise a budget-terminated episode never reaches
        # the iteration cap and the agent is never rewarded for final quality,
        # so it just stops ASAP). Not time-discounted: it is quality-at-budget.
        terminal_reached = self.iteration >= int(self.opt.iterations)
        if self.budget_conditioned and self.time_budget and self.training_seconds >= self.time_budget:
            terminal_reached = True
        terminal_bonus = 0.0
        if terminal_reached:
            tq_discount = 1.0 if self.budget_conditioned else time_value
            terminal_bonus = float(reward_cfg.get("terminal_quality_weight", 0.1)) * (
                validation["quality"] - self.initial_validation_quality
            ) * complexity_discount * tq_discount
            terminal_bonus -= float(reward_cfg.get("terminal_count_penalty_weight", 0.5)) * excess

        reward = (
            quality_term
            + qpc_term
            - time_penalty
            - count_penalty
            - growth_penalty
            + compaction_bonus
            - vram_penalty
            - instability_penalty
            + terminal_bonus
        )
        terms = {
            "quality_per_compute": qpc_term,
            "quality_gain": quality_gain,
            "psnr_gain": psnr_gain,
            "ssim_gain": ssim_gain,
            "time_value": time_value,
            "complexity_discount": complexity_discount,
            "count_excess": excess,
            "reference_gaussian_count": n_ref,
            "time_penalty": -time_penalty,
            "count_excess_penalty": -count_penalty,
            "gaussian_growth_penalty": -growth_penalty,
            "compaction_bonus": compaction_bonus,
            "vram_penalty": -vram_penalty,
            "instability_penalty": -instability_penalty,
            "terminal_quality_bonus": terminal_bonus,
        }
        return reward, terms

    def _is_done(self, controls, block_stats: dict[str, float]) -> bool:
        if self.iteration >= int(self.opt.iterations):
            return True
        # Budget-conditioning: the sampled wall-clock training budget is the
        # primary episode terminator. The stop head may still fire earlier.
        if self.budget_conditioned and self.time_budget and self.training_seconds >= self.time_budget:
            return True
        max_wall = float(self.config.get("max_wall_seconds", 0.0))
        if max_wall > 0 and self.elapsed_seconds >= max_wall:
            return True
        if block_stats["numerical_failure"] or block_stats["hard_budget_exceeded"]:
            return True
        # In budget-conditioning the wall-clock budget is the stopping criterion,
        # so the stop head does not terminate the episode (every episode runs to
        # its budget and always collects the terminal quality bonus).
        if self.budget_conditioned:
            return False
        min_stop_iter = int(self.config.get("safety", {}).get("min_iterations_before_stop", 500))
        return controls.stop == "stop" and self.iteration >= min_stop_iter

    def _hard_stop_gaussians(self) -> int:
        """Population that ENDS the episode.

        Separate from safety.hard_max_gaussians, which also normalises five observation features
        (count, growth, added, pruned). Raising that one to let a run continue would rescale the
        policy's inputs and so change the controller itself; safety.hard_stop_gaussians moves only
        the termination threshold and defaults to hard_max_gaussians, so every existing config
        keeps its current behaviour.
        """
        safety = self.config.get("safety", {})
        return int(safety.get("hard_stop_gaussians",
                              safety.get("hard_max_gaussians", 3_000_000)))

    def _hard_budget_exceeded(self) -> bool:
        safety = self.config.get("safety", {})
        if int(self.gaussians.get_xyz.shape[0]) > self._hard_stop_gaussians():
            return True
        max_vram = float(safety.get("hard_max_vram_gb", 24.0))
        return _memory_gb()["peak"] > max_vram

    def _finalize_fastergs_telemetry(self, block_stats: dict[str, float], controls) -> None:
        """Reduce the block's per-iteration CUDA events + coverage into FasterGS obs.
        Called after the block-level cuda.synchronize(), so all events are complete."""
        fwd = bwd = opt = dns = 0.0
        for ev in self._fgs_ev:
            try:
                fwd += ev[0].elapsed_time(ev[1])
                bwd += ev[1].elapsed_time(ev[2])
                dns += ev[2].elapsed_time(ev[3])
                opt += ev[3].elapsed_time(ev[4])
            except Exception:
                continue
        total = fwd + bwd + opt + dns
        if total > 1e-6:
            block_stats["fgs_frac_forward"] = fwd / total
            block_stats["fgs_frac_backward"] = bwd / total
            block_stats["fgs_frac_optimizer"] = opt / total
            block_stats["fgs_frac_densify"] = dns / total
        # Densification gradient: |densification_info[1]| among visible Gaussians, reported
        # relative to the densify_grad_threshold so the policy sees how strongly the
        # population wants to densify (the signal the fork writes correctly on COLMAP).
        grad, vis = self._fgs_last_grad, self._fgs_last_radii
        if grad is not None and vis is not None and grad.numel() > 0:
            g = grad.abs().float()
            mask = vis > 0
            gv = g[mask] if bool(mask.any()) else g
            if gv.numel() > 0:
                thr = max(1e-8, float(getattr(self.opt, "densify_grad_threshold", 2e-4)))
                block_stats["fgs_densify_grad_q50"] = min(1.0, float(torch.quantile(gv, 0.5).item()) / (10.0 * thr))
                block_stats["fgs_densify_grad_q90"] = min(1.0, float(torch.quantile(gv, 0.9).item()) / (10.0 * thr))
                block_stats["fgs_densify_grad_active_fraction"] = float((gv > thr).float().mean().item())
        block_stats["fgs_antialiasing_on"] = 1.0 if controls.fastergs_antialiasing == "on" else 0.0

    def _build_observation(self) -> np.ndarray:
        max_iter = max(1, int(self.config.get("max_episode_iterations", 7000)))
        max_wall = max(1e-6, float(self.config.get("max_wall_seconds", 1.0)))
        hard_max_gaussians = max(1, int(self.config.get("safety", {}).get("hard_max_gaussians", 3_000_000)))
        target_vram = max(1e-6, float(self.config.get("safety", {}).get("target_vram_gb", 20.0)))
        block_stats = self.last_block_stats
        mem = _memory_gb()
        gstats = self._gaussian_stats()
        loss_slope = 0.0
        if len(self.loss_history) >= 2:
            loss_slope = self.loss_history[-1] - self.loss_history[-2]

        # Budget-conditioned: the elapsed/remaining "budget" fractions become
        # wall-clock-relative to this episode's sampled time budget.
        if self.budget_conditioned and self.time_budget:
            elapsed_frac = self.training_seconds / self.time_budget
            remaining_frac = max(0.0, 1.0 - self.training_seconds / self.time_budget)
        else:
            elapsed_frac = self.elapsed_seconds / max_wall
            remaining_frac = max(0.0, 1.0 - self.iteration / max_iter)

        obs = [
            self.iteration / max_iter,
            elapsed_frac,
            remaining_frac,
            self.block_index / max(1.0, max_iter / 100.0),
            block_stats["l1"],
            block_stats["dssim"],
            block_stats["total_loss"],
            0.5 + 0.5 * np.tanh(loss_slope),
            self.last_validation["psnr"] / 50.0,
            self.last_validation["ssim"],
            0.0,
            0.5 + 0.5 * np.tanh(self.last_validation["quality"] - self.prev_validation_quality),
            block_stats["time_per_iteration"] / 0.2,
            (self.last_validation.get("seconds", 0.0) / max(1, self.last_validation.get("views", 1))) / 0.5,
            mem["peak"] / target_vram,
            mem["allocated"] / target_vram,
            self._current_res_scale,
            gstats["count"] / hard_max_gaussians,
            max(0.0, block_stats["gaussian_growth"]) / hard_max_gaussians,
            gstats["count"] / hard_max_gaussians,
            *action_to_observation_values(self.prev_action),
            1.0 if self.prev_action_improved else 0.0,
            gstats["opacity_q10"],
            gstats["opacity_q50"],
            gstats["opacity_q90"],
            gstats["near_transparent_fraction"],
            gstats["scale_q50"],
            gstats["scale_q90"],
            gstats["anisotropy_q50"],
            block_stats["last_visible_fraction"],
            block_stats["gaussians_added"] / hard_max_gaussians,
            block_stats["gaussians_pruned"] / hard_max_gaussians,
            gstats["active_sh_degree_fraction"],
        ]
        if self.budget_conditioned:
            # Absolute budget (normalized) so the policy can adapt strategy to
            # small-vs-large budgets, not just its position within one.
            obs.append((self.time_budget or 0.0) / max(1e-6, self.budget_max_seconds))
        if self.fastergs_ext:
            # FasterGS-specific channels (order matches spaces._fastergs_obs()).
            obs.extend([
                block_stats.get("fgs_frac_forward", 0.0),
                block_stats.get("fgs_frac_backward", 0.0),
                block_stats.get("fgs_frac_optimizer", 0.0),
                block_stats.get("fgs_frac_densify", 0.0),
                block_stats.get("fgs_densify_grad_q50", 0.0),
                block_stats.get("fgs_densify_grad_q90", 0.0),
                block_stats.get("fgs_densify_grad_active_fraction", 0.0),
            ])
            if self.fastergs_aa:
                obs.append(block_stats.get("fgs_antialiasing_on", 0.0))
        obs = np.asarray([_clip01(float(v)) for v in obs], dtype=np.float32)
        if obs.shape[0] != len(self._full_observation_names):
            raise RuntimeError(
                f"Observation length mismatch: {obs.shape[0]} != {len(self._full_observation_names)}")
        if len(self._obs_keep_indices) != obs.shape[0]:
            obs = obs[self._obs_keep_indices]          # state-vector ablation
        if obs.shape[0] != len(self.observation_names):
            raise RuntimeError(f"Observation length mismatch: {obs.shape[0]} != {len(self.observation_names)}")
        return obs

    @torch.no_grad()
    def _gaussian_stats(self) -> dict[str, float]:
        if self.gaussians is None:
            return {
                "count": 0.0,
                "opacity_q10": 0.0,
                "opacity_q50": 0.0,
                "opacity_q90": 0.0,
                "near_transparent_fraction": 0.0,
                "scale_q50": 0.0,
                "scale_q90": 0.0,
                "anisotropy_q50": 0.0,
                "active_sh_degree_fraction": 0.0,
            }
        opacity = self.gaussians.get_opacity.detach().flatten().float()
        scaling = self.gaussians.get_scaling.detach().float()
        max_scale = scaling.max(dim=1).values
        min_scale = torch.clamp(scaling.min(dim=1).values, min=1e-8)
        anisotropy = max_scale / min_scale
        extent = max(1e-6, float(self.scene.cameras_extent))

        def q(tensor, value):
            if tensor.numel() == 0:
                return 0.0
            return float(torch.quantile(tensor, value).item())

        return {
            "count": float(self.gaussians.get_xyz.shape[0]),
            "opacity_q10": q(opacity, 0.10),
            "opacity_q50": q(opacity, 0.50),
            "opacity_q90": q(opacity, 0.90),
            "near_transparent_fraction": float((opacity < 0.01).float().mean().item()) if opacity.numel() else 0.0,
            "scale_q50": min(q(max_scale / extent, 0.50), 1.0),
            "scale_q90": min(q(max_scale / extent, 0.90), 1.0),
            "anisotropy_q50": min(np.log1p(q(anisotropy, 0.50)) / 5.0, 1.0),
            "active_sh_degree_fraction": self.gaussians.active_sh_degree / max(1, self.gaussians.max_sh_degree),
        }

    def _empty_block_stats(self) -> dict[str, float]:
        return {
            "start_iteration": 0,
            "iterations_ran": 0,
            "l1": 0.0,
            "dssim": 0.0,
            "total_loss": 0.0,
            "visible_gaussians": 0.0,
            "last_visible_fraction": 0.0,
            "densify_events": 0.0,
            "gaussians_added": 0.0,
            "gaussians_pruned": 0.0,
            "start_gaussians": 0.0,
            "end_gaussians": 0.0,
            "gaussian_growth": 0.0,
            "block_seconds": 0.0,
            "time_per_iteration": 0.0,
            "numerical_failure": 0.0,
            "hard_budget_exceeded": 0.0,
            # FasterGS extension telemetry (populated only when fastergs_ext is on).
            "fgs_frac_forward": 0.0,
            "fgs_frac_backward": 0.0,
            "fgs_frac_optimizer": 0.0,
            "fgs_frac_densify": 0.0,
            "fgs_densify_grad_q50": 0.0,
            "fgs_densify_grad_q90": 0.0,
            "fgs_densify_grad_active_fraction": 0.0,
            "fgs_antialiasing_on": 0.0,
        }

    def _init_block_log(self) -> None:
        fields = [
            "scene",
            "block",
            "iteration",
            "reward",
            "validation_psnr",
            "validation_psnr_min",
            "validation_ssim",
            "validation_l1",
            "quality_gain",
            "complexity_discount",
            "count_excess",
            "reference_gaussian_count",
            "time_penalty",
            "count_excess_penalty",
            "gaussian_growth_penalty",
            "compaction_bonus",
            "vram_penalty",
            "instability_penalty",
            "terminal_quality_bonus",
            "block_seconds",
            "validation_seconds",
            "elapsed_seconds",
            "time_per_iteration",
            "gaussian_count",
            "gaussians_added",
            "gaussians_pruned",
            "peak_vram_gb",
            "action_json",
        ]
        with self.block_log_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    def _append_block_log(self, info: dict[str, Any], reward: float) -> None:
        reward_terms = info["reward_terms"]
        block_stats = info["block_stats"]
        validation = info["validation"]
        row = {
            "scene": info["scene"],
            "block": info["block_index"],
            "iteration": info["iteration"],
            "reward": reward,
            "validation_psnr": validation["psnr"],
            "validation_psnr_min": validation.get("psnr_min", 0.0),
            "validation_ssim": validation["ssim"],
            "validation_l1": validation["l1"],
            "quality_gain": reward_terms.get("quality_gain", 0.0),
            "complexity_discount": reward_terms.get("complexity_discount", 1.0),
            "count_excess": reward_terms.get("count_excess", 0.0),
            "reference_gaussian_count": reward_terms.get("reference_gaussian_count", 0.0),
            "time_penalty": reward_terms.get("time_penalty", 0.0),
            "count_excess_penalty": reward_terms.get("count_excess_penalty", 0.0),
            "gaussian_growth_penalty": reward_terms.get("gaussian_growth_penalty", 0.0),
            "compaction_bonus": reward_terms.get("compaction_bonus", 0.0),
            "vram_penalty": reward_terms.get("vram_penalty", 0.0),
            "instability_penalty": reward_terms.get("instability_penalty", 0.0),
            "terminal_quality_bonus": reward_terms.get("terminal_quality_bonus", 0.0),
            "block_seconds": block_stats["block_seconds"],
            "validation_seconds": validation.get("seconds", 0.0),
            "elapsed_seconds": self.elapsed_seconds,
            "time_per_iteration": block_stats["time_per_iteration"],
            "gaussian_count": block_stats["end_gaussians"],
            "gaussians_added": block_stats["gaussians_added"],
            "gaussians_pruned": block_stats["gaussians_pruned"],
            "peak_vram_gb": _memory_gb()["peak"],
            "action_json": json.dumps(info["action"], sort_keys=True),
        }
        with self.block_log_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)

    def _write_episode_metadata(self) -> None:
        metadata = {
            "scene": self.scene_id,
            "model_path": str(self.model_path),
            "train_camera_count": len(self.train_cameras),
            "validation_camera_count": len(self.validation_cameras),
            "train_camera_names": [camera.image_name for camera in self.train_cameras],
            "validation_camera_names": [camera.image_name for camera in self.validation_cameras],
            "test_camera_accessed_for_reward_or_state": False,
            "observation_names": self.observation_names,
            # horizon the Dash curriculum was built against (None when the backend is not Dash);
            # equals the episode cap unless dash.schedule_horizon_iterations pins it
            "dash_schedule_horizon": self.dash_schedule_horizon,
            "episode_iteration_cap": int(self.opt.iterations) if self.opt is not None else None,
            "config": self.config,
        }
        with (self.model_path / "agentic_episode_metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)
        cfg_args = {
            "sh_degree": self.dataset.sh_degree,
            "source_path": self.dataset.source_path,
            "model_path": self.dataset.model_path,
            "images": self.dataset.images,
            "depths": self.dataset.depths,
            "resolution": self.dataset.resolution,
            "white_background": self.dataset.white_background,
            "train_test_exp": self.dataset.train_test_exp,
            "data_device": self.dataset.data_device,
            "eval": self.dataset.eval,
            "convert_SHs_python": self.pipe.convert_SHs_python,
            "compute_cov3D_python": self.pipe.compute_cov3D_python,
            "debug": self.pipe.debug,
            "antialiasing": self.pipe.antialiasing,
        }
        namespace_items = ", ".join(f"{key}={repr(value)}" for key, value in cfg_args.items())
        with (self.model_path / "cfg_args").open("w", encoding="utf-8") as f:
            f.write(f"Namespace({namespace_items})")
