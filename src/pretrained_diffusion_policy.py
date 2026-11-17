"""Adapter for the original pretrained low-dimensional Push-T Diffusion Policy.

The original model receives 20D keypoints, not this project's raw 10D state.
Weights, normalization and denoiser stay in the original repo. DDPM retains the
checkpoint's full sampling chain; explicit DDIM mode supports fewer steps.
This adapter supplies observations and executes aligned action chunks through
the project's common live simulator runner. Optional model dependencies are
imported only when loading an original checkpoint.
"""

from collections import deque
from dataclasses import asdict, dataclass
import inspect
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

try:
    from .pusht_runtime import PROJECT_ROOT
except ImportError:
    from pusht_runtime import PROJECT_ROOT


def _reference_repository():
    reference = PROJECT_ROOT.parent / "diffusion_policy"
    if not (reference / "diffusion_policy" / "env" / "pusht").is_dir():
        raise FileNotFoundError(f"Original Diffusion Policy repository not found: {reference}")
    if str(reference) not in sys.path:
        sys.path.insert(0, str(reference))
    return reference


def original_block_keypoints():
    """Return the exact deterministic 9-point local map used by the original env."""
    _reference_repository()
    from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv

    parameters = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    return np.asarray(parameters["local_keypoint_map"]["block"], dtype=np.float64).copy()


def state_to_keypoints(states, block_keypoints):
    """Convert raw (...,10) states to original (...,20) keypoint observations.

    The nine local T points are rigidly transformed by the recorded T body
    origin/angle, followed by actuator XY. Velocity features are not supplied
    to the pretrained model. The full-visibility mask is supplied separately.
    """
    states = np.asarray(states, dtype=np.float64)
    points = np.asarray(block_keypoints, dtype=np.float64)
    if states.ndim < 1 or states.shape[-1] != 10 or not np.isfinite(states).all():
        raise ValueError("states must contain finite raw 10D state vectors")
    if points.shape != (9, 2) or not np.isfinite(points).all():
        raise ValueError("The pretrained policy requires nine finite local T keypoints")
    cosine, sine = np.cos(states[..., 8, None]), np.sin(states[..., 8, None])
    x = cosine * points[:, 0] - sine * points[:, 1] + states[..., 4, None]
    y = sine * points[:, 0] + cosine * points[:, 1] + states[..., 5, None]
    keypoints = np.stack([x, y], axis=-1).reshape(states.shape[:-1] + (18,))
    return np.concatenate([keypoints, states[..., :2]], axis=-1).astype(np.float32)


def make_ddim_scheduler(original_schedule):
    """Build an optional fast sampler with the checkpoint's trained noise schedule.

    Import lazily so keypoint conversion and adapter tests do not require the
    original model's optional Diffusers dependency. DDIM changes sampling, not
    the pretrained denoiser or its normalization.
    """
    from diffusers import DDIMScheduler

    return DDIMScheduler.from_config(original_schedule.config)


def _validate_sampling(sampler, steps, diffusion_steps):
    if sampler not in ("ddpm", "ddim"):
        raise ValueError("sampler must be 'ddpm' or 'ddim'")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= diffusion_steps:
        raise ValueError("inference_steps must be an integer in [1, diffusion_steps]")
    if sampler == "ddpm" and steps != diffusion_steps:
        raise ValueError(
            "DDPM must use all diffusion_steps: the pinned diffusers 0.11.1 DDPM scheduler "
            "does not correctly support skipped timesteps. To reduce steps, use "
            "--sampler ddim --inference-steps 20.")


@dataclass
class PretrainedPolicyConfig:
    obs_horizon: int = 2
    original_horizon: int = 16
    action_offset: int = 1
    prediction_horizon: int = 15
    action_horizon: int = 8
    max_action_horizon: int = 8
    inference_steps: int = 100
    diffusion_steps: int = 100
    sampler: str = "ddpm"

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "sampler":
                continue
            minimum = 0 if name == "action_offset" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.action_offset != self.obs_horizon - 1:
            raise ValueError("The supported checkpoint must use pre-action observation alignment")
        if self.prediction_horizon != self.original_horizon - self.action_offset:
            raise ValueError("prediction_horizon must exclude the historical action prefix")
        if not 1 <= self.action_horizon <= self.max_action_horizon <= self.prediction_horizon:
            raise ValueError("action_horizon must not exceed the checkpoint's executable action chunk")
        _validate_sampling(self.sampler, self.inference_steps, self.diffusion_steps)


class PretrainedDiffusionPolicy:
    """Expose the common act/reset interface while retaining the native policy.

    ``last_plan`` contains only current/future actions. For the published
    horizon-16, observation-2 checkpoint, index zero is a historical action;
    the returned plan therefore has 15 actions, beginning at native index one.
    Native global Torch RNG use is isolated into an episode-local RNG stream.
    """

    name = "pretrained_diffusion_policy"
    display_name = "Pretrained Diffusion Policy"
    dt = 0.1

    def __init__(self, native_policy, config=None, block_keypoints=None,
                 metadata=None, device="cpu", seed=42):
        self.config = config or PretrainedPolicyConfig()
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("The pretrained adapter supports cpu or cuda devices")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable; use --device cpu")
            self.device = torch.device("cuda", self.device.index if self.device.index is not None
                                       else torch.cuda.current_device())
        self.native_policy = native_policy.to(self.device).eval()
        self._original_noise_scheduler = self.native_policy.noise_scheduler
        self._ddim_scheduler = None
        self.block_keypoints = (original_block_keypoints() if block_keypoints is None else
                                np.asarray(block_keypoints, dtype=np.float64).copy())
        state_to_keypoints(np.zeros(10), self.block_keypoints)
        self.metadata = dict(metadata or {})
        self.metadata.update(
            original_sampler="ddpm",
            original_scheduler=type(self._original_noise_scheduler).__name__,
            observation_encoding="nine_original_T_keypoints_xy_then_agent_xy",
            original_prediction_horizon=self.config.original_horizon,
            historical_action_prefix=self.config.action_offset,
            future_prediction_horizon=self.config.prediction_horizon,
            action_bounds=[0.0, 512.0],
            action_clipping="Native physical XY targets are clipped to the common simulator action space.")
        self.reset(seed)

    def _seeded_rng_states(self, seed):
        cpu = torch.Generator(device="cpu").manual_seed(int(seed)).get_state()
        cuda = None
        if self.device.type == "cuda":
            cuda = torch.Generator(device=self.device).manual_seed(int(seed)).get_state()
        return cpu, cuda

    def reset(self, seed=None):
        self.history = deque(maxlen=self.config.obs_horizon)
        self.pending_actions = deque()
        self._cpu_rng_state, self._cuda_rng_state = self._seeded_rng_states(42 if seed is None else seed)
        self.native_policy.reset()
        self.last_plan = np.empty((0, 2), dtype=np.float32)
        self.last_sampling_seconds = 0.0
        self.last_replanned = False
        self.last_clipped_coordinates = 0

    @torch.no_grad()
    def predict_action(self, history, seed=None, inference_steps=None):
        history = np.asarray(history, dtype=np.float64)
        if history.ndim == 1:
            history = history[None]
        if history.ndim != 2 or history.shape[1] != 10 or not len(history) or not np.isfinite(history).all():
            raise ValueError("history must contain finite raw states with shape (T,10), T >= 1")
        history = history[-self.config.obs_horizon:]
        if len(history) < self.config.obs_horizon:
            history = np.concatenate([np.repeat(history[:1], self.config.obs_horizon - len(history), axis=0), history])
        steps = self.config.inference_steps if inference_steps is None else inference_steps
        _validate_sampling(self.config.sampler, steps, self.config.diffusion_steps)
        if self.config.sampler == "ddim" and self._ddim_scheduler is None:
            self._ddim_scheduler = make_ddim_scheduler(self._original_noise_scheduler)
        observations = torch.as_tensor(state_to_keypoints(history, self.block_keypoints)[None],
                                        device=self.device)
        observation_dict = {"obs": observations, "obs_mask": torch.ones_like(observations, dtype=torch.bool)}
        cpu_rng, cuda_rng = ((self._cpu_rng_state, self._cuda_rng_state) if seed is None else
                             self._seeded_rng_states(seed))
        devices = [self.device.index] if self.device.type == "cuda" else []
        old_steps = self.native_policy.num_inference_steps
        old_scheduler = self.native_policy.noise_scheduler
        old_kwargs = self.native_policy.kwargs
        try:
            self.native_policy.num_inference_steps = steps
            self.native_policy.noise_scheduler = (self._ddim_scheduler if self.config.sampler == "ddim"
                                                   else self._original_noise_scheduler)
            if self.config.sampler == "ddim":
                # A deterministic DDIM transition still starts from random noise;
                # keep that noise in the adapter's existing isolated RNG stream.
                self.native_policy.kwargs = dict(old_kwargs, eta=0.0)
            self.native_policy.eval()
            # The original predict_action does not pass a Generator through to
            # its DDPM scheduler. Preserve caller RNG and advance only our stream.
            with torch.random.fork_rng(devices=devices):
                torch.set_rng_state(cpu_rng)
                if devices:
                    torch.cuda.set_rng_state(cuda_rng, device=self.device)
                result = self.native_policy.predict_action(observation_dict)
                if seed is None:
                    self._cpu_rng_state = torch.get_rng_state().clone()
                    if devices:
                        self._cuda_rng_state = torch.cuda.get_rng_state(self.device).clone()
        finally:
            self.native_policy.num_inference_steps = old_steps
            self.native_policy.noise_scheduler = old_scheduler
            self.native_policy.kwargs = old_kwargs
        if not isinstance(result, dict) or "action_pred" not in result or "action" not in result:
            raise RuntimeError("Native policy must return action_pred and action tensors")
        prediction = result["action_pred"].detach().cpu().numpy()
        selected = result["action"].detach().cpu().numpy()
        if prediction.shape != (1, self.config.original_horizon, 2) or not np.isfinite(prediction).all():
            raise RuntimeError("Native action_pred has an invalid shape or nonfinite values")
        future = prediction[0, self.config.action_offset:]
        if (selected.shape != (1, self.config.max_action_horizon, 2)
                or not np.isfinite(selected).all()
                or not np.allclose(selected[0], future[:self.config.max_action_horizon], rtol=1e-6, atol=1e-5)):
            raise RuntimeError("Native executable actions disagree with the checkpoint's action alignment")
        self.last_clipped_coordinates = int(np.count_nonzero((future < 0) | (future > 512)))
        return np.clip(future, 0.0, 512.0).astype(np.float32)

    def act(self, state):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (10,) or not np.isfinite(state).all():
            raise ValueError("state must contain ten finite coordinates")
        self.history.append(state.copy())
        self.last_replanned = not self.pending_actions
        self.last_sampling_seconds = 0.0
        if self.last_replanned:
            started = perf_counter()
            self.last_plan = self.predict_action(np.asarray(self.history))
            self.last_sampling_seconds = perf_counter() - started
            self.pending_actions.extend(self.last_plan[:self.config.action_horizon].copy())
        return self.pending_actions.popleft().copy()


def _supported_config(cfg):
    """Validate the original policy/observation contract before constructing it."""
    policy = cfg["policy"]
    runner = cfg["task"]["env_runner"]
    if policy.get("_target_") != "diffusion_policy.policy.diffusion_unet_lowdim_policy.DiffusionUnetLowdimPolicy":
        raise ValueError("Only the original low-dimensional DiffusionUnetLowdimPolicy checkpoint is supported")
    if runner.get("_target_") != "diffusion_policy.env_runner.pusht_keypoints_runner.PushTKeypointsRunner":
        raise ValueError("Checkpoint must use the original PushTKeypointsRunner")
    if (policy.get("obs_dim") != 20 or policy.get("action_dim") != 2
            or runner.get("agent_keypoints", False)
            or runner.get("keypoint_visible_rate", 1.0) != 1.0
            or runner.get("past_action", False) or runner.get("n_latency_steps", 0) != 0):
        raise ValueError("Expected fully visible 9 T keypoints + agent XY, without past-action or latency conditioning")
    if policy.get("pred_action_steps_only", False) or not policy.get("oa_step_convention", False):
        raise ValueError("Checkpoint must predict full trajectories with oa_step_convention=True")
    if policy["noise_scheduler"].get("_target_") != "diffusers.schedulers.scheduling_ddpm.DDPMScheduler":
        raise ValueError("The supported original checkpoint uses a DDPM scheduler")
    obs_horizon, original_horizon = policy["n_obs_steps"], policy["horizon"]
    action_horizon = policy["n_action_steps"]
    if (runner.get("n_obs_steps", obs_horizon) != obs_horizon
            or runner.get("n_action_steps", action_horizon) != action_horizon):
        raise ValueError("Checkpoint runner and policy observation/action horizons disagree")
    diffusion_steps = policy["noise_scheduler"]["num_train_timesteps"]
    return PretrainedPolicyConfig(
        obs_horizon=obs_horizon, original_horizon=original_horizon,
        action_offset=obs_horizon-1, prediction_horizon=original_horizon-obs_horizon+1,
        action_horizon=action_horizon, max_action_horizon=action_horizon,
        inference_steps=policy.get("num_inference_steps") or diffusion_steps,
        diffusion_steps=diffusion_steps)


def load_policy(path, device="cpu"):
    """Load trusted original .ckpt weights/config on CPU, then select the device.

    Original checkpoints contain Dill/OmegaConf metadata. Load only a checkpoint
    obtained from a trusted source, such as the official published experiment.
    The project's self-built .pt checkpoints use a different loader/schema.
    """
    _reference_repository()
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained Push-T checkpoint not found: {path}")
    try:
        import dill
        import hydra
        from omegaconf import OmegaConf
    except ImportError as error:
        raise ImportError("Original checkpoint dependencies are missing; activate the robodiff environment") from error
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    kwargs = {"map_location": "cpu", "pickle_module": dill}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    payload = torch.load(path, **kwargs)
    if not isinstance(payload, dict) or "cfg" not in payload or "state_dicts" not in payload:
        raise ValueError("Expected an original Diffusion Policy checkpoint with cfg and state_dicts")
    cfg = payload["cfg"]
    plain_config = OmegaConf.to_container(cfg, resolve=True)
    config = _supported_config(plain_config)
    native = hydra.utils.instantiate(cfg.policy)
    weight_key = "ema_model" if bool(cfg.training.get("use_ema", False)) else "model"
    if weight_key not in payload["state_dicts"]:
        raise ValueError(f"Checkpoint is missing the requested {weight_key} weights")
    native.load_state_dict(payload["state_dicts"][weight_key], strict=True)
    metadata = {"checkpoint": str(path.resolve()), "weights": weight_key,
                "architecture": plain_config["policy"]["_target_"],
                "original_runner_config": plain_config["task"]["env_runner"],
                "original_config": plain_config,
                "training_seed": plain_config.get("training", {}).get("seed"),
                "note": "Original pretrained model; the shared runner defines evaluation reset and coverage threshold."}
    del payload
    return PretrainedDiffusionPolicy(native, config=config, metadata=metadata, device=device)
