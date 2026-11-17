"""State-conditioned diffusion imitation policy, implemented with PyTorch only.

Train: python -B src/Diffusion_Policy_model.py --epochs 100
Run:   python -B src/run_diffusion_policy.py --render --continuous

The policy learns absolute actuator XY targets, not next-state dynamics or MPC
costs. Observation history ends at t; predicted actions start at t. The original
demonstrations have a fixed goal. See docs/diffusion_policy.md for conventions.
"""

import argparse
from collections import deque
import copy
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from .diffusion_policy_data import (PolicySequenceDataset, encode_states,
                                        make_policy_normalizer, prepare_policy_data)
    from .pusht_runtime import PROJECT_ROOT
    from .run_mpc import write_json
except ImportError:
    from diffusion_policy_data import (PolicySequenceDataset, encode_states,
                                       make_policy_normalizer, prepare_policy_data)
    from pusht_runtime import PROJECT_ROOT
    from run_mpc import write_json


@dataclass
class DiffusionPolicyConfig:
    obs_horizon: int = 2
    prediction_horizon: int = 16
    action_horizon: int = 8
    diffusion_steps: int = 100
    inference_steps: int = 20
    channels: int = 64
    residual_blocks: int = 4
    time_embed_dim: int = 64

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.diffusion_steps < 2:
            raise ValueError("diffusion_steps must be >= 2")
        if not 1 <= self.inference_steps <= self.diffusion_steps:
            raise ValueError("inference_steps must be in [1, diffusion_steps]")
        if self.action_horizon > self.prediction_horizon:
            raise ValueError("action_horizon must not exceed prediction_horizon")
        if self.channels < 2:
            raise ValueError("channels must be >= 2")
        if self.time_embed_dim < 4 or self.time_embed_dim % 2:
            raise ValueError("time_embed_dim must be even and >= 4")


class ConditionalResidualBlock(nn.Module):
    """Dilated temporal convolution modulated by observation/time FiLM."""

    def __init__(self, channels, condition_dim, dilation):
        super().__init__()
        # Keep at least two channels per group even for a one-action horizon.
        groups = max(g for g in range(1, min(8, channels // 2) + 1) if channels % g == 0)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.film = nn.Linear(condition_dim, 2 * channels)
        self.activation = nn.SiLU()

    def forward(self, x, condition):
        scale, shift = self.film(condition).unsqueeze(-1).chunk(2, dim=1)
        h = self.norm1(self.conv1(x))
        h = self.activation(h * (1 + scale) + shift)
        h = self.activation(self.norm2(self.conv2(h)))
        return (x + h) / math.sqrt(2)


class ConditionalTemporalDenoiser(nn.Module):
    """Compact temporal residual network; this is not the original U-Net."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.time_embed_dim
        frequencies = torch.exp(-math.log(10000) * torch.arange(dim // 2) / (dim // 2 - 1))
        self.register_buffer("frequencies", frequencies)
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))
        condition_dim = dim * 2
        self.condition = nn.Sequential(nn.Linear(dim + config.obs_horizon * 11, condition_dim),
                                       nn.SiLU(), nn.Linear(condition_dim, condition_dim))
        self.input = nn.Conv1d(2, config.channels, 1)
        self.blocks = nn.ModuleList([
            ConditionalResidualBlock(config.channels, condition_dim, 2 ** (i % 4))
            for i in range(config.residual_blocks)])
        self.output = nn.Sequential(nn.SiLU(), nn.Conv1d(config.channels, 2, 1))

    def forward(self, noisy_actions, timestep, observations):
        if noisy_actions.ndim != 3 or noisy_actions.shape[1:] != (self.config.prediction_horizon, 2):
            raise ValueError("noisy_actions must have shape (B, prediction_horizon, 2)")
        if observations.shape != (len(noisy_actions), self.config.obs_horizon, 11):
            raise ValueError("observations must have shape (B, obs_horizon, 11)")
        timestep = torch.as_tensor(timestep, device=noisy_actions.device).reshape(-1)
        timestep = timestep.expand(len(noisy_actions))
        phase = timestep.float()[:, None] * self.frequencies[None]
        time_embedding = self.time_mlp(torch.cat([phase.sin(), phase.cos()], dim=-1))
        condition = self.condition(torch.cat([observations.flatten(1), time_embedding], dim=-1))
        x = self.input(noisy_actions.transpose(1, 2))
        for block in self.blocks:
            x = block(x, condition)
        return self.output(x).transpose(1, 2)


class DiffusionSchedule(nn.Module):
    """Cosine DDPM forward process and deterministic DDIM reverse process."""

    def __init__(self, steps=100):
        super().__init__()
        if not isinstance(steps, int) or steps < 2:
            raise ValueError("steps must be an integer >= 2")
        grid = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
        cumulative = torch.cos(((grid / steps + 0.008) / 1.008) * math.pi / 2).square()
        cumulative = cumulative / cumulative[0]
        betas = (1 - cumulative[1:] / cumulative[:-1]).clamp(1e-8, 0.999)
        self.register_buffer("alpha_bar", (1 - betas).cumprod(0).float())
        self.steps = steps

    def add_noise(self, clean, noise, timesteps):
        alpha = self.alpha_bar[timesteps].reshape(-1, 1, 1)
        return alpha.sqrt() * clean + (1 - alpha).sqrt() * noise

    def inference_timesteps(self, inference_steps):
        if not isinstance(inference_steps, int) or not 1 <= inference_steps <= self.steps:
            raise ValueError("inference_steps must be in [1, diffusion_steps]")
        return np.linspace(self.steps - 1, 0, inference_steps).round().astype(int).tolist()

    def ddim_step(self, sample, epsilon, timestep, previous_timestep):
        """Use the next *selected* timestep, with alpha_bar(-1) = 1."""
        alpha = self.alpha_bar[timestep]
        previous = self.alpha_bar[previous_timestep] if previous_timestep >= 0 else alpha.new_tensor(1.)
        clean = ((sample - (1 - alpha).sqrt() * epsilon) / alpha.sqrt()).clamp(-1, 1)
        # Recompute noise after clipping so the DDIM direction remains consistent.
        epsilon = (sample - alpha.sqrt() * clean) / (1 - alpha).sqrt()
        return previous.sqrt() * clean + (1 - previous).sqrt() * epsilon


def masked_noise_loss(predicted, noise, mask):
    """Mean epsilon MSE over real demonstration actions, excluding tail padding."""
    if predicted.shape != noise.shape or mask.shape != predicted.shape[:-1]:
        raise ValueError("Noise/mask shapes disagree")
    weights = mask.to(predicted.dtype).unsqueeze(-1)
    count = weights.sum() * predicted.shape[-1]
    if count.item() == 0:
        raise ValueError("A training batch must contain valid actions")
    return ((predicted - noise).square() * weights).sum() / count


class DiffusionPolicy:
    """Inference wrapper for raw 10D states and bounded physical XY actions.

    predict_action(history) samples a full H-action plan without changing the
    execution queue. act(state) records every new observation and executes A
    actions per sampled plan. reset(seed) starts an independent episode.
    """

    def __init__(self, model, normalizer, config=None, device="cpu", seed=42):
        self.config = config or model.config
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.schedule = DiffusionSchedule(self.config.diffusion_steps).to(self.device)
        self.normalizer = copy.deepcopy(normalizer)
        self.dt = float(normalizer["dt"])
        self.obs_mean = np.asarray(normalizer["obs_mean"], dtype=np.float32)
        self.obs_std = np.asarray(normalizer["obs_std"], dtype=np.float32)
        self.action_low = np.asarray(normalizer["action_low"], dtype=np.float32)
        self.action_high = np.asarray(normalizer["action_high"], dtype=np.float32)
        if (self.obs_mean.shape != (11,) or self.obs_std.shape != (11,)
                or self.action_low.shape != (2,) or self.action_high.shape != (2,)
                or not np.isfinite(np.r_[self.obs_mean, self.obs_std, self.action_low,
                                          self.action_high, self.dt]).all()
                or np.any(self.obs_std <= 0) or np.any(self.action_high <= self.action_low)
                or self.dt <= 0):
            raise ValueError("Invalid policy normalization")
        self.reset(seed)

    def reset(self, seed=None):
        self.history = deque(maxlen=self.config.obs_horizon)
        self.pending_actions = deque()
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(42 if seed is None else int(seed))
        self.last_plan = np.empty((0, 2), dtype=np.float32)
        self.last_sampling_seconds = 0.
        self.last_replanned = False

    @torch.no_grad()
    def predict_action(self, history, seed=None, inference_steps=None):
        history = np.asarray(history, dtype=np.float32)
        if history.ndim == 1:
            history = history[None]
        if history.ndim != 2 or history.shape[1] != 10 or len(history) == 0 or not np.isfinite(history).all():
            raise ValueError("history must contain finite states with shape (T, 10), T >= 1")
        history = history[-self.config.obs_horizon:]
        if len(history) < self.config.obs_horizon:
            history = np.concatenate([np.repeat(history[:1], self.config.obs_horizon - len(history), axis=0), history])
        normalized = (encode_states(history) - self.obs_mean) / self.obs_std
        observations = torch.as_tensor(normalized[None], device=self.device)
        generator = self.generator if seed is None else torch.Generator(device=self.device).manual_seed(int(seed))
        sample = torch.randn((1, self.config.prediction_horizon, 2), device=self.device, generator=generator)
        timesteps = self.schedule.inference_timesteps(
            self.config.inference_steps if inference_steps is None else inference_steps)
        self.model.eval()
        for index, timestep in enumerate(timesteps):
            epsilon = self.model(sample, timestep, observations)
            previous = timesteps[index + 1] if index + 1 < len(timesteps) else -1
            sample = self.schedule.ddim_step(sample, epsilon, timestep, previous)
        normalized_actions = sample[0].cpu().numpy()
        actions = self.action_low + (normalized_actions + 1) * .5 * (self.action_high - self.action_low)
        if not np.isfinite(actions).all():
            raise RuntimeError("Diffusion policy generated nonfinite actions")
        return np.clip(actions, 0., 512.).astype(np.float32)

    def act(self, state):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (10,) or not np.isfinite(state).all():
            raise ValueError("state must contain 10 finite coordinates")
        self.history.append(state.copy())
        self.last_replanned = not self.pending_actions
        self.last_sampling_seconds = 0.
        if self.last_replanned:
            started = perf_counter()
            self.last_plan = self.predict_action(np.asarray(self.history))
            self.last_sampling_seconds = perf_counter() - started
            self.pending_actions.extend(self.last_plan[:self.config.action_horizon].copy())
        return self.pending_actions.popleft().copy()


def save_policy(policy, path, metadata=None):
    """Portable CPU weights plus architecture, normalizer and data provenance."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "architecture": "conditional_temporal_diffusion_v1",
               "config": asdict(policy.config), "normalizer": policy.normalizer,
               "model_state": {key: value.detach().cpu().clone()
                               for key, value in policy.model.state_dict().items()},
               "metadata": metadata or {}}
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_policy(path, device="cpu"):
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1 or payload.get("architecture") != "conditional_temporal_diffusion_v1":
        raise ValueError("Unsupported diffusion-policy checkpoint")
    config = DiffusionPolicyConfig(**payload["config"])
    model = ConditionalTemporalDenoiser(config)
    model.load_state_dict(payload["model_state"], strict=True)
    policy = DiffusionPolicy(model, payload["normalizer"], config, device=device)
    policy.metadata = payload.get("metadata", {})
    return policy


def _epoch(model, schedule, loader, device, generator, optimizer=None, ema=None, ema_updates=0):
    training = optimizer is not None
    model.train(training)
    numerator, denominator = 0., 0.
    with torch.set_grad_enabled(training):
        for batch in loader:
            observations, actions, mask = [batch[key].to(device) for key in ("obs", "action", "mask")]
            noise = torch.randn(actions.shape, generator=generator, device=device)
            timesteps = torch.randint(schedule.steps, (len(actions),), generator=generator, device=device)
            noisy = schedule.add_noise(actions, noise, timesteps)
            loss = masked_noise_loss(model(noisy, timesteps, observations), noise, mask)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite diffusion loss; inspect data/learning rate")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                ema_updates += 1
                decay = min(.995, (1 + ema_updates) / (10 + ema_updates))
                with torch.no_grad():
                    for target, source in zip(ema.parameters(), model.parameters()):
                        target.lerp_(source, 1 - decay)
            count = int(mask.sum()) * actions.shape[-1]
            numerator += float(loss.detach()) * count
            denominator += count
    if denominator == 0:
        raise ValueError("Empty training/validation loader")
    return numerator / denominator, ema_updates


def train_policy(datasets=None, *, data_pkl=None, config=None, epochs=100, batch_size=256,
                 lr=1e-4, weight_decay=1e-6, control_hz=10., val_ratio=.1, seed=42,
                 device="cpu", torch_threads=1, resume=None, output=None,
                 last_output=None, history_path=None):
    """Train imitation diffusion; --resume fine-tunes EMA weights with new AdamW.

    All sequence targets are expert actions. prepare_policy_data delegates to
    preprocess_onestep_data; normalization is fit on train states only. Best
    checkpoint selection uses fixed-noise validation epsilon MSE of EMA weights.
    """
    if epochs < 1 or batch_size < 1 or torch_threads < 1:
        raise ValueError("epochs, batch_size and torch_threads must be positive")
    if not np.isfinite(lr) or lr <= 0 or not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("Invalid learning rate or weight decay")
    torch.set_num_threads(torch_threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device)
    output = Path(output or PROJECT_ROOT / "checkpoints/DiffusionPolicy_best.pt")
    last_output = Path(last_output or output.with_name(output.stem + "_last" + output.suffix))
    history_path = Path(history_path or PROJECT_ROOT / "experiments/diffusion_policy_training.json")
    if len({p.resolve() for p in (output, last_output, history_path)}) != 3:
        raise ValueError("Best, last and history output paths must differ")
    cache = prepare_policy_data(datasets, control_hz=control_hz, val_ratio=val_ratio,
                                seed=seed, data_pkl=data_pkl)
    if resume is not None:
        policy = load_policy(resume, device=device)
        if config is not None and config != policy.config:
            raise ValueError("Fine-tuning must retain the checkpoint architecture/config")
        if not np.isclose(policy.dt, cache["stats"]["dt"]):
            raise ValueError("Checkpoint and demonstration control frequencies disagree")
        previous_splits = {ep["initial_group"]: ep["split"]
                           for ep in policy.metadata.get("episodes", [])}
        if any(ep["initial_group"] in previous_splits
               and previous_splits[ep["initial_group"]] != ep["split"]
               for ep in cache["episodes"]):
            raise ValueError("Fine-tuning changes a previously seen initial-pose group's train/val split; "
                             "reuse the original preprocessing cache or preserve its split assignments")
        config, normalizer, model = policy.config, policy.normalizer, policy.model
    else:
        config = config or DiffusionPolicyConfig()
        normalizer = make_policy_normalizer(cache)
        model = ConditionalTemporalDenoiser(config).to(device)
    train_set = PolicySequenceDataset(cache, "train", config.obs_horizon, config.prediction_horizon, normalizer)
    val_set = PolicySequenceDataset(cache, "val", config.obs_horizon, config.prediction_horizon, normalizer)
    shuffle = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=shuffle, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    schedule = DiffusionSchedule(config.diffusion_steps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    noise_generator = torch.Generator(device=device).manual_seed(seed + 1)
    ema_policy = DiffusionPolicy(ema, normalizer, config, device=device)
    provenance = {"preprocessing": cache["config"], "episodes": cache["episodes"],
                  "train_samples": len(train_set), "val_samples": len(val_set),
                  "seed": seed, "batch_size": batch_size, "lr": lr,
                  "weight_decay": weight_decay, "resume": str(resume) if resume else None,
                  "selection_metric": "fixed_noise_validation_epsilon_mse_ema",
                  "goal_pose": [256., 256., math.pi / 4],
                  "action_convention": "absolute_xy_at_current_state",
                  "torch_version": str(torch.__version__)}
    history = {"config": asdict(config), "normalizer": normalizer, "metadata": provenance,
               "epochs": [], "best_epoch": None, "best_val_loss": None}
    print(f"Sources: {len(cache['config']['sources'])} | Episodes: {len(cache['episodes'])} | "
          f"Train: {len(train_set)} | Validation: {len(val_set)}", flush=True)
    best, updates = float("inf"), 0
    for epoch in range(1, epochs + 1):
        started = perf_counter()
        train_loss, updates = _epoch(model, schedule, train_loader, device, noise_generator,
                                     optimizer, ema, updates)
        val_generator = torch.Generator(device=device).manual_seed(seed + 2)
        val_loss, _ = _epoch(ema, schedule, val_loader, device, val_generator)
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                  "seconds": perf_counter() - started, "ema_updates": updates}
        history["epochs"].append(record)
        metadata = {**provenance, **record}
        if val_loss < best:
            best = val_loss
            history.update(best_epoch=epoch, best_val_loss=best)
            save_policy(ema_policy, output, metadata)
        save_policy(ema_policy, last_output, metadata)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(history_path, history)
        print(f"Epoch {epoch}/{epochs}: train={train_loss:.6f} val(EMA)={val_loss:.6f} "
              f"best={best:.6f} ({record['seconds']:.1f}s)", flush=True)
    return load_policy(output, device=device), history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", "--dataset", nargs="+", help="Zarr paths, directories or glob patterns; default: all data/demonstrations/**/*.zarr")
    parser.add_argument("--data-pkl", type=Path, help="Reuse a schema-2 cache from preprocess_onestep_data.py")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--control-hz", type=float, default=10.)
    parser.add_argument("--val-ratio", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--resume", type=Path, help="Fine-tune saved policy with its normalizer; new optimizer")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "checkpoints/DiffusionPolicy_best.pt")
    parser.add_argument("--last-output", type=Path, default=PROJECT_ROOT / "checkpoints/DiffusionPolicy_last.pt")
    parser.add_argument("--history", type=Path, default=PROJECT_ROOT / "experiments/diffusion_policy_training.json")
    for name, value in asdict(DiffusionPolicyConfig()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=None,
                            help=f"Default {value}; retained from checkpoint with --resume")
    args = parser.parse_args(argv)
    config_values = {name: getattr(args, name) for name in asdict(DiffusionPolicyConfig())
                     if getattr(args, name) is not None}
    try:
        if args.resume:
            existing = load_policy(args.resume, device="cpu").config
            if any(getattr(existing, key) != value for key, value in config_values.items()):
                parser.error("Architecture/sampling settings must match --resume checkpoint")
            config = None
        else:
            config = DiffusionPolicyConfig(**config_values)
        _, history = train_policy(args.datasets, data_pkl=args.data_pkl, config=config,
                                  epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                                  weight_decay=args.weight_decay, control_hz=args.control_hz,
                                  val_ratio=args.val_ratio, seed=args.seed, device=args.device,
                                  torch_threads=args.torch_threads, resume=args.resume,
                                  output=args.output, last_output=args.last_output, history_path=args.history)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(f"Saved best: {args.output} | epoch={history['best_epoch']} | "
          f"val epsilon MSE={history['best_val_loss']:.6f}")


if __name__ == "__main__":
    main()
