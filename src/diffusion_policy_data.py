"""Causal action-sequence data built on the existing one-step preprocessing.

The shared cache owns episode boundaries, backwards-difference velocities,
initial-pose-group splits, and content deduplication. This module changes the
learning target to recorded actions, including an episode's last action. It
never reads images or changes a Zarr store.
"""

import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .preprocess_onestep_data import load_data, prepare_data
except ImportError:
    from preprocess_onestep_data import load_data, prepare_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIRECTORY = PROJECT_ROOT / "data" / "demonstrations"


def discover_zarr_datasets(inputs=None):
    """Find all ``*.zarr`` stores under paths/directories/globs, deterministically.

    ``None`` recursively searches this project's ``data/demonstrations``.
    Explicit paths can name a directory store or a Zarr archive. Resolved-path
    duplicates are removed here; ``prepare_data`` also removes identical store
    contents. An explicit missing path/glob or an empty search is an error.
    """
    if inputs is None:
        inputs = [DEFAULT_DATA_DIRECTORY]
    elif isinstance(inputs, (str, Path)):
        inputs = [inputs]
    stores = set()
    for item in inputs:
        expression = str(Path(item).expanduser())
        matches = glob.glob(expression, recursive=True) if glob.has_magic(expression) else [expression]
        if not matches:
            raise FileNotFoundError(f"No paths match: {item}")
        for match in matches:
            path = Path(match)
            if not path.exists():
                raise FileNotFoundError(f"Dataset path does not exist: {path}")
            if path.name.endswith(".zarr"):
                stores.add(path.resolve())
            elif path.is_dir():
                stores.update(p.resolve() for p in path.rglob("*.zarr") if p.exists())
            else:
                raise ValueError(f"Expected a .zarr store or directory: {path}")
    if not stores:
        raise ValueError("No .zarr datasets found; provide a directory or dataset paths")
    return sorted(stores, key=str)


def _validate_cache(cache):
    """Check the shared episode schema without trusting cached normalization."""
    if not isinstance(cache, dict) or cache.get("schema_version") != 2:
        raise ValueError("Diffusion Policy requires a schema_version=2 preprocessing cache with raw episodes")
    try:
        state = np.asarray(cache["raw"]["state"])
        action = np.asarray(cache["raw"]["action"])
        records = cache["episodes"]
        dt = float(cache["stats"]["dt"])
        control_hz = float(cache["config"]["control_hz"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Cache must include raw state/action, episodes, stats.dt and config.control_hz") from error
    if state.ndim != 2 or state.shape[1] != 10 or action.shape != (len(state), 2) or len(state) < 2:
        raise ValueError("Cache raw arrays must have shape (N,10) and (N,2), N >= 2")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("Cache contains nonfinite state/action values")
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("Cache control frequency and dt must be positive and finite")
    if not np.isclose(dt * control_hz, 1.0):
        raise ValueError("Cache dt and control_hz disagree")
    # Original mouse recordings contain occasional coordinates 513 on a
    # nominal [0,512] workspace. Preserve these labels exactly; inference clips
    # the sampled physical action. Larger overshoots indicate another schema.
    if np.any(action < -1.0) or np.any(action > 513.0):
        raise ValueError("Actions must be absolute XY targets in [0,512], allowing 1-unit recorder overshoot")
    groups, splits, previous_stop = {}, set(), 0
    if not records:
        raise ValueError("Cache contains no episodes")
    for record in records:
        try:
            start, stop = record["start"], record["stop"]
            split, group = record["split"], record["initial_group"]
        except (KeyError, TypeError) as error:
            raise ValueError("Each cache episode needs start, stop, split, initial_group") from error
        if not isinstance(start, (int, np.integer)) or not isinstance(stop, (int, np.integer)):
            raise ValueError("Episode boundaries must be integer indices")
        if start != previous_stop or not start < stop <= len(state):
            raise ValueError("Cache episodes must be nonempty contiguous partitions of the raw arrays")
        if split not in {"train", "val"} or not isinstance(group, str) or not group:
            raise ValueError("Each episode requires train/val split and a nonempty initial_group")
        if group in groups and groups[group] != split:
            raise ValueError("Initial-pose group appears in both train and validation")
        groups[group] = split
        splits.add(split)
        previous_stop = stop
    if previous_stop != len(state) or splits != {"train", "val"}:
        raise ValueError("Cache must partition all states into nonempty train and validation episodes")
    return cache


def prepare_policy_data(datasets=None, control_hz=10.0, val_ratio=0.1, seed=42, data_pkl=None):
    """Reuse ``preprocess_onestep_data.prepare_data`` or its trusted saved cache.

    Return the unchanged schema-2 cache, preserving its split/provenance. When
    loading a cache, its split parameters take precedence over ``seed`` and
    ``val_ratio``; the requested control frequency must still match. Pickle
    caches must come from a trusted local preprocessing run.
    """
    if not np.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("control_hz must be finite and positive")
    if data_pkl is not None:
        if datasets is not None:
            raise ValueError("Provide datasets or data_pkl, not both")
        cache = load_data(data_pkl)
    else:
        cache = prepare_data(discover_zarr_datasets(datasets), control_hz=control_hz,
                             val_ratio=val_ratio, seed=seed)
    _validate_cache(cache)
    if not np.isclose(float(cache["stats"]["dt"]), 1.0 / control_hz):
        raise ValueError("Requested control_hz differs from the preprocessing cache")
    return cache


def encode_states(states):
    """Encode raw ``(...,10)`` states as 11 features using sin/cos of T angle.

    Feature order: agent XY/VXY, block XY/VXY, sin(theta), cos(theta), omega.
    Only the angle representation changes; velocities remain causal.
    """
    states = np.asarray(states, dtype=np.float32)
    if states.ndim < 1 or states.shape[-1] != 10 or not np.isfinite(states).all():
        raise ValueError("states must contain finite raw 10D state vectors")
    return np.concatenate((states[..., :8], np.sin(states[..., 8:9]),
                           np.cos(states[..., 8:9]), states[..., 9:10]), axis=-1)


def make_policy_normalizer(cache):
    """Train-only observation statistics and fixed workspace action scaling.

    All train states, including terminal recorded states, contribute once.
    Action scaling is ``a / 256 - 1`` from the known [0,512] workspace; recorded
    513-valued targets are retained as 1.00390625. Metadata makes those small
    boundary overshoots visible. No validation statistics enter normalization.
    """
    _validate_cache(cache)
    state = cache["raw"]["state"]
    train_states = np.concatenate([state[ep["start"]:ep["stop"]]
                                   for ep in cache["episodes"] if ep["split"] == "train"])
    if len(train_states) < 2:
        raise ValueError("Observation normalization requires at least two recorded train states")
    features = encode_states(train_states).astype(np.float64)
    actions = np.asarray(cache["raw"]["action"])
    return {
        "obs_mean": features.mean(axis=0).astype(np.float32).tolist(),
        "obs_std": np.maximum(features.std(axis=0, ddof=1), 1e-6).astype(np.float32).tolist(),
        "action_low": [0.0, 0.0], "action_high": [512.0, 512.0],
        "dt": float(cache["stats"]["dt"]),
        "action_out_of_bounds_count": int(np.any((actions < 0) | (actions > 512), axis=1).sum()),
        "recorded_action_min": actions.min(axis=0).astype(float).tolist(),
        "recorded_action_max": actions.max(axis=0).astype(float).tolist(),
    }


class PolicySequenceDataset(Dataset):
    """Causal history and future demonstration actions within one episode.

    ``indices[i]`` is ``(cache_episode_index, local_current_step)``. For current
    step t, observations cover [t-O+1,t], repeating the episode's first state
    when history is short. Targets cover actions [t,t+H-1], repeating the last
    action only for storage; ``mask`` excludes that future padding from loss.
    Every actual recorded action is a target, including the episode's last one.
    Items are torch tensors: obs (O,11), action (H,2), mask (H,) bool.
    """

    def __init__(self, cache, split, obs_horizon=2, prediction_horizon=16, normalizer=None):
        _validate_cache(cache)
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        for value, name in ((obs_horizon, "obs_horizon"), (prediction_horizon, "prediction_horizon")):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        normalizer = make_policy_normalizer(cache) if normalizer is None else normalizer
        try:
            mean = np.asarray(normalizer["obs_mean"], dtype=np.float32)
            std = np.asarray(normalizer["obs_std"], dtype=np.float32)
            low = np.asarray(normalizer["action_low"], dtype=np.float32)
            high = np.asarray(normalizer["action_high"], dtype=np.float32)
            dt = float(normalizer["dt"])
        except (KeyError, ValueError, TypeError) as error:
            raise ValueError("Invalid policy normalizer") from error
        if (mean.shape != (11,) or std.shape != (11,) or low.shape != (2,) or high.shape != (2,)
                or not all(np.isfinite(x).all() for x in (mean, std, low, high))
                or np.any(std <= 0) or np.any(high <= low) or not np.isfinite(dt)
                or not np.isclose(dt, cache["stats"]["dt"])):
            raise ValueError("Policy normalizer has invalid dimensions, scales, or control period")
        self.cache, self.split = cache, split
        self.normalizer = normalizer
        self.obs_horizon, self.prediction_horizon = int(obs_horizon), int(prediction_horizon)
        self.observations = (encode_states(cache["raw"]["state"]) - mean) / std
        self.actions = 2.0 * (np.asarray(cache["raw"]["action"], dtype=np.float32) - low) / (high - low) - 1.0
        self.indices = [(episode_index, local_step)
                        for episode_index, ep in enumerate(cache["episodes"]) if ep["split"] == split
                        for local_step in range(ep["stop"] - ep["start"])]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        episode_index, step = self.indices[index]
        episode = self.cache["episodes"][episode_index]
        start, stop = episode["start"], episode["stop"]
        observation_indices = start + np.maximum(0, step + np.arange(1-self.obs_horizon, 1))
        action_indices = start + step + np.arange(self.prediction_horizon)
        mask = action_indices < stop
        action_indices = np.minimum(action_indices, stop-1)
        return {
            "obs": torch.from_numpy(self.observations[observation_indices].copy()),
            "action": torch.from_numpy(self.actions[action_indices].copy()),
            "mask": torch.from_numpy(mask.copy()),
        }
