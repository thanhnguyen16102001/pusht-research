"""Prepare episode-aware Push-T transitions and train-only normalization.

Caches retain raw episodes for recursive model evaluation. Recordings sharing the
same initial physical pose are assigned to the same split, including recordings
from different source datasets. No input Zarr store is modified.
"""

import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np
import zarr

try:
    from .episode_kinematics import load_episode_kinematics
except ImportError:
    from episode_kinematics import load_episode_kinematics

OUTPUT_FILE = Path("data/processed/demo_actions_v1.pkl")
POSITION_COLUMNS = [0, 1, 4, 5, 8]


def dataset_signature(source):
    """SHA-256 over recorded state/action arrays and exclusive episode ends."""
    root = zarr.open_group(str(source), mode="r")
    digest = hashlib.sha256()
    for key in ("state", "action"):
        digest.update(np.asarray(root["data"][key][:]).tobytes())
    digest.update(np.asarray(root["meta"]["episode_ends"][:]).tobytes())
    return digest.hexdigest()


def _initial_group(state):
    pose = np.asarray(state[POSITION_COLUMNS], dtype=np.float64).copy()
    pose[-1] = np.arctan2(np.sin(pose[-1]), np.cos(pose[-1]))
    return hashlib.sha256(np.round(pose, 4).tobytes()).hexdigest()[:16]


def prepare_data(datasets, control_hz=10.0, val_ratio=0.1, seed=42):
    """Return an in-memory cache; split whole initial-pose groups, never steps."""
    if isinstance(datasets, (str, Path)):
        datasets = [datasets]
    if not datasets or not np.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("Provide datasets and a positive finite control frequency")
    if not np.isfinite(val_ratio) or not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be strictly between zero and one")

    sources = list(dict.fromkeys(str(Path(p).expanduser().resolve()) for p in datasets))
    states, actions, episodes, signatures = [], [], [], {}
    seen_datasets = set()
    offset = 0
    for source in sources:
        root = zarr.open_group(source, mode="r")
        # Detect exact copied datasets before they can appear in both splits.
        signature = dataset_signature(source)
        if signature in seen_datasets:
            continue
        seen_datasets.add(signature)
        signatures[source] = signature
        for episode_id in range(len(root["meta"]["episode_ends"])):
            episode = load_episode_kinematics(source, episode_id, control_hz)
            state = episode.state_augmented.astype(np.float32)
            action = np.column_stack([episode.action_target.x,
                                      episode.action_target.y]).astype(np.float32)
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError(f"Nonfinite values in {source}, episode {episode_id}")
            states.append(state)
            actions.append(action)
            episodes.append({"source": source, "id": int(episode_id),
                             "start": offset, "stop": offset + len(state),
                             "initial_group": _initial_group(state[0]),
                             "dataset_signature": signature})
            offset += len(state)

    groups = sorted({ep["initial_group"] for ep in episodes if ep["stop"]-ep["start"] >= 2})
    if len(groups) < 2:
        raise ValueError("At least two distinct initial-pose groups with transitions are required")
    rng = np.random.default_rng(seed)
    n_val = min(len(groups)-1, max(1, round(len(groups)*val_ratio)))
    val_groups = set(rng.permutation(groups)[:n_val])
    parts = {"train": [[], []], "val": [[], []]}
    for record, state, action in zip(episodes, states, actions):
        split = "val" if record["initial_group"] in val_groups else "train"
        record["split"] = split
        if len(state) < 2:
            continue
        x = np.column_stack([state[:-1], action[:-1]])
        y = state[1:, POSITION_COLUMNS]-state[:-1, POSITION_COLUMNS]
        y[:, 4] = np.arctan2(np.sin(y[:, 4]), np.cos(y[:, 4]))
        parts[split][0].append(x)
        parts[split][1].append(y)
    if not all(parts[split][0] for split in parts):
        raise ValueError("Both splits must contain consecutive transition pairs")
    x_train, y_train = [np.concatenate(arrays) for arrays in parts["train"]]
    if len(x_train) < 2:
        raise ValueError("Training requires at least two transition samples")
    stats = {"x_mean": x_train.mean(0),
             "x_std": np.maximum(x_train.std(0, ddof=1), 1e-6),
             "y_mean": y_train.mean(0),
             "y_std": np.maximum(y_train.std(0, ddof=1), 1e-6),
             "dt": float(1.0/control_hz)}
    raw_state, raw_action = np.concatenate(states), np.concatenate(actions)
    cache = {"schema_version": 2, "episodes": episodes,
             "raw": {"state": raw_state, "action": raw_action},
             "normalized": {"state": (raw_state-stats["x_mean"][:10])/stats["x_std"][:10],
                            "action": (raw_action-stats["x_mean"][10:])/stats["x_std"][10:]},
             "stats": stats,
             "config": {"control_hz": float(control_hz), "val_ratio": float(val_ratio),
                        "seed": int(seed), "split_strategy": "initial_pose_groups",
                        "sources": list(signatures), "dataset_signatures": signatures}}
    for split in parts:
        x, y = [np.concatenate(arrays) for arrays in parts[split]]
        cache[split] = {"x": (x-stats["x_mean"])/stats["x_std"],
                        "y": (y-stats["y_mean"])/stats["y_std"]}
    return cache


def preprocess_data(datasets, output=OUTPUT_FILE, control_hz=10.0, val_ratio=0.1, seed=42):
    """Prepare and save a trusted pickle cache; existing input data stays unchanged."""
    cache = prepare_data(datasets, control_hz=control_hz, val_ratio=val_ratio, seed=seed)
    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        pickle.dump(cache, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved: {output} | Episodes: {len(cache['episodes'])} | States: {len(cache['raw']['state'])}")
    print(f"Train: {len(cache['train']['x'])} pairs | Val: {len(cache['val']['x'])} pairs")
    return cache


def load_data(path=OUTPUT_FILE):
    """Load a trusted cache produced by this project, including legacy caches."""
    with Path(path).expanduser().open("rb") as file:
        return pickle.load(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Push-T world-model transitions")
    parser.add_argument("--datasets", nargs="+", default=["data/demonstrations/demo_actions_v1.zarr"])
    parser.add_argument("--output", default=str(OUTPUT_FILE))
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    preprocess_data(args.datasets, args.output, args.control_hz, args.val_ratio, args.seed)
