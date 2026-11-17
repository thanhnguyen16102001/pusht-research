"""Learned Push-T dynamics: state 10D + target action 2D -> next state 10D.

The backward-compatible MLP predicts five pose changes. Velocities are derived
from those changes using the recorded control period. Batch inference and
recursive rollouts stay in Torch for MPC. Checkpoints include split provenance.
"""

import copy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

MODEL_FILE = Path(__file__).with_suffix(".pkl")
PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_DIR / "checkpoints" / "WorldModel_best.pt"
POSITION_COLUMNS = [0, 1, 4, 5, 8]
_STAT_SHAPES = {"x_mean": (12,), "x_std": (12,), "y_mean": (5,), "y_std": (5,)}


def build_model(architecture="residual_mlp_v1", stats=None):
    """Legacy-compatible 12 -> 128 -> 128 -> 5 residual dynamics network."""
    if architecture == "contact_local_v2":
        if stats is None:
            raise ValueError("Contact dynamics requires normalization statistics")
        try:
            from .Model_ContactDynamics import ContactDynamicsModel
        except ImportError:
            from Model_ContactDynamics import ContactDynamicsModel
        return ContactDynamicsModel(stats)
    if architecture != "residual_mlp_v1":
        raise ValueError(f"Unknown model architecture: {architecture}")
    return nn.Sequential(nn.Linear(12, 128), nn.SiLU(),
                         nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 5))


def prepare_stats(stats, device="cpu", dtype=torch.float32):
    """Validate normalization once and place it on the inference device."""
    dt = float(stats["dt"])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("stats['dt'] must be positive and finite")
    result = {"dt": dt}
    for key, shape in _STAT_SHAPES.items():
        value = torch.as_tensor(stats[key], dtype=dtype, device=device)
        if tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Invalid normalization {key}: expected finite shape {shape}")
        if key.endswith("std") and not bool((value > 0).all()):
            raise ValueError(f"{key} must be strictly positive")
        result[key] = value
    if "metadata" in stats:
        result["metadata"] = stats["metadata"]
    return result


def _predict_batch_impl(model, stats, states, actions):
    """Unchecked tensor-only inner loop; caller validates dimensions and inputs."""
    x = torch.cat((states, actions), dim=-1)
    delta = model((x-stats["x_mean"])/stats["x_std"])
    delta = delta*stats["y_std"]+stats["y_mean"]
    position = states[:, POSITION_COLUMNS]+delta
    velocity = delta/stats["dt"]
    return torch.stack((position[:, 0], position[:, 1], velocity[:, 0], velocity[:, 1],
                        position[:, 2], position[:, 3], velocity[:, 2], velocity[:, 3],
                        position[:, 4], velocity[:, 4]), dim=-1)


@torch.no_grad()
def predict_batch(model, stats, states, actions):
    """Return next states (B,10) from states (B,10) and absolute XY actions (B,2)."""
    parameter = next(model.parameters())
    states = torch.as_tensor(states, dtype=parameter.dtype, device=parameter.device)
    actions = torch.as_tensor(actions, dtype=parameter.dtype, device=parameter.device)
    if states.ndim != 2 or states.shape[1] != 10 or actions.shape != (len(states), 2):
        raise ValueError("Expected states (B,10) and actions (B,2)")
    if len(states) == 0 or not bool(torch.isfinite(states).all() & torch.isfinite(actions).all()):
        raise ValueError("States and actions must be nonempty and finite")
    prepared = prepare_stats(stats, parameter.device, parameter.dtype)
    model.eval()
    return _predict_batch_impl(model, prepared, states, actions)


@torch.no_grad()
def rollout(model, stats, initial_states, actions):
    """Recursively predict (B,H+1,10), including initial states at index zero.

    initial_states: (10,), (1,10), or (B,10).
    actions: (B,H,2), or a single sequence (H,2). No true intermediate states
    are injected. Invalid model predictions remain nonfinite so a planner can
    reject individual candidates without losing the other trajectories.
    """
    parameter = next(model.parameters())
    actions = torch.as_tensor(actions, dtype=parameter.dtype, device=parameter.device)
    state = torch.as_tensor(initial_states, dtype=parameter.dtype, device=parameter.device)
    if actions.ndim == 2:
        actions = actions.unsqueeze(0)
    if actions.ndim != 3 or actions.shape[-1] != 2 or min(actions.shape[:2]) < 1:
        raise ValueError("Expected action sequences (B,H,2) with B,H >= 1")
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.ndim != 2 or state.shape[-1] != 10 or state.shape[0] not in (1, actions.shape[0]):
        raise ValueError("Expected initial states (10,), (1,10), or (B,10)")
    if not bool(torch.isfinite(state).all() & torch.isfinite(actions).all()):
        raise ValueError("Initial states and actions must be finite")
    state = state.expand(actions.shape[0], -1)
    prepared = prepare_stats(stats, parameter.device, parameter.dtype)
    model.eval()
    trajectory = [state]
    for index in range(actions.shape[1]):
        state = _predict_batch_impl(model, prepared, state, actions[:, index])
        trajectory.append(state)
    return torch.stack(trajectory, dim=1)


def predict(model, stats, state, action):
    """Legacy scalar API, returning a NumPy next state with shape (10,)."""
    state, action = np.asarray(state, dtype=np.float32), np.asarray(action, dtype=np.float32)
    if state.shape != (10,) or action.shape != (2,):
        raise ValueError("Expected state (10,) and action (2,)")
    return predict_batch(model, stats, state[None], action[None])[0].cpu().numpy()


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cpu_copy(item) for item in value]
    return copy.deepcopy(value)


def save_model(model, stats, path=MODEL_FILE):
    """Save portable weights/stats without moving the caller's model off device."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    prepare_stats(stats)
    checkpoint = {"schema_version": 2, "architecture": getattr(model, "architecture", "residual_mlp_v1"),
                  "weights": _cpu_copy(model.state_dict()), "stats": _cpu_copy(stats)}
    torch.save(checkpoint, path)
    print(f"Saved: {path}")


def load_model(path=MODEL_FILE, device="cpu"):
    """Load a trusted old/new checkpoint; old provenance remains explicitly unknown."""
    checkpoint = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    model = build_model(checkpoint.get("architecture", "residual_mlp_v1"), checkpoint["stats"])
    model.load_state_dict(checkpoint["weights"])
    model.to(device).eval()
    stats = prepare_stats(checkpoint["stats"], device)
    stats.setdefault("metadata", {"provenance": "unknown", "episodes": [],
                                   "training_episodes": [], "validation_is_heldout": False})
    return model, stats


def _training_metadata(cache, pretrained, prior, config):
    episodes = copy.deepcopy(cache.get("episodes", []))
    current_train = [ep for ep in episodes if ep.get("split") == "train"]
    previous_train = copy.deepcopy(prior.get("training_episodes", [])) if prior else []
    if prior and not previous_train:
        previous_train = [ep for ep in prior.get("episodes", []) if ep.get("split") == "train"]
    reproducible = bool(episodes) and all(ep.get("initial_group") and ep.get("dataset_signature")
                                         for ep in episodes)
    known = reproducible and (pretrained is None or prior.get("provenance") == "known")
    seen_ids = {(ep.get("source"), ep.get("id")) for ep in previous_train}
    seen_groups = {ep.get("initial_group") for ep in previous_train if ep.get("initial_group")}
    for ep in episodes:
        ep["seen_in_pretraining"] = ((ep.get("source"), ep.get("id")) in seen_ids
                                      or ep.get("initial_group") in seen_groups)
    heldout = known and all(not ep["seen_in_pretraining"] for ep in episodes if ep["split"] == "val")
    return {"provenance": "known" if known else "partial", "episodes": episodes,
            "training_episodes": previous_train+current_train,
            "validation_is_heldout": heldout,
            "pretrained_checkpoint": str(Path(pretrained).resolve()) if pretrained else None,
            "data_config": copy.deepcopy(cache.get("config", {})),
            "training_config": config, "history": [], "best_epoch": None}


def train_model(zarr_path=None, epochs=100, batch_size=256, lr=1e-3, control_hz=10.0,
                seed=42, pretrained=None, data_pkl=None, val_ratio=0.1, device=None,
                output=None, last_output=None, history_path=None, torch_threads=1,
                architecture=None):
    """Train/fine-tune, return the best validation model and checkpoint metadata.

    zarr_path may be one path or several. Existing cache splits are honored.
    Best and final-epoch checkpoints are separate when paths are supplied.
    Fine-tuning restores weights/normalization, not the optimizer state.
    """
    if epochs < 1 or batch_size < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("epochs/batch_size must be positive and lr positive finite")
    if torch_threads is not None:
        if torch_threads < 1:
            raise ValueError("torch_threads must be positive")
        torch.set_num_threads(torch_threads)
    torch.manual_seed(seed)
    try:
        from .preprocess_onestep_data import prepare_data, load_data
    except ImportError:
        from preprocess_onestep_data import prepare_data, load_data
    if data_pkl is not None:
        cache = load_data(data_pkl)
    elif zarr_path is not None:
        cache = prepare_data(zarr_path, control_hz=control_hz, val_ratio=val_ratio, seed=seed)
    else:
        raise ValueError("Provide zarr_path or data_pkl")
    cached = prepare_stats(cache["stats"])
    pairs = {}
    for split in ("train", "val"):
        x = torch.as_tensor(cache[split]["x"], dtype=torch.float32)
        y = torch.as_tensor(cache[split]["y"], dtype=torch.float32)
        if x.ndim != 2 or x.shape[1] != 12 or y.shape != (len(x), 5) or len(x) < 1:
            raise ValueError(f"Invalid {split} transition arrays")
        if not bool(torch.isfinite(x).all() & torch.isfinite(y).all()):
            raise ValueError(f"Nonfinite {split} transitions")
        pairs[split] = (x*cached["x_std"]+cached["x_mean"],
                        y*cached["y_std"]+cached["y_mean"])
    if pretrained is not None:
        model, stats = load_model(pretrained)
        if not np.isclose(cached["dt"], stats["dt"]):
            raise ValueError("Dataset and checkpoint must have the same control period")
        prior = copy.deepcopy(stats.get("metadata", {}))
    else:
        model, stats, prior = build_model(architecture or "residual_mlp_v1", cached), cached, None
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = {"epochs": int(epochs), "batch_size": int(batch_size), "lr": float(lr),
              "seed": int(seed), "device": str(selected_device), "torch_threads": torch_threads,
              "data_pkl": str(Path(data_pkl).resolve()) if data_pkl else None,
              "architecture": getattr(model, "architecture", "residual_mlp_v1")}
    stats["metadata"] = metadata = _training_metadata(cache, pretrained, prior, config)
    metadata["train_pairs"], metadata["val_pairs"] = len(pairs["train"][0]), len(pairs["val"][0])
    model.to(selected_device)
    norm = prepare_stats(stats, selected_device)
    tensors = {}
    for split, (x, y) in pairs.items():
        tensors[split] = ((x.to(selected_device)-norm["x_mean"])/norm["x_std"],
                          (y.to(selected_device)-norm["y_mean"])/norm["y_std"])
    x_train, y_train = tensors["train"]
    x_val, y_val = tensors["val"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss, best_weights = float("inf"), None
    started = perf_counter()
    print(f"Training {len(x_train)} transitions; validating {len(x_val)}; device={selected_device}")
    for epoch in range(1, epochs+1):
        model.train()
        order = torch.randperm(len(x_train), device=selected_device)
        total_loss = 0.0
        for indices in order.split(batch_size):
            loss = nn.functional.mse_loss(model(x_train[indices]), y_train[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())*len(indices)
        model.eval()
        with torch.no_grad():
            val_loss = float(nn.functional.mse_loss(model(x_val), y_val))
        if not np.isfinite(val_loss):
            raise RuntimeError("Training produced a nonfinite validation loss")
        metadata["history"].append({"epoch": epoch, "train_loss": total_loss/len(x_train),
                                     "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss, best_weights = val_loss, _cpu_copy(model.state_dict())
            metadata["best_epoch"] = epoch
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs} | Train: {total_loss/len(x_train):.6f} | Val: {val_loss:.6f}", flush=True)
    metadata["training_seconds"] = perf_counter()-started
    metadata["best_val_loss"] = best_loss
    metadata["last_epoch"] = epochs
    if last_output is not None:
        last_stats = copy.deepcopy(stats)
        last_stats["metadata"]["checkpoint_role"] = "last"
        last_stats["metadata"]["epoch"] = epochs
        save_model(model, last_stats, last_output)
    model.load_state_dict(best_weights)
    model.cpu().eval()
    stats = prepare_stats(stats)
    stats["metadata"]["checkpoint_role"] = "best"
    stats["metadata"]["epoch"] = metadata["best_epoch"]
    if output is not None:
        save_model(model, stats, output)
    if history_path is not None:
        history_path = Path(history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(stats["metadata"], indent=2)+"\n")
    return model, stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train or inspect Push-T residual world model")
    parser.add_argument("--dataset", nargs="+", default=["data/demonstrations/demo_actions_v1.zarr"])
    parser.add_argument("--data-pkl", help="Use an existing cache and its saved split")
    parser.add_argument("--resume", nargs="?", const=str(MODEL_FILE), help="Fine-tune a trusted checkpoint")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--last-output")
    parser.add_argument("--history")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--architecture", choices=("residual_mlp_v1", "contact_local_v2"))
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--step", type=int, default=0)
    args = parser.parse_args()
    if args.predict:
        try:
            from .episode_kinematics import load_episode_kinematics
        except ImportError:
            from episode_kinematics import load_episode_kinematics
        model, stats = load_model(args.resume or MODEL_FILE, device=args.device)
        episode = load_episode_kinematics(args.dataset[0], args.episode, 1.0/stats["dt"])
        if not 0 <= args.step < episode.num_steps-1:
            parser.error("step must have a next state in the same episode")
        action = [episode.action_target.x[args.step], episode.action_target.y[args.step]]
        print("Predicted:", np.round(predict(model, stats, episode.state_augmented[args.step], action), 3))
        print("Actual:   ", np.round(episode.state_augmented[args.step+1], 3))
        return
    output = Path(args.output)
    last_output = Path(args.last_output) if args.last_output else output.with_name(output.stem+"_last"+output.suffix)
    history = Path(args.history) if args.history else output.with_suffix(".training.json")
    train_model(args.dataset, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                control_hz=args.control_hz, seed=args.seed, pretrained=args.resume,
                data_pkl=args.data_pkl, val_ratio=args.val_ratio, device=args.device,
                output=output, last_output=last_output, history_path=history,
                torch_threads=args.torch_threads, architecture=args.architecture)


if __name__ == "__main__":
    main()
