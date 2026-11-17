"""Evaluate recursive Push-T model rollouts without crossing episode boundaries.

Metrics use physical simulation units, not normalized training targets. A report
is called held-out validation only when checkpoint provenance establishes that
the selected episodes were excluded from all training. Legacy checkpoints need
--allow-exploratory; explicit --episodes always produces an exploratory report.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import zarr

try:
    from .episode_kinematics import load_episode_kinematics
except ImportError:
    from episode_kinematics import load_episode_kinematics


def select_episodes(stats, datasets, episode_ids=None, allow_exploratory=False):
    """Return source/id pairs and an honest evaluation-split description."""
    paths = list(dict.fromkeys(str(Path(p).expanduser().resolve()) for p in datasets))
    if not paths:
        raise ValueError("At least one dataset is required")
    counts = {}
    for path in paths:
        root = zarr.open_group(path, mode="r")
        counts[path] = len(root["meta"]["episode_ends"])

    metadata = stats.get("metadata", {})
    known = (metadata.get("provenance") == "known"
             and metadata.get("validation_is_heldout", False))
    if episode_ids is not None:
        selected = []
        for path in paths:
            for episode_id in sorted(set(int(i) for i in episode_ids)):
                if not 0 <= episode_id < counts[path]:
                    raise ValueError(f"Episode {episode_id} is outside {path}")
                selected.append({"source": path, "id": episode_id})
        return selected, "exploratory_explicit_episodes"

    records = metadata.get("episodes", [])
    # Check recorded numerical data, not just the file name or episode count.
    signatures = {}
    for record in records:
        source = str(Path(record["source"]).expanduser().resolve())
        expected = record.get("dataset_signature")
        if source in counts and expected:
            if source not in signatures:
                try:
                    from .preprocess_onestep_data import dataset_signature
                except ImportError:
                    from preprocess_onestep_data import dataset_signature
                signatures[source] = dataset_signature(source)
            if signatures[source] != expected:
                raise ValueError(f"Dataset contents changed since training: {source}. "
                                 "Use explicit --episodes only for exploratory evaluation of changed data.")
    selected = []
    for record in records:
        if record.get("split") != "val":
            continue
        source = str(Path(record["source"]).expanduser().resolve())
        if source in counts:
            episode_id = int(record["id"])
            if not 0 <= episode_id < counts[source]:
                raise ValueError("Checkpoint episode metadata exceeds dataset bounds")
            selected.append({"source": source, "id": episode_id})
    selected = list({(r["source"], r["id"]): r for r in selected}.values())
    if known and selected:
        return selected, "heldout_validation"
    if not allow_exploratory:
        raise ValueError("Held-out episodes cannot be established for these datasets and checkpoint. "
                         "Use --allow-exploratory to measure errors without claiming held-out performance.")
    if not selected:
        selected = [{"source": p, "id": i} for p in paths for i in range(counts[p])]
    return selected, "exploratory_unknown_training_history"


def collect_windows(selected, horizon, dt, stride=5, max_windows=1024, seed=42):
    """Read aligned windows; contact[k+1] labels the effect of action[k]."""
    if horizon < 1 or stride < 1 or max_windows < 1 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("horizon, stride, max_windows, and dt must be positive")
    episodes, candidates = [], []
    for record in selected:
        source, episode_id = record["source"], int(record["id"])
        episode = load_episode_kinematics(source, episode_id, control_hz=1.0 / dt)
        if episode.num_steps <= horizon:
            continue
        root = zarr.open_group(source, mode="r")
        ends = root["meta"]["episode_ends"]
        begin = 0 if episode_id == 0 else int(ends[episode_id - 1])
        contacts = None
        if "n_contacts" in root["data"]:
            contacts = np.asarray(root["data"]["n_contacts"][begin:begin + episode.num_steps]).reshape(-1)
        states = episode.state_augmented.astype(np.float32)
        actions = np.column_stack((episode.action_target.x, episode.action_target.y)).astype(np.float32)
        index = len(episodes)
        episodes.append((record, states, actions, contacts))
        candidates.extend((index, step) for step in range(0, episode.num_steps - horizon, stride))
    if not candidates:
        raise ValueError("No complete windows fit the requested maximum horizon")
    if len(candidates) > max_windows:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(candidates), max_windows, replace=False))
        candidates = [candidates[i] for i in indices]

    truth, actions, contact_flags, sources = [], [], [], []
    for index, start in candidates:
        record, state, action, contacts = episodes[index]
        truth.append(state[start:start + horizon + 1])
        actions.append(action[start:start + horizon])
        # Unknown contact annotations stay unknown, rather than being counted as no contact.
        contact_flags.append(np.full(horizon, -1, dtype=np.int8) if contacts is None
                             else (contacts[start + 1:start + horizon + 1] > 0).astype(np.int8))
        sources.append({**record, "start_step": start})
    return np.stack(truth), np.stack(actions), np.stack(contact_flags), sources


def baseline_rollouts(initial_states, horizon, dt):
    """Two action-independent baselines in the same 10D state representation."""
    initial = np.asarray(initial_states, dtype=np.float32)
    persistence = np.repeat(initial[:, None, :], horizon + 1, axis=1)
    constant_velocity = persistence.copy()
    times = np.arange(horizon + 1, dtype=np.float32)[None, :] * dt
    for position, velocity in ((0, 2), (1, 3), (4, 6), (5, 7), (8, 9)):
        constant_velocity[:, :, position] += initial[:, velocity, None] * times
    return {"persistence": persistence, "constant_velocity": constant_velocity}


def endpoint_metrics(prediction, truth):
    """Euclidean vector RMSEs and wrapped angular errors at one horizon."""
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    if prediction.shape != truth.shape or prediction.ndim != 2 or prediction.shape[1] != 10:
        raise ValueError("Expected matching arrays of shape (B, 10)")
    count = len(truth)
    valid = np.isfinite(prediction).all(axis=1) & np.isfinite(truth).all(axis=1)
    metrics = {"windows": count, "valid_windows": int(valid.sum()),
               "invalid_fraction": float(1.0 - valid.mean()) if count else None}
    names = ("agent_position_rmse", "block_position_rmse", "agent_velocity_rmse",
             "block_velocity_rmse", "block_omega_rmse", "block_angle_mae_deg",
             "block_angle_rmse_deg", "block_position_p95", "block_angle_p95_deg")
    if not valid.any():
        metrics.update({key: None for key in names})
        return metrics
    error = prediction[valid].astype(np.float64) - truth[valid].astype(np.float64)
    angle = np.rad2deg(np.arctan2(np.sin(error[:, 8]), np.cos(error[:, 8])))
    for name, columns in (("agent_position", [0, 1]), ("block_position", [4, 5]),
                          ("agent_velocity", [2, 3]), ("block_velocity", [6, 7])):
        metrics[name + "_rmse"] = float(np.sqrt(np.mean(np.sum(error[:, columns] ** 2, axis=1))))
    metrics.update(block_omega_rmse=float(np.sqrt(np.mean(error[:, 9] ** 2))),
                   block_angle_mae_deg=float(np.mean(np.abs(angle))),
                   block_angle_rmse_deg=float(np.sqrt(np.mean(angle ** 2))),
                   block_position_p95=float(np.percentile(np.linalg.norm(error[:, [4, 5]], axis=1), 95)),
                   block_angle_p95_deg=float(np.percentile(np.abs(angle), 95)))
    return metrics


def evaluate_predictions(predictions, truth, contacts, horizons, dt):
    """Compare identical windows for every method and requested horizon."""
    rows = []
    for method, prediction in predictions.items():
        if prediction.shape != truth.shape:
            raise ValueError("Prediction trajectory shape does not match ground truth")
        for horizon in horizons:
            if not 1 <= horizon < truth.shape[1]:
                raise ValueError("Metric horizon is outside the trajectory")
            known = (contacts[:, :horizon] >= 0).all(axis=1)
            touched = (contacts[:, :horizon] > 0).any(axis=1)
            subsets = {"all": np.ones(len(truth), dtype=bool),
                       "contact": known & touched, "no_contact": known & ~touched}
            for subset, selected in subsets.items():
                rows.append({"method": method, "horizon": int(horizon),
                             "seconds": float(horizon * dt), "subset": subset,
                             **endpoint_metrics(prediction[selected, horizon], truth[selected, horizon])})
    return rows


def _save_plots(output, rows, truth, predictions, dt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for method in predictions:
        subset = [r for r in rows if r["method"] == method and r["subset"] == "all"]
        for ax, metric, label in zip(axes,
                                    ("agent_position_rmse", "block_position_rmse", "block_angle_rmse_deg"),
                                    ("Agent position [sim units]", "T position [sim units]", "T angle [degrees]")):
            ax.plot([r["seconds"] for r in subset], [r[metric] for r in subset], "o-", label=method)
            ax.set(xlabel="Prediction horizon [s]", ylabel="Endpoint RMSE", title=label)
            ax.grid(alpha=0.3)
    axes[-1].legend()
    fig.savefig(output / "rollout_errors.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    times = np.arange(truth.shape[1]) * dt
    all_series = {"recorded": truth, **predictions}
    for name, states in all_series.items():
        axes[0].plot(states[0, :, 4], states[0, :, 5], ".-", label=name)
        angle = np.unwrap(states[0, :, 8])
        # Align the first angle to the same branch for readable comparisons.
        angle += 2 * np.pi * np.round((truth[0, 0, 8] - angle[0]) / (2 * np.pi))
        axes[1].plot(times, np.rad2deg(angle), label=name)
    axes[0].set(xlabel="T x [sim units]", ylabel="T y [sim units]", title="Example recursive T trajectory")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].invert_yaxis()
    axes[1].set(xlabel="Nominal time [s]", ylabel="T angle [degrees]", title="Example T orientation")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[1].legend()
    fig.savefig(output / "example_rollout.png", dpi=150)
    plt.close(fig)


def evaluate_model(model, stats, datasets, output_dir=None, horizons=(1, 5, 10, 20),
                   max_windows=1024, stride=5, seed=42, batch_size=256,
                   episode_ids=None, allow_exploratory=False, rollout_fn=None,
                   checkpoint=None):
    """Evaluate free-running rollouts, optionally writing a compact report bundle."""
    horizons = sorted(set(int(h) for h in horizons))
    if not horizons or horizons[0] < 1 or batch_size < 1:
        raise ValueError("Positive horizons and batch_size are required")
    dt = float(stats["dt"])
    selected, split = select_episodes(stats, datasets, episode_ids, allow_exploratory)
    truth, actions, contacts, windows = collect_windows(selected, max(horizons), dt,
                                                       stride, max_windows, seed)
    if rollout_fn is None:
        try:
            from .Model_MLP_OneStepPredict import rollout
        except ImportError:
            from Model_MLP_OneStepPredict import rollout
        rollout_fn = rollout
    predicted = []
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(truth), batch_size):
            end = min(begin + batch_size, len(truth))
            result = rollout_fn(model, stats, torch.from_numpy(truth[begin:end, 0]),
                                torch.from_numpy(actions[begin:end]))
            if isinstance(result, torch.Tensor):
                result = result.detach().cpu().numpy()
            predicted.append(np.asarray(result))
    predictions = {"world_model": np.concatenate(predicted),
                   **baseline_rollouts(truth[:, 0], max(horizons), dt)}
    rows = evaluate_predictions(predictions, truth, contacts, horizons, dt)
    report = {"schema_version": 1, "evaluation_split": split,
              "checkpoint": str(Path(checkpoint).resolve()) if checkpoint else None,
              "checkpoint_provenance": stats.get("metadata", {}).get("provenance", "unknown"),
              "dt": dt, "horizons": horizons, "windows": len(truth),
              "selected_episodes": selected, "window_sources": windows,
              "sampling": {"stride": stride, "max_windows": max_windows, "seed": seed},
              "metric_definition": "Endpoint error after recursive prediction; vector RMSE is sqrt(mean(||error||^2)).",
              "contact_definition": "Any recorded contact in the future transitions of this window; n_contacts[t+1] labels action[t].",
              "interpretation": "Validation estimates model error; it does not establish closed-loop task success.",
              "metrics": rows}
    if output_dir is not None:
        output = Path(output_dir).expanduser()
        output.mkdir(parents=True, exist_ok=True)
        (output / "model_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        with (output / "metrics.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        np.savez_compressed(output / "example_rollout.npz", truth=truth[0], actions=actions[0],
                            contacts=contacts[0], **{name: values[0] for name, values in predictions.items()})
        _save_plots(output, rows, truth, predictions, dt)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/WorldModel_best.pt")
    parser.add_argument("--datasets", nargs="+", default=["data/demonstrations/demo_actions_v1.zarr",
                                                         "data/demonstrations/pusht_demo.zarr"])
    parser.add_argument("--output", default="results/world_model")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--max-windows", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, nargs="+", help="Explicit exploratory episode selection")
    parser.add_argument("--allow-exploratory", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        from .Model_MLP_OneStepPredict import load_model
    except ImportError:
        from Model_MLP_OneStepPredict import load_model
    model, stats = load_model(args.checkpoint)
    model.to(args.device)
    report = evaluate_model(model, stats, args.datasets, args.output, args.horizons,
                            args.max_windows, args.stride, args.seed, args.batch_size,
                            args.episodes, args.allow_exploratory, checkpoint=args.checkpoint)
    print(f"Evaluation: {report['evaluation_split']} | Windows: {report['windows']} | dt: {report['dt']}")
    for row in report["metrics"]:
        if row["method"] == "world_model" and row["subset"] == "all":
            print(f"H={row['horizon']:2d}: T position RMSE={row['block_position_rmse']}, "
                  f"angle RMSE={row['block_angle_rmse_deg']} deg, invalid={row['invalid_fraction']}")
    print(f"Reports: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
