"""Run learned MPC in Push-T and save reproducible closed-loop measurements.

Example (from pusht-research):
    python -B src/run_mpc.py --seeds 1000 1001 --output results/mpc_v1
    python -B src/run_mpc.py --continuous --start-seed 1000 --render --realtime-plotting --pace

Prediction uses the learned model; only the selected action enters the simulator.
The reference environment and the recorded demonstrations are never modified.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter, sleep
from types import SimpleNamespace

import numpy as np
import torch

PUSHT_TARGET_AREA = 120.0 * 30.0 + 30.0 * 90.0

try:
    from .pusht_runtime import PROJECT_ROOT, StateObserver, coverage, make_env, physical_metrics
except ImportError:
    from pusht_runtime import PROJECT_ROOT, StateObserver, coverage, make_env, physical_metrics


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return _json_value(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    """Atomically replace a UTF-8 JSON file; invalid numeric values become null."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_json_value(value), indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def user_requested_stop(render=False, plotter=None):
    """Service both GUI event loops and detect Q/Esc or a closed window."""
    if plotter is not None:
        if not plotter.closed:
            plotter.fig.canvas.flush_events()
        if plotter.closed:
            return True
    if render:
        import pygame
        if pygame.display.get_init():
            return any(event.type == pygame.QUIT or
                       (event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE))
                       for event in pygame.event.get())
    return False


def coverage_reached(value, target_coverage, legacy=False):
    """Use the original strict threshold, or an explicitly requested coverage.

    A 1e-9 tolerance at exactly 100% only accommodates polygon roundoff; it
    does not relax a practical target such as 99.5%.
    """
    if legacy:
        return value > target_coverage
    tolerance = 1e-9 if target_coverage == 1.0 else 0.0
    return value >= target_coverage - tolerance


def plot_episode(path, data):
    """An independent Agg figure also works while a live GUI backend is active."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    states, goal = data["states"], data["goal"]
    dt = float(data["dt"])
    times = np.arange(len(states)) * dt
    delta_angle = states[:, 8] - goal[2]
    fig = Figure(figsize=(16, 8), constrained_layout=True)
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 4).ravel()
    axes[0].plot(states[:, 0], states[:, 1], label="Actuator")
    axes[0].plot(states[:, 4], states[:, 5], label="T pose origin")
    axes[0].scatter(*goal[:2], marker="*", s=100, color="green", label="Goal origin")
    axes[0].set(xlim=(0, 512), ylim=(512, 0), title="Actual paths", xlabel="X", ylabel="Y")
    axes[0].legend(fontsize=8)
    axes[1].plot(times, np.linalg.norm(states[:, 4:6] - goal[:2], axis=1))
    axes[1].set(title="T position error", ylabel="Simulation units")
    axes[2].plot(times, np.degrees(np.abs(np.arctan2(np.sin(delta_angle), np.cos(delta_angle)))))
    axes[2].set(title="T angle error", ylabel="Degrees")
    axes[3].plot(times, data["coverage"])
    target_coverage = float(data.get("target_coverage", 0.95))
    original_threshold = float(data.get("original_success_threshold", 0.95))
    axes[3].axhline(target_coverage, color="green", linestyle="--",
                    label=f"Requested coverage {target_coverage:.1%}")
    if not np.isclose(target_coverage, original_threshold):
        axes[3].axhline(original_threshold, color="gray", linestyle=":",
                        label=f"Original threshold {original_threshold:.0%}")
    axes[3].set(title="Actual target coverage", ylim=(0, 1.02))
    axes[3].legend(fontsize=8)
    axes[4].plot(times, np.linalg.norm(states[:, 2:4], axis=1))
    axes[4].set(title="Actuator speed", ylabel="Simulation units/s")
    acceleration = np.diff(states[:, 2:4], axis=0) / dt
    axes[5].plot(times[1:], np.linalg.norm(acceleration, axis=1))
    axes[5].set(title="Actuator acceleration", ylabel="Simulation units/s²")
    distance = np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1)
    axes[6].plot(times, np.r_[0.0, np.cumsum(distance)])
    axes[6].set(title="Cumulative actuator distance", ylabel="Simulation units")
    axes[7].plot(times[:-1], 1000 * data["planning_seconds"])
    axes[7].axhline(1000 * dt, color="red", linestyle="--")
    axes[7].set(title="Planning latency", ylabel="ms")
    for ax in axes:
        ax.grid(alpha=0.25)
    for ax in axes[1:]:
        ax.set_xlabel("Simulation time [s]")
    fig.savefig(path, dpi=110)


def run_episode(env, controller, model, stats, seed, max_steps=300,
                mode="mpc", render=False, realtime_plotting=False, pace=False,
                plotter=None, target_coverage=None):
    """Run one episode, retaining coherent completed transitions when interrupted.

    Return (metrics, arrays, planner_diagnostics). A caller-supplied live plotter
    is reset and kept open for reuse; an internally created plotter is closed.
    Q/Esc/close in either GUI or Ctrl+C stops the episode and preserves history.
    Simulation time is completed actions * dt, independently of GUI/CPU latency.
    target_coverage=None retains the environment's strict original threshold;
    an explicit fraction in (0,1] lets the episode refine alignment past it.
    """
    try:
        from .Model_MLP_OneStepPredict import predict
    except ImportError:
        from Model_MLP_OneStepPredict import predict

    if target_coverage is not None and (not np.isfinite(target_coverage)
                                        or not 0 < target_coverage <= 1):
        raise ValueError("target_coverage must be finite and in (0, 1]")
    env.reset()
    legacy_threshold = target_coverage is None
    target_coverage = float(env.success_threshold if legacy_threshold else target_coverage)
    dt = float(stats["dt"])
    if not np.isclose(1.0 / env.control_hz, dt):
        raise ValueError("Checkpoint dt must match the environment control period")
    observer = StateObserver(dt)
    info = env._get_info()
    state = observer.from_info(info)
    goal = np.asarray(info["goal_pose"], dtype=np.float32)
    if controller is not None:
        controller.reset(seed=int(seed))
    rng = np.random.default_rng(seed)
    initial_state = state.copy()
    initial_coverage, initial_contacts = coverage(env), info["n_contacts"]
    # Each append commits one complete transition. Ctrl+C during planning or
    # env.step cannot leave the states/actions/diagnostics with different lengths.
    records = []
    termination_reason = "time_limit"
    owns_plotter = plotter is None and realtime_plotting
    started = perf_counter()
    if owns_plotter:
        try:
            from .episode_kinematics import visualize_episode
        except ImportError:
            from episode_kinematics import visualize_episode
        plotter = visualize_episode(realtime_plotting=True, episode_id=int(seed), control_hz=1 / dt)
    if plotter is not None:
        if not np.isclose(plotter.dt_nominal, dt):
            raise ValueError("Plotter dt must match the environment control period")
        plotter.quit_hint = "Q/Esc: stop and save"
        plotter.reset(episode_id=int(seed))

    try:
        for _ in range(max_steps):
            step_started = perf_counter()
            frame = None
            if render or realtime_plotting:
                frame = env.render(mode="human" if render else "rgb_array")
            if render:
                import pygame
                current_coverage = records[-1]["coverage"] if records else initial_coverage
                pygame.display.set_caption(
                    f"Push-T AI | Seed {seed} | Coverage {current_coverage:.2%} / {target_coverage:.2%} "
                    "| Q/Esc: stop and save")
            if user_requested_stop(render, plotter):
                termination_reason = "user_stop"
                break
            if coverage_reached(records[-1]["coverage"] if records else initial_coverage,
                                target_coverage, legacy_threshold):
                termination_reason = "success"
                break

            if mode == "mpc":
                plan = controller.plan(state, goal)
            else:
                action = state[:2].copy() if mode == "hold" else rng.uniform(
                    env.action_space.low, env.action_space.high).astype(np.float32)
                next_prediction = predict(model, stats, state, action)
                plan = SimpleNamespace(action=action, actions=action[None],
                                       states=np.stack([state, next_prediction]),
                                       cost=None, components={}, planning_seconds=0.0, fallback=False)

            action = np.asarray(plan.action, dtype=np.float32)
            if action.shape != (2,) or not np.isfinite(action).all():
                raise RuntimeError("Controller returned an invalid action")
            if np.any(action < env.action_space.low) or np.any(action > env.action_space.high):
                raise RuntimeError("Controller returned an action outside environment bounds")
            if plotter is not None:
                plotter.update(state, action, image=frame)
            # Planning and plot refresh can take time: honor events received
            # during that work before sending another action to the simulator.
            if user_requested_stop(render, plotter):
                termination_reason = "user_stop"
                break

            _, _, _, info = env.step(action)
            state = observer.from_info(info)
            records.append(dict(state=state.copy(), coverage=coverage(env), contacts=info["n_contacts"],
                                action=action.copy(), predicted=np.asarray(plan.states[1], dtype=np.float32),
                                latency=float(plan.planning_seconds), cost=plan.cost, terms=plan.components,
                                plan=np.asarray(plan.actions, dtype=np.float32),
                                planned_states=np.asarray(plan.states, dtype=np.float32),
                                fallback=bool(plan.fallback)))
            if pace:
                sleep(max(0.0, dt - (perf_counter() - step_started)))
    except KeyboardInterrupt:
        termination_reason = "keyboard_interrupt"
    finally:
        try:
            if plotter is not None and not plotter.closed:
                if records:
                    # Terminal observation with the last issued target still held.
                    frame = None if termination_reason in ("user_stop", "keyboard_interrupt") else env.render(
                        mode="human" if render else "rgb_array")
                    plotter.update(records[-1]["state"], records[-1]["action"], image=frame)
                plotter.refresh()
        except KeyboardInterrupt:
            termination_reason = "keyboard_interrupt"
        finally:
            if owns_plotter:
                plotter.close()

    states = np.asarray([initial_state] + [r["state"] for r in records], dtype=np.float32)
    coverages = [initial_coverage] + [r["coverage"] for r in records]
    contacts = [initial_contacts] + [r["contacts"] for r in records]
    actions = np.asarray([r["action"] for r in records], dtype=np.float32).reshape(-1, 2)
    latencies = np.asarray([r["latency"] for r in records])
    steps = len(records)
    success = coverage_reached(coverages[-1], target_coverage, legacy_threshold)
    uncovered = np.clip(1.0 - np.asarray(coverages), 0.0, 1.0)
    if success and termination_reason == "time_limit":
        termination_reason = "success"
    stopped_by_user = termination_reason in ("user_stop", "keyboard_interrupt")
    prediction = np.asarray([r["predicted"] for r in records], dtype=np.float32).reshape(-1, 10)
    fallback_flags = np.asarray([r["fallback"] for r in records], dtype=bool)
    model_prediction = ~fallback_flags
    finite = np.isfinite(prediction).all(axis=1)
    valid = finite & model_prediction
    prediction_error = np.linalg.norm(prediction[valid, 4:6] - states[1:][valid, 4:6], axis=1)
    metrics = {
        "seed": int(seed), "controller": mode, "steps": steps,
        "success": bool(success), "completion_time_seconds": steps * dt if success else None,
        "target_coverage": target_coverage,
        "target_coverage_comparison": ">" if legacy_threshold else ">=",
        "original_success_threshold": float(env.success_threshold),
        "original_environment_success": bool(coverages[-1] > env.success_threshold),
        "original_environment_reached": bool(max(coverages) > env.success_threshold),
        "target_area": PUSHT_TARGET_AREA,
        "final_uncovered_fraction": float(uncovered[-1]),
        "final_uncovered_area": float(uncovered[-1] * PUSHT_TARGET_AREA),
        "min_uncovered_fraction": float(uncovered.min()),
        "min_uncovered_area": float(uncovered.min() * PUSHT_TARGET_AREA),
        "elapsed_simulation_seconds": steps * dt, "wall_seconds": perf_counter() - started,
        "time_limit_seconds": max_steps * dt, "stopped_by_user": stopped_by_user,
        "termination_reason": termination_reason,
        "initial_coverage": coverages[0], "final_coverage": coverages[-1],
        "max_coverage": max(coverages), "fallback_count": int(fallback_flags.sum()),
        "planning_ms_mean": float(latencies.mean() * 1000) if steps else 0.0,
        "planning_ms_p50": float(np.percentile(latencies, 50) * 1000) if steps else 0.0,
        "planning_ms_p95": float(np.percentile(latencies, 95) * 1000) if steps else 0.0,
        "planning_deadline_misses": int((latencies > dt).sum()),
        "one_step_T_position_error_mean": float(prediction_error.mean()) if len(prediction_error) else None,
        "invalid_prediction_count": int((~finite & model_prediction).sum()),
        "scored_model_predictions": int(valid.sum()),
        **physical_metrics(states, goal, dt),
    }
    horizon = controller.config.horizon if mode == "mpc" and controller is not None and hasattr(
        controller, "config") else 1
    data = dict(states=states, actions=actions,
                predicted_next_states=prediction, coverage=np.asarray(coverages),
                uncovered_fraction=uncovered, uncovered_area=uncovered * PUSHT_TARGET_AREA,
                target_coverage=np.array(target_coverage), target_area=np.array(PUSHT_TARGET_AREA),
                original_success_threshold=np.array(env.success_threshold),
                contacts=np.asarray(contacts), planning_seconds=latencies, goal=goal, dt=np.array(dt),
                planned_actions=np.asarray([r["plan"] for r in records], dtype=np.float32).reshape(-1, horizon, 2),
                planned_states=np.asarray([r["planned_states"] for r in records], dtype=np.float32).reshape(
                    -1, horizon + 1, 10),
                prediction_is_model=model_prediction)
    return metrics, data, {"cost": [r["cost"] for r in records], "components": [r["terms"] for r in records]}


def save_episode(output, metrics, data, diagnostic, save_plot=True):
    """Persist one episode; NPZ and JSON replacements are individually atomic."""
    stem = Path(output) / f"episode_{metrics['seed']}"
    temporary = stem.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **data)
    temporary.replace(stem.with_suffix(".npz"))
    write_json(stem.with_suffix(".json"), {**metrics, "planner_diagnostics": diagnostic})
    if save_plot:
        plot_episode(stem.with_suffix(".png"), data)


def build_summary(metadata, results, run_status):
    """Aggregate completed episodes only; retain interrupted episodes separately."""
    completed = [r for r in results if not r["stopped_by_user"]]
    successes = [r for r in completed if r["success"]]

    def mean(key, records=completed):
        return float(np.mean([r[key] for r in records])) if records else None

    def available_mean(key):
        records = [r for r in completed if key in r]
        return mean(key, records)

    return {
        **metadata, "run_status": run_status, "episodes": list(results), "num_episodes": len(results),
        "num_completed_episodes": len(completed), "num_interrupted_episodes": len(results) - len(completed),
        "success_rate": len(successes) / len(completed) if completed else None,
        "original_environment_success_rate": available_mean("original_environment_success"),
        "original_environment_reached_rate": available_mean("original_environment_reached"),
        "mean_final_uncovered_fraction": available_mean("final_uncovered_fraction"),
        "mean_final_uncovered_area": available_mean("final_uncovered_area"),
        "mean_max_coverage": mean("max_coverage"),
        "mean_completion_time_successes_only": mean("completion_time_seconds", successes),
        "mean_actuator_path_length": mean("actuator_path_length"),
        "mean_acceleration_rms": mean("acceleration_rms"),
        "notes": ("Aggregates use completed success/time_limit episodes and exclude user-interrupted episodes. "
                  "Completion times are censored (null) for failures. Success uses each episode's requested "
                  "actual geometric coverage threshold; the original environment's >95% result is reported separately. "
                  "Uncovered area is measured in squared simulation units.")
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints/WorldModel_contact_best.pt")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/mpc_default.json")
    parser.add_argument("--seeds", type=int, nargs="+", help="Finite episode seeds (default: 1000 through 1004)")
    parser.add_argument("--continuous", action="store_true",
                        help="Start a new episode after success/timeout until Q/Esc/window close or Ctrl+C")
    parser.add_argument("--start-seed", type=int,
                        help="First seed for --continuous (default: 1000); increment by one per episode")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--target-coverage", type=float,
                        help="Required covered target fraction (0,1]; default: cost.goal_coverage if enabled, "
                             "otherwise the original >95%% threshold. Also sets the planner completion target.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/mpc_v1")
    parser.add_argument("--controller", choices=["mpc", "hold", "random"], default="mpc")
    parser.add_argument("--initial-state", type=float, nargs=5, metavar=("AX", "AY", "TX", "TY", "ANGLE"))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--realtime-plotting", action="store_true")
    parser.add_argument("--pace", action="store_true", help="Pace at 10 control steps/s when computation permits")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing matching files from an earlier run")
    args = parser.parse_args(argv)
    if args.target_coverage is not None and (not np.isfinite(args.target_coverage)
                                           or not 0 < args.target_coverage <= 1):
        parser.error("--target-coverage must be finite and in (0, 1]")
    if args.continuous and args.seeds is not None:
        parser.error("Use --start-seed with --continuous; --seeds is for finite runs")
    if not args.continuous and args.start_seed is not None:
        parser.error("--start-seed requires --continuous")
    seeds = args.seeds if args.seeds is not None else [1000, 1001, 1002, 1003, 1004]
    start_seed = args.start_seed if args.start_seed is not None else 1000
    if args.max_steps < 1 or args.threads < 1 or len(set(seeds)) != len(seeds):
        parser.error("max-steps/threads must be positive and seeds must be unique")
    if any(seed < 0 or seed >= 2**32 for seed in ([start_seed] if args.continuous else seeds)):
        parser.error("Seeds must be integers in [0, 2**32 - 1]")
    if args.output.exists() and not args.overwrite and (
            (args.output / "summary.json").exists() or any(args.output.glob("episode_*"))):
        parser.error("Output already contains a run; choose another directory or use --overwrite")
    if args.overwrite and args.output.exists():
        print("Overwrite enabled: matching files may be replaced; other existing files are retained. "
              "summary.json describes this invocation only.", flush=True)
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    if not args.render:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    torch.set_num_threads(args.threads)

    try:
        from .Model_MLP_OneStepPredict import load_model
        from .MPC_Controller import MPCController, MPCConfig
        from .CostFunctions import CostConfig
    except ImportError:
        from Model_MLP_OneStepPredict import load_model
        from MPC_Controller import MPCController, MPCConfig
        from CostFunctions import CostConfig

    model, stats = load_model(args.checkpoint, device=args.device)
    config = json.loads(args.config.read_text())
    if args.target_coverage is not None:
        config["cost"]["goal_coverage"] = args.target_coverage
    mpc_config, cost_config = MPCConfig(**config["mpc"]), CostConfig(**config["cost"])
    target_coverage = float(getattr(cost_config, "goal_coverage", 0.0)) or None
    if not np.isclose(float(stats["dt"]), cost_config.dt):
        raise ValueError("Cost dt and checkpoint dt must match")
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_provenance": stats.get("metadata", {}).get("provenance", "unknown"),
        "mpc_config": asdict(mpc_config), "cost_config": asdict(cost_config),
        "controller": args.controller, "initial_state": args.initial_state,
        "dt": float(stats["dt"]), "device": args.device, "threads": args.threads,
        "continuous": args.continuous, "start_seed": start_seed if args.continuous else None,
        "requested_seeds": None if args.continuous else seeds,
        "target_coverage": target_coverage,
        "target_coverage_comparison": ">" if target_coverage is None else ">=",
    }
    summary_path = args.output / "summary.json"
    write_json(summary_path, build_summary(metadata, results, "running"))
    env, plotter = None, None
    run_status = "error"
    episode_seeds = range(start_seed, 2**32) if args.continuous else seeds
    try:
        # Reuse both windows across episodes; reset only physical and chart history.
        env = make_env(start_seed if args.continuous else seeds[0], args.initial_state)
        metadata["target_coverage"] = (float(getattr(env, "success_threshold", 0.95))
                                       if target_coverage is None else target_coverage)
        metadata["original_success_threshold"] = float(getattr(env, "success_threshold", 0.95))
        controller = MPCController(model, stats, config=mpc_config, cost_config=cost_config,
                                   action_low=env.action_space.low, action_high=env.action_space.high)
        if args.realtime_plotting:
            try:
                from .episode_kinematics import visualize_episode
            except ImportError:
                from episode_kinematics import visualize_episode
            plotter = visualize_episode(realtime_plotting=True, control_hz=1 / float(stats["dt"]))
        for seed in episode_seeds:
            if user_requested_stop(args.render, plotter):
                run_status = "stopped_by_user"
                break
            env.seed(int(seed))
            metrics, data, diagnostic = run_episode(
                env, controller, model, stats, seed, args.max_steps, args.controller,
                args.render, args.realtime_plotting, args.pace, plotter=plotter,
                target_coverage=target_coverage)
            # Commit the numeric record and summary before the slower optional
            # image export, so Ctrl+C during export cannot hide a saved episode.
            save_episode(args.output, metrics, data, diagnostic, save_plot=False)
            results.append(metrics)
            run_status = ("keyboard_interrupt" if metrics["termination_reason"] == "keyboard_interrupt"
                          else "stopped_by_user") if metrics["stopped_by_user"] else "running"
            write_json(summary_path, build_summary(metadata, results, run_status))
            if not args.no_plots:
                plot_episode(args.output / f"episode_{seed}.png", data)
            print(f"Seed {seed}: success={metrics['success']} steps={metrics['steps']} "
                  f"max_coverage={metrics['max_coverage']:.3f} "
                  f"planner_p95={metrics['planning_ms_p95']:.1f}ms "
                  f"end={metrics['termination_reason']}", flush=True)
            if metrics["stopped_by_user"]:
                break
        else:
            run_status = "seed_limit" if args.continuous else "finished"
    except KeyboardInterrupt:
        # Between episodes (e.g. during image export), completed saves remain.
        run_status = "keyboard_interrupt"
    except Exception:
        # A later failure must not leave a completed invocation labeled running.
        run_status = "error"
        raise
    finally:
        if plotter is not None:
            plotter.close()
        if env is not None:
            env.close()
        summary = build_summary(metadata, results, run_status)
        write_json(summary_path, summary)
    print(f"Saved {args.output / 'summary.json'}", flush=True)
    return summary


if __name__ == "__main__":
    main()
