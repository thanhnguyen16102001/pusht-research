"""Run an imitation Diffusion Policy in the original Push-T simulator.

The policy samples action chunks from observed states. It does not use a World
Model, evaluate MPC costs, or predict future simulator states. Control remains
at the checkpoint's recorded frequency (normally 10 Hz).
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import os
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import torch

try:
    from .pusht_runtime import PROJECT_ROOT, StateObserver, coverage, make_env, physical_metrics
    from .run_mpc import (PUSHT_TARGET_AREA, build_summary, coverage_reached,
                          plot_episode, user_requested_stop, write_json)
except ImportError:
    from pusht_runtime import PROJECT_ROOT, StateObserver, coverage, make_env, physical_metrics
    from run_mpc import (PUSHT_TARGET_AREA, build_summary, coverage_reached,
                         plot_episode, user_requested_stop, write_json)


def load_policy(path, device="cpu"):
    """Load the standalone policy without importing a World Model."""
    try:
        from .Diffusion_Policy_model import load_policy as load
    except ImportError:
        from Diffusion_Policy_model import load_policy as load
    return load(path, device=device)


def checkpoint_sha256(path):
    """Hash large pretrained checkpoints without duplicating them in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_episode(env, policy, seed, max_steps=300, target_coverage=0.99,
                render=False, realtime_plotting=False, pace=False, plotter=None):
    """Return metrics, trajectory arrays, and policy diagnostics for one episode.

    ``policy.act`` receives every causal observation, including steps that use
    an existing action chunk. Completion checks actual polygon coverage after
    each executed action, regardless of the simulator's original >95% flag.
    Only completed transitions are retained if the user interrupts a step.
    A supplied plotter is reset and reused; an internally created one is closed.
    """
    if not np.isfinite(target_coverage) or not 0 < target_coverage <= 1:
        raise ValueError("target_coverage must be finite and in (0, 1]")
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    controller_name = getattr(policy, "name", "diffusion_policy")
    display_name = getattr(policy, "display_name", "Diffusion Policy")
    dt = float(policy.dt)
    if not np.isfinite(dt) or dt <= 0 or not np.isclose(dt, 1.0 / env.control_hz):
        raise ValueError("Policy dt must match the environment control period")
    env.reset()
    policy.reset(seed=int(seed))
    observer = StateObserver(dt)
    info = env._get_info()
    initial_state = observer.from_info(info)
    state = initial_state.copy()
    goal = np.asarray(info["goal_pose"], dtype=np.float32)
    initial_coverage, initial_contacts = coverage(env), info["n_contacts"]
    records = []
    termination_reason = "time_limit"
    started = perf_counter()
    owns_plotter = plotter is None and realtime_plotting
    if owns_plotter:
        try:
            from .episode_kinematics import visualize_episode
        except ImportError:
            from episode_kinematics import visualize_episode
        plotter = visualize_episode(realtime_plotting=True, episode_id=int(seed), control_hz=1 / dt)
    if plotter is not None:
        if not np.isclose(plotter.dt_nominal, dt):
            raise ValueError("Plotter dt must match the policy control period")
        plotter.quit_hint = "Q/Esc: stop and save"
        plotter.reset(episode_id=int(seed))

    try:
        for _ in range(max_steps):
            step_started = perf_counter()
            current_coverage = records[-1]["coverage"] if records else initial_coverage
            frame = env.render(mode="human" if render else "rgb_array") if render or realtime_plotting else None
            if render:
                import pygame
                pygame.display.set_caption(
                    f"Push-T {display_name} | Seed {seed} | Coverage {current_coverage:.2%} "
                    f"/ {target_coverage:.2%} | Q/Esc: stop and save")
            if user_requested_stop(render, plotter):
                termination_reason = "user_stop"
                break
            if coverage_reached(current_coverage, target_coverage):
                termination_reason = "success"
                break
            inference_started = perf_counter()
            action = np.asarray(policy.act(state), dtype=np.float32)
            inference_seconds = perf_counter() - inference_started
            if action.shape != (2,) or not np.isfinite(action).all():
                raise RuntimeError("Policy returned an invalid action")
            if np.any(action < env.action_space.low) or np.any(action > env.action_space.high):
                raise RuntimeError("Policy returned an action outside environment bounds")
            replanned = bool(policy.last_replanned)
            sampled_chunk = np.asarray(policy.last_plan, dtype=np.float32).copy() if replanned else None
            if replanned and (sampled_chunk.shape != (policy.config.prediction_horizon, 2)
                              or not np.isfinite(sampled_chunk).all()):
                raise RuntimeError("Policy returned an invalid sampled action chunk")
            sampling_seconds = float(policy.last_sampling_seconds) if replanned else 0.0
            if plotter is not None:
                plotter.update(state, action, image=frame)
            if user_requested_stop(render, plotter):
                termination_reason = "user_stop"
                break
            _, _, _, info = env.step(action)
            state = observer.from_info(info)
            records.append(dict(state=state.copy(), action=action.copy(), coverage=coverage(env),
                                contacts=info["n_contacts"], inference_seconds=inference_seconds,
                                replanned=replanned, sampled_chunk=sampled_chunk,
                                sampling_seconds=sampling_seconds))
            if pace:
                sleep(max(0.0, dt - (perf_counter() - step_started)))
    except KeyboardInterrupt:
        termination_reason = "keyboard_interrupt"
    finally:
        try:
            if plotter is not None and not plotter.closed:
                if records:
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
    actions = np.asarray([r["action"] for r in records], dtype=np.float32).reshape(-1, 2)
    coverages = np.asarray([initial_coverage] + [r["coverage"] for r in records])
    contacts = np.asarray([initial_contacts] + [r["contacts"] for r in records])
    latencies = np.asarray([r["inference_seconds"] for r in records])
    sampling = np.asarray([r["sampling_seconds"] for r in records])
    replanned = np.asarray([r["replanned"] for r in records], dtype=bool)
    chunk_steps = np.flatnonzero(replanned)
    chunks = np.asarray([r["sampled_chunk"] for r in records if r["replanned"]],
                        dtype=np.float32).reshape(-1, policy.config.prediction_horizon, 2)
    steps = len(records)
    success = coverage_reached(coverages[-1], target_coverage)
    if success and termination_reason == "time_limit":
        termination_reason = "success"
    uncovered = np.clip(1.0 - coverages, 0.0, 1.0)
    metrics = {
        "seed": int(seed), "controller": controller_name, "steps": steps,
        "success": bool(success), "completion_time_seconds": steps * dt if success else None,
        "target_coverage": float(target_coverage), "target_coverage_comparison": ">=",
        "original_success_threshold": float(env.success_threshold),
        "original_environment_success": bool(coverages[-1] > env.success_threshold),
        "original_environment_reached": bool(coverages.max() > env.success_threshold),
        "target_area": PUSHT_TARGET_AREA,
        "final_uncovered_fraction": float(uncovered[-1]),
        "final_uncovered_area": float(uncovered[-1] * PUSHT_TARGET_AREA),
        "min_uncovered_fraction": float(uncovered.min()),
        "min_uncovered_area": float(uncovered.min() * PUSHT_TARGET_AREA),
        "elapsed_simulation_seconds": steps * dt, "wall_seconds": perf_counter() - started,
        "time_limit_seconds": max_steps * dt,
        "stopped_by_user": termination_reason in ("user_stop", "keyboard_interrupt"),
        "termination_reason": termination_reason,
        "initial_coverage": float(coverages[0]), "final_coverage": float(coverages[-1]),
        "max_coverage": float(coverages.max()), "num_action_chunks": len(chunks),
        "inference_ms_mean": float(latencies.mean() * 1000) if steps else 0.0,
        "inference_ms_p95": float(np.percentile(latencies, 95) * 1000) if steps else 0.0,
        "inference_deadline_misses": int((latencies > dt).sum()),
        "sampling_ms_mean": float(sampling[replanned].mean() * 1000) if len(chunks) else 0.0,
        **physical_metrics(states, goal, dt),
    }
    data = dict(states=states, actions=actions, coverage=coverages, contacts=contacts,
                uncovered_fraction=uncovered, uncovered_area=uncovered * PUSHT_TARGET_AREA,
                target_coverage=np.array(target_coverage), target_area=np.array(PUSHT_TARGET_AREA),
                original_success_threshold=np.array(env.success_threshold), goal=goal, dt=np.array(dt),
                inference_seconds=latencies, sampling_seconds=sampling, replanned=replanned,
                sampled_action_chunks=chunks, chunk_start_steps=chunk_steps,
                # Shared chart helper calls the latency column planning_seconds.
                planning_seconds=latencies)
    diagnostics = {"chunk_start_steps": chunk_steps.tolist(),
                   "action_horizon": policy.config.action_horizon,
                   "prediction_horizon": policy.config.prediction_horizon,
                   "inference_steps": policy.config.inference_steps,
                   "latency_note": "Inference includes history update and action queue access; sampling occurs only at chunk starts."}
    if hasattr(policy.config, "sampler"):
        diagnostics["sampler"] = policy.config.sampler
    return metrics, data, diagnostics


def save_episode(output, metrics, data, diagnostics):
    """Save numeric trajectories and metadata atomically, without state forecasts."""
    stem = Path(output) / f"episode_{metrics['seed']}"
    temporary = stem.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **data)
    temporary.replace(stem.with_suffix(".npz"))
    write_json(stem.with_suffix(".json"), {**metrics, "policy_diagnostics": diagnostics})


def main(argv=None, *, policy_loader=None, default_checkpoint=None,
         default_output=None, default_target_coverage=0.99, sampler_choices=None,
         description=None):
    """Shared live runner; optional loaders keep checkpoint formats separate."""
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=default_checkpoint or PROJECT_ROOT / "checkpoints/DiffusionPolicy_best.pt")
    parser.add_argument("--seeds", type=int, nargs="+", help="Finite seeds (default: 1000 through 1004)")
    parser.add_argument("--continuous", action="store_true", help="Run episodes until Q/Esc/window close or Ctrl+C")
    parser.add_argument("--start-seed", type=int, help="First continuous seed (default: 1000)")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--target-coverage", type=float, default=default_target_coverage,
                        help=f"Complete when actual coverage reaches this fraction (default: {default_target_coverage})")
    parser.add_argument("--action-horizon", type=int, help="Execute this many sampled actions before resampling")
    parser.add_argument("--inference-steps", type=int, help="Denoising steps per sampled action chunk")
    if sampler_choices is not None:
        parser.add_argument("--sampler", choices=sampler_choices,
                            help="Sampling method; original DDPM requires all training steps; DDIM supports fewer steps")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--realtime-plotting", action="store_true")
    parser.add_argument("--pace", action="store_true")
    parser.add_argument("--initial-state", type=float, nargs=5, metavar=("AX", "AY", "TX", "TY", "ANGLE"))
    parser.add_argument("--output", type=Path, default=default_output or PROJECT_ROOT / "results/diffusion_policy")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not np.isfinite(args.target_coverage) or not 0 < args.target_coverage <= 1:
        parser.error("target-coverage must be finite and in (0, 1]")
    if args.continuous and args.seeds is not None:
        parser.error("Use --start-seed with --continuous; --seeds is for finite runs")
    if not args.continuous and args.start_seed is not None:
        parser.error("--start-seed requires --continuous")
    seeds = args.seeds if args.seeds is not None else [1000, 1001, 1002, 1003, 1004]
    start_seed = args.start_seed if args.start_seed is not None else 1000
    if args.max_steps < 1 or args.threads < 1 or len(set(seeds)) != len(seeds):
        parser.error("max-steps/threads must be positive and seeds must be unique")
    if any(seed < 0 or seed >= 2**32 for seed in ([start_seed] if args.continuous else seeds)):
        parser.error("Seeds must be integers in [0, 2**32)")
    if args.output.exists() and not args.overwrite and (
            (args.output / "summary.json").exists() or any(args.output.glob("episode_*"))):
        parser.error("Output already contains a run; choose another directory or use --overwrite")
    if args.overwrite and args.output.exists():
        print("Overwrite replaces matching files only; summary describes this invocation.", flush=True)
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    if not args.render:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    torch.set_num_threads(args.threads)
    loader = load_policy if policy_loader is None else policy_loader
    print(f"Loading policy: {args.checkpoint} | device={args.device}", flush=True)
    policy = loader(args.checkpoint, device=args.device)
    overrides = {key: getattr(args, key) for key in ("action_horizon", "inference_steps")
                 if getattr(args, key) is not None}
    if getattr(args, "sampler", None) is not None:
        overrides["sampler"] = args.sampler
    if overrides:
        try:
            policy.config = replace(policy.config, **overrides)
        except ValueError as error:
            parser.error(str(error))
    print(f"Policy ready | denoising steps={policy.config.inference_steps} "
          f"| action chunk={policy.config.action_horizon} "
          f"| sampler={getattr(policy.config, 'sampler', 'policy default')}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
        "controller": getattr(policy, "name", "diffusion_policy"), "policy_config": asdict(policy.config),
        "policy_metadata": getattr(policy, "metadata", {}),
        "dt": float(policy.dt), "device": args.device, "threads": args.threads,
        "continuous": args.continuous, "start_seed": start_seed if args.continuous else None,
        "requested_seeds": None if args.continuous else seeds,
        "initial_state": args.initial_state, "target_coverage": args.target_coverage,
        "target_coverage_comparison": ">=",
        "policy_note": "Imitation policy conditioned on observed state history, without a World Model or MPC cost optimization.",
    }
    results = []
    summary_path = args.output / "summary.json"
    write_json(summary_path, build_summary(metadata, results, "running"))
    env, plotter = None, None
    run_status = "error"
    episode_seeds = range(start_seed, 2**32) if args.continuous else seeds
    try:
        env = make_env(start_seed if args.continuous else seeds[0], args.initial_state)
        metadata["environment_legacy"] = bool(getattr(env, "legacy", False))
        if args.realtime_plotting:
            try:
                from .episode_kinematics import visualize_episode
            except ImportError:
                from episode_kinematics import visualize_episode
            plotter = visualize_episode(realtime_plotting=True, control_hz=1 / float(policy.dt))
        for seed in episode_seeds:
            if user_requested_stop(args.render, plotter):
                run_status = "stopped_by_user"
                break
            env.seed(int(seed))
            print(f"Starting episode {seed} | coverage target={args.target_coverage:.1%} "
                  f"| render={args.render} | Q/Esc: stop and save", flush=True)
            metrics, data, diagnostics = run_episode(
                env, policy, seed, max_steps=args.max_steps, target_coverage=args.target_coverage,
                render=args.render, realtime_plotting=args.realtime_plotting, pace=args.pace, plotter=plotter)
            # PushTEnv creates success_threshold during reset, not construction.
            metadata["original_success_threshold"] = metrics["original_success_threshold"]
            save_episode(args.output, metrics, data, diagnostics)
            results.append(metrics)
            run_status = ("keyboard_interrupt" if metrics["termination_reason"] == "keyboard_interrupt"
                          else "stopped_by_user") if metrics["stopped_by_user"] else "running"
            write_json(summary_path, build_summary(metadata, results, run_status))
            if not args.no_plots:
                plot_episode(args.output / f"episode_{seed}.png", data)
            print(f"Seed {seed}: success={metrics['success']} steps={metrics['steps']} "
                  f"max_coverage={metrics['max_coverage']:.3f} "
                  f"end={metrics['termination_reason']}", flush=True)
            if metrics["stopped_by_user"]:
                break
        else:
            run_status = "seed_limit" if args.continuous else "finished"
    except KeyboardInterrupt:
        run_status = "keyboard_interrupt"
    except Exception:
        run_status = "error"
        raise
    finally:
        if plotter is not None:
            plotter.close()
        if env is not None:
            env.close()
        summary = build_summary(metadata, results, run_status)
        write_json(summary_path, summary)
    print(f"Saved {summary_path}", flush=True)
    return summary


if __name__ == "__main__":
    main()
