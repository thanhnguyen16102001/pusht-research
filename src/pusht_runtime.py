"""Push-T runtime adapters; use the same state convention as demonstration data."""

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StateObserver:
    """Causal 10D states with interval-average velocities, not body velocities.

    Start a new observer (or reset it) at every environment reset. Pymunk's raw
    angle from info is retained; only angular differences are wrapped.
    """

    def __init__(self, dt=0.1):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        self.dt = float(dt)
        self.previous = None

    def reset(self):
        self.previous = None

    def observe(self, raw_state):
        raw = np.asarray(raw_state, dtype=np.float64)
        if raw.shape != (5,) or not np.isfinite(raw).all():
            raise ValueError("raw_state must be five finite pose coordinates")
        velocity = np.zeros(5)
        if self.previous is not None:
            velocity[:4] = (raw[:4] - self.previous[:4]) / self.dt
            angles = np.unwrap([self.previous[4], raw[4]])
            velocity[4] = (angles[1] - angles[0]) / self.dt
        self.previous = raw.copy()
        return np.array([*raw[:2], *velocity[:2], *raw[2:4], *velocity[2:4],
                         raw[4], velocity[4]], dtype=np.float32)

    def from_info(self, info):
        return self.observe(np.concatenate([info["pos_agent"], info["block_pose"]]))


def make_env(seed=0, initial_state=None, render_size=96):
    """Import the bundled environment without editing or installing its sources."""
    reference = PROJECT_ROOT.parent / "diffusion_policy"
    if not (reference / "diffusion_policy" / "env" / "pusht").is_dir():
        raise FileNotFoundError(f"Push-T reference repository not found: {reference}")
    if str(reference) not in sys.path:
        sys.path.insert(0, str(reference))
    from diffusion_policy.env.pusht.pusht_env import PushTEnv

    if initial_state is not None:
        initial_state = np.asarray(initial_state, dtype=np.float64)
        if initial_state.shape != (5,) or not np.isfinite(initial_state).all():
            raise ValueError("initial_state must be five finite coordinates")
    env = PushTEnv(render_size=render_size, render_action=False,
                   reset_to_state=initial_state)
    env.seed(int(seed))
    return env


def coverage(env):
    """Actual geometric target coverage, the simulator's success criterion."""
    from diffusion_policy.env.pusht.pusht_env import pymunk_to_shapely

    target_body = env._get_goal_pose_body(env.goal_pose)
    target = pymunk_to_shapely(target_body, env.block.shapes)
    actual = pymunk_to_shapely(env.block, env.block.shapes)
    return float(target.intersection(actual).area / target.area)


def physical_metrics(states, goal, dt):
    """Metrics in simulation units; time-to-success is recorded by the runner."""
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 10 or len(states) == 0:
        raise ValueError("states must have shape (N, 10), N >= 1")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    goal = np.asarray(goal, dtype=float)
    displacement = np.diff(states[:, :2], axis=0)
    velocity_change = np.diff(states[:, 2:4], axis=0)
    accelerations = velocity_change / dt
    theta = states[-1, 8] - goal[2]
    return {
        "actuator_path_length": float(np.linalg.norm(displacement, axis=1).sum()),
        "velocity_total_variation": float(np.linalg.norm(velocity_change, axis=1).sum()),
        "acceleration_rms": float(np.sqrt(np.mean(np.sum(accelerations**2, axis=1))))
        if len(accelerations) else 0.0,
        "acceleration_peak": float(np.linalg.norm(accelerations, axis=1).max())
        if len(accelerations) else 0.0,
        "final_block_position_error": float(np.linalg.norm(states[-1, 4:6] - goal[:2])),
        "final_block_angle_error_deg": float(np.degrees(abs(np.arctan2(np.sin(theta), np.cos(theta))))),
    }
