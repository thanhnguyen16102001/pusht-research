"""CEM model-predictive control using the learned Push-T World Model.

The optimizer searches absolute actuator target-XY sequences. It returns the
best sequence actually evaluated, executes no simulator steps, and never updates
model weights. A caller executes the first action and replans from a real state.
"""
from dataclasses import asdict, dataclass
import math
from time import perf_counter
from typing import Dict, Optional

import numpy as np
import torch

try:
    from .CostFunctions import CostConfig, tee_coverage, tee_signed_distance, total_cost
except ImportError:
    from CostFunctions import CostConfig, tee_coverage, tee_signed_distance, total_cost


@dataclass
class MPCConfig:
    horizon: int = 10
    population: int = 256
    iterations: int = 4
    elite_fraction: float = 0.1
    initial_std: float = 70.0
    min_std: float = 5.0
    refinement_min_std: Optional[float] = None
    smoothing: float = 0.2
    noise_correlation: float = 0.8
    max_action_delta: float = 60.0
    max_predicted_speed: float = 2500.0
    prediction_margin: float = 100.0
    contact_candidates: bool = True
    route_candidates: bool = False
    adaptive_sampling: bool = False
    precision_contact: bool = False
    seed: int = 42

    def __post_init__(self):
        for name in ("horizon", "population", "iterations"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.population < 4:
            raise ValueError("population must be at least four")
        for name in ("initial_std", "min_std", "max_action_delta", "max_predicted_speed"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.min_std > self.initial_std:
            raise ValueError("min_std cannot exceed initial_std")
        if self.refinement_min_std is not None and (
                not math.isfinite(self.refinement_min_std)
                or not 0 < self.refinement_min_std <= self.initial_std):
            raise ValueError("refinement_min_std must be positive and no larger than initial_std")
        if not 0 < self.elite_fraction <= 1:
            raise ValueError("elite_fraction must be in (0, 1]")
        if not 0 <= self.smoothing < 1 or not 0 <= self.noise_correlation < 1:
            raise ValueError("smoothing and noise_correlation must be in [0, 1)")
        if not math.isfinite(self.prediction_margin) or self.prediction_margin < 0:
            raise ValueError("prediction_margin must be finite and nonnegative")


@dataclass
class PlanResult:
    action: np.ndarray
    actions: np.ndarray
    states: np.ndarray
    cost: float
    components: Dict[str, float]
    planning_seconds: float
    fallback: bool = False
    valid_candidates: int = 0


class MPCController:
    """Sampling MPC with warm starts, smooth noise, and contact-directed proposals.

    Contact proposals are only candidate actions, not an alternate controller:
    every proposal is rolled out by the learned model and ranked by the same cost.
    Bounds and model-consistency heuristics do not constitute hardware guarantees.
    """

    def __init__(self, model, stats, config=None, cost_config=None,
                 action_low=None, action_high=None, rollout_fn=None):
        self.model = model
        self.stats = stats
        self.config = MPCConfig(**config) if isinstance(config, dict) else (config or MPCConfig())
        self.cost_config = (CostConfig(**cost_config) if isinstance(cost_config, dict)
                            else (cost_config or CostConfig()))
        if not math.isclose(float(stats["dt"]), self.cost_config.dt, rel_tol=1e-6):
            raise ValueError("Checkpoint dt and cost dt must match")
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")
        self.model.eval()
        self.low = torch.as_tensor([0.0, 0.0] if action_low is None else action_low,
                                   dtype=torch.float32, device=self.device)
        self.high = torch.as_tensor([512.0, 512.0] if action_high is None else action_high,
                                    dtype=torch.float32, device=self.device)
        if (self.low.shape != (2,) or self.high.shape != (2,)
                or not bool(torch.isfinite(self.low).all() & torch.isfinite(self.high).all())
                or not bool((self.high > self.low).all())):
            raise ValueError("Action bounds must be finite 2-vectors, high > low")
        if rollout_fn is None:
            try:
                from .Model_MLP_OneStepPredict import rollout
            except ImportError:
                from Model_MLP_OneStepPredict import rollout
            rollout_fn = rollout
        self._rollout = rollout_fn
        self.generator = torch.Generator(device=self.device)
        self.reset()

    def reset(self, seed=None):
        """Start a new episode without carrying plans or RNG state across episodes."""
        self.generator.manual_seed(self.config.seed if seed is None else int(seed))
        self._previous_actions = None
        self._previous_action = None

    def _bound_actions(self, actions, anchor):
        bounded = actions.clamp(self.low, self.high).clone()
        previous = anchor.clamp(self.low, self.high)
        for step in range(self.config.horizon):
            difference = bounded[:, step] - previous
            length = torch.linalg.vector_norm(difference, dim=-1, keepdim=True)
            difference = difference * (self.config.max_action_delta / length.clamp_min(1e-8)).clamp_max(1)
            bounded[:, step] = (previous + difference).clamp(self.low, self.high)
            previous = bounded[:, step]
        return bounded

    def _noise(self):
        noise = torch.randn((self.config.population, self.config.horizon, 2),
                            generator=self.generator, device=self.device)
        rho = self.config.noise_correlation
        for step in range(1, self.config.horizon):
            noise[:, step] = rho * noise[:, step - 1] + math.sqrt(1 - rho * rho) * noise[:, step]
        return noise

    def _refinement_active(self, state, goal):
        """Use precision sampling only once the current T is sufficiently covered."""
        threshold = self.cost_config.overlap_activation_coverage
        return not threshold or float(tee_coverage(state[4:6], state[8], goal)) >= threshold

    def _contact_proposals(self, state, goal):
        """Geometry-informed starting guesses; all evaluated by learned dynamics."""
        h = self.config.horizon
        start = state[:2]
        proposals = [start.expand(h, 2).clone()]
        if not self.config.contact_candidates:
            return torch.stack(proposals)
        local = state.new_tensor([
            [-45, 0], [0, 0], [45, 0], [-60, 15], [60, 15],
            [-40, 30], [40, 30], [-15, 60], [15, 60],
            [-15, 100], [15, 100], [0, 120]])
        normals = state.new_tensor([
            [0, -1], [0, -1], [0, -1], [-1, 0], [1, 0],
            [0, 1], [0, 1], [-1, 0], [1, 0], [-1, 0], [1, 0], [0, 1]])

        def rotate(points, angle):
            c, s = angle.cos(), angle.sin()
            return torch.stack([c * points[:, 0] - s * points[:, 1],
                                s * points[:, 0] + c * points[:, 1]], -1)

        surface = state[4:6] + rotate(local, state[8])
        world_normals = rotate(normals, state[8])
        contact = surface + world_normals * (self.cost_config.agent_radius + 3.0)
        desired_surface = goal[:2] + rotate(local, goal[2])
        push = desired_surface - surface
        if self.config.precision_contact and self._refinement_active(state, goal):
            position_error = (state[4:6] - goal[:2]).norm()
            angle_error = torch.atan2((state[8] - goal[2]).sin(),
                                      (state[8] - goal[2]).cos()).abs()
            if float(position_error + 120.0 * angle_error) <= 20.0:
                # The 3-unit approach clearance otherwise consumes corrections
                # smaller than 3 units: the agent never reaches the T. On faces
                # that can push inward, finish at the agent center tangent to
                # the goal surface, including its desired normal orientation.
                inward = (push * world_normals).sum(-1) < -1e-6
                desired_contact = (desired_surface
                                   + rotate(normals, goal[2]) * self.cost_config.agent_radius)
                push = torch.where(inward[:, None], desired_contact - contact, push)
        push *= (80.0 / torch.linalg.vector_norm(push, dim=-1, keepdim=True).clamp_min(1e-8)).clamp_max(1)
        for target, push_vector in zip(contact, push):
            # Different approach durations allow both immediate pushing and reaching.
            for approach_fraction in (0.35, 0.7):
                split = max(1, min(h, round(h * approach_fraction)))
                path = start.new_empty((h, 2))
                alpha = torch.arange(1, split + 1, device=self.device)[:, None] / split
                path[:split] = start + alpha * (target - start)
                if split < h:
                    alpha = torch.arange(1, h - split + 1, device=self.device)[:, None] / (h - split)
                    path[split:] = target + alpha * push_vector
                proposals.append(path)
            if self.config.route_candidates and h >= 4:
                # Reach the far side along either arc around an enclosing circle,
                # instead of requiring a straight approach through the T body.
                center = state[4:6] + rotate(state.new_tensor([[0, 45]]), state[8])[0]
                from_center, to_center = start - center, target - center
                angle_start = torch.atan2(from_center[1], from_center[0])
                angle_end = torch.atan2(to_center[1], to_center[0])
                difference = torch.atan2((angle_end-angle_start).sin(), (angle_end-angle_start).cos())
                radius = torch.stack([from_center.norm(), to_center.norm(), state.new_tensor(98.)]).max()
                for delta in (difference, difference - torch.sign(difference + 1e-6) * (2 * math.pi)):
                    split = max(3, min(h - 1, round(h * .7)))
                    path = start.new_empty((h, 2))
                    alpha = torch.arange(1, split, device=self.device) / (split - 1)
                    angles = angle_start + alpha * delta
                    radii = from_center.norm() + alpha * (radius - from_center.norm())
                    path[:split - 1] = center + radii[:, None] * torch.stack([angles.cos(), angles.sin()], -1)
                    path[split - 1] = target
                    alpha = torch.arange(1, h - split + 1, device=self.device)[:, None] / (h - split)
                    path[split:] = target + alpha * push_vector
                    proposals.append(path)
        return torch.stack(proposals)

    def _valid_predictions(self, trajectories):
        valid = torch.isfinite(trajectories).all(-1).all(-1)
        future = trajectories[:, 1:]
        xy = future[..., [0, 1, 4, 5]].reshape(future.shape[0], future.shape[1], 2, 2)
        lower = self.low - self.config.prediction_margin
        upper = self.high + self.config.prediction_margin
        valid &= ((xy >= lower) & (xy <= upper)).all(-1).all(-1).all(-1)
        speeds = future[..., [2, 3, 6, 7]].reshape(future.shape[0], future.shape[1], 2, 2)
        valid &= (torch.linalg.vector_norm(speeds, dim=-1) <= self.config.max_predicted_speed).all(-1).all(-1)
        return valid

    def _fallback(self, state, started):
        # No imagined progress is returned when every candidate is invalid.
        action = state[:2].clamp(self.low, self.high)
        actions = action.expand(self.config.horizon, 2).cpu().numpy().copy()
        states = state.expand(self.config.horizon + 1, 10).cpu().numpy().copy()
        self._previous_actions = None
        self._previous_action = action.clone()
        return PlanResult(action.cpu().numpy().copy(), actions, states, float("inf"),
                          {}, perf_counter() - started, fallback=True)

    @torch.no_grad()
    def plan(self, state, goal):
        """Return the best evaluated action sequence, without advancing an environment."""
        started = perf_counter()
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        goal = torch.as_tensor(goal, dtype=torch.float32, device=self.device)
        if state.shape != (10,) or not bool(torch.isfinite(state).all()):
            raise ValueError("state must contain ten finite physical-unit values")
        if goal.shape != (3,) or not bool(torch.isfinite(goal).all()):
            raise ValueError("goal must contain finite x, y, angle")
        h = self.config.horizon
        anchor = state[:2] if self._previous_action is None else self._previous_action
        if self._previous_actions is None:
            mean = state[:2].expand(h, 2).clone()
        else:
            mean = torch.cat([self._previous_actions[1:], self._previous_actions[-1:]], 0)
        deviation = self.config.initial_std
        refining = self._refinement_active(state, goal)
        minimum_std = (self.config.refinement_min_std if refining
                       and self.config.refinement_min_std is not None else self.config.min_std)
        if self.config.adaptive_sampling and refining:
            position_error = (state[4:6] - goal[:2]).norm()
            angle_error = torch.atan2((state[8] - goal[2]).sin(), (state[8] - goal[2]).cos()).abs()
            gap = (tee_signed_distance(state[:2], state[4:6], state[8])
                   - self.cost_config.agent_radius).clamp_min(0)
            deviation = min(deviation, max(minimum_std * 2,
                                           float(position_error * .5 + angle_error * 40 + gap * .5)))
        std = torch.full_like(mean, deviation)
        priors = self._contact_proposals(state, goal)
        best_actions = best_states = best_terms = None
        best_cost = float("inf")
        valid_count = 0
        elites = max(2, min(self.config.population, round(self.config.population * self.config.elite_fraction)))

        for _ in range(self.config.iterations):
            actions = mean[None] + std[None] * self._noise()
            actions[0] = mean
            # Keep a scored incumbent and contact guesses in the search population.
            offset = 1
            if best_actions is not None:
                actions[offset] = best_actions
                offset += 1
            n_priors = min(len(priors), max(0, self.config.population // 4 - offset))
            if n_priors:
                actions[offset:offset + n_priors] = priors[:n_priors]
            actions = self._bound_actions(actions, anchor)
            trajectories = self._rollout(self.model, self.stats, state, actions)
            expected = (self.config.population, h + 1, 10)
            if trajectories.shape != expected:
                raise ValueError(f"World Model rollout must return {expected}")
            costs, terms = total_cost(trajectories, goal, self.cost_config, return_components=True)
            valid = self._valid_predictions(trajectories) & torch.isfinite(costs)
            costs = torch.where(valid, costs, torch.full_like(costs, float("inf")))
            valid_count += int(valid.sum())
            count = min(elites, int(valid.sum()))
            if count == 0:
                continue
            indices = torch.topk(costs, count, largest=False).indices
            minimum = float(costs[indices[0]])
            if minimum < best_cost:
                best_cost = minimum
                best_actions = actions[indices[0]].clone()
                best_states = trajectories[indices[0]].clone()
                best_terms = {name: float(value[indices[0]]) for name, value in terms.items()}
            selected = actions[indices]
            alpha = self.config.smoothing
            mean = alpha * mean + (1 - alpha) * selected.mean(0)
            std = (alpha * std + (1 - alpha) * selected.std(0, unbiased=False)).clamp_min(minimum_std)

        if best_actions is None:
            return self._fallback(state, started)
        self._previous_actions = best_actions.clone()
        self._previous_action = best_actions[0].clone()
        return PlanResult(best_actions[0].cpu().numpy().copy(), best_actions.cpu().numpy(),
                          best_states.cpu().numpy(), best_cost, best_terms,
                          perf_counter() - started, valid_candidates=valid_count)

    def configuration(self):
        return {"mpc": asdict(self.config), "cost": asdict(self.cost_config)}
