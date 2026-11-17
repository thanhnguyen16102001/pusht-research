"""Push-T trajectory costs in physical simulation units.

State: agent x/y/vx/vy, block x/y/vx/vy/angle/angular velocity.
Trajectories include the current state: (..., H+1, 10). Calling total_cost
without config preserves the original three-term objective. Optional exact
polygon overlap measures the target area still uncovered by the predicted T.
"""
from dataclasses import dataclass, fields
import math
import torch

DT = 0.1
L = 512.0 * math.sqrt(2.0)
ACCEL_REF = 2000.0
Q = torch.diag(torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=torch.float32))


@dataclass
class CostConfig:
    dt: float = DT
    goal_weight: float = 1.0
    distance_weight: float = 0.02
    smoothness_weight: float = 0.01
    time_weight: float = 0.1
    approach_weight: float = 0.15
    consistency_weight: float = 0.1
    penetration_weight: float = 0.05
    overlap_weight: float = 0.0
    overlap_activation_coverage: float = 0.0
    position_scale: float = 100.0
    angle_scale: float = math.pi / 2
    acceleration_ref: float = ACCEL_REF
    terminal_weight: float = 0.7
    geometry_mix: float = 0.0
    goal_position_tolerance: float = 2.0
    goal_angle_tolerance: float = 0.02
    goal_hold_steps: int = 2
    goal_coverage: float = 0.0
    agent_radius: float = 15.0
    contact_margin: float = 10.0
    penetration_tolerance: float = 3.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{field.name} must be finite and nonnegative")
        for name in ("dt", "position_scale", "angle_scale", "acceleration_ref", "agent_radius"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.terminal_weight <= 1:
            raise ValueError("terminal_weight must be between zero and one")
        if not 0 <= self.geometry_mix <= 1:
            raise ValueError("geometry_mix must be between zero and one")
        if not 0 <= self.goal_coverage <= 1:
            raise ValueError("goal_coverage must be between zero and one (zero disables it)")
        if not 0 <= self.overlap_activation_coverage <= 1:
            raise ValueError("overlap_activation_coverage must be between zero and one")
        if not isinstance(self.goal_hold_steps, int) or self.goal_hold_steps < 1:
            raise ValueError("goal_hold_steps must be a positive integer")


def _validate(states, goal=None):
    if not isinstance(states, torch.Tensor) or not states.is_floating_point():
        raise ValueError("states must be a floating-point torch Tensor")
    if states.ndim not in (2, 3) or states.shape[-1] != 10 or states.shape[-2] < 2:
        raise ValueError("Expected states (H+1, 10) or (B, H+1, 10), H >= 1")
    if goal is not None:
        goal = torch.as_tensor(goal, dtype=states.dtype, device=states.device)
        if goal.shape != (3,) or not bool(torch.isfinite(goal).all()):
            raise ValueError("goal must contain finite block x, y, and angle")
    return goal


def _angle_error(angles, target):
    delta = angles - target
    return torch.atan2(delta.sin(), delta.cos())


def goal_cost(states, goal):
    """Original average/terminal normalized pose error."""
    goal = _validate(states, goal)
    position_error = ((states[..., 1:, 4:6] - goal[:2]) / L).square().sum(-1)
    angle_error = (_angle_error(states[..., 1:, 8], goal[2]) / math.pi).square()
    step_cost = (position_error + angle_error) / 2.0
    return (step_cost.mean(-1) + step_cost[..., -1]) / 2.0


def distance_cost(states):
    """Original mean distance per step divided by workspace diagonal."""
    _validate(states)
    displacement = states[..., 1:, :2] - states[..., :-1, :2]
    return torch.linalg.vector_norm(displacement, dim=-1).mean(-1) / L


def smoothness_cost(states, dt=DT, acceleration_ref=ACCEL_REF):
    """Changes in the velocity vector, including changes in direction."""
    _validate(states)
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(acceleration_ref) or acceleration_ref <= 0:
        raise ValueError("dt and acceleration_ref must be positive and finite")
    delta_velocity = states[..., 1:, 2:4] - states[..., :-1, 2:4]
    return (delta_velocity / (dt * acceleration_ref)).square().sum(-1).mean(-1)


def tee_signed_distance(agent_xy, block_xy, block_angle):
    """Point-to-T distance for PushTEnv's two local rectangles, negative inside.

    Body origin is not its centroid. Rectangles are [-60,60]x[0,30] and
    [-15,15]x[30,120], transformed using the block's body pose.
    """
    relative = agent_xy - block_xy
    cosine, sine = block_angle.cos(), block_angle.sin()
    local = torch.stack([cosine * relative[..., 0] + sine * relative[..., 1],
                         -sine * relative[..., 0] + cosine * relative[..., 1]], -1)

    vertices = local.new_tensor([[-60, 0], [60, 0], [60, 30], [15, 30],
                                 [15, 120], [-15, 120], [-15, 30], [-60, 30]])
    edges = torch.roll(vertices, -1, 0) - vertices
    relative_edge = local.unsqueeze(-2) - vertices
    fraction = (relative_edge * edges).sum(-1) / edges.square().sum(-1)
    nearest = vertices + fraction.clamp(0, 1).unsqueeze(-1) * edges
    distance = torch.linalg.vector_norm(local.unsqueeze(-2) - nearest, dim=-1).amin(-1)
    x, y = local[..., 0], local[..., 1]
    inside = (((x >= -60) & (x <= 60) & (y >= 0) & (y <= 30))
              | ((x >= -15) & (x <= 15) & (y >= 30) & (y <= 120)))
    return torch.where(inside, -distance, distance)


def _rectangle_intersection_area(vertices, bounds):
    """Batch clip CCW rectangles against axis-aligned rectangles, without rasterization.

    vertices: (N,4,2); bounds: (N,4) ordered xmin/xmax/ymin/ymax.
    Sutherland-Hodgman clipping adds at most one vertex at each half-plane.
    Padded vertices are excluded by counts; the final polygon has at most eight.
    """
    count = torch.full((len(vertices),), 4, dtype=torch.long, device=vertices.device)
    for axis, boundary_index, sign in ((0, 0, -1), (0, 1, 1), (1, 2, -1), (1, 3, 1)):
        capacity = vertices.shape[1]
        index = torch.arange(capacity, device=vertices.device).expand(len(vertices), -1)
        valid = index < count[:, None]
        previous_index = torch.remainder(index - 1, count[:, None].clamp_min(1))
        previous = vertices.gather(1, previous_index[..., None].expand(-1, -1, 2))
        bound = bounds[:, boundary_index, None]
        inside = (sign * (vertices[..., axis] - bound) <= 0) & valid
        previous_inside = sign * (previous[..., axis] - bound) <= 0
        crossing = (inside != previous_inside) & valid
        denominator = vertices[..., axis] - previous[..., axis]
        # Parallel/non-crossing edges are masked out; avoid 0/0 in their slots.
        safe_denominator = torch.where(crossing, denominator, torch.ones_like(denominator))
        fraction = ((bound - previous[..., axis]) / safe_denominator).clamp(0, 1)
        intersection = previous + fraction[..., None] * (vertices - previous)
        intersection[..., axis] = bound
        candidates = torch.stack((intersection, vertices), 2).flatten(1, 2)
        keep = torch.stack((crossing, inside), 2).flatten(1, 2)
        destination = (keep.long().cumsum(-1) - 1).clamp_min(0)
        output = vertices.new_zeros((len(vertices), capacity + 1, 2))
        output.scatter_add_(1, destination[..., None].expand(-1, -1, 2),
                            torch.where(keep[..., None], candidates, torch.zeros_like(candidates)))
        vertices, count = output, keep.sum(-1)
    index = torch.arange(vertices.shape[1], device=vertices.device).expand(len(vertices), -1)
    following_index = torch.remainder(index + 1, count[:, None].clamp_min(1))
    following = vertices.gather(1, following_index[..., None].expand(-1, -1, 2))
    cross = vertices[..., 0] * following[..., 1] - vertices[..., 1] * following[..., 0]
    return (0.5 * (cross * (index < count[:, None])).sum(-1)).clamp_min(0)


def tee_coverage(block_xy, block_angle, goal):
    """Exact geometric area(T intersect goal) / 6300 for arbitrary leading dims.

    block_xy is a floating Torch tensor (...,2); block_angle broadcasts to (...).
    Return same-device/dtype coverage in [0,1]. Nonfinite block poses return NaN
    individually, so total_cost can reject them without discarding other plans.
    The two rectangles in each T have disjoint interiors, hence the sum of four
    rectangle-pair intersection areas equals the T intersection area. Computation
    in the goal frame avoids subtracting large world-coordinate shoelace terms.
    No simulator, Shapely, image pixels, or sampling approximation is used.
    """
    if (not isinstance(block_xy, torch.Tensor) or not block_xy.is_floating_point()
            or block_xy.ndim < 1 or block_xy.shape[-1] != 2):
        raise ValueError("block_xy must be a floating Torch tensor with shape (...,2)")
    angle = torch.as_tensor(block_angle, dtype=block_xy.dtype, device=block_xy.device)
    target = torch.as_tensor(goal, dtype=block_xy.dtype, device=block_xy.device)
    if target.shape != (3,) or not bool(torch.isfinite(target).all()):
        raise ValueError("goal must contain finite block x, y, and angle")
    try:
        x, y, angle = torch.broadcast_tensors(block_xy[..., 0], block_xy[..., 1], angle)
    except RuntimeError as error:
        raise ValueError("block_angle must broadcast with block_xy leading dimensions") from error
    shape = angle.shape
    finite = torch.isfinite(x) & torch.isfinite(y) & torch.isfinite(angle)
    x = torch.where(finite, x - target[0], torch.zeros_like(x))
    y = torch.where(finite, y - target[1], torch.zeros_like(y))
    angle = torch.where(finite, angle - target[2], torch.zeros_like(angle))
    c, s = target[2].cos(), target[2].sin()
    translation = torch.stack((c * x + s * y, -s * x + c * y), -1).reshape(-1, 1, 1, 2)
    local = block_xy.new_tensor([[[-60, 0], [60, 0], [60, 30], [-60, 30]],
                                 [[-15, 30], [15, 30], [15, 120], [-15, 120]]])
    c, s = angle.cos().reshape(-1, 1, 1), angle.sin().reshape(-1, 1, 1)
    vertices = torch.stack((c * local[..., 0] - s * local[..., 1],
                            s * local[..., 0] + c * local[..., 1]), -1) + translation
    # Pair each predicted rectangle with both target rectangles.
    n = vertices.shape[0]
    vertices = vertices[:, :, None].expand(-1, -1, 2, -1, -1).reshape(-1, 4, 2)
    bounds = block_xy.new_tensor([[-60, 60, 0, 30], [-15, 15, 30, 120]])
    bounds = bounds[None, None].expand(n, 2, -1, -1).reshape(-1, 4)
    minimum, maximum = vertices.amin(1), vertices.amax(1)
    intersects = ((maximum[:, 0] > bounds[:, 0]) & (minimum[:, 0] < bounds[:, 1])
                  & (maximum[:, 1] > bounds[:, 2]) & (minimum[:, 1] < bounds[:, 3]))
    # AABB rejection is exact for zero-area pairs and saves most work far away.
    pair_area = vertices.new_zeros(len(vertices))
    pair_area[intersects] = _rectangle_intersection_area(vertices[intersects], bounds[intersects])
    area = pair_area.reshape(n, 2, 2).sum((1, 2))
    result = (area / 6300.0).reshape(shape).clamp(0, 1)
    return torch.where(finite, result, torch.full_like(result, float("nan")))


def _overlap_term(coverages, terminal_weight):
    uncovered = (1.0 - coverages).clamp(0, 1)
    return (1 - terminal_weight) * uncovered.mean(-1) + terminal_weight * uncovered[..., -1]


def overlap_cost(states, goal, config=None):
    """Mean/terminal fraction of green target area not covered by the gray T."""
    config = CostConfig() if config is None else config
    config = CostConfig(**config) if isinstance(config, dict) else config
    goal = _validate(states, goal)
    future = states[..., 1:, :]
    return _overlap_term(tee_coverage(future[..., 4:6], future[..., 8], goal), config.terminal_weight)


def _completion_from_within(within, config, dtype):
    suffix = torch.flip(torch.cumprod(torch.flip(within.to(torch.int64), [-1]), -1), [-1]).bool()
    horizon = within.shape[-1]
    indices = torch.arange(1, horizon + 1, device=within.device)
    enough_remaining = (horizon - indices + 1) >= config.goal_hold_steps
    arrival = torch.where(suffix & enough_remaining, indices, horizon + 1).amin(-1)
    return arrival.to(dtype) / float(horizon + 1)


def _refinement_coverages(states, goal, config):
    """Return active mask and initial/future coverage, avoiding inactive rollouts.

    Activation depends only on the observed initial state, not a candidate's
    imagined future. A zero threshold enables refinement everywhere. Mixed
    batches compute future polygon intersections only for active trajectories.
    """
    if not config.overlap_weight and not config.goal_coverage:
        return torch.zeros_like(states[..., 0, 8], dtype=torch.bool), None, None
    if not config.overlap_activation_coverage:
        coverage = tee_coverage(states[..., 4:6], states[..., 8], goal)
        return torch.ones_like(coverage[..., 0], dtype=torch.bool), coverage[..., 0], coverage[..., 1:]
    initial = tee_coverage(states[..., 0, 4:6], states[..., 0, 8], goal)
    active = initial >= config.overlap_activation_coverage
    if not bool(active.any()):
        return active, initial, None
    future = states[..., 1:, :]
    if bool(active.all()):
        covered = tee_coverage(future[..., 4:6], future[..., 8], goal)
    else:
        covered = torch.zeros_like(future[..., 8])
        selected = future[active]
        covered[active] = tee_coverage(selected[..., 4:6], selected[..., 8], goal)
    return active, initial, covered


def _completion_within(future, goal, config, refinement, covered):
    within = ((torch.linalg.vector_norm(future[..., 4:6] - goal[:2], dim=-1)
               <= config.goal_position_tolerance)
              & (_angle_error(future[..., 8], goal[2]).abs() <= config.goal_angle_tolerance))
    if config.goal_coverage and covered is not None:
        within = torch.where(refinement[..., None], covered >= config.goal_coverage, within)
    return within


def completion_time_cost(states, goal, config=None):
    """First sustained arrival / (H+1); unreached trajectories cost one.

    During active refinement, the candidate must remain at/above goal_coverage
    (when nonzero); otherwise legacy pose tolerances apply. The condition must
    hold through the horizon's end for at least goal_hold_steps.
    Reaching then leaving earns no time reward.
    """
    config = CostConfig() if config is None else config
    config = CostConfig(**config) if isinstance(config, dict) else config
    goal = _validate(states, goal)
    future = states[..., 1:, :]
    if config.goal_coverage:
        refinement, _, covered = _refinement_coverages(states, goal, config)
    else:
        refinement, covered = torch.zeros_like(states[..., 0, 8], dtype=torch.bool), None
    within = _completion_within(future, goal, config, refinement, covered)
    return _completion_from_within(within, config, states.dtype)


def cost_components(states, goal, config=None):
    """Unweighted dimensionless terms, one scalar per candidate trajectory."""
    config = CostConfig() if config is None else config
    config = CostConfig(**config) if isinstance(config, dict) else config
    goal = _validate(states, goal)
    future = states[..., 1:, :]
    refinement, initial_coverage, future_coverage = _refinement_coverages(states, goal, config)
    position = ((future[..., 4:6] - goal[:2]) / config.position_scale).square().sum(-1)
    angle = (_angle_error(future[..., 8], goal[2]) / config.angle_scale).square()
    pose = position + angle
    if config.geometry_mix:
        # Corresponding exterior vertices couple translation and rotation using
        # the actual asymmetric T geometry, rather than a centroid approximation.
        local = states.new_tensor([[-60, 0], [60, 0], [60, 30], [15, 30],
                                    [15, 120], [-15, 120], [-15, 30], [-60, 30]])
        c, s = future[..., 8, None].cos(), future[..., 8, None].sin()
        predicted = torch.stack([c * local[:, 0] - s * local[:, 1],
                                 s * local[:, 0] + c * local[:, 1]], -1)
        predicted = predicted + future[..., None, 4:6]
        c, s = goal[2].cos(), goal[2].sin()
        target = torch.stack([c * local[:, 0] - s * local[:, 1],
                              s * local[:, 0] + c * local[:, 1]], -1) + goal[:2]
        geometry = ((predicted - target) / config.position_scale).square().sum(-1).mean(-1)
        pose = (1 - config.geometry_mix) * pose + config.geometry_mix * geometry
    goal_term = ((1 - config.terminal_weight) * pose.mean(-1)
                 + config.terminal_weight * pose[..., -1])
    displacement = states[..., 1:, :2] - states[..., :-1, :2]
    path = torch.linalg.vector_norm(displacement, dim=-1).sum(-1) / L
    gap = tee_signed_distance(states[..., :2], states[..., 4:6], states[..., 8]) - config.agent_radius
    initial_error = torch.linalg.vector_norm(states[..., 0, 4:6] - goal[:2], dim=-1)
    initial_angle = _angle_error(states[..., 0, 8], goal[2]).abs()
    active = ((initial_error > config.goal_position_tolerance)
              | (initial_angle > config.goal_angle_tolerance)).to(states.dtype)
    if config.goal_coverage:
        active = torch.where(refinement, (initial_coverage < config.goal_coverage).to(states.dtype), active)
    approach = active * (gap[..., 1:].clamp_min(0) / config.position_scale).square().mean(-1)
    # Soft physical-consistency heuristics, not an exact collision model.
    far = torch.minimum(gap[..., :-1], gap[..., 1:]) > config.contact_margin
    block_motion = ((states[..., 1:, 4:6] - states[..., :-1, 4:6]) / 10.0).square().sum(-1)
    block_turn = (_angle_error(states[..., 1:, 8], states[..., :-1, 8]) / 0.1).square()
    consistency = ((block_motion + block_turn) * far.to(states.dtype)).mean(-1)
    penetration = ((-gap[..., 1:] - config.penetration_tolerance).clamp_min(0)
                   / config.agent_radius).square().mean(-1)
    within = _completion_within(future, goal, config, refinement, future_coverage)
    time_term = _completion_from_within(within, config, states.dtype)
    overlap = (torch.where(refinement, _overlap_term(future_coverage, config.terminal_weight),
                           torch.zeros_like(goal_term))
               if future_coverage is not None else torch.zeros_like(goal_term))
    return {"goal": goal_term, "distance": path,
            "smoothness": smoothness_cost(states, config.dt, config.acceleration_ref),
            "time": time_term, "approach": approach,
            "consistency": consistency, "penetration": penetration, "overlap": overlap}


def total_cost(states, goal, config=None, return_components=False):
    """Cost per trajectory; invalid predictions receive infinity for rejection.

    Without config, preserves the original equal-weight three-term objective.
    """
    _validate(states, goal)
    if config is None:
        components = {"goal": goal_cost(states, goal), "distance": distance_cost(states),
                      "smoothness": smoothness_cost(states)}
        result = (torch.stack(list(components.values()), -1) * Q.diagonal().to(states)).sum(-1)
    else:
        config = CostConfig(**config) if isinstance(config, dict) else config
        components = cost_components(states, goal, config)
        result = sum(value * getattr(config, f"{name}_weight") for name, value in components.items())
    finite = torch.isfinite(states).all(dim=-1).all(dim=-1) & torch.isfinite(result)
    result = torch.where(finite, result, torch.full_like(result, float("inf")))
    return (result, components) if return_components else result


if __name__ == "__main__":
    trajectory = torch.zeros(4, 10)
    trajectory[:, 0] = torch.tensor([100.0, 110.0, 120.0, 130.0])
    trajectory[:, 2] = 100.0
    trajectory[:, 4:6] = torch.tensor([250.0, 250.0])
    value, terms = total_cost(trajectory, [256.0, 256.0, math.pi / 4],
                              CostConfig(), return_components=True)
    print({name: float(term) for name, term in terms.items()})
    print("Total MPC cost:", float(value))
