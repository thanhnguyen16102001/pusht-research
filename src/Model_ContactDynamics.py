"""Equivariant learned Push-T dynamics with explicit contact geometry features.

This model evaluates no simulator. It uses known T geometry and actuator radius
as input features and conservatively suppresses impossible stationary-block
motion when the predicted swept actuator path cannot reach the block.
"""

import torch
from torch import nn


class ContactDynamicsModel(nn.Module):
    architecture = "contact_local_v2"

    def __init__(self, stats):
        super().__init__()
        for name in ("x_mean", "x_std", "y_mean", "y_std"):
            self.register_buffer(name, torch.as_tensor(stats[name], dtype=torch.float32).clone())
        self.dt = float(stats["dt"])
        # Scalar coefficients of two vectors preserve rotation/translation
        # equivariance while learning the actuator's response from demonstrations.
        self.agent_net = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, 32),
                                       nn.SiLU(), nn.Linear(32, 2))
        nn.init.zeros_(self.agent_net[-1].weight)
        nn.init.constant_(self.agent_net[-1].bias, -0.85)
        self.block_net = nn.Sequential(nn.Linear(19, 128), nn.SiLU(),
                                       nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 3))
        self.register_buffer("block_output_scale", torch.tensor([5.0, 5.0, 0.1]))
        self.register_buffer("path_fractions", torch.linspace(0.0, 1.0, 5))

    @staticmethod
    def to_local(vector, cosine, sine):
        return torch.stack((cosine*vector[..., 0]+sine*vector[..., 1],
                            -sine*vector[..., 0]+cosine*vector[..., 1]), dim=-1)

    @staticmethod
    def to_world(vector, cosine, sine):
        return torch.stack((cosine*vector[..., 0]-sine*vector[..., 1],
                            sine*vector[..., 0]+cosine*vector[..., 1]), dim=-1)

    @staticmethod
    def _box_distance(points, center, halfsize):
        center = points.new_tensor(center)
        halfsize = points.new_tensor(halfsize)
        relative = points-center
        q = relative.abs()-halfsize
        signed = torch.linalg.vector_norm(q.clamp_min(0), dim=-1)+q.max(dim=-1).values.clamp_max(0)
        closest = relative.clamp(-halfsize, halfsize)+center
        return signed, closest

    @classmethod
    def surface_geometry(cls, points):
        """Signed distance to the union of the original T's two rectangles."""
        bar_distance, bar_point = cls._box_distance(points, (0., 15.), (60., 15.))
        stem_distance, stem_point = cls._box_distance(points, (0., 75.), (15., 45.))
        choose_bar = bar_distance <= stem_distance
        signed = torch.where(choose_bar, bar_distance, stem_distance)
        closest = torch.where(choose_bar[..., None], bar_point, stem_point)
        normal = points-closest
        normal = normal/torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(1e-6)
        return signed, closest, normal

    def physical_delta(self, state, action):
        """Predict physical [agent_dx,agent_dy,T_dx,T_dy,T_dtheta]."""
        target_delta = action-state[:, :2]
        agent_velocity = state[:, 2:4]
        travel_from_velocity = agent_velocity*self.dt
        invariants = torch.stack((torch.linalg.vector_norm(target_delta, dim=-1)/128.,
                                  torch.linalg.vector_norm(travel_from_velocity, dim=-1)/32.,
                                  (target_delta*travel_from_velocity).sum(-1)/(128.*32.)), dim=-1)
        coefficients = self.agent_net(invariants).sigmoid()
        agent_delta = coefficients[:, :1]*target_delta+coefficients[:, 1:]*travel_from_velocity
        angle = state[:, 8]
        cosine, sine = angle.cos(), angle.sin()
        position = self.to_local(state[:, :2]-state[:, 4:6], cosine, sine)
        command = self.to_local(target_delta, cosine, sine)
        local_agent_velocity = self.to_local(agent_velocity, cosine, sine)
        local_block_velocity = self.to_local(state[:, 6:8], cosine, sine)
        local_delta = self.to_local(agent_delta, cosine, sine)
        signed, closest, normal = self.surface_geometry(position)
        gap = signed-15.0
        end_gap = self.surface_geometry(position+local_delta)[0]-15.0
        lever = closest-position.new_tensor([0., 45.])
        torque_feature = lever[:, 0]*local_agent_velocity[:, 1]-lever[:, 1]*local_agent_velocity[:, 0]
        features = torch.cat((position/150., command/150., local_agent_velocity/150.,
                              local_block_velocity/150., state[:, 9:10]/5., local_delta/30.,
                              closest/120., normal, gap[:, None]/60., torque_feature[:, None]/3000.,
                              (normal*local_agent_velocity).sum(-1, keepdim=True)/150.,
                              end_gap[:, None]/60.), dim=-1)
        block_delta_local = self.block_net(features)*self.block_output_scale

        # A Lipschitz lower bound on clearance along the five-sample swept
        # segment. Additional velocity allowance covers a curved actuator path.
        path = position[:, None]+self.path_fractions[None, :, None]*local_delta[:, None]
        path_gap = self.surface_geometry(path)[0]-15.0
        spacing_allowance = torch.linalg.vector_norm(local_delta, dim=-1)/8.0
        curve_allowance = torch.linalg.vector_norm(travel_from_velocity-agent_delta, dim=-1)/4.0
        clearance_lower_bound = path_gap.min(-1).values-spacing_allowance-curve_allowance
        contact_gate = ((5.0-clearance_lower_bound)/5.0).clamp(0., 1.)
        block_motion = torch.linalg.vector_norm(state[:, 6:8], dim=-1)*self.dt+state[:, 9].abs()*self.dt*120.
        gate = torch.where(block_motion > 0.05, torch.ones_like(contact_gate), contact_gate)
        block_delta_local = block_delta_local*gate[:, None]
        block_delta_world = self.to_world(block_delta_local[:, :2], cosine, sine)
        return torch.cat((agent_delta, block_delta_world, block_delta_local[:, 2:3]), dim=-1)

    def forward(self, normalized_input):
        physical_input = normalized_input*self.x_std+self.x_mean
        delta = self.physical_delta(physical_input[:, :10], physical_input[:, 10:])
        return (delta-self.y_mean)/self.y_std
