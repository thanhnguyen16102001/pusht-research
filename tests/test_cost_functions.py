"""Objective semantics: time, vector smoothness, geometry, and compatibility."""
import math
from pathlib import Path
import sys
import unittest

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from CostFunctions import (CostConfig, completion_time_cost, cost_components, distance_cost,
                           goal_cost, smoothness_cost, tee_signed_distance, total_cost)


class CostTests(unittest.TestCase):
    def setUp(self):
        self.states = torch.zeros(7, 10)
        self.states[:, 4:6] = torch.tensor([200., 200.])
        self.goal = torch.tensor([256., 256., 0.])

    def test_original_three_term_call_unchanged(self):
        self.states[:, 0] = torch.arange(7) * 10
        self.states[:, 2] = torch.arange(7) * 20
        expected = (goal_cost(self.states, self.goal) + distance_cost(self.states)
                    + smoothness_cost(self.states)) / 3
        torch.testing.assert_close(total_cost(self.states, self.goal), expected)
        batch = self.states.unsqueeze(0).repeat(3, 1, 1)
        torch.testing.assert_close(total_cost(batch, self.goal), expected.repeat(3))

    def test_earlier_stable_completion_is_better_and_leaving_is_not_completion(self):
        early, late, leaves = [self.states.clone() for _ in range(3)]
        early[1:, 4:6] = self.goal[:2]
        late[4:, 4:6] = self.goal[:2]
        leaves[1:4, 4:6] = self.goal[:2]
        values = completion_time_cost(torch.stack([early, late, leaves]), self.goal)
        self.assertLess(values[0], values[1])
        self.assertLess(values[1], values[2])
        self.assertEqual(float(values[2]), 1.)

    def test_smoothness_penalizes_direction_change_even_at_constant_speed(self):
        steady, reversing = self.states.clone(), self.states.clone()
        steady[:, 2] = 100
        reversing[:, 2] = torch.tensor([100., -100., 100., -100., 100., -100., 100.])
        self.assertEqual(float(smoothness_cost(steady)), 0.)
        self.assertGreater(float(smoothness_cost(reversing)), 0.)

    def test_tee_surface_respects_body_origin_and_rotation(self):
        xy = torch.tensor([[0., 100.], [-100., 0.], [100., 100.]])
        block = torch.zeros_like(xy)
        distance = tee_signed_distance(xy, block, torch.tensor([0., math.pi / 2, 0.]))
        torch.testing.assert_close(distance[:2], torch.tensor([-15., -15.]), atol=1e-5, rtol=1e-5)
        self.assertGreater(float(distance[2]), 0.)
        # The rectangles' shared seam is inside the union, not a T boundary.
        seam = tee_signed_distance(torch.tensor([0., 30.]), torch.zeros(2), torch.tensor(0.))
        self.assertAlmostEqual(float(seam), -15.)

    def test_geometry_goal_cost_respects_equivalent_rotations(self):
        self.states[:, 4:6] = self.goal[:2]
        self.states[:, 8] = 2 * math.pi
        values = cost_components(self.states, self.goal, CostConfig(geometry_mix=1.0))
        self.assertLess(float(values['goal']), 1e-10)

    def test_motion_without_contact_is_penalized(self):
        self.states[:, :2] = 10
        still = cost_components(self.states, self.goal)
        moving = self.states.clone()
        moving[:, 4] += torch.arange(7) * 10
        imagined = cost_components(moving, self.goal)
        self.assertEqual(float(still['consistency']), 0.)
        self.assertGreater(float(imagined['consistency']), 0.)

    def test_invalid_trajectories_rejected_and_bad_configuration_fails(self):
        self.states[-1, 4] = float('nan')
        self.assertTrue(torch.isinf(total_cost(self.states, self.goal, CostConfig())))
        with self.assertRaises(ValueError):
            CostConfig(dt=0)
        with self.assertRaises(ValueError):
            total_cost(torch.zeros(10), self.goal)


if __name__ == '__main__':
    unittest.main()
