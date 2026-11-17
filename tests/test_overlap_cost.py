"""Exact T overlap, finishing objective, and coverage-based completion."""

import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from CostFunctions import (CostConfig, completion_time_cost, cost_components,
                           overlap_cost, tee_coverage, total_cost)


VERTICES = np.array([[-60., 0.], [60., 0.], [60., 30.], [15., 30.],
                     [15., 120.], [-15., 120.], [-15., 30.], [-60., 30.]])


def polygon(pose):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    rotation = np.array([[c, -s], [s, c]])
    return Polygon(VERTICES @ rotation.T + np.asarray(pose[:2]))


class OverlapCostTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_exact_full_zero_and_small_remaining_gap(self):
        goal = torch.tensor([256., 256., 0.], dtype=torch.float64)
        offsets = torch.tensor([0., 0.001, 0.1, 1., 5.], dtype=torch.float64)
        positions = goal[:2].expand(5, 2).clone()
        positions[:, 0] += offsets
        covered = tee_coverage(positions, torch.zeros(5, dtype=torch.float64), goal)
        # For 0 <= dx <= 30, uncovered area is (30+90)*dx = 120*dx.
        torch.testing.assert_close(covered, 1 - 120 * offsets / 6300, rtol=1e-11, atol=1e-11)
        self.assertTrue(bool((covered[:-1] > covered[1:]).all()))
        self.assertEqual(float(tee_coverage(goal[:2] + 1000, goal[2], goal)), 0.)
        self.assertEqual(float(tee_coverage(goal[:2], goal[2], goal)), 1.)

    def test_random_and_near_goal_poses_match_independent_shapely_oracle(self):
        rng = np.random.default_rng(77)
        goal = np.array([256., 256., math.pi / 4])
        poses = np.column_stack([rng.uniform(100, 410, (100, 2)), rng.uniform(-8, 8, 100)])
        near = goal + rng.normal(size=(100, 3)) * [2, 2, 0.04]
        tiny = goal + rng.normal(size=(30, 3)) * [0.001, 0.001, 1e-5]
        poses = np.concatenate([poses, near, tiny, goal[None], [goal + [0, 0, 2 * math.pi]]])
        target = polygon(goal)
        oracle = np.array([polygon(p).intersection(target).area / target.area for p in poses])
        for dtype, tolerance in ((torch.float64, 5e-12), (torch.float32, 1.5e-6)):
            tensor = torch.tensor(poses, dtype=dtype)
            measured = tee_coverage(tensor[:, :2], tensor[:, 2], torch.tensor(goal, dtype=dtype))
            np.testing.assert_allclose(measured.numpy(), oracle, rtol=0, atol=tolerance)

    def test_coverage_is_symmetric_periodic_and_rigid_transform_invariant(self):
        a = torch.tensor([245., 260., 0.74], dtype=torch.float64)
        b = torch.tensor([256., 256., 0.81], dtype=torch.float64)
        expected = tee_coverage(a[:2], a[2], b)
        torch.testing.assert_close(tee_coverage(b[:2], b[2], a), expected)
        torch.testing.assert_close(tee_coverage(a[:2], a[2] + 2 * math.pi, b), expected)
        phi = torch.tensor(1.31, dtype=torch.float64)
        rotation = torch.stack([torch.stack([phi.cos(), -phi.sin()]),
                                torch.stack([phi.sin(), phi.cos()])])
        shift = torch.tensor([-400., 800.], dtype=torch.float64)
        aa = torch.cat([rotation @ a[:2] + shift, (a[2] + phi)[None]])
        bb = torch.cat([rotation @ b[:2] + shift, (b[2] + phi)[None]])
        torch.testing.assert_close(tee_coverage(aa[:2], aa[2], bb), expected)

    def test_batch_shape_nonfinite_isolation_and_input_validation(self):
        goal = torch.tensor([256., 256., math.pi / 4])
        xy = goal[:2].expand(2, 3, 2).clone()
        angles = goal[2].expand(2, 3).clone()
        xy[0, 0, 0] = float("nan")
        angles[1, 2] = float("inf")
        result = tee_coverage(xy, angles, goal)
        self.assertEqual(result.shape, (2, 3))
        self.assertTrue(torch.isnan(result[0, 0]))
        self.assertTrue(torch.isnan(result[1, 2]))
        torch.testing.assert_close(result[0, 1:], torch.ones(2))
        with self.assertRaises(ValueError):
            tee_coverage(torch.ones(4, 3), 0., goal)
        with self.assertRaises(ValueError):
            tee_coverage(torch.ones(4, 2), torch.zeros(5), goal)
        with self.assertRaises(ValueError):
            CostConfig(goal_coverage=1.001)
        with self.assertRaises(ValueError):
            CostConfig(overlap_weight=-1.)

    def test_terminal_uncovered_area_and_weight_enter_cost(self):
        goal = torch.tensor([256., 256., 0.])
        states = torch.zeros(2, 5, 10)
        states[..., 4:6] = goal[:2]
        states[..., :2] = 10
        states[0, :, 4] += 2.
        states[1, 0, 4] += 2.
        states[1, 1:, 4] += torch.tensor([1.5, 1., 0.5, 0.])
        config = CostConfig(overlap_weight=5., terminal_weight=0.8)
        values = overlap_cost(states, goal, config)
        self.assertGreater(float(values[0]), float(values[1]))
        expected = 0.2 * (120 * torch.tensor([1.5, 1., .5, 0.]) / 6300).mean()
        torch.testing.assert_close(values[1], expected, atol=1e-7, rtol=1e-5)
        no_overlap = CostConfig(terminal_weight=0.8)
        with_overlap = total_cost(states, goal, config)
        without = total_cost(states, goal, no_overlap)
        torch.testing.assert_close(with_overlap - without, 5 * values)
        invalid = states.clone()
        invalid[0, -1, 4] = float("nan")
        result = total_cost(invalid, goal, config)
        self.assertTrue(torch.isinf(result[0]))
        self.assertTrue(torch.isfinite(result[1]))

    def test_completion_requires_coverage_and_staying_covered(self):
        goal = torch.tensor([256., 256., 0.])
        states = torch.zeros(4, 7, 10)
        states[..., 4:6] = goal[:2]
        states[..., 4] += 1.  # Within old pose tolerance but only 98.095% covered.
        states[0, 1:, 4] = 256.
        states[1, 4:, 4] = 256.
        states[2, 1:4, 4] = 256.  # Arrives then leaves.
        config = CostConfig(goal_coverage=.995, goal_hold_steps=2)
        time = completion_time_cost(states, goal, config)
        torch.testing.assert_close(time, torch.tensor([1/7, 4/7, 1., 1.]))
        terms = cost_components(states, goal, config)
        torch.testing.assert_close(terms["time"], time)
        self.assertTrue(bool((terms["approach"] > 0).all()))

    def test_old_config_skips_exact_overlap_work(self):
        states = torch.zeros(4, 10)
        goal = torch.tensor([256., 256., 0.])
        with patch("CostFunctions.tee_coverage", side_effect=AssertionError("Unexpected coverage call")):
            terms = cost_components(states, goal, CostConfig())
        self.assertEqual(float(terms["overlap"]), 0.)

    def test_inactive_refinement_preserves_original_cost_without_future_clipping(self):
        goal = torch.tensor([256., 256., 0.])
        states = torch.zeros(2, 5, 10)
        states[..., :2] = 100.
        states[..., 4:6] = 50.
        states[1, :, 4] += torch.arange(5)
        old = CostConfig()
        phased = CostConfig(overlap_weight=5., goal_coverage=.995,
                            overlap_activation_coverage=.90)
        with patch("CostFunctions.tee_coverage", wraps=tee_coverage) as geometric:
            terms = cost_components(states, goal, phased)
        self.assertEqual(geometric.call_count, 1)
        self.assertEqual(tuple(geometric.call_args.args[0].shape), (2, 2))
        torch.testing.assert_close(terms["overlap"], torch.zeros(2))
        torch.testing.assert_close(total_cost(states, goal, phased),
                                   total_cost(states, goal, old), rtol=0, atol=0)
        torch.testing.assert_close(completion_time_cost(states, goal, phased),
                                   completion_time_cost(states, goal, old), rtol=0, atol=0)

    def test_mixed_refinement_batch_matches_individual_scores_and_only_clips_active_futures(self):
        goal = torch.tensor([256., 256., 0.])
        states = torch.zeros(2, 6, 10)
        states[..., :2] = 100.
        states[0, :, 4:6] = 50.
        states[1, :, 4:6] = goal[:2]
        states[1, :, 4] += torch.tensor([1., 1., .5, 0., 0., 0.])
        phased = CostConfig(overlap_weight=5., goal_coverage=.995,
                            overlap_activation_coverage=.90)
        with patch("CostFunctions.tee_coverage", wraps=tee_coverage) as geometric:
            batch, terms = total_cost(states, goal, phased, return_components=True)
        shapes = [tuple(call.args[0].shape) for call in geometric.call_args_list]
        self.assertEqual(shapes, [(2, 2), (1, 5, 2)])
        individual = torch.stack([total_cost(state, goal, phased) for state in states])
        torch.testing.assert_close(batch, individual)
        self.assertEqual(float(terms["overlap"][0]), 0.)
        self.assertGreater(float(terms["overlap"][1]), 0.)
        torch.testing.assert_close(terms["time"], completion_time_cost(states, goal, phased))
        always_active = CostConfig(overlap_weight=5., goal_coverage=.995)
        torch.testing.assert_close(batch[1], total_cost(states[1], goal, always_active))
        with self.assertRaises(ValueError):
            CostConfig(overlap_activation_coverage=1.01)


if __name__ == "__main__":
    unittest.main()
