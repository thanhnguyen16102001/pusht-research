"""Planner regression tests using controlled, explicitly known toy dynamics."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from MPC_Controller import MPCConfig, MPCController
from CostFunctions import CostConfig, total_cost


def toy_rollout(model, stats, state, actions):
    state = state.expand(len(actions), 10).clone()
    history = [state]
    for action in actions.unbind(1):
        following = state.clone()
        delta = action - state[:, :2]
        following[:, :2] = action
        following[:, 2:4] = delta / stats['dt']
        following[:, 4:6] = state[:, 4:6] + 0.5 * delta
        following[:, 6:8] = 0.5 * delta / stats['dt']
        history.append(following)
        state = following
    return torch.stack(history, 1)


class PlannerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = torch.nn.Linear(1, 1)
        self.state = np.array([100, 100, 0, 0, 200, 200, 0, 0, 0, 0], dtype=np.float32)
        self.goal = [250, 250, 0]
        self.config = MPCConfig(horizon=5, population=64, iterations=4,
                                contact_candidates=False, max_action_delta=50)
        self.cost = CostConfig(distance_weight=0, smoothness_weight=0, time_weight=0,
                               approach_weight=0, consistency_weight=0, penetration_weight=0)

    def make_controller(self, **kwargs):
        return MPCController(self.model, {'dt': .1}, self.config, self.cost,
                             rollout_fn=kwargs.get('rollout_fn', toy_rollout))

    def test_cem_improves_over_hold_and_returns_a_scored_plan(self):
        controller = self.make_controller()
        result = controller.plan(self.state, self.goal)
        hold = torch.as_tensor(self.state[:2]).expand(1, self.config.horizon, 2)
        hold_states = toy_rollout(self.model, {'dt': .1}, torch.as_tensor(self.state), hold)
        hold_cost = float(total_cost(hold_states, self.goal, self.cost)[0])
        self.assertFalse(result.fallback)
        self.assertLess(result.cost, hold_cost * .5)
        rescored = float(total_cost(torch.as_tensor(result.states), self.goal, self.cost))
        self.assertAlmostEqual(result.cost, rescored, places=5)
        np.testing.assert_array_equal(result.action, result.actions[0])
        np.testing.assert_array_equal(result.states[0], self.state)
        self.assertGreater(result.valid_candidates, 0)

    def test_action_bounds_and_increment_limits(self):
        controller = self.make_controller()
        result = controller.plan(self.state, [500, 500, 0])
        self.assertTrue(np.all(result.actions >= 0) and np.all(result.actions <= 512))
        differences = np.diff(np.vstack([self.state[:2], result.actions]), axis=0)
        self.assertLessEqual(float(np.linalg.norm(differences, axis=-1).max()), 50.0001)

    def test_reset_is_reproducible_and_warm_start_remains_valid(self):
        controller = self.make_controller()
        first = controller.plan(self.state, self.goal)
        following = controller.plan(first.states[1], self.goal)
        self.assertEqual(following.actions.shape, (5, 2))
        controller.reset()
        repeated = controller.plan(self.state, self.goal)
        np.testing.assert_array_equal(first.actions, repeated.actions)

    def test_all_invalid_predictions_return_explicit_hold_fallback(self):
        def broken(model, stats, state, actions):
            return torch.full((len(actions), actions.shape[1] + 1, 10), float('nan'))
        controller = self.make_controller(rollout_fn=broken)
        result = controller.plan(self.state, self.goal)
        self.assertTrue(result.fallback)
        self.assertTrue(np.isinf(result.cost))
        np.testing.assert_array_equal(result.action, self.state[:2])
        self.assertEqual(result.components, {})

    def test_invalid_state_and_mismatched_timing_fail(self):
        with self.assertRaises(ValueError):
            MPCController(self.model, {'dt': .01}, self.config, self.cost, rollout_fn=toy_rollout)
        controller = self.make_controller()
        with self.assertRaises(ValueError):
            controller.plan(np.zeros(5), self.goal)
        self.state[0] = np.nan
        with self.assertRaises(ValueError):
            controller.plan(self.state, self.goal)

    def test_planning_does_not_modify_model_parameters(self):
        before = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        self.make_controller().plan(self.state, self.goal)
        for name, parameter in self.model.named_parameters():
            torch.testing.assert_close(parameter, before[name])
            self.assertIsNone(parameter.grad)

    def test_contact_routes_and_adaptive_search_remain_bounded(self):
        self.config.contact_candidates = True
        self.config.route_candidates = True
        self.config.adaptive_sampling = True
        self.config.population = 128
        controller = self.make_controller()
        proposals = controller._contact_proposals(torch.as_tensor(self.state), torch.tensor(self.goal))
        self.assertGreater(len(proposals), 25)
        self.assertTrue(bool(torch.isfinite(proposals).all()))
        result = controller.plan(self.state, self.goal)
        self.assertFalse(result.fallback)
        self.assertTrue(np.all(result.actions >= 0) and np.all(result.actions <= 512))

    def test_precision_contact_closes_staging_gap_for_small_push(self):
        self.config.contact_candidates = True
        self.config.horizon = 20
        controller = self.make_controller()
        # T is two units right of target; actuator is by its right stem face.
        state = torch.tensor([234., 260., 0., 0., 202., 200., 0., 0., 0., 0.])
        goal = torch.tensor([200., 200., 0.])
        original = controller._contact_proposals(state, goal)
        self.config.precision_contact = True
        refined = controller._contact_proposals(state, goal)
        # Contact index 8 is local (15,60) with normal (+1,0), two approach times.
        first_right_stem = 1 + 8 * 2
        original_end = original[first_right_stem, -1]
        refined_end = refined[first_right_stem, -1]
        torch.testing.assert_close(original_end, torch.tensor([233., 260.]))
        torch.testing.assert_close(refined_end, torch.tensor([230., 260.]))
        # Original endpoint is 16 units from the current face (radius=15), so
        # it cannot push; corrected endpoint crosses current contact by 2 units.
        self.assertGreater(float(original_end[0] - (state[4] + 15)), 15.)
        self.assertLess(float(refined_end[0] - (state[4] + 15)), 15.)
        # Staging segment and the hold candidate keep their existing behavior.
        torch.testing.assert_close(refined[first_right_stem, :7], original[first_right_stem, :7])
        torch.testing.assert_close(refined[0], original[0])

    def test_precision_contact_keeps_far_goal_proposals_unchanged(self):
        self.config.contact_candidates = True
        self.config.route_candidates = True
        controller = self.make_controller()
        original = controller._contact_proposals(torch.as_tensor(self.state), torch.tensor(self.goal))
        self.config.precision_contact = True
        refined = controller._contact_proposals(torch.as_tensor(self.state), torch.tensor(self.goal))
        torch.testing.assert_close(refined, original, rtol=0, atol=0)

    def test_refinement_gate_preserves_far_plan_and_uses_fine_sampling_near_goal(self):
        baseline = self.make_controller().plan(self.state, self.goal)
        self.config.precision_contact = True
        self.config.adaptive_sampling = True
        self.config.refinement_min_std = 0.5
        self.cost.overlap_weight = 5.0
        self.cost.goal_coverage = 0.995
        self.cost.overlap_activation_coverage = 0.90
        controller = self.make_controller()
        self.assertFalse(controller._refinement_active(torch.tensor(self.state), torch.tensor(self.goal)))
        result = controller.plan(self.state, self.goal)
        np.testing.assert_array_equal(result.actions, baseline.actions)
        self.assertAlmostEqual(result.cost, baseline.cost, places=6)
        state = torch.tensor(self.state)
        state[4:6] = torch.tensor(self.goal[:2]) + 1
        self.assertTrue(controller._refinement_active(state, torch.tensor(self.goal)))
        with self.assertRaises(ValueError):
            MPCConfig(refinement_min_std=0)


if __name__ == '__main__':
    unittest.main()
