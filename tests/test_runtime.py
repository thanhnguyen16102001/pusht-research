"""Online state and closed-loop reporting regressions."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from pusht_runtime import StateObserver, coverage, make_env, physical_metrics


class RuntimeTests(unittest.TestCase):
    def test_observer_uses_raw_angle_and_causal_finite_differences(self):
        observer = StateObserver(0.1)
        first = observer.observe([10, 20, 100, 200, 2 * np.pi - 0.01])
        second = observer.observe([11, 18, 103, 204, 0.01])
        np.testing.assert_allclose(first[[2, 3, 6, 7, 9]], 0)
        np.testing.assert_allclose(second[[2, 3, 6, 7, 9]], [10, -20, 30, 40, 0.2])
        self.assertAlmostEqual(second[8], 0.01)
        observer.reset()
        np.testing.assert_allclose(observer.observe([11, 18, 103, 204, 0.01])[[2, 3, 6, 7, 9]], 0)

    def test_bad_sample_does_not_advance_history(self):
        observer = StateObserver()
        observer.observe(np.zeros(5))
        with self.assertRaises(ValueError):
            observer.observe([1, 2, 3, 4, np.nan])
        self.assertEqual(observer.observe([1, 0, 0, 0, 0])[2], 10)

    def test_metrics_measure_vector_change_not_only_speed(self):
        states = np.zeros((3, 10))
        states[:, :2] = [[0, 0], [3, 4], [3, 9]]
        states[:, 2:4] = [[10, 0], [0, 10], [-10, 0]]
        goal = [0, 0, 2 * np.pi]
        metrics = physical_metrics(states, goal, 0.1)
        self.assertAlmostEqual(metrics["actuator_path_length"], 10)
        self.assertAlmostEqual(metrics["velocity_total_variation"], 20 * np.sqrt(2))
        self.assertAlmostEqual(metrics["acceleration_rms"], 100 * np.sqrt(2))
        self.assertAlmostEqual(metrics["final_block_angle_error_deg"], 0, places=5)

    def test_simulator_goal_and_timing_match_runtime(self):
        env = make_env(seed=0, initial_state=[100, 100, 256, 256, np.pi / 4])
        self.addCleanup(env.close)
        env.reset()
        self.assertEqual(env.control_hz, 10)
        self.assertEqual(env.sim_hz, 100)
        self.assertGreater(coverage(env), 0.95)
        observer = StateObserver()
        state = observer.from_info(env._get_info())
        np.testing.assert_allclose(state[[4, 5, 8]], env.goal_pose, atol=1e-5)

    def test_fallback_is_not_reported_as_a_world_model_prediction(self):
        from run_mpc import run_episode

        class FallbackController:
            def reset(self, seed=None):
                pass

            def plan(self, state, goal):
                return SimpleNamespace(action=state[:2].copy(), actions=state[None, :2],
                                       states=np.stack([state, state]), cost=float("inf"),
                                       components={}, planning_seconds=0.001, fallback=True)

        env = make_env(seed=0)
        self.addCleanup(env.close)
        metrics, data, _ = run_episode(env, FallbackController(), None, {"dt": 0.1},
                                       seed=0, max_steps=2)
        self.assertEqual(metrics["fallback_count"], 2)
        self.assertEqual(metrics["scored_model_predictions"], 0)
        self.assertIsNone(metrics["one_step_T_position_error_mean"])
        self.assertFalse(data["prediction_is_model"].any())
        self.assertEqual(data["states"].shape, (3, 10))
        self.assertEqual(data["actions"].shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
