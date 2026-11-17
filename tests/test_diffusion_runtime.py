"""Closed-loop imitation execution, completion, interruption, and persistence."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

import run_diffusion_policy as runner
from pusht_runtime import make_env


@dataclass
class FakeConfig:
    prediction_horizon: int = 4
    action_horizon: int = 2
    inference_steps: int = 10


class FakePolicy:
    """Hold commands with chunk diagnostics; records every observed state."""

    dt = 0.1

    def __init__(self):
        self.config = FakeConfig()
        self.reset_seeds = []
        self.seen = []

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self.seen = []

    def act(self, state):
        self.last_replanned = len(self.seen) % self.config.action_horizon == 0
        self.seen.append(state.copy())
        self.last_sampling_seconds = 0.002
        if self.last_replanned:
            self.last_plan = np.tile(state[:2], (self.config.prediction_horizon, 1))
        return state[:2].copy()


class ScriptedEnvironment:
    control_hz = 10
    success_threshold = 0.95
    action_space = SimpleNamespace(low=np.zeros(2), high=np.full(2, 512.0))

    def __init__(self, coverages=(0.96, 0.98, 0.99)):
        self.coverages = coverages
        self.index = 0
        self.current_seed = None
        self.seed = Mock(side_effect=self._seed)
        self.close = Mock()

    def _seed(self, seed):
        self.current_seed = seed

    def reset(self):
        self.index = 0

    def _get_info(self):
        return dict(pos_agent=np.array([50.0 + self.index, 50.0]),
                    block_pose=np.array([256.0 + self.index, 256.0, np.pi / 4]),
                    goal_pose=np.array([256.0, 256.0, np.pi / 4]), n_contacts=0)

    def step(self, action):
        self.index += 1
        return None, 1.0, self.coverages[self.index] > self.success_threshold, self._get_info()

    def render(self, mode="rgb_array"):
        return np.zeros((96, 96, 3), dtype=np.uint8)


def scripted_coverage(env):
    return env.coverages[env.index]


class DiffusionRuntimeTests(unittest.TestCase):
    def test_policy_observes_each_step_and_continues_past_original_done_to_exact_99(self):
        env = ScriptedEnvironment((0.96, 0.98, 0.9899, 0.99))
        policy = FakePolicy()
        with patch.object(runner, "coverage", side_effect=scripted_coverage):
            metrics, data, diagnostics = runner.run_episode(env, policy, 42, max_steps=6)
        self.assertTrue(metrics["success"])
        self.assertEqual(metrics["termination_reason"], "success")
        self.assertEqual(metrics["steps"], 3)
        self.assertEqual(policy.reset_seeds, [42])
        self.assertEqual(len(policy.seen), 3)
        np.testing.assert_allclose(np.asarray(policy.seen)[:, 2], [0, 10, 10])
        np.testing.assert_allclose(data["states"], np.r_[policy.seen, data["states"][-1:]])
        np.testing.assert_array_equal(data["chunk_start_steps"], [0, 2])
        np.testing.assert_allclose(data["sampling_seconds"], [0.002, 0.0, 0.002])
        self.assertEqual(data["sampled_action_chunks"].shape, (2, 4, 2))
        self.assertAlmostEqual(metrics["final_uncovered_area"], 63.0)
        self.assertNotIn("predicted_next_states", data)
        self.assertNotIn("planned_states", data)
        self.assertNotIn("cost", diagnostics)

    def test_stop_before_first_command_keeps_empty_action_shapes(self):
        env, policy = ScriptedEnvironment(), FakePolicy()
        # First event check runs before inference, second catches Q during inference.
        with patch.object(runner, "coverage", side_effect=scripted_coverage), patch.object(
                runner, "user_requested_stop", side_effect=[False, True]):
            metrics, data, _ = runner.run_episode(env, policy, 42)
        self.assertEqual(env.index, 0)
        self.assertTrue(metrics["stopped_by_user"])
        self.assertEqual(metrics["termination_reason"], "user_stop")
        self.assertEqual(data["states"].shape, (1, 10))
        self.assertEqual(data["actions"].shape, (0, 2))
        self.assertEqual(data["sampled_action_chunks"].shape, (0, 4, 2))
        self.assertEqual(data["chunk_start_steps"].shape, (0,))

    def test_interrupt_during_step_retains_only_completed_transitions(self):
        env, policy = ScriptedEnvironment(), FakePolicy()
        step = env.step

        def interrupted(action):
            result = step(action)
            if env.index == 2:
                raise KeyboardInterrupt
            return result

        with patch.object(runner, "coverage", side_effect=scripted_coverage), patch.object(
                env, "step", side_effect=interrupted):
            metrics, data, _ = runner.run_episode(env, policy, 42)
        self.assertEqual(metrics["termination_reason"], "keyboard_interrupt")
        self.assertEqual(metrics["steps"], 1)
        self.assertEqual(data["states"].shape, (2, 10))
        self.assertEqual(data["actions"].shape, (1, 2))
        self.assertAlmostEqual(metrics["final_coverage"], 0.98)

    def test_invalid_threshold_and_dt_rejected_before_reset(self):
        env, policy = Mock(control_hz=10), FakePolicy()
        for value in (0, -0.1, 1.01, np.nan, np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.run_episode(env, policy, 42, target_coverage=value)
        policy.dt = 0.05
        with self.assertRaisesRegex(ValueError, "control period"):
            runner.run_episode(env, policy, 42)
        env.reset.assert_not_called()

    def test_continuous_saves_successes_and_partial_episode_before_stop(self):
        env, policy = ScriptedEnvironment(), FakePolicy()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "policy.pt"
            checkpoint.write_bytes(b"fake policy for runner test")
            output = root / "run"

            def stop(*args):
                if env.current_seed == 43 and env.index == 1:
                    # Previous episodes must be persisted before starting a new one.
                    self.assertTrue((output / "episode_41.npz").exists())
                    self.assertTrue((output / "episode_42.json").exists())
                    return True
                return False

            with patch.object(runner, "load_policy", return_value=policy), patch.object(
                    runner, "make_env", return_value=env) as make, patch.object(
                    runner, "coverage", side_effect=scripted_coverage), patch.object(
                    runner, "user_requested_stop", side_effect=stop):
                summary = runner.main(["--checkpoint", str(checkpoint), "--continuous", "--start-seed", "41",
                                       "--output", str(output), "--no-plots"])
            make.assert_called_once()
            env.close.assert_called_once()
            self.assertEqual(policy.reset_seeds, [41, 42, 43])
            self.assertEqual(summary["num_completed_episodes"], 2)
            self.assertEqual(summary["num_interrupted_episodes"], 1)
            self.assertEqual(summary["success_rate"], 1.0)
            self.assertEqual(summary["run_status"], "stopped_by_user")
            self.assertEqual(json.loads((output / "summary.json").read_text()), summary)
            episode = json.loads((output / "episode_43.json").read_text())
            self.assertEqual(episode["termination_reason"], "user_stop")
            self.assertIn("policy_diagnostics", episode)
            self.assertNotIn("planner_diagnostics", episode)
            with np.load(output / "episode_43.npz") as trace:
                self.assertEqual(trace["actions"].shape, (1, 2))
                self.assertEqual(trace["states"].shape, (2, 10))
            self.assertFalse(list(output.glob("*.tmp")))

    def test_cli_overrides_sampling_and_action_horizons(self):
        env, policy = ScriptedEnvironment(), FakePolicy()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "policy.pt"
            checkpoint.write_bytes(b"test")
            with patch.object(runner, "load_policy", return_value=policy), patch.object(
                    runner, "make_env", return_value=env), patch.object(
                    runner, "coverage", side_effect=scripted_coverage):
                summary = runner.main(["--checkpoint", str(checkpoint), "--seeds", "42", "--no-plots",
                                       "--output", str(root / "run"), "--action-horizon", "1",
                                       "--inference-steps", "5"])
            self.assertEqual(summary["policy_config"]["action_horizon"], 1)
            self.assertEqual(summary["policy_config"]["inference_steps"], 5)
            self.assertEqual(summary["episodes"][0]["num_action_chunks"], 2)

    def test_original_environment_runs_without_a_world_model(self):
        env = make_env(seed=0)
        self.addCleanup(env.close)
        metrics, data, _ = runner.run_episode(env, FakePolicy(), 0, max_steps=1)
        self.assertEqual(metrics["steps"], 1)
        self.assertEqual(data["states"].shape, (2, 10))
        self.assertEqual(data["actions"].shape, (1, 2))
        self.assertTrue(np.isfinite(data["states"]).all())

    def test_cli_uses_original_environment_reset_before_reading_success_threshold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "policy.pt"
            checkpoint.write_bytes(b"fake policy for simulator lifecycle test")
            with patch.object(runner, "load_policy", return_value=FakePolicy()):
                summary = runner.main(["--checkpoint", str(checkpoint), "--seeds", "0", "--max-steps", "1",
                                       "--output", str(root / "run"), "--no-plots"])
            self.assertEqual(summary["original_success_threshold"], 0.95)
            self.assertEqual(summary["num_completed_episodes"], 1)
            self.assertEqual(summary["episodes"][0]["steps"], 1)
            self.assertTrue((root / "run/episode_0.npz").exists())

    def test_output_guard_protects_existing_episode(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            old = output / "episode_42.npz"
            old.write_bytes(b"preserve")
            with self.assertRaises(SystemExit):
                runner.main(["--output", str(output)])
            self.assertEqual(old.read_bytes(), b"preserve")


if __name__ == "__main__":
    unittest.main()
