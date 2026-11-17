"""Requested overlap completion must not stop at the simulator's old 95%."""

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

import run_mpc
from pusht_runtime import coverage, make_env


class HoldController:
    config = SimpleNamespace(horizon=1)

    def reset(self, seed=None):
        pass

    def plan(self, state, goal):
        return SimpleNamespace(action=state[:2].copy(), actions=state[None, :2],
                               states=np.stack([state, state]), cost=0.0,
                               components={}, planning_seconds=0.001, fallback=False)


class ScriptedEnvironment:
    control_hz = 10
    success_threshold = 0.95
    action_space = SimpleNamespace(low=np.zeros(2), high=np.full(2, 512.0))

    def __init__(self, coverages):
        self.coverages = coverages
        self.index = 0

    def reset(self):
        self.index = 0

    def _get_info(self):
        return dict(pos_agent=np.array([50.0, 50.0]),
                    block_pose=np.array([256.0 + self.index, 256.0, np.pi / 4]),
                    goal_pose=np.array([256.0, 256.0, np.pi / 4]), n_contacts=0)

    def step(self, action):
        self.index += 1
        return None, 1.0, self.coverages[self.index] > self.success_threshold, self._get_info()


class CoverageRuntimeTests(unittest.TestCase):
    def run_scripted(self, coverages, target=None, max_steps=5):
        env = ScriptedEnvironment(coverages)
        with patch.object(run_mpc, "coverage", side_effect=lambda value: value.coverages[value.index]):
            result = run_mpc.run_episode(env, HoldController(), None, {"dt": 0.1}, 42,
                                         max_steps=max_steps, target_coverage=target)
        return env, result

    def test_strict_target_continues_after_original_done_until_coverage_reached(self):
        env, (metrics, data, _) = self.run_scripted([0.96, 0.98, 0.994, 0.995], target=0.995)
        self.assertEqual(env.index, 3)
        self.assertEqual(env.success_threshold, 0.95)
        self.assertTrue(metrics["success"])
        self.assertTrue(metrics["original_environment_success"])
        self.assertEqual(metrics["termination_reason"], "success")
        self.assertEqual(metrics["target_coverage"], 0.995)
        self.assertEqual(metrics["target_coverage_comparison"], ">=")
        self.assertAlmostEqual(metrics["final_uncovered_area"], 31.5)
        np.testing.assert_allclose(data["uncovered_fraction"], [0.04, 0.02, 0.006, 0.005])
        np.testing.assert_allclose(data["uncovered_area"], data["uncovered_fraction"] * 6300)
        self.assertEqual(data["states"].shape, (4, 10))
        self.assertEqual(data["actions"].shape, (3, 2))

    def test_timeout_reports_original_success_separately_from_requested_success(self):
        _, (metrics, _, _) = self.run_scripted([0.97, 0.985], target=0.995, max_steps=1)
        self.assertFalse(metrics["success"])
        self.assertTrue(metrics["original_environment_success"])
        self.assertTrue(metrics["original_environment_reached"])
        self.assertIsNone(metrics["completion_time_seconds"])
        summary = run_mpc.build_summary({}, [metrics], "finished")
        self.assertEqual(summary["success_rate"], 0.0)
        self.assertEqual(summary["original_environment_success_rate"], 1.0)
        self.assertAlmostEqual(summary["mean_final_uncovered_area"], 94.5)

    def test_default_retains_original_strict_threshold(self):
        env, (metrics, _, _) = self.run_scripted([0.95, 0.951])
        self.assertEqual(env.index, 1)
        self.assertTrue(metrics["success"])
        self.assertEqual(metrics["target_coverage_comparison"], ">")
        env, (metrics, _, _) = self.run_scripted([0.96])
        self.assertEqual(env.index, 0)
        self.assertTrue(metrics["success"])

    def test_full_coverage_roundoff_does_not_hide_meaningful_gap(self):
        self.assertTrue(run_mpc.coverage_reached(1.0 - 5e-10, 1.0))
        self.assertFalse(run_mpc.coverage_reached(1.0 - 1e-7, 1.0))
        self.assertFalse(run_mpc.coverage_reached(0.995 - 5e-10, 0.995))

    def test_original_environment_keeps_threshold_while_runner_refines(self):
        env = make_env(seed=0, initial_state=[100, 100, 257, 256, np.pi / 4])
        self.addCleanup(env.close)
        env.reset()
        initial = coverage(env)
        self.assertGreater(initial, 0.95)
        self.assertLess(initial, 0.995)
        metrics, data, _ = run_mpc.run_episode(env, HoldController(), None, {"dt": 0.1},
                                               0, max_steps=1, target_coverage=0.995)
        self.assertEqual(metrics["steps"], 1)
        self.assertTrue(metrics["original_environment_success"])
        self.assertFalse(metrics["success"])
        self.assertEqual(env.success_threshold, 0.95)
        self.assertEqual(data["target_area"], 6300)

    def test_invalid_target_is_rejected_before_reset(self):
        for target in (0.0, -0.1, 1.01, np.nan, np.inf):
            env = Mock()
            with self.subTest(target=target), self.assertRaises(ValueError):
                run_mpc.run_episode(env, None, None, {"dt": 0.1}, 0, target_coverage=target)
            env.reset.assert_not_called()

    def test_cli_resolves_config_and_override_for_both_planner_and_runner(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint = directory / "checkpoint.pt"
            checkpoint.write_bytes(b"test")
            config = json.loads((project / "configs/mpc_default.json").read_text())
            config["cost"]["goal_coverage"] = 0.995
            config_path = directory / "config.json"
            config_path.write_text(json.dumps(config))
            for override in (None, 0.997):
                expected = 0.995 if override is None else override
                env = SimpleNamespace(action_space=ScriptedEnvironment.action_space,
                                      success_threshold=0.95, seed=Mock(), close=Mock())
                metrics = dict(seed=42, success=False, stopped_by_user=False, steps=0,
                               termination_reason="time_limit", max_coverage=0.97, planning_ms_p95=0.0,
                               completion_time_seconds=None, actuator_path_length=0.0, acceleration_rms=0.0)
                args = ["--seeds", "42", "--no-plots", "--output", str(directory / str(expected)),
                        "--checkpoint", str(checkpoint), "--config", str(config_path)]
                if override is not None:
                    args += ["--target-coverage", str(override)]
                with patch.object(run_mpc, "make_env", return_value=env), patch.object(
                        run_mpc, "run_episode", return_value=(metrics, {}, {})) as episode, patch(
                        "Model_MLP_OneStepPredict.load_model", return_value=(None, {"dt": 0.1})), patch(
                        "MPC_Controller.MPCController", return_value=HoldController()) as controller:
                    summary = run_mpc.main(args)
                self.assertEqual(episode.call_args.kwargs["target_coverage"], expected)
                self.assertEqual(controller.call_args.kwargs["cost_config"].goal_coverage, expected)
                self.assertEqual(summary["target_coverage"], expected)


if __name__ == "__main__":
    unittest.main()
