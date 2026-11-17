"""Continuous-run lifecycle, interruption, and incremental persistence checks."""

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent
import numpy as np

from episode_kinematics import EpisodePlotter
from pusht_runtime import make_env
import run_mpc


class HoldController:
    def reset(self, seed=None):
        pass

    def plan(self, state, goal):
        return SimpleNamespace(action=state[:2].copy(), actions=state[None, :2],
                               states=np.stack([state, state]), cost=0.0,
                               components={}, planning_seconds=0.001, fallback=False)


class ContinuousRuntimeTests(unittest.TestCase):
    def test_plot_reset_reuses_window_and_restarts_velocity_history(self):
        with patch.object(plt, "show"):
            plotter = EpisodePlotter(realtime_plotting=True)
        self.addCleanup(plotter.close)
        figure = plotter.fig
        plotter.update([10, 20, 100, 200, 0], [15, 25])
        plotter.update([11, 20, 100, 200, 0], [16, 25])
        self.assertEqual(plotter.episode.num_steps, 2)
        self.assertTrue(plotter.reset(episode_id=7))
        self.assertIs(plotter.fig, figure)
        self.assertIsNone(plotter.episode)
        self.assertTrue(all(len(line.get_xdata()) == 0 for plot in plotter.plots for line in plot["lines"]))
        plotter.update([300, 400, 100, 200, 0], [305, 405])
        self.assertEqual(plotter.episode.episode_id, 7)
        np.testing.assert_array_equal(plotter.episode.state_augmented[0, [2, 3, 6, 7, 9]], 0)
        np.testing.assert_array_equal(plotter.episode.time_nominal, [0])

    def test_plot_q_stops_before_another_action_and_keeps_zero_step_shapes(self):
        with patch.object(plt, "show"):
            plotter = EpisodePlotter(realtime_plotting=True)
        self.addCleanup(plotter.close)
        env = make_env(seed=0)
        self.addCleanup(env.close)
        update = plotter.update

        def quit_on_update(*args, **kwargs):
            result = update(*args, **kwargs)
            plotter.fig.canvas.callbacks.process(
                "key_press_event", KeyEvent("key_press_event", plotter.fig.canvas, key="q"))
            return result

        with patch.object(plotter, "update", side_effect=quit_on_update):
            metrics, data, _ = run_mpc.run_episode(
                env, HoldController(), None, {"dt": 0.1}, 0, max_steps=3,
                realtime_plotting=True, plotter=plotter)
        self.assertEqual(metrics["termination_reason"], "user_stop")
        self.assertTrue(metrics["stopped_by_user"])
        self.assertEqual(data["states"].shape, (1, 10))
        self.assertEqual(data["actions"].shape, (0, 2))
        self.assertEqual(data["planned_actions"].shape, (0, 1, 2))
        self.assertEqual(data["planned_states"].shape, (0, 2, 10))

    def test_ctrl_c_during_environment_step_keeps_only_committed_transitions(self):
        env = make_env(seed=0)
        self.addCleanup(env.close)
        step = env.step
        calls = 0

        def interrupted_step(action):
            nonlocal calls
            calls += 1
            result = step(action)
            if calls == 2:
                raise KeyboardInterrupt
            return result

        with patch.object(env, "step", side_effect=interrupted_step):
            metrics, data, diagnostic = run_mpc.run_episode(
                env, HoldController(), None, {"dt": 0.1}, 0, max_steps=3)
        self.assertEqual(metrics["termination_reason"], "keyboard_interrupt")
        self.assertEqual(metrics["steps"], 1)
        self.assertEqual(data["states"].shape, (2, 10))
        self.assertEqual(data["actions"].shape, (1, 2))
        self.assertEqual(data["predicted_next_states"].shape, (1, 10))
        self.assertEqual(len(diagnostic["cost"]), 1)

    def test_ctrl_c_during_plot_finalization_still_returns_episode(self):
        with patch.object(plt, "show"):
            plotter = EpisodePlotter(realtime_plotting=True)
        self.addCleanup(plotter.close)
        env = make_env(seed=0)
        self.addCleanup(env.close)
        # update is allowed; only the final explicit refresh is interrupted.
        with patch.object(plotter, "update", return_value=True), patch.object(
                plotter, "refresh", side_effect=KeyboardInterrupt):
            metrics, data, _ = run_mpc.run_episode(
                env, HoldController(), None, {"dt": 0.1}, 0, max_steps=1, plotter=plotter)
        self.assertEqual(metrics["termination_reason"], "keyboard_interrupt")
        self.assertEqual(data["states"].shape, (2, 10))
        self.assertEqual(data["actions"].shape, (1, 2))

    def test_continuous_saves_each_episode_reuses_env_and_excludes_interruption(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"test-checkpoint")
            env = SimpleNamespace(action_space=SimpleNamespace(low=np.zeros(2), high=np.full(2, 512)),
                                  seed=Mock(), close=Mock())
            seen = []

            def episode(current_env, controller, model, stats, seed, *args, **kwargs):
                self.assertIs(current_env, env)
                before = json.loads((output / "summary.json").read_text())
                self.assertEqual(before["num_episodes"], len(seen))
                for previous in seen:
                    self.assertTrue((output / f"episode_{previous}.npz").exists())
                    self.assertTrue((output / f"episode_{previous}.json").exists())
                seen.append(seed)
                stopped = seed == 43
                success = seed == 41
                metrics = dict(seed=seed, success=success, stopped_by_user=stopped, steps=1,
                               termination_reason="user_stop" if stopped else "success" if success else "time_limit",
                               max_coverage=0.99 if success else 0.3, planning_ms_p95=1.0,
                               completion_time_seconds=0.1 if success else None,
                               actuator_path_length=1.0, acceleration_rms=2.0)
                return metrics, dict(states=np.zeros((2, 10)), actions=np.zeros((1, 2))), {}

            with patch.object(run_mpc, "make_env", return_value=env) as make, patch.object(
                    run_mpc, "run_episode", side_effect=episode), patch(
                    "Model_MLP_OneStepPredict.load_model", return_value=(None, {"dt": 0.1})), patch(
                    "MPC_Controller.MPCController", return_value=HoldController()):
                summary = run_mpc.main([
                    "--continuous", "--start-seed", "41", "--max-steps", "1", "--no-plots",
                    "--checkpoint", str(checkpoint), "--config", str(project / "configs/mpc_default.json"),
                    "--output", str(output)])
            make.assert_called_once()
            env.close.assert_called_once()
            self.assertEqual(seen, [41, 42, 43])
            self.assertEqual([call.args[0] for call in env.seed.call_args_list], seen)
            self.assertEqual(summary["run_status"], "stopped_by_user")
            self.assertEqual(summary["num_completed_episodes"], 2)
            self.assertEqual(summary["num_interrupted_episodes"], 1)
            self.assertEqual(summary["success_rate"], 0.5)
            self.assertEqual(json.loads((output / "summary.json").read_text()), summary)
            self.assertFalse(list(output.glob("*.tmp")))

    def test_output_guard_also_detects_episode_files_without_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "episode_1.npz").write_bytes(b"existing")
            with self.assertRaises(SystemExit):
                run_mpc.main(["--output", str(output)])
            self.assertEqual((output / "episode_1.npz").read_bytes(), b"existing")

    def test_later_failure_preserves_saved_episode_and_marks_summary_error(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"test-checkpoint")
            env = SimpleNamespace(action_space=SimpleNamespace(low=np.zeros(2), high=np.full(2, 512)),
                                  seed=Mock(), close=Mock())
            metrics = dict(seed=41, success=False, stopped_by_user=False, steps=1,
                           termination_reason="time_limit", max_coverage=0.3, planning_ms_p95=1.0,
                           completion_time_seconds=None, actuator_path_length=1.0, acceleration_rms=2.0)
            first = (metrics, dict(states=np.zeros((2, 10)), actions=np.zeros((1, 2))), {})
            with patch.object(run_mpc, "make_env", return_value=env), patch.object(
                    run_mpc, "run_episode", side_effect=[first, RuntimeError("simulator failed")]), patch(
                    "Model_MLP_OneStepPredict.load_model", return_value=(None, {"dt": 0.1})), patch(
                    "MPC_Controller.MPCController", return_value=HoldController()):
                with self.assertRaisesRegex(RuntimeError, "simulator failed"):
                    run_mpc.main([
                        "--seeds", "41", "42", "--no-plots", "--checkpoint", str(checkpoint),
                        "--config", str(project / "configs/mpc_default.json"), "--output", str(output)])
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["run_status"], "error")
            self.assertEqual(summary["num_completed_episodes"], 1)
            self.assertTrue((output / "episode_41.npz").exists())
            self.assertTrue((output / "episode_41.json").exists())
            env.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
