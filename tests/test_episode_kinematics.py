"""Headless regression checks for recorded and live episode visualization.

Run from the project directory with:
    python -B -m unittest discover -s tests -v
"""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent
import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import episode_kinematics as kinematics


class EpisodeKinematicsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(plt.close, "all")
        # The angle crosses 2*pi, so naive subtraction would create a spike.
        self.states = np.array([
            [10.0, 20.0, 100.0, 200.0, 6.25],
            [10.2, 20.4, 100.1, 200.2, 0.01],
            [10.7, 20.9, 100.3, 200.1, 0.04],
            [11.1, 21.3, 100.5, 200.0, 0.07],
        ])
        self.actions = np.array([
            [15.0, 25.0], [16.0, 26.0], [17.0, 27.0], [18.0, 28.0],
        ])

    def make_dataset(self, states=None, actions=None, episode_ends=None):
        states = self.states if states is None else states
        actions = self.actions if actions is None else actions
        if episode_ends is None:
            episode_ends = [len(states)]
        path = Path(self.temp.name) / "episode.zarr"
        root = zarr.open_group(str(path), mode="w")
        data = root.create_group("data")
        data.array("state", states)
        data.array("action", actions)
        data.array("img", np.zeros((len(states), 8, 8, 3), dtype=np.uint8))
        root.create_group("meta").array("episode_ends", np.array(episode_ends))
        return path

    @staticmethod
    def signal_axis(plotter, title):
        return next(ax for ax in plotter.fig.axes if ax.get_title() == title)

    @staticmethod
    def signal_line(axis, label):
        return next(line for line in axis.lines if line.get_label() == label)

    def assert_no_future_points(self, plotter, now):
        # Includes both signal traces and current-value markers/cursors.
        for axis in plotter.fig.axes:
            for line in axis.lines:
                times = np.asarray(line.get_xdata(), dtype=float)
                if times.size:
                    self.assertLessEqual(float(times.max()), now + 1e-12)

    def test_recorded_default_remains_10_hz_and_angles_unwrap(self):
        episode = kinematics.load_episode_kinematics(self.make_dataset())
        self.assertEqual(episode.dt_nominal, 0.1)
        np.testing.assert_allclose(episode.time_nominal, [0.0, 0.1, 0.2, 0.3])
        np.testing.assert_allclose(episode.agent.vel.x, [0.0, 2.0, 5.0, 4.0])
        np.testing.assert_allclose(
            episode.block.omega,
            [0.0, (2 * np.pi + 0.01 - 6.25) / 0.1, 0.3, 0.3],
        )

    def test_velocities_restart_at_episode_boundary(self):
        states = self.states.copy()
        states[2:, :4] += 10000.0
        path = self.make_dataset(states=states, episode_ends=[2, 4])
        episode = kinematics.load_episode_kinematics(path, episode_id=1)
        self.assertEqual(episode.num_steps, 2)
        np.testing.assert_array_equal(episode.raw_state, states[2:])
        np.testing.assert_array_equal(episode.state_augmented[0, [2, 3, 6, 7, 9]], 0.0)
        np.testing.assert_allclose(episode.agent.vel.x[1], 4.0)
        np.testing.assert_allclose(episode.time_nominal, [0.0, 0.1])

    def test_replay_reveals_only_prefix_and_rewind_hides_future_again(self):
        states = self.states.copy()
        states[-1, 0] = 1_000_000.0
        episode = kinematics.load_episode_kinematics(self.make_dataset(states=states))
        plotter = kinematics.EpisodePlotter(episode_id=0, realtime_plotting=False)
        self.addCleanup(plotter.close)
        axis = self.signal_axis(plotter, "Agent position")

        for step in (0, 2, 3, 1):
            plotter.render_episode(episode, step)
            line = self.signal_line(axis, "x")
            np.testing.assert_allclose(line.get_xdata(), episode.time_nominal[:step + 1])
            np.testing.assert_allclose(line.get_ydata(), states[:step + 1, 0])
            self.assert_no_future_points(plotter, episode.time_nominal[step])
            # Future values must not affect the visible chart's vertical scale,
            # including after rewinding from a previously displayed future step.
            if step < 3:
                self.assertLess(axis.get_ylim()[1], 1000.0)
            else:
                self.assertGreater(axis.get_ylim()[1], 1_000_000.0)

    def test_live_raw_states_match_offline_kinematics(self):
        offline = kinematics.load_episode_kinematics(self.make_dataset())
        with patch.object(plt, "show"):
            plotter = kinematics.visualize_episode(realtime_plotting=True, episode_id=0)
        self.addCleanup(plotter.close)
        self.assertIsNone(plotter.episode)

        for state, action in zip(self.states, self.actions):
            plotter.update(state, action)
        plotter.refresh()

        live = plotter.episode
        self.assertEqual(live.num_steps, offline.num_steps)
        np.testing.assert_allclose(live.time_nominal, offline.time_nominal)
        np.testing.assert_allclose(live.state_augmented, offline.state_augmented)
        np.testing.assert_allclose(live.action_target.x, offline.action_target.x)
        np.testing.assert_allclose(live.action_target.y, offline.action_target.y)
        self.assert_no_future_points(plotter, live.time_nominal[-1])

    def test_live_preserves_supplied_augmented_velocity_values(self):
        # These measured velocities intentionally differ from finite differences.
        supplied = np.array([
            [10.0, 20.0, 12.0, -3.0, 100.0, 200.0, 4.0, -5.0, 0.1, 0.6],
            [10.2, 20.4, 13.0, -2.0, 100.1, 200.2, 6.0, -7.0, 0.2, 0.8],
        ])
        with patch.object(plt, "show"):
            plotter = kinematics.visualize_episode(realtime_plotting=True)
        self.addCleanup(plotter.close)
        for state, action in zip(supplied, self.actions):
            plotter.update(state, action)
        np.testing.assert_array_equal(plotter.episode.state_augmented, supplied)
        np.testing.assert_array_equal(
            plotter.episode.raw_state, supplied[:, [0, 1, 4, 5, 8]],
        )

    def test_live_visualizer_opens_nonblocking_and_closes(self):
        with patch.object(plt, "show") as show:
            plotter = kinematics.visualize_episode(realtime_plotting=True)
        self.addCleanup(plotter.close)
        self.assertFalse(plotter.closed)
        self.assertTrue(show.called)
        for call in show.call_args_list:
            self.assertIs(call.kwargs.get("block"), False)
        plotter.close()
        self.assertTrue(plotter.closed)

    def test_invalid_live_sample_does_not_partially_append(self):
        with patch.object(plt, "show"):
            plotter = kinematics.visualize_episode(realtime_plotting=True)
        self.addCleanup(plotter.close)
        plotter.update(self.states[0], self.actions[0])
        before = plotter.episode.state_augmented.copy()
        invalid_samples = [
            (np.zeros(4), self.actions[1], None),
            (np.full(5, np.nan), self.actions[1], None),
            (self.states[1], np.zeros(3), None),
            (self.states[1], [0.0, np.inf], None),
            (self.states[1], self.actions[1], np.zeros((8, 8))),
            (self.states[1], self.actions[1], np.zeros((0, 8, 3))),
        ]
        for state, action, image in invalid_samples:
            with self.subTest(state_shape=np.shape(state), image_shape=np.shape(image)):
                with self.assertRaises(ValueError):
                    plotter.update(state, action, image)
                np.testing.assert_array_equal(plotter.episode.state_augmented, before)
                self.assertEqual(plotter.episode.num_steps, 1)
        plotter.update(self.states[1], self.actions[1])
        np.testing.assert_allclose(plotter.episode.agent.vel.x, [0.0, 2.0])

    def test_refresh_throttle_retains_all_control_samples(self):
        with patch.object(plt, "show"):
            plotter = kinematics.visualize_episode(realtime_plotting=True)
        self.addCleanup(plotter.close)
        axis = self.signal_axis(plotter, "Agent position")
        # No wall-clock time elapses while four control samples arrive.
        with patch.object(kinematics, "monotonic", return_value=10.0):
            for state, action in zip(self.states, self.actions):
                plotter.update(state, action)
            self.assertEqual(plotter.episode.num_steps, 4)
            self.assertEqual(len(self.signal_line(axis, "x").get_xdata()), 1)
            plotter.refresh()
        np.testing.assert_allclose(self.signal_line(axis, "x").get_xdata(),
                                   [0.0, 0.1, 0.2, 0.3])
        np.testing.assert_allclose(self.signal_line(axis, "x").get_ydata(),
                                   self.states[:, 0])

    def test_paused_display_still_collects_and_resumes_at_latest_sample(self):
        with patch.object(plt, "show"):
            plotter = kinematics.visualize_episode(realtime_plotting=True)
        self.addCleanup(plotter.close)
        axis = self.signal_axis(plotter, "Agent position")
        with patch.object(kinematics, "monotonic", return_value=10.0) as clock:
            plotter.update(self.states[0], self.actions[0])
            event = KeyEvent("key_press_event", plotter.fig.canvas, key="space")
            plotter.fig.canvas.callbacks.process("key_press_event", event)
            for state, action in zip(self.states[1:], self.actions[1:]):
                clock.return_value += 1.0
                plotter.update(state, action)
            self.assertEqual(plotter.episode.num_steps, 4)
            np.testing.assert_allclose(self.signal_line(axis, "x").get_xdata(), [0.0])
            plotter.fig.canvas.callbacks.process("key_press_event", event)
        np.testing.assert_allclose(self.signal_line(axis, "x").get_xdata(),
                                   [0.0, 0.1, 0.2, 0.3])


if __name__ == "__main__":
    unittest.main()
