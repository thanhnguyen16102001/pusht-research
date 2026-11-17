"""
Push-T Episode Kinematics
=========================

Read a recorded Push-T episode from Zarr and estimate linear/angular velocities
using backward finite differences.

Original state:
    [agent_x, agent_y, block_x, block_y, block_angle]

Augmented state:
    [agent_x, agent_y, agent_vx, agent_vy,
     block_x, block_y, block_vx, block_vy,
     block_angle, block_omega]

Notes:
    - Estimated velocity components at the first sample are zero by convention.
    - Time is nominal (default control frequency: 10 Hz), not recorded timestamps.
    - Replay reveals only the history up to the current step.
    - Realtime visualization receives samples from the caller's running loop.
    - Zarr is always opened read-only.
"""

from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import numpy as np
import zarr


# ============================================================
# 1. Data structures
# ============================================================

@dataclass
class Vector2Series:
    x: np.ndarray
    y: np.ndarray


@dataclass
class AgentData:
    pos: Vector2Series
    vel: Vector2Series


@dataclass
class BlockData:
    pos: Vector2Series
    vel: Vector2Series
    angle: np.ndarray
    omega: np.ndarray


@dataclass
class EpisodeKinematics:
    episode_id: int
    num_steps: int
    dt_nominal: float
    time_nominal: np.ndarray

    agent: AgentData
    block: BlockData
    action_target: Vector2Series

    raw_state: np.ndarray
    state_augmented: np.ndarray


# ============================================================
# 2. Numerical differentiation
# ============================================================

def calculate_velocity(values, dt):
    """Backward difference: velocity[0] = 0; velocity[k] = (x[k] - x[k-1]) / dt."""
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("Expected a 1D array")

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")

    velocity = np.zeros_like(values)
    velocity[1:] = np.diff(values) / dt
    return velocity


# ============================================================
# 3. Load and process an episode
# ============================================================

def load_episode_kinematics(zarr_path, episode_id=0, control_hz=10.0):
    """Load one episode and return its positions, velocities, actions, and 10D state."""

    # 3.1. Validate inputs and open Zarr (read-only)
    if not np.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("control_hz must be positive and finite")

    dataset_path = Path(zarr_path).expanduser()

    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    root = zarr.open_group(str(dataset_path), mode="r")
    data = root["data"]
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)

    if len(episode_ends) == 0:
        raise ValueError("Dataset contains no episodes")

    if episode_ends[0] <= 0 or np.any(np.diff(episode_ends) <= 0):
        raise ValueError("Invalid episode boundaries")

    if not 0 <= episode_id < len(episode_ends):
        raise IndexError(f"episode_id must be in [0, {len(episode_ends) - 1}]")

    # 3.2. Slice one episode without crossing into another
    start = (0 if episode_id == 0 else int(episode_ends[episode_id - 1]))
    stop = int(episode_ends[episode_id])

    if stop > data["state"].shape[0] or stop > data["action"].shape[0]:
        raise ValueError("Episode boundary exceeds recorded data length")

    raw_state = np.asarray(data["state"][start:stop], dtype=np.float64)
    raw_action = np.asarray(data["action"][start:stop], dtype=np.float64)

    if raw_state.ndim != 2 or raw_state.shape[1] != 5:
        raise ValueError("Expected state shape (N, 5)")

    num_steps = raw_state.shape[0]

    if raw_action.shape != (num_steps, 2):
        raise ValueError("Expected action shape (N, 2)")

    return _make_episode(episode_id, raw_state, raw_action, 1.0 / control_hz)


def _make_episode(episode_id, raw_state, raw_action, dt, state_augmented=None):
    """Build the same structured state view for a recording or a live snapshot."""
    num_steps = len(raw_state)
    time_nominal = np.arange(num_steps, dtype=np.float64) * dt

    agent_x = raw_state[:, 0]
    agent_y = raw_state[:, 1]
    block_x = raw_state[:, 2]
    block_y = raw_state[:, 3]
    block_angle = raw_state[:, 4]

    if state_augmented is None:
        agent_vx = calculate_velocity(agent_x, dt)
        agent_vy = calculate_velocity(agent_y, dt)
        block_vx = calculate_velocity(block_x, dt)
        block_vy = calculate_velocity(block_y, dt)
        block_omega = calculate_velocity(np.unwrap(block_angle), dt)
        state_augmented = np.column_stack([
            agent_x, agent_y, agent_vx, agent_vy,
            block_x, block_y, block_vx, block_vy, block_angle, block_omega])
    else:
        agent_vx, agent_vy = state_augmented[:, 2], state_augmented[:, 3]
        block_vx, block_vy = state_augmented[:, 6], state_augmented[:, 7]
        block_omega = state_augmented[:, 9]

    # 3.5. Object-oriented view of one episode
    agent = AgentData(pos=Vector2Series(x=agent_x, y=agent_y),
                      vel=Vector2Series(x=agent_vx, y=agent_vy))

    block = BlockData(pos=Vector2Series(x=block_x, y=block_y),
                      vel=Vector2Series(x=block_vx, y=block_vy),
                      angle=block_angle,
                      omega=block_omega)

    action_target = Vector2Series(x=raw_action[:, 0], y=raw_action[:, 1])

    return EpisodeKinematics(episode_id=episode_id,
                             num_steps=num_steps,
                             dt_nominal=dt,
                             time_nominal=time_nominal,
                             agent=agent,
                             block=block,
                             action_target=action_target,
                             raw_state=raw_state,
                             state_augmented=state_augmented)


# ============================================================
# 4. Progressive replay and realtime visualization
# ============================================================


def _plot_font_family():
    """Use installed Times New Roman; cache WSL fonts locally for fast redraws."""
    import shutil
    from matplotlib import get_cachedir
    from matplotlib import font_manager

    if not any(font.name == "Times New Roman" for font in font_manager.fontManager.ttflist):
        for font_path in sorted(Path("/mnt/c/Windows/Fonts").glob("times*.ttf")):
            # Repeated font access over /mnt/c makes interactive drawing slow.
            # Keep the installed font unchanged and cache only in Matplotlib's
            # user cache, never in the source tree or demonstration datasets.
            local_font = font_path
            try:
                cache_dir = Path(get_cachedir()) / "pusht-fonts"
                cache_dir.mkdir(parents=True, exist_ok=True)
                cached_font = cache_dir / font_path.name
                if (not cached_font.exists()
                        or cached_font.stat().st_size != font_path.stat().st_size
                        or cached_font.stat().st_mtime_ns != font_path.stat().st_mtime_ns):
                    shutil.copy2(font_path, cached_font)
                local_font = cached_font
            except OSError:
                pass  # A read-only cache still permits using the installed font.
            try:
                font_manager.fontManager.addfont(str(local_font))
            except (OSError, RuntimeError):
                continue

    available = {font.name for font in font_manager.fontManager.ttflist}
    for family in ("Times New Roman", "Liberation Serif", "DejaVu Serif"):
        if family in available:
            return family
    return "serif"


class EpisodePlotter:
    """One episode's charts, updated on the GUI/main thread.

    In realtime mode, update() accepts either the recorded 5D state (velocities
    are estimated) or the existing augmented 10D state (velocities are retained).
    Each call represents one control step, not elapsed wall-clock training time.
    All samples are retained; GUI refreshes are limited to 30 per second.
    Closing this window does not stop the caller's simulation or training.
    """

    def __init__(self, episode_id=0, control_hz=10.0, realtime_plotting=False):
        import matplotlib.pyplot as plt

        if not np.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("control_hz must be positive and finite")

        self.episode_id = episode_id
        self.dt_nominal = 1.0 / control_hz
        self.realtime_plotting = realtime_plotting
        self._raw_states = []
        self._actions = []
        self._states = []
        self._latest_image = None
        self._closed = False
        self._paused = False
        self._last_refresh = float("-inf")
        self._refresh_period = 1.0 / 30.0
        self._layout_pending = True
        self._plt = plt
        self._font_family = _plot_font_family()
        self.quit_hint = "Q: close plot"
        self.plots = []

        # Preserve the existing layout, colors, line widths, and grid style.
        with plt.rc_context({"font.family": self._font_family}):
            self.fig = plt.figure(figsize=(16, 9), constrained_layout=True)
            grid = self.fig.add_gridspec(3, 3, width_ratios=[1.25, 1.0, 1.0])
            self.ax_image = self.fig.add_subplot(grid[0:2, 0])
            ax_action = self.fig.add_subplot(grid[2, 0])
            ax_agent_pos = self.fig.add_subplot(grid[0, 1])
            ax_agent_vel = self.fig.add_subplot(grid[0, 2])
            ax_block_pos = self.fig.add_subplot(grid[1, 1])
            ax_block_vel = self.fig.add_subplot(grid[1, 2])
            ax_angle = self.fig.add_subplot(grid[2, 1])
            ax_omega = self.fig.add_subplot(grid[2, 2])

            self.image_artist = self.ax_image.imshow(
                np.full((96, 96, 3), 255, dtype=np.uint8), interpolation="nearest")
            self._image_placeholder = self.ax_image.text(
                0.5, 0.5, "Waiting for an image", ha="center", va="center",
                transform=self.ax_image.transAxes, color="gray")
            self.ax_image.set_title("Push-T | Realtime" if realtime_plotting
                                    else "Recorded Push-T | Replay")
            self.ax_image.set_axis_off()
            self.heading = self.fig.suptitle("Waiting for the first sample", fontsize=12)

            self._add_plot(ax_agent_pos, "Agent position", "Position [sim units]",
                           [("x", 0), ("y", 1)])
            self._add_plot(ax_agent_vel, "Agent velocity", "Velocity [sim units/s]",
                           [("vx", 2), ("vy", 3)])
            self._add_plot(ax_block_pos, "Block position", "Position [sim units]",
                           [("x", 4), ("y", 5)])
            self._add_plot(ax_block_vel, "Block velocity", "Velocity [sim units/s]",
                           [("vx", 6), ("vy", 7)])
            self._add_plot(ax_angle, "Block angle", "Angle [rad]", [("theta", 8)])
            self._add_plot(ax_omega, "Block angular velocity", "Angular velocity [rad/s]",
                           [("omega", 9)])
            self._add_plot(ax_action,
                           "Action target" if realtime_plotting else "Recorded action target",
                           "Target [sim units]", [("target x", 10), ("target y", 11)])

        self.fig.canvas.mpl_connect("close_event", self._on_close)
        self.fig.canvas.mpl_connect("draw_event", self._on_draw)
        self.fig.canvas.mpl_connect("resize_event", self._on_resize)
        if realtime_plotting:
            self.fig.canvas.mpl_connect("key_press_event", self._on_live_key)
            plt.show(block=False)

    def _add_plot(self, ax, title, ylabel, series):
        lines, markers, columns = [], [], []
        for label, column in series:
            line, = ax.plot([], [], label=label, linewidth=1.5)
            marker, = ax.plot([], [], "o", color=line.get_color(), markersize=5)
            lines.append(line)
            markers.append(marker)
            columns.append(column)
        ax.set_title(title)
        ax.set_xlabel("Nominal time [s]")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        cursor = ax.axvline(0.0, color="red", linestyle="--", linewidth=1.2)
        self.plots.append(dict(ax=ax, columns=columns, lines=lines,
                               markers=markers, cursor=cursor))

    @property
    def closed(self):
        return self._closed or not self._plt.fignum_exists(self.fig.number)

    @property
    def episode(self):
        """Snapshot of received live samples, or None before the first update."""
        if not self._raw_states:
            return None
        return _make_episode(self.episode_id, np.asarray(self._raw_states),
                             np.asarray(self._actions), self.dt_nominal,
                             state_augmented=np.asarray(self._states))

    def _on_close(self, _event):
        self._closed = True

    def _on_draw(self, _event):
        if self._layout_pending:
            # Solve the unchanged eight-panel layout once, not at every sample.
            self.fig.set_layout_engine("none")
            self._layout_pending = False

    def _on_resize(self, _event):
        self.fig.set_layout_engine("constrained")
        self._layout_pending = True

    def _on_live_key(self, event):
        if event.key in ("q", "Q", "escape"):
            self.close()
        elif event.key in (" ", "space"):
            self._paused = not self._paused
            if self._paused:
                self.heading.set_text(self.heading.get_text().replace("Realtime", "Paused display"))
                self.fig.canvas.draw_idle()
            else:
                self.refresh()

    def reset(self, episode_id=None):
        """Clear live history for the next episode without recreating the window.

        Return False if the window has closed. The nominal control period and
        plot style are retained. No simulation state or saved data is changed.
        """
        if self.closed:
            return False
        if not self.realtime_plotting:
            raise ValueError("reset() requires realtime_plotting=True")
        if episode_id is not None:
            self.episode_id = episode_id
        self._raw_states.clear()
        self._actions.clear()
        self._states.clear()
        self._latest_image = None
        self._paused = False
        self._last_refresh = float("-inf")
        self.image_artist.set_data(np.full((96, 96, 3), 255, dtype=np.uint8))
        self._image_placeholder.set_visible(True)
        self.heading.set_text(f"Episode {self.episode_id} | Waiting for the first sample | {self.quit_hint}")
        for plot in self.plots:
            for line in plot["lines"] + plot["markers"]:
                line.set_data([], [])
            plot["cursor"].set_xdata([0.0, 0.0])
            plot["ax"].set_xlim(0.0, 1.0)
            plot["ax"].set_ylim(-0.055, 0.055)
            plot["ax"].set_autoscaley_on(True)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        return not self.closed

    def update(self, state, action, image=None):
        """Append one live sample; return False if the figure has been closed.

        Pass matching pre-action state, action target XY, and optional RGB image.
        5D input: [agent_x, agent_y, block_x, block_y, block_angle].
        10D input: the augmented state order documented at the top of this file.
        No simulator steps, file writes, or blocking pyplot.show() are performed.
        """
        if self.closed:
            return False
        if not self.realtime_plotting:
            raise ValueError("update() requires realtime_plotting=True")

        state = np.asarray(state, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)
        if state.shape not in ((5,), (10,)) or not np.all(np.isfinite(state)):
            raise ValueError("state must contain 5 or 10 finite values")
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("action must contain two finite target coordinates")
        if image is not None:
            image = self._validate_image(image)

        if state.shape == (10,):
            augmented = state.copy()
            raw_state = state[[0, 1, 4, 5, 8]].copy()
        else:
            raw_state = state.copy()
            velocity = np.zeros(5, dtype=np.float64)
            if self._raw_states:
                previous = self._raw_states[-1]
                velocity[:4] = (raw_state[:4] - previous[:4]) / self.dt_nominal
                angles = np.unwrap([previous[4], raw_state[4]])
                velocity[4] = (angles[1] - angles[0]) / self.dt_nominal
            augmented = np.array([
                *raw_state[:2], *velocity[:2], *raw_state[2:4], *velocity[2:4],
                raw_state[4], velocity[4]])

        self._raw_states.append(raw_state)
        self._actions.append(action.copy())
        self._states.append(augmented)
        if image is not None:
            self._latest_image = image.copy()

        if monotonic() - self._last_refresh >= self._refresh_period:
            if self._paused:
                # Keep receiving samples while the display is paused.
                self.fig.canvas.flush_events()
                self._last_refresh = monotonic()
            else:
                self.refresh()
        return True

    @staticmethod
    def _validate_image(image):
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) == 0:
            raise ValueError("Expected image shape (height, width, 3)")
        return image

    def render_episode(self, episode, step, image=None):
        """Render just a prefix, including rescaling Y from that prefix only."""
        if self.closed:
            return
        if not 0 <= step < episode.num_steps:
            raise IndexError("step is outside this episode")
        if image is not None:
            self.image_artist.set_data(self._validate_image(image))
            self._image_placeholder.set_visible(False)

        # Never hand future values to a Line2D artist or to axis autoscaling.
        times = episode.time_nominal[:step + 1]
        values = np.column_stack([
            episode.state_augmented[:step + 1],
            episode.action_target.x[:step + 1], episode.action_target.y[:step + 1]])
        now = float(times[-1])
        if self.realtime_plotting:
            xmax = max(1.0, float(np.ceil(now + episode.dt_nominal)))
            status = "Paused display" if self._paused else "Realtime"
            detail = f"{status} | Space: pause display | {self.quit_hint}"
            step_label = str(step + 1)
        else:
            xmax = max(float(episode.time_nominal[-1]), episode.dt_nominal)
            detail = "Space: pause | Arrows: step | R | Q"
            step_label = f"{step + 1}/{episode.num_steps}"

        self.heading.set_text(f"Episode {episode.episode_id} | Step {step_label} | "
                              f"Time {now:.2f}s (nominal) | {detail}")
        for plot in self.plots:
            for column, line, marker in zip(plot["columns"], plot["lines"], plot["markers"]):
                line.set_data(times, values[:, column])
                marker.set_data([now], [values[-1, column]])
            plot["cursor"].set_xdata([now, now])
            plot["ax"].set_xlim(0.0, xmax)
            plot["ax"].relim()
            plot["ax"].autoscale_view(scalex=False, scaley=True)

        # New tick labels must also inherit the requested font after axis changes.
        from matplotlib.text import Text
        for text_artist in self.fig.findobj(Text):
            text_artist.set_fontfamily(self._font_family)

    def refresh(self):
        """Force the display to catch up, e.g. after the last training step."""
        if self.closed or not self._states:
            return
        episode = self.episode
        self.render_episode(episode, episode.num_steps - 1, self._latest_image)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self._last_refresh = monotonic()

    def close(self):
        """Close only this figure; the training or simulation loop may continue."""
        self._closed = True
        self._plt.close(self.fig)


def visualize_episode(zarr_path=None, episode_id=-1, control_hz=10.0,
                      realtime_plotting=False):
    """Replay an episode, or create a nonblocking live plotter.

    Default (False): replay Zarr images and reveal curves through the current
    step only; -1 selects a random episode. Returns EpisodeKinematics on close.
    Keys: Space=play/pause, Left/Right=step, R=restart, Q/Esc=quit.

    True: omit zarr_path and receive an EpisodePlotter immediately. Call
    plotter.update(state, action, image=None) once per control step, on the main
    thread. Call refresh() to show the final sample, and close() when finished.
    No trainer or environment is created by this visualization function.
    """
    if not np.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("control_hz must be positive and finite")
    if realtime_plotting:
        if zarr_path is not None:
            raise ValueError("In realtime mode, omit zarr_path and use plotter.update(...)")
        return EpisodePlotter(episode_id=0 if episode_id == -1 else episode_id,
                              control_hz=control_hz, realtime_plotting=True)
    if zarr_path is None:
        raise ValueError("zarr_path is required for recorded-episode playback")

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    root = zarr.open_group(str(Path(zarr_path).expanduser()), mode="r")
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    if len(episode_ends) == 0:
        raise ValueError("Dataset contains no episodes")
    if episode_id == -1:
        episode_id = int(np.random.default_rng().integers(0, len(episode_ends)))

    episode = load_episode_kinematics(zarr_path, episode_id=episode_id, control_hz=control_hz)
    start = 0 if episode_id == 0 else int(episode_ends[episode_id - 1])
    # Read each frame lazily instead of loading the entire episode's RGB array.
    frames = root["data"]["img"]
    if (frames.ndim != 4 or frames.shape[0] < int(episode_ends[episode_id])
            or frames.shape[-1] != 3):
        raise ValueError("Expected image array shape (N, height, width, 3)")

    plotter = EpisodePlotter(episode_id=episode_id, control_hz=control_hz)
    step = 0
    playing = True

    def render():
        plotter.render_episode(episode, step, frames[start + step])

    def on_key(event):
        nonlocal step, playing
        if event.key in ("q", "escape"):
            plotter.close()
            return
        if event.key in (" ", "space"):
            playing = not playing
        elif event.key == "right":
            playing = False
            step = min(step + 1, episode.num_steps - 1)
        elif event.key == "left":
            playing = False
            step = max(step - 1, 0)
        elif event.key == "r":
            step = 0
            playing = True
        render()
        plotter.fig.canvas.draw_idle()

    def update(_frame):
        nonlocal step, playing
        if playing and step < episode.num_steps - 1:
            step += 1
            render()
        elif playing:
            playing = False

    plotter.fig.canvas.mpl_connect("key_press_event", on_key)
    render()
    # init_func prevents the first canvas draw from skipping sample zero.
    animation = FuncAnimation(plotter.fig, update, init_func=render,
                              interval=1000.0 / control_hz, blit=False,
                              cache_frame_data=False)
    # Keep the animation alive for the complete lifetime of its figure.
    plotter.fig._episode_animation = animation

    print(f"[INFO] Episode: {episode_id} | Steps: {episode.num_steps}")
    print(f"[INFO] Nominal sampling period: {episode.dt_nominal:.3f} s")
    print("[INFO] Progressive replay is read-only; no physics is rerun.")
    plt.show()
    return episode


# ============================================================
# 5. Standalone execution
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect or visualize Push-T episode kinematics")
    parser.add_argument("--dataset", type=str, default="data/demonstrations/demo_actions_v1.zarr")
    parser.add_argument("--episode", type=int, default=0, help="Episode ID, or -1 for random view")
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--visualize", action="store_true", help="Replay images and state plots")
    args = parser.parse_args()

    if args.visualize:
        visualize_episode(zarr_path=args.dataset,
                          episode_id=args.episode,
                          control_hz=args.control_hz)
    else:
        episode = load_episode_kinematics(zarr_path=args.dataset,
                                          episode_id=args.episode,
                                          control_hz=args.control_hz)

        print("\n========== EPISODE KINEMATICS ==========")
        print("Episode ID :", episode.episode_id)
        print("Steps      :", episode.num_steps)
        print("Nominal dt :", episode.dt_nominal)
        print("State shape:", episode.state_augmented.shape)

        print("\nAugmented state columns:")
        print("[agent_x, agent_y, agent_vx, agent_vy,")
        print(" block_x, block_y, block_vx, block_vy,")
        print(" block_angle, block_omega]")

        np.set_printoptions(precision=3, suppress=True)
        print("\nFirst 5 steps:")
        print(episode.state_augmented[:5])

        print("\nInitial velocities:")
        print("Agent vx    :", episode.agent.vel.x[0])
        print("Agent vy    :", episode.agent.vel.y[0])
        print("Block vx    :", episode.block.vel.x[0])
        print("Block vy    :", episode.block.vel.y[0])
        print("Block omega:", episode.block.omega[0])

        velocity_indices = [2, 3, 6, 7, 9]
        assert np.allclose(episode.state_augmented[0, velocity_indices], 0.0)
        print("\n[OK] Initial velocities are zero.")
