"""Integration of the original pretrained adapter with visible project runs."""

import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import episode_kinematics
import run_diffusion_policy as shared
import run_mpc
import run_pretrained_diffusion as pretrained
from pretrained_diffusion_policy import PretrainedPolicyConfig
from pusht_runtime import make_env
from test_coverage_runtime import HoldController
from test_diffusion_runtime import FakePolicy, ScriptedEnvironment, scripted_coverage


class FakePretrainedPolicy(FakePolicy):
    name = "pretrained_diffusion_policy"
    display_name = "GitHub pretrained Diffusion Policy"

    def __init__(self):
        super().__init__()
        self.config = PretrainedPolicyConfig()
        self.metadata = {
            "weights": "ema_model",
            "original_runner": {"legacy_test": True, "test_start_seed": 4300000},
        }


class PretrainedRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "checkpoints/pretrained/PushT_original_lowdim.ckpt"
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b"stand-in checkpoint for runner integration")

    def test_pretrained_defaults_reach_visible_loop_with_correct_identity(self):
        env, policy = ScriptedEnvironment((.94, .9499, .95)), FakePretrainedPolicy()
        env.legacy = False
        env.render = Mock(wraps=env.render)
        plotter = Mock(dt_nominal=0.1, closed=False)
        with patch.object(pretrained, "PROJECT_ROOT", self.root), patch.object(
                pretrained, "load_policy", return_value=policy) as load, patch.object(
                shared, "load_policy", side_effect=AssertionError("Own-model loader must not be called")), patch.object(
                shared, "make_env", return_value=env), patch.object(
                shared, "coverage", side_effect=scripted_coverage), patch.object(
                shared, "user_requested_stop", return_value=False), patch.object(
                shared, "sleep") as sleep, patch.object(
                episode_kinematics, "visualize_episode", return_value=plotter) as visualize, patch(
                "pygame.display.set_caption") as caption:
            summary = pretrained.main(["--seeds", "42", "--max-steps", "3", "--render",
                                       "--realtime-plotting", "--pace", "--no-plots"])
        load.assert_called_once_with(self.checkpoint, device="cpu")
        visualize.assert_called_once_with(realtime_plotting=True, control_hz=10.0)
        self.assertTrue(all(call.kwargs["mode"] == "human" for call in env.render.call_args_list))
        self.assertIn("GitHub pretrained", caption.call_args.args[0])
        self.assertEqual(sleep.call_count, 2)
        plotter.reset.assert_called_once_with(episode_id=42)
        self.assertGreaterEqual(plotter.update.call_count, 2)
        plotter.close.assert_called_once()
        self.assertEqual(summary["controller"], policy.name)
        self.assertEqual(summary["policy_metadata"], policy.metadata)
        self.assertFalse(summary["environment_legacy"])
        self.assertTrue(summary["policy_metadata"]["original_runner"]["legacy_test"])
        self.assertEqual(summary["target_coverage"], .95)
        self.assertEqual(summary["policy_config"]["sampler"], "ddpm")
        self.assertEqual(summary["policy_config"]["inference_steps"], 100)
        episode = summary["episodes"][0]
        self.assertEqual(episode["controller"], policy.name)
        self.assertEqual(episode["steps"], 2)
        self.assertTrue(episode["success"])
        output = self.root / "results/pretrained_diffusion"
        self.assertEqual(json.loads((output / "summary.json").read_text()), summary)
        self.assertTrue((output / "episode_42.npz").exists())

    def test_ddim_sampler_and_reduced_steps_reach_policy_and_saved_metadata_together(self):
        env, policy = ScriptedEnvironment((.94, .9499, .95)), FakePretrainedPolicy()
        output = self.root / "ddim_fast"
        with patch.object(pretrained, "load_policy", return_value=policy) as load, patch.object(
                shared, "make_env", return_value=env), patch.object(
                shared, "coverage", side_effect=scripted_coverage):
            summary = pretrained.main([
                "--checkpoint", str(self.checkpoint), "--sampler", "ddim", "--inference-steps", "20",
                "--seeds", "42", "--max-steps", "3", "--no-plots", "--output", str(output)])
        load.assert_called_once_with(self.checkpoint, device="cpu")
        self.assertEqual(policy.config.sampler, "ddim")
        self.assertEqual(policy.config.inference_steps, 20)
        self.assertEqual(policy.config.diffusion_steps, 100)
        self.assertEqual(summary["policy_config"]["sampler"], "ddim")
        self.assertEqual(summary["policy_config"]["inference_steps"], 20)
        self.assertEqual(summary["target_coverage"], .95)
        self.assertTrue(summary["episodes"][0]["success"])
        self.assertEqual(json.loads((output / "summary.json").read_text()), summary)
        with np.load(output / "episode_42.npz") as data:
            self.assertEqual(data["sampled_action_chunks"].shape, (1, 15, 2))

    def test_reduced_ddpm_exits_with_guidance_before_creating_environment_or_output(self):
        for sampler_args in ([], ["--sampler", "ddpm"]):
            with self.subTest(sampler_args=sampler_args):
                output = self.root / "invalid_ddpm"
                policy = FakePretrainedPolicy()
                stderr = io.StringIO()
                with patch.object(pretrained, "load_policy", return_value=policy), patch.object(
                        shared, "make_env") as make_environment, patch("sys.stderr", stderr), patch(
                        "sys.stdout", io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        pretrained.main([
                            "--checkpoint", str(self.checkpoint), "--inference-steps", "20",
                            "--output", str(output), *sampler_args])
                self.assertEqual(error.exception.code, 2)
                self.assertIn("--sampler ddim --inference-steps 20", stderr.getvalue())
                self.assertEqual(policy.config.sampler, "ddpm")
                self.assertEqual(policy.config.inference_steps, 100)
                make_environment.assert_not_called()
                self.assertFalse(output.exists())

    def test_shared_own_policy_cli_does_not_expose_native_sampler_switch(self):
        loader = Mock()
        output = self.root / "own_invalid_sampler"
        stderr = io.StringIO()
        with patch.object(shared, "make_env") as make_environment, patch("sys.stderr", stderr):
            with self.assertRaises(SystemExit) as error:
                shared.main([
                    "--checkpoint", str(self.checkpoint), "--sampler", "ddim",
                    "--inference-steps", "20", "--output", str(output)], policy_loader=loader)
        self.assertEqual(error.exception.code, 2)
        self.assertIn("unrecognized arguments: --sampler ddim", stderr.getvalue())
        loader.assert_not_called()
        make_environment.assert_not_called()
        self.assertFalse(output.exists())

    def test_continuous_timeouts_and_quit_save_separate_committed_histories(self):
        env = ScriptedEnvironment((.1, .2, .3))
        policy = FakePretrainedPolicy()
        output = self.root / "continuous"

        def stop(*args):
            if env.current_seed == 33 and env.index == 1:
                self.assertTrue((output / "episode_31.json").exists())
                self.assertTrue((output / "episode_32.npz").exists())
                return True
            return False

        with patch.object(pretrained, "load_policy", return_value=policy), patch.object(
                shared, "make_env", return_value=env), patch.object(
                shared, "coverage", side_effect=scripted_coverage), patch.object(
                shared, "user_requested_stop", side_effect=stop):
            summary = pretrained.main(["--checkpoint", str(self.checkpoint), "--continuous",
                                       "--start-seed", "31", "--max-steps", "2",
                                       "--output", str(output), "--no-plots"])
        self.assertEqual(policy.reset_seeds, [31, 32, 33])
        self.assertEqual(summary["num_completed_episodes"], 2)
        self.assertEqual(summary["num_interrupted_episodes"], 1)
        self.assertEqual(summary["run_status"], "stopped_by_user")
        self.assertEqual(summary["success_rate"], 0.0)
        self.assertEqual([ep["termination_reason"] for ep in summary["episodes"]],
                         ["time_limit", "time_limit", "user_stop"])
        for seed, steps in ((31, 2), (32, 2), (33, 1)):
            episode = json.loads((output / f"episode_{seed}.json").read_text())
            self.assertEqual(episode["controller"], policy.name)
            with np.load(output / f"episode_{seed}.npz") as data:
                self.assertEqual(data["states"].shape, (steps + 1, 10))
                self.assertEqual(data["actions"].shape, (steps, 2))
                np.testing.assert_array_equal(data["chunk_start_steps"], [0])
                self.assertNotIn("predicted_next_states", data)
        env.close.assert_called_once()

    def test_pretrained_wrapper_runs_original_environment_without_own_model(self):
        policy = FakePretrainedPolicy()
        created = []

        def original_env(*args, **kwargs):
            env = make_env(*args, **kwargs)
            created.append(env)
            return env

        with patch.object(pretrained, "load_policy", return_value=policy), patch.object(
                shared, "load_policy", side_effect=AssertionError("Wrong checkpoint loader")), patch.object(
                shared, "make_env", side_effect=original_env):
            summary = pretrained.main(["--checkpoint", str(self.checkpoint), "--seeds", "1000",
                                       "--max-steps", "1", "--no-plots", "--output", str(self.root / "real")])
        self.assertEqual(len(created), 1)
        self.assertFalse(created[0].legacy)
        self.assertFalse(summary["environment_legacy"])
        self.assertEqual(summary["episodes"][0]["steps"], 1)
        self.assertEqual(summary["original_success_threshold"], .95)
        self.assertEqual(summary["controller"], "pretrained_diffusion_policy")

    def test_shared_runner_keeps_own_policy_identity_when_loader_is_injected(self):
        env, policy = ScriptedEnvironment(), FakePolicy()
        loader = Mock(return_value=policy)
        with patch.object(shared, "make_env", return_value=env), patch.object(
                shared, "coverage", side_effect=scripted_coverage), patch.object(
                shared, "load_policy", side_effect=AssertionError("Injected loader was ignored")):
            summary = shared.main(["--checkpoint", str(self.checkpoint), "--seeds", "1", "--no-plots",
                                   "--output", str(self.root / "own")], policy_loader=loader)
        self.assertEqual(summary["controller"], "diffusion_policy")
        self.assertEqual(summary["episodes"][0]["controller"], "diffusion_policy")
        self.assertEqual(summary["policy_metadata"], {})
        self.assertEqual(summary["target_coverage"], .99)

    def test_pretrained_and_mpc_defaults_stop_at_exact_95_and_policy_override_can_require_99(self):
        coverages = (.94, .9499, .95, .99)
        project = Path(__file__).resolve().parents[1]
        for config_name in ("mpc_default.json", "mpc_overlap.json"):
            configured = json.loads((project / "configs" / config_name).read_text())["cost"]["goal_coverage"]
            self.assertEqual(configured, .95)
            env = ScriptedEnvironment(coverages)
            with patch.object(run_mpc, "coverage", side_effect=scripted_coverage):
                metrics, _, _ = run_mpc.run_episode(env, HoldController(), None, {"dt": .1},
                                                   42, max_steps=4, target_coverage=configured)
            self.assertEqual(metrics["steps"], 2)
            self.assertTrue(metrics["success"])
            self.assertFalse(metrics["original_environment_success"])

        for target, expected_steps in ((None, 2), (.99, 3)):
            env, policy = ScriptedEnvironment(coverages), FakePretrainedPolicy()
            arguments = ["--checkpoint", str(self.checkpoint), "--seeds", "42", "--max-steps", "4",
                         "--no-plots", "--output", str(self.root / f"threshold_{target}")]
            if target is not None:
                arguments.extend(["--target-coverage", str(target)])
            with patch.object(pretrained, "load_policy", return_value=policy), patch.object(
                    shared, "make_env", return_value=env), patch.object(
                    shared, "coverage", side_effect=scripted_coverage):
                summary = pretrained.main(arguments)
            self.assertEqual(summary["target_coverage"], .95 if target is None else target)
            self.assertEqual(summary["episodes"][0]["steps"], expected_steps)
            self.assertTrue(summary["episodes"][0]["success"])

    def test_checkpoint_hash_streams_and_matches_content_digest(self):
        expected = hashlib.sha256(b"stand-in checkpoint for runner integration").hexdigest()
        # Real pretrained files are ~1 GiB; checksum calculation must not
        # allocate another whole-file byte array while model weights are held.
        with patch.object(Path, "read_bytes", side_effect=AssertionError("Do not read full checkpoint")):
            self.assertEqual(shared.checkpoint_sha256(self.checkpoint), expected)


if __name__ == "__main__":
    unittest.main()
