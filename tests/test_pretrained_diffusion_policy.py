"""Native checkpoint adapter: keypoints, temporal alignment and isolated RNG."""

import copy
from dataclasses import replace
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from pretrained_diffusion_policy import (PretrainedDiffusionPolicy, PretrainedPolicyConfig,
                                         _supported_config, original_block_keypoints,
                                         state_to_keypoints)


class FakeNativePolicy(torch.nn.Module):
    def __init__(self, horizon=6, offset=1, action_steps=3, stochastic=False):
        super().__init__()
        self.register_buffer("anchor", torch.zeros(1))
        self.horizon, self.offset, self.action_steps = horizon, offset, action_steps
        self.num_inference_steps = 8
        self.noise_scheduler = SimpleNamespace(config={"num_train_timesteps": 8})
        self.kwargs = {}
        self.stochastic = stochastic
        self.seen = []
        self.seen_schedulers = []
        self.seen_kwargs = []
        self.resets = 0
        self.nonfinite = False
        self.bad_alignment = False
        self.outside = False
        self.fail = False

    def reset(self):
        self.resets += 1

    def predict_action(self, observations):
        self.seen.append((observations["obs"].clone(), observations["obs_mask"].clone(),
                          self.num_inference_steps))
        self.seen_schedulers.append(self.noise_scheduler)
        self.seen_kwargs.append(dict(self.kwargs))
        if self.fail:
            torch.rand(3)
            raise RuntimeError("Deliberate failure")
        values = torch.arange(self.horizon, device=self.anchor.device, dtype=torch.float32)
        prediction = torch.stack([100 + values, 200 + values], dim=-1)[None]
        if self.stochastic:
            prediction += torch.rand(1, device=self.anchor.device)
        if self.nonfinite:
            prediction[0, 1, 0] = float("nan")
        if self.outside:
            prediction[0, 1] = torch.tensor([-1., 513.], device=self.anchor.device)
        selected = prediction[:, self.offset:self.offset + self.action_steps].clone()
        if self.bad_alignment:
            selected += 10
        return {"action_pred": prediction, "action": selected}


class PretrainedAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.points = np.arange(18, dtype=float).reshape(9, 2)
        self.config = PretrainedPolicyConfig(obs_horizon=2, original_horizon=6,
                                             action_offset=1, prediction_horizon=5,
                                             action_horizon=2, max_action_horizon=3,
                                             inference_steps=8, diffusion_steps=8)

    def policy(self, stochastic=False):
        native = FakeNativePolicy(stochastic=stochastic)
        return PretrainedDiffusionPolicy(native, self.config, self.points)

    def test_keypoints_match_the_actual_original_environment(self):
        points = original_block_keypoints()
        from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
        parameters = PushTKeypointsEnv.genenerate_keypoint_manager_params()
        env = PushTKeypointsEnv(**parameters)
        self.addCleanup(env.close)
        for seed in range(8):
            env.seed(seed)
            observation = env.reset()
            state = np.zeros(10, dtype=np.float64)
            state[:2] = env.agent.position
            state[4:6] = env.block.position
            state[8] = env.block.angle
            np.testing.assert_allclose(state_to_keypoints(state, points),
                                       observation[:20].astype(np.float32), rtol=1e-6, atol=5e-5)
            np.testing.assert_array_equal(observation[20:], np.ones(20))

    def test_keypoint_features_ignore_velocities_and_are_periodic(self):
        states = np.zeros((3, 10))
        states[:, :2] = [50, 60]
        states[:, 4:6] = [200, 250]
        states[:, 8] = [.2, .2 + 2 * np.pi, .2]
        states[2, [2, 3, 6, 7, 9]] = [10, 20, 30, 40, 50]
        encoded = state_to_keypoints(states, self.points)
        np.testing.assert_allclose(encoded[0], encoded[1], atol=1e-5)
        np.testing.assert_array_equal(encoded[0], encoded[2])
        np.testing.assert_array_equal(encoded[:, -2:], states[:, :2])

    def test_historical_action_prefix_is_excluded(self):
        policy = self.policy()
        prediction = policy.predict_action(np.zeros(10))
        self.assertEqual(prediction.shape, (5, 2))
        np.testing.assert_array_equal(prediction[0], [101, 201])
        np.testing.assert_array_equal(policy.act(np.zeros(10)), [101, 201])
        np.testing.assert_array_equal(policy.act(np.ones(10)), [102, 202])
        self.assertFalse(policy.last_replanned)

    def test_history_updates_on_queued_actions_and_reset_flushes(self):
        policy = self.policy()
        for step in range(3):
            policy.act(np.full(10, step, dtype=np.float32))
        observations, mask, _ = policy.native_policy.seen[-1]
        expected = state_to_keypoints(np.stack([np.ones(10), np.full(10, 2)]), self.points)
        np.testing.assert_array_equal(observations[0].numpy(), expected)
        self.assertTrue(bool(mask.all()))
        self.assertEqual(len(policy.native_policy.seen), 2)
        self.assertTrue(policy.last_replanned)
        policy.reset(19)
        self.assertEqual(len(policy.history), 0)
        self.assertEqual(len(policy.pending_actions), 0)
        self.assertEqual(policy.last_plan.shape, (0, 2))
        policy.act(np.full(10, 4))
        observations = policy.native_policy.seen[-1][0][0].numpy()
        np.testing.assert_array_equal(observations[0], observations[1])

    def test_native_global_rng_is_isolated_and_episode_seed_is_reproducible(self):
        policy = self.policy(stochastic=True)
        torch.manual_seed(123)
        caller_state = torch.get_rng_state().clone()
        policy.reset(7)
        first = policy.predict_action(np.zeros(10))
        second = policy.predict_action(np.zeros(10))
        self.assertFalse(np.array_equal(first, second))
        torch.testing.assert_close(torch.get_rng_state(), caller_state)
        policy.reset(7)
        np.testing.assert_array_equal(policy.predict_action(np.zeros(10)), first)
        np.testing.assert_array_equal(policy.predict_action(np.zeros(10)), second)
        fixed1 = policy.predict_action(np.zeros(10), seed=99)
        fixed2 = policy.predict_action(np.zeros(10), seed=99)
        np.testing.assert_array_equal(fixed1, fixed2)
        torch.testing.assert_close(torch.get_rng_state(), caller_state)

    @patch("pretrained_diffusion_policy.make_ddim_scheduler")
    def test_inference_overrides_are_applied_and_restored(self, factory):
        policy = self.policy()
        original_scheduler = policy.native_policy.noise_scheduler
        policy.config = replace(policy.config, sampler="ddim", inference_steps=4, action_horizon=1)
        policy.act(np.zeros(10))
        self.assertEqual(policy.native_policy.seen[-1][2], 4)
        self.assertIs(policy.native_policy.seen_schedulers[-1], factory.return_value)
        self.assertEqual(policy.native_policy.seen_kwargs[-1], {"eta": 0.0})
        self.assertIs(policy.native_policy.noise_scheduler, original_scheduler)
        self.assertEqual(policy.native_policy.num_inference_steps, 8)
        policy.predict_action(np.zeros(10), inference_steps=2)
        self.assertEqual(policy.native_policy.seen[-1][2], 2)
        self.assertEqual(policy.native_policy.num_inference_steps, 8)
        policy.act(np.ones(10))
        self.assertTrue(policy.last_replanned)
        factory.assert_called_once_with(original_scheduler)
        with self.assertRaises(ValueError):
            replace(policy.config, action_horizon=4)
        with self.assertRaises(ValueError):
            replace(policy.config, inference_steps=9)

    @patch("pretrained_diffusion_policy.make_ddim_scheduler")
    def test_rng_and_scheduler_are_restored_when_native_inference_fails(self, factory):
        policy = self.policy()
        policy.config = replace(policy.config, sampler="ddim", inference_steps=4)
        original_scheduler = policy.native_policy.noise_scheduler
        original_kwargs = {"eta": 0.75}
        policy.native_policy.kwargs = original_kwargs
        policy.native_policy.fail = True
        before = torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, "Deliberate"):
            policy.predict_action(np.zeros(10), inference_steps=2)
        torch.testing.assert_close(before, torch.get_rng_state())
        self.assertEqual(policy.native_policy.num_inference_steps, 8)
        self.assertIs(policy.native_policy.seen_schedulers[-1], factory.return_value)
        self.assertEqual(policy.native_policy.seen_kwargs[-1], {"eta": 0.0})
        self.assertIs(policy.native_policy.noise_scheduler, original_scheduler)
        self.assertIs(policy.native_policy.kwargs, original_kwargs)
        self.assertEqual(original_kwargs, {"eta": 0.75})

    def test_reduced_ddpm_is_rejected_before_native_inference(self):
        policy = self.policy()
        with self.assertRaisesRegex(ValueError, "--sampler ddim --inference-steps 20"):
            replace(policy.config, inference_steps=4)
        with self.assertRaisesRegex(ValueError, "--sampler ddim --inference-steps 20"):
            policy.predict_action(np.zeros(10), inference_steps=4)
        self.assertEqual(policy.native_policy.seen, [])
        with self.assertRaisesRegex(ValueError, "sampler must be"):
            replace(policy.config, sampler="unknown")

    @patch("pretrained_diffusion_policy.make_ddim_scheduler")
    def test_switching_back_to_ddpm_uses_original_full_chain(self, factory):
        policy = self.policy(stochastic=True)
        original_scheduler = policy.native_policy.noise_scheduler
        expected = policy.predict_action(np.zeros(10), seed=17)
        factory.assert_not_called()
        policy.config = replace(policy.config, sampler="ddim", inference_steps=4)
        policy.predict_action(np.zeros(10), seed=17)
        policy.config = replace(policy.config, sampler="ddpm", inference_steps=8)
        actual = policy.predict_action(np.zeros(10), seed=17)
        np.testing.assert_array_equal(actual, expected)
        self.assertIs(policy.native_policy.seen_schedulers[-1], original_scheduler)
        self.assertEqual(policy.native_policy.seen[-1][2], 8)
        self.assertEqual(policy.native_policy.seen_kwargs[-1], {})
        self.assertEqual(policy.metadata["original_sampler"], "ddpm")

    def test_action_bounds_nonfinite_outputs_and_alignment_are_checked(self):
        policy = self.policy()
        policy.native_policy.outside = True
        prediction = policy.predict_action(np.zeros(10))
        np.testing.assert_array_equal(prediction[0], [0, 512])
        self.assertEqual(policy.last_clipped_coordinates, 2)
        policy.native_policy.outside = False
        policy.native_policy.nonfinite = True
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            policy.predict_action(np.zeros(10))
        policy.native_policy.nonfinite = False
        policy.native_policy.bad_alignment = True
        with self.assertRaisesRegex(RuntimeError, "alignment"):
            policy.predict_action(np.zeros(10))
        for invalid in (np.zeros(5), np.full(10, np.nan), np.empty((0, 10))):
            with self.assertRaises(ValueError):
                policy.predict_action(invalid)

    def test_checkpoint_contract_rejects_incompatible_observations(self):
        cfg = {
            "policy": {"_target_": "diffusion_policy.policy.diffusion_unet_lowdim_policy.DiffusionUnetLowdimPolicy",
                       "obs_dim": 20, "action_dim": 2, "horizon": 16, "n_obs_steps": 2,
                       "n_action_steps": 8, "num_inference_steps": 100, "oa_step_convention": True,
                       "noise_scheduler": {"_target_": "diffusers.schedulers.scheduling_ddpm.DDPMScheduler",
                                           "num_train_timesteps": 100}},
            "task": {"env_runner": {"_target_": "diffusion_policy.env_runner.pusht_keypoints_runner.PushTKeypointsRunner"}}}
        config = _supported_config(cfg)
        self.assertEqual(config.prediction_horizon, 15)
        cases = (("policy", "obs_dim", 10), ("policy", "oa_step_convention", False),
                 ("runner", "keypoint_visible_rate", .5), ("runner", "agent_keypoints", True),
                 ("runner", "past_action", True), ("runner", "n_latency_steps", 1),
                 ("runner", "n_obs_steps", 3), ("runner", "n_action_steps", 4))
        for section, key, value in cases:
            with self.subTest(key=key):
                changed = copy.deepcopy(cfg)
                target = changed["policy"] if section == "policy" else changed["task"]["env_runner"]
                target[key] = value
                with self.assertRaises(ValueError):
                    _supported_config(changed)


if __name__ == "__main__":
    unittest.main()
