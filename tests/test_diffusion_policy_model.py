"""Regression tests for diffusion math, imitation training and causal control."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from Diffusion_Policy_model import (ConditionalTemporalDenoiser, DiffusionPolicy,
                                    DiffusionPolicyConfig, DiffusionSchedule,
                                    load_policy, masked_noise_loss, save_policy,
                                    train_policy)
from diffusion_policy_data import prepare_policy_data


class DiffusionModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.config = DiffusionPolicyConfig(obs_horizon=2, prediction_horizon=4,
                                            action_horizon=2, diffusion_steps=8,
                                            inference_steps=4, channels=16,
                                            residual_blocks=1, time_embed_dim=8)
        self.normalizer = {"obs_mean": [0.] * 11, "obs_std": [1.] * 11,
                           "action_low": [0., 0.], "action_high": [512., 512.],
                           "dt": .1}

    def policy(self):
        return DiffusionPolicy(ConditionalTemporalDenoiser(self.config), self.normalizer)

    def demonstration(self):
        path = self.directory / "demo.zarr"
        root = zarr.open_group(str(path), mode="w")
        states, actions = [], []
        for episode in range(4):
            for step in range(6):
                states.append([30 + episode * 20 + step, 80 + step,
                               150 + episode * 30 + .2 * step, 240 - step, .3 + .01 * step])
                actions.append([50 + episode * 20 + step, 90 + step])
        data = root.create_group("data")
        data.array("state", np.asarray(states, dtype=np.float32))
        data.array("action", np.asarray(actions, dtype=np.float32))
        root.create_group("meta").array("episode_ends", np.arange(6, 25, 6))
        return path

    def test_forward_noise_and_ddim_oracle_recover_clean_actions(self):
        schedule = DiffusionSchedule(17).double()
        clean = torch.linspace(-.75, .75, 24, dtype=torch.float64).reshape(3, 4, 2)
        noise = torch.randn(clean.shape, dtype=torch.float64)
        for count in (1, 2, 6, 17):
            with self.subTest(inference_steps=count):
                timesteps = schedule.inference_timesteps(count)
                self.assertEqual(timesteps[0], 16)
                self.assertEqual(len(set(timesteps)), count)
                sample = schedule.add_noise(clean, noise, torch.full((3,), timesteps[0]))
                for index, timestep in enumerate(timesteps):
                    alpha = schedule.alpha_bar[timestep]
                    true_noise = (sample - alpha.sqrt() * clean) / (1-alpha).sqrt()
                    previous = timesteps[index+1] if index+1 < len(timesteps) else -1
                    sample = schedule.ddim_step(sample, true_noise, timestep, previous)
                torch.testing.assert_close(sample, clean, rtol=0, atol=2e-11)

    def test_ddim_clipping_recomputes_consistent_direction(self):
        schedule = DiffusionSchedule(12).double()
        clean = torch.tensor([[[1.5, -1.4], [.2, -.3]]], dtype=torch.float64)
        noise = torch.tensor([[[.3, .4], [-.1, .2]]], dtype=torch.float64)
        sample = schedule.add_noise(clean, noise, torch.tensor([9]))
        alpha, previous = schedule.alpha_bar[9], schedule.alpha_bar[3]
        clipped = clean.clamp(-1, 1)
        corrected_noise = (sample-alpha.sqrt()*clipped)/(1-alpha).sqrt()
        expected = previous.sqrt()*clipped + (1-previous).sqrt()*corrected_noise
        actual = schedule.ddim_step(sample, noise, 9, 3)
        torch.testing.assert_close(actual, expected)
        uncorrected = previous.sqrt()*clipped + (1-previous).sqrt()*noise
        self.assertGreater(float((actual-uncorrected).abs().max()), .01)
        torch.testing.assert_close(schedule.ddim_step(sample, noise, 9, -1), clipped)

    def test_masked_loss_excludes_padding_and_its_gradients(self):
        predicted = torch.tensor([[[1., 2.], [3., 4.], [999., -888.]]], requires_grad=True)
        noise = torch.zeros_like(predicted)
        mask = torch.tensor([[True, True, False]])
        loss = masked_noise_loss(predicted, noise, mask)
        self.assertAlmostEqual(float(loss), 7.5)
        loss.backward()
        torch.testing.assert_close(predicted.grad[0, :2], predicted.detach()[0, :2]/2)
        torch.testing.assert_close(predicted.grad[0, 2], torch.zeros(2))
        changed_noise = noise.clone()
        changed_noise[0, 2] = 1e6
        torch.testing.assert_close(masked_noise_loss(predicted, changed_noise, mask), loss)
        with self.assertRaises(ValueError):
            masked_noise_loss(predicted, noise, torch.zeros_like(mask))

    def test_denoiser_can_fit_conditional_noise(self):
        model = ConditionalTemporalDenoiser(self.config)
        observations = torch.randn(4, 2, 11)
        # Identical noisy inputs but different conditioning require the model
        # to actually use observations instead of learning only a noise prior.
        sample = torch.randn(1, 4, 2).repeat(4, 1, 1)
        timestep = torch.full((4,), 4)
        target = observations[:, :1, :2].repeat(1, 4, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        before = float((model(sample, timestep, observations)-target).square().mean())
        for _ in range(60):
            optimizer.zero_grad()
            loss = (model(sample, timestep, observations)-target).square().mean()
            loss.backward()
            optimizer.step()
        after = float((model(sample, timestep, observations)-target).square().mean())
        self.assertLess(after, before * .1)

    def test_action_queue_records_every_observation_and_reset_flushes(self):
        policy = self.policy()
        sampled_histories = []
        plan = np.array([[10, 20], [11, 21], [12, 22], [13, 23]], dtype=np.float32)

        def sample(history):
            sampled_histories.append(history.copy())
            return plan.copy()

        states = [np.full(10, step, dtype=np.float32) for step in range(4)]
        with patch.object(policy, "predict_action", side_effect=sample):
            np.testing.assert_array_equal(policy.act(states[0]), plan[0])
            self.assertTrue(policy.last_replanned)
            np.testing.assert_array_equal(policy.act(states[1]), plan[1])
            self.assertFalse(policy.last_replanned)
            self.assertEqual(policy.last_sampling_seconds, 0.)
            np.testing.assert_array_equal(policy.act(states[2]), plan[0])
            self.assertTrue(policy.last_replanned)
            np.testing.assert_array_equal(sampled_histories[1], np.stack(states[1:3]))
            policy.reset(123)
            self.assertEqual(len(policy.history), 0)
            self.assertEqual(len(policy.pending_actions), 0)
            np.testing.assert_array_equal(policy.act(states[3]), plan[0])
            np.testing.assert_array_equal(sampled_histories[-1], states[3][None])

    def test_checkpoint_reproduces_seeded_actions_and_bounded_inference(self):
        policy = self.policy()
        history = np.arange(20, dtype=np.float32).reshape(2, 10)
        expected = policy.predict_action(history, seed=53)
        self.assertTrue(np.isfinite(expected).all())
        self.assertTrue(((expected >= 0) & (expected <= 512)).all())
        path = self.directory / "policy.pt"
        save_policy(policy, path, {"test": "round_trip"})
        restored = load_policy(path)
        np.testing.assert_array_equal(restored.predict_action(history, seed=53), expected)
        self.assertEqual(restored.metadata["test"], "round_trip")
        policy.reset(9)
        first = policy.act(history[-1])
        policy.reset(9)
        np.testing.assert_array_equal(policy.act(history[-1]), first)

    def test_history_padding_matches_repeated_initial_state(self):
        policy = self.policy()
        state = np.arange(10, dtype=np.float32)
        np.testing.assert_array_equal(policy.predict_action(state, seed=2),
                                      policy.predict_action(np.stack([state, state]), seed=2))
        for bad in (np.zeros(5), np.empty((0, 10)), np.full(10, np.nan)):
            with self.assertRaises(ValueError):
                policy.predict_action(bad)
        with self.assertRaises(ValueError):
            policy.predict_action(state, inference_steps=0)

    def test_single_action_horizon_supports_small_channel_counts(self):
        for channels in (2, 8):
            config = DiffusionPolicyConfig(prediction_horizon=1, action_horizon=1,
                                            channels=channels, residual_blocks=1,
                                            time_embed_dim=8, diffusion_steps=8,
                                            inference_steps=4)
            policy = DiffusionPolicy(ConditionalTemporalDenoiser(config), self.normalizer)
            action = policy.predict_action(np.zeros(10), seed=3)
            self.assertEqual(action.shape, (1, 2))
            self.assertTrue(np.isfinite(action).all())
        with self.assertRaises(ValueError):
            DiffusionPolicyConfig(channels=1)

    def test_tiny_zarr_train_save_reload_and_fine_tune(self):
        source = self.demonstration()
        output, last, history_path = [self.directory / name for name in ("best.pt", "last.pt", "history.json")]
        with contextlib.redirect_stdout(io.StringIO()):
            policy, history = train_policy([source], config=self.config, epochs=2,
                                            batch_size=8, lr=1e-3, seed=7,
                                            output=output, last_output=last,
                                            history_path=history_path)
        self.assertEqual(history["metadata"]["train_samples"] + history["metadata"]["val_samples"], 24)
        self.assertEqual(len(history["epochs"]), 2)
        self.assertTrue(all(np.isfinite(record["val_loss"]) for record in history["epochs"]))
        self.assertEqual(policy.metadata["epoch"], history["best_epoch"])
        self.assertAlmostEqual(policy.metadata["val_loss"], history["best_val_loss"])
        self.assertEqual(load_policy(last).metadata["epoch"], 2)
        self.assertEqual(json.loads(history_path.read_text())["best_epoch"], history["best_epoch"])
        with contextlib.redirect_stdout(io.StringIO()):
            tuned, _ = train_policy([source], resume=output, epochs=1, batch_size=8,
                                    seed=7,
                                    output=self.directory/"tuned.pt",
                                    last_output=self.directory/"tuned_last.pt",
                                    history_path=self.directory/"tuned.json")
        self.assertEqual(tuned.normalizer, policy.normalizer)
        self.assertEqual(tuned.config, policy.config)

        # Resplitting the same episodes while fine-tuning would mix the old
        # training/validation populations and invalidate held-out selection.
        old_groups = {ep["initial_group"]: ep["split"] for ep in policy.metadata["episodes"]}
        changed_seed = next(seed for seed in range(20)
                            if any(old_groups[ep["initial_group"]] != ep["split"]
                                   for ep in prepare_policy_data([source], seed=seed)["episodes"]))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "split|validation|group"):
            train_policy([source], resume=output, epochs=1, batch_size=8,
                          seed=changed_seed,
                          output=self.directory/"leak.pt",
                          last_output=self.directory/"leak_last.pt",
                          history_path=self.directory/"leak.json")


if __name__ == "__main__":
    unittest.main()
