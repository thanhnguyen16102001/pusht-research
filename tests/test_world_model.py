"""Regression tests for causal learned dynamics, data splits, and checkpoints."""
import copy
import pickle
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from Model_MLP_OneStepPredict import (build_model, load_model, predict, predict_batch,
                                     rollout, save_model, train_model)
from preprocess_onestep_data import prepare_data, preprocess_data


class WorldModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.model = build_model()
        self.stats = {"x_mean": torch.zeros(12), "x_std": torch.ones(12),
                      "y_mean": torch.zeros(5), "y_std": torch.ones(5)*0.01, "dt": 0.1}

    def dataset(self, name="one.zarr", action_offset=0.0):
        root = zarr.open_group(str(self.directory/name), mode="w")
        states, actions = [], []
        for episode in range(4):
            start = np.array([20+episode*40, 30, 100+episode*20, 200, 0.2])
            for t in range(5):
                states.append(start+np.array([t, t*0.5, t*0.2, -t*0.1, t*0.01]))
                actions.append([40+episode*40+t+action_offset, 40+t])
        data = root.create_group("data")
        data.array("state", np.asarray(states, dtype=np.float32))
        data.array("action", np.asarray(actions, dtype=np.float32))
        root.create_group("meta").array("episode_ends", np.arange(5, 21, 5))
        return self.directory/name

    def test_batch_matches_scalar_and_preserves_velocity_definition(self):
        states = np.random.default_rng(3).normal(size=(7, 10)).astype(np.float32)
        actions = np.random.default_rng(4).normal(size=(7, 2)).astype(np.float32)
        batch = predict_batch(self.model, self.stats, states, actions).numpy()
        scalar = np.stack([predict(self.model, self.stats, s, a) for s, a in zip(states, actions)])
        np.testing.assert_allclose(batch, scalar, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(batch[:, [2,3,6,7,9]],
                                   (batch[:, [0,1,4,5,8]]-states[:, [0,1,4,5,8]])/0.1,
                                   rtol=1e-3, atol=2e-6)

    def test_rollout_matches_recursive_predictions_and_keeps_initial_state(self):
        state = torch.arange(10, dtype=torch.float32)
        actions = torch.ones(3, 4, 2)
        prediction = rollout(self.model, self.stats, state, actions)
        self.assertEqual(tuple(prediction.shape), (3, 5, 10))
        torch.testing.assert_close(prediction[:, 0], state.expand(3, -1))
        current = state.expand(3, -1)
        for step in range(4):
            current = predict_batch(self.model, self.stats, current, actions[:, step])
            torch.testing.assert_close(prediction[:, step+1], current)
        torch.testing.assert_close(rollout(self.model, self.stats, state, actions[:1, :1])[:, 1],
                                   predict_batch(self.model, self.stats, state[None], actions[0, :1]))

    def test_invalid_inputs_and_dt_are_rejected(self):
        with self.assertRaises(ValueError):
            rollout(self.model, self.stats, torch.zeros(10), torch.zeros(1, 0, 2))
        with self.assertRaises(ValueError):
            predict_batch(self.model, self.stats, torch.full((1,10), float("nan")), torch.zeros(1,2))
        with self.assertRaises(ValueError):
            predict_batch(self.model, {**self.stats, "dt": 0}, torch.zeros(1,10), torch.zeros(1,2))

    def test_split_groups_initial_states_and_never_crosses_episodes(self):
        one, two = self.dataset(), self.dataset("two.zarr", action_offset=2)
        cache = prepare_data([one, one, two], val_ratio=0.5)
        self.assertEqual(len(cache["episodes"]), 8)
        groups = {}
        for record in cache["episodes"]:
            groups.setdefault(record["initial_group"], set()).add(record["split"])
        self.assertTrue(all(len(splits)==1 for splits in groups.values()))
        self.assertEqual(len(cache["train"]["x"])+len(cache["val"]["x"]), 8*4)
        for split in ("train", "val"):
            raw_y = cache[split]["y"]*cache["stats"]["y_std"]+cache["stats"]["y_mean"]
            np.testing.assert_allclose(raw_y[:, 0], 1.0, atol=1e-5)
            np.testing.assert_allclose(raw_y[:, 4], 0.01, atol=1e-5)

    def test_legacy_and_new_checkpoint_round_trip(self):
        legacy = self.directory/"legacy.pkl"
        torch.save({"weights": self.model.state_dict(), "stats": self.stats}, legacy)
        _, legacy_stats = load_model(legacy)
        self.assertEqual(legacy_stats["metadata"]["provenance"], "unknown")
        target = self.directory/"new.pt"
        stats = {**self.stats, "metadata": {"provenance": "known", "episodes": []}}
        before_device = next(self.model.parameters()).device
        save_model(self.model, stats, target)
        self.assertEqual(next(self.model.parameters()).device, before_device)
        restored, restored_stats = load_model(target)
        torch.testing.assert_close(predict_batch(self.model, stats, torch.zeros(1,10), torch.zeros(1,2)),
                                   predict_batch(restored, restored_stats, torch.zeros(1,10), torch.zeros(1,2)))
        self.assertEqual(restored_stats["metadata"]["provenance"], "known")

    def test_tiny_training_honors_cache_split_and_saves_best_and_last(self):
        dataset = self.dataset()
        cache_path = self.directory/"cache.pkl"
        cache = preprocess_data(dataset, cache_path, val_ratio=0.5)
        best, last, history = self.directory/"best.pt", self.directory/"last.pt", self.directory/"history.json"
        _, stats = train_model(data_pkl=cache_path, epochs=3, batch_size=8, val_ratio=0.1,
                               output=best, last_output=last, history_path=history)
        self.assertTrue(best.exists() and last.exists() and history.exists())
        meta = stats["metadata"]
        self.assertEqual(meta["val_pairs"], len(cache["val"]["x"]))
        self.assertTrue(meta["validation_is_heldout"])
        self.assertEqual(meta["best_val_loss"], min(x["val_loss"] for x in meta["history"]))
        _, saved_last = load_model(last)
        self.assertEqual(saved_last["metadata"]["checkpoint_role"], "last")
        _, saved_best = load_model(best)
        self.assertEqual(saved_best["metadata"]["epoch"], meta["best_epoch"])

    def test_contact_model_is_rigid_transform_equivariant(self):
        model = build_model("contact_local_v2", self.stats)
        state = torch.tensor([[124., 215., 30., -10., 200., 200., 0., 0., 0., 0.]])
        action = torch.tensor([[170., 215.]])
        angle = 0.7
        rotation = torch.tensor([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]], dtype=torch.float32)
        translation = torch.tensor([50., -30.])
        transformed = state.clone()
        for columns in ([0,1], [4,5]):
            transformed[:, columns] = state[:, columns] @ rotation.T+translation
        for columns in ([2,3], [6,7]):
            transformed[:, columns] = state[:, columns] @ rotation.T
        transformed[:, 8] += angle
        expected = predict_batch(model, self.stats, state, action)
        actual = predict_batch(model, self.stats, transformed, action @ rotation.T+translation)
        for columns in ([0,1], [4,5]):
            torch.testing.assert_close(actual[:, columns], expected[:, columns] @ rotation.T+translation,
                                       atol=1e-4, rtol=1e-4)
        for columns in ([2,3], [6,7]):
            torch.testing.assert_close(actual[:, columns], expected[:, columns] @ rotation.T,
                                       atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(actual[:, 8], expected[:, 8]+angle)
        torch.testing.assert_close(actual[:, 9], expected[:, 9], atol=1e-4, rtol=1e-4)

    def test_contact_gate_blocks_free_space_hallucination_but_keeps_moving_block(self):
        model = build_model("contact_local_v2", self.stats)
        torch.nn.init.zeros_(model.block_net[-1].weight)
        torch.nn.init.ones_(model.block_net[-1].bias)
        state = torch.tensor([[450., 450., 0., 0., 100., 100., 0., 0., 0., 0.]])
        action = state[:, :2].clone()
        next_state = predict_batch(model, self.stats, state, action)
        torch.testing.assert_close(next_state[:, 4:], state[:, 4:])
        state[:, 6] = 5.
        moving = predict_batch(model, self.stats, state, action)
        self.assertGreater(float((moving[:, 4:6]-state[:, 4:6]).abs().sum()), 0.)
        target = self.directory/"contact.pt"
        save_model(model, self.stats, target)
        restored, restored_stats = load_model(target)
        self.assertEqual(restored.architecture, "contact_local_v2")
        torch.testing.assert_close(predict_batch(restored, restored_stats, state, action), moving)

    def test_legacy_cache_without_fingerprints_is_not_claimed_verified_heldout(self):
        cache = prepare_data(self.dataset(), val_ratio=0.5)
        for record in cache["episodes"]:
            record.pop("initial_group")
            record.pop("dataset_signature")
        cache_path = self.directory/"legacy_cache.pkl"
        with cache_path.open("wb") as file:
            pickle.dump(cache, file)
        _, stats = train_model(data_pkl=cache_path, epochs=1, batch_size=8)
        self.assertEqual(stats["metadata"]["provenance"], "partial")
        self.assertFalse(stats["metadata"]["validation_is_heldout"])


if __name__ == "__main__":
    unittest.main()
