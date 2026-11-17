"""Alignment, causality, split and normalization regressions for imitation data."""

import copy
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
import unittest

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from diffusion_policy_data import (PolicySequenceDataset, discover_zarr_datasets,
                                   encode_states, make_policy_normalizer, prepare_policy_data)
from preprocess_onestep_data import prepare_data


class DiffusionPolicyDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def store(self, name="one.zarr", action_offset=0.0):
        path = self.directory / name
        root = zarr.open_group(str(path), mode="w")
        states, actions = [], []
        for episode in range(4):
            for step in range(5):
                states.append([20+episode*50+step, 30+step*2,
                               100+episode*20+step*0.2, 200-step*0.1, 0.2+step*0.03])
                actions.append([40+episode*50+step+action_offset, 60+step*3])
        root.create_group("data").array("state", np.asarray(states, dtype=np.float32))
        root["data"].array("action", np.asarray(actions, dtype=np.float32))
        root.create_group("meta").array("episode_ends", np.arange(5, 21, 5))
        return path

    def cache(self):
        return prepare_policy_data(self.store(), val_ratio=0.5, seed=9)

    def test_discovery_recurses_deduplicates_paths_and_validates_empty_inputs(self):
        one, two = self.store(), self.store("nested/two.zarr")
        actual = discover_zarr_datasets([self.directory, one, str(self.directory / "**/*.zarr")])
        self.assertEqual(actual, sorted([one, two], key=str))
        with self.assertRaises(FileNotFoundError):
            discover_zarr_datasets(self.directory / "missing*.zarr")
        empty = self.directory / "empty"
        empty.mkdir()
        with self.assertRaises(ValueError):
            discover_zarr_datasets(empty)

    def test_reuses_exact_preprocessor_splits_and_deduplicates_contents(self):
        one, two = self.store(), self.store("two.zarr", action_offset=2)
        duplicate = self.directory / "copied.zarr"
        shutil.copytree(one, duplicate)
        cache = prepare_policy_data([one, two, duplicate], val_ratio=0.5, seed=9)
        reference = prepare_data(discover_zarr_datasets([one, two, duplicate]), val_ratio=0.5, seed=9)
        self.assertEqual(cache["episodes"], reference["episodes"])
        self.assertEqual(len(cache["episodes"]), 8)
        self.assertEqual(len(cache["config"]["sources"]), 2)
        groups = {}
        for episode in cache["episodes"]:
            groups.setdefault(episode["initial_group"], set()).add(episode["split"])
        self.assertTrue(all(len(split) == 1 for split in groups.values()))
        train = PolicySequenceDataset(cache, "train")
        val = PolicySequenceDataset(cache, "val")
        self.assertEqual(len(train)+len(val), 40)
        self.assertFalse({i for i, _ in train.indices} & {i for i, _ in val.indices})

    def test_sequence_alignment_start_padding_and_masked_terminal_targets(self):
        cache = self.cache()
        normalizer = make_policy_normalizer(cache)
        dataset = PolicySequenceDataset(cache, "train", obs_horizon=3,
                                        prediction_horizon=4, normalizer=normalizer)
        episode_index, step = dataset.indices[0]
        self.assertEqual(step, 0)
        episode = cache["episodes"][episode_index]
        start, stop = episode["start"], episode["stop"]
        first = dataset[0]
        encoded = encode_states(cache["raw"]["state"])
        expected = (encoded-np.asarray(normalizer["obs_mean"])) / normalizer["obs_std"]
        np.testing.assert_allclose(first["obs"], np.repeat(expected[start:start+1], 3, axis=0), atol=1e-6)
        np.testing.assert_allclose(first["action"], cache["raw"]["action"][start:start+4]/256-1)
        self.assertTrue(first["mask"].all())
        final_index = dataset.indices.index((episode_index, stop-start-1))
        final = dataset[final_index]
        np.testing.assert_array_equal(final["mask"], [True, False, False, False])
        expected_final = cache["raw"]["action"][stop-1]/256-1
        np.testing.assert_allclose(final["action"], np.repeat(expected_final[None], 4, axis=0))
        np.testing.assert_allclose(final["obs"], expected[stop-3:stop], atol=1e-6)

    def test_future_states_cannot_leak_into_observation_history(self):
        cache = self.cache()
        normalizer = make_policy_normalizer(cache)
        original = PolicySequenceDataset(cache, "train", 3, 4, normalizer)
        episode_index, _ = original.indices[0]
        start = cache["episodes"][episode_index]["start"]
        changed = copy.deepcopy(cache)
        changed["raw"]["state"][start+3:start+5] += 100
        mutated = PolicySequenceDataset(changed, "train", 3, 4, normalizer)
        sample_index = original.indices.index((episode_index, 2))
        np.testing.assert_array_equal(original[sample_index]["obs"], mutated[sample_index]["obs"])
        np.testing.assert_array_equal(original[sample_index]["action"], mutated[sample_index]["action"])

    def test_validation_states_do_not_influence_observation_normalization(self):
        cache = self.cache()
        before = make_policy_normalizer(cache)
        for episode in cache["episodes"]:
            if episode["split"] == "val":
                cache["raw"]["state"][episode["start"]:episode["stop"]] += 10000
        after = make_policy_normalizer(cache)
        self.assertEqual(before["obs_mean"], after["obs_mean"])
        self.assertEqual(before["obs_std"], after["obs_std"])
        raw_train = np.concatenate([cache["raw"]["state"][e["start"]:e["stop"]]
                                    for e in cache["episodes"] if e["split"] == "train"])
        np.testing.assert_allclose(before["obs_mean"], encode_states(raw_train).mean(0), atol=1e-5)

    def test_angle_encoding_is_continuous_at_wrap_and_preserves_velocity_features(self):
        state = np.arange(10, dtype=np.float32)
        wrapped = state.copy()
        state[8], wrapped[8] = -np.pi+1e-6, np.pi+1e-6
        np.testing.assert_allclose(encode_states(state), encode_states(wrapped), atol=1e-6)
        np.testing.assert_array_equal(encode_states(state)[:8], state[:8])
        self.assertEqual(encode_states(state)[-1], state[-1])
        with self.assertRaises(ValueError):
            encode_states(np.zeros(5))

    def test_trusted_cache_preserves_splits_and_rejects_legacy_or_wrong_frequency(self):
        cache = self.cache()
        path = self.directory / "cache.pkl"
        with path.open("wb") as stream:
            pickle.dump(cache, stream)
        loaded = prepare_policy_data(data_pkl=path, val_ratio=0.1, seed=123)
        self.assertEqual(loaded["episodes"], cache["episodes"])
        self.assertEqual(loaded["config"]["seed"], 9)
        with self.assertRaises(ValueError):
            prepare_policy_data(data_pkl=path, control_hz=100)
        with self.assertRaises(ValueError):
            prepare_policy_data([self.directory], data_pkl=path)
        with path.open("wb") as stream:
            pickle.dump({"train": {}}, stream)
        with self.assertRaisesRegex(ValueError, "schema_version"):
            prepare_policy_data(data_pkl=path)

    def test_recorded_513_targets_are_preserved_and_reported(self):
        cache = self.cache()
        episode = next(ep for ep in cache["episodes"] if ep["split"] == "train")
        cache["raw"]["action"][episode["start"]] = [513.0, 512.0]
        normalizer = make_policy_normalizer(cache)
        self.assertEqual(normalizer["action_out_of_bounds_count"], 1)
        self.assertEqual(normalizer["recorded_action_max"][0], 513.0)
        dataset = PolicySequenceDataset(cache, "train", normalizer=normalizer)
        self.assertAlmostEqual(float(dataset[0]["action"][0, 0]), 1.00390625)
        cache["raw"]["action"][episode["start"], 0] = 600
        with self.assertRaisesRegex(ValueError, "absolute XY"):
            make_policy_normalizer(cache)

    def test_invalid_cache_boundaries_and_split_leakage_are_rejected(self):
        cache = self.cache()
        changed = copy.deepcopy(cache)
        changed["episodes"][1]["start"] -= 1
        with self.assertRaisesRegex(ValueError, "contiguous"):
            PolicySequenceDataset(changed, "train")
        train = next(ep for ep in cache["episodes"] if ep["split"] == "train")
        val = next(ep for ep in cache["episodes"] if ep["split"] == "val")
        val["initial_group"] = train["initial_group"]
        with self.assertRaisesRegex(ValueError, "both train and validation"):
            PolicySequenceDataset(cache, "train")


if __name__ == "__main__":
    unittest.main()
