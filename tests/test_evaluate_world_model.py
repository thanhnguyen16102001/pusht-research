"""Regression tests for honest split selection and recursive rollout metrics."""

import json
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate_world_model import (baseline_rollouts, collect_windows, endpoint_metrics,
                                  evaluate_model, evaluate_predictions, select_episodes)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "data.zarr"
        root = zarr.open_group(str(self.path), mode="w")
        # Two independent episodes with a large discontinuity between them.
        raw = np.zeros((8, 5), dtype=np.float32)
        raw[:, 0] = [1, 2, 3, 4, 101, 102, 103, 104]
        raw[:, 2:4] = [250, 250]
        data = root.create_group("data")
        data.array("state", raw)
        data.array("action", np.zeros((8, 2), dtype=np.float32))
        data.array("n_contacts", np.array([[9], [0], [1], [0], [9], [0], [0], [0]], dtype=np.float32))
        root.create_group("meta").array("episode_ends", np.array([4, 8]))
        self.stats = {"dt": 0.1, "metadata": {
            "provenance": "known", "validation_is_heldout": True,
            "episodes": [{"source": str(self.path), "id": 0, "split": "val"},
                         {"source": str(self.path), "id": 1, "split": "train"}]}}

    def test_evaluator_supports_package_import(self):
        project = str(Path(__file__).resolve().parents[1])
        with patch.object(sys, "path", [project] + sys.path):
            module = importlib.import_module("src.evaluate_world_model")
        self.assertTrue(callable(module.evaluate_model))

    def test_unknown_checkpoint_requires_explicit_exploratory_selection(self):
        with self.assertRaisesRegex(ValueError, "Held-out episodes cannot"):
            select_episodes({"dt": 0.1}, [self.path])
        records, label = select_episodes({"dt": 0.1}, [self.path], allow_exploratory=True)
        self.assertEqual(len(records), 2)
        self.assertTrue(label.startswith("exploratory"))
        records, label = select_episodes(self.stats, [self.path], episode_ids=[1])
        self.assertEqual(records[0]["id"], 1)
        self.assertEqual(label, "exploratory_explicit_episodes")

    def test_known_checkpoint_uses_only_recorded_heldout_episodes(self):
        records, label = select_episodes(self.stats, [self.path])
        self.assertEqual(label, "heldout_validation")
        self.assertEqual([r["id"] for r in records], [0])
        self.stats["metadata"]["validation_is_heldout"] = False
        with self.assertRaises(ValueError):
            select_episodes(self.stats, [self.path])

    def test_changed_dataset_signature_cannot_claim_heldout_evaluation(self):
        self.stats["metadata"]["episodes"][0]["dataset_signature"] = "not-the-recorded-content"
        with self.assertRaisesRegex(ValueError, "contents changed"):
            select_episodes(self.stats, [self.path])

    def test_windows_do_not_cross_episodes_and_contacts_are_aligned_forward(self):
        records = [{"source": str(self.path), "id": i} for i in (0, 1)]
        truth, actions, contacts, windows = collect_windows(records, 2, 0.1, stride=1)
        np.testing.assert_array_equal(truth[:, :, 0], [[1, 2, 3], [2, 3, 4],
                                                     [101, 102, 103], [102, 103, 104]])
        np.testing.assert_array_equal(contacts, [[0, 1], [1, 0], [0, 0], [0, 0]])
        self.assertEqual(actions.shape, (4, 2, 2))
        self.assertEqual([w["start_step"] for w in windows], [0, 1, 0, 1])

    def test_endpoint_metrics_use_vector_errors_and_periodic_angles(self):
        truth = np.zeros((2, 10))
        prediction = truth.copy()
        prediction[:, 4:6] = [3, 4]
        prediction[:, 8] = 2 * np.pi
        metrics = endpoint_metrics(prediction, truth)
        self.assertEqual(metrics["block_position_rmse"], 5.0)
        self.assertAlmostEqual(metrics["block_angle_rmse_deg"], 0.0)
        prediction[0, 0] = np.nan
        metrics = endpoint_metrics(prediction, truth)
        self.assertEqual(metrics["invalid_fraction"], 0.5)
        self.assertEqual(metrics["valid_windows"], 1)

    def test_constant_velocity_baseline_matches_state_order(self):
        initial = np.array([[10, 20, 2, 3, 30, 40, 4, 5, 0.5, 0.1]], dtype=np.float32)
        baseline = baseline_rollouts(initial, 2, 0.1)
        np.testing.assert_allclose(baseline["constant_velocity"][0, 2],
                                   [10.4, 20.6, 2, 3, 30.8, 41, 4, 5, 0.52, 0.1])
        np.testing.assert_array_equal(baseline["persistence"][0, 2], initial[0])

    def test_unknown_contacts_are_not_silently_labeled_no_contact(self):
        truth = np.zeros((1, 3, 10))
        rows = evaluate_predictions({"model": truth.copy()}, truth,
                                    np.full((1, 2), -1), [2], 0.1)
        self.assertEqual([r["windows"] for r in rows], [1, 0, 0])
        self.assertIsNone(rows[1]["block_position_rmse"])

    def test_report_uses_recursive_states_and_saves_reproducible_artifacts(self):
        # This deliberately imperfect dynamics model doubles its own previous x.
        # A teacher-forced second step would predict 4 from true x=2; starting at
        # x=2 must recursively predict 8 at horizon2, rather than 6 from true x=3.
        def recursive_model(model, stats, initial, actions):
            states = [initial]
            for _ in range(actions.shape[1]):
                next_state = states[-1].clone()
                next_state[:, 0] *= 2
                states.append(next_state)
            return torch.stack(states, dim=1)

        output = Path(self.temp.name) / "report"
        report = evaluate_model(torch.nn.Identity(), self.stats, [self.path], output,
                                horizons=[1, 2], stride=1, rollout_fn=recursive_model)
        self.assertEqual(report["evaluation_split"], "heldout_validation")
        self.assertEqual(report["windows"], 2)
        h2 = next(row for row in report["metrics"] if row["method"] == "world_model"
                  and row["horizon"] == 2 and row["subset"] == "all")
        self.assertAlmostEqual(h2["agent_position_rmse"], np.sqrt((1 ** 2 + 4 ** 2) / 2))
        self.assertEqual(json.loads((output / "model_report.json").read_text())["windows"], 2)
        with np.load(output / "example_rollout.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["world_model"][:, 0], [1, 2, 4])
        self.assertTrue((output / "metrics.csv").is_file())
        self.assertTrue((output / "rollout_errors.png").is_file())


if __name__ == "__main__":
    unittest.main()
