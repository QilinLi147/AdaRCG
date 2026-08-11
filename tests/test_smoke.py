from __future__ import annotations

import unittest

import numpy as np
import torch

from adarcg.config import PROFILES, profile
from adarcg.data import dataset_folds
from adarcg.engine import causal_integrate
from adarcg.model import AdaRCG
from adarcg.prepare import temporal_sequence
from adarcg.regions import dataset_regions


class PublicPackageSmokeTest(unittest.TestCase):
    def test_four_dataset_profiles(self) -> None:
        self.assertEqual(set(PROFILES), {"seed", "mped", "seediv", "seedv"})
        self.assertEqual(profile("SEED-IV").classes, 4)
        self.assertEqual(profile("SEED-V").classes, 5)

    def test_temporal_cache_layout(self) -> None:
        values = np.arange(12 * 62 * 5, dtype=np.float32).reshape(12, 62, 5)
        windows = temporal_sequence(values)
        self.assertEqual(windows.shape, (3, 62, 50))
        np.testing.assert_array_equal(windows[0, 0].reshape(10, 5), values[:10, 0])

    def test_model_forward(self) -> None:
        current = profile("seed")
        membership, adjacency = dataset_regions("seed")
        model = AdaRCG(
            torch.from_numpy(membership),
            torch.from_numpy(adjacency),
            input_dim=50,
            classes=current.classes,
            hidden_dim=current.hidden_dim,
            rank=current.graph_rank,
            top_regions=current.selected_regions,
            dropout=current.dropout,
            structured_evidence=True,
        ).eval()
        with torch.no_grad():
            outputs = model(torch.randn(2, 62, 50))
            logits = model.combined_logits(outputs)
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(tuple(outputs["adjacency"].shape), (7, 7))

    def test_trial_boundary_reset(self) -> None:
        probability = torch.tensor([[0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
        result = causal_integrate(probability, [(0, 1), (0, 1), (0, 2)], 0.5)
        torch.testing.assert_close(result[1], torch.tensor([0.5, 0.5]))
        torch.testing.assert_close(result[2], probability[2])

    def test_four_dataset_partition_interfaces(self) -> None:
        seed_session = np.repeat(np.arange(3), 15)
        seed_trial = np.tile(np.arange(15), 3)
        seed_data = {
            "subject": np.zeros(45, dtype=np.int64),
            "session": seed_session,
            "trial": seed_trial,
            "y": seed_trial % 3,
        }
        self.assertEqual(len(dataset_folds(seed_data, profile("seed"), 0)), 1)

        for name, classes, trials_per_session in (
            ("seediv", 4, 12),
            ("seedv", 5, 15),
        ):
            session = np.repeat(np.arange(3), trials_per_session)
            trial = np.tile(np.arange(trials_per_session), 3)
            data = {
                "subject": np.zeros(len(session), dtype=np.int64),
                "session": session,
                "trial": trial,
                "y": trial % classes,
            }
            fold = dataset_folds(data, profile(name), 0)[0]
            combined = np.concatenate((fold.train, fold.validation, fold.test))
            self.assertEqual(len(np.unique(combined)), len(combined))

        mped_trial = np.arange(28, dtype=np.int64)
        mped_data = {
            "subject": np.zeros(28, dtype=np.int64),
            "session": np.zeros(28, dtype=np.int64),
            "trial": mped_trial,
            "emotion": mped_trial % 7,
            "y": mped_trial % 3,
        }
        self.assertEqual(len(dataset_folds(mped_data, profile("mped"), 0)), 4)


if __name__ == "__main__":
    unittest.main()
