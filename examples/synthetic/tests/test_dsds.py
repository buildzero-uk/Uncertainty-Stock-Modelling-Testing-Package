"""Focused tests of seeded DSDS behaviour and the shared wall factor."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from generate_data import AGES, generate, training
from mwe_core.dsds import HouseMCModel


class TestSyntheticDSDS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = HouseMCModel(random_state=42).fit(training())

    def test_fixtures_are_reproducible_and_probability_orientation(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            generate(Path(a))
            generate(Path(b))
            for name in ["buildings.csv", "training.csv", "age_probabilities.csv", "hmi.csv"]:
                self.assertEqual((Path(a) / name).read_bytes(), (Path(b) / name).read_bytes())
            probs = pd.read_csv(Path(a) / "age_probabilities.csv", index_col="sampled_age")
            np.testing.assert_allclose(probs.sum(axis=0), 1)
            self.assertEqual(list(probs.columns), AGES)
            self.assertEqual(list(probs.index), AGES)
            # Asymmetry makes accidental transposition detectable.
            self.assertFalse(np.allclose(probs.sum(axis=1), 1))
            hmi = pd.read_csv(Path(a) / "hmi.csv")
            self.assertTrue((hmi["MI"] > 0).all())
            self.assertTrue(hmi["unit"].isin(["kg/m2", "kg/m", "kg/item"]).all())
            for age in AGES:
                self.assertEqual(set(hmi.loc[hmi[age] == "YES", "Layer"]),
                                 {"Structure", "Skin", "Space", "Services"})

    def test_seeded_counts_and_lengths_repeat(self):
        row = pd.DataFrame({"T": [120.0], "P": [32.0], "F": [2]})
        draws = [self.model.sample_sequential(row, n_samples=8,
                 rng=np.random.default_rng(11), wall_rng=np.random.default_rng(43))
                 for _ in range(2)]
        for key in ["R", "W", "D", "L"]:
            np.testing.assert_array_equal(draws[0][key], draws[1][key])
            self.assertEqual(draws[0][key].shape, (1, 8))
            self.assertTrue(np.isfinite(draws[0][key]).all())
        self.assertTrue((draws[0]["L"] > 0).all())
        self.assertTrue((draws[0]["W"] >= 2).all())

    def test_wall_factor_shared_but_counts_have_separate_streams(self):
        n = 8
        expected = stats.norm.ppf(0.05 + 0.90 * np.random.default_rng(43).random(n))
        count_draws = []
        for count_seed, area, perimeter in [(11, 120.0, 32.0), (12, 175.0, 38.0)]:
            row = pd.DataFrame({"T": [area], "P": [perimeter], "F": [2]})
            out = self.model.sample_sequential(row, n_samples=n,
                rng=np.random.default_rng(count_seed), wall_rng=np.random.default_rng(43))
            conditional_rows = pd.DataFrame({"T": np.repeat(area, n),
                "P": np.repeat(perimeter, n), "F": np.repeat(2, n),
                "R": out["R"][0], "D": out["D"][0]})
            mu = self.model._predict_mu_lnL(conditional_rows)
            factor = (np.log(out["L"][0]) - mu) / self.model.sigma_L
            np.testing.assert_allclose(factor, expected, rtol=1e-10, atol=1e-10)
            count_draws.append(out["R"][0])
        self.assertFalse(np.array_equal(*count_draws))


if __name__ == "__main__":
    unittest.main()
