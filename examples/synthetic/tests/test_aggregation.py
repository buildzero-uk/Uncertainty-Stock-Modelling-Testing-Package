"""Regression tests for sparse-column aggregation and Figure 3 screening."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mwe_core.aggregation import (  # noqa: E402
    LAYERS,
    MATERIALS,
    aggregate_draws,
    classify_material,
    summarise_targets,
)


def path(layer, material, technology="example"):
    return f"{layer}///wall///load bearing///{technology}///synthetic specification///{material}"


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.brick = path("structure", "Brick")
        self.timber = path("space", "Timber stud")
        self.glass = path("skin", "Glass")
        self.a = pd.DataFrame({self.brick: [1., 2., 3.]})
        # Nonzero materials absent from the first building must survive.
        self.b = pd.DataFrame({self.glass: [4., 5., 6.], self.timber: [7., 8., 9.]})

    def test_later_building_unique_material_is_retained(self):
        areas, city = aggregate_draws({"a": self.a, "b": self.b}, {"a": "A", "b": "A"})
        np.testing.assert_array_equal(city[self.glass], [4, 5, 6])
        np.testing.assert_array_equal(city[self.timber], [7, 8, 9])
        np.testing.assert_array_equal(city.sum(axis=1), [12, 15, 18])
        assert_frame_equal(areas["A"], city)

    def test_draw_and_first_seen_column_order_are_preserved(self):
        _, city = aggregate_draws({"a": self.a, "b": self.b}, {"a": "A", "b": "B"})
        self.assertEqual(city.columns.tolist(), [self.brick, self.glass, self.timber])
        self.assertEqual(city.index.tolist(), [0, 1, 2])
        np.testing.assert_array_equal(city.iloc[0], [1, 4, 7])
        # Reversing building or column order must not change paired values.
        _, reverse = aggregate_draws({"b": self.b.iloc[:, ::-1], "a": self.a}, {"a": "A", "b": "B"})
        assert_frame_equal(city.sort_index(axis=1), reverse.sort_index(axis=1))

    def test_area_city_and_target_mass_conservation(self):
        extra = pd.DataFrame({path("services", "Copper"): [10., 20., 30.], path("skin", "slate"): [2., 2., 2.]})
        areas, city = aggregate_draws({"a": self.a, "b": self.b, "c": extra}, {"a": "A", "b": "B", "c": "A"})
        expected = self.a.sum(axis=1) + self.b.sum(axis=1) + extra.sum(axis=1)
        np.testing.assert_allclose(city.sum(axis=1), expected)
        assert_frame_equal(areas["A"] + areas["B"], city)
        targets = summarise_targets(city)
        np.testing.assert_allclose(targets.total, expected)
        np.testing.assert_allclose(targets[[f"layer__{x}" for x in LAYERS]].sum(axis=1), expected)
        np.testing.assert_allclose(targets[[f"material__{x}" for x in MATERIALS]].sum(axis=1), expected)
        self.assertEqual(targets.columns.tolist(), ["total", *[f"layer__{x}" for x in LAYERS], *[f"material__{x}" for x in MATERIALS]])

    def test_screening_uses_terminal_material_and_original_priority(self):
        expected = {
            "Timber": "other", "  TIMBER  ": "other", "Timber stud": "timber",
            "slate": "other", "plywood": "other", "Granite": "stone", "ballast": "stone",
            "metal-frame Glass": "metal", "aluminium": "metal", "Zinc": "metal",
            "Concrete brick": "concrete", "Glass": "glass", "brick": "brick",
        }
        for material, group in expected.items():
            with self.subTest(material=material):
                self.assertEqual(classify_material(material), group)
        d = pd.DataFrame({path("structure", "slate", technology="Stone"): [5.], path("skin", "plywood", technology="Timber"): [7.]})
        t = summarise_targets(d)
        self.assertEqual(t["material__stone"].iloc[0], 0.)
        self.assertEqual(t["material__timber"].iloc[0], 0.)
        self.assertEqual(t["material__other"].iloc[0], 12.)

    def test_totals_count_each_input_once(self):
        d = pd.DataFrame({path("structure", "steel reinforced concrete"): [11.], path("space", "Timber stud"): [3.]})
        t = summarise_targets(d)
        self.assertEqual(t.total.iloc[0], 14.)
        self.assertEqual(t["material__metal"].iloc[0], 11.)
        self.assertEqual(t["material__concrete"].iloc[0], 0.)
        self.assertEqual(t.filter(like="material__").sum(axis=1).iloc[0], 14.)
        # A target frame cannot accidentally be used as a raw-material frame.
        with self.assertRaises(ValueError):
            summarise_targets(t)

    def test_misaligned_or_duplicate_draws_are_rejected(self):
        for index in ([1, 2, 3], [0, 0, 1], [1, 0, 2], [0., 1., 2.]):
            with self.subTest(index=index):
                bad = self.b.copy()
                bad.index = index
                with self.assertRaises(ValueError):
                    aggregate_draws({"a": self.a, "b": bad}, {"a": "A", "b": "A"})
        with self.assertRaises(ValueError):
            aggregate_draws({"a": self.a, "b": self.b.iloc[:2]}, {"a": "A", "b": "A"})

    def test_nonfinite_negative_and_non_numeric_masses_are_rejected(self):
        for value in (np.nan, np.inf, -np.inf, -0.01, "2", 1j, True):
            with self.subTest(value=value):
                bad = pd.DataFrame({self.brick: [value]})
                with self.assertRaises(ValueError):
                    aggregate_draws({"a": bad}, {"a": "A"})
                with self.assertRaises(ValueError):
                    summarise_targets(bad)

    def test_duplicate_columns_and_bad_paths_are_rejected(self):
        duplicate = pd.DataFrame([[1., 2.]], columns=[self.brick, self.brick])
        with self.assertRaises(ValueError):
            aggregate_draws({"a": duplicate}, {"a": "A"})
        for column in ("brick", "structure///wall///brick", path("invalid", "brick"), path("structure", "")):
            with self.subTest(column=column):
                with self.assertRaises(ValueError):
                    aggregate_draws({"a": pd.DataFrame({column: [1.]})}, {"a": "A"})

    def test_mapping_must_cover_all_buildings(self):
        with self.assertRaises(ValueError):
            aggregate_draws({"a": self.a, "b": self.b}, {"a": "A"})
        with self.assertRaises(ValueError):
            aggregate_draws({"a": self.a}, {"a": ""})
        # A larger lookup can legitimately be reused for a building subset.
        _, city = aggregate_draws({"a": self.a}, {"a": "A", "unused": "B"})
        assert_frame_equal(city, self.a)

    def test_all_zero_columns_can_be_absent_without_mutating_inputs(self):
        empty = pd.DataFrame(index=pd.RangeIndex(3))
        before = self.a.copy(deep=True)
        areas, city = aggregate_draws({"empty": empty, "a": self.a}, {"empty": "E", "a": "A"})
        assert_frame_equal(city, before)
        assert_frame_equal(self.a, before)
        np.testing.assert_array_equal(areas["E"].to_numpy(), np.zeros((3, 1)))
        zero_targets = summarise_targets(empty)
        np.testing.assert_array_equal(zero_targets.to_numpy(), np.zeros((3, 12)))

    def test_empty_building_collection_or_draw_sequence_is_rejected(self):
        with self.assertRaises(ValueError):
            aggregate_draws({}, {})
        with self.assertRaises(ValueError):
            aggregate_draws({"a": self.a.iloc[:0]}, {"a": "A"})


if __name__ == "__main__":
    unittest.main()
