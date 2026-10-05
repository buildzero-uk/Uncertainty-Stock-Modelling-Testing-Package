"""Hand-calculated checks for reused material-tree and dimension operations.

These synthetic masses check research-code semantics, not realistic Bristol
material intensities. The fixture deliberately spans nine working dimension
routes and does not claim to cover all research heating configurations.
"""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mwe_core.materials import (  # noqa: E402
    TreeNode, MaterialTreeBuilder, UncertaintyForestBuilder,
)
from mwe_core.dimensions import apply_dimensions  # noqa: E402


# Expected dimensions and masses are hand-calculated, rather than obtained
# through the production dimension function or a copied conditional dispatch.
ROUTES = [
    (('Structure', 'Wall', 'Load bearing'), 200., 172.8, 34560.),
    (('Structure', 'Floor', 'Structure'), 150., 60., 9000.),
    (('Skin', 'Roof', 'Finish'), 40., 70., 2800.),
    (('Space', 'Wall', 'Internal partition'), 80., 54., 4320.),
    (('Space', 'Floor', 'Finish'), 10., 60., 600.),
    (('Skin', 'Opening', 'Window'), 15., 6., 90.),
    (('Skin', 'Opening', 'Door'), 20., 2., 40.),
    (('Space', 'Opening', 'Door'), 10., 5., 50.),
    (('Services', 'Electrical', 'Wiring'), .2, 48., 9.6),
]


def material_path(route):
    return [*route, 'Synthetic technology', 'Synthetic specification', 'Fixture material']


def fixed_tree():
    tree = TreeNode('root', -1)
    for route, mi, _, _ in ROUTES:
        tree.add_path(material_path(route), mi)
    return tree


class DimensionAndMaterialTests(unittest.TestCase):
    def setUp(self):
        self.ngd = dict(area=60., floor=2, perimeter=32., wall_height=5.4, roof_area=70.)
        self.inferred = dict(room=4, window=6, interior_door=5, interior_wall=20.)

    def scale(self, tree, ngd=None):
        dims = self.ngd if ngd is None else ngd
        apply_dimensions(tree, dims, self.inferred, dims['area'], floor_count=dims['floor'])

    def test_nine_dimension_routes_match_hand_calculated_masses(self):
        tree = fixed_tree()
        self.scale(tree)
        masses = []
        for route, mi, dimension, mass in ROUTES:
            with self.subTest(route=route):
                actual = tree.query(material_path(route))
                self.assertAlmostEqual(actual[0], mi)
                self.assertAlmostEqual(actual[1], dimension)
                self.assertAlmostEqual(actual[2], mass)
                masses.append(actual[2])
        self.assertAlmostEqual(sum(masses), 51469.6)

    def test_single_storey_space_floor_is_zero_without_removing_ground_floor(self):
        tree = fixed_tree()
        single = {**self.ngd, 'floor': 1, 'wall_height': 2.7}
        self.scale(tree, single)
        np.testing.assert_allclose(
            tree.query(material_path(('Space', 'Floor', 'Finish'))), [10., 0., 0.])
        np.testing.assert_allclose(
            tree.query(material_path(('Structure', 'Floor', 'Structure'))), [150., 60., 9000.])

    def test_sampling_retains_all_materials_of_selected_specification(self):
        rows = []
        for technology in ['Metal frame', 'Timber frame']:
            for specification in ['Specification A', 'Specification B']:
                for material, mi in [('Glass', 12.), ('Frame material', 3.)]:
                    rows.append({
                        'Layer': 'Skin', 'Function': 'Opening', 'Sub-Function': 'Window',
                        'Technology': technology, 'Specification': specification,
                        'Material': material, 'MI': mi, 'Synthetic cohort': 'YES',
                    })
        builder = MaterialTreeBuilder(pd.DataFrame(rows), ['Synthetic cohort'])
        original = builder.build_tree_for_age_band('Synthetic cohort')
        before = deepcopy(original.to_dict())
        forest = UncertaintyForestBuilder(['Synthetic cohort'], None)
        for sample in forest.run_one_sample(original, n_samples=8, random_state=17):
            technology_nodes = sample.children['Skin'].children['Opening'].children['Window'].children
            self.assertEqual(len(technology_nodes), 1)
            technology = next(iter(technology_nodes.values()))
            self.assertEqual(len(technology.children), 1)
            specification = next(iter(technology.children.values()))
            self.assertEqual(set(specification.children), {'Glass', 'Frame material'})
            self.assertEqual(specification.children['Glass'].mi, 12.)
            self.assertEqual(specification.children['Frame material'].mi, 3.)
        self.assertEqual(original.to_dict(), before)

    def test_copying_pool_candidate_prevents_cross_building_dimension_mutation(self):
        candidate = fixed_tree()
        pool = [candidate]
        before = deepcopy(candidate.to_dict())
        first = deepcopy(pool[0])
        self.scale(first)
        second = deepcopy(pool[0])
        self.scale(second, {**self.ngd, 'area': 120.})
        self.assertEqual(candidate.to_dict(), before)
        self.assertEqual(pool[0].to_dict(), before)
        floor_path = material_path(('Structure', 'Floor', 'Structure'))
        self.assertEqual(candidate.query(floor_path), 150.)
        np.testing.assert_allclose(first.query(floor_path), [150., 60., 9000.])
        np.testing.assert_allclose(second.query(floor_path), [150., 120., 18000.])
        self.assertIsNot(first, second)


if __name__ == '__main__':
    unittest.main()
