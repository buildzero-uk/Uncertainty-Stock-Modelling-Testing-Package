"""Small end-to-end tests of the actual sampling primitives and their assembly."""
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from mwe_core.age import generate_slots_for_age_material
from mwe_core.materials import TreeNode
from mwe_core.pipeline import read_inputs,simulate,SharedPools

HERE=Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b,cls.training,cls.hmi,cls.matrix=read_inputs(HERE/'data')
        cls.buildings=b.iloc[[0,1,12,13]].copy()

    def test_repeatable_pipeline_and_input_preservation(self):
        original=self.hmi.copy(deep=True)
        kwargs=dict(draws=12,pool_size=2,refreshes=3,seed=17)
        a=simulate(self.buildings,self.training,self.hmi,self.matrix,**kwargs)
        b=simulate(self.buildings,self.training,self.hmi,self.matrix,**kwargs)
        pd.testing.assert_frame_equal(a.city_paths,b.city_paths,check_exact=True)
        pd.testing.assert_frame_equal(a.draw_metadata,b.draw_metadata,check_exact=True)
        pd.testing.assert_frame_equal(self.hmi,original)
        self.assertTrue(a.checks['passed'])
        self.assertEqual(len(a.area_paths),2)
        self.assertTrue(a.draw_metadata.groupby('draw').shared_wall_z.nunique().eq(1).all())
        self.assertTrue(a.draw_metadata.groupby('draw').internal_wall_length_m.nunique().gt(1).any())

    def test_pool_identity_and_refresh_boundaries(self):
        p=SharedPools(self.hmi,list(self.matrix.index),3,42)
        b=self.buildings.iloc[0]
        prototype=(b.wall_material,b.roof_material,b.roof_shape)
        a=p.get(b.area_id,b.age,prototype,0)
        self.assertIs(a,p.get(b.area_id,b.age,prototype,0))
        self.assertIsNot(a,p.get(b.area_id,b.age,prototype,1))
        self.assertIsNot(a,p.get('AnotherArea',b.age,prototype,0))

    def test_asymmetric_age_matrix_direction(self):
        ages=['old','middle','new']
        # Observed columns: old -> new, middle -> old, new -> middle.
        csv_values=np.array([[0,1,0],[0,0,1],[1,0,0]],dtype=float)
        pools=[]
        for i in range(3):
            tree=TreeNode('root',-1)
            tree.mi=float(i)
            pools.append([[tree]])
        trees,indices=generate_slots_for_age_material(8,ages,[],csv_values.T,pools,42,'old','lazy')
        self.assertTrue((indices==2).all())
        self.assertTrue(all(t.mi==2 for t in trees))
        self.assertIsNot(trees[0],pools[2][0][0])

    def test_invalid_refresh_configuration(self):
        with self.assertRaisesRegex(ValueError,'divisible'):
            simulate(self.buildings,self.training,self.hmi,self.matrix,draws=11,refreshes=3)


if __name__=='__main__':
    unittest.main()
