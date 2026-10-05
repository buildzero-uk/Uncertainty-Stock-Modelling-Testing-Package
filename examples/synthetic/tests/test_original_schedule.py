"""Regression oracles for the original v4 age/prototype and wall schedules.

The small material inventory below has a known choice sequence: one of three
technologies, then one of two specifications. Expected pools and age/candidate
draws are calculated directly from those production steps, without calling the
public pool/slot helpers to construct the expected values.
"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mwe_core.pipeline import (SharedPools,read_inputs,resolve_area_seeds,
                              sample_material_block,simulate)

HERE=Path(__file__).resolve().parents[1]
AGES=['old','new']
TECHNOLOGIES=['Brick option','Stone option','Concrete option']
SPECIFICATIONS=['Option A','Option B']
PROTOTYPE=('Unknown','Unknown','pitched')


def fixture_hmi():
    rows=[]
    for technology in TECHNOLOGIES:
        for specification in SPECIFICATIONS:
            for material,mi in [('Marker A',1.),('Marker B',2.)]:
                rows.append({'Layer':'Structure','Function':'Wall','Sub-Function':'Load bearing',
                             'Technology':technology,'Specification':specification,
                             'Material':material,'MI':mi,'old':'YES','new':'YES'})
    return pd.DataFrame(rows)


def signature(tree):
    technology_nodes=tree.children['Structure'].children['Wall'].children['Load bearing'].children
    assert len(technology_nodes)==1
    technology,node=next(iter(technology_nodes.items()))
    assert len(node.children)==1
    specification,leaf_parent=next(iter(node.children.items()))
    assert set(leaf_parent.children)=={'Marker A','Marker B'}
    return technology,specification


def oracle_pool(area_seed,block,pool_size):
    rng=np.random.default_rng(area_seed+block)
    return [(TECHNOLOGIES[rng.integers(3)],SPECIFICATIONS[rng.integers(2)])
            for _ in range(pool_size)]


def oracle_slots(area_seed,probabilities,block_size,pool_size):
    rng=np.random.default_rng(area_seed)
    ages=rng.choice(2,size=block_size,p=probabilities)
    choices=np.array([rng.integers(pool_size) for _ in ages])
    return ages,choices


class OriginalScheduleTests(unittest.TestCase):
    def test_area_seed_default_uses_first_appearance_before_building_sort(self):
        buildings=pd.DataFrame({'building_id':['z_first','a_second','b_third'],
                                'area_id':['AreaB','AreaA','AreaB']})
        self.assertEqual(resolve_area_seeds(buildings,42),{'AreaB':42,'AreaA':43})
        explicit={'AreaA':901,'AreaB':287}
        self.assertEqual(resolve_area_seeds(buildings,42,explicit),explicit)
        self.assertEqual(explicit,{'AreaA':901,'AreaB':287})
        for invalid in ({'AreaA':901},{'AreaA':901,'AreaB':287,'Extra':5}):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):resolve_area_seeds(buildings,42,invalid)

    def test_explicit_map_never_falls_back_for_unlisted_area(self):
        pools=SharedPools(fixture_hmi(),AGES,3,42,area_seeds={'A':287})
        self.assertEqual(pools.area_seed('A'),287)
        with self.assertRaises((ValueError,KeyError)):pools.area_seed('B')
        fallback=SharedPools(fixture_hmi(),AGES,3,42)
        self.assertEqual(fallback.area_seed('A'),42)
        self.assertEqual(fallback.area_seed('B'),42)

    def test_all_50_refresh_blocks_match_original_pool_seed_reset(self):
        pools=SharedPools(fixture_hmi(),AGES,50,42,area_seeds={'A':287})
        second_prototype=('Unknown','Metal','flat')  # Roof route is absent from the toy HMI.
        for block in range(50):
            expected=oracle_pool(287,block,50)
            for age,prototype in [('old',PROTOTYPE),('new',PROTOTYPE),('old',second_prototype)]:
                with self.subTest(block=block,age=age,prototype=prototype):
                    actual=pools.get('A',age,prototype,block)
                    self.assertEqual([signature(tree) for tree in actual],expected)
                    self.assertIs(actual,pools.get('A',age,prototype,block))

    def test_slot_sequence_resets_for_each_building_and_refresh_block(self):
        probabilities=np.array([[.65,.35],[.20,.80]])
        pools=SharedPools(fixture_hmi(),AGES,50,42,area_seeds={'A':287})
        expected_age,candidate=oracle_slots(287,probabilities[0],20,50)
        for block in range(50):
            expected_pool=oracle_pool(287,block,50)
            expected_signatures=[expected_pool[i] for i in candidate]
            for building_call in range(2):
                with self.subTest(block=block,building_call=building_call):
                    trees,age_indices=sample_material_block(pools,AGES,probabilities,'old',
                                                           'A',PROTOTYPE,block,20)
                    np.testing.assert_array_equal(age_indices,expected_age)
                    self.assertEqual([signature(tree) for tree in trees],expected_signatures)
            # A returned tree can be scaled/mutated without changing the shared pool.
            original=deepcopy(pools.get('A','old',PROTOTYPE,block)[0].to_dict())
            trees[0].children.clear()
            self.assertEqual(pools.get('A','old',PROTOTYPE,block)[0].to_dict(),original)

    def test_only_available_ages_are_requested(self):
        pools=SharedPools(fixture_hmi(),AGES,3,42,area_seeds={'A':287})
        trees,indices=sample_material_block(pools,AGES,np.array([[1.,0.],[0.,1.]]),
                                           'old','A',PROTOTYPE,0,20)
        np.testing.assert_array_equal(indices,np.zeros(20,dtype=int))
        self.assertTrue(trees)
        self.assertEqual({key[1] for key in pools.pools}, {'old'})

    def test_other_area_uses_its_explicit_seed(self):
        pools=SharedPools(fixture_hmi(),AGES,5,42,area_seeds={'A':287,'B':288})
        probabilities=np.array([[.65,.35],[.20,.80]])
        trees,indices=sample_material_block(pools,AGES,probabilities,'new','B',PROTOTYPE,4,20)
        expected_ages,choices=oracle_slots(288,probabilities[1],20,5)
        expected_pool=oracle_pool(288,4,5)
        np.testing.assert_array_equal(indices,expected_ages)
        self.assertEqual([signature(tree) for tree in trees],[expected_pool[i] for i in choices])

    def test_simulate_preserves_area_order_and_fixed_wall_stream_43(self):
        buildings,training,hmi,matrix=read_inputs(HERE/'data')
        buildings=buildings.iloc[:2].copy()
        buildings['building_id']=['z_first','a_second']
        buildings['area_id']=['AreaB','AreaA']
        result=simulate(buildings,training,hmi,matrix,draws=12,pool_size=2,refreshes=3,seed=123)
        self.assertEqual(result.sampling_schedule['version'],'original-v4')
        self.assertEqual(result.sampling_schedule['area_seeds'],{'AreaB':123,'AreaA':124})
        self.assertEqual(result.sampling_schedule['wall_seed'],43)
        expected=norm.ppf(.05+.90*np.random.default_rng(43).random(12))
        for _,data in result.draw_metadata.groupby('building_id'):
            np.testing.assert_array_equal(data.sort_values('draw').shared_wall_z,expected)


if __name__=='__main__':unittest.main()
