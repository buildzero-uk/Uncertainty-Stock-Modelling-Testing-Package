"""Saved-output checks catch stale files, misassigned areas and invalid draws."""

from pathlib import Path
from contextlib import redirect_stdout
from copy import deepcopy
import gzip
from io import StringIO
import json
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mwe_core.aggregation import aggregate_draws, summarise_targets
from verify_example import (DEFAULT_PARAMETERS,compare_default_reference,
                            reference_skip_reason,verify_saved_outputs)


def material_path(layer,material):
    return f'{layer}///wall///subfunction///technology///specification///{material}'


class SavedOutputTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out=Path(self.temp.name)
        self.brick=material_path('structure','Brick')
        self.glass=material_path('skin','Glass')
        self.stone=material_path('structure','Stone')
        buildings={
            '001':pd.DataFrame({self.brick:[1.,2.,3.]}),
            '002':pd.DataFrame({self.glass:[4.,5.,6.]}),
            '003':pd.DataFrame({self.stone:[7.,8.,9.]}),
        }
        for frame in buildings.values():frame.index.name='draw'
        assignment={'001':'01','002':'01','003':'02'}
        areas,city=aggregate_draws(buildings,assignment)
        building_targets=[];area_targets=[];summary=[]
        for scope,frames in [('building',buildings),('area',areas),('city',{'synthetic_city':city})]:
            for unit,frame in frames.items():
                targets=summarise_targets(frame)
                if scope=='building':
                    self.write(frame,self.out/'draws/buildings'/f'{unit}_paths.csv.gz',index=True)
                    saved=targets.reset_index();saved.insert(0,'building_id',unit);saved.insert(1,'area_id',assignment[unit])
                    building_targets.append(saved)
                elif scope=='area':
                    self.write(frame,self.out/'draws/areas'/f'{unit}_paths.csv.gz',index=True)
                    saved=targets.reset_index();saved.insert(0,'area_id',unit);area_targets.append(saved)
                else:
                    self.write(frame,self.out/'draws/city_paths.csv.gz',index=True)
                    self.write(targets,self.out/'city_targets.csv',index=True)
                for target in targets:
                    x=targets[target].to_numpy();lo,mid,hi=np.quantile(x,[.05,.5,.95])
                    summary.append(dict(scope=scope,unit=unit,target=target,mean_kg=x.mean(),p05_kg=lo,median_kg=mid,p95_kg=hi))
        self.write(pd.concat(building_targets,ignore_index=True),self.out/'building_targets.csv.gz')
        self.write(pd.concat(area_targets,ignore_index=True),self.out/'area_targets.csv.gz')
        self.write(pd.DataFrame(summary),self.out/'summary.csv')
        metadata=dict(synthetic=True,parameters=dict(draws=3,pool_size=2,refreshes=1,seed=42),buildings=3,areas=2,input_sha256={})
        (self.out/'run_metadata.json').write_text(json.dumps(metadata))
        (self.out/'checks.json').write_text(json.dumps(dict(passed=True)))

    @staticmethod
    def write(frame,path,index=False):
        path.parent.mkdir(parents=True,exist_ok=True)
        frame.to_csv(path,index=index,compression='infer')

    def test_valid_complete_output_including_numeric_string_ids(self):
        metadata,_=verify_saved_outputs(self.out)
        self.assertEqual(metadata['buildings'],3)

    def test_wrong_area_assignment_is_rejected_even_when_city_sums_match(self):
        path=self.out/'building_targets.csv.gz'
        d=pd.read_csv(path,dtype={'building_id':str,'area_id':str})
        d.loc[d.building_id.eq('002'),'area_id']='02'
        self.write(d,path)
        with self.assertRaisesRegex(AssertionError,'assigned area'):
            verify_saved_outputs(self.out)

    def test_missing_or_extra_raw_building_file_is_rejected(self):
        original=self.out/'draws/buildings/001_paths.csv.gz'
        extra=self.out/'draws/buildings/stale_paths.csv.gz'
        shutil.copy2(original,extra)
        with self.assertRaisesRegex(AssertionError,'extra='):
            verify_saved_outputs(self.out)
        extra.unlink();original.unlink()
        with self.assertRaisesRegex(AssertionError,'missing='):
            verify_saved_outputs(self.out)

    def test_extra_raw_area_file_is_rejected(self):
        shutil.copy2(self.out/'draws/areas/01_paths.csv.gz',self.out/'draws/areas/stale_paths.csv.gz')
        with self.assertRaisesRegex(AssertionError,'extra='):
            verify_saved_outputs(self.out)

    def test_duplicate_saved_target_draw_is_rejected(self):
        path=self.out/'building_targets.csv.gz'
        d=pd.read_csv(path,dtype={'building_id':str,'area_id':str})
        d.loc[d.building_id.eq('001')&d.draw.eq(1),'draw']=0
        self.write(d,path)
        with self.assertRaisesRegex(AssertionError,'ordered row'):
            verify_saved_outputs(self.out)

    def test_raw_material_values_must_match_targets_not_only_total(self):
        path=self.out/'draws/city_paths.csv.gz'
        d=pd.read_csv(path,index_col='draw')
        d[self.brick]+=1;d[self.glass]-=1
        self.write(d,path,index=True)
        with self.assertRaisesRegex(AssertionError,'raw paths disagree'):
            verify_saved_outputs(self.out)

    def test_negative_nonfinite_and_invalid_path_raw_data_are_rejected(self):
        path=self.out/'draws/buildings/001_paths.csv.gz'
        original=pd.read_csv(path,index_col='draw')
        for value in [-1.,np.inf]:
            with self.subTest(value=value):
                bad=original.copy();bad.iloc[0,0]=value
                self.write(bad,path,index=True)
                with self.assertRaises(ValueError):verify_saved_outputs(self.out)
        self.write(original.rename(columns={self.brick:'invalid path'}),path,index=True)
        with self.assertRaisesRegex(ValueError,'six non-empty'):
            verify_saved_outputs(self.out)

    def test_duplicate_raw_headers_are_not_silently_renamed(self):
        path=self.out/'draws/buildings/001_paths.csv.gz'
        with gzip.open(path,'wt') as stream:
            stream.write(f'draw,{self.brick},{self.brick}\n0,1,0\n1,2,0\n2,3,0\n')
        with self.assertRaisesRegex(AssertionError,'duplicate column'):
            verify_saved_outputs(self.out)

    def test_raw_draw_order_is_not_silently_sorted(self):
        path=self.out/'draws/areas/01_paths.csv.gz'
        d=pd.read_csv(path,index_col='draw').iloc[[1,0,2]]
        self.write(d,path,index=True)
        with self.assertRaisesRegex(AssertionError,'ordered row'):
            verify_saved_outputs(self.out)


class ReferenceScheduleTests(unittest.TestCase):
    def setUp(self):
        self.metadata={
            'parameters':dict(DEFAULT_PARAMETERS),'input_sha256':{'fixture.csv':'fixture-hash'},
            'sampling_schedule':{
                'version':'original-v4','area_seeds':{'A':42,'B':43},'wall_seed':43,
                'pool_seed_rule':'area_seed + refresh_block; shared across ages and exterior prototypes',
                'slot_seed_rule':'area_seed; reset for every building and refresh block',
                'count_stream':'SHA256(base_seed, counts, building_id); independent reproducible count draws',
            },
        }

    def test_full_matching_schedule_is_eligible(self):
        self.assertIsNone(reference_skip_reason(self.metadata,deepcopy(self.metadata)))

    def test_each_changed_schedule_field_prevents_reference_comparison(self):
        changes={'version':'hashed-v1','area_seeds':{'A':287,'B':288},'wall_seed':99,
                 'pool_seed_rule':'independent age pools','slot_seed_rule':'independent building slots',
                 'count_stream':'different counts'}
        for key,value in changes.items():
            with self.subTest(key=key):
                run=deepcopy(self.metadata);run['sampling_schedule'][key]=value
                self.assertIsNotNone(reference_skip_reason(run,self.metadata))

    def test_older_or_incomplete_schedule_metadata_is_not_assumed_compatible(self):
        old=deepcopy(self.metadata);old.pop('sampling_schedule')
        self.assertIn('missing',reference_skip_reason(self.metadata,old))
        self.assertIn('missing',reference_skip_reason(old,self.metadata))
        incomplete=deepcopy(self.metadata);incomplete['sampling_schedule'].pop('wall_seed')
        self.assertIn('incomplete',reference_skip_reason(incomplete,self.metadata))

    def test_changed_parameters_or_input_hashes_prevent_comparison(self):
        run=deepcopy(self.metadata);run['parameters']['seed']=99
        self.assertIn('configuration',reference_skip_reason(run,self.metadata))
        run=deepcopy(self.metadata);run['input_sha256']['fixture.csv']='different-hash'
        self.assertIn('input hashes',reference_skip_reason(run,self.metadata))

    def test_schedule_mismatch_skips_numbers_but_matching_schedule_checks_them(self):
        expected=pd.DataFrame([dict(target='total',mean_kg=100.,p05_kg=90.,median_kg=100.,p95_kg=110.)])
        observed=expected.assign(scope='city')
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            expected.to_csv(directory/'default_city_summary.csv',index=False)
            (directory/'default_run.json').write_text(json.dumps(self.metadata))
            wrong_numbers=observed.copy();wrong_numbers['mean_kg']=1.e9
            wrong_schedule=deepcopy(self.metadata);wrong_schedule['sampling_schedule']['wall_seed']=44
            output=StringIO()
            with redirect_stdout(output):
                self.assertFalse(compare_default_reference(wrong_schedule,wrong_numbers,directory))
            self.assertIn('sampling_schedule differs',output.getvalue())
            self.assertNotIn('reference results match',output.getvalue())
            with self.assertRaises(AssertionError):
                compare_default_reference(self.metadata,wrong_numbers,directory)
            with redirect_stdout(StringIO()):
                self.assertTrue(compare_default_reference(self.metadata,observed,directory))


if __name__=='__main__':
    unittest.main()
