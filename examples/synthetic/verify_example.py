"""Check saved draws and compare the default run with the supplied reference."""
from pathlib import Path
import argparse
import csv
import gzip
import json

import numpy as np
import pandas as pd

from mwe_core.aggregation import LAYERS, MATERIALS, summarise_targets

HERE=Path(__file__).resolve().parent
TARGET_COLUMNS=['total',*[f'layer__{x}' for x in LAYERS],*[f'material__{x}' for x in MATERIALS]]


def read_saved_csv(path, **kwargs):
    """Reject duplicate CSV headers before pandas can silently rename them."""
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8-sig',newline='') as stream:
        header=next(csv.reader(stream),[])
    if not header or len(header)!=len(set(header)):
        raise AssertionError(f'Empty or duplicate column headers: {path.name}')
    return pd.read_csv(path,**kwargs)


def check_draw_index(frame,draws,label):
    index=frame.index
    if (not index.is_unique or not pd.api.types.is_integer_dtype(index.dtype)
            or pd.api.types.is_bool_dtype(index.dtype)
            or not np.array_equal(index.to_numpy(),np.arange(draws))):
        raise AssertionError(f'{label}: expected one ordered row for every draw 0..{draws-1}')


def check_target_values(frame,draws,label):
    check_draw_index(frame,draws,label)
    if list(frame.columns)!=TARGET_COLUMNS:
        raise AssertionError(f'{label}: unexpected target columns or column order')
    values=frame.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values<0).any():
        raise AssertionError(f'{label}: non-finite or negative target masses')
    for prefix in ['layer__','material__']:
        np.testing.assert_allclose(frame.filter(like=prefix).sum(axis=1),frame.total,
                                   rtol=1e-12,atol=1e-6,err_msg=f'{label}: {prefix} mass conservation')


def check_file_set(directory,expected_names,regular_files=True):
    if not directory.is_dir():
        raise AssertionError(f'Missing raw-output directory: {directory}')
    actual={path.name for path in directory.iterdir()}
    expected=set(expected_names)
    if actual!=expected:
        raise AssertionError(f'{directory.name} raw files mismatch; missing={sorted(expected-actual)}, '
                             f'extra={sorted(actual-expected)}')
    if regular_files and any(not (directory/name).is_file() for name in expected):
        raise AssertionError(f'Expected regular raw CSV files in {directory}')


def verify_saved_outputs(out):
    """Verify saved raw paths, each assigned area, city and target partitions.

    Returned metadata and summary are also used by the optional default-reference
    comparison in ``main``. No checks rely only on the saved ``checks.json`` flag.
    """
    out=Path(out)
    meta=json.loads((out/'run_metadata.json').read_text())
    checks=json.loads((out/'checks.json').read_text())
    if meta.get('synthetic') is not True or checks.get('passed') is not True:
        raise AssertionError('Run metadata does not identify a successful synthetic run')
    draws=meta['parameters']['draws']
    if isinstance(draws,bool) or not isinstance(draws,int) or draws<1:
        raise AssertionError('Metadata draw count must be a positive integer')
    building=read_saved_csv(out/'building_targets.csv.gz',dtype={'building_id':str,'area_id':str})
    area=read_saved_csv(out/'area_targets.csv.gz',dtype={'area_id':str})
    city=read_saved_csv(out/'city_targets.csv',index_col='draw')
    summary=read_saved_csv(out/'summary.csv')
    if list(building.columns)!=['building_id','area_id','draw',*TARGET_COLUMNS]:
        raise AssertionError('Unexpected building target schema')
    if list(area.columns)!=['area_id','draw',*TARGET_COLUMNS]:
        raise AssertionError('Unexpected area target schema')
    check_target_values(city,draws,'city')
    for frame,columns in [(building,['building_id','area_id']),(area,['area_id'])]:
        for column in columns:
            if frame[column].isna().any() or not frame[column].str.fullmatch(r'[A-Za-z0-9_-]+').all():
                raise AssertionError(f'Invalid saved identifier in {column}')
    building_ids=list(building.building_id.unique())
    area_ids=list(area.area_id.unique())
    if len(building_ids)!=meta['buildings'] or len(area_ids)!=meta['areas']:
        raise AssertionError('Unexpected number of building or area output sequences')
    if not building.groupby('building_id').area_id.nunique().eq(1).all():
        raise AssertionError('A building is assigned to more than one area')
    if set(building.area_id)!=set(area_ids):
        raise AssertionError('Building assignments and area outputs do not cover the same areas')

    check_file_set(out/'draws', ['buildings','areas','city_paths.csv.gz'],regular_files=False)
    building_frames={}
    for bid in building_ids:
        target=building.loc[building.building_id.eq(bid),['draw',*TARGET_COLUMNS]].set_index('draw')
        check_target_values(target,draws,f'building {bid}')
        building_frames[bid]=target
    area_frames={}
    for aid in area_ids:
        target=area.loc[area.area_id.eq(aid),['draw',*TARGET_COLUMNS]].set_index('draw')
        check_target_values(target,draws,f'area {aid}')
        area_frames[aid]=target
        # Compare every target at every draw, using the saved building assignment.
        summed=building.loc[building.area_id.eq(aid)].groupby('draw')[TARGET_COLUMNS].sum()
        np.testing.assert_allclose(summed,target,rtol=1e-12,atol=1e-6,
                                   err_msg=f'Buildings do not sum to their assigned area {aid}')
    for frame,label in [(building,'building'),(area,'area')]:
        np.testing.assert_allclose(frame.groupby('draw')[TARGET_COLUMNS].sum(),city,
                                   rtol=1e-12,atol=1e-6,err_msg=f'{label} targets do not sum to city')

    check_file_set(out/'draws/buildings',[f'{bid}_paths.csv.gz' for bid in building_ids])
    check_file_set(out/'draws/areas',[f'{aid}_paths.csv.gz' for aid in area_ids])
    raw_units=[(out/'draws/buildings'/f'{bid}_paths.csv.gz',target,f'building {bid}')
               for bid,target in building_frames.items()]
    raw_units.extend((out/'draws/areas'/f'{aid}_paths.csv.gz',target,f'area {aid}')
                     for aid,target in area_frames.items())
    raw_units.append((out/'draws/city_paths.csv.gz',city,'city'))
    for path,target,label in raw_units:
        raw=read_saved_csv(path,index_col='draw')
        check_draw_index(raw,draws,f'{label} raw paths')
        # This also validates six-level paths, legal layer names, unique columns,
        # real numeric values, finite masses and nonnegative masses.
        recalculated=summarise_targets(raw)
        np.testing.assert_allclose(recalculated,target,rtol=1e-12,atol=1e-6,
                                   err_msg=f'{label} raw paths disagree with saved targets')
    if not ((summary.p05_kg<=summary.median_kg)&(summary.median_kg<=summary.p95_kg)).all():
        raise AssertionError('Invalid quantile order')
    return meta,summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'outputs')
    args=parser.parse_args()
    out=args.output
    meta,summary=verify_saved_outputs(out)
    reference=HERE/'expected/default_city_summary.csv'
    reference_meta=HERE/'expected/default_run.json'
    default={'draws':200,'pool_size':5,'refreshes':10,'seed':42}
    if reference.exists() and reference_meta.exists() and meta['parameters']==default:
        expected_meta=json.loads(reference_meta.read_text())
        if meta['input_sha256']==expected_meta['input_sha256']:
            expected=pd.read_csv(reference).set_index('target').sort_index()
            observed=summary[summary.scope.eq('city')].set_index('target').sort_index()
            cols=['mean_kg','p05_kg','median_kg','p95_kg']
            np.testing.assert_allclose(observed[cols],expected[cols],rtol=1e-7,atol=.01)
            print('Default synthetic reference results match (floating-point tolerance applied).')
        else:
            print('Reference comparison skipped: input files differ from the supplied fixture.')
    else:
        print('Reference comparison skipped: configuration differs or reference is not installed.')
    print('Saved-output checks passed: exact raw file inventory, raw paths -> targets, '
          'building -> assigned area -> city, complete paired draws, material/layer conservation and quantiles.')


if __name__=='__main__':
    main()
