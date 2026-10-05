"""Small orchestration layer around the research model's sampling primitives."""
from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from io import StringIO
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from .age import generate_slots_for_age_material
from .aggregation import aggregate_draws, summarise_targets, LAYERS, MATERIALS
from .dimensions import apply_dimensions
from .dsds import HouseMCModel
from .materials import MaterialTreeBuilder, TreeTrimmer, UncertaintyForestBuilder

HIERARCHY = ['Layer', 'Function', 'Sub-Function', 'Technology', 'Specification', 'Material']


def stream_seed(seed: int, *labels) -> int:
    """Stable named streams; Python's process-randomised hash() is not used."""
    payload = json.dumps([int(seed), *labels], ensure_ascii=True, separators=(',', ':'))
    return int.from_bytes(sha256(payload.encode()).digest()[:8], 'little')


def read_inputs(directory: Path):
    buildings = pd.read_csv(directory / 'buildings.csv', dtype={'building_id':str,'area_id':str})
    training = pd.read_csv(directory / 'training.csv')
    hmi = pd.read_csv(directory / 'hmi.csv')
    matrix = pd.read_csv(directory / 'age_probabilities.csv', index_col=0)
    ages = list(matrix.index)
    if not ages or len(ages) != len(set(ages)) or list(matrix.columns) != ages:
        raise ValueError('Age probability rows and columns must have identical unique labels')
    p = matrix.to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=0), 1, rtol=0, atol=1e-12):
        raise ValueError('Age probabilities must be nonnegative, finite and COLUMN-normalised')
    required = ['building_id', 'area_id', 'age', 'footprint_m2', 'perimeter_m', 'storeys',
                'height_m', 'roof_area_m2', 'wall_material', 'roof_material', 'roof_shape']
    if not set(required) <= set(buildings) or buildings[required].isna().any().any():
        raise ValueError('Incomplete synthetic building attributes')
    if buildings.empty or not buildings.building_id.is_unique:
        raise ValueError('Buildings must have unique IDs and at least one record')
    for c in ['building_id', 'area_id']:
        if not buildings[c].astype(str).str.fullmatch(r'[A-Za-z0-9_-]+').all():
            raise ValueError(f'{c} must contain only letters, numbers, underscores or hyphens')
    numeric = buildings[['footprint_m2','perimeter_m','storeys','height_m','roof_area_m2']].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric <= 0).any() or not (buildings.storeys % 1 == 0).all():
        raise ValueError('Building dimensions must be positive; storeys must be integers')
    if not buildings.age.isin(ages).all() or not buildings.roof_shape.isin(['flat','pitched']).all():
        raise ValueError('Unknown building age or roof shape')
    if not set([*HIERARCHY,'MI',*ages]) <= set(hmi) or hmi[HIERARCHY].isna().any().any():
        raise ValueError('HMI must include all six path levels, MI and the age flags')
    if hmi[HIERARCHY].astype(str).apply(lambda c: c.str.contains('///', regex=False)).any().any():
        raise ValueError('HMI labels cannot contain the path delimiter ///')
    if hmi.duplicated(HIERARCHY).any():
        raise ValueError('HMI has duplicate paths; give distinct specifications separate labels')
    coefficients = pd.to_numeric(hmi.MI, errors='raise').to_numpy(dtype=float)
    if not np.isfinite(coefficients).all() or (coefficients < 0).any():
        raise ValueError('Synthetic MI coefficients must be numeric, finite and nonnegative')
    for age in ages:
        if not hmi[age].isin(['YES','NO']).all():
            raise ValueError('Age flags must be YES or NO')
        if set(hmi.loc[hmi[age].eq('YES'),'Layer'].str.lower()) != set(LAYERS):
            raise ValueError(f'All four layers must have feasible entries for age {age}')
    return buildings, training, hmi, matrix


def leaf_masses(tree):
    """Read the mass element of [MI, dimension, mass] once per material leaf."""
    result = {}

    def visit(node, path):
        if node.mi is not None:
            if len(path) != 6 or not isinstance(node.mi, list) or len(node.mi) != 3:
                raise ValueError('Expected a scaled six-level material leaf')
            mass = float(node.mi[2])
            if not np.isfinite(mass) or mass < 0:
                raise ValueError('Synthetic material mass is not finite and nonnegative')
            result['///'.join(path)] = mass
        else:
            for child in node.children.values():
                visit(child, [*path, str(child.name)])

    visit(tree, [])
    return result


class SharedPools:
    """One prototype pool per area, sampled age, exterior prototype and block."""
    def __init__(self, hmi, ages, pool_size, seed):
        self.builder = MaterialTreeBuilder(hmi, ages)
        self.sampler = UncertaintyForestBuilder(ages, None)
        self.trimmer = TreeTrimmer()
        self.pool_size, self.seed = pool_size, seed
        self.templates, self.pools, self.uses = {}, {}, {}

    def get(self, area, age, prototype, block):
        key = (area, age, *prototype, int(block))
        if key not in self.pools:
            base_key = (age, *prototype)
            if base_key not in self.templates:
                tree = self.builder.build_tree_for_age_band(age)
                info = dict(zip(['wall_material','roof_material','roof_shape'], prototype))
                # The source trimmer prints each deletion; keep CLI output compact.
                with redirect_stdout(StringIO()):
                    self.trimmer.run_trimmer(tree, info)
                self.templates[base_key] = tree
            self.pools[key] = self.sampler.run_one_sample(
                self.templates[base_key], self.pool_size,
                random_state=stream_seed(self.seed, 'pool', *key))
            self.uses[key] = 0
        self.uses[key] += 1
        return self.pools[key]

    def inventory(self):
        rows = []
        for key in sorted(self.pools):
            area, age, wall, roof, shape, block = key
            serial = [t.to_dict() for t in self.pools[key]]
            rows.append(dict(area_id=area, sampled_age=age, wall_material=wall,
                roof_material=roof, roof_shape=shape, refresh_block=block,
                pool_size=self.pool_size, building_requests=self.uses[key],
                candidate_sha256=sha256(json.dumps(serial,sort_keys=True,default=float).encode()).hexdigest()))
        return pd.DataFrame(rows)


@dataclass
class SimulationResult:
    building_paths: dict
    area_paths: dict
    city_paths: pd.DataFrame
    draw_metadata: pd.DataFrame
    pool_inventory: pd.DataFrame
    checks: dict


def simulate(buildings, training, hmi, matrix, *, draws=200, pool_size=5, refreshes=10, seed=42, progress=False):
    if min(draws, pool_size, refreshes) < 1 or draws % refreshes:
        raise ValueError('Positive parameters required, with draws divisible by refreshes (K % C == 0)')
    if seed < 0:
        raise ValueError('seed must be nonnegative')
    ages = list(matrix.index)
    probabilities_by_observed_age = matrix.to_numpy(dtype=float).T
    dsds = HouseMCModel(valid_ratio=.2, random_state=42).fit(training)
    pools = SharedPools(hmi, ages, pool_size, seed)
    block_size = draws // refreshes
    buildings = buildings.sort_values('building_id').reset_index(drop=True)
    building_frames, diagnostics = {}, []
    wall_seed = stream_seed(seed, 'shared-wall')
    wall_z = norm.ppf(.05 + .90 * np.random.default_rng(wall_seed).random(draws))
    for i, b in enumerate(buildings.itertuples(index=False)):
        exterior = pd.DataFrame([dict(T=b.footprint_m2*b.storeys,P=b.perimeter_m,F=b.storeys)])
        interior = dsds.sample_sequential(exterior, n_samples=draws, alpha=.05, w_floor=int(b.storeys),
            rng=np.random.default_rng(stream_seed(seed,'counts',b.building_id)),
            wall_rng=np.random.default_rng(wall_seed))
        # Retain the v4 model's hard bounds after conditional sampling.
        limits = {'R':(3,38),'W':(2,42),'D':(1,32),'L':(3,200)}
        values = {k:np.clip(v[0],*limits[k]) for k,v in interior.items()}
        if any(not np.isfinite(v).all() for v in values.values()):
            raise ValueError('DSDS produced non-finite values on synthetic inputs')
        geometry = dict(area=float(b.footprint_m2),perimeter=float(b.perimeter_m),
            floor=int(b.storeys),wall_height=float(b.height_m),roof_area=float(b.roof_area_m2))
        prototype = (b.wall_material,b.roof_material,b.roof_shape)
        mass_rows = []
        for block in range(refreshes):
            tree_matrix = [[pools.get(b.area_id,age,prototype,block)] for age in ages]
            # This original primitive draws ages, selects candidates, and deep-copies them.
            trees, sampled_ages = generate_slots_for_age_material(
                block_size, ages, [], probabilities_by_observed_age, tree_matrix,
                stream_seed(seed,'slots',b.building_id,block), b.age, 'lazy')
            for offset, (tree, age_index) in enumerate(zip(trees,sampled_ages)):
                k = block*block_size+offset
                inferred = dict(room=int(values['R'][k]),window=int(values['W'][k]),
                    interior_door=int(values['D'][k]),interior_wall=float(values['L'][k]))
                apply_dimensions(tree,geometry,inferred,geometry['area'],floor_count=geometry['floor'])
                mass_rows.append(leaf_masses(tree))
                diagnostics.append(dict(building_id=b.building_id,area_id=b.area_id,draw=k,
                    observed_age=b.age,sampled_age=ages[age_index],refresh_block=block,
                    rooms=inferred['room'],windows=inferred['window'],doors=inferred['interior_door'],
                    internal_wall_length_m=inferred['interior_wall'],shared_wall_z=wall_z[k]))
        frame = pd.DataFrame(mass_rows).fillna(0.)
        frame.index = pd.RangeIndex(draws,name='draw')
        # Sparse saved schemas deliberately exercise the corrected aggregator.
        building_frames[b.building_id] = frame.loc[:,frame.ne(0).any(axis=0)]
        if progress and ((i+1)%4==0 or i+1==len(buildings)):
            print(f'Simulated {i+1}/{len(buildings)} synthetic buildings',flush=True)
    areas, city = aggregate_draws(building_frames,dict(zip(buildings.building_id,buildings.area_id)))
    targets = summarise_targets(city)
    building_sum = np.stack([x.sum(axis=1).to_numpy() for x in building_frames.values()]).sum(axis=0)
    area_sum = np.stack([x.sum(axis=1).to_numpy() for x in areas.values()]).sum(axis=0)
    errors = {
        'building_to_city_max_error_kg':float(np.max(np.abs(building_sum-targets.total))),
        'area_to_city_max_error_kg':float(np.max(np.abs(area_sum-targets.total))),
        'layers_to_total_max_error_kg':float(np.max(np.abs(targets[[f'layer__{l}' for l in LAYERS]].sum(axis=1)-targets.total))),
        'materials_to_total_max_error_kg':float(np.max(np.abs(targets[[f'material__{m}' for m in MATERIALS]].sum(axis=1)-targets.total))),
    }
    tolerance = 1e-10*max(1.,float(targets.total.max()))
    if any(e>tolerance for e in errors.values()):
        raise RuntimeError(f'Mass conservation failed: {errors}')
    checks = {'passed':True,'buildings':len(buildings),'areas':len(areas),'draws':draws,
              'mass_conservation':errors,'absolute_tolerance_kg':tolerance,
              'finite_nonnegative_masses':True,'units':'kg'}
    return SimulationResult(building_frames,areas,city,pd.DataFrame(diagnostics),pools.inventory(),checks)


def target_summaries(result):
    rows=[]
    for scope, frames in [('building',result.building_paths),('area',result.area_paths),('city',{'synthetic_city':result.city_paths})]:
        for unit,frame in frames.items():
            targets=summarise_targets(frame)
            for col in targets:
                x=targets[col].to_numpy()
                p05,median,p95=np.quantile(x,[.05,.5,.95])
                rows.append(dict(scope=scope,unit=unit,target=col,mean_kg=x.mean(),p05_kg=p05,
                    median_kg=median,p95_kg=p95))
    return pd.DataFrame(rows)
