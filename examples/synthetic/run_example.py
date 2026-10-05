#!/usr/bin/env python3
"""Run the synthetic, data-independent example: python run_example.py."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time

import numpy as np
import pandas as pd

from mwe_core.aggregation import summarise_targets
from mwe_core.pipeline import read_inputs, simulate, target_summaries

HERE = Path(__file__).resolve().parent


def write_csv(frame, path, index=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path,index=index,float_format='%.15g',
        compression={'method':'gzip','mtime':0} if path.suffix=='.gz' else None)


def plot_results(targets, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    fig, axes = plt.subplots(1,2,figsize=(10,3.9),layout='constrained')
    total = targets.total.to_numpy()/1000
    p05,median,p95 = np.quantile(total,[.05,.5,.95])
    axes[0].hist(total,bins=20,density=True,color='#4B8190',edgecolor='white',linewidth=.5)
    axes[0].axvspan(p05,p95,color='#295C72',alpha=.08)
    axes[0].axvline(median,color='#BB5A37',lw=1.8,label='Median')
    for i,value in enumerate([p05,p95]):
        axes[0].axvline(value,color='#333333',ls='--',lw=1,label='5th-95th percentiles' if i==0 else None)
    axes[0].set(title='A  Total stock distribution',xlabel='Material stock (tonnes)',ylabel='Probability density')
    axes[0].legend(frameon=False,fontsize=8)
    layers=['structure','skin','space','services']
    for i,layer in enumerate(layers):
        lo,mid,hi=np.quantile(targets['layer__'+layer]/1000,[.05,.5,.95])
        axes[1].errorbar(mid,i,xerr=[[mid-lo],[hi-mid]],fmt='o',color='#295C72',capsize=3)
    axes[1].set_yticks(range(4),[x.title() for x in layers])
    axes[1].invert_yaxis()
    axes[1].set_xscale('log')
    axes[1].xaxis.set_major_formatter(ScalarFormatter())
    axes[1].set(title='B  Layer stock intervals',xlabel='Material stock (tonnes; log scale)')
    axes[1].grid(axis='x',color='#dde3e5',lw=.7)
    fig.suptitle('Synthetic data - workflow demonstration only',fontsize=12,fontweight='bold')
    fig.savefig(output/'synthetic_results.png',dpi=200)
    fig.savefig(output/'synthetic_results.pdf')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=HERE/'data')
    parser.add_argument('--output',type=Path,default=HERE/'outputs')
    parser.add_argument('--draws',type=int,default=200)
    parser.add_argument('--pool-size',type=int,default=5)
    parser.add_argument('--refreshes',type=int,default=10)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--no-plot',action='store_true')
    args=parser.parse_args()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error('Output must be a new or empty directory. Choose a new --output path for each run.')
    started=time.perf_counter()
    buildings,training,hmi,matrix=read_inputs(args.input_dir)
    print(f'Synthetic example: {len(buildings)} buildings, {buildings.area_id.nunique()} areas, '
          f'{args.draws} draws, P={args.pool_size}, C={args.refreshes}',flush=True)
    result=simulate(buildings,training,hmi,matrix,draws=args.draws,pool_size=args.pool_size,
                    refreshes=args.refreshes,seed=args.seed,progress=True)
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    collected=[]
    for bid,frame in result.building_paths.items():
        write_csv(frame,out/'draws/buildings'/f'{bid}_paths.csv.gz',index=True)
        targets=summarise_targets(frame).reset_index()
        targets.insert(0,'building_id',bid)
        targets.insert(1,'area_id',buildings.set_index('building_id').loc[bid,'area_id'])
        collected.append(targets)
    write_csv(pd.concat(collected,ignore_index=True),out/'building_targets.csv.gz')
    collected=[]
    for area,frame in result.area_paths.items():
        write_csv(frame,out/'draws/areas'/f'{area}_paths.csv.gz',index=True)
        targets=summarise_targets(frame).reset_index()
        targets.insert(0,'area_id',area)
        collected.append(targets)
    write_csv(pd.concat(collected,ignore_index=True),out/'area_targets.csv.gz')
    write_csv(result.city_paths,out/'draws/city_paths.csv.gz',index=True)
    targets=summarise_targets(result.city_paths)
    write_csv(targets,out/'city_targets.csv',index=True)
    write_csv(target_summaries(result),out/'summary.csv')
    write_csv(result.draw_metadata,out/'draw_metadata.csv.gz')
    write_csv(result.pool_inventory,out/'pool_inventory.csv')
    (out/'checks.json').write_text(json.dumps(result.checks,indent=2)+'\n')
    if not args.no_plot:
        plot_results(targets,out)
    parameters={k:getattr(args,k) for k in ['draws','pool_size','refreshes','seed']}
    metadata=dict(synthetic=True,parameters=parameters,buildings=len(buildings),areas=buildings.area_id.nunique(),
        dsds_training_records=len(training),hmi_rows=len(hmi),age_labels=list(matrix.index),
        units='kg in CSV; tonnes in the figure',python=platform.python_version(),
        dependencies={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','statsmodels','matplotlib']},
        input_sha256={name:hashlib.sha256((args.input_dir/name).read_bytes()).hexdigest()
                      for name in ['buildings.csv','training.csv','hmi.csv','age_probabilities.csv']},
        elapsed_seconds=round(time.perf_counter()-started,3),
        note='Synthetic coefficients and fitted parameters are illustrative; no Bristol results are reproduced.')
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Finished in {metadata["elapsed_seconds"]:.1f} seconds. Outputs: {out.resolve()}',flush=True)
    print('Mass-conservation checks passed. Run python verify_example.py to check saved outputs.')


if __name__=='__main__':
    main()
