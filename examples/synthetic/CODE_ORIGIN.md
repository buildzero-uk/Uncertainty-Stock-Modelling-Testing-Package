# Code origin and adaptations

This example includes local copies of the research primitives it needs.
It can run after being copied out of the repository and does not import the
production scripts or read their private data paths. Source file paths below
are relative to the repository root; their SHA-256 fingerprints are recorded
in `provenance.json`.

## Reused primitives

The source directory is `stock_modelling/proof_of_concept_uncertainty/`.

| Example module | Original source | Reused implementation |
| --- | --- | --- |
| `mwe_core/dsds.py` | `building_sampling/house_mc_sampling.py` | `HouseMCModel` fitting, conditional prediction and sequential sampling |
| `mwe_core/materials.py` | `material_sampling/material_sampling_v2.py` | `TreeNode`, `MaterialTreeBuilder`, `TreeTrimmer`; `UncertaintyForestBuilder.__init__`, `_trim_tree_one_sample`, `run_one_sample` |
| `mwe_core/age.py` | `material_sampling/pre_sampler.py` | `generate_slots_for_age_material` |
| `mwe_core/dimensions.py` | `stock_calculation/uncertainty_stock_calculation_v2.py` | `apply_dimensions` |
| `mwe_core/aggregation.py` | `variance_decomposition/processing_files.py` | Terminal-material classification rules from `_merge_material_name`, with all six Figure 3 groups enabled and unmatched names retained as `Other` |

The material-tree definitions and listed sampling methods, the age-slot
function, and `apply_dimensions` were copied without changes to their bodies.
Import-time I/O, private file paths, unused plotting methods and demonstration
entrypoints were excluded. `UncertaintyForestBuilder` contains only the three
methods used by the example; the full class is retained in its original file.

## Explicit adaptations

- **Reproducible DSDS randomness.** The Poisson sampler uses its supplied RNG
  instead of constructing an unseeded generator internally. A separate
  wall-length RNG makes the shared standardised wall disturbance explicit.
  Conditional model forms and truncation are retained; the runner also retains
  the production v4 bounds for rooms, windows, doors and internal-wall length.
  Fitted coefficients come exclusively from the synthetic training data.
- **Numeric dimensions.** The `evaluate_expression` adapter accepts numeric
  values only. It does not evaluate arbitrary formula strings. The component
  scaling function itself is unchanged.
- **Small shared pools.** `pipeline.py` replaces the production file-backed,
  multiprocessing orchestration with an explicit in-memory pool cache.
  Candidate generation uses `area_seed + block` for every age and exterior
  prototype. Slot generation restarts with constant `area_seed` for each
  building and block, sharing age/candidate-index sequences among buildings
  with the same recorded age in an area. Pools remain distinct by area,
  sampled age, exterior prototype and block. Candidates are copied before
  scaling, and draw indices remain paired.
- **Explicit area ordering.** Historical area seeds depended on unsorted
  directory enumeration. The example uses `base_seed + ordinal` with areas
  ordered by first appearance in the input, or a complete `--area-seeds`
  JSON mapping. It records the effective mapping.
- **Original wall seed.** The default shared wall seed is 43, matching the
  original model seed 42 plus one. `--wall-seed` exposes a sensitivity option;
  changing it changes that original default. Count seeds are stable hashes of
  the base seed, `counts` and building ID. They make new draws reproducible
  but cannot recover the research helper's historical unseeded counts.
- **Complete aggregation.** The aggregation fix developed during revision
  uses a complete material schema. This example builds that schema from the
  union of all building paths, including columns absent from the first
  building. Missing columns mean zero; non-finite or negative masses are
  rejected. Raw research-data exceptions require an explicit external
  adapter; public input validation is not relaxed.
  It does not reuse the earlier first-building-column accumulation behaviour.
- **Standalone fixtures and reporting.** The data generator, command-line
  runner, output verifier, reference summaries and tests were added for this
  example. No real Bristol input records or original HMI coefficients are
  included. The synthetic inventory is intentionally small and does not cover
  every branch or construction specification of the full database.

The original research scripts remain separate. Successful execution of this
example establishes that this supplied workflow can run and that the tested
software invariants hold. It does not assert that every historical research
entrypoint has been converted into a portable package, or that synthetic
results validate empirical stock estimates.

## Revision boundary

This revision supersedes the earlier example's hash-per-building material
slot seeds and hash-per-key pool seeds. Those streams demonstrated the same
primitives but did not preserve the original shared sampling schedule.
Expected summaries and the figure must use this revision's regenerated
reference files.

Code agreement checks tested calculations and scheduling, not predictive
accuracy against measured building inventories. A paired local replay can
hold recovered historical dimensions fixed. A fresh run with newly seeded
count draws answers a different reproducibility question and need not be
bitwise identical. Private replay data and legacy adapters are not included
in this synthetic example.
