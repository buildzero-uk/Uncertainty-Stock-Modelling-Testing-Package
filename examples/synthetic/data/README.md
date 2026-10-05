# Synthetic inputs

Every record, coefficient and probability in this directory is synthetic. No
building records, surveyed training observations or numerical HMI coefficients
from the study are redistributed here. These fixtures test software behaviour;
they do not reproduce or validate Bristol estimates.

Regenerate them with `python generate_data.py`. The default generation seed is
`20261005`. Identifiers and areas are fictional and contain no coordinates.

| File | Meaning |
| --- | --- |
| `buildings.csv` | 24 houses in two fictional areas; footprint (m2), perimeter (m), storeys, total wall height (m), roof area (m2), recorded age and external material/roof labels. |
| `training.csv` | 300 generated observations: total floor area `T` (m2), footprint perimeter `P` (m), storeys `F`, rooms `R`, windows `W`, internal doors `D`, internal-wall length `L` (m). Counts follow illustrative conditional Poisson mechanisms and wall lengths a noisy lognormal mechanism. |
| `age_probabilities.csv` | Rows = sampled age, columns = recorded age. Each column sums to one; select a **column** to sample an age conditional on the recorded label. Probabilities are illustrative assumptions. |
| `hmi.csv` | A six-level hierarchy: Layer / Function / Sub-Function / Technology / Specification / Material. `MI` is a component intensity, with its denominator in `unit`. Age columns contain `YES` or `NO`. |

The geometry is deliberately simple: total floor area is footprint multiplied
by storeys; total wall height is 2.7 m multiplied by storeys; pitched roof area
is 1.18 times the footprint. This is example geometry, not a survey model.

HMI coefficients use kg/m2 for walls/floors/roofs, kg/item for openings/fans and
kg/m for wiring. The material sampler chooses technologies/specifications;
**all materials within a chosen specification are retained together**.
Some specifications are available only in selected synthetic age bands.
External wall and roof labels use the study code's vocabulary to exercise its
existing trimming rules. The source trimmer preferentially trims Skin walls
when that node exists; it does not impose the same trimming on Structure walls.

The original Figure 3 grouping is reproduced, including its reporting
conventions: bare `Timber`, chipboard, plywood and slate fall in `Other`, while
`Timber studding`, `Timber structure`, etc. enter Timber. `Other` is the complete
remainder, not a chemically homogeneous or separately recyclable material.
