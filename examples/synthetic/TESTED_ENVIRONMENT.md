# Tested environment and verification

Verified on 5 October 2026 using CPython 3.11.5 on macOS / Apple Silicon.

## Execution checks

- Created a new virtual environment without system-site packages.
- Installed the five direct dependencies from public PyPI; the complete resolved
  dependency set is recorded in `requirements-lock.txt`. `pip check` passed.
- Copied the example directory outside the research repository, excluding
  generated outputs and Python caches.
- Invoked the relocated runner from a different working directory, using the
  new virtual environment and only the included synthetic inputs.
- Completed the default run: 24 buildings, two areas, 200 draws, `P=5`, `C=10`,
  seed 42. Simulation and reporting took approximately 32 seconds in that run;
  dependency installation and initial library imports are additional.
- Passed **31 automated tests**, including the sparse-column aggregation
  regression and independent checks against hand-calculated component masses.
- Verified raw-output inventory, paired draws, building-to-assigned-area-to-city
  sums, layer/material conservation and default numerical reference summaries.
- Compared all saved target sequences, summaries, sampled-attribute records and
  pool fingerprints with the development-environment run; they matched within
  floating-point tolerance.
- Confirmed the runner rejects a nonempty output directory, preventing stale
  files from a previous configuration from being mixed with new results.
- Rendered the example PDF and inspected both the PDF rendering and PNG output.

The copied material-tree primitives, age-slot sampler and component-dimension
conversion were checked against their original Python syntax trees. The DSDS
random-stream adaptations are documented in `CODE_ORIGIN.md`.

These checks concern the supplied synthetic example. Other operating systems
and Python versions have not been independently tested. The pinned packages
target Python 3.11; do not assume compatibility with every newer Python release.

## Interpreting the checks

Mass conservation is checked within floating-point tolerance rather than exact
decimal equality. The default-reference comparison uses relative tolerance
`1e-7` and absolute tolerance `0.01 kg`; it permits minor numerical differences
between numerical-library builds.

Synthetic results are a repeatable software baseline. Their magnitude,
uncertainty ranges and correlations do not constitute independent evidence
about real buildings. A different input inventory or a different simulation
seed can legitimately produce different results.
