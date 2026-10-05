# Tested environment and verification

This record applies to the **original-schedule revision**, dated 5 October
2026. Earlier timings, test counts and baselines do not certify this revision.

## Environment and execution record

| Item | Verified result |
| --- | --- |
| Python version and platform | Python 3.11.5; macOS 26.6.2; ARM64 |
| Isolated environment and dependency check | Existing isolated venv with the supplied dependency pins; `pip check` passed |
| Default synthetic run | Completed: 24 buildings, two areas, 200 draws, `P=5`, `C=10`; original material schedule; wall seed 43 |
| Automated tests | All 43 tests passed in 2.62 seconds |
| Relocated standalone run | Completed from a separate temporary directory, invoked from outside the example directory; no production modules or data needed |
| Saved-output and new default-reference verification | Passed for both runs, including parameters, input hashes and complete sampling schedule |
| Figure render inspection | PNG inspected; PDF rendered and inspected; labels, units and intervals legible |
| Runtime, excluding installation/setup where applicable | Default run 26.8 seconds; relocated run 33.3 seconds, with other local work running concurrently |

The supplied pins target Python 3.11. `requirements.txt` lists the five direct
dependencies; `requirements-lock.txt` records the complete pinned set. Other
operating systems and Python versions are not implied to be tested.

## Checks for this revision

The default configuration uses 24 buildings, two areas, 200 draws, `P=5`,
`C=10`, base seed 42 and shared wall seed 43. Verification covers:

- Hand-calculated component masses and retention of all materials in a
  selected specification.
- Age-matrix orientation, deterministic area seeds, shared age/candidate slots,
  pool refreshes and the explicit wall RNG convention.
- Separate reproducible count streams and repeatability for fixed inputs,
  seeds and software versions.
- Paired building-to-area-to-city sums and complete-column aggregation when
  later buildings introduce new materials.
- Separate conservation of layer and material partitions.
- Saved-output integrity and the newly generated original-schedule reference.
  The previous schedule's expected values are superseded.
- Rejection of unsupported inputs and nonempty output directories.

## Interpretation

The reference comparison permits numerical-library differences: relative
tolerance `1e-7` and absolute tolerance `0.01 kg`. Conservation is also checked
within floating-point tolerance.

Synthetic reference values are software regression baselines. Code agreement
or replay of saved model quantities does not establish predictive accuracy
against measured buildings. A historical material seed cannot recover the
original unseeded count draws. Fresh explicit count seeds can preserve the
implemented model and material schedule while producing different sampled
outputs.

Real-data replay and any required legacy adapters are maintained outside this
public example. No real input data or recovered model outputs are included.

A separate local check of the revised sampling helpers on retained real
inputs is summarised in [REAL_DATA_CHECK.md](REAL_DATA_CHECK.md).
