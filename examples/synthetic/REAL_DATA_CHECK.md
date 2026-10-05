# Real-input computational check

The revised public `SharedPools` and `sample_material_block` functions were
also checked locally on a randomly selected study area containing 351
buildings, using 1,000 draws, 50 candidates per pool and 50 refresh blocks.
All 351 buildings were retained. Original inputs and stored outputs were
kept outside the public example.

Holding recovered historical DSDS quantities fixed, the new material schedule
was compared with 17,550 stored building/block files. All 351,000 sampled-age
assignments matched. Across 75,707,040 material-path mass
comparisons, the maximum absolute difference was
1.16e-10 kg, consistent with floating-point rounding.
The largest building-target difference from the previous original-code replay
was 1.16e-09 kg.

This local check used an external adapter for historical input formatting and
mass-cleaning conventions, including one zero-storey input. It invoked the
actual public material-sampling helpers; it did not run the unmodified
synthetic-only CLI on raw research records. The public CLI keeps its stricter
input validation. The original script bodies were loaded separately as the
reference implementation.

A second local run generated new count draws using explicit independent
streams, with the original material schedule and shared wall seed 43. Such
a fresh run need not reproduce unseeded historical count realisations.
The paired comparison above tests implementation fidelity; it is not a
comparison against measured building inventories or evidence of empirical
predictive accuracy. The 43 public tests and synthetic reference outputs
provide checks that readers can repeat without the private data.
