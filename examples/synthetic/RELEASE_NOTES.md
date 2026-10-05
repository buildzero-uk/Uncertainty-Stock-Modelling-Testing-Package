# Original-schedule revision - 5 October 2026

Material sampling follows the original research scheduling convention:
area-seeded candidate pools refresh by block, while area-seeded age and
candidate-index slots restart for every building and block. This replaces
the earlier per-building material-slot and per-key pool seeds.

The portable default derives area seeds from first appearance in the input.
`--area-seeds` accepts an explicit complete mapping; `--wall-seed` exposes the
shared wall seed, defaulting to the original value of 43.

Counts retain explicit independent building streams for reproducibility.
Historical unseeded counts cannot be recovered from an area seed alone;
matching the material schedule is distinct from reproducing every historical
random output.

The synthetic inputs, model forms, dimension formulas, complete-column
aggregation and Figure 3 material rules are unchanged. Strict public-input
validation remains in place. Expected summaries and the figure are regenerated
for the revised schedule. All included data remain synthetic.
