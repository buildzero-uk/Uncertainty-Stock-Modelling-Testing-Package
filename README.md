# Uncertainty-Aware Building Material Stock Modelling

## About this project

This repository contains the research code for a probabilistic framework for estimating building material stocks and their uncertainties.

The framework combines uncertainty in building dimensions, construction technologies and building ages through Monte Carlo sampling. It produces material stock distributions at building, neighbourhood and city scales, with results reported by material category and building layer.

The accompanying study applies the framework to residential buildings in Bristol, United Kingdom.

## Synthetic test example

The `examples/` directory contains a runnable test example using entirely synthetic data.

The example demonstrates the main calculation workflow, including conditional building-dimension sampling, age and material-prototype sampling, component-mass calculation and aggregation. It includes installation instructions, reference outputs and automated tests.

See [the example README](examples/synthetic/README.md) for instructions.

The synthetic inputs and outputs are provided for software testing and demonstration. They do not represent real buildings or reproduce the Bristol stock estimates.

## Data availability

The real data used in the study are not included in the current release and will be uploaded in a future update.

Until then, the synthetic example allows users to run and test the workflow without access to the original datasets.
