# S-PStorm
Source code for the work *A Variance-Reduced and Stabilized Proximal Stochastic Gradient Method with Support Identification Guarantees for Structured Optimization accepted* by **AISTATS2023**.

This repo contains implementations of a collection of stochastic first-order methods, including `ProxSVRG`, `SAGA`, `RDA`, `PStorm`, and `S-PStorm`. The repo contains two main directories, `src` and `test`. `src/solvers` contain the source code for all algorithm implementations. `test` directory contains the scripts necessary to reproduce the results reported in the paper.  

# Usgae of the code

## Data Preparation
Navigate to the `test/data_prep` directory and do the following steps.

1. Run `bash download.sh` to download 10 datasets.
2. Run `bash process.sh` to perform data preprocessing.
3. Run `python compute_Lip` to get the estimate of the Lipschitz constants.

## Perform tests

1. Navigate to the directory: `cd test/bash`.
2. Generate the bash scripts: `python create_bash.py`
3. Run the command `bash submit` and experiments will run in the background. The logs on each run can be found at `test/bash/log` and the results will be saved at `test/experiments`



