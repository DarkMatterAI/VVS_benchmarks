# 6. VVS Benchmarks

**Prerequisites:** [Train Models](02_training.md),
[Set Up Scoring Endpoints](04_scoring_endpoints.md)

This section evaluates the VVS algorithm.

## Setup

Several retrieval indices are pre-computed for use in downstream benchmarks:

```bash
cd src/vvs_local
./run.sh generate_indices
```

## Embedding size latency

Evaluates retrieval latency at different embedding sizes:

```bash
cd src/vvs_local
./run.sh compression_eval
```

## Learning rate sweeps

Evaluates VVS sensitivity across a wide range of learning rates. Sweeps are run in
the enumerated space using KNN retrieval (`src/vvs_local/lr_sweep_knn.yaml`) and in
the decomposed space using BBKNN (`src/vvs_local/lr_sweep_bbknn.yaml`):

```bash
cd src/vvs_local
./run.sh lr_sweep --cfg lr_sweep_bbknn.yaml
./run.sh lr_sweep --cfg lr_sweep_knn.yaml
```

## VVS hyperparameter sweeps

Performs hyperparameter sweeps for VVS (`src/vvs_local/hyperparam_configs/bbknn_sweep.yaml`):

```bash
cd src/vvs_local
./run.sh hyperparam_sweep --cfg bbknn_sweep.yaml --run_type sweep
```

> **Note:** These sweeps also serve as the VVS hyperparameter sweep for the method
> comparisons in [Comparison Benchmarks](07_comparisons.md).

## Data analysis

Generates plots of embedding compression retrieval performance, VVS hyperparameter
boxplots, VVS learning rate sweeps, and the VVS repeat-SMILES histogram:

```bash
cd src/data_analysis
./src/vvs_local/run.sh sweep
./src/vvs_local/run.sh embed
```