# Reproduction Guides

These guides provide step-by-step instructions for reproducing all datasets,
models, experiments, and figures from the paper. They are intended to be followed
in order, as later steps depend on artifacts produced by earlier ones.

Note: executing all steps requires several days of runtime 

## Order and dependencies

| Step | Guide | Depends on |
|------|-------|------------|
| 0 | [System Requirements](00_system_requirements.md) | — |
| 1 | [Download Datasets and Create Artifacts](01_datasets.md) | 0 |
| 2 | [Train Models](02_training.md) | 1 |
| 3 | [Model Benchmarks and Plot Generation](03_model_evaluation.md) | 2 |
| 4 | [Set Up Scoring Endpoints](04_scoring_endpoints.md) | 1, 2 |
| 5 | [BBKNN Evaluations](05_bbknn.md) | 2, 4 |
| 6 | [VVS Benchmarks](06_vvs.md) | 2, 4 |
| 7 | [Comparison Benchmarks](07_comparisons.md) | 4, 6 |

```
datasets (1) → models (2) → ┬─→ model evaluation (3)
                            └─→ scoring endpoints (4) → ┬─→ BBKNN (5)
                                                        └─→ VVS (6) → comparisons (7)
```

## Reproducing figures exactly
Running the code may produce slightly different results due to the stochastic
nature of several steps. To reproduce the paper figures exactly, download the
[Zenodo data archive](https://zenodo.org/uploads/18615633) and place it in the
`src/blob_store/internal` directory.
