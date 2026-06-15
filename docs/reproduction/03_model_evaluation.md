# 3. Model Benchmarks and Plot Generation

**Prerequisites:** [Train Models](02_training.md)

This section evaluates the Embedding Compression and Embedding Decomposer models
and generates the corresponding paper figures.

> **Note:** Some analysis scripts rely on hard-coded paths to specific model
> training runs. To reproduce, download the `training_runs` directory from the
> [Zenodo data archive](https://zenodo.org/uploads/18615633).

## Embedding Compression Evaluation

```bash
cd src/data_analysis
./src/model_evaluation/embedding_compression/run.sh generate
./src/model_evaluation/embedding_compression/run.sh plot
```

Generates:

- Four-panel figure showing ablations for loss function and model size
- Qualitative comparison plots of retrieval performance at different embedding sizes

## Embedding Decomposer Evaluation

```bash
cd src/data_analysis
./src/model_evaluation/enamine_decomposer/run.sh generate
./src/model_evaluation/enamine_decomposer/run.sh plot
```

Generates:

- Loss function ablation plot
- Qualitative retrieval performance plots
- Heatmap of retrieval accuracy for different input/output embedding size combinations
- Plot of "flipped predictions" and retrieval miss rate as a function of SMILES length