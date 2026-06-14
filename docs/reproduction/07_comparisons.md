# 7. Comparison Benchmarks

**Prerequisites:** [Set Up Scoring Endpoints](04_scoring_endpoints.md),
[VVS Benchmarks](06_vvs.md) (provides the VVS hyperparameter sweep)

We compare VVS to several other search methods. Each method first runs a
hyperparameter sweep: three replicas per hyperparameter configuration against the
`erbb1_mlp` score function. The results are analyzed to select the best
configuration, and each method is then run for five replicas using its best
hyperparameters against all score functions.

## Hyperparameter sweeps

Hyperparameters are configured via a `sweep_space.yaml` file in each benchmark
folder.

**SyntheMol:**
```bash
cd src/benchmarks_synthemol/
./run.sh sweep
```

**Thompson Sampling:**
```bash
cd src/benchmarks_ts/
./run.sh sweep
```

**RxnFlow:**
```bash
cd src/benchmarks_rxnflow/
./run.sh sweep
```

**RAD:**
```bash
cd src/benchmarks_rad/
./process_dataset.sh   # only needs to be run once
./run.sh sweep
```

**VVS:** The VVS hyperparameter sweep is run in [VVS Benchmarks](06_vvs.md).

## Compile best results

Parse all sweep runs to determine the best hyperparameters for each method:

```bash
cd data_analysis
./src/benchmarks/run.sh sweep
```

This produces a summary CSV at
`src/blob_store/internal/data_analysis/benchmarks/sweep/best_hyperparams.csv`, which
can be used to update each method's final hyperparameters.

## Run final benchmarks

For each method, update the corresponding `final_space.yaml` config and run:

```bash
cd src/{benchmarks_synthemol | benchmarks_ts | benchmarks_rxnflow | benchmarks_rad}
./src/benchmarks/run.sh final
```

For VVS final benchmarks:

```bash
cd src/vvs_local
./run.sh hyperparam_sweep --cfg bbknn_final.yaml --run_type final
```

## Plot results

Generate the benchmark comparison plots:

```bash
cd src/data_analysis
./src/benchmarks/run.sh final
```