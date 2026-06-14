# 5. BBKNN Evaluations

**Prerequisites:** [Train Models](02_training.md),
[Set Up Scoring Endpoints](04_scoring_endpoints.md) (required for the EGFR evaluation)

This section evaluates the BBKNN algorithm and generates the corresponding figures.

## Generate data

```bash
cd src/bbknn
./run.sh bbknn
./run.sh natprod
./run.sh egfr
```

- `./run.sh bbknn` — runs BBKNN with query molecules from the Enamine Assembled and
  Lyu 140M datasets
- `./run.sh natprod` — runs BBKNN with query molecules from the COCONUT natural
  product dataset
- `./run.sh egfr` — runs BBKNN with known EGFR ligands as queries and scores the
  results with the OpenEye docking endpoint. Also produces poses of specific known
  EGFR ligands and similar molecules retrieved by BBKNN.

## Evaluate and plot

```bash
cd src/data_analysis
./src/bbknn/run.sh cosine
./src/bbknn/run.sh sampling
./src/bbknn/run.sh egfr
```

- `./src/bbknn/run.sh cosine` — compares query and result molecules by cosine and
  Tanimoto similarity. Generates aggregate correlation plots and panel plots of
  molecules where cosine and Tanimoto disagree strongly.
- `./src/bbknn/run.sh sampling` — generates bar plots of query/result similarity for
  the Enamine Assembled, Lyu 140M, and COCONUT datasets; plots of total results vs.
  the BBKNN `k` parameter; and molecule grid plots of example query/result pairs.
- `./src/bbknn/run.sh egfr` — generates plots of docking score gain vs. sampling
  depth and example query/result pairs with corresponding docking scores.