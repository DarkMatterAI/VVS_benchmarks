# VVS Benchmarks

Benchmark code for the paper [Efficient Search of Ultra-Large Synthesis On-Demand
Libraries with Chemical Language Models](https://www.biorxiv.org/content/10.1101/2025.09.04.674350v1.full).

Code subject to change during the peer review process.

This repository reproduces all experiments and figures in the paper. Full
reproduction instructions are split across the guides in [`docs/reproduction/`](docs/reproduction).
A minimal demo is provided below.

## 1. System Requirements

### Software dependencies
The pipeline runs through a combination of shell scripts, Python, and Docker
Compose services. All internal dependencies are pinned in the provided container
images and environment files. Host requirements:

- **Operating system:** Linux (tested on Ubuntu 20.04 LTS)
- **Docker Engine:** ≥ 24.0 (tested on 24.0.7)
- **Docker Compose:** ≥ 2.29 (tested on 2.29.2)
- **NVIDIA Driver:** ≥ 550 (tested on 550.90.07)
- **NVIDIA Container Toolkit:** ≥ 1.15 (required for GPU passthrough)

### Hardware requirements
- An NVIDIA GPU is required for embedding and model inference.
  **Tested on:** NVIDIA RTX 3090 (24 GB) and NVIDIA A100 (40/80 GB).
- **Disk:** Full dataset processing requires ~1000 GB of working space (~500 GB of
  persistent artifacts). The demo requires only a few GB.
- **Recommended RAM:** ≥ 32 GB.

### Licenses
Some benchmarks use OpenEye docking and ROCS scoring, which require a valid
OpenEye license saved at `./src/score_consumer/secrets/oe_license.txt`. The demo
does **not** require an OpenEye license.

## 2. Installation Guide

### Instructions
```bash
git clone https://github.com/DarkMatterAI/VVS_benchmarks
cd vvs_benchmarks
```

All benchmarks and analysis are executed via shell script in containerized environments - no further installation is necessary. See Instructions for Use section for instructions on running benchmarks.

## 3. Demo

A minimal, self-contained demo runs a building block space search (BBKNN) using a
toy dataset of 100 Enamine building blocks. It demonstrates the core
decomposition → retrieval → assembly pipeline: query molecules are decomposed into
building block embeddings, nearest-neighbor building blocks are retrieved from the
toy library, and the retrieved blocks are assembled into synthesizable analogues
ranked by similarity to the query. The demo requires a GPU but **no OpenEye
license** and only a few GB of disk.

### Instructions
From the repository root:

```bash
cd src/bbknn
./run_demo.sh
```

This builds the container, embeds the demo building blocks into a small vector
database, and runs BBKNN against the demo query molecules. The demo files
(`demo_bbs.csv`, `demo_smiles.csv`, `demo_reactions.json`) are located in
`src/bbknn/src/demo/`. To run on your own data, replace these files following the
same column schema.

### Expected output
Results are written to `blob_store/demo/results/demo_results.csv`. Each row
contains a query molecule, an assembled analogue (`result`), the building blocks
used, and the cosine/Tanimoto similarity of the analogue to the query. A preview of
the top results is printed to the console on completion. An example result file is provided at `src/bbknn/src/demo/demo_results.csv`.

### Expected run time
The demo completes in approximately 5 on a normal desktop with a single
GPU (the first run is slower, as containers are built for the first time and model weights are downloaded from HuggingFace).

## 4. Instructions for Use

Reproducing all datasets, models, experiments, and figures are broken down into sections, with detailed instructions provided in [`docs/reproduction/`](docs/reproduction): 

0. [System Requirements](docs/reproduction/00_system_requirements.md)
1. [Download Datasets and Create Artifacts](docs/reproduction/01_datasets.md)
2. [Train Models](docs/reproduction/02_training.md)
3. [Model Benchmarks and Plot Generation](docs/reproduction/03_model_evaluation.md)
4. [Set Up Scoring Endpoints](docs/reproduction/04_scoring_endpoints.md)
5. [BBKNN Evaluations](docs/reproduction/05_bbknn.md)
6. [VVS Benchmarks](docs/reproduction/06_vvs.md)
7. [Comparison Benchmarks](docs/reproduction/07_comparisons.md)

> Results may differ slightly from the paper due to stochasticity in several steps.
> To reproduce paper figures exactly, download the
> [Zenodo data archive](https://zenodo.org/uploads/18615633) and place it in
> `src/blob_store/internal`.

