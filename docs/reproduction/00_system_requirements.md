# System Requirements

## Software dependencies
The pipeline runs through a combination of shell scripts, Python, and Docker
Compose services. All internal dependencies are pinned in the provided container
images and environment files. Host requirements:

- **Operating system:** Linux (tested on Ubuntu 20.04 LTS)
- **Docker Engine:** ≥ 24.0 (tested on 24.0.7)
- **Docker Compose:** ≥ 2.29 (tested on 2.29.2)
- **NVIDIA Driver:** ≥ 550 (tested on 550.90.07)
- **NVIDIA Container Toolkit:** ≥ 1.15 (required for GPU passthrough)

## Hardware requirements
- An NVIDIA GPU is required for embedding and model inference.
  **Tested on:** NVIDIA RTX 3090 (24 GB) and NVIDIA A100 (40/80 GB).
- **Disk:** Full dataset processing requires \~1000 GB of working space (\~500 GB of
  persistent artifacts).
- **Recommended RAM:** ≥ 32 GB.

## Licenses
Some benchmarks use OpenEye docking and ROCS scoring, which require a valid
OpenEye license. Save the license file at:

```
./src/score_consumer/secrets/oe_license.txt
```

> **Note:** Only the OpenEye-based scoring functions require this license. BBKNN
> retrieval, model training, and the MLP/random-forest scoring functions do not.

## Reproducing figures exactly
Running the code may produce slightly different results due to the stochastic
nature of several steps. To reproduce the paper figures exactly, download the
[Zenodo data archive](https://zenodo.org/records/18615633) and place it in the
`src/blob_store/internal` directory.