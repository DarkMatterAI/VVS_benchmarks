# 4. Set Up Scoring Endpoints

**Prerequisites:** [Download Datasets and Create Artifacts](01_datasets.md),
[Train Models](02_training.md) (for the EGFR MLP score)

Subsequent benchmarks use several scoring functions as optimization targets. These
are served via Docker Compose, which creates the following containers:

- `score_consumer` — runs CPU scoring functions
- `score_consumer_gpu` — runs GPU scoring functions
- `rabbitmq` — queue system used by benchmarks to dispatch scoring jobs

> **Note:** The OpenEye docking and ROCS scoring functions require a valid OpenEye
> license at `./src/score_consumer/secrets/oe_license.txt`. See
> [System Requirements](00_system_requirements.md).

## Instructions

Start the scoring services:

```bash
cd src/score_consumer
docker compose -f docker-compose-self-contained.yml up -d --build
docker compose -f docker-compose-self-contained.yml scale score_consumer={SCORE_WORKERS}
```

> **Note:** The `scale` command creates replicas of the CPU score consumer for
> parallel processing. Data for the paper was generated with `SCORE_WORKERS=64`.

## Cleanup

To tear down the scoring services when no longer needed:

```bash
docker compose -f docker-compose-self-contained.yml down
```