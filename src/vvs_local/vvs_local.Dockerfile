# vvs_local.Dockerfile  ───────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /code
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/code

# ── minimal system deps (git for HF pull; lib RDKit deps come via wheel)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# ── Python deps ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install -r requirements.txt

# ── project code ─────────────────────────────────────────────────────
COPY src /code/vvs_local
COPY hyperparam_configs      /code/hyperparam_configs
COPY lr_sweep_bbknn.yaml     /code/lr_sweep_bbknn.yaml
COPY lr_sweep_knn.yaml       /code/lr_sweep_knn.yaml

CMD ["bash", "-c", "echo 'Run via ./run.sh lr_sweep <extra-args>'"]
