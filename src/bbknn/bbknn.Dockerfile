# bbknn.Dockerfile
FROM python:3.10-slim

WORKDIR /code
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/code

# ── system deps (if any) ─────────────────────────────────────────
RUN apt-get update && \
    apt-get install --no-install-recommends -y git && \
    rm -rf /var/lib/apt/lists/*

# ── Python deps ─────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install -r requirements.txt

# ── project code ────────────────────────────────────────────────
COPY src            /code/bbknn
COPY config.yaml    /code/bbknn/config.yaml

# ── default command (debug) ─────────────────────────────────────
CMD ["python", "-m", "bbknn.bbknn_eval"]
