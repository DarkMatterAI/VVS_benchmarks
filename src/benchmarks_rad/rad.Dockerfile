FROM python:3.11-slim

# ----- system deps (gcc for usearch & RDKit, redis-cli for simple tests)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git\
    libxrender1\
    libxtst6\
    libxi6\
    make\
    g++\
    build-essential\
    redis\
    curl &&\
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /code
ENV PYTHONUNBUFFERED=1

# ---------- Python deps ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- RAD (pinned commit) ----------
RUN git clone --recursive https://github.com/keiserlab/rad.git && \
    cd rad && \
    git checkout a526b30cca5fb05fd4729e20375fc6045fd9244e && \
    pip install .

# project files
COPY src/ /code/src/

COPY rad_data_params.yaml /code/rad_data_params.yaml
COPY sweep_space.yaml /code/sweep_space.yaml
COPY debug_space.yaml /code/debug_space.yaml
COPY final_space.yaml /code/final_space.yaml

ENV PYTHONPATH=/code

CMD ["python", "-m", "src.rad_runner"]
