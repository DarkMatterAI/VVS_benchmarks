FROM python:3.10-slim

WORKDIR /code
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/code \
    PLUGIN_MAP_PATH=/plugin_map/plugin_ids.json

# ---------- Python deps ----------
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY requirements_gpu.txt ./requirements_gpu.txt
RUN pip install -r requirements_gpu.txt

# ---------- project code ----------
COPY . /code

# ---------- default entry ----------
CMD ["python", "-m", "src.main"]

