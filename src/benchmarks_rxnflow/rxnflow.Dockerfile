FROM python:3.12-slim
WORKDIR /code

# system deps for RDKit & build
RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential \
        libxrender1 libxtst6 libxi6 && \
    rm -rf /var/lib/apt/lists/*

# RxnFlow (editable install, pulls PyG wheels automatically)
RUN git clone https://github.com/SeonghwanSeo/RxnFlow.git && \
    cd RxnFlow && pip install --no-cache-dir -e . \
        --find-links https://data.pyg.org/whl/torch-2.5.1+cu121.html

# python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1 PYTHONPATH=/code
COPY src/ /code/src/
COPY debug_space.yaml /code/debug_space.yaml
COPY sweep_space.yaml /code/sweep_space.yaml
COPY final_space.yaml /code/final_space.yaml

# default command (overridden by run.sh)
CMD ["python", "-m", "src.rxnflow_runner", "--cfg", "/code/debug_space.yaml"]
