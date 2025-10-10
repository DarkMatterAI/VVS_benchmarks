FROM python:3.10-slim
WORKDIR /code

RUN apt-get update && \
    apt-get install --no-install-recommends -y git libxrender1 libxtst6 libxi6 && \
    rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/code

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /code

# default run type = debug
CMD ["python", "-m", "run_benchmark", "--run_type", "debug", "--run_name", "dev"]
