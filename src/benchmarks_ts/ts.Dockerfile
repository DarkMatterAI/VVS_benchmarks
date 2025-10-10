FROM python:3.10-slim
WORKDIR /code

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PYTHONPATH=/code

RUN apt-get update && \
    apt-get install --no-install-recommends -y git && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/PatWalters/TS.git 

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /code

CMD ["python", "-m", "run_benchmark", "--run_type", "debug", "--run_name", "dev"]
