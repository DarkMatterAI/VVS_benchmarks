FROM python:3.10-slim

WORKDIR /code

RUN apt-get update && \
    apt-get install --no-install-recommends -y git libxrender1 libxtst6 libxi6 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements_download.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY src_download_files /code/src_download_files
COPY src_processing /code/src_processing
COPY src_create_datasets /code/src_create_datasets

# default blob-store location (override with -e or docker-compose)
ENV BLOB_STORE=/code/blob_store
VOLUME ["/code/blob_store"]

CMD ["python", "-u", "-m", "src_download_files"]

