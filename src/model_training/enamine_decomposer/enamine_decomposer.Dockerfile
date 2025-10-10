FROM python:3.10-slim

# Faster installs
ENV PIP_NO_CACHE_DIR=1

WORKDIR /code

# ---------------- Python libs ----------------
COPY requirements.txt /code/requirements.txt
RUN pip install --upgrade -r /code/requirements.txt

# ---------------- Source code ----------------
COPY src /code/src

# Path inside container that the outer script will bind-mount
ENV BLOB_STORE=/code/blob_store
VOLUME ["/code/blob_store"]

CMD ["python", "-u", "-m", "src.process_dataset"]
