FROM python:3.10-slim

WORKDIR /code
ENV PIP_NO_CACHE_DIR=1

# --- OS deps for RDKit and latex ---

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    libxrender1 \
    libxtst6 \ 
    libxi6 \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    cm-super \
    dvipng && \
    rm -rf /var/lib/apt/lists/*


COPY requirements.txt /code/
RUN pip install --upgrade -r /code/requirements.txt

COPY src /code/src
ENV BLOB_STORE=/code/blob_store
VOLUME ["/code/blob_store"]

CMD ["python", "-u", "-m", "model_evaluation.embedding_compression.generate_data"]
