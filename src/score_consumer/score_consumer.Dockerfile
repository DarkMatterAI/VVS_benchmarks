FROM python:3.10-slim

WORKDIR /code
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/code \
    PLUGIN_MAP_PATH=/plugin_map/plugin_ids.json

# deps
RUN pip install -i https://pypi.anaconda.org/OpenEye/simple OpenEye-toolkits

COPY requirements.txt .
RUN pip install -r requirements.txt

# project
COPY . /code

# Default process: run the RabbitMQ consumer
CMD ["python", "-m", "src.main"]
