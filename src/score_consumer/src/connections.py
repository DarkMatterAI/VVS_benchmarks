import os, pika
from sqlalchemy import create_engine
from .utils import log

EXCHANGE = os.getenv('RABBITMQ_EXCHANGE_NAME')
BINDING_KEY = "request.benchmark_score.score.*.*.*"
BINDING_KEY_GPU = "request.benchmark_score_gpu.score.*.*.*"

def make_channel():
    params = pika.ConnectionParameters(
        host=os.getenv("RABBITMQ_HOST"),
        port=int(os.getenv("RABBITMQ_PORT")),
        credentials=pika.PlainCredentials(
            os.getenv("RABBITMQ_USER"), 
            os.getenv("RABBITMQ_PASS")
        ),
        heartbeat=180,
    )

    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    return channel, connection 

