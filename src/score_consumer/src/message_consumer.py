import os 
import json, time, importlib, random
from functools import partial
import signal
from contextlib import contextmanager
import pika
from .connections import make_channel, EXCHANGE, BINDING_KEY, BINDING_KEY_GPU
from .utils import log, TimeoutException
from dotenv import load_dotenv; load_dotenv()

SELF_CONTAINED = os.getenv("SELF_CONTAINED", "false")

@contextmanager
def time_limit(seconds: int):
    """Context-manager that raises TimeoutException after *seconds*."""
    def _handler(_signo, _frame):
        raise TimeoutException()
    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)                   # cancel alarm
        signal.signal(signal.SIGALRM, old_handler)


try:
    # CPU consumer doesn't have torch installed
    import torch 
    from .erbb1_mlp import ERBB1
    erbb1_mlp = ERBB1 
    erbb1_mlp.load_models()
except:
    erbb1_mlp = None 

try:
    # GPU consumer doesn't have opensys installed
    from .synthemol_rf import score as synthemol_score
    from .openeye import dock_2zdt, dock_6lud, rocs_2chw
    # batch_dummy = dummy
    batch_dummy = lambda x: [{"valid": True, "score": random.random()} for i in x]
except:
    synthemol_score, dock_2zdt, dock_6lud, rocs_2chw, batch_dummy = None, None, None, None, None

if erbb1_mlp is not None:
    binding_key = BINDING_KEY_GPU
    device='GPU'
else:
    binding_key = BINDING_KEY
    device='CPU'

PLUGIN_MAP_PATH = "/code/plugin_ids.json"   # created by create_records.py

if SELF_CONTAINED == "false":
    with open(PLUGIN_MAP_PATH) as f:
        PLUGIN_IDS = json.load(f)               # {plugin_name: id}
else:
    PLUGIN_IDS = {}

PLUGIN_TIMEOUTS = {
    "docking_6lud": 30,
    "docking_2zdt": 30,
    "rocs_2chw":    30,
    "synthemol_rf":  30,
    "erbb1_mlp":     30,
    "bench_dummy":   5,
}
DEFAULT_TIMEOUT = 60

EXECUTE_MAP = {
    "bench_dummy": batch_dummy,
    "synthemol_rf": synthemol_score,
    "docking_2zdt": dock_2zdt,
    "docking_6lud": dock_6lud,
    "rocs_2chw": rocs_2chw,
    "erbb1_mlp": erbb1_mlp
}
EXECUTE_MAP = {k:v for k,v in EXECUTE_MAP.items() if v is not None}
log(f"Loaded {device} consumer with {len(EXECUTE_MAP)} scores active")
log(f"Listening at {binding_key}")

def execute_plugin(plugin_name: str, item):
    """
    Call the score function with a hard timeout.
    Raises TimeoutException if the limit is reached.
    """
    func = EXECUTE_MAP[plugin_name]
    timeout = PLUGIN_TIMEOUTS.get(plugin_name, DEFAULT_TIMEOUT)

    log(f"Scoring with {plugin_name}  (timeout {timeout}s)")

    with time_limit(timeout):
        if isinstance(item, list):
            return func(item)
        else:
            return func([item])[0]

def fail_response(item):
    fail_default = {"valid": False, "score": None}
    return [fail_default for i in item] if type(item)==list else fail_default

def callback(ch, method, props, body):
    try:
        data = json.loads(body)
        _pfx, group, ptype, plugin_id, item_id, req_id = (
            method.routing_key.split(".")
        )
        is_external = item_id == "internal"

        # which plugin?
        plugin_name = (plugin_id if is_external else
                       next(k for k, v in PLUGIN_IDS.items()
                            if str(v) == plugin_id))

        # ── run score function (may raise TimeoutException) ───────────
        value     = execute_plugin(plugin_name, data)
        resp_body = json.dumps(value)

    except TimeoutException:
        log("⏲  time-limit exceeded", "red")
        resp_body = json.dumps(fail_response(data))

    except Exception as e:
        # raise e 
        log(f"✖ error {e}, {str(e)}, {type(e)}", "red")
        resp_body = json.dumps(fail_response(data))

    finally:
        # ---------- reply to RPC caller --------------------------------
        if props.reply_to:
            ch.basic_publish(
                exchange="",
                routing_key=props.reply_to,
                body=resp_body,
                properties=pika.BasicProperties(
                    correlation_id=props.correlation_id,
                    delivery_mode=2,
                ),
            )
        ch.basic_ack(method.delivery_tag)
        return

def start_consumer():
    if SELF_CONTAINED == "false":
        channel, connection = make_channel()
        result = channel.queue_declare(queue=f'vvs_benchmark_consumer_{device}', 
                                       durable=True, auto_delete=True, arguments={
            "x-dead-letter-exchange": f"{EXCHANGE}.dlx"
        })
        queue_name = result.method.queue
        channel.queue_bind(
            exchange=EXCHANGE,
            queue=queue_name,
            routing_key=binding_key
        )

        channel.basic_qos(prefetch_count=1)
        consumer_tag = channel.basic_consume(
            queue=queue_name,
            on_message_callback=callback
        )
    else:
        print("Creating Exchange")
        channel, connection = make_channel()
        channel.exchange_declare(exchange=EXCHANGE, exchange_type='topic')
        result = channel.queue_declare(queue=f'vvs_benchmark_consumer_{device}', 
                                       durable=True, auto_delete=True)
        queue_name = result.method.queue
        channel.queue_bind(
            exchange=EXCHANGE,
            queue=queue_name,
            routing_key=binding_key
        )

        channel.basic_qos(prefetch_count=1)
        consumer_tag = channel.basic_consume(
            queue=queue_name,
            on_message_callback=callback
        )

    return channel, connection, consumer_tag
