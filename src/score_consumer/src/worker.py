import time, signal, sys, os, random 
from .message_consumer import start_consumer
from .utils import log
from dotenv import load_dotenv; load_dotenv()

RECYCLE_S = int(float(os.getenv("RECYCLE_MIN", "15")) * 60)
RECYCLE_J = int(float(os.getenv("RECYCLE_JITTER", "3")) * 60)

def run_once():
    ch, conn, tag = start_consumer()
    start = time.time()
    log("Consumer started")
    recycle_time = RECYCLE_S + (-1 if random.random()>0.5 else 1)*RECYCLE_J*random.random()

    try:
        while True:
            ch.connection.process_data_events(time_limit=0.25)
            if time.time() - start > recycle_time:
                ch.basic_cancel(tag)
                log("⟳ recycle interval reached - shutting down", "yellow")
                break
    finally:
        ch.close(); conn.close()

def worker():
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True:
        run_once()
        time.sleep(1)

