import json, uuid, pika, os, time

RABBIT     = dict(
    host=os.getenv("RABBITMQ_HOST", "rabbitmq"),
    port=int(os.getenv("RABBITMQ_PORT", 5672)),
    credentials=pika.PlainCredentials(
        os.getenv("RABBITMQ_USER", "rabbitmq_user"),
        os.getenv("RABBITMQ_PASS", "rabbitmq_password"),
    ),
    heartbeat=180
)

class BatchScorer:
    def __init__(self, exchange, routing_key, timeout=15):
        self.exchange = exchange
        self.routing  = routing_key
        self.timeout  = timeout
        self._connect()
    # --------------------------------------------------------------
    def _connect(self):
        self.conn = pika.BlockingConnection(pika.ConnectionParameters(**RABBIT))
        self.ch   = self.conn.channel()
        qname = self.ch.queue_declare(queue="", exclusive=True).method.queue
        self.callback_q = qname
        self.ch.basic_consume(qname, on_message_callback=self._on_reply, auto_ack=True)
        self._pending = {}

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def score_batch(self, items):
        """
        items : List[dict-like]  - payloads for score function
        returns: List[float]     - same length, same order
        """
        self._pending = {}                           # reset for this batch
        for idx, payload in enumerate(items):
            cid = str(uuid.uuid4())
            self._pending[cid] = (idx, None)         # placeholder

            self.ch.basic_publish(
                exchange=self.exchange,
                routing_key=self.routing,
                body=json.dumps(payload),
                properties=pika.BasicProperties(
                    reply_to=self.callback_q,
                    correlation_id=cid,
                    delivery_mode=2,
                ),
            )

        # wait until every corr_id filled
        t0 = time.time()
        while any(score is None for _, score in self._pending.values()):
            self.conn.process_data_events(time_limit=0.5)
            if time.time() - t0 > self.timeout:
                print("score_items timeout")
                break
                # raise TimeoutError("score_items timeout")

        # restore original order
        out = [None] * len(items)
        for cid, (i, res) in self._pending.items():
            out[i] = res
        failed = [i for i in out if i is None]
        if failed:
            print(f"{len(failed)} messages timed out")
        return out

    def _on_reply(self, ch, method, props, body):
        cid = props.correlation_id
        if cid in self._pending:
            _idx, _ = self._pending[cid]
            res = json.loads(body)
            self._pending[cid] = (_idx, res)
