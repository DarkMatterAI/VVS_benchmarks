"""
score_rpc.py
─────────────────────────────────────────────────────────────────────
Thin wrapper around the internal RabbitMQ RPC scoring service used
for the EGFR benchmark.

• BatchScorer - AMQP helper that keeps at most *batch_size* requests
                inflight; each message has its own timeout timer.
• RPCScore    - SMILES→score callable with an in-memory result cache.
"""

from __future__ import annotations
import json, uuid, os, time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from dotenv import load_dotenv; load_dotenv()
import numpy as np
import pika

from rich.console import Console
console  = Console()


# ───────────────────────── Rabbit settings (env-configurable)
RABBIT: Dict = dict(
    host=os.getenv("RABBITMQ_HOST", "rabbitmq"),
    port=int(os.getenv("RABBITMQ_PORT", 5672)),
    credentials=pika.PlainCredentials(
        os.getenv("RABBITMQ_USER", "rabbitmq_user"),
        os.getenv("RABBITMQ_PASS", "rabbitmq_password"),
    ),
    heartbeat=180
)
EXCHANGE = os.getenv("RABBITMQ_EXCHANGE_NAME")


# ╭───────────────────────  AMQP helper  ─────────────────────────────╮
class BatchScorer:
    """
    Fire *N* messages off to RabbitMQ, keep at most **batch_size**
    outstanding at any moment, wait for the replies and preserve the
    original order.

    Parameters
    ----------
    exchange    : str
    routing_key : str
    timeout     : int     seconds *per message* before declaring it lost
    batch_size  : int     max un-acked messages allowed simultaneously
    """

    def __init__(
        self,
        exchange: str,
        routing_key: str,
        timeout: int = 15,
        batch_size: int = 128,
    ):
        self.exchange   = exchange
        self.routing    = routing_key
        self.timeout    = timeout
        self.batch_size = batch_size
        self._connect()

    # ───────────────────────── connection boiler-plate
    def _connect(self):
        self.conn = pika.BlockingConnection(pika.ConnectionParameters(**RABBIT))
        self.ch   = self.conn.channel()
        qname = self.ch.queue_declare(queue="", exclusive=True).method.queue
        self.callback_q = qname
        self.ch.basic_consume(
            qname,
            on_message_callback=self._on_reply,
            auto_ack=True,
        )
        self._pending: Dict[str, Tuple[int, float]] = {}   # cid -> (idx, send_time)
        self._results: Dict[str, float] = {}               # cid -> score

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    # ───────────────────────── public scoring API
    def score_batch(self, items: List[dict]) -> List[float]:
        """
        items  : list[dict]     user-defined payloads
        returns: list[float]    score per item (original order),
                                `None` when individual timeout is hit.
        """
        out = [None] * len(items)
        send_idx = 0
        self._pending.clear()
        self._results.clear()
        last_log = time.time()

        # main loop -------------------------------------------------------
        while send_idx < len(items) or self._pending:
            # ----- issue new messages until the concurrency cap ----------
            if time.time() - last_log > 2:
                console.log(f"[blue]Sending messages {send_idx} / {len(items)}, {len(self._pending)} outstanding")
                last_log = time.time()

            while send_idx < len(items) and len(self._pending) < self.batch_size:
                payload = items[send_idx]
                cid = str(uuid.uuid4())
                self._pending[cid] = (send_idx, time.time())

                self.ch.basic_publish(
                    exchange=self.exchange,
                    routing_key=self.routing,
                    body=json.dumps(payload),
                    properties=pika.BasicProperties(
                        reply_to=self.callback_q,
                        correlation_id=cid,
                        delivery_mode=2,      # persistent
                    ),
                )
                send_idx += 1

            # ----- process network events / replies ----------------------
            self.conn.process_data_events(time_limit=0.25)

            # ----- commit freshly arrived scores -------------------------
            for cid, score in list(self._results.items()):
                idx, _ = self._pending.pop(cid, (None, None))
                if idx is not None:
                    out[idx] = score
                self._results.pop(cid, None)          # consume once

            # ----- mark individual timeouts ------------------------------
            now = time.time()
            timed_out = [
                cid for cid, (idx, sent) in self._pending.items()
                if now - sent > self.timeout
            ]
            for cid in timed_out:
                idx, _ = self._pending.pop(cid)
                out[idx] = None                       # timeout marker

        n_timeouts = len([i for i in out if i is None])
        console.log(f"[blue]Finished message batch with {n_timeouts} timeouts")
        return out

    # ───────────────────────── internal callback
    def _on_reply(self, _ch, _method, props, body):
        cid = props.correlation_id
        if cid in self._pending:
            try:
                score = json.loads(body)["score"]
            except Exception:
                score = None
            self._results[cid] = score
# ╰────────────────────────────────────────────────────────────────────╯


# ╭────────────────────  simple cache wrapper  ───────────────────────╮
class RPCScore:
    """
    Callable SMILES-to-score helper with in-memory caching.

    Parameters
    ----------
    plugin      : str     backend scoring model (e.g. “docking_6lud”)
    timeout     : int     per-message timeout in seconds
    batch_size  : int     max concurrent messages inside **BatchScorer**
    device      : str     "gpu" switches routing key to GPU queue
    save_dir    : Optional[str] = None, save directory for docking poses
    """

    def __init__(
        self,
        plugin: str,
        timeout: int,
        batch_size: int = 128,
        retries: int = 0,
        device: str = "cpu",
        save_dir: Optional[str] = None,
    ):
        group = "benchmark_score_gpu" if device == "gpu" else "benchmark_score"
        self.rpc = BatchScorer(
            exchange=EXCHANGE,
            routing_key=f"request.{group}.score.{plugin}.internal.internal",
            timeout=timeout,
            batch_size=batch_size,
        )
        self.retries = retries
        self.cache: Dict[str, float] = {}
        self.save_dir = save_dir 

    # public ------------------------------------------------------------
    def close(self):
        self.rpc.close()

    def rpc_loop(self, payloads: list[dict]):
        retry_count = self.retries 
        while retry_count >= 0:
            print("posting")
            retry_payloads = []
            results = self.rpc.score_batch(payloads)
            for payload, score in zip(payloads, results):
                if score is None:
                    retry_payloads.append(payload)
                else:
                    self.cache[payload["item_data"]["item"]] = score

            # set up next loop
            payloads = retry_payloads 
            retry_count -= 1
            if not payloads:
                break 
        
        # any remaining after retry count
        for payload in payloads:
            self.cache[payload["item_data"]["item"]] = -np.inf


    def __call__(self, smiles: List[str]) -> List[float]:
        """
        List of SMILES in ➜ list[float] out (same order).
        Unscored / timed-out entries come back as ``-np.inf``.
        """
        missing_payloads: List[dict] = []
        idx_lookup: defaultdict[str, List[int]] = defaultdict(list)

        for i, smi in enumerate(smiles):
            idx_lookup[smi].append(i)
            if smi not in self.cache:
                missing_payloads.append({"item_data": {"item": smi}})

        if missing_payloads:
            if self.save_dir is not None:
                for p in missing_payloads:
                    p["save_dir"] = self.save_dir 
            results = self.rpc.score_batch(missing_payloads)
            for payload, score in zip(missing_payloads, results):
                if score is None:
                    score = -np.inf
                self.cache[payload["item_data"]["item"]] = score

        out = [None] * len(smiles)
        for smi, idxs in idx_lookup.items():
            sc = self.cache.get(smi, -np.inf)
            for idx in idxs:
                out[idx] = sc
        return out

    def score_pairs(
        self,
        pairs: List[Tuple[str, str]],
        save_dir: str,
    ) -> List[Tuple[float, float]]:
        """
        Score (query, result) SMILES pairs, saving docked poses as
        ``query_<i>.sdf`` / ``result_<i>.sdf`` under *save_dir* on the
        docking server.

        Always sends every molecule to the server (bypasses the score
        cache) so that each pair gets its own uniquely-named pose files.

        Parameters
        ----------
        pairs    : list of (query_smiles, result_smiles)
        save_dir : sub-directory under SAVE_DIR / protein on the server

        Returns
        -------
        list of (query_score, result_score) tuples.
        Failed / timed-out entries are returned as ``-np.inf``.
        """
        payloads: List[dict] = []
        for i, (q_smi, r_smi) in enumerate(pairs):
            payloads.append({
                "item_data": {"item": q_smi},
                "save_dir":  save_dir,
                "save_name": f"query_{i}",
            })
            payloads.append({
                "item_data": {"item": r_smi},
                "save_dir":  save_dir,
                "save_name": f"result_{i}",
            })

        raw = self.rpc.score_batch(payloads)

        pair_scores: List[Tuple[float, float]] = []
        for i, (q_smi, r_smi) in enumerate(pairs):
            q_sc = raw[2 * i]     if raw[2 * i]     is not None else -np.inf
            r_sc = raw[2 * i + 1] if raw[2 * i + 1] is not None else -np.inf
            self.cache[q_smi] = q_sc
            self.cache[r_smi] = r_sc
            pair_scores.append((q_sc, r_sc))

        return pair_scores

# ╰────────────────────────────────────────────────────────────────────╯
