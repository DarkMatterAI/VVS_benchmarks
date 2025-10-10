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
import json, uuid, time
import pika
from datetime import datetime
from typing import Dict, List 
import numpy as np 

from .constants import RABBIT, EXCHANGE, console


# ╭───────────────────────  AMQP helper  ─────────────────────────────╮
class BatchScorer:
    """
    Fire *N* messages off to RabbitMQ and keep **batch_size** (or fewer)
    batches in-flight at once.  Each batch is one AMQP message that may
    contain 1 … batch_size payload objects.  Per-batch timeouts are
    enforced and original ordering is preserved.

    Parameters
    ----------
    exchange    : str
    routing_key : str
    timeout     : int     seconds (applied per BATCH, not per item)
    batch_size  : int     number of payloads per Rpc message;
                          keep at most this many batches un-acked.
    """

    def __init__(
        self,
        exchange: str,
        routing_key: str,
        timeout: int = 15,
        batch_size: int = 128,
        concurrency: int = 1,
    ):
        self.exchange    = exchange
        self.routing     = routing_key
        self.timeout     = timeout
        self.batch_size  = max(1, batch_size)   # 1 ⇒ legacy, un-batched mode
        self.concurrency = concurrency
        self._connect()

    # ───────────────────────── connection boiler-plate
    def _connect(self):
        self.conn = pika.BlockingConnection(pika.ConnectionParameters(**RABBIT))
        self.ch   = self.conn.channel()

        # reply-to queue
        qname = self.ch.queue_declare(queue="", exclusive=True).method.queue
        self.callback_q = qname
        self.ch.basic_consume(
            qname,
            on_message_callback=self._on_reply,
            auto_ack=True,
        )

        # bookkeeping
        self._pending: dict[str, tuple[list[int], float]] = {}
        self._results: dict[str, list[float | None]]      = {}

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    # ───────────────────────── public API
    def score_batch(self, items: list[dict]) -> list[float | None]:
        """
        Parameters
        ----------
        items : list[dict]   arbitrary payloads to forward to the scorer

        Returns
        -------
        list[float | None]  score per item (timeout → None)
        """
        out   : list[float | None] = [None] * len(items)
        out_ts: list[float | None] = [None] * len(items)
        send_idx   = 0                      # cursor into *items*
        last_log   = time.time()

        self._pending.clear()
        self._results.clear()

        # ── helper for publishing one chunk ───────────────────────────
        def _publish(chunk: list[dict], indices: list[int]) -> None:
            cid   = str(uuid.uuid4())
            body  = json.dumps(chunk[0] if len(chunk) == 1 else chunk)
            self._pending[cid] = (indices, time.time())

            self.ch.basic_publish(
                exchange=self.exchange,
                routing_key=self.routing,
                body=body,
                properties=pika.BasicProperties(
                    reply_to=self.callback_q,
                    correlation_id=cid,
                    delivery_mode=2,          # persistent
                ),
            )

        # ── main loop: send, poll, harvest, timeout ──────────────────
        while send_idx < len(items) or self._pending:
            # progress log every ~2 s
            if time.time() - last_log > 2:
                console.log(
                    f"[blue]=== Score RPC: {send_idx}/{len(items)} sent - "
                    f"{len(self._pending)} batches awaiting replies"
                )
                last_log = time.time()

            # — issue more batches (respecting concurrency cap) —
            while send_idx < len(items) and len(self._pending) <= self.concurrency:
                chunk   = items[send_idx : send_idx + self.batch_size]
                idxs    = list(range(send_idx, send_idx + len(chunk)))
                _publish(chunk, idxs)
                send_idx += len(chunk)

            # — process replies —
            self.conn.process_data_events(time_limit=0.25)

            # — commit completed batches —
            completed_ts = datetime.now()
            for cid, scores in list(self._results.items()):
                idxs, _sent = self._pending.pop(cid, ([], None))
                for k, idx in enumerate(idxs):
                    out[idx] = scores[k] if k < len(scores) else None
                    out_ts[idx] = completed_ts if k < len(scores) else None
                self._results.pop(cid, None)

            # — handle timeouts —
            now = time.time()
            for cid, (idxs, sent) in list(self._pending.items()):
                if now - sent > self.timeout:
                    for idx in idxs:
                        out[idx] = None       # mark timeout
                        out_ts[idx] = datetime.now()
                    self._pending.pop(cid)

        console.log(f"[blue]=== Score RPC: Batch done - {sum(o is None for o in out)} timeouts")
        return out, out_ts 

    # ───────────────────────── internal callback
    def _on_reply(self, _ch, _method, props, body):
        cid = props.correlation_id
        if cid not in self._pending:
            return                                      # stray reply

        try:
            payload = json.loads(body)
            # normalise: Single dict -> list[dict]
            payload_list = payload if isinstance(payload, list) else [payload]
            scores = [
                p.get("score") if isinstance(p, dict) else None
                for p in payload_list
            ]
        except Exception:
            # malformed reply = fail whole batch
            idxs, _ = self._pending[cid]
            scores = [None] * len(idxs)

        self._results[cid] = scores
# ╰────────────────────────────────────────────────────────────────────╯

# ╭────────────────────  simple cache wrapper  ───────────────────────╮
class RPCScore:
    """
    Callable SMILES-to-score helper with in-memory caching **plus**
    inference-budget *and* wall-clock runtime limits.

    Parameters
    ----------
    plugin         : str   - backend scoring model name
    timeout        : int   - batch-level timeout (sec) for `BatchScorer`
    batch_size     : int   - payloads per AMQP message
    device         : str   - "cpu" | "gpu"  (selects routing-key group)
    budget         : int   - max # `__call__` invocations allowed
    runtime_limit  : int   - max seconds allowed since **first** call
    retries        : int   - how many retry passes for timed-out items
    """

    def __init__(self,
                 plugin: str,
                 timeout: int,
                 batch_size: int = 128,
                 concurrency: int = 1,
                 *,
                 device: str      = "cpu",
                 budget: int      = 10_000,
                 runtime_limit: int | None = None,
                 retries: int     = 0,
                 check_lmt: bool  = True):

        group = "benchmark_score_gpu" if device == "gpu" else "benchmark_score"
        self.rpc = BatchScorer(
            exchange=EXCHANGE,
            routing_key=f"request.{group}.score.{plugin}.internal.internal",
            timeout=timeout,
            batch_size=batch_size,
            concurrency=concurrency,
        )

        # limits / accounting ------------------------------------------------
        self.budget        = int(budget)
        self.runtime_limit = runtime_limit
        self.retries       = retries
        self.used          = 0
        self.check_lmt     = check_lmt
        self.start_time: float | None = None

        # bookkeeping -------------------------------------------------------
        self.cache: Dict[str, float] = {}
        self.records: list[dict]     = []      # [{ts,item,score}, …]

    # ───────────────────────────── helpers ────────────────────────────
    def _check_limits(self, n_new: int = 1) -> None:
        """Raise `RuntimeError` if budget or wall-clock limit exceeded."""
        # budget ------------------------------------------------------------
        if self.used > self.budget:
            raise RuntimeError("RPCScore budget exhausted "
                               f"({self.used}/{self.budget})")

        # runtime -----------------------------------------------------------
        if self.runtime_limit is not None:
            now = time.time()
            if self.start_time is None:           # first ever call
                self.start_time = now
            elif now - self.start_time > self.runtime_limit:
                raise RuntimeError("RPCScore runtime limit exceeded "
                                   f"({int(now-self.start_time)} s)")

    def _stash_records(self,
                       smiles: List[str],
                       scores: List[float | None],
                       timestamps: List[float],
                       ) -> None:
        """Append per-item log records (ts, SMILES, score)."""
        for s, sc, ts in zip(smiles, scores, timestamps):
            self.records.append({"ts": ts, "item": s,
                                 "score": sc if sc is not None else -np.inf})

    # ───────────────────────── public API (callable) ──────────────────
    def __call__(self, smiles: List[str]) -> List[float]:
        """
        Vectorised scoring - pass a *list* of SMILES, get list[float].

        *   cached results are returned instantly  
        *   new SMILES are sent in **batched** RPC calls  
        *   timed-outs / failures → `-np.inf`
        """
        
        unlist = False
        if type(smiles) == str:
            smiles = [smiles]
            unlist = True 

        # 1.  limits check (counts only *new* SMILES to be evaluated)
        uniq_missing = {s for s in smiles if s not in self.cache}
        if self.check_lmt:
            self._check_limits(len(uniq_missing))

        # 2.  build payloads for unseen SMILES -----------------------------
        payloads: list[dict] = [
            {"item_data": {"item": s}} for s in uniq_missing
        ]
        payload_timestamps = {}

        # 3.  fetch scores (with retry logic) ------------------------------
        if payloads:
            remaining = payloads
            for attempt in range(self.retries + 1):
                results, timestamps = self.rpc.score_batch(remaining)
                next_round = []
                for pld, sc, ts in zip(remaining, results, timestamps):
                    smi = pld["item_data"]["item"]
                    payload_timestamps[smi] = ts
                    if sc is None:
                        next_round.append(pld)            # retry
                    else:
                        self.cache[smi] = sc
                remaining = next_round
                if not next_round:                        # all done
                    break

            # anything still missing after retries → -inf
            for pld in remaining:
                self.cache[pld["item_data"]["item"]] = -np.inf

            # update usage counters / records
            scored_smiles = [pld["item_data"]["item"] for pld in payloads]
            scored_vals   = [self.cache[s] for s in scored_smiles]
            scored_ts = [payload_timestamps[s] for s in scored_smiles]
            self._stash_records(scored_smiles, scored_vals, scored_ts)
            self.used += len(payloads)

        # 4.  assemble output in original order ----------------------------
        output = [self.cache.get(s, -np.inf) for s in smiles]
        if unlist:
            output = output[0]
        return output

    # ───────────────────────── housekeeping ───────────────────────────
    def close(self):
        self.rpc.close()

