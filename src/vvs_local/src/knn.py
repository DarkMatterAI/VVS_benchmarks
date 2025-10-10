"""
Light-weight wrapper around a USearch HNSW index that mimics the public
`run()` signature of `BBKNN`.

Returned objects are *schema-compatible* with the BB-KNN implementation:

    products_df : DataFrame
        ┌───────────┬─────────┬───────┐
        │ query_idx │ result  │ rank  │
        └───────────┴─────────┴───────┘
        · one row per (query, neighbour) pair
        · **rank** is 0-indexed

    prod_emb : torch.Tensor  -  `[n_unique, dim]`  (CPU, float32)
        Embeddings for every *unique* `result` in the same order as
        `products_df["result"].unique()`
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Dict

import duckdb, torch, numpy as np, pandas as pd
from usearch.index import Index

from .constants import console 

EmbeddingsT = torch.Tensor


class KNN:
    # ──────────────────────────────────────────────────────────────────
    def __init__(
        self,
        *,
        index:   Index,          # restored USearch index
        db_path: str,
        db_tbl:  str,
    ):
        self.index   = index
        self.db_tbl  = db_tbl
        # one shared read-only DuckDB connection
        self.con     = duckdb.connect(db_path, read_only=True)

    # -----------------------------------------------------------------
    def _smiles_lookup(self, np_keys: np.ndarray) -> Dict[int, str]:
        """
        Fetch SMILES for a *set* of row-ids using one SQL `IN (…)` call.
        """
        keys_unique = np.unique(np_keys)
        if keys_unique.size == 0:
            return {}

        placeholders = ",".join(map(str, keys_unique))
        query = f"""
            SELECT rowid, item
            FROM {self.db_tbl}
            WHERE rowid IN ({placeholders})
        """
        rows = self.con.execute(query).fetchall()
        return {int(rid): smi for rid, smi in rows}

    # -----------------------------------------------------------------
    def close(self):
        pass
    
    def run(
        self,
        queries: torch.Tensor,          # [B, dim]  (already compressed)
        *,
        k_nn: int,
        embed_products: bool = True,
    ) -> Tuple[pd.DataFrame, Optional[EmbeddingsT]]:
        
        queries_np = queries.detach().cpu().numpy()
        console.log(f"[cyan]=== KNN: retrieval on {queries.size(0)} queries")

        # USearch returns .keys (ids) & .counts (actual neighbours per row)
        res = self.index.search(queries_np, count=k_nn)
        # keys   : np.ndarray = res.keys          # (B, k_nn) int64
        # counts : np.ndarray = res.counts        # (B,)      int32

        if queries_np.shape[0]>1:
            keys   : np.ndarray = res.keys
            counts : np.ndarray = res.counts
        else:
            keys   : np.ndarray = res.keys[None]
            counts : np.ndarray = np.array([res.keys.shape[0]])

        # -----------------------------------------------------------------
        # Build DataFrame rows  (query_idx, result_smi, rank)
        # -----------------------------------------------------------------
        # slice keys per row according to counts to ignore padding −1
        mask = np.arange(k_nn)[None, :] < counts[:, None]
        valid_keys = keys[mask]                # flattened (N_total,) ids
        query_idx  = np.repeat(np.arange(len(queries_np)), counts)
        rank_vals  = np.tile(np.arange(k_nn), len(queries_np)).reshape(mask.shape)[mask]

        smi_map = self._smiles_lookup(valid_keys)
        smiles  = [smi_map.get(k, "") for k in valid_keys]

        products_df = pd.DataFrame(
            {"query_idx": query_idx.astype(int),
             "result":    smiles,
             "rank":      rank_vals.astype(int)}
        )

        # -----------------------------------------------------------------
        # Collect unique neighbour ids & their embeddings (optional)
        # -----------------------------------------------------------------
        prod_emb: Optional[EmbeddingsT] = None
        if embed_products:
            uniq_keys, inverse = np.unique(valid_keys, return_inverse=True)
            emb_np = self.index.get(uniq_keys)[inverse]
            prod_emb = (
                torch.from_numpy(emb_np.astype(np.float32))
                      .to("cpu")
            )

        return products_df, prod_emb
