from __future__ import annotations

from collections import defaultdict
from typing      import Dict, Tuple, List, Optional

import torch
import pandas as pd
import numpy as np 

from .constants import console 
from .bbknn     import BBKNN
from .score_rpc import RPCScore
from .gradient  import cosine_gradient


EmbeddingsT = torch.Tensor   # convenience alias


# ╭───────────────────────────── main optimiser ─────────────────────────────╮
class VVS:
    """
    Variational Virtual Screen (VVS) - iterative BB-KNN + policy-gradient
    optimiser.

    Parameters
    ----------
    bbknn        : BBKNN
        *Shared* instance - no duplicate model weights.

    scorer       : RPCScore
        Any callable that maps ``list[smiles] ➜ list[float]`` and enforces
        budget / runtime limits internally.

    update_type  : {"top1", "standard"}, default="top1"
        * ``"standard"``  - hill climbing around the original query.
        * ``"top1"``      - re-centre gradient on **best product embedding**
          (highest score) before computing ∇.
          
    norm_scaling : {True, False}, default=False
        If True, gradient step queries are scaled to have the same norm as 
        the original query
    """

    # ────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        bbknn: BBKNN,
        scorer: RPCScore,
        update_type: str = "top1",
        norm_scaling: bool = False
    ) -> None:
        if update_type not in {"top1", "standard"}:
            raise ValueError("update_type must be 'top1' or 'standard'")
        self.bbknn   = bbknn
        self.scorer  = scorer
        self.update_type = update_type
        self.norm_scaling = norm_scaling

        # last-iteration artefacts (for inspection)
        self.result_dfs: List[pd.DataFrame]      = []
        self.last_product_dict: Optional[Dict]   = None
        self.last_update_dict:  Optional[Dict]   = None

    # ╭──────────────────────────── helpers ────────────────────────────╮
    @staticmethod
    def _expand_queries(
        base_q: EmbeddingsT,              # [B, d]
        base_i: EmbeddingsT,              # [B]
        lrs:    EmbeddingsT,              # [L]  learning-rates
        grads:  Optional[EmbeddingsT],    # [B, d] or None
        scale:  bool, 
    ) -> Tuple[EmbeddingsT, torch.Tensor]:
        """
        Generate the actual query set for BB-KNN:

        * if ``grads`` is *None* → just return the original queries.
        * else → create *(B x L)* perturbed queries:  
          `q' = q - lr · grad`

        Returns ``(expanded_queries, orig_idx)`` where ``orig_idx`` maps every
        expanded row to its originating base index (needed for bookkeeping).
        """
        if grads is None:
            return base_q, base_i
        
        # broadcast: [B, 1, d] - [  , L,  ]   -> [B, L, d]
        expanded = base_q[:, None, :] - lrs[None, :, None] * grads[:, None, :]
        if scale:
            base_norm = torch.norm(base_q, p=2, dim=-1)
            expanded_norm = torch.norm(expanded, p=2, dim=-1)
            norm_scaling = base_norm[:,None].div(expanded_norm)
            expanded = expanded * norm_scaling.unsqueeze(-1)
        orig_idx = base_i.repeat_interleave(lrs.numel())
        return expanded.reshape(-1, base_q.size(1)), orig_idx

    # ────────────────────────────────────────────────────────────────────
    def _retrieve_and_score(
        self,
        queries: EmbeddingsT,          # [N, d]
        orig_idx: torch.Tensor,        # [N]  maps -> base query row
        k_nn: int,
        skip_score: bool = False,
    ) -> Dict:
        """
        1. BB-KNN enumeration → products (+ their compressed embeddings)
        2. call `self.scorer` unless *skip_score* is True
        3. build & return a *product_dict* keyed by ``"{baseIdx}_{smiles}"``.
        """
        # 1. BB-KNN enumeration
        prod_df, prod_emb = self.bbknn.run(queries, k_nn=k_nn)  # prod_emb on CPU

        # 2. scoring -----------------------------------------------------
        if skip_score:
            scores = [0.0] * len(prod_df)
        else:
            scores = self.scorer(prod_df["result"].tolist())

        prod_df = prod_df.assign(score=scores)
        self.result_dfs.append(prod_df)
        prod_df = prod_df[~(np.isinf(prod_df["score"]))].reset_index(drop=True)
        
        # 3. aggregate duplicates across (base idx, product) -------------
        product_dict: Dict[str, Dict] = {}
        score_np   = np.asarray(prod_df["score"])
        res_np     = prod_df["result"].to_numpy()

        for row_id, q_id in enumerate(prod_df["query_idx"].to_numpy()):
            base_i  = orig_idx[q_id].item()
            smi     = res_np[row_id]
            key     = f"{base_i}_{smi}"

            entry = product_dict.setdefault(
                key,
                {
                    "idx": base_i,
                    "query_idx": q_id,
                    "result": smi,
                    "score": score_np[row_id],
                    "embedding": prod_emb[row_id],
                    "count": 0,
                },
            )
            entry["count"] += 1
            # keep highest score seen for this (idx, smi)
            if score_np[row_id] > entry["score"]:
                entry["score"] = score_np[row_id]
                entry["embedding"] = prod_emb[row_id]

        self.last_product_dict = product_dict
        return product_dict

    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def _build_update_dict(
        base_q: EmbeddingsT,
        product_dict: Dict,
    ) -> Dict[int, Dict[str, EmbeddingsT]]:
        """
        Collate per-base-query tensors needed for gradient computation.

        Returns
        -------
        {base_idx: {"query": [1,d], "embeddings": [n,d], "scores": [n]}}
        """
        coll: Dict[int, Dict[str, List]] = defaultdict(
            lambda: {"embeddings": [], "scores": []}
        )
        for entry in product_dict.values():
            i = entry["idx"]
            coll[i]["embeddings"].append(entry["embedding"])
            coll[i]["scores"].append(entry["score"])

        update = {}
        for i, buf in coll.items():
            update[i] = {
                "query":       base_q[i][None],
                "embeddings":  torch.stack(buf["embeddings"]),
                "scores":      torch.as_tensor(buf["scores"], dtype=torch.float32),
            }
        return update

    # ────────────────────────────────────────────────────────────────────
    def _apply_policy_update(
        self,
        update_dict: Dict[int, Dict[str, EmbeddingsT]],
    ) -> Tuple[EmbeddingsT, EmbeddingsT]:
        """
        From the collected product embeddings/scores compute a *new* query set
        + their gradients.

        Returns ``(new_queries, gradients)`` with shape ``[B, d]`` each.
        """
        new_queries, new_idxs, grads = [], [], []

        for idx,data in update_dict.items():
            q   = data["query"].cpu()               # [1,d]
            emb = data["embeddings"]                # [N,d]
            scr = data["scores"]                    # [N]
            assert q.shape[0]>0

            if self.update_type == "top1":
                q = emb[scr.argmax()].unsqueeze(0)  # [1,d]

            assert q.shape[0]>0
            grad = cosine_gradient(q, emb, scr)     # [d]
            new_queries.append(q)
            new_idxs.append(len(new_queries)-1)
            grads.append(grad)

        return torch.cat(new_queries, 0), torch.tensor(new_idxs), torch.stack(grads, 0)
    
    def close(self):
        self.scorer.close()
        self.bbknn.close()

    # ╭────────────────────────── high-level loops ───────────────────────╮
    def search_iteration(
        self,
        base_queries: EmbeddingsT,          # [B, d], FloatTensor
        base_idxs:    EmbeddingsT,          # [B], LongTensor
        lrs:          EmbeddingsT,          # [L], FloatTensor
        k_nn:         int,
        prev_grads:   Optional[EmbeddingsT] = None,
        *,
        skip_score:   bool = False,
    ) -> Tuple[EmbeddingsT, EmbeddingsT]:
        """
        One optimisation step:
          1. perturb queries  (if `prev_grads` is given)
          2. enumerate + score products
          3. compute new query embeddings and gradients
        """
        # 1. expand
        queries, orig_idx = self._expand_queries(base_queries, 
                                                 base_idxs, 
                                                 lrs, 
                                                 prev_grads, 
                                                 self.norm_scaling)

        # 2. enumerate & score
        prod_dict = self._retrieve_and_score(
            queries, orig_idx, k_nn, skip_score=skip_score
        )
        if not prod_dict:
            return False, None

        # 3. gradient step
        upd_dict = self._build_update_dict(base_queries, prod_dict)
        self.last_update_dict = upd_dict
        if not upd_dict:
            return False, None 
        
        updated = self._apply_policy_update(upd_dict)
        return True, updated

    # ────────────────────────────────────────────────────────────────────
    def search(
        self,
        init_queries: EmbeddingsT,          # [B, d]
        iterations:   int,
        lrs:          EmbeddingsT,          # [L]
        k_nn:         int,
        *,
        skip_score:   bool = False,
        check_lmt:    bool = False,
    ) -> Tuple[EmbeddingsT, EmbeddingsT]:
        """
        Run *iterations* optimisation steps and return the **final** query
        embeddings and their gradients from the last iteration.
        """
        queries, grads = init_queries, None
        idxs = torch.arange(queries.size(0))
        for step in range(iterations):
            if check_lmt:
                self.scorer._check_limits()

            console.log(f"[magenta]=== VVS iteration {step+1}/{iterations}")
            cont, outputs = self.search_iteration(
                queries, idxs, lrs, k_nn, grads, skip_score=skip_score
            )
            if not cont:
                console.log(f"[red]=== VVS iteration {step+1}/{iterations} produced 0 results")
                queries, idxs, grads = None, None, None
                break
                
            queries, idxs, grads = outputs
        return queries, idxs, grads
