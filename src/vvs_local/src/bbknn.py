from __future__ import annotations

from itertools import product
from typing    import Sequence, Dict, List, Tuple, Optional 

import pandas as pd 
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding

from .reaction_assembly import ReactionAssembly
from .constants import console 

# ────────────────────────────── type aliases
IdxPair     = Tuple[int, int]              # (bb1_idx, bb2_idx)
RankPair    = Tuple[int, int]              # (bb1_rank, bb2_rank)
NeighbourT  = torch.Tensor                 # [B, 2, k]  (long, cpu)
EmbeddingsT = torch.Tensor                 # [N, d]

# ╭──────────────────────── mini utilities ────────────────────────────╮
def _chunk(seq: Sequence, n: int) -> List[Sequence]:
    return [seq[i : i + n] for i in range(0, len(seq), n)]

def _gather_by_unique(
    seq: Sequence[str]
) -> tuple[list[str], dict[str, list[int]]]:
    """
    Split *seq* into the list of unique strings **in first-appearance
    order** and a mapping string → list[row-indices].

    Returns
    -------
    uniq   : list[str]             (deduplicated)
    index  : dict[str, list[int]]  ( `seq[i]` ∈ uniq ⇒ i ∈ index[seq[i]] )
    """
    uniq, index = [], {}
    for i, s in enumerate(seq):
        if s not in index:
            uniq.append(s)
            index[s] = []
        index[s].append(i)
    return uniq, index
# ╰─────────────────────────────────────────────────────────────────────╯


class BBKNN:
    """
    **Single-size** BB-KNN wrapper.

    A single instance handles exactly one (input_size → output_size) pair.

    Parameters
    ----------
    tok, coll, embed_model      :   *pre-loaded* encoder assets
    decomposer                  :   Enamine-decomposer (pre-loaded)
    bb_table                    :   Enamine BB embedding matrix [N, output_size] (CPU tensor)
    bb_smiles                   :   list[str] - same order as `bb_table`
    assembly                    :   ReactionAssembly instance
    input_size, output_size     :   int
    device                      :   torch device string
    """

    # --------------------------------------------------------------------
    def __init__(
        self,
        *,
        tok:              AutoTokenizer,
        collator:         DataCollatorWithPadding,
        encoder:          AutoModel,
        decomposer:       AutoModel,
        bb_table:         EmbeddingsT,
        bb_smiles:        List[str],
        assembly:         ReactionAssembly,
        input_size:       int,
        output_size:      int,
        device:           str = "cpu",
    ):
        self.tok        = tok
        self.coll       = collator
        self.encoder    = encoder.eval().to(device)
        self.decomposer = decomposer.eval().to(device)

        self.bb_table   = F.normalize(bb_table, 2, -1).to(device)  # [N,d]
        self.bb_smiles  = bb_smiles                                 # list[str]
        self.assembly   = assembly

        self.in_size    = int(input_size)
        self.out_size   = int(output_size)
        self.device     = device

    # ╭────────────────── embedding / compress helpers ─────────────────╮
    @torch.no_grad()
    def _encode(self, smiles: Sequence[str], batch: int = 1024) -> EmbeddingsT:
        out = []
        for chunk in _chunk(smiles, batch):
            toks = self.coll(self.tok(chunk, truncation=True, max_length=256))
            toks = {k: v.to(self.device) for k, v in toks.items()}
            last = self.encoder(**toks, output_hidden_states=True).hidden_states[-1]
            mask = toks["attention_mask"]
            emb  = (last * mask.unsqueeze(-1)).sum(1) / mask.sum(-1, keepdim=True)
            out.append(emb)
        return torch.cat(out)                               # on device

    @torch.no_grad()
    def _compress(self, emb: EmbeddingsT) -> EmbeddingsT:    # [B, in_size]
        z = self.decomposer.compress(emb, [self.in_size])[self.in_size]
        return z                                            # on device

    # ╭──────────────── neighbour retrieval & reaction build─────────────╮
    @torch.no_grad()
    def _retrieve_bb(self, z: EmbeddingsT, k: int) -> NeighbourT:
        """
        z       : [B, in_size] (device) - compressed query vectors
        returns : [B, 2, k]   (cpu long) - BB indices per position
        """
        de = self.decomposer.decompose({self.in_size: z.to(self.device)}, 
                                       [self.out_size])
        bbq = F.normalize(de[self.out_size][0], 2, -1)          # [B,2,d]
        idx = (bbq @ self.bb_table.T).topk(k, dim=-1).indices   # device
        return idx.cpu().long()                                 # to cpu
    
    @torch.no_grad()
    def _embed_unique(
        self,
        smiles: Sequence[str],
        batch:  int = 1024,
    ) -> EmbeddingsT:
        """
        Compute compressed embeddings **once per unique SMILES** and scatter
        them back so the output matches *smiles* order (duplicates included).

        Returns
        -------
        torch.Tensor  -  shape ``[len(smiles), in_size]``  (CPU)
        """
        if not smiles:
            return torch.empty(0, self.in_size)

        uniq, idx_map = _gather_by_unique(smiles)          # deduplicate
        console.log(f"[cyan]=== BBKNN: embedding {len(uniq)} unique items")
        emb_uniq = self._compress(self._encode(uniq, batch)) \
                       .cpu()                              # [U, in_size]

        out = torch.empty(len(smiles), self.in_size, dtype=emb_uniq.dtype)
        for j, s in enumerate(uniq):
            out[idx_map[s]] = emb_uniq[j]                  # scatter
        return out

    # ╭────────────────────────── public API  ───────────────────────────╮
    def close(self):
        self.assembly.close()

    def run(
        self,
        queries: torch.Tensor,              # shape [B, in_size]  - **compressed
                                            # query embeddings**
        *,
        k_nn: int,                          # nearest-neighbour count per BB-slot
        embed_products: bool = True         # embed products 
    ) -> Tuple[pd.DataFrame, Optional[EmbeddingsT]]:
        """
        Enumerate reaction products for a batch of **pre-embedded queries**
        using BB-KNN and the ReactionAssembly helper.

        Workflow
        --------
        1. **Decompose** each query embedding into two BB vectors and fetch
           the **top-*k* nearest neighbours** for each building-block slot
           (cosine similarity against `self.bb_table`).

        2. Build the cross-product of retrieved BB pairs *(bb1 x bb2)* for
           every query → de-duplicate across the whole batch.

        3. **React** every unique BB pair in parallel (via
           `self.assembly.react`); scatter the resulting product SMILES back
           to the originating queries.

        4. **Embed all unique products** (encode → compress) so downstream
           code can rank/cluster them without another model call.

        Returns
        -------
        products_df : pandas.DataFrame
            One row per *(query, product)* pair — duplicates removed.
            Columns
              - ``query_idx``  :  row index within the input batch  
              - ``result``     :  product SMILES  
              - ``bb1_idx``    :  Enamine row-index of first BB  
              - ``bb2_idx``    :  Enamine row-index of second BB  
              - ``bb1_rank``   :  K-retrieval rank of first BB
              - ``bb2_rank``   :  K-retrieval rank of second BB

        prod_emb : torch.Tensor  -  shape ``[n_products, in_size]`` (*CPU*)
            Compressed embeddings for every **unique** product in
            ``products_df["result"]`` (same order).

        Notes
        -----
        * `queries` must already be **compressed to `self.in_size`**.  If you
          have raw 768-d RoBERTa embeddings, call ``bbknn._compress()`` first.
        * The function is **deterministic** given identical `queries`,
          `k_nn`, and ReactionAssembly settings.
        """
        B = queries.shape[0]
        console.log(f"[cyan]=== BBKNN: retrieval on {queries.size(0)} queries")
        idx_bb = self._retrieve_bb(queries, k_nn)            # [B,2,k]

        rec_entries = []
        for q_idx in range(B):
            bb1_ids = idx_bb[q_idx, 0].tolist()
            bb2_ids = idx_bb[q_idx, 1].tolist()
            for r1_rank, r1 in enumerate(bb1_ids):
                for r2_rank, r2 in enumerate(bb2_ids):
                    rec_entries.append(
                        (q_idx, r1, r2, r1_rank, r2_rank)
            )

        # deduplicate BB pairs for reaction assembly --------------------
        uniq_pairs: Dict[IdxPair, List[int]] = {}
        uniq_ranks: Dict[IdxPair, List[RankPair]] = {}
        for q_idx, r1, r2, r1_rank, r2_rank in rec_entries:
            r_to_rank = {r1:r1_rank, r2:r2_rank}
            key   = tuple(sorted((r1, r2)))
            ranks = tuple([r_to_rank[i] for i in key])
            uniq_pairs.setdefault(key, []).append(q_idx)
            uniq_ranks.setdefault(key, []).append(ranks)

        key_list  = list(uniq_pairs.keys())    # fixed order
        pair_idxs = list(range(len(key_list)))
        pair_list = [(self.bb_smiles[a], self.bb_smiles[b]) for (a, b) in key_list]

        # 3. parallel reaction enumeration ------------------------------
        console.log(f"[cyan]=== BBKNN: assembling {len(pair_list)} pairs")
        reacts = self.assembly.react(pair_idxs, pair_list)
        #  reacts : list[{"idx":pair_idx,"result":smiles}]

        # scatter back to queries + build output DataFrame --------------
        rows = []
        for rec in reacts:
            pair_idx   = rec["idx"]
            prod_smi   = rec["result"]
            bb1_idx, bb2_idx = key_list[pair_idx]
            for q_idx, ranks in zip(uniq_pairs[(bb1_idx, bb2_idx)], 
                                    uniq_ranks[(bb1_idx, bb2_idx)]):
                bb1_rank, bb2_rank = ranks
                rows.append({"query_idx": q_idx,
                             "result":    prod_smi,
                             "bb1_idx":   bb1_idx,
                             "bb2_idx":   bb2_idx,
                             "bb1_rank":  bb1_rank,
                             "bb2_rank":  bb2_rank,
                             })

        products_df = pd.DataFrame(rows)
        # drop duplicates within query embedding
        products_df = (products_df.drop_duplicates(["query_idx", "result"])
                                  .reset_index(drop=True))

        # 4. embed all unique products ----------------------------------
        prod_emb = None 
        if embed_products:
            prod_emb = self._embed_unique(products_df["result"].tolist())

        return products_df, prod_emb
