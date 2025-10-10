"""
bbknn.py
─────────────────────────────────────────────────────────────────────
Core implementation of **B B-K N N** (“building-block K-nearest
neighbours”).  The class:

* embeds query SMILES with a transformer encoder
* compresses to multiple latent sizes with the Enamine-decomposer
* decomposes each latent into two building-block embeddings
* retrieves top-*k* nearest neighbours from the Enamine BB library
* enumerates products by applying every 2-BB reaction template
  (parallelised over CPUs)

Public helpers
──────────────
`eval_bbknn_similarity`  - convenience to attach cosine & Tanimoto scores
                           to a reactions DataFrame.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import List, Dict, Sequence

import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding

from .reaction_utils import parallel_react, compute_fps, tanimoto_similarity
from .constants import DEVICE


# ╭─────────────────────────────────────────── helpers ─────────────────────╮
def _chunk(seq: Sequence, n: int) -> List[Sequence]:
    """Split *seq* into chunks of at most *n* items (simple list slice)."""
    return [seq[i : i + n] for i in range(0, len(seq), n)]
# ╰──────────────────────────────────────────────────────────────────────────╯


class BBKNN:
    """
    **B B-K N N** modeller.

    Parameters
    ----------
    embed_model_name : str
        Hugging-Face model name for the SMILES encoder
        (e.g. ``entropy/roberta_zinc_480m``).

    decomp_model_name : str
        Hugging-Face model name for the Enamine-decomposer.

    bb_emb_path : pathlib.Path
        Path to the *bb_embeddings.pt* tensor (look-up table).

    bb_csv_path : pathlib.Path
        Path to *processed/enamine/data.csv* (contains SMILES for every BB).

    device : str
        Torch device string (`"cuda"` or `"cpu"`).  Defaults to autodetected
        value from :data:`bbknn.constants.DEVICE`.
    """

    # ╭──────────────────────── init / model loading ───────────────────────╮
    def __init__(
        self,
        embed_model_name: str,
        decomp_model_name: str,
        bb_emb_path: Path,
        bb_csv_path: Path,
        device: str = DEVICE,
    ) -> None:
        self.embed_model_name  = embed_model_name
        self.decomp_model_name = decomp_model_name
        self.bb_emb_path       = bb_emb_path
        self.bb_csv_path       = bb_csv_path
        self.device            = device
        self._load_models()

    def _load_models(self) -> None:
        """Download / load all transformer weights & reference tables."""
        # ─────────────────── encoder ────────────────────
        self.tok  = AutoTokenizer.from_pretrained(self.embed_model_name)
        self.coll = DataCollatorWithPadding(self.tok, return_tensors="pt")

        self.embed_model = (
            AutoModel.from_pretrained(self.embed_model_name,
                                      add_pooling_layer=False)
            .to(self.device)
            .eval()
        )

        # ─────────────────── decomposer ──────────────────
        self.decomposer = (
            AutoModel.from_pretrained(self.decomp_model_name,
                                      trust_remote_code=True)
            .to(self.device)
            .eval()
        )

        # ─────────────────── BB lookup ───────────────────
        bb_state = torch.load(self.bb_emb_path, map_location="cpu")
        self.bb_lookup = nn.ModuleDict(
            {
                k.split(".")[0]: nn.Embedding.from_pretrained(v, freeze=True)
                for k, v in bb_state.items()
            }
        ).to(self.device)
        for emb in self.bb_lookup.values():        # unit-norm rows
            emb.weight.data = F.normalize(emb.weight.data, 2, -1)

        #  CSV with SMILES (for pretty output)
        self.bb_csv = pd.read_csv(self.bb_csv_path)

    # ╭──────────────────────── embeddings ───────────────────────────────╮
    @torch.no_grad
    def embed(
        self,
        smiles: Sequence[str],
        *,
        batch_size: int = 1024,
        to_cpu: bool = True,
    ) -> torch.Tensor:
        """
        Encode a list of SMILES into 768-d RoBERTa embeddings (row-wise).

        Returns
        -------
        torch.Tensor  - shape ``[len(smiles), 768]`` on *CPU* by default.
        """
        outs = []
        for batch in _chunk(smiles, batch_size):
            toks = self.coll(self.tok(batch, truncation=True, max_length=256))
            toks = {k: v.to(self.device) for k, v in toks.items()}
            last = self.embed_model(**toks,
                                    output_hidden_states=True).hidden_states[-1]
            mask = toks["attention_mask"]
            emb  = (last * mask.unsqueeze(-1)).sum(1) / mask.sum(-1, keepdim=True)
            outs.append(emb if not to_cpu else emb.cpu())
        return torch.cat(outs)

    @torch.no_grad
    def compress(
        self,
        embeddings: torch.Tensor,
        input_sizes: List[int],
        *,
        to_cpu: bool = True,
    ) -> Dict[int, torch.Tensor]:
        """
        Compress *embedding* to every size in *input_sizes*.

        Returns
        -------
        Dict[size, Tensor]  - each tensor has shape ``[B, size]``.
        """
        z = self.decomposer.compress(embeddings.to(self.device), input_sizes)
        return {k: v.cpu() for k, v in z.items()} if to_cpu else z

    # ╭────────────────── decomposition + retrieval ───────────────────────╮
    @torch.no_grad
    def decompose_and_retrieve(
        self,
        embeddings: Dict[int, torch.Tensor],
        output_sizes: List[int],
        k: int,
    ) -> Dict[int, torch.Tensor]:
        """
        Decompose each *input* tensor and fetch the **top-k** nearest
        neighbours (cos-sim) from the corresponding BB table.

        Returns
        -------
        Dict[out_size, LongTensor]
            Shape ``[n_input_sizes, B, 2, k]`` — the last axis contains
            Enamine BB row-indices.
        """
        embeds_gpu = {k: v.to(self.device) for k, v in embeddings.items()}
        decomposed = self.decomposer.decompose(embeds_gpu, output_sizes)

        idx_dict: Dict[int, torch.Tensor] = {}
        for out_sz in output_sizes:
            ref      = self.bb_lookup[str(out_sz)].weight               # [N,d]
            queries  = F.normalize(decomposed[out_sz], p=2, dim=-1)     # [n_in,B,2,d]
            idx      = (queries @ ref.T).topk(k, dim=-1).indices.cpu()
            idx_dict[out_sz] = idx.long()
        return idx_dict

    # ╭──────────────────────── table helpers ─────────────────────────────╮
    def _idx_to_df(self, indices: torch.Tensor, bb_pos: int) -> pd.DataFrame:
        """
        Convert a vector of Enamine indices → pretty DataFrame
        with `rank`, `item`, `item_id` columns (prefix `bb{pos}_`).
        """
        sub = (self.bb_csv
               .iloc[indices]
               .reset_index(drop=True)
               .assign(item_id=indices))
        sub = (sub.drop(columns="external_id")
                  .reset_index()
                  .rename(columns={"index": "rank"}))
        sub.columns = [f"bb{bb_pos}_{c}" for c in sub.columns]
        return sub

    def decompose_and_retrieve_df(
        self,
        embeddings: Dict[int, torch.Tensor],
        output_sizes: List[int],
        k: int,
    ) -> Dict[str, List[pd.DataFrame]]:
        """
        Wrapper around :pymeth:`decompose_and_retrieve` that converts every
        neighbour tensor into a **list of DataFrames** - one per query
        molecule.

        Returns
        -------
        dict[str, list[pd.DataFrame]]
            Keys are ``"{input}->{output}"`` size pairs.
        """
        idx_map = self.decompose_and_retrieve(embeddings, output_sizes, k)
        tables: Dict[str, List[pd.DataFrame]] = {}

        for out_sz, idx in idx_map.items():              # idx shape [n_in,B,2,k]
            row = 0
            for in_sz in self.decomposer.config.comp_sizes:
                if in_sz not in embeddings:
                    continue
                key = f"{in_sz}->{out_sz}"
                dfs: List[pd.DataFrame] = []
                for b in range(idx.size(1)):             # iterate over batch
                    bb1 = self._idx_to_df(idx[row, b, 0], 1)
                    bb2 = self._idx_to_df(idx[row, b, 1], 2)
                    dfs.append(pd.concat([bb1, bb2], axis=1))
                tables.setdefault(key, []).extend(dfs)
                row += 1
        return tables

    # ╭──────────────────── pair enumeration helpers ──────────────────────╮
    def build_reaction_pairs(
        self,
        queries: Sequence[str],
        retrieved: Dict[str, List[pd.DataFrame]],
    ) -> List[dict]:
        """
        Cartesian-combine every BB-pair into a dict describing the
        prospective reaction *input*.
        """
        pairs: List[dict] = []
        for key, frames in retrieved.items():
            in_sz, out_sz = map(int, key.split("->"))
            for q_smi, frame in zip(queries, frames):
                bb1_rows = frame[["bb1_rank", "bb1_item", "bb1_item_id"]].to_dict("records")
                bb2_rows = frame[["bb2_rank", "bb2_item", "bb2_item_id"]].to_dict("records")
                for r1, r2 in product(bb1_rows, bb2_rows):
                    pairs.append(
                        {
                            "query": q_smi,
                            "in_size": in_sz,
                            "out_size": out_sz,
                            **r1,
                            **r2,
                            "max_rank": max(r1["bb1_rank"], r2["bb2_rank"]),
                        }
                    )
        return pairs

    # ╭────────────────────── public multiscale driver ─────────────────────╮
    def smiles_query_multiscale(
        self,
        queries: Sequence[str],
        k: int,
        input_sizes: List[int],
        output_sizes: List[int],
        *,
        reaction_cpus: int = 8,
    ) -> pd.DataFrame:
        """
        *High-level one-liner* — from SMILES queries all the way to
        enumerated reaction products.

        Workflow
        --------
        1. embed → compress (every *input_sizes*)
        2. decompose → top-*k* neighbours (every *output_sizes*)
        3. enumerate all reaction templates
        4. return a de-duplicated product table

        Returns
        -------
        pandas.DataFrame
            Columns include the query/building-block metadata plus
            the ``result`` SMILES.
        """
        print("🔢  Embedding queries")
        emb        = self.embed(queries, to_cpu=False)
        comp       = self.compress(emb, input_sizes, to_cpu=False)
        print("📐  Retrieving building blocks")
        retrieved  = self.decompose_and_retrieve_df(comp, output_sizes, k)

        print("🧮  Reacting pairs")
        pair_dicts = self.build_reaction_pairs(queries, retrieved)
        products = parallel_react(pair_dicts, reaction_cpus)
        df = (pd.DataFrame(products)
                .sort_values(["query", "in_size", "out_size", "max_rank"],
                             ascending=[True, False, False, True])
                .drop_duplicates(["query", "in_size", "out_size", "result"])
                .reset_index(drop=True))
        return df
# ╰──────────────────────────────────────────────────────────────────────────╯


# ╭──────────────────── evaluation convenience function ────────────────────╮
def eval_bbknn_similarity(df: pd.DataFrame, bbknn: BBKNN) -> pd.DataFrame:
    """
    Attach **cosine** and **Tanimoto** similarities to *df* (in-place copy).

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :pymeth:`BBKNN.smiles_query_multiscale`
        - must contain ``query`` and ``result`` columns.

    bbknn : BBKNN
        Instance with loaded encoder (re-used for embedding).

    Returns
    -------
    pandas.DataFrame
        Same shape as input, with two new float columns
        ``cosine_similarity`` and ``tanimoto_similarity``.
    """
    # unify SMILES set
    unique_smiles = list(set(df["query"]).union(df["result"]))
    smi_to_idx    = {s: i for i, s in enumerate(unique_smiles)}

    # ── embed once ───────────────────────────────────────────────────
    print("🔢  Embedding unique SMILES …")
    emb = bbknn.embed(unique_smiles)

    df = df.copy()
    df["query_idx"]  = df["query"].map(smi_to_idx)
    df["result_idx"] = df["result"].map(smi_to_idx)
    idx_pairs = torch.tensor(df[["query_idx", "result_idx"]].values, dtype=torch.long)

    # ── cosine similarity (batched) ─────────────────────────────────
    print("📐  Cosine similarity …")
    cos = []
    for batch in idx_pairs.split(2048):
        q = emb[batch[:, 0]].to(bbknn.device)
        r = emb[batch[:, 1]].to(bbknn.device)
        cos.append(F.cosine_similarity(q, r).cpu())
    df["cosine_similarity"] = torch.cat(cos).numpy()

    # ── Tanimoto (ECFP-4) ───────────────────────────────────────────
    print("🧮  Tanimoto similarity …")
    fps = compute_fps(unique_smiles)
    df["tanimoto_similarity"] = [tanimoto_similarity(pair, fps)
                                 for pair in idx_pairs.tolist()]
    
    df = (df.sort_values(["query", "in_size", "out_size", "max_rank", "cosine_similarity"],
                         ascending=[True, False, False, True, False])
                         .reset_index(drop=True))

    return df

