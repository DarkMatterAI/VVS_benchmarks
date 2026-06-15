"""
run_demo.py
Minimal BBKNN demo: retrieve synthesizable analogues for a set of query SMILES
using a toy building block library. Run AFTER prepare_demo.py.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from bbknn.bbknn import BBKNN, eval_bbknn_similarity
from bbknn.constants import BLOB, EMB_MODEL_NM, DECOMP_MODEL_NM, DEVICE

DEMO_DIR      = Path(__file__).parent
DEMO_BB_CSV   = DEMO_DIR / "demo_bbs.csv"
DEMO_SMI_CSV  = DEMO_DIR / "demo_smiles.csv"
BB_EMB_PATH   = BLOB / "demo_artifacts" / "bb_embeddings.pt"
OUT_PATH      = BLOB / "results" / "demo_results.csv"

SIZE = 256
K    = 5


def main():
    queries = pd.read_csv(DEMO_SMI_CSV)
    assert "item" in queries.columns, "demo_smiles.csv must contain an 'item' column"
    query_smiles = queries["item"].tolist()

    print("Loading models \u2026")
    bbknn = BBKNN(EMB_MODEL_NM, DECOMP_MODEL_NM, BB_EMB_PATH, DEMO_BB_CSV, DEVICE)

    df = bbknn.smiles_query_multiscale(
        query_smiles,
        k=K,
        input_sizes=[SIZE],
        output_sizes=[SIZE],
        reaction_cpus=4,
    )

    # Optional: attach cosine / Tanimoto similarity to the query
    df = eval_bbknn_similarity(df, bbknn).drop(columns=["query_idx", "result_idx"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\u2713 wrote {len(df):,} results \u2192 {OUT_PATH}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()