"""
Assemble random pairs of Enamine building blocks using the
two-building-block reactions stored in
blob_store/internal/processed/enamine/enamine_id_to_reaction.pkl
and write a CSV at
blob_store/internal/training_datasets/enamine_assembled.csv
"""
from __future__ import annotations
import os, random, multiprocessing, pickle as pkl
from pathlib import Path
from collections import defaultdict

import pandas as pd
import duckdb
from rdkit import Chem
from rdkit.Chem import AllChem
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeRemainingColumn

console = Console()

# ─── paths & params ─────────────────────────────────────────────
BLOB         = Path(os.environ.get("BLOB_STORE", "/code/blob_store")).resolve()
ENAMINE_CSV  = BLOB / "internal" / "processed" / "enamine" / "data.csv"
REACTION_PKL = BLOB / "internal" / "processed" / "enamine" / "enamine_id_to_reaction.pkl"

OUT_DIR      = BLOB / "internal" / "training_datasets" / "enamine_assembled"
OUT_CSV      = OUT_DIR / "enamine_assembled.csv"
OUT_DB       = OUT_DIR / "enamine_assembled.db"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = 50_000_000       # desired number of assembled molecules
N_SAMPLE    = 400              # random partners per seed BB
TOP_K       = 3                # max products per (bb1,bb2,reaction_ids)
CPU_COUNT   = multiprocessing.cpu_count()

random.seed(42)

# ─── load reaction templates ────────────────────────────────────
if not REACTION_PKL.exists():
    console.log(f"[red]✗ missing {REACTION_PKL}")
    raise SystemExit(1)

with open(REACTION_PKL, "rb") as f:
    id_to_smarts: dict[int, str] = pkl.load(f)

REACTIONS = {}
for rid, smarts in id_to_smarts.items():
    rxn = AllChem.ReactionFromSmarts(smarts)
    rxn.Initialize()
    REACTIONS[rid] = {"reaction": rxn,
                      "reactants": list(rxn.GetReactants())}

console.log(f"Loaded {len(REACTIONS):,} two-BB reactions")

# # ─── chemistry helpers ─────────────────────────────────────────
def run_reaction(mol1, mol2, rxn_dict):
    rxn, (r1, r2) = rxn_dict["reaction"], rxn_dict["reactants"]
    prods = set()
    for a, b in ((mol1, mol2), (mol2, mol1)):
        if a.HasSubstructMatch(r1) and b.HasSubstructMatch(r2):
            for plist in rxn.RunReactants((a, b)):
                for p in plist:
                    prods.add(Chem.MolToSmiles(Chem.RemoveHs(p)))
    return prods

def react_pair(seed_row, df, n=N_SAMPLE, k=TOP_K):
    id1, smi1 = seed_row["index"], seed_row["item"]
    mol1 = Chem.AddHs(Chem.MolFromSmiles(smi1))
    results = defaultdict(list)

    for _, row2 in df.sample(n=n).iterrows():
        id2, smi2 = row2["index"], row2["item"]
        mol2 = Chem.AddHs(Chem.MolFromSmiles(smi2))
        for rid, rxn_dict in REACTIONS.items():
            for p in run_reaction(mol1, mol2, rxn_dict):
                key = f"{p}__{min(id1,id2)}__{max(id1,id2)}"
                results[key].append(rid)

    if not results:
        return pd.DataFrame()
    
    rows = []
    for key, rxns in results.items():
        prod, bb1, bb2 = key.split("__")
        rows.append(
            {
                "product": prod,
                "bb1_id": int(bb1),
                "bb2_id": int(bb2),
                # store as tuple for hashing
                "reaction_ids": tuple(sorted(rxns)),
            }
        )

    if not rows:
        return pd.DataFrame()

    out = (
        pd.DataFrame(rows)
        .sample(frac=1)
        .groupby(["bb1_id", "bb2_id", "reaction_ids"])
        .head(k)
    )

    # convert back to list for JSON-style CSV cell
    out["reaction_ids"] = out["reaction_ids"].apply(list)
    return out

# ─── main ───────────────────────────────────────────────────────
def main():
    # ----- early exits ------------------------------------------------------
    if not ENAMINE_CSV.exists():
        console.log(f"[red]Missing building blocks CSV: {ENAMINE_CSV}")
        return

    append_mode = OUT_CSV.exists()
    total = 0
    if append_mode:
        # count existing rows (skip header) ---------------------------------
        with OUT_CSV.open() as f:
            total = sum(1 for _ in f) - 1        # header line

        if total >= TARGET_SIZE:
            console.log(f"[green]✓ {OUT_CSV.name} already ≥ target - skipping")
            return
        console.log(f"[yellow]Resuming CSV ({total:,}/{TARGET_SIZE:,})")

    # ----- DB setup (may or may not exist) ----------------------------------
    db_con = None
    table_created = False
    if OUT_DB.exists():
        console.log("🗄  Found existing DuckDB - appending batches")
        db_con = duckdb.connect(str(OUT_DB))
        table_created = True

    # ----- load BBs ---------------------------------------------------------
    console.rule("[bold]Generating Enamine-Assembled dataset")
    df_bb = pd.read_csv(str(ENAMINE_CSV)).reset_index()

    header_written = append_mode
    csv_mode = "a" if append_mode else "w"
    with OUT_CSV.open(csv_mode) as writer, Progress(
        SpinnerColumn(), 
        "[progress.description]{task.description}",
        TimeRemainingColumn(), 
        console=console
    ) as prog:
        task = prog.add_task("assembling", total=TARGET_SIZE, completed=total)

        while total < TARGET_SIZE:
            seeds = random.sample(range(len(df_bb)), k=min(2000, len(df_bb)))
            seed_df = df_bb.iloc[seeds]

            with multiprocessing.Pool(CPU_COUNT) as pool:
                batches = pool.starmap(
                    react_pair,
                    [(row, df_bb) for _, row in seed_df.iterrows()],
                )

            batch_df = pd.concat([b for b in batches if not b.empty], ignore_index=True)
            if batch_df.empty:
                continue

            # ----- CSV append ------------------------------------------------
            batch_df.to_csv(writer, header=not header_written, index=False)
            header_written = True

            # ----- DB append/creation ---------------------------------------
            if db_con:
                db_con.register("tmp_batch", batch_df)
                if not table_created:
                    db_con.execute("CREATE TABLE enamine AS SELECT * FROM tmp_batch")
                    table_created = True
                else:
                    db_con.execute("INSERT INTO enamine SELECT * FROM tmp_batch")
                db_con.unregister("tmp_batch")

            total += len(batch_df)
            prog.update(task, completed=total)

    # create DB if it didn't exist yet ---------------------------------------
    if not OUT_DB.exists():
        console.log("🗄  Creating DuckDB from CSV (first run)")
        con = duckdb.connect(str(OUT_DB))
        con.execute(f"CREATE TABLE enamine AS SELECT * FROM read_csv_auto('{OUT_CSV}')")
        con.execute("CREATE INDEX idx_prod ON enamine(product)")
        con.close()

    elif db_con:
        db_con.close()

    console.rule(f"[bold green]Done → {OUT_CSV.name} ({total:,} rows)  +  {OUT_DB.name}")

if __name__ == "__main__":
    main()
