"""
Chunk-wise canonicalisation + DuckDB creation
"""
import gc, os, tarfile, gzip, zipfile
from pathlib import Path
from multiprocessing import Pool
from functools import partial
import pickle as pkl
import math

from chembl_webresource_client.settings import Settings
Settings.Instance().CONCURRENT_SIZE = 100
Settings.Instance().MAX_LIMIT = 50
from chembl_webresource_client.new_client import new_client

from synthemol.reactions import REAL_REACTIONS

import duckdb, pandas as pd
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeRemainingColumn, SpinnerColumn

from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover

from .paths import (
    RAW_DIR, 
    PROCESSED_DIR,
    ZINC_DIR, 
    D4_DIR, 
    ENAMINE_DIR,
    NATPROD_DIR,
    ZINC_TAR_GZ, 
    D4_GZ, 
    DATA_ZIP,
    NATPROD_ZIP,
    CHEMBL_DIR,
    CHEMBL_CSV,
)

console = Console()

# ────────────────────────────────────────────────────────────────
# Helper: generic decompressors (run once, idempotent)
# ────────────────────────────────────────────────────────────────
def _extract_if_missing(archive: Path, dest_file: Path, member: str | None = None):
    if dest_file.exists():
        return dest_file
    console.log(f"[blue]→ extracting {archive.name}")
    dest_file.parent.mkdir(parents=True, exist_ok=True)

    if archive.suffixes[-2:] == [".tar", ".gz"]:
        with tarfile.open(archive, "r:gz") as tf:
            tf.extract(member or dest_file.name, path=RAW_DIR)
    elif archive.suffix == ".gz":
        # single gzip’d csv
        with gzip.open(archive, "rb") as gz, open(dest_file, "wb") as out:
            out.write(gz.read())
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extract(member, path=RAW_DIR)
    else:
        raise ValueError(f"Unknown archive type: {archive}")
    return dest_file


# ────────────────────────────────────────────────────────────────
# RDKit helpers  (stand-alone → avoids importing your common.*)
# ────────────────────────────────────────────────────────────────
REMOVER = SaltRemover()

def canon_smile(smile: str, remove_stereo: bool = False):
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return None
        mol = REMOVER.StripMol(mol, dontRemoveEverything=True)
        if remove_stereo:
            Chem.RemoveStereochemistry(mol)
        return Chem.CanonSmiles(Chem.MolToSmiles(mol))
    except Exception:
        return None

def canonicalize_and_filter_df(df: pd.DataFrame, smile_col: str, processes=None):
    smiles = df[smile_col].tolist()
    with Pool(processes or os.cpu_count()) as p:
        cleaned = p.map(canon_smile, smiles)
    df[smile_col] = cleaned
    before = len(df)
    df = df[df[smile_col].notna()].reset_index(drop=True)
    if before != len(df):
        console.log(f"[yellow]⚠ {before - len(df)} SMILES failed canonicalisation")
    return df


# ────────────────────────────────────────────────────────────────
# Generic DuckDB creator
# ────────────────────────────────────────────────────────────────
def csv_to_duckdb(csv_file: Path, db_dir: Path, table_name: str):
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "database.db"
    if db_path.exists():
        console.log(f"[green]✓ {table_name} DuckDB already present - skipping")
        return
    console.log(f"[blue]→ creating DuckDB for {table_name}")
    con = duckdb.connect(str(db_path))
    query = f"""
        CREATE TABLE {table_name} AS
        SELECT ROW_NUMBER() OVER() AS row_id, *
        FROM read_csv_auto('{csv_file}')
    """
    con.execute(query)
    con.execute(f"CREATE INDEX idx_{table_name}_row_id ON {table_name}(row_id)")
    con.close()
    console.log(f"[green]✓ DuckDB created → {db_path.relative_to(PROCESSED_DIR)}")


# ────────────────────────────────────────────────────────────────
# Dataset-specific processors
# ────────────────────────────────────────────────────────────────
def _chunk_loop(reader, out_file: Path, cols_map: dict):
    with Progress(SpinnerColumn(), "[progress.description]{task.description}",
                  BarColumn(), TimeRemainingColumn(), transient=True,
                  console=console) as prog:
        t = prog.add_task("chunks", total=None)
        for chunk_idx, df in enumerate(reader):
            prog.update(t, advance=1, description=f"chunk {chunk_idx}")
            df = df.rename(columns=cols_map)[list(cols_map.values())]
            df = canonicalize_and_filter_df(df, "item")
            df.to_csv(out_file, mode="a",
                      header=not out_file.exists(), index=False)
            gc.collect()


# ────────────────────────────────────────────────────────────────
# ZINC dataset
# ────────────────────────────────────────────────────────────────
def process_zinc():
    csv_path = _extract_if_missing(ZINC_TAR_GZ, RAW_DIR / "zinc15_10M_2D.csv",
                                   member="zinc15_10M_2D.csv")
    out_dir = ZINC_DIR; out_csv = out_dir / "data.csv"
    if (out_dir / "database.db").exists():
        console.log("[green]✓ ZINC-10M already processed - skipping")
        return
    console.rule("[bold]ZINC 10 M")
    reader = pd.read_csv(csv_path, usecols=["smiles", "zinc_id"],
                         chunksize=1_000_000)
    _chunk_loop(reader, out_csv, {"smiles": "item", "zinc_id": "external_id"})
    csv_to_duckdb(out_csv, out_dir, "zinc_10m")


# ────────────────────────────────────────────────────────────────
# D4 dataset
# ────────────────────────────────────────────────────────────────
def process_d4():
    csv_path = _extract_if_missing(D4_GZ, RAW_DIR / "D4_screen_table.csv")
    out_dir = D4_DIR; out_csv = out_dir / "data.csv"
    if (out_dir / "database.db").exists():
        console.log("[green]✓ D4-138M already processed - skipping")
        return
    console.rule("[bold]D4 138 M")
    reader = pd.read_csv(csv_path, usecols=["zincid", "smiles"],
                         chunksize=5_000_000)
    _chunk_loop(reader, out_csv, {"zincid": "external_id",
                                  "smiles": "item"})
    csv_to_duckdb(out_csv, out_dir, "d4_138m")


# ────────────────────────────────────────────────────────────────
# Natural product dataset
# ────────────────────────────────────────────────────────────────
def process_natural_products():
    out_csv = NATPROD_DIR / "data.csv"
    if out_csv.exists():
        console.log("[green]✓ Natural-products CSV already processed - skipping")
        return

    csv_path = _extract_if_missing(NATPROD_ZIP, 
                                   RAW_DIR / "coconut_csv_lite-05-2025.csv",
                                   member="coconut_csv_lite-05-2025.csv")

    console.rule("[bold]Natural-Products dataset")

    df = pd.read_csv(
        csv_path,
        usecols=[
            "identifier",
            "canonical_smiles",
            "heavy_atom_count",
            "number_of_minimal_rings",
            "contains_sugar",
            "chemical_class",
            "chemical_super_class",
        ],
    )

    df = df.rename(
        columns={
            "canonical_smiles": "item",
            "identifier": "external_id",
        }
    )

    df = canonicalize_and_filter_df(df, "item")
    NATPROD_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    console.log(f"[green]✓ wrote natural-products → {out_csv.relative_to(PROCESSED_DIR)} "
                f"({len(df):,} rows)")


# ────────────────────────────────────────────────────────────────
# Enamine dataset
# ────────────────────────────────────────────────────────────────
def _extract_enamine_csv() -> Path | None:
    if (RAW_DIR / "Data/4_real_space/building_blocks.csv").exists():
        return RAW_DIR / "Data/4_real_space/building_blocks.csv"
    if not DATA_ZIP.exists():
        console.log("[red]✗ Data.zip (Enamine) missing")
        return None
    console.log(f"[blue]→ extracting {DATA_ZIP.name}")
    with zipfile.ZipFile(DATA_ZIP) as zf:
        zf.extractall(RAW_DIR)
    return RAW_DIR / "Data/4_real_space/building_blocks.csv"

def _get_reaction_bb_pickle() -> Path | None:
    """
    Return path to reaction_to_building_blocks_filtered.pkl,
    extracting Data.zip if necessary.
    """
    pkl_path = RAW_DIR / "Data/4_real_space/reaction_to_building_blocks_filtered.pkl"
    if pkl_path.exists():
        return pkl_path
    if not DATA_ZIP.exists():
        console.log("[red]✗ Data.zip missing - need Enamine reactions")
        return None
    console.log(f"[blue]→ extracting {DATA_ZIP.name} for reaction pickle")
    with zipfile.ZipFile(DATA_ZIP) as zf:
        zf.extractall(RAW_DIR)
    return pkl_path if pkl_path.exists() else None

def _enamine_outputs_exist() -> bool:
    """True when CSV, DB and three pickle files already exist."""
    req_files = [
        ENAMINE_DIR / "data.csv",
        ENAMINE_DIR / "database.db",
        ENAMINE_DIR / "enamine_reactions.pkl",
        ENAMINE_DIR / "enamine_id_to_reaction.pkl",
        ENAMINE_DIR / "enamine_reaction_to_bb.pkl",
    ]
    return all(f.exists() for f in req_files)

def process_enamine():
    if _enamine_outputs_exist():
        console.log("[green]✓ Enamine already processed - skipping")
        return

    src_csv = _extract_enamine_csv()
    rxn_pkl = _get_reaction_bb_pickle()
    if src_csv is None or rxn_pkl is None:
        return

    ENAMINE_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = ENAMINE_DIR / "data.csv"

    console.rule("[bold]Enamine BB   (csv → pickle → DuckDB)")

    # 1) CSV → canonicalised dataframe
    df = pd.read_csv(src_csv, usecols=["smiles", "Catalog_ID"])
    df = df.rename(columns={"smiles": "item", "Catalog_ID": "external_id"})
    df = canonicalize_and_filter_df(df, "item")
    unique_smiles = set(df["item"].tolist())

    # 2) Reaction pickles ------------------------------------------------------
    with open(rxn_pkl, "rb") as f:
        reaction_to_bb = pkl.load(f)

    valid_reactions      = set()
    reaction_id_to_str   = {}
    two_bb_reaction_ids  = set()

    for reaction in REAL_REACTIONS:
        if len(reaction.reactants) == 2:           # only 2-building-block
            valid_reactions.add(reaction)
            reaction_id_to_str[reaction.id] = reaction.reaction_smarts
            two_bb_reaction_ids.add(reaction.id)

    # Filter the original mapping to 2-BB reactions only
    reaction_to_two_bb = {
        rid: bb_map
        for rid, bb_map in reaction_to_bb.items()
        if rid in two_bb_reaction_ids
    }

    # Sync building blocks with the canonicalised Enamine SMILES
    two_bb_smiles = set()
    for rid, reagent_map in reaction_to_two_bb.items():
        for reagent_id, bb_list in reagent_map.items():
            new_set = set(bb_list).intersection(unique_smiles)
            reagent_map[reagent_id] = new_set
            two_bb_smiles.update(new_set)

    # 3) Save the three pickle artefacts --------------------------------------
    (ENAMINE_DIR / "enamine_reactions.pkl").write_bytes(pkl.dumps(valid_reactions))
    (ENAMINE_DIR / "enamine_id_to_reaction.pkl").write_bytes(pkl.dumps(reaction_id_to_str))
    (ENAMINE_DIR / "enamine_reaction_to_bb.pkl").write_bytes(pkl.dumps(reaction_to_two_bb))
    console.log("[green]✓ reaction pickles written")

    # 4) Final CSV filtered to 2-BB subset ------------------------------------
    df_unique = (
        df[df["item"].isin(two_bb_smiles)]
        .drop_duplicates("item")
        .reset_index(drop=True)
    )
    df_unique.to_csv(out_csv, index=False)
    console.log(f"[green]✓ data.csv written  ({len(df_unique):,} rows)")

    # 5) DuckDB ---------------------------------------------------------------
    csv_to_duckdb(out_csv, ENAMINE_DIR, "enamine")


# ────────────────────────────────────────────────────────────────
# ChEMBL dataset
# ────────────────────────────────────────────────────────────────
def _fetch_chembl_smiles(molecule_ids: list[str]) -> pd.DataFrame:
    """
    Page through the ChEMBL API to obtain canonical SMILES for the given ids.
    """
    mol_client = new_client.molecule
    batch_size = Settings.Instance().MAX_LIMIT
    records = []

    with Progress(SpinnerColumn(), "[progress.description]{task.description}",
                  BarColumn(), transient=True, console=console) as prog:
        task = prog.add_task("ChEMBL pages", total=len(molecule_ids))
        for i in range(0, len(molecule_ids), batch_size):
            batch = molecule_ids[i : i + batch_size]
            page = mol_client.filter(molecule_chembl_id__in=batch).only(
                "molecule_chembl_id", "molecule_structures"
            )
            records.extend(page)
            prog.update(task, advance=len(batch))
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    df = df.dropna(subset=["molecule_structures"])
    df["smiles"] = df.molecule_structures.map(
        lambda x: x.get("canonical_smiles") if x else None
    )
    df = df.drop(columns=["molecule_structures"]).dropna()
    df = df.drop_duplicates("molecule_chembl_id")
    return df

def _chembl_outputs_exist() -> bool:
    req = [CHEMBL_DIR / f for f in ("data.csv", )]
    return all(p.exists() for p in req)

def process_chembl():
    if _chembl_outputs_exist():
        console.log("[green]✓ ChEMBL dataset already processed - skipping")
        return

    if not CHEMBL_CSV.exists():
        console.log(f"[red]✗ Missing raw CSV {CHEMBL_CSV.name}")
        return

    CHEMBL_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = CHEMBL_DIR / "data.csv"

    console.rule("[bold]ChEMBL ErbB1 IC-50")

    # 1) Raw CSV filtering ----------------------------------------------------
    df = pd.read_csv(CHEMBL_CSV)
    df = df[
        (df.target_organism == "Homo sapiens") &
        (df.standard_units == "nM")
    ].reset_index(drop=True)

    df = df.drop_duplicates("molecule_chembl_id").rename(
        columns={"standard_value": "IC50"}
    )
    df = df.dropna(subset=["IC50"]).reset_index(drop=True)

    # 2) Retrieve canonical SMILES -------------------------------------------
    smiles_df = _fetch_chembl_smiles(df.molecule_chembl_id.tolist())
    if smiles_df.empty:
        console.log("[red]✗ No SMILES retrieved - aborting")
        return

    # 3) Merge + pIC50 --------------------------------------------------------
    merged = (
        df[["molecule_chembl_id", "IC50"]]
        .merge(smiles_df, on="molecule_chembl_id")
        .reset_index(drop=True)
    )
    merged["pIC50"] = merged.IC50.map(lambda x: 9 - math.log10(x))

    merged = merged.rename(
        columns={
            "molecule_chembl_id": "external_id",
            "smiles": "item",
        }
    )[["external_id", "item", "IC50", "pIC50"]]

    # Canonicalise SMILES once more (cheap safety)
    merged = canonicalize_and_filter_df(merged, "item")

    merged.to_csv(out_csv, index=False)
    console.log(f"[green]✓ data.csv written  ({len(merged):,} rows)")
