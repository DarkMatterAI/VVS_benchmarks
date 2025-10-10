"""
Chunked DuckDB → multiprocessing fingerprints → USearch HNSW index.
Outputs
  • packed-bit fingerprints  (optional, for inspection)
  • rad_index.hnsw           (HNSW graph on disk)
"""
import os, json, multiprocessing, time, yaml
from functools import partial
from pathlib import Path
import argparse
import time 

import duckdb, numpy as np, pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeRemainingColumn
from usearch.index import Index

console = Console()

p = argparse.ArgumentParser()
p.add_argument("--cfg", required=True)
p.add_argument("--mode", choices=["full", "debug"], default="full")
args = p.parse_args()

# ----------------------------------------------------------------
BLOB_ROOT  = Path("/code/blob_store").resolve()
DATA_ROOT  = BLOB_ROOT / "internal" / "processed"
OUT_ROOT   = DATA_ROOT / "rad"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
MAX_CPU = max(multiprocessing.cpu_count() - 2, 2)

def fp_one(smiles: str, fp_size: int, radius: int):
    fpg = GetMorganGenerator(radius=radius, fpSize=fp_size)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    arr = np.zeros((fp_size,), np.uint8)
    DataStructs.ConvertToNumpyArray(fpg.GetFingerprint(mol), arr)
    return np.packbits(arr)

def build_index(ds_cfg: dict, mode: str):
    name      = ds_cfg["name"]
    fp_size   = ds_cfg["fp_size"]
    radius    = ds_cfg["fp_radius"]
    chunk_sz  = ds_cfg["chunksize"]
    m         = ds_cfg["m"]
    ef_con    = ds_cfg["ef_con"]

    limit     = ds_cfg["debug_rows"] if mode=="debug" else ds_cfg["full_rows"]
    out_name  = f"{name}{'_debug' if mode=='debug' else ''}.hnsw"
    out_path  = OUT_ROOT / out_name
    if out_path.exists():
        out_path.unlink()

    console.rule(f"[bold]{name}: fingerprints → HNSW  (limit={limit:,})")

    fp_func = partial(fp_one, fp_size=fp_size, radius=radius)

    db_path = str(DATA_ROOT / name / "database.db")
    conn = duckdb.connect(db_path, read_only=True)

    hnsw = Index(
        ndim=fp_size, dtype="b1", metric="tanimoto",
        connectivity=m, expansion_add=ef_con
    )

    off = 0
    seen = 0
    failed = 0
    start = time.time()
    with Progress(SpinnerColumn(), "[progress.description]{task.description}",
                  TimeRemainingColumn(), console=console) as prog:
        t = prog.add_task("build", total=limit)
        while seen < limit:
            remaining = limit - seen 
            console.rule(f"[bold]{name}: processing batch - {seen}/{limit}")
            chunk_sz_iter = min(chunk_sz, remaining)
            df = conn.sql(f"""
                SELECT item
                FROM {ds_cfg['table']}
                LIMIT {chunk_sz_iter} OFFSET {off}
            """).df()
            if df.empty: break

            # reset pool every chunk to clear memory 
            with multiprocessing.Pool(MAX_CPU) as pool:
                fps = pool.map(fp_func, df.item.values)
            # drop None (bad SMILES) and keep keys aligned
            good = [(seen + i, fp) for i, fp in enumerate(fps) if fp is not None]
            if len(good) != len(fps):
                n_failed = len(fps) - len(good)
                failed += n_failed
                console.rule(f"[bold]{name}: {n_failed} fingerprints failed to compute")
            if not good:
                off += chunk_sz
                continue
            
            keys, arr = zip(*good)
            hnsw.add(np.array(keys, dtype=np.uint32), np.stack(arr))
            seen += len(arr)
            off  += chunk_sz
            prog.update(t, completed=seen)

    hnsw.save(str(out_path))

    elapsed = time.time() - start 
    log = {
        'config' : ds_cfg,
        'time' : elapsed,
        'failed' : failed 
    }
    log_out_path  = OUT_ROOT / out_name.replace('.hnsw', '.json')
    with open(log_out_path, 'w') as f:
        json.dump(log, f)

    console.log(f"[cyan]{name}: saved → {out_path}  ({seen:,} vectors)")

# ------------------------------ main loop ------------------------------------
def main():
    config = yaml.safe_load(Path(args.cfg).read_text())["datasets"]
    for ds_cfg in config:
        build_index(ds_cfg, args.mode)

if __name__ == "__main__":
    main()
