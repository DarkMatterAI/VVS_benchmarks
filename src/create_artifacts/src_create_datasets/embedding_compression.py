"""
Generate embedding_compression.csv (TARGET_SIZE rows) from

  • Zinc-10M     - full table
  • D4-138M      - sample if table > SAMPLE_EACH
  • Enamine-Asmbl - sample if table > SAMPLE_EACH
"""
from __future__ import annotations
import os
from pathlib import Path
import pyarrow as pa
import duckdb, pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeRemainingColumn

console = Console()

# ─── paths ──────────────────────────────────────────────────────
BLOB      = Path(os.environ.get("BLOB_STORE", "/code/blob_store")).resolve()
PROC      = BLOB / "internal" / "processed"
TRAIN     = BLOB / "internal" / "training_datasets"

ZINC_DB   = PROC / "zinc_10m"            / "database.db"
D4_DB     = PROC / "d4_138m"             / "database.db"
ENA_DB    = TRAIN / "enamine_assembled"  / "enamine_assembled.db"

OUT_DIR   = TRAIN / "embedding_compression"
OUT_CSV   = OUT_DIR / "embedding_compression.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── parameters ─────────────────────────────────────────────────
TARGET_SIZE   = 30_000_003
SAMPLE_EACH   = TARGET_SIZE // 3
BATCH         = 100_000

# ─── helpers ────────────────────────────────────────────────────
def n_rows(db_path: Path, table: str) -> int:
    con = duckdb.connect(str(db_path), read_only=True)
    cnt = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    con.close()
    return cnt

def stream(con: duckdb.DuckDBPyConnection, sql: str,
           writer, header_written: bool) -> tuple[bool, int]:
    """
    Stream-query `sql` and append to `writer` in batches of BATCH rows.
    Returns (header_written, rows_written).
    """
    reader_or_table = con.execute(sql).fetch_record_batch(BATCH)
    rows = 0

    # Case 1: RecordBatchReader (common when rows_per_batch is given)
    if isinstance(reader_or_table, pa.RecordBatchReader):
        for rb in reader_or_table:
            df = rb.to_pandas()
            df.to_csv(writer, header=not header_written, index=False)
            header_written = True
            rows += len(df)

    # Case 2: single pyarrow.Table (when result fits in one batch)
    else:  # pyarrow.Table
        tbl = reader_or_table
        if tbl.num_rows:
            df = tbl.to_pandas()
            df.to_csv(writer, header=not header_written, index=False)
            header_written = True
            rows = tbl.num_rows

    return header_written, rows

# ─── main ───────────────────────────────────────────────────────
def main():
    if OUT_CSV.exists() and sum(1 for _ in OUT_CSV.open()) - 1 >= TARGET_SIZE:
        console.log("[green]✓ embedding_compression.csv already ≥ target - skipping")
        return

    console.rule("[bold]Building embedding-compression dataset")

    sources = [
        ("ZINC-10M" , ZINC_DB, "zinc_10m", SAMPLE_EACH),
        ("D4-138M"  , D4_DB , "d4_138m" , SAMPLE_EACH),
        ("Enamine-Assembled", ENA_DB, "enamine", SAMPLE_EACH),
    ]

    header_written, total_written = False, 0
    OUT_CSV.unlink(missing_ok=True)

    with OUT_CSV.open("w") as writer, Progress(
        SpinnerColumn(), "[progress.description]{task.description}",
        BarColumn(), TimeRemainingColumn(), console=console
    ) as prog:
        bar = prog.add_task("writing", total=TARGET_SIZE)

        for name, db_path, table, target in sources:
            if not db_path.exists():
                console.log(f"[red]✗ missing {db_path}")
                return
            rows_in_table = n_rows(db_path, table)
            con = duckdb.connect(str(db_path), read_only=True)

            if target is None or rows_in_table <= target:
                sql = f"SELECT item FROM {table}" if table != "enamine" else \
                      f"SELECT product AS item FROM enamine"
                console.log(f"🔹 {name}: streaming full table ({rows_in_table:,})")
            else:
                sql = f"""
                    SELECT { 'item' if table!='enamine' else 'product AS item' }
                    FROM {table}
                    ORDER BY RANDOM()
                    LIMIT {target}
                """
                console.log(f"🔹 {name}: sampling {target:,} of {rows_in_table:,}")

            header_written, rows = stream(con, sql, writer, header_written)
            total_written += rows
            prog.update(bar, advance=rows)
            con.close()

    console.rule(f"[bold green]Done → {OUT_CSV.name}  ({total_written:,} rows)")

if __name__ == "__main__":
    main()
