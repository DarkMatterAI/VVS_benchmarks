"""
Concurrent, resumable download utility for the raw artifacts.

* All files land in  $BLOB_STORE/internal/raw_downloads/
* Already-present files are skipped.
* HTTP downloads use httpx + HTTP/2 + streaming.
* The PatWalters/TS repo is cloned with `git` (small).
* Adjust the DOWNLOADS / GIT_REPOS lists as needed.
"""
import asyncio
import os
import subprocess
from pathlib import Path
from typing import List

from chembl_webresource_client.settings import Settings
Settings.Instance().CONCURRENT_SIZE = 100
Settings.Instance().MAX_LIMIT = 50
from chembl_webresource_client.new_client import new_client
import pandas as pd   

import httpx
import aiofiles
from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    DownloadColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

console = Console()
DOWNLOAD_CONCURRENCY = 4

# ────────────────────────────────────────────────────────────────
# What to fetch
# ────────────────────────────────────────────────────────────────
DOWNLOADS: List[dict] = [
    # 1) ZINC-10M
    dict(
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/zinc15_10M_2D.tar.gz",
        name="zinc15_10M_2D.tar.gz",
    ),
    # 2) D4 screen table 
    dict(
        url="https://figshare.com/ndownloader/files/13599404",
        name="D4_screen_table.csv.gz",
    ),
    # 3) Enamine BB csv / pickle
    dict(
        url="https://zenodo.org/records/10257839/files/Data.zip",
        name="Data.zip",
    ),
    # 4) SyntheMol random-forest models
    dict(
        url="https://zenodo.org/records/10257839/files/Models.zip",
        name="Models.zip",
    ),
    # 5) Coconut-lite natural-products subset 
    dict(
        url="https://coconut.s3.uni-jena.de/prod/downloads/2025-05/coconut_csv_lite-05-2025.zip",
        name="natural_products.zip",
    ),
]

GIT_REPOS = [
    dict(
        url="https://github.com/PatWalters/TS.git",
        dest="TS",
    )
]

# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────
RAW_DIR = (
    Path(os.environ.get("BLOB_STORE", "/code/blob_store"))
    / "internal"
    / "raw_downloads"
)
RAW_DIR.mkdir(parents=True, exist_ok=True)


async def _download_file(client: httpx.AsyncClient, task, semaphore, progress):
    url, name = task["url"], task["name"]
    dest_path = RAW_DIR / name

    if dest_path.exists():
        console.log(f"[bold green]✓ {name} already present - skipping")
        return

    async with semaphore:
        task_id = progress.add_task(f"[cyan]{name}", start=False)
        console.log(f"[blue]→ downloading {url}")

        async with client.stream("GET", url, follow_redirects=True, timeout=None) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            progress.update(task_id, total=total, start=True)

            async with aiofiles.open(dest_path, "wb") as out:
                async for chunk in r.aiter_bytes(chunk_size=1 << 20):  # 1 MiB
                    await out.write(chunk)
                    progress.update(task_id, advance=len(chunk))

        progress.update(task_id, completed=total)
        console.log(f"[bold green]✓ finished {name}")


async def download_all(concurrency: int = DOWNLOAD_CONCURRENCY):
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(http2=True, timeout=None) as client:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            await asyncio.gather(
                *[
                    _download_file(client, task, semaphore, progress)
                    for task in DOWNLOADS
                ]
            )


def clone_repos():
    for repo in GIT_REPOS:
        dest_path = RAW_DIR / repo["dest"]
        if dest_path.exists():
            console.log(f"[bold green]✓ {repo['dest']} already cloned - skipping")
            continue

        console.log(f"[blue]→ cloning {repo['url']}")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo["url"], str(dest_path)],
            check=True,
        )
        console.log(f"[bold green]✓ cloned {repo['dest']}")


def fetch_chembl_erbb1():
    CHEMBL_CSV = RAW_DIR / "chembl_erbb1_ic50.csv"   
    if CHEMBL_CSV.exists():
        console.log("[bold green]✓ chembl_erbb1_ic50.csv already present - skipping")
        return

    console.log("[blue]→ querying ChEMBL for ErbB1 IC-50 data")

    bioactivities = (
        new_client.activity
        .filter(target_chembl_id="CHEMBL203", type="IC50",
                relation="=", assay_type="B")
        .only(
            "activity_id", 
            "assay_chembl_id", 
            "assay_description", 
            "assay_type",
            "molecule_chembl_id", 
            "type", 
            "standard_units", 
            "relation",
            "standard_value", 
            "target_chembl_id", 
            "target_organism"
        )
    )

    df = pd.DataFrame.from_dict(bioactivities)
    df.to_csv(CHEMBL_CSV, index=False)
    console.log(f"[bold green]✓ wrote {CHEMBL_CSV.name}  ({len(df):,} rows)")


# ────────────────────────────────────────────────────────────────
# Public entrypoint
# ────────────────────────────────────────────────────────────────
def main():
    console.rule("[bold]Raw-file downloader")
    console.print(f"Downloads will be saved under: [italic]{RAW_DIR}[/]")

    clone_repos()               # git
    asyncio.run(download_all()) # HTTP
    fetch_chembl_erbb1()        # ChEMBL

    console.rule("[bold green]All downloads complete")

if __name__ == "__main__":
    main()

