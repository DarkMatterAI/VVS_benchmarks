from rich.console import Console

from .paths import (
    INTERNAL, 
    PROCESSED_DIR,
    OPENEYE_DOCK, 
    OPENEYE_ROCS, 
    SYNTHEMOL_RF,
    ZINC_DIR, 
    D4_DIR, 
    ENAMINE_DIR,
    CHEMBL_DIR,
)

console = Console()

def setup_internal_dirs() -> None:
    """
    Ensure the complete blob-store hierarchy exists.
    Idempotent: re-runs are harmless.
    """
    # root segments
    for p in (INTERNAL, PROCESSED_DIR):
        p.mkdir(parents=True, exist_ok=True)

    # sub-trees for artefacts and datasets
    for p in (
        OPENEYE_DOCK,
        OPENEYE_ROCS,
        SYNTHEMOL_RF,
        ZINC_DIR,
        D4_DIR,
        ENAMINE_DIR,
        CHEMBL_DIR,
    ):
        p.mkdir(parents=True, exist_ok=True)

    console.log("[bold green]✓ directory tree verified")

