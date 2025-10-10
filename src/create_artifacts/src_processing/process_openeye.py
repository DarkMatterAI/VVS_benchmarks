import shutil
from pathlib import Path
from rich.console import Console
from rich.table import Table

from .paths import OPENEYE_DOCK, OPENEYE_ROCS, TS_REPO_DATA, PROVIDED_6LUD

console = Console()

def preprocess_openeye_files() -> None:
    """
    Place three OpenEye files into the new internal layout:

        internal/openeye/docking/2zdt_receptor.oedu
        internal/openeye/docking/6lud.oedu
        internal/openeye/rocs/2chw_lig.sdf
    """
    mapping = {
        TS_REPO_DATA / "2zdt_receptor.oedu": OPENEYE_DOCK / "2zdt_receptor.oedu",
        TS_REPO_DATA / "2chw_lig.sdf"      : OPENEYE_ROCS / "2chw_lig.sdf",
        PROVIDED_6LUD                      : OPENEYE_DOCK / "6lud.oedu",
    }

    moved, skipped, missing = [], [], []

    for src, dst in mapping.items():
        if dst.exists():            # already there
            skipped.append(dst.name)
            continue
        if not src.exists():        # missing source
            missing.append(src)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, dst)
        moved.append(dst.name)

    # Pretty summary
    table = Table(title="OpenEye asset placement", show_header=True, header_style="bold blue")
    table.add_column("Status"); table.add_column("File")
    for f in moved:   table.add_row("[green]moved/copied", f)
    for f in skipped: table.add_row("[yellow]skipped",     f)
    for f in missing: table.add_row("[red]MISSING",        str(f))
    console.print(table)
