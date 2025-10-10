import shutil, zipfile, tempfile
from pathlib import Path
from rich.console import Console

from .paths import SYNTHEMOL_RF, MODELS_ZIP, RAW_DIR

console = Console()

_EXPECTED_FILE = "syntenol_model.pkl"   # pick one file that must exist

def _extract_models() -> Path | None:
    """
    Unzip Models.zip into a temp dir under raw_downloads/,
    return the path, or None on failure.
    """
    if not MODELS_ZIP.exists():
        console.log(f"[red]✗ Missing {MODELS_ZIP.name} - cannot extract")
        return None

    tmp_dir = RAW_DIR / "Models_extracted"
    if tmp_dir.exists():
        return tmp_dir  # extracted previously

    console.log(f"[blue]→ extracting {MODELS_ZIP.name}")
    with zipfile.ZipFile(MODELS_ZIP) as zf:
        zf.extractall(tmp_dir)
    return tmp_dir

def preprocess_synthemol_artifacts() -> None:
    """
    Move antibiotic_random_forest/ into internal/synthemol_rf
    (but only if the destination is absent or empty).
    """
    if SYNTHEMOL_RF.exists() and any(SYNTHEMOL_RF.iterdir()):
        console.log("[yellow]- SyntheMol RF already present - skipping")
        return

    extracted = _extract_models()
    if extracted is None:
        return  # earlier log explains why

    src_dir = extracted / "Models" / "antibiotic_random_forest"
    if not src_dir.exists():
        console.log(f"[red]✗ {src_dir} not found inside Models.zip")
        return

    SYNTHEMOL_RF.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src=str(src_dir), dst=str(SYNTHEMOL_RF), dirs_exist_ok=True)
    console.log("[green]✓ moved antibiotic_random_forest → internal/synthemol_rf")
