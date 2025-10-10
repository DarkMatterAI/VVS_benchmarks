from rich.console import Console
from rich.panel import Panel

from .setup_dirs import setup_internal_dirs
from .process_openeye import preprocess_openeye_files
from .process_synthemol import preprocess_synthemol_artifacts
from .dataset_processing import (process_zinc, 
                                 process_d4, 
                                 process_enamine, 
                                 process_chembl,
                                 process_natural_products)

console = Console()

def main():
    console.print(Panel("[bold]Light + Heavy processing stage", style="blue"))
    setup_internal_dirs()
    preprocess_openeye_files()
    preprocess_synthemol_artifacts()
    process_chembl()

    # heavy datasets
    process_enamine()
    process_natural_products()
    process_zinc()
    process_d4()

    console.print(Panel("[bold green]All processing complete"))

if __name__ == "__main__":
    main()
