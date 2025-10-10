from rich.console import Console

from .enamine_assembled import main as enamine_main 
from .embedding_compression import main as embed_comp_main

console = Console()

# ────────────────────────────────────────────────────────────────
# Public entrypoint
# ────────────────────────────────────────────────────────────────
def main():
    console.rule("[bold]Building Datasets")

    enamine_main()
    embed_comp_main()

    console.rule("[bold green]Building Datasets Complete")

if __name__ == "__main__":
    main()
