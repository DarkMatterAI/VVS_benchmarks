from __future__ import annotations
import multiprocessing, pickle
from pathlib import Path
import pandas as pd
from rich.console import Console
from .constants import (ENV_DIR, ENV_BASE, TEMPLATE_TXT, BB_SMI,
                        BB_CSV_PTH, REACTION_PKL, N_CPU)

console = Console()

def ensure_env() -> None:
    # 1) reaction SMARTS ▶ template.txt
    if not TEMPLATE_TXT.exists():
        TEMPLATE_TXT.parent.mkdir(parents=True, exist_ok=True)
        with open(REACTION_PKL, "rb") as fh:
            id2s = pickle.load(fh)
        uniq = []
        seen = set()
        for sm in id2s.values():
            if sm not in seen:
                seen.add(sm)
                uniq.append(sm)
        TEMPLATE_TXT.write_text("\n".join(uniq))
        console.log(f"[cyan]template.txt written ({len(uniq)} SMARTS)")

    # 2) building_block.smi   (SMILES<TAB>ID)
    if not BB_SMI.exists():
        df = pd.read_csv(BB_CSV_PTH, usecols=["item", "external_id"])
        df = df[~df.item.str.contains("Si")]     # drop silicon blocks
        BB_SMI.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(BB_SMI, sep="\t", index=False, header=False)
        console.log(f"[cyan]building_block.smi written ({len(df):,} blocks)")

    # 3) fingerprints / descriptors / masks
    NEED = {"bb_mask.npy", "bb_desc.npy", "bb_fp_2_1024.npy"}
    missing = [f for f in NEED if not (ENV_DIR / f).exists()]
    if missing:
        console.rule("[bold]generating RxnFlow env arrays")

        import sys
        sys.path.append("/code/RxnFlow")
        from RxnFlow.data.scripts._b_smi_to_env import get_block_data

        get_block_data(
            block_path      = str(BB_SMI),
            template_path   = str(TEMPLATE_TXT),
            save_directory_path = str(ENV_DIR),
            num_cpus        = N_CPU,
        )
    else:
        console.log("[green]✓ RxnFlow env already complete")
