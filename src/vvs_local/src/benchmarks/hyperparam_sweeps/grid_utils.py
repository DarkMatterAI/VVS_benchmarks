"""
Cartesian-product expander for the *runs:* section of a sweep YAML.
"""

from __future__ import annotations
from itertools import product
from typing import Dict, List, Tuple

def expand_grid(run_block: Dict) -> Tuple[str, List[Tuple[str, Dict]]]:
    """
    Returns
    -------
    run_name  - base name from YAML
    configs   - list of (cfg_id, merged-dict)
                cfg_id is <grid#>-<scorer#>
    """
    axes, keys = [], []
    for k, v in run_block["params"].items():
        keys.append(k)
        axes.append(v if isinstance(v, list) else [v])

    configs: List[Tuple[str, Dict]] = []
    for g_idx, combo in enumerate(product(*axes), 1):
        grid_cfg = dict(zip(keys, combo))

        for s_idx, scorer_cfg in enumerate(run_block["scorers"], 1):
            merged = {
                **grid_cfg,
                **run_block["run_params"],    # inference budget, etc.
                **scorer_cfg,
            }
            cfg_id = f"{g_idx:02d}-{s_idx:02d}"
            configs.append((cfg_id, merged))

    return run_block["name"], configs
