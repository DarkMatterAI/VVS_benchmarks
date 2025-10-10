from itertools import product
def expand_grid(entry: dict):
    """Return list[(cfg_id, merged_dict)].  Same format as other benchmarks."""
    axes  = {k:(v if isinstance(v,list) else [v]) for k,v in entry["params"].items()}
    keys  = list(axes)
    base  = entry["name"]
    score_cfgs = entry["score_params"]
    const = entry["run_params"]

    combos = product(*axes.values())
    result = []
    for gi, combo in enumerate(combos, 1):
        grid = dict(zip(keys, combo))
        for si, sc in enumerate(score_cfgs, 1):
            cfg_id = f"{gi:02d}-{si:02d}"
            merged = grid | sc | const
            result.append((cfg_id, merged))
    return base, result
