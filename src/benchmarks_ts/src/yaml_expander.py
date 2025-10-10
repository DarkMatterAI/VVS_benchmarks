from itertools import product

def expand_grid(entry: dict):
    """
    Returns a list of tuples (cfg_id, merged_params_dict).

    cfg_id is a 2-part identifier: <grid#>-<score#>
    """
    grid_fields = entry["params"]
    score_list  = entry["score_params"]         # list of dicts
    base_name   = entry["name"]
    run_const   = entry["run_params"]

    # -------- split scalars vs iterable sweep axes ----------------------
    axes = []
    keys = []
    for k, v in grid_fields.items():
        keys.append(k)
        axes.append(v if isinstance(v, list) else [v])

    configs = []
    g_idx = 0
    for combo in product(*axes):               # cartesian grid
        g_idx += 1
        grid_cfg = dict(zip(keys, combo))

        s_idx = 0
        for score_cfg in score_list:
            s_idx += 1
            merged = {**grid_cfg, **score_cfg, **run_const}
            cfg_id = f"{g_idx:02d}-{s_idx:02d}"
            configs.append((cfg_id, merged))

    return base_name, configs
