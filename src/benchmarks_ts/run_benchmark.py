import yaml, argparse, random, multiprocessing as mp
from pathlib import Path
from src.yaml_expander   import expand_grid
from src.ts_runner import run_ts

p = argparse.ArgumentParser()
p.add_argument("--cfg", required=True)
args = p.parse_args()

root = yaml.safe_load(Path(args.cfg).read_text())
all_tasks = []
run_type  = Path(args.cfg).stem.split("_")[0].lower()

for raw in root["runs"]:
    base, configs = expand_grid(raw)
    replicas  = raw["run_params"]["replicas"]
    num_proc  = raw["run_params"].get("num_proc", 1)

    for cfg_id, prm in configs:
        for r in range(replicas):
            run_name = f"{base}_{cfg_id}_{r+1}"
            prm = prm.copy()
            prm["rng_seed"] = random.randrange(1_000_000_000)
            all_tasks.append((run_type, run_name, prm, num_proc))

# ------------------------------------------------------------------ launcher
# group tasks by num_proc so blocks with different num_proc don't mix
from itertools import groupby
for n_proc, group in groupby(sorted(all_tasks, key=lambda t: t[3]), key=lambda t: t[3]):
    grp_tasks = [(rt, rn, p) for rt, rn, p, _ in group]
    if n_proc > 1:
        cpus = min(n_proc, len(grp_tasks))
        with mp.Pool(processes=cpus) as pool:
            pool.starmap(run_ts, grp_tasks)
    else:
        for t in grp_tasks:
            run_ts(*t)
