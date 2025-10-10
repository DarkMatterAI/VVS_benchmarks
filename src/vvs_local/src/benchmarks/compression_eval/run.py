"""
benchmarks.compression_eval.run
────────────────────────────────────────────────────────────────────────────
Measure *query latency* for every pre-built USearch index under
``vvs_local.constants.INDEX_PATH``.

For each <size>/index.usearch we

    • load the index (this also mmap-opens the vector file);
    • run one “warm-up” search so that the first timed call is realistic;
    • perform *N* timed searches - each on 20 x 1024 random rows with
      small Gaussian noise - and record per-call wall-clock time.

Results are written next to the index as:

    <INDEX_PATH>/<size>/query_latency.json
"""

from __future__ import annotations
import argparse, json, time, os
from pathlib import Path
from datetime import datetime

import numpy as np
from rich.console import Console
from usearch.index import Index 

from vvs_local.constants import INDEX_PATH, console


# ╭──────────────────────── benchmark helper ─────────────────────────╮
def _benchmark_index(idx_dir: Path,
                     n_iter: int = 100,
                     n_queries: int = 1024,
                     k: int = 100,
                     seed: int | None = None):
    """
    Run *n_iter* searches and return timing statistics.
    """
    idx_path = idx_dir / "index.usearch"
    out_path = idx_dir / "query_latency.json"

    if not idx_path.exists():
        console.log(f"[yellow]⚠  {idx_path} missing - skipped")
        return

    # ---------------------------------------------------------------- warm-up
    console.rule(f"[bold blue]Latency test - {idx_dir.name}-d")
    index = Index.restore(str(idx_path))
    rng   = np.random.default_rng(seed)

    console.log(f"[cyan]Warm-up query")
    vecs  = rng.standard_normal((32, index.ndim), dtype=np.float32)
    res = index.search(vecs, count=k)                # warm-up call
    _ = index.vectors[res.keys]

    # ---------------------------------------------------------------- timed runs
    console.log(f"[cyan]Executing timed runs")
    wall_times: list[float] = []
    for i in range(n_iter):
        query_ids = rng.integers(0, index.size, n_queries, dtype=np.int64)
        q = index.get(query_ids).astype(np.float32)
        q += rng.standard_normal(q.shape, dtype=q.dtype) / 10.0

        t0 = time.perf_counter()
        res = index.search(q, count=k)
        _ = index.vectors[res.keys]
        wall_times.append(time.perf_counter() - t0)

        if (i + 1) % 10 == 0:
            console.log(f"  → iter {i+1:3d}/{n_iter}  "
                        f"mean {np.mean(wall_times):.4f} s")

    wall_arr = np.asarray(wall_times)
    stats = {
        "size":          int(idx_dir.name),
        "iters":         n_iter,
        "n_queries":     n_queries,
        "k":             k,
        "mean_seconds":  float(wall_arr.mean()),
        "p50_seconds":   float(np.percentile(wall_arr, 50)),
        "p95_seconds":   float(np.percentile(wall_arr, 95)),
    }

    out_path.write_text(json.dumps(stats, indent=2))
    console.log(f"[green]✓ wrote latency stats → {out_path.relative_to(INDEX_PATH)}")


# ╭────────────────────────────────── main ────────────────────────────╮
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100,
                    help="Timed searches per index (default: 100)")
    ap.add_argument("--n_queries", type=int, default=1024,
                    help="Queries per timed search (default: 1024)")
    args = ap.parse_args()

    all_sizes = sorted(int(p.name) for p in INDEX_PATH.iterdir() if p.is_dir())
    console.log(f"[cyan]Selected sizes: {all_sizes}")

    for s in all_sizes:
        _benchmark_index(INDEX_PATH / str(s),
                         n_iter=args.iters,
                         n_queries=args.n_queries)


if __name__ == "__main__":
    main()
