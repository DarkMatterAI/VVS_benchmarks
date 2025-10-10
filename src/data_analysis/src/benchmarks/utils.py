"""
Common helpers used by both sweep-selection and final-run analysis.
"""

from __future__ import annotations
import json, re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .constants import TOPKS, console

# ───────────────────────── regex helpers ──────────────────────────
_RUN_RE = re.compile(r"(.+?)_(\d{2,3}-\d{2})_(\d+)$")


def parse_run_name(name: str) -> Tuple[str, str, str]:
    """
    Convert   <base>_<gridId>-<scoreId>_<replica>
    into     (base, gridId-scoreId, replica)
    """
    m = _RUN_RE.match(name)
    return (m.group(1), m.group(2), m.group(3)) if m else (name, "??", "??")


# ───────────────────────── I/O + post-processing ──────────────────────────
def load_one_run(run_dir: Path,
                 trunc: int = 50_000
                 ) -> Tuple[pd.DataFrame, dict] | Tuple[None, None]:
    """
    Return (df, params) or (None, None) when either file is missing /
    empty / unreadable.  The *df* is always sorted newest-score-first,
    truncated to *trunc* rows.
    """
    log_p   = run_dir / "score_log.csv"
    param_p = run_dir / "params.json"
    if not (log_p.exists() and param_p.exists()):
        return None, None

    try:
        df = (pd.read_csv(log_p)
                .assign(ts=lambda d: pd.to_datetime(d["ts"], format="mixed"))
                .sort_values(["ts", "score"], ascending=[True, False])
                .head(trunc))
        if df.empty:
            return None, None
    except Exception as exc:
        console.log(f"[red]x failed reading {log_p}: {exc}")
        return None, None

    with param_p.open() as fh:
        params = json.load(fh)

    return df, params


def summarise_scores(df: pd.DataFrame,
                     *,
                     strip_bb: bool = False) -> dict:
    """
    Calculate n-results, runtime, and Top-K means.
    If *strip_bb* is True and column `is_bb` exists, rows flagged as
    building-blocks are ignored for n_results **and** Top-K scoring.
    """
    if strip_bb and "is_bb" in df:
        df = df[~df.is_bb].copy()

    rec = {
        "n_results": len(df),
        "runtime":   (df.ts.iloc[-1] - df.ts.iloc[0]).total_seconds(),
    }
    for k in TOPKS:
        rec[f"top{k}"] = df.score.nlargest(k).mean()
    return rec
