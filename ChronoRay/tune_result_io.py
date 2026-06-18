"""Helpers for exporting Ray Tune :class:`~ray.tune.ResultGrid` runs."""

from __future__ import annotations

import math
from typing import Any


def trials_as_dicts(result_grid: Any, metric_key: str = "objective") -> list[dict]:
    """Build a list of trial dicts from a ResultGrid.

    Uses :meth:`~ray.tune.ResultGrid.get_dataframe` when non-empty; otherwise
    iterates trials directly (some Ray / searcher combinations leave the
    dataframe empty while per-trial metrics are still available).
    """
    trials: list[dict] = []
    df = result_grid.get_dataframe()
    if not df.empty:
        col = metric_key
        if col not in df.columns:
            candidates = [c for c in df.columns if metric_key.lower() in c.lower()]
            if not candidates:
                raise KeyError(f"No metric column matching {metric_key!r} in Tune dataframe")
            col = candidates[0]
        for i, row in df.iterrows():
            obj = float(row[col])
            params = {
                k.replace("config/", ""): float(row[k])
                for k in df.columns
                if k.startswith("config/") and row[k] == row[k]
            }
            trials.append({
                "trial_id": int(i),
                "params": params,
                "rmse": math.sqrt(max(obj, 0.0)),
                "objective": obj,
                "duration_s": float(row.get("time_total_s", 0.0) or 0.0),
            })
    else:
        for i, result in enumerate(result_grid):
            if getattr(result, "error", None):
                continue
            metrics = result.metrics or {}
            obj = metrics.get(metric_key)
            if obj is None:
                continue
            obj = float(obj)
            config = result.config or {}
            params = {k: float(v) for k, v in config.items() if isinstance(v, (int, float))}
            trials.append({
                "trial_id": i,
                "params": params,
                "rmse": math.sqrt(max(obj, 0.0)),
                "objective": obj,
                "duration_s": float(metrics.get("time_total_s", 0.0) or 0.0),
            })
    return trials
