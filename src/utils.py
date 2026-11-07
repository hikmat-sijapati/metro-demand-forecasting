import re
import numpy as np
import pandas as pd


def norm_name(s) -> str:
    """Normalise station names so ridership and reference tables can be joined."""
    s = str(s).lower()
    s = re.sub(r"\b(metro|station|stn)\b", " ", s)
    return re.sub(r"[^a-z0-9]+", "", s)


def resolve_cols(actual, wanted: dict) -> dict:
    """Map configured column names to actual ones (case-insensitive); fail loudly."""
    lower = {c.lower(): c for c in actual}
    out, missing = {}, []
    for key, name in wanted.items():
        if name.lower() in lower:
            out[key] = lower[name.lower()]
        else:
            missing.append(name)
    if missing:
        raise KeyError(f"Columns {missing} not found. Actual columns: {list(actual)}. "
                       "Update COLS in src/config.py.")
    return out


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "R2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        # MAPE explodes on near-zero counts, so report WAPE instead.
        "WAPE": float(np.sum(np.abs(err)) / np.sum(np.abs(y_true))),
    }


def is_weekend(dates, rule) -> np.ndarray:
    """1 if the date falls on the weekend that applied on that date (UAE changed on rule['change_date'])."""
    d = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    dow = d.dt.dayofweek
    before = d < pd.Timestamp(rule["change_date"])
    return np.where(before, dow.isin(rule["before"]), dow.isin(rule["after"])).astype(int)
