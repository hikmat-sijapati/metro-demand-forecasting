"""Features + chronological split + baselines at the configured GRAIN (day or hour).

  python src/04_features_baselines.py

Leakage rule: every lag >= horizon, so a forecast for period t only uses data up to
t - horizon. Outputs data/processed/features.parquet (with a `split` column, used by the
models on Day 2) and reports/baseline_metrics.csv.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from utils import regression_metrics, is_weekend

P, hourly = C.P, C.GRAIN == "hour"
assert min(P["lags"]) >= P["horizon"], "A lag shorter than the horizon would leak the future."
wk = [P["week"] * k for k in (1, 2, 3, 4)]
assert all(l in P["lags"] for l in wk), "LAGS must contain 1..4 weeks"

df = pd.read_parquet(C.INTERIM_DIR / "counts.parquet").sort_values(["station", "ts"]).reset_index(drop=True)
g = df.groupby("station")["checkins"]
for L in P["lags"]:
    df[f"lag_{L}"] = g.shift(L)              # grid is regular per station: shift by rows == periods
df["lag_mean_4w"] = df[[f"lag_{l}" for l in wk]].mean(axis=1, skipna=False)

df["date"] = df["ts"].dt.normalize()
n_missing = int(df["checkins"].isna().sum())
df["dow"] = df["ts"].dt.dayofweek
df["is_weekend"] = is_weekend(df["date"], C.WEEKEND_RULE)
df["month"] = df["ts"].dt.month
if hourly:
    df["hour"] = df["ts"].dt.hour
try:
    import holidays
    hol = holidays.country_holidays("AE", years=sorted(df["ts"].dt.year.unique()))
    df["is_holiday"] = df["date"].dt.date.map(lambda d: int(d in hol))
except Exception as e:
    warnings.warn(f"holiday calendar unavailable ({e}); is_holiday=0. Add dates manually.")
    df["is_holiday"] = 0
df["station_id"] = df["station"].astype("category").cat.codes
df = df[df["checkins"].notna()].copy()          # lags were built on the full grid; now drop missing-target rows
print(f"dropped {n_missing:,} missing-target periods (dates with no / unreliable data)")

dates = np.sort(df["date"].unique())
i1, i2 = int(len(dates) * C.SPLIT[0]), int(len(dates) * (C.SPLIT[0] + C.SPLIT[1]))
split_of = {d: "train" for d in dates[:i1]} | {d: "val" for d in dates[i1:i2]} | {d: "test" for d in dates[i2:]}
df["split"] = df["date"].map(split_of)
print("grain:", C.GRAIN, "| date ranges:",
      {s: (str(x["date"].min().date()), str(x["date"].max().date())) for s, x in df.groupby("split")})

keys = ["station", "dow"] + (["hour"] if hourly else [])
train = df[df["split"] == "train"]
hist = train.groupby(keys)["checkins"].mean().rename("hist_avg").reset_index()
df = df.merge(hist, on=keys, how="left")
df["hist_avg"] = df["hist_avg"].fillna(train["checkins"].mean())
baselines = {"seasonal_naive_1w": f"lag_{P['week']}", "same_period_yesterday": f"lag_{P['day']}",
             "seasonal_mean_4w": "lag_mean_4w", "historical_avg": "hist_avg"}

rows = []
for split in ("val", "test"):
    # same rows for every baseline (all baseline inputs available) -> a fair comparison
    part = df[(df["split"] == split) & df[list(baselines.values())].notna().all(axis=1)]
    for name, col in baselines.items():
        rows.append({"split": split, "model": name, "n": len(part),
                     **regression_metrics(part["checkins"], part[col])})
res = pd.DataFrame(rows)
C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
res.to_csv(C.REPORTS_DIR / "baseline_metrics.csv", index=False)
print(res.round(3).to_string(index=False))
C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
df.to_parquet(C.PROCESSED_DIR / "features.parquet", index=False)
print("saved features:", df.shape)
