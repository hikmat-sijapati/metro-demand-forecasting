"""Train, tune and evaluate the approved models + baselines on the chronological split.

  python src/06_models.py            # full run
  python src/06_models.py --fast     # small grids/subsamples: for debugging only
  python src/06_models.py --models ridge bayes rf      # subset

Protocol: hyper-parameters are chosen by rolling-origin (expanding-window) CV on train+val
(time-ordered subsample, with folds grouped by date); the chosen model is refit on ALL train+val
rows and scored ONCE on the test period. Baselines are scored on exactly the same rows.
Outputs go to reports/results/.
Notes: ridge/lasso/SVR use a standardised target and standardised numeric features; station/hour/
weekday are one-hot for those models and integer-coded for the forest. LinearSVR uses the squared
epsilon-insensitive loss (liblinear primal solver) for speed. Predictions are clipped at 0.
"""
import argparse, json, platform, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, Lasso, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR, LinearSVR

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from utils import regression_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--fast", action="store_true")
ap.add_argument("--svr-rows", type=int, default=None, help="override the RBF-SVR training subsample size")
ap.add_argument("--models", nargs="+", default=["ridge", "lasso", "bayes", "rf", "svr_linear", "svr_rbf"])
a = ap.parse_args()
warnings.filterwarnings("ignore")
P, hourly, SEED = C.P, C.GRAIN == "hour", C.SEED
RES = C.REPORTS_DIR / "results"; RES.mkdir(parents=True, exist_ok=True)
tune_rows = 15_000 if a.fast else C.TUNE_ROWS
svr_rows = a.svr_rows or (6_000 if a.fast else C.SVR_RBF_TRAIN_ROWS)

df = pd.read_parquet(C.PROCESSED_DIR / "features.parquet")
lag_cols = [f"lag_{l}" for l in P["lags"]] + ["lag_mean_4w"]
df["month_sin"], df["month_cos"] = np.sin(2 * np.pi * df["month"] / 12), np.cos(2 * np.pi * df["month"] / 12)
if hourly:
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * df["hour"] / 24), np.cos(2 * np.pi * df["hour"] / 24)
cat_cols = ["station", "dow"] + (["hour"] if hourly else [])
num_cols = lag_cols + ["month_sin", "month_cos"] + (["hour_sin", "hour_cos"] if hourly else [])
bin_cols = ["is_weekend", "is_holiday"]
geo = [c for c in ("lat", "lon") if c in df.columns]
rf_cols = lag_cols + bin_cols + ["dow", "month", "station_id"] + (["hour"] if hourly else []) + geo
if geo:
    df[geo] = df[geo].fillna(-1)

ok = df[lag_cols].notna().all(axis=1) & df["checkins"].notna()
tv = df[ok & df["split"].isin(["train", "val"])].sort_values("ts").reset_index(drop=True)
te = df[ok & (df["split"] == "test")].sort_values(["ts", "station"]).reset_index(drop=True)
if tv.empty or te.empty:
    sys.exit("Empty train/test after dropping rows with missing lags. Check the window / data coverage.")
print(f"grain={C.GRAIN} | train+val rows {len(tv):,} | test rows {len(te):,} | fast={a.fast}")


def sub(d, n):
    return d if len(d) <= n else d.sample(n=n, random_state=SEED).sort_values("ts").reset_index(drop=True)


def linear_pre():
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
                              ("num", StandardScaler(), num_cols), ("bin", "passthrough", bin_cols)])


def scaled(pipe):
    return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())


def date_blocked_splits(data, n_splits):
    """Create expanding-window folds without splitting observations from one date."""
    dates = np.sort(pd.to_datetime(data["date"]).unique())
    if len(dates) <= n_splits:
        raise ValueError(f"Need more than {n_splits} distinct dates for date-blocked CV; found {len(dates)}.")
    date_values = pd.to_datetime(data["date"]).to_numpy()
    return [
        (np.flatnonzero(np.isin(date_values, dates[train_dates])),
         np.flatnonzero(np.isin(date_values, dates[test_dates])))
        for train_dates, test_dates in TimeSeriesSplit(n_splits=n_splits).split(dates)
    ]


SCORE = "neg_root_mean_squared_error"
specs = {
    "ridge": (scaled(Pipeline([("pre", linear_pre()), ("m", Ridge())])), {"regressor__m__alpha": [0.1, 1, 10, 100]}, "grid"),
    "lasso": (scaled(Pipeline([("pre", linear_pre()), ("m", Lasso(max_iter=5000))])),
              {"regressor__m__alpha": [0.001, 0.01, 0.1]}, "grid"),
    "svr_linear": (scaled(Pipeline([("pre", linear_pre()), ("m", LinearSVR(loss="squared_epsilon_insensitive", dual=False,
                    max_iter=5000, random_state=SEED))])), {"regressor__m__C": [0.01, 0.1, 1]}, "grid"),
    "svr_rbf": (scaled(Pipeline([("pre", linear_pre()), ("m", SVR(kernel="rbf", epsilon=0.05))])),
                {"regressor__m__C": [1, 10, 100]}, "grid"),
    "rf": (RandomForestRegressor(n_estimators=60 if a.fast else 150, n_jobs=-1, random_state=SEED),
           {"max_depth": [None, 20, 30], "min_samples_leaf": [1, 3, 10], "max_features": [0.4, 0.7, 1.0]}, "random"),
}
names = {"ridge": "Ridge", "lasso": "Lasso", "bayes": "Bayesian ridge", "rf": "Random Forest",
         "svr_linear": "SVR (linear)", "svr_rbf": "SVR (RBF)"}

out = te[["station", "ts", "date", "dow", "is_holiday", "checkins"]].rename(columns={"checkins": "y"}).copy()
if hourly:
    out["hour"] = te["hour"].values
out["seasonal_naive"], out["same_period_yesterday"] = te[f"lag_{P['week']}"].values, te[f"lag_{P['day']}"].values
out["seasonal_mean_4w"], out["historical_avg"] = te["lag_mean_4w"].values, te["hist_avg"].values

rows, best_params, extra = [], {}, {}

def save_all():
    base_rows = [{"model": k, "name": k, **regression_metrics(out["y"], out[k]), "cv_rmse": np.nan, "fit_seconds": 0, "n": len(out)}
                 for k in ("seasonal_naive", "same_period_yesterday", "seasonal_mean_4w", "historical_avg")]
    pd.DataFrame(rows + base_rows).to_csv(RES / "metrics_test.csv", index=False)
    out.to_parquet(RES / "test_predictions.parquet", index=False)
    rng = lambda d: [str(d.min().date()), str(d.max().date())]
    (RES / "run_info.json").write_text(json.dumps({
        "grain": C.GRAIN, "fast": a.fast, "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "best_params": best_params, "train_rows": int((tv["split"] == "train").sum()),
        "val_rows": int((tv["split"] == "val").sum()), "test_rows": int(len(te)),
        "train_range": rng(tv.loc[tv.split == "train", "date"]), "val_range": rng(tv.loc[tv.split == "val", "date"]),
        "test_range": rng(te["date"]), "n_stations": int(df["station"].nunique()), "n_rows_modelling_table": int(len(df)),
        "tune_rows": tune_rows, "svr_rbf_train_rows": extra.get("svr_rbf_train_rows"), "models_run": [r["model"] for r in rows]}, indent=1))



for key in a.models:
    t0 = time.time()
    if key == "bayes":
        pre = linear_pre()
        Xtr = pre.fit_transform(tv); ytr = tv["checkins"].to_numpy(float)
        mu, sd = ytr.mean(), ytr.std()
        br = BayesianRidge().fit(Xtr, (ytr - mu) / sd)
        m, s = br.predict(pre.transform(te), return_std=True)
        pred, out["bayes_sigma"] = m * sd + mu, s * sd
        pd.DataFrame({"feature": pre.get_feature_names_out(), "coef_standardised_target": br.coef_}).assign(
            abs=lambda d: d.coef_standardised_target.abs()).sort_values("abs", ascending=False).drop(columns="abs") \
          .to_csv(RES / "bayes_coefficients.csv", index=False)
        bp, cvr = {"alpha_": float(br.alpha_), "lambda_": float(br.lambda_)}, np.nan
    else:
        est, grid, kind = specs[key]
        train = sub(tv, svr_rows) if key == "svr_rbf" else tv
        tune = sub(train, tune_rows if key != "svr_rbf" else svr_rows)
        Xt = tune[rf_cols] if key == "rf" else tune
        cv_splits = date_blocked_splits(tune, C.CV_SPLITS)
        print(f"  [{names[key]}] tuning on {len(tune):,} rows, final fit on {len(train):,} rows, then predicting {len(te):,} test rows ...", flush=True)
        search = (RandomizedSearchCV(est, grid, n_iter=3 if a.fast else 6, cv=cv_splits, scoring=SCORE,
                                     n_jobs=-1 if key == "rf" else 1, random_state=SEED, refit=False)
                  if kind == "random" else GridSearchCV(est, grid, cv=cv_splits, scoring=SCORE, refit=False, n_jobs=1, verbose=2 if key == "svr_rbf" else 0))
        search.fit(Xt, tune["checkins"])
        bp, cvr = search.best_params_, float(-search.best_score_)
        print(f"  [{names[key]}] best params found: {bp} -> final fit ...", flush=True)
        model = clone(est).set_params(**bp)
        if key == "rf":
            model.set_params(max_samples=(C.RF_MAX_SAMPLES / len(train)) if len(train) > C.RF_MAX_SAMPLES else None)
            model.fit(train[rf_cols], train["checkins"])
            pred = model.predict(te[rf_cols])
            pd.DataFrame({"feature": rf_cols, "importance": model.feature_importances_}).sort_values(
                "importance", ascending=False).to_csv(RES / "rf_importance.csv", index=False)
        else:
            model.fit(train, train["checkins"])
            pred = model.predict(te)
            if key == "ridge":
                pre = model.regressor_.named_steps["pre"]
                pd.DataFrame({"feature": pre.get_feature_names_out(),
                              "coef_standardised_target": model.regressor_.named_steps["m"].coef_}).assign(
                    abs=lambda d: d.coef_standardised_target.abs()).sort_values("abs", ascending=False) \
                  .drop(columns="abs").to_csv(RES / "ridge_coefficients.csv", index=False)
        if key == "svr_rbf":
            extra["svr_rbf_train_rows"] = int(len(train))
    pred = np.clip(pred, 0, None)
    out[key] = pred
    fit_s = time.time() - t0
    m_ = regression_metrics(te["checkins"], pred)
    rows.append({"model": key, "name": names[key], **m_, "cv_rmse": cvr, "fit_seconds": round(fit_s, 1), "n": len(te)})
    best_params[key] = {k: (v if isinstance(v, (int, float, str)) or v is None else str(v)) for k, v in bp.items()}
    print(f"{names[key]:<16} RMSE {m_['RMSE']:.3f} | MAE {m_['MAE']:.3f} | R2 {m_['R2']:.3f} | WAPE {m_['WAPE']:.3f} | {fit_s:.0f}s | best {best_params[key]}", flush=True)
    save_all()

save_all()
print("saved ->", RES)
