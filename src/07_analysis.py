"""Model comparison, bootstrap test, error slices, uncertainty check, figures, placeholder values.

  python src/07_analysis.py

Reads reports/results/{metrics_test.csv,test_predictions.parquet,run_info.json} (from 06_models.py)
and reports/data_quality.json (from 03_aggregate.py combine). Writes CSVs/JSON to reports/results/
and PNG figures to reports/figures/. `placeholder_values.json` feeds 08_fill_report.py.
"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

RES, FIGS = C.REPORTS_DIR / "results", C.FIGS_DIR
FIGS.mkdir(parents=True, exist_ok=True)
hourly = C.GRAIN == "hour"
met = pd.read_csv(RES / "metrics_test.csv").set_index("model")
pred = pd.read_parquet(RES / "test_predictions.parquet")
info = json.loads((RES / "run_info.json").read_text())
dq = json.loads((C.REPORTS_DIR / "data_quality.json").read_text()) if (C.REPORTS_DIR / "data_quality.json").exists() else {}
plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "legend.fontsize": 6})

DISP = {"ridge": "Ridge regression", "lasso": "Lasso regression", "bayes": "Bayesian ridge", "rf": "Random Forest",
        "svr_linear": "SVR (linear)", "svr_rbf": "SVR (RBF)", "seasonal_naive": "seasonal naive (same period one week earlier)",
        "same_period_yesterday": "same period yesterday", "seasonal_mean_4w": "four-week seasonal mean",
        "historical_avg": "historical average"}
approved = [m for m in ("ridge", "lasso", "bayes", "rf", "svr_linear", "svr_rbf") if m in met.index]
baselines = ["seasonal_naive", "same_period_yesterday", "seasonal_mean_4w", "historical_avg"]
lin = [m for m in ("ridge", "lasso") if m in met.index]
linear_best = min(lin, key=lambda m: met.loc[m, "cv_rmse"]) if lin else None      # chosen by CV, not by test error
families = ([linear_best] if linear_best else []) + [m for m in ("bayes", "rf", "svr_linear", "svr_rbf") if m in met.index]
best = min(families, key=lambda m: met.loc[m, "RMSE"])
best_base = min(baselines, key=lambda m: met.loc[m, "RMSE"])
rm_best, rm_base = met.loc[best, "RMSE"], met.loc[best_base, "RMSE"]
improve = (rm_base - rm_best) / rm_base * 100

# ---- Table 2: report every baseline and every fitted model variant ----
t2_keys = baselines + approved
t2_labels = ["Seasonal naive (t − 1 week)", "Same hour yesterday", "Four-week seasonal mean", "Historical average",
             "Ridge", "Lasso", "Bayesian ridge", "Random Forest", "SVR (linear)", "SVR (RBF)"]
t2 = pd.DataFrame([{"label": l, "key": k, "RMSE": met.loc[k, "RMSE"], "MAE": met.loc[k, "MAE"], "R2": met.loc[k, "R2"],
                    "WAPE": met.loc[k, "WAPE"] * 100} for l, k in zip(t2_labels, t2_keys) if k in met.index])
t2.to_csv(RES / "table2.csv", index=False)

# ---- paired block bootstrap over test dates: baseline RMSE minus model RMSE ----
rng = np.random.default_rng(C.SEED)
g = pd.DataFrame({"d": pred["date"], "sa": (pred["y"] - pred[best_base]) ** 2, "sb": (pred["y"] - pred[best]) ** 2}) \
      .groupby("d").agg(sa=("sa", "sum"), sb=("sb", "sum"), n=("sa", "size"))
sa, sb, n = g["sa"].to_numpy(), g["sb"].to_numpy(), g["n"].to_numpy()
idx = rng.integers(0, len(g), size=(C.BOOTSTRAP_B, len(g)))
delta = np.sqrt(sa[idx].sum(1) / n[idx].sum(1)) - np.sqrt(sb[idx].sum(1) / n[idx].sum(1))
boot = {"model": best, "baseline": best_base, "B": C.BOOTSTRAP_B, "n_dates": int(len(g)), "mean_rmse_reduction": float(delta.mean()),
        "ci95": [float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))], "share_model_better": float((delta > 0).mean())}
boot["significant"] = bool(boot["ci95"][0] > 0)
(RES / "bootstrap.json").write_text(json.dumps(boot, indent=1))

# ---- error slices for the best model vs the best baseline ----
d = pred.copy()
for tag, col in (("m", best), ("b", best_base)):
    d[f"se_{tag}"], d[f"ae_{tag}"] = (d["y"] - d[col]) ** 2, (d["y"] - d[col]).abs()


def slice_by(by):
    gg = d.groupby(by).agg(n=("y", "size"), mean_actual=("y", "mean"), total=("y", "sum"), se_m=("se_m", "mean"), se_b=("se_b", "mean"),
                           ae_m=("ae_m", "sum"), ae_b=("ae_b", "sum"))
    return pd.DataFrame({"n": gg["n"], "mean_actual": gg["mean_actual"], "RMSE_model": np.sqrt(gg["se_m"]), "WAPE_model": gg["ae_m"] / gg["total"],
                         "RMSE_baseline": np.sqrt(gg["se_b"]), "WAPE_baseline": gg["ae_b"] / gg["total"]})


by_station = slice_by("station").sort_values("WAPE_model", ascending=False)
by_station.to_csv(RES / "errors_by_station.csv")
d["holiday"] = d["is_holiday"].map({0: "regular day", 1: "public holiday"})
slice_by("holiday").to_csv(RES / "errors_by_holiday.csv")
slice_by("dow").to_csv(RES / "errors_by_weekday.csv")
if hourly:
    by_hour = slice_by("hour"); by_hour.to_csv(RES / "errors_by_hour.csv")
    d["peak"] = np.where(d["hour"].isin(C.PEAK_HOURS), "peak", "off-peak")
    slice_by("peak").to_csv(RES / "errors_by_peak.csv")
    worst_t = by_hour["RMSE_model"].nlargest(2).index.tolist()
    peak_txt = " and ".join(f"{int(h):02d}:00" for h in sorted(worst_t))
else:
    by_hour = slice_by("dow")
    peak_txt = ", ".join(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][int(i)] for i in by_hour["RMSE_model"].nlargest(2).index)
big = by_station[by_station["mean_actual"] >= by_station["mean_actual"].median()]
worst_st = "; ".join(f"{s} (WAPE {r.WAPE_model * 100:.0f}%)" for s, r in big.head(3).iterrows())

# ---- Bayesian interval calibration ----
cov = width = None
if "bayes_sigma" in pred and "bayes" in pred:
    z = norm.ppf(0.5 + C.PI_LEVEL / 2)
    lo, hi = np.clip(pred["bayes"] - z * pred["bayes_sigma"], 0, None), pred["bayes"] + z * pred["bayes_sigma"]
    cov, width = float(((pred["y"] >= lo) & (pred["y"] <= hi)).mean()), float((hi - lo).mean())
    (RES / "bayes_intervals.json").write_text(json.dumps({"level": C.PI_LEVEL, "coverage": cov, "mean_width": width}, indent=1))

# ---- figures ----
st = pred.groupby("station")["y"].sum().idxmax()
sdf = pred[pred["station"] == st].sort_values("ts")
last = sdf["date"].max()
span = 7 if hourly else 28
sdf = sdf[sdf["date"] > last - pd.Timedelta(days=span)]
fig, ax = plt.subplots(figsize=(4.6, 3.0))
if cov is not None:
    z = norm.ppf(0.5 + C.PI_LEVEL / 2)
    ax.fill_between(sdf["ts"], np.clip(sdf["bayes"] - z * sdf["bayes_sigma"], 0, None), sdf["bayes"] + z * sdf["bayes_sigma"],
                    color="#9ecae1", alpha=0.5, label=f"Bayesian ridge {int(C.PI_LEVEL * 100)}% interval")
ax.plot(sdf["ts"], sdf["y"], color="black", lw=0.9, label="Actual")
ax.plot(sdf["ts"], sdf[best], color="#d95f02", lw=0.9, label=DISP[best])
ax.plot(sdf["ts"], sdf[best_base], color="grey", lw=0.8, ls="--", label=DISP[best_base].split(" (")[0])
ax.set_ylabel("Check-ins per " + ("hour" if hourly else "day")); ax.set_title(st); ax.legend(loc="upper left", ncol=2)
fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(FIGS / "fig1a_forecast_vs_actual.png", dpi=220); plt.close(fig)
test_week = f"{sdf['date'].min().date()} to {sdf['date'].max().date()}"

fig, ax = plt.subplots(figsize=(4.6, 3.0))
xs = by_hour.index.astype(int)
ax.plot(xs, by_hour["RMSE_model"], marker="o", ms=3, color="#d95f02", label=DISP[best])
ax.plot(xs, by_hour["RMSE_baseline"], marker="s", ms=3, color="grey", ls="--", label=DISP[best_base].split(" (")[0])
ax.set_xlabel("Hour of day" if hourly else "Weekday (0 = Monday)"); ax.set_ylabel("RMSE"); ax.legend()
ax.set_title("Error by " + ("hour of day" if hourly else "weekday")); fig.tight_layout()
fig.savefig(FIGS / "fig1b_error_by_hour.png", dpi=220); plt.close(fig)

top_feat = []
if (RES / "rf_importance.csv").exists():
    imp = pd.read_csv(RES / "rf_importance.csv").head(12).iloc[::-1]
    top_feat = imp["feature"].tail(4).tolist()[::-1]
    fig, ax = plt.subplots(figsize=(4.6, 3.0)); ax.barh(imp["feature"], imp["importance"], color="#1b9e77")
    ax.set_xlabel("Random Forest importance"); ax.set_title("Top features"); fig.tight_layout()
    fig.savefig(FIGS / "fig_feature_importance.png", dpi=220); plt.close(fig)

# ---- placeholder values ----
fam_rank = sorted(approved, key=lambda m: met.loc[m, "RMSE"])
ranking = ", then ".join(DISP[m] for m in fam_rank) + " (lowest to highest RMSE)"
gap = (met.loc[linear_best, "RMSE"] - rm_best) / rm_best * 100 if linear_best else float("nan")
bp = info["best_params"]
hp = "; ".join(f"{DISP.get(k, k)} {', '.join(f'{p.split(chr(95) * 2)[-1]}={v}' for p, v in vals.items() if not p.endswith('_'))}"
               for k, vals in bp.items() if vals and k != "bayes")
fmt_n = lambda x: f"{int(x):,}"
rows_read, chunks = dq.get("rows_read"), dq.get("chunks_processed")
chunk_total = dq.get("chunks_total", C.total_chunks())
legacy_chunks = dq.get("legacy_chunks_in_accumulator", 0)
chunk_coverage = (f"{chunks:,} of {chunk_total:,} currently indexed chunk files processed"
                  if chunks is not None else "[chunk coverage unavailable]")
if legacy_chunks:
    chunk_coverage += f"; {legacy_chunks:,} older accumulator files retained"
rows_read_text = f"{rows_read:,} rows across {dq.get('accumulator_chunks', chunks):,} accumulator files" if rows_read else "[n/a]"
vals = {
    "BEST_MODEL": DISP[best], "RMSE_BEST": f"{rm_best:.2f}", "WAPE_BEST": f"{met.loc[best, 'WAPE'] * 100:.1f}%",
    "BEST_BASELINE": DISP[best_base], "RMSE_BASELINE_BEST": f"{rm_base:.2f}", "IMPROVEMENT": f"{abs(improve):.1f}", "_beat_baseline": "yes" if improve > 0 else "no",
    "RANKING": ranking, "GAP": f"{gap:.1f}", "N_TEST_ROWS": fmt_n(info["test_rows"]), "N_ROWS": fmt_n(info["n_rows_modelling_table"]),
    "N_STATIONS": str(info["n_stations"]), "TRAIN_RANGE": " to ".join(info["train_range"]), "VAL_RANGE": " to ".join(info["val_range"]),
    "TEST_RANGE": " to ".join(info["test_range"]), "STATION": st, "TEST_WEEK": test_week, "PEAK_HOURS": peak_txt,
    "WORST_STATIONS": worst_st, "TOP_FEATURES": ", ".join(top_feat) if top_feat else "[run 06 with rf]",
    "CI": f"(95% CI for the RMSE reduction {boot['ci95'][0]:.2f} to {boot['ci95'][1]:.2f})",
    "SIGNIFICANT / NOT SIGNIFICANT": "significant" if boot["significant"] else "not significant",
    "PAIRED_BOOTSTRAP_RESULT": (f"A paired block bootstrap over test dates ({C.BOOTSTRAP_B:,} resamples) puts the RMSE reduction of {DISP[best]} "
                                f"over the best baseline at {boot['mean_rmse_reduction']:.2f} (95% CI {boot['ci95'][0]:.2f} to {boot['ci95'][1]:.2f})"),
    "PYTHON_VERSION": info["python"], "HARDWARE": f"{info['platform']}, {info['processor'] or 'CPU n/a'}, {os.cpu_count()} logical cores",
    "SVR_SUBSAMPLE_SIZE": f"of {fmt_n(info['svr_rbf_train_rows'])} rows" if info.get("svr_rbf_train_rows") else "",
    "FINAL_HYPERPARAMETERS": f"(selected: {hp})", "WINDOW_START": dq.get("window", [C.WINDOW_START, C.WINDOW_END])[0],
    "WINDOW_END": dq.get("window", [C.WINDOW_START, C.WINDOW_END])[1],
    "CHUNKS_PROCESSED": chunk_coverage,
    "ROWS_READ": rows_read_text,
    "PLACEHOLDER_SHARE": f"{dq['time_unknown_share'] * 100:.1f}" if "time_unknown_share" in dq else "[n/a]",
    "TOTAL_ROWS_READ": (f"{dq['rows_read']:,}" + ("" if dq.get("chunks_processed") == dq.get("chunks_total") else
                        f" read from {chunks:,} of {dq['chunks_total']:,} chunks")) if rows_read else "[n/a]",
}
if cov is not None:
    vals["COVERAGE"] = f"{cov * 100:.1f}"
    vals["WELL_CALIBRATED / OVERCONFIDENT"] = ("well calibrated" if abs(cov - C.PI_LEVEL) <= 0.03 else
                                               "overconfident" if cov < C.PI_LEVEL else "conservative")
(RES / "placeholder_values.json").write_text(json.dumps(vals, indent=1))

warn = []
if improve <= 0:
    warn.append(f"WARNING: {DISP[best]} does NOT beat the best baseline ({DISP[best_base]}). Rewrite the results narrative honestly: "
                "say so, and discuss why (strong seasonality, little non-linear structure, or too little training data).")
if not boot["significant"]:
    warn.append("NOTE: the bootstrap interval for the RMSE reduction includes 0 (or is negative): do not claim a significant improvement.")
summary = warn + [f"# Results summary ({C.GRAIN} grain, FAST={info['fast']})", "",
           f"Best approved model: {DISP[best]} (RMSE {rm_best:.3f}); best baseline: {DISP[best_base]} (RMSE {rm_base:.3f}); improvement {improve:.1f}%.",
           f"Ranking: {ranking}.", f"Bootstrap: {json.dumps(boot)}", "",
           f"Current index coverage: {chunk_coverage}. Rows read: {rows_read_text}.",
           "## Table 2", t2.round(3).to_string(index=False), "",
           "## Worst stations (high volume)", big.head(5).round(3).to_string(),
           "", f"Bayesian {int(C.PI_LEVEL * 100)}% interval coverage: {cov}", f"Top RF features: {top_feat}"]
(RES / "results_summary.md").write_text("\n".join(summary))
print("\n".join(summary))
print("\nsaved placeholder values for", len(vals), "report fields ->", RES / "placeholder_values.json")
