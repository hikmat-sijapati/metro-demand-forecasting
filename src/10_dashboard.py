"""Local Gradio explorer for saved held-out station-hour predictions.

Run after src/06_models.py:
    python src/10_dashboard.py

This presents historical test-set forecasts, not live or future predictions.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

RES = C.REPORTS_DIR / "results"
MODEL_LABELS = {
    "ridge": ("Ridge", "Regularised / Bayesian linear"),
    "lasso": ("Lasso", "Regularised / Bayesian linear"),
    "bayes": ("Bayesian ridge", "Regularised / Bayesian linear"),
    "rf": ("Random Forest", "Random Forest"),
    "svr_linear": ("SVR (linear)", "Support Vector Regression"),
    "svr_rbf": ("SVR (RBF)", "Support Vector Regression"),
}
BASELINE_LABELS = {
    "seasonal_naive": "Seasonal naive (one week)",
    "same_period_yesterday": "Same hour yesterday",
    "seasonal_mean_4w": "Four-week seasonal mean",
    "historical_avg": "Historical average",
}
MODEL_FAMILY = {
    "ridge": "Regularised / Bayesian linear",
    "lasso": "Regularised / Bayesian linear",
    "bayes": "Regularised / Bayesian linear",
    "rf": "Random Forest",
    "svr_linear": "Support Vector Regression",
    "svr_rbf": "Support Vector Regression",
}
RANGE_CANDIDATES = [
    {
        "label": "Primary: stable pre-pandemic",
        "start": "2018-10-01",
        "end": "2020-01-31",
        "reason": "Selected: long, comparatively stable period with useful hourly coverage; matches the report's evaluated model window.",
    },
    {
        "label": "Pandemic recovery / Expo period",
        "start": "2021-09-01",
        "end": "2022-03-31",
        "reason": "Good hourly coverage, but pandemic recovery and Expo-era changes make it a distinct operating regime.",
    },
    {
        "label": "Recent hourly block",
        "start": "2024-08-01",
        "end": "2025-02-28",
        "reason": "More recent, but coverage is fragmented and includes missing or unreliable days.",
    },
    {
        "label": "Latest archive block",
        "start": "2026-04-01",
        "end": "2026-07-11",
        "reason": "Latest transactions in the archive, but too short and fragmented for stable lagged training and evaluation.",
    },
]
ERROR_FILES = {
    "Hour": ("errors_by_hour.csv", "hour"),
    "Weekday": ("errors_by_weekday.csv", "dow"),
    "Holiday": ("errors_by_holiday.csv", None),
    "Station": ("errors_by_station.csv", "station"),
}


def load_results():
    predictions_path = RES / "test_predictions.parquet"
    metrics_path = RES / "metrics_test.csv"
    info_path = RES / "run_info.json"
    missing = [str(p) for p in (predictions_path, metrics_path, info_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Saved model results are missing: " + ", ".join(missing)
            + ". Run `python src/06_models.py` first."
        )

    import json

    info = json.loads(info_path.read_text())
    if info.get("grain") != "hour":
        raise ValueError("This dashboard requires hourly results; rerun the pipeline with METRO_GRAIN=hour.")

    predictions = pd.read_parquet(predictions_path)
    metrics = pd.read_csv(metrics_path)
    required = {"date", "ts", "station", "hour", "y", *MODEL_LABELS, *BASELINE_LABELS}
    absent = sorted(required - set(predictions.columns))
    if absent:
        raise ValueError(
            f"Saved test predictions are missing columns: {', '.join(absent)}. "
            "Run the full model set with `python src/06_models.py`."
        )
    predictions["date"] = pd.to_datetime(predictions["date"]).dt.normalize()
    predictions["hour"] = predictions["hour"].astype(int)
    metrics = metrics[metrics["model"].isin([*MODEL_LABELS, *BASELINE_LABELS])].copy()
    return predictions, metrics, info


def format_metrics(metrics):
    labels = {**{k: v[0] for k, v in MODEL_LABELS.items()}, **BASELINE_LABELS}
    families = {**{k: v[1] for k, v in MODEL_LABELS.items()}, **{k: "Baseline" for k in BASELINE_LABELS}}
    shown = metrics.copy()
    shown["Method"] = shown["model"].map(labels)
    shown["Family"] = shown["model"].map(families)
    shown["WAPE (%)"] = shown["WAPE"] * 100
    columns = ["Family", "Method", "RMSE", "MAE", "R2", "WAPE (%)"]
    return shown[columns].rename(columns={"R2": "R²"}).round(3)


def load_range_quality():
    """Summarize candidate periods from the current accumulator, independently of model scores."""
    import duckdb

    db_path = C.INTERIM_DIR / "counts.duckdb"
    if not db_path.exists():
        return pd.DataFrame(), None, (
            f"Range diagnostics unavailable: {db_path} was not found. "
            "The saved model comparison remains available."
        )

    con = duckdb.connect(str(db_path), read_only=True)
    daily = con.execute(
        """
        SELECT date,
               sum(n) FILTER (WHERE kind = 'in' AND hour >= 0) AS known_time_in,
               sum(n) FILTER (WHERE kind = 'in') AS total_in,
               count(DISTINCT station) FILTER (WHERE kind = 'in' AND hour >= 0) AS stations_with_time
        FROM counts
        GROUP BY date
        ORDER BY date
        """
    ).fetchdf()
    con.close()
    if daily.empty:
        raise ValueError(f"No daily rows found in {db_path}.")

    daily["date"] = pd.to_datetime(daily["date"])
    calendar = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = daily.set_index("date").reindex(calendar)
    daily.index.name = "date"
    daily["known_share"] = daily["known_time_in"] / daily["total_in"].replace(0, np.nan)
    median_volume = daily["total_in"].rolling(29, center=True, min_periods=7).median()
    daily["usable"] = (
        daily["known_share"].ge(C.MIN_KNOWN_TIME_SHARE)
        & daily["total_in"].ge(C.LOW_DAY_RATIO * median_volume)
        & daily["stations_with_time"].gt(0)
    )
    latest = daily.index.max().date().isoformat()

    summaries = []
    for candidate in RANGE_CANDIDATES:
        start, end = pd.Timestamp(candidate["start"]), pd.Timestamp(candidate["end"])
        part = daily.loc[start:end]
        usable = part["usable"].fillna(False).astype(bool)
        longest_run = current_run = 0
        for value in usable:
            current_run = current_run + 1 if value else 0
            longest_run = max(longest_run, current_run)
        summaries.append({
            "Candidate": candidate["label"],
            "Dates": f"{candidate['start']} to {candidate['end']}",
            "Calendar days": len(part),
            "Recorded days": int(part["total_in"].notna().sum()),
            "Usable hourly days": int(usable.sum()),
            "Usable-day share (%)": round(100 * usable.mean(), 1) if len(part) else 0,
            "Longest continuous usable run (days)": longest_run,
            "Median daily check-ins": round(float(part["total_in"].median()), 0)
            if part["total_in"].notna().any() else np.nan,
            "Rationale": candidate["reason"],
        })
    return pd.DataFrame(summaries), latest, (
        "Quality screen: at least 90% known-time check-ins, plus daily volume at least "
        "30% of its centered 29-day rolling median. This is an archive-quality comparison, "
        "not a head-to-head model evaluation. The accumulator combines chunk exports; "
        "export overlap can affect absolute volumes, so interpret counts as indicative."
    )


def build_range_chart(range_quality):
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = range_quality["Candidate"].tolist()
    values = range_quality["Usable hourly days"].to_numpy()
    bars = ax.barh(labels, values, color=["#167d8d", "#8aa6a3", "#8aa6a3", "#c79a48"])
    ax.set_xlabel("Quality-screened days")
    ax.set_title("Hourly data availability across candidate periods")
    ax.grid(axis="x", alpha=0.25)
    ax.bar_label(bars, padding=4, fmt="%.0f")
    fig.tight_layout()
    plt.close(fig)
    return fig


def build_performance_chart(metrics, family, metric):
    model_rows = metrics[metrics["model"].isin(MODEL_LABELS)].copy()
    model_rows["family"] = model_rows["model"].map(MODEL_FAMILY)
    if family != "All model families":
        model_rows = model_rows[model_rows["family"] == family]
    if metric not in {"RMSE", "MAE", "WAPE"}:
        raise ValueError(f"Unsupported performance metric: {metric}")
    model_rows["display"] = model_rows["model"].map({k: v[0] for k, v in MODEL_LABELS.items()})

    baseline = metrics[metrics["model"].isin(BASELINE_LABELS)].copy()
    baseline["display"] = baseline["model"].map(BASELINE_LABELS)
    shown = pd.concat([model_rows[["display", metric]], baseline[["display", metric]],], ignore_index=True)
    if metric == "WAPE":
        shown[metric] = shown[metric] * 100
    shown = shown.sort_values(metric, ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(3.2, 0.42 * len(shown))))
    colors = ["#167d8d" if name in model_rows["display"].values else "#999999"
              for name in shown["display"]]
    bars = ax.barh(shown["display"], shown[metric], color=colors)
    ax.set_xlabel(f"{metric} (%)" if metric == "WAPE" else metric)
    ax.set_title(f"Test-set {metric} by model and baseline")
    ax.grid(axis="x", alpha=0.25)
    ax.bar_label(bars, padding=4, fmt="%.2f")
    fig.tight_layout()
    plt.close(fig)
    return shown.rename(columns={"display": "Method", metric: f"{metric} (%)" if metric == "WAPE" else metric}), fig


def load_error_slice(category, metric):
    filename, index_column = ERROR_FILES[category]
    path = RES / filename
    if not path.exists():
        raise FileNotFoundError(f"Error-slice results are missing: {path}. Run `python src/07_analysis.py`.")
    df = pd.read_csv(path)
    if index_column and index_column in df.columns:
        df = df.set_index(index_column).reset_index()
    elif "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "Group"})
    if metric not in {"RMSE_model", "WAPE_model"}:
        raise ValueError(f"Unsupported error metric: {metric}")
    if metric == "WAPE_model":
        df[metric] = df[metric] * 100
    label_column = index_column or ("holiday" if "holiday" in df.columns else "Group")
    df = df.sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(3.2, 0.34 * min(len(df), 18))))
    chart_df = df.head(18).sort_values(metric)
    bars = ax.barh(chart_df[label_column].astype(str), chart_df[metric], color="#d27942")
    ax.set_xlabel("WAPE (%)" if metric == "WAPE_model" else "RMSE")
    ax.set_title(f"Test-set error by {category.lower()}")
    ax.grid(axis="x", alpha=0.25)
    ax.bar_label(bars, padding=4, fmt="%.2f")
    fig.tight_layout()
    plt.close(fig)
    return df.round(3), fig


def build_forecast_view(predictions, date, station, hour_label, chart_model, info):
    if not date or not station or not hour_label or not chart_model:
        raise ValueError("Choose a test date, station, hour, and model.")
    if chart_model not in MODEL_LABELS:
        raise ValueError(f"Unknown model: {chart_model}")
    hour = int(hour_label[:2])
    selected = predictions[
        (predictions["date"] == pd.Timestamp(date))
        & (predictions["station"] == station)
        & (predictions["hour"] == hour)
    ]
    if len(selected) != 1:
        raise ValueError(f"No unique held-out prediction exists for {station}, {date}, {hour:02d}:00.")
    row = selected.iloc[0]

    forecast_rows = []
    for key, (label, family) in MODEL_LABELS.items():
        forecast_rows.append({
            "Family": family,
            "Method": label,
            "Forecast": float(row[key]),
            "Actual": float(row["y"]),
            "Forecast error": float(row[key] - row["y"]),
        })
    for key, label in BASELINE_LABELS.items():
        forecast_rows.append({
            "Family": "Baseline",
            "Method": label,
            "Forecast": float(row[key]),
            "Actual": float(row["y"]),
            "Forecast error": float(row[key] - row["y"]),
        })
    forecast_table = pd.DataFrame(forecast_rows).round(2)

    day = predictions[
        (predictions["date"] == pd.Timestamp(date))
        & (predictions["station"] == station)
    ].sort_values("hour")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(day["hour"], day["y"], marker="o", linewidth=2, label="Actual")
    ax.plot(day["hour"], day[chart_model], marker=".", linewidth=1.5,
            label=MODEL_LABELS[chart_model][0])
    ax.plot(day["hour"], day["seasonal_mean_4w"], linestyle="--", linewidth=1.3,
            label=BASELINE_LABELS["seasonal_mean_4w"])
    ax.axvline(hour, color="grey", linestyle=":", linewidth=1)
    ax.set(xticks=range(24), xlabel="Hour of day", ylabel="Check-ins",
           title=f"{station} — {date} (held-out test day)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    plt.close(fig)

    summary = (
        f"**Selected forecast:** {station}, {date} at {hour:02d}:00  \n"
        f"**Actual check-ins:** {row['y']:,.0f}  \n"
        f"**{MODEL_LABELS[chart_model][0]} forecast:** {row[chart_model]:,.1f}  \n"
        f"**Test period:** {info['test_range'][0]} to {info['test_range'][1]}. "
        "This is a saved historical holdout prediction, not a live forecast."
    )
    return forecast_table, fig, summary


def make_dashboard():
    import gradio as gr

    predictions, metrics, info = load_results()
    range_quality, archive_latest, range_note = load_range_quality()
    dates = sorted(predictions["date"].dt.strftime("%Y-%m-%d").unique())
    stations = sorted(predictions["station"].dropna().unique())
    if not dates or not stations:
        raise ValueError("The saved test predictions contain no dates or stations.")

    model_metrics = metrics[metrics["model"].isin(MODEL_LABELS)]
    best_model = model_metrics.loc[model_metrics["RMSE"].idxmin()]
    baseline_metrics = metrics[metrics["model"].isin(BASELINE_LABELS)]
    best_baseline = baseline_metrics.loc[baseline_metrics["RMSE"].idxmin()]
    improvement = 100 * (best_baseline["RMSE"] - best_model["RMSE"]) / best_baseline["RMSE"]

    bootstrap_path = RES / "bootstrap.json"
    bootstrap_note = ""
    if bootstrap_path.exists():
        import json
        bootstrap = json.loads(bootstrap_path.read_text())
        low, high = bootstrap["ci95"]
        bootstrap_note = (
            f"Paired date-block bootstrap: mean RMSE reduction "
            f"{bootstrap['mean_rmse_reduction']:.2f} "
            f"(95% CI {low:.2f} to {high:.2f}; {bootstrap['n_dates']} test dates)."
        )

    interval_path = RES / "bayes_intervals.json"
    interval_note = ""
    if interval_path.exists():
        import json
        interval = json.loads(interval_path.read_text())
        interval_note = (
            f"Bayesian {interval['level']:.0%} interval empirical coverage: "
            f"{interval['coverage']:.1%}."
        )

    initial_date = dates[-1]
    initial_station = "BurJuman Metro Station" if "BurJuman Metro Station" in stations else stations[0]
    model_choices = [(label, key) for key, (label, _) in MODEL_LABELS.items()]
    metrics_view = format_metrics(metrics)

    def station_choices(date):
        on_date = predictions[predictions["date"] == pd.Timestamp(date)]
        choices = sorted(on_date["station"].dropna().unique())
        selected = initial_station if initial_station in choices else (choices[0] if choices else None)
        return gr.update(choices=choices, value=selected)

    def inspect_forecast(date, station, hour_label, chart_model):
        return build_forecast_view(predictions, date, station, hour_label, chart_model, info)

    def update_performance(family, metric):
        table, fig = build_performance_chart(metrics, family, metric)
        return table, fig

    def update_error_slice(category, metric):
        return load_error_slice(category, metric)

    quality_view = range_quality if not range_quality.empty else pd.DataFrame(
        {"Diagnostic": [range_note]}
    )
    initial_perf_table, initial_perf_chart = build_performance_chart(
        metrics, "All model families", "RMSE"
    )
    initial_error_table, initial_error_chart = load_error_slice("Hour", "RMSE_model")
    feature_path = RES / "rf_importance.csv"
    feature_view = pd.read_csv(feature_path).head(12) if feature_path.exists() else pd.DataFrame(
        {"Note": ["Random Forest feature importances unavailable; run the full model pipeline."]}
    )

    with gr.Blocks(title="Dubai Metro Demand Forecast Explorer") as demo:
        gr.Markdown(
            """
# Dubai Metro Demand Forecasting
### Historical evaluation and station-hour forecast explorer
Use the tabs to review the study-window choice, test-set evidence, station/hour predictions,
and where errors are concentrated.
"""
        )
        gr.Markdown(
            f"**Scope:** saved model predictions are from the historical test set "
            f"({info['test_range'][0]} to {info['test_range'][1]}). This dashboard does not "
            f"produce current/future forecasts. Archive transactions currently run through "
            f"**{archive_latest or 'unknown date'}**; export date is not the same as transaction date."
        )

        with gr.Tabs():
            with gr.Tab("Overview & range choice"):
                gr.Markdown("## Executive summary")
                gr.Markdown(
                    f"""
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px">
  <div style="padding:16px;border-radius:12px;background:#e8f4f4"><b>Best model</b><br><span style="font-size:22px">{MODEL_LABELS[best_model['model']][0]}</span></div>
  <div style="padding:16px;border-radius:12px;background:#e8f4f4"><b>Best-model RMSE</b><br><span style="font-size:22px">{best_model['RMSE']:.2f}</span></div>
  <div style="padding:16px;border-radius:12px;background:#e8f4f4"><b>Improvement vs best baseline</b><br><span style="font-size:22px">{improvement:.1f}% RMSE</span></div>
  <div style="padding:16px;border-radius:12px;background:#e8f4f4"><b>Test observations</b><br><span style="font-size:22px">{len(predictions):,}</span></div>
  <div style="padding:16px;border-radius:12px;background:#e8f4f4"><b>Stations / test dates</b><br><span style="font-size:22px">{predictions['station'].nunique()} / {predictions['date'].nunique()}</span></div>
</div>
"""
                )
                gr.Markdown(
                    f"**Best baseline:** {BASELINE_LABELS[best_baseline['model']]} "
                    f"(RMSE {best_baseline['RMSE']:.2f}). {bootstrap_note} {interval_note}"
                )
                gr.Markdown("## Why this project uses 1 Oct 2018–31 Jan 2020 as its primary range")
                gr.Markdown(
                    "The primary range is the **best-supported study window for this project**, "
                    "not a claim that older data are inherently better or that this model is "
                    "validated for current operations. It offers substantially more usable "
                    "hourly history than the later fragmented blocks, relatively stable "
                    "coverage and station composition, and avoids mixing the COVID recovery/"
                    "Expo period into the primary experiment. The chronological test remains "
                    "within this same pre-pandemic regime."
                )
                gr.Dataframe(
                    value=quality_view,
                    interactive=False,
                    wrap=True,
                    label="Quality-screened archive ranges (DuckDB accumulator)",
                )
                if not range_quality.empty:
                    gr.Plot(value=build_range_chart(range_quality), label="Usable hourly days by candidate period")
                gr.Markdown(f"*{range_note}*")
                gr.Markdown(
                    "### Test-set model comparison\n"
                    "These are rolling one-day-ahead predictions over the historical holdout. "
                    "They are not a single multi-week forecast made at the start of the test period."
                )
                with gr.Row():
                    family_filter = gr.Dropdown(
                        choices=["All model families", *sorted(set(MODEL_FAMILY.values()))],
                        value="All model families",
                        label="Model family",
                    )
                    metric_filter = gr.Dropdown(
                        choices=["RMSE", "MAE", "WAPE"],
                        value="RMSE",
                        label="Comparison metric",
                    )
                performance_table = gr.Dataframe(
                    value=initial_perf_table,
                    interactive=False,
                    label="Test metric (lower is better)",
                )
                performance_chart = gr.Plot(value=initial_perf_chart, label="Models and baselines")
                family_filter.change(
                    update_performance,
                    inputs=[family_filter, metric_filter],
                    outputs=[performance_table, performance_chart],
                )
                metric_filter.change(
                    update_performance,
                    inputs=[family_filter, metric_filter],
                    outputs=[performance_table, performance_chart],
                )
                with gr.Accordion("Feature importance (association, not causation)", open=False):
                    gr.Dataframe(value=feature_view, interactive=False, label="Top Random Forest features")
            with gr.Tab("Station / hour explorer"):
                gr.Markdown(
                    "Select any date, station, and hour in the held-out period. The table shows "
                    "all model variants and baselines for the selected observation; the chart "
                    "compares the selected model over the whole day."
                )
                with gr.Row():
                    date_input = gr.Dropdown(choices=dates, value=initial_date, label="Historical test date")
                    station_input = gr.Dropdown(choices=stations, value=initial_station, label="Station")
                    hour_input = gr.Dropdown(
                        choices=[f"{hour:02d}:00" for hour in range(24)],
                        value="08:00",
                        label="Hour",
                    )
                chart_model = gr.Dropdown(
                    choices=model_choices,
                    value="rf",
                    label="Model for the 24-hour profile",
                )
                run_button = gr.Button("Show selected observation", variant="primary")
                summary_output = gr.Markdown()
                with gr.Row():
                    forecast_output = gr.Dataframe(
                        interactive=False,
                        label="Predictions at the selected station and hour",
                    )
                    plot_output = gr.Plot(label="Actual vs selected model across the day")
                date_input.change(station_choices, inputs=date_input, outputs=station_input)
                run_button.click(
                    inspect_forecast,
                    inputs=[date_input, station_input, hour_input, chart_model],
                    outputs=[forecast_output, plot_output, summary_output],
                )
            with gr.Tab("Error analysis"):
                gr.Markdown(
                    "Explore held-out errors by hour, weekday, holiday status, or station. "
                    "Relative errors can be large in low-volume hours; interpret them alongside RMSE and actual counts."
                )
                with gr.Row():
                    error_category = gr.Dropdown(
                        choices=list(ERROR_FILES),
                        value="Hour",
                        label="Breakdown",
                    )
                    error_metric = gr.Dropdown(
                        choices=["RMSE_model", "WAPE_model"],
                        value="RMSE_model",
                        label="Error metric",
                    )
                error_chart = gr.Plot(value=initial_error_chart, label="Test-set error")
                error_table = gr.Dataframe(
                    value=initial_error_table,
                    interactive=False,
                    label="Detailed error slice (model versus baselines)",
                )
                error_category.change(
                    update_error_slice,
                    inputs=[error_category, error_metric],
                    outputs=[error_table, error_chart],
                )
                error_metric.change(
                    update_error_slice,
                    inputs=[error_category, error_metric],
                    outputs=[error_table, error_chart],
                )
    return demo


if __name__ == "__main__":
    make_dashboard().launch(server_name="127.0.0.1", share=False)
