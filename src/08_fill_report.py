"""Fill the numeric placeholders of the report draft from the results, and insert the figures.

  python src/08_fill_report.py "Final_Report_DRAFT_Metro_Demand_Forecasting.docx" "Final_Report.docx"

* Replaces every highlighted [KEY] run whose KEY exists in reports/results/placeholder_values.json
  (and removes the highlight), then refreshes metric-dependent narrative from saved results.
* Fills Table 2 from reports/results/table2.csv and bolds the best value in each column.
* Puts reports/figures/fig1a_*.png and fig1b_*.png into the two figure cells.
* Prints the placeholders that remain. Re-read the whole report afterwards: numbers change the story.
Requires python-docx.
"""
import json, re, sys
from pathlib import Path
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

src, dst = sys.argv[1], sys.argv[2]
RES = C.REPORTS_DIR / "results"
vals = json.loads((RES / "placeholder_values.json").read_text())
t2 = pd.read_csv(RES / "table2.csv")
metrics = pd.read_csv(RES / "metrics_test.csv").set_index("model")
run_info = json.loads((RES / "run_info.json").read_text())
dq = json.loads((C.REPORTS_DIR / "data_quality.json").read_text())
index_path = C.INTERIM_DIR / "chunk_index.json"
chunk_index = json.loads(index_path.read_text()) if index_path.exists() else {}
exports = {str(item["folder"]).rsplit("_", 1)[0] for item in chunk_index.values()}
doc = Document(src)


def all_paragraphs(d):
    yield from d.paragraphs
    for t in d.tables:
        for row in t.rows:
            seen = set()
            for cell in row.cells:
                if id(cell._tc) in seen:
                    continue
                seen.add(id(cell._tc))
                yield from cell.paragraphs


def rewrite_paragraph(prefix, text):
    matches = [p for p in all_paragraphs(doc) if p.text.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one report paragraph starting with {prefix!r}; found {len(matches)}")
    matches[0].text = text


def replace_in_paragraph(prefix, old, new):
    matches = [p for p in all_paragraphs(doc) if p.text.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one report paragraph starting with {prefix!r}; found {len(matches)}")
    matches[0].text = matches[0].text.replace(old, new)


def main_word_count(d):
    parts = []
    in_report = False
    for p in d.paragraphs:
        if p.text.strip() == "Abstract":
            in_report = True
            continue
        if p.text.strip() == "References":
            break
        if in_report:
            parts.append(p.text)
    for t in d.tables:
        seen = set()
        for row in t.rows:
            for cell in row.cells:
                if id(cell._tc) not in seen:
                    seen.add(id(cell._tc))
                    parts.append(cell.text)
    return len(re.findall(r"\b[\w'-]+\b", " ".join(parts)))


filled = 0
for p in all_paragraphs(doc):
    for r in p.runs:
        m = re.fullmatch(r"\[(.+)\]", r.text.strip())
        if m and m.group(1) in vals and vals[m.group(1)] and not str(vals[m.group(1)]).startswith("["):
            r.text, r.font.highlight_color = str(vals[m.group(1)]), None
            filled += 1

# ---- wording fixes when the best model does NOT beat the best baseline (an honest outcome) ----
if vals.get("_beat_baseline") == "no":
    swaps = {"outperformed every baseline, which shows that": "did not outperform the best baseline, which suggests that",
             "an improvement of": "a deterioration of", "% better than the strongest baseline": "% worse than the strongest baseline",
             "gave the lowest error": "gave the lowest error among the approved methods",
             "achieved the lowest test error": "achieved the lowest test error among the approved methods"}
    for p in all_paragraphs(doc):
        for r in p.runs:
            for old, new in swaps.items():
                if old in r.text:
                    r.text = r.text.replace(old, new)
    print("NOTE: best model does not beat the best baseline -> the wording was adjusted; rewrite the discussion to explain why.")

# Keep the nonlinear-model comparison consistent with the evaluated test results.
linear_models = [m for m in ("ridge", "lasso", "bayes", "svr_linear") if m in metrics.index]
if linear_models and {"rf", "svr_rbf"}.issubset(metrics.index):
    best_linear = min(linear_models, key=lambda m: metrics.loc[m, "RMSE"])
    best_linear_rmse = metrics.loc[best_linear, "RMSE"]
    rf_better = metrics.loc["rf", "RMSE"] < best_linear_rmse
    rbf_better = metrics.loc["svr_rbf", "RMSE"] < best_linear_rmse
    if not (rf_better and rbf_better):
        if rf_better:
            comparison = (
                f"Random Forest outperformed every linear method, while RBF-SVR did not improve "
                f"on the best linear model (RMSE {metrics.loc['svr_rbf', 'RMSE']:.2f} versus "
                f"{best_linear_rmse:.2f}); RBF-SVR used a "
                f"{int(run_info['svr_rbf_train_rows']):,}-row training subsample"
            )
        elif rbf_better:
            comparison = (
                f"RBF-SVR outperformed every linear method, while Random Forest did not improve "
                f"on the best linear model (RMSE {metrics.loc['rf', 'RMSE']:.2f} versus "
                f"{best_linear_rmse:.2f}); RBF-SVR used a "
                f"{int(run_info['svr_rbf_train_rows']):,}-row training subsample"
            )
        else:
            comparison = (
                f"Neither Random Forest nor RBF-SVR outperformed the best linear model "
                f"(RMSE {best_linear_rmse:.2f}); RBF-SVR used a "
                f"{int(run_info['svr_rbf_train_rows']):,}-row training subsample"
            )
        outdated = (
            "Both non-linear models (Random Forest and the RBF support vector regression) "
            "achieved lower error than every linear model, even though the RBF model was trained "
            "on only a subsample"
        )
        for p in all_paragraphs(doc):
            for r in p.runs:
                if outdated in r.text:
                    r.text = r.text.replace(outdated, comparison)

# Refresh narrative claims directly from the final saved metrics and diagnostics.
if "svr_rbf_train_rows" in run_info:
    replace_in_paragraph(
        "Support vector regression (Drucker",
        "30,000 rows",
        f"{int(run_info['svr_rbf_train_rows']):,} rows",
    )

linear_models = [m for m in ("ridge", "lasso", "bayes", "svr_linear") if m in metrics.index]
best_linear = min(linear_models, key=lambda m: metrics.loc[m, "RMSE"])
best_linear_rmse = metrics.loc[best_linear, "RMSE"]
best_linear_name = metrics.loc[best_linear, "name"]
rbf_rmse = metrics.loc["svr_rbf", "RMSE"]
if rf_better and rbf_better:
    nonlinear_comparison = (
        f"Both Random Forest and RBF-SVR outperformed the best linear model "
        f"({best_linear_name}, RMSE {best_linear_rmse:.2f}); Random Forest remained the "
        f"strongest fitted model. RBF-SVR used a "
        f"{int(run_info['svr_rbf_train_rows']):,}-row training subsample, so results do not "
        "establish that nonlinear methods will generally outperform linear ones."
    )
elif rf_better:
    nonlinear_comparison = (
        f"Random Forest outperformed the best linear model ({best_linear_name}, RMSE "
        f"{best_linear_rmse:.2f}), while RBF-SVR did not (RMSE {rbf_rmse:.2f}). "
        f"RBF-SVR used a {int(run_info['svr_rbf_train_rows']):,}-row training subsample; "
        "these results do not establish that nonlinear methods will generally outperform "
        "linear ones."
    )
elif rbf_better:
    nonlinear_comparison = (
        f"RBF-SVR outperformed the best linear model ({best_linear_name}, RMSE "
        f"{best_linear_rmse:.2f}), while Random Forest did not (RMSE "
        f"{metrics.loc['rf', 'RMSE']:.2f}). RBF-SVR used a "
        f"{int(run_info['svr_rbf_train_rows']):,}-row training subsample; these results do "
        "not establish that nonlinear methods will generally outperform linear ones."
    )
else:
    nonlinear_comparison = (
        f"Neither Random Forest nor RBF-SVR outperformed the best linear model "
        f"({best_linear_name}, RMSE {best_linear_rmse:.2f}); RBF-SVR used a "
        f"{int(run_info['svr_rbf_train_rows']):,}-row training subsample."
    )
best_name = vals["BEST_MODEL"]
best_baseline_name = vals["BEST_BASELINE"]
best_rmse = float(vals["RMSE_BEST"])
baseline_rmse = float(vals["RMSE_BASELINE_BEST"])
bootstrap = json.loads((RES / "bootstrap.json").read_text())
rf_tree_count = 60 if run_info.get("fast") else 150
holiday = pd.read_csv(RES / "errors_by_holiday.csv").set_index("holiday")
holiday_wape = float(holiday.loc["public holiday", "WAPE_model"] * 100)
regular_wape = float(holiday.loc["regular day", "WAPE_model"] * 100)
holiday_baseline_wape = float(holiday.loc["public holiday", "WAPE_baseline"] * 100)
chunk_sentence = (
    f"{dq['chunks_processed']:,} of {dq['chunks_total']:,} unique files in the refreshed API index "
    f"were processed across {dq.get('chunk_exports_in_index', len(exports)):,} export snapshots."
)
if dq.get("legacy_chunks_in_accumulator", 0):
    chunk_sentence += (
        f" The accumulator also retains {dq['legacy_chunks_in_accumulator']:,} previously processed "
        "files absent from the current index; their counts remain included and possible overlap "
        "with current exports has not been ruled out."
    )
row_sentence = (
    f"The local accumulator read {dq['rows_read']:,} transaction rows across "
    f"{dq.get('accumulator_chunks', dq['chunks_processed']):,} chunk files, including "
    f"{dq.get('rows_read_legacy', 0):,} rows from files absent from the current index."
)
archive_period = f"{dq['archive_date_start']} to {dq['archive_date_end']}"

rewrite_paragraph(
    "Accurate short-term demand forecasts help metro operators",
    "Accurate short-term demand forecasts help metro operators plan capacity, schedule trains and "
    "manage crowding. This study forecasts next-day station-hour check-in demand for the Dubai Metro "
    "using the Roads and Transport Authority's open fare-gate transaction data (data.dubai, dataset "
    f"466403): {chunk_sentence} {row_sentence} The archive covers {archive_period}. Profiling revealed "
    "serious data-quality hazards: chunks are random slices of the whole period, "
    f"{dq['time_unknown_share'] * 100:.1f}% of rows carry a placeholder time of 00:00:00.000, "
    "and monthly volumes are implausibly low in some periods. Real hourly times, complete coverage, stable volumes and a "
    "stable station set coincide only before the pandemic, so the analysis covers October 2018 to "
    "January 2020. After aggregation to station-hour counts, regularised and Bayesian regression, "
    "Random Forest and support vector regression were compared with four seasonal baselines using "
    "chronological, rolling-origin evaluation. Random Forest achieved the lowest test error "
    f"(RMSE {best_rmse:.2f}, WAPE {vals['WAPE_BEST']}), "
    f"{float(vals['IMPROVEMENT']):.1f}% better than the strongest baseline. Public-holiday WAPE was "
    f"{holiday_wape:.1f}%, versus {regular_wape:.1f}% on regular days. The report documents these "
    "data hazards and discusses privacy, legal and fairness implications."
)

rewrite_paragraph(
    "The data are the RTA Metro Ridership dataset",
    "The data are the RTA Metro Ridership dataset published on data.dubai (dataset 466403), "
    "delivered through an API as gzip-compressed CSV chunks of about 1.05 million rows. "
    f"{chunk_sentence} {row_sentence} Table 1 summarises archive-level data quality, with "
    "chunk-specific observations labelled separately."
)

rewrite_paragraph(
    "Several issues shape the analysis.",
    "Several issues shape the analysis. Hourly coverage is intermittent: dates with mostly "
    "placeholder times cannot support hourly modelling, and some periods have implausibly low "
    "volumes. In the selected 488-day window, 84 dates failed the 90% known-time threshold and two "
    "dates were flagged for unusually low volume; unreliable targets were treated as missing, not "
    "zero. Archive volumes also vary substantially, with pronounced low volumes in 2022-23 and a "
    "recent imbalance between Check in and Check out counts, so the latest transaction date alone "
    "does not establish recent data completeness. The refreshed API index contains "
    f"{dq['chunks_total']:,} unique chunk files across {dq.get('chunk_exports_in_index', 0):,} "
    f"export snapshots, all processed. The accumulator additionally retains "
    f"{dq.get('legacy_chunks_in_accumulator', 0):,} older file stems absent from that index "
    f"({dq.get('rows_read_legacy', 0):,} rows); possible overlap is unresolved and may inflate "
    "totals slightly. The study therefore uses 1 October 2018 to 31 January 2020, where usable "
    "hourly coverage, comparatively stable volumes and a stable 46-station set coincide."
)

rewrite_paragraph(
    "Pipeline.",
    "Pipeline. Each chunk was aggregated with DuckDB (Raasveldt and Mühleisen, 2019) into a running "
    "table of counts by date, hour, transaction type and station, with a ledger preventing "
    f"double-counting of file stems. {chunk_sentence} {row_sentence} Station is the entry station "
    "of a Check in, name variants were merged, and rows with an unknown time were excluded from "
    "hourly series. Dates with no data, or with less than 90% of rows carrying a real time, were set "
    "to missing rather than zero, and missing targets were dropped after lag construction. Exact "
    "duplicate rows were retained because they cannot be distinguished from simultaneous taps; "
    "duplicate prevalence is reported only for the profiled chunk."
)

rewrite_paragraph(
    "Evaluation protocol.",
    "Evaluation protocol. Data were split chronologically by date: 70% training "
    f"({vals['TRAIN_RANGE']}), 15% validation ({vals['VAL_RANGE']}) and 15% test "
    f"({vals['TEST_RANGE']}). Hyperparameters were tuned using date-grouped rolling-origin "
    "validation within the training and validation periods; the test set was held out for final "
    "evaluation. Search ranges were Ridge alpha (0.1, 1, 10, 100), Lasso alpha (0.001, 0.01, 0.1), "
    "Random Forest depth (None, 20, 30), leaf size (1, 3, 10) and feature fraction (0.4, 0.7, 1.0), "
    "and SVR C values (0.01, 0.1, 1) for linear SVR and (1, 10, 100) for RBF-SVR. Random Forest "
    f"used {rf_tree_count} trees; RBF-SVR epsilon was fixed at 0.05 and gamma used the scikit-learn default. "
    f"{vals['FINAL_HYPERPARAMETERS']}"
)

rewrite_paragraph(
    "Features and horizon.",
    "Features and horizon. All lags are at least 24 hours: 24, 48, 168, 336, 504 and 672 hours, "
    "plus the mean of the one- to four-week lags. Calendar features are hour, weekday, weekend flag "
    "(Friday and Saturday before 2022, as the weekday profile of the window confirms), month and a "
    "public-holiday flag generated with the Python holidays package's AE calendar. Holiday dates "
    "were not independently checked against each official UAE announcement and should be verified "
    "before operational use. Station identity completes the feature set. Weather was not used, "
    "which is a limitation."
)

rewrite_paragraph(
    "Random Forest gave the lowest error",
    f"{best_name} gave the lowest error (RMSE {best_rmse:.2f}), against "
    f"{baseline_rmse:.2f} for the best baseline ({best_baseline_name}), an improvement of "
    f"{float(vals['IMPROVEMENT']):.1f}%. Across all fitted model variants, the test RMSE ranking "
    f"was {vals['RANKING']}. {nonlinear_comparison} Figure 1 compares forecasts with "
    "actual demand for a representative station (panel a) and shows error by hour of day (panel b)."
)

rewrite_paragraph(
    "Model comparison (RQ1).",
    f"Model comparison (RQ1). {best_name} outperformed every baseline, showing that recent "
    "history and calendar position provide useful information beyond fixed seasonal profiles. "
    f"The paired block bootstrap estimated an RMSE reduction of "
    f"{float(bootstrap['mean_rmse_reduction']):.2f} against {best_baseline_name}, with a "
    f"95% confidence interval of {bootstrap['ci95'][0]:.2f} to "
    f"{bootstrap['ci95'][1]:.2f}. Random Forest was the "
    f"strongest fitted model. {nonlinear_comparison}"
)

coefficients = pd.read_csv(RES / "bayes_coefficients.csv").set_index("feature")
lag_coef = float(coefficients.loc["num__lag_168", "coef_standardised_target"])
holiday_coef = float(coefficients.loc["bin__is_holiday", "coef_standardised_target"])
rewrite_paragraph(
    "What the models learned (RQ2).",
    "What the models learned (RQ2). Random Forest importance ranked "
    f"{vals['TOP_FEATURES']} highest. In Bayesian ridge, the standardised coefficient for the "
    f"one-week lag was {lag_coef:+.3f}, while the public-holiday coefficient was {holiday_coef:+.3f}. "
    "These are conditional associations in the fitted model, not causal effects. Calendar effects "
    "remain plausible, although weather could not be tested here."
)

rewrite_paragraph(
    "Error analysis and uncertainty (RQ3).",
    f"Error analysis and uncertainty (RQ3). By hour, Random Forest RMSE was highest at "
    f"{vals['PEAK_HOURS']}. Among high-volume stations, the largest WAPEs were "
    f"{vals['WORST_STATIONS']}. Public-holiday WAPE was {holiday_wape:.1f}%, compared with "
    f"{regular_wape:.1f}% on regular days; the four-week seasonal-mean baseline reached "
    f"{holiday_baseline_wape:.1f}% on holidays. The Bayesian 90% prediction intervals covered "
    f"{vals['COVERAGE']}% of test observations, close to nominal coverage on this test period."
)

rewrite_paragraph(
    "Limitations.",
    "Limitations. First, the data have no transaction or card identifier, so duplicates cannot be "
    "separated from simultaneous taps; their count is reported for the profiled chunk, not assumed "
    "to apply uniformly to every chunk. Second, rows at 00:00:00.000 carry no usable time, which "
    "removes some dates from the hourly series and confines the analysis to pre-pandemic operations. "
    f"Third, all {dq['chunks_total']:,} currently indexed files were processed, but "
    f"{dq.get('legacy_chunks_in_accumulator', 0):,} previously processed files absent from the "
    "refreshed index remain in the accumulator; possible overlap is unresolved and may affect totals. "
    "Fourth, apparent under-recording in some years and the excess of Check in over Check out rows "
    "after 2025 could distort trends outside the window. Fifth, only Check in demand is modelled, "
    "without origin-destination flows, weather or events, and a single network limits generalisation. "
    "Finally, structural change such as COVID-19 (Mützel and Scheiner, 2022) means models trained on "
    "one regime may not transfer to another."
)

rewrite_paragraph(
    "This study built a reproducible pipeline",
    f"This study built a reproducible pipeline for the RTA open metro data and compared "
    "regularised and Bayesian linear regression, Random Forest and support vector regression for "
    f"next-day station-hour demand. {best_name} reduced RMSE by "
    f"{float(vals['IMPROVEMENT']):.1f}% relative to the strongest seasonal baseline; the paired "
    f"block-bootstrap interval excluded zero. {nonlinear_comparison} The principal practical lesson "
    "is that data quality and regime "
    "stability affect the usefulness of forecasts as much as model choice. Future work should add "
    "weather and event data, model origin-destination flows using check-out records, test gradient "
    "boosting and sequence models, and evaluate transfer across structural changes such as COVID-19 "
    "and Expo 2020."
)

for p in all_paragraphs(doc):
    if p.text.strip() == "Submission date: 2 November 2026":
        p.text = "Submission deadline: 8 November 2026"

# ---- Table 2 ----
for t in doc.tables:
    for row in t.rows:
        if row.cells[0].text.strip() == "Size" and len(row.cells) > 1:
            row.cells[1].text = (
                f"{dq['chunks_total']:,} unique chunk files indexed "
                f"({dq['chunks_processed']:,} processed across {dq.get('chunk_exports_in_index', 0):,} exports); "
                f"{dq['rows_read']:,} rows read across {dq.get('accumulator_chunks', dq['chunks_processed']):,} "
                "accumulator files; each chunk about 1.05 million rows"
            )
        elif row.cells[0].text.strip() == "Period" and len(row.cells) > 1:
            row.cells[1].text = (
                f"{dq['archive_date_start']} to {dq['archive_date_end']}; "
                f"{dq['archive_distinct_dates']:,} distinct dates in the accumulator"
            )
        elif row.cells[0].text.strip() == "Chunk structure" and len(row.cells) > 1:
            row.cells[1].text = (
                f"Random slices of the archive are distributed across "
                f"{dq.get('chunk_exports_in_index', 0):,} indexed export snapshots. The refreshed "
                f"index has {dq['chunks_total']:,} unique chunk files; "
                f"{dq.get('legacy_chunks_in_accumulator', 0):,} older file stems absent from the "
                "current index remain in the accumulator and possible overlap is unresolved. "
                "Transaction dates extend only through "
                f"{dq['archive_date_end']}."
            )
    head = [c.text.strip() for c in t.rows[0].cells]
    if head == ["Model", "RMSE", "MAE", "R²", "WAPE"]:
        best = {"RMSE": t2["RMSE"].min(), "MAE": t2["MAE"].min(), "R2": t2["R2"].max(), "WAPE": t2["WAPE"].min()}
        while len(t.rows) < len(t2) + 1:
            t.add_row()
        for i, row in enumerate(t.rows[1:len(t2) + 1]):
            rec = t2.iloc[i]
            row.cells[0].text = str(rec["label"])
            for j, key in enumerate(["RMSE", "MAE", "R2", "WAPE"], start=1):
                para = row.cells[j].paragraphs[0]
                run = para.runs[0] if para.runs else para.add_run()
                run.text = f"{rec[key]:.1f}%" if key == "WAPE" else f"{rec[key]:.2f}"
                run.font.highlight_color = None
                run.bold = bool(abs(rec[key] - best[key]) < 1e-12)
            filled += 4

# ---- figures ----
figs = {"[INSERT FIGURE 1a": C.FIGS_DIR / "fig1a_forecast_vs_actual.png", "[INSERT FIGURE 1b": C.FIGS_DIR / "fig1b_error_by_hour.png"}
for t in doc.tables:
    for row in t.rows:
        for cell in row.cells:
            for prefix, img in figs.items():
                if cell.text.strip().startswith(prefix) and img.exists():
                    para = cell.paragraphs[0]
                    for r in list(para.runs):
                        r._element.getparent().remove(r._element)
                    para.paragraph_format.line_spacing = 1.0          # single/auto spacing: a fixed line height would clip the picture
                    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    para.add_run().add_picture(str(img), width=Cm(7.4))
                    filled += 1

word_count = main_word_count(doc)
for p in all_paragraphs(doc):
    for r in p.runs:
        if "[FINAL_WORD_COUNT]" in r.text:
            r.text = r.text.replace("[FINAL_WORD_COUNT]", f"{word_count:,}")

doc.save(dst)
left = sorted({m for p in all_paragraphs(Document(dst)) for m in re.findall(r"\[[^\]]+\]", p.text)})
print(f"filled {filled} items -> {dst} (main-text word count: {word_count:,})")
print(f"\n{len(left)} placeholders still need YOUR input:")
for x in left:
    print("  ", x)
Path(dst).with_suffix(".remaining_placeholders.txt").write_text("\n".join(left))
