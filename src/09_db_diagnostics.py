"""One-file diagnostics of the accumulator, to decide the analysis window and any corrective actions.

  python src/09_db_diagnostics.py                      # window from config.py
  python src/09_db_diagnostics.py --start 2024-01-01 --end 2026-06-30

Opens data/interim/counts.duckdb READ-ONLY (stop any running download first: only one process can
open the file). Prints to the screen AND saves reports/db_diagnostics.txt: upload that file.
"""
import argparse, sys
from pathlib import Path
import duckdb, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from aggcore import DB_PATH

ap = argparse.ArgumentParser()
ap.add_argument("--start", default=C.WINDOW_START); ap.add_argument("--end", default=C.WINDOW_END)
a = ap.parse_args()


class Tee:
    def __init__(self, *s): self.s = s
    def write(self, t):
        for x in self.s: x.write(t); x.flush()
    def flush(self):
        for x in self.s: x.flush()


C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
TXT = C.REPORTS_DIR / "db_diagnostics.txt"
_log = open(TXT, "w", encoding="utf-8")
sys.stdout, sys.stderr = Tee(sys.__stdout__, _log), Tee(sys.__stderr__, _log)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 400); pd.set_option("display.max_columns", 40)
try:
    con = duckdb.connect(str(DB_PATH), read_only=True)
except Exception as e:
    sys.exit(f"Cannot open {DB_PATH} read-only: {e}\nIs the download still running? Stop it (Ctrl+C) and retry.")


def show(title, sql, params=None, max_rows=None):
    df = con.execute(sql, params or []).fetchdf()
    print(f"\n== {title} ==")
    print(df.head(max_rows).to_string(index=False) if max_rows else df.to_string(index=False))
    if max_rows and len(df) > max_rows:
        print(f"... ({len(df)} rows total)")
    return df


print(f"DB diagnostics | window {a.start} .. {a.end} | grain {C.GRAIN}\n")
show("1. progress and ledger totals", """
  SELECT count(*) AS chunks_done, sum(rows_total) AS rows_read, sum(rows_bad_date) AS bad_date,
         sum(rows_time_unknown) AS time_unknown, round(100.0 * sum(rows_time_unknown) / sum(rows_total), 2) AS time_unknown_pct,
         sum(rows_other_kind) AS other_kind, sum(rows_no_station) AS no_station FROM chunks_done""")
show("1b. chunk numbers processed from MORE THAN ONE export (possible double counting; expect no rows)", """
  SELECT regexp_extract(stem, '_([0-9]{4})$', 1) AS chunk_no, count(*) AS versions, string_agg(stem, ', ') AS files
  FROM chunks_done GROUP BY 1 HAVING count(*) > 1""")
show("1c. exports present in the ledger (folder prefix = export timestamp)", """
  SELECT regexp_replace(stem, '_[0-9]{4}$', '') AS export_prefix, count(*) AS chunks,
         min(regexp_extract(stem, '_([0-9]{4})$', 1)) AS first_chunk, max(regexp_extract(stem, '_([0-9]{4})$', 1)) AS last_chunk
  FROM chunks_done GROUP BY 1 ORDER BY 1""")
show("1d. integrity: counted rows should equal rows read minus bad dates minus no-station rows", """
  SELECT (SELECT sum(rows_total - rows_bad_date - rows_no_station) FROM chunks_done) AS expected_rows,
         (SELECT sum(n) FROM counts) AS counted_rows""")
show("2. date coverage", """
  SELECT min(date) AS first_date, max(date) AS last_date, count(DISTINCT date) AS days_with_data,
         datediff('day', min(date), max(date)) + 1 AS calendar_days,
         datediff('day', min(date), max(date)) + 1 - count(DISTINCT date) AS days_without_any_data FROM counts""")
show("3. rows per year by kind, in/out balance and unknown-time share", """
  SELECT year(date) AS year, sum(n) FILTER (WHERE kind = 'in') AS check_in, sum(n) FILTER (WHERE kind = 'out') AS check_out,
         round(sum(n) FILTER (WHERE kind = 'in') * 1.0 / nullif(sum(n) FILTER (WHERE kind = 'out'), 0), 3) AS in_out_ratio,
         round(100.0 * sum(n) FILTER (WHERE hour = -1) / sum(n), 1) AS unknown_time_pct
  FROM counts GROUP BY 1 ORDER BY 1""")
show("4. monthly volumes (look for under-recorded months and months dominated by unknown times)", """
  SELECT strftime(date, '%Y-%m') AS month, sum(n) FILTER (WHERE kind = 'in') AS check_in, sum(n) FILTER (WHERE kind = 'out') AS check_out,
         round(100.0 * sum(n) FILTER (WHERE hour = -1) / sum(n), 1) AS unknown_time_pct,
         count(DISTINCT date) AS days_with_data FROM counts GROUP BY 1 ORDER BY 1""")
show("5. days with no data at all, per year", """
  WITH cal AS (SELECT CAST(g AS DATE) AS d FROM generate_series((SELECT min(date) FROM counts), (SELECT max(date) FROM counts), INTERVAL 1 DAY) t(g))
  SELECT year(d) AS year, count(*) AS missing_days FROM cal WHERE d NOT IN (SELECT DISTINCT date FROM counts) GROUP BY 1 ORDER BY 1""")
show("5b. first missing days (up to 60)", """
  WITH cal AS (SELECT CAST(g AS DATE) AS d FROM generate_series((SELECT min(date) FROM counts), (SELECT max(date) FROM counts), INTERVAL 1 DAY) t(g))
  SELECT d AS missing_date FROM cal WHERE d NOT IN (SELECT DISTINCT date FROM counts) ORDER BY d""", max_rows=60)
show("6. outage candidates: days with check-ins below 30% of their month's median day, count per year", """
  WITH d AS (SELECT date, sum(n) AS n FROM counts WHERE kind = 'in' GROUP BY 1),
       m AS (SELECT date, n, median(n) OVER (PARTITION BY date_trunc('month', date)) AS med FROM d)
  SELECT year(date) AS year, count(*) AS low_days FROM m WHERE n < 0.3 * med GROUP BY 1 ORDER BY 1""")
show("7. hourly profile, real times only (hour >= 0), all years: expect ~0 overnight and clear peaks", """
  SELECT hour, sum(n) FILTER (WHERE kind = 'in') AS check_in, sum(n) FILTER (WHERE kind = 'out') AS check_out
  FROM counts WHERE hour >= 0 GROUP BY 1 ORDER BY 1""")
show(f"8. hourly profile inside the window {a.start}..{a.end}", """
  SELECT hour, sum(n) AS check_in FROM counts WHERE kind = 'in' AND hour >= 0 AND date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
  GROUP BY 1 ORDER BY 1""", [a.start, a.end])
show(f"9. weekday profile in the window (mean check-ins per day; confirms the weekend definition)", """
  WITH d AS (SELECT date, sum(n) AS n FROM counts WHERE kind = 'in' AND date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE) GROUP BY 1)
  SELECT dayname(date) AS weekday, round(avg(n)) AS mean_daily_check_ins, count(*) AS days FROM d GROUP BY 1, dayofweek(date) ORDER BY dayofweek(date)""",
     [a.start, a.end])
show("10. lowest 15 days inside the window (cross-check against holidays, weather, disruptions)", """
  SELECT date, dayname(date) AS weekday, sum(n) AS check_in FROM counts WHERE kind = 'in' AND date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
  GROUP BY 1 ORDER BY 3 LIMIT 15""", [a.start, a.end])
show("11. unknown-time share per month inside the window (decides hour vs day grain)", """
  SELECT strftime(date, '%Y-%m') AS month, round(100.0 * sum(n) FILTER (WHERE hour = -1) / sum(n), 1) AS unknown_time_pct
  FROM counts WHERE date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE) GROUP BY 1 ORDER BY 1""", [a.start, a.end])
show("12. station timeline (Check in): first/last date, total, active days; spots renames and new stations", """
  SELECT station, min(date) AS first_date, max(date) AS last_date, count(DISTINCT date) AS active_days, sum(n) AS check_ins
  FROM counts WHERE kind = 'in' GROUP BY 1 ORDER BY first_date, station""")
show(f"13. busiest stations inside the window", """
  SELECT station, sum(n) AS check_ins, count(DISTINCT date) AS active_days FROM counts
  WHERE kind = 'in' AND date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE) GROUP BY 1 ORDER BY 2 DESC""", [a.start, a.end], max_rows=25)
show("14. rows with no usable station or kind, by kind", "SELECT kind, count(*) AS cells, sum(n) AS rows FROM counts GROUP BY 1 ORDER BY 1")
print(f"\nOutput saved to: {TXT}")
