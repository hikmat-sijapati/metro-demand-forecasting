"""Deeper data-quality profile of ONE chunk (every chunk is a random slice of all dates, so one
chunk is representative). Answers: how many rows have a placeholder time (00:00:00.000), are
whole dates placeholders, how many duplicates are real taps vs placeholders, what the hourly
profile looks like, which months look under-recorded, and when each station name appears
(to spot renamed/new stations).

  python src/02b_time_quality.py data/raw/<chunk>.csv.gz
Saves reports/time_quality_<chunk>.txt (upload that file).
"""
import sys
from pathlib import Path
import duckdb, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from aggcore import canon, date_expr, sniff_src
from utils import resolve_cols


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for st in self.streams:
            st.write(text)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


path = Path(sys.argv[1])
C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
TXT = C.REPORTS_DIR / f"time_quality_{path.name.split('.')[0]}.txt"
_log = open(TXT, "w", encoding="utf-8")
sys.stdout, sys.stderr = Tee(sys.__stdout__, _log), Tee(sys.__stderr__, _log)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 500)
print(f"Time-quality profile of {path.name}\n")

con = duckdb.connect()
src = sniff_src(path)
cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()]
c = resolve_cols(cols, C.COLS)
Q = {k: '"' + v + '"' for k, v in c.items()}
typ = f"lower(trim({Q['txn_type']}))"
kind = f"CASE WHEN {typ} LIKE '%in' THEN 'in' WHEN {typ} LIKE '%out' THEN 'out' ELSE 'other' END"
load = '"load_timestamp"' if "load_timestamp" in [x.lower() for x in cols] else "NULL"
con.execute(f"""CREATE TEMP TABLE x AS SELECT {date_expr(Q['txn_date'])} AS d, trim({Q['txn_time']}) AS t,
  {kind} AS kind, {canon(Q['start'])} AS s_in, {canon(Q['end'])} AS s_out, {load} AS load_ts FROM {src}""")
con.execute("CREATE TEMP TABLE y AS SELECT *, (t LIKE '00:00:00%') AS mid FROM x")
show = lambda title, sql: print(f"\n== {title} ==\n" + con.execute(sql).fetchdf().to_string(index=False))

show("1. placeholder times (exactly 00:00:00): overall", """
  SELECT count(*) AS rows, count(*) FILTER (WHERE mid) AS midnight_rows,
         round(100.0 * count(*) FILTER (WHERE mid) / count(*), 2) AS midnight_pct FROM y""")
show("1b. placeholder share by txn kind", """
  SELECT kind, count(*) AS rows, count(*) FILTER (WHERE mid) AS midnight_rows,
         round(100.0 * count(*) FILTER (WHERE mid) / count(*), 2) AS midnight_pct FROM y GROUP BY 1 ORDER BY 1""")
show("1c. placeholder share by year", """
  SELECT year(d) AS year, count(*) AS rows, count(*) FILTER (WHERE mid) AS midnight_rows,
         round(100.0 * count(*) FILTER (WHERE mid) / count(*), 2) AS midnight_pct FROM y GROUP BY 1 ORDER BY 1""")
show("2. are WHOLE DATES placeholders? (per-date placeholder share)", """
  WITH per AS (SELECT d, count(*) AS n, count(*) FILTER (WHERE mid) AS m FROM y GROUP BY d)
  SELECT count(*) AS dates, count(*) FILTER (WHERE m = 0) AS dates_no_placeholder,
         count(*) FILTER (WHERE m * 1.0 / n > 0.5) AS dates_over_50pct,
         count(*) FILTER (WHERE m * 1.0 / n > 0.9) AS dates_over_90pct,
         coalesce(sum(n) FILTER (WHERE m * 1.0 / n > 0.9), 0) AS rows_on_over_90pct_dates FROM per""")
show("3. duplicates (same date, time, kind, stations)", """
  SELECT (SELECT count(*) FROM x) - (SELECT count(*) FROM (SELECT DISTINCT d, t, kind, s_in, s_out FROM x)) AS dup_rows_all,
         (SELECT count(*) FROM y WHERE NOT mid) - (SELECT count(*) FROM (SELECT DISTINCT d, t, kind, s_in, s_out FROM y WHERE NOT mid)) AS dup_rows_excluding_midnight""")
print("\n== 4. hourly profile of rows WITH a real time (rows per hour of day) ==")
h = con.execute("SELECT CAST(substr(t, 1, 2) AS INTEGER) AS hour, kind, count(*) AS n FROM y WHERE NOT mid AND kind IN ('in','out') GROUP BY 1, 2").fetchdf()
print(h.pivot(index="hour", columns="kind", values="n").fillna(0).astype(int).to_string())
show("5. rows per month (look for under-recorded periods)", """
  SELECT strftime(d, '%Y-%m') AS month, count(*) AS rows, count(*) FILTER (WHERE kind = 'in') AS check_in,
         count(*) FILTER (WHERE mid) AS midnight_rows FROM y GROUP BY 1 ORDER BY 1""")
show("6. station name timeline (Check in): first/last date seen per name", """
  SELECT COALESCE(NULLIF(s_in, ''), NULLIF(s_out, '')) AS station, min(d) AS first_date, max(d) AS last_date, count(*) AS rows
  FROM y WHERE kind = 'in' GROUP BY 1 ORDER BY first_date, station""")
show("7. Check in rows with no station at all", """
  SELECT count(*) AS rows_no_station FROM y WHERE kind = 'in' AND NULLIF(s_in, '') IS NULL AND NULLIF(s_out, '') IS NULL""")
show("8. load_timestamp", "SELECT count(DISTINCT load_ts) AS distinct_loads, min(load_ts) AS first_load, max(load_ts) AS last_load FROM y")
print(f"\nOutput saved to: {TXT}")
