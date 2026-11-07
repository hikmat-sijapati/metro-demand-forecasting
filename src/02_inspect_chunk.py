"""Profile one chunk. Answers the questions that decide the whole design:
  1. Is the HOUR of day recoverable from txn_time?  (sample looked like mm:ss.0)
  2. Which date formats occur, and what date range / row order does a chunk have?
     (decides whether a time window needs few chunks or all 3,104)
  3. Blank stations, name variants, duplicate rows.

  python src/02_inspect_chunk.py data/raw/metro_ridership_..._0001.csv.gz
"""
import json, sys
from pathlib import Path
import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from aggcore import canon, date_expr
from utils import resolve_cols

def sniff_src(p: Path) -> str:
    """read_csv() source that works whether or not the file is really gzipped."""
    with open(p, "rb") as f:
        gz = f.read(2) == b"\x1f\x8b"
    if gz != p.name.endswith(".gz"):
        print(f"NOTE: {p.name} is {'gzip' if gz else 'plain text'} despite its extension; reading it as such.")
    return (f"read_csv('{p.as_posix()}', all_varchar=true, ignore_errors=true, "
            f"compression='{'gzip' if gz else 'none'}')")


class Tee:
    """Send everything printed (and any error traceback) to the screen AND a text file."""
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
TXT_PATH = C.REPORTS_DIR / f"chunk_profile_{path.name.split('.')[0]}.txt"
_log = open(TXT_PATH, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _log)
sys.stderr = Tee(sys.__stderr__, _log)
print(f"Profile of {path.name}\n")
con = duckdb.connect()
src = sniff_src(path)
cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()]
print("COLUMNS:", cols)
print(con.execute(f"SELECT * FROM {src} LIMIT 5").df().to_string())
c = resolve_cols(cols, C.COLS)
Q = {k: '"' + v + '"' for k, v in c.items()}
D, T = f"trim({Q['txn_date']})", f"trim({Q['txn_time']})"
dmy = f"try_strptime({D}, '%d/%m/%Y')"
iso = f"try_strptime({D}, '%Y-%m-%d')"
dd = date_expr(Q["txn_date"])
def _py(v):
    if hasattr(v, "item") and not hasattr(v, "year"):        # numpy scalar -> python
        return v.item()
    return v if isinstance(v, (int, float, str)) or v is None else str(v)


one = lambda sql: {k: _py(v) for k, v in con.execute(sql).fetchdf().iloc[0].to_dict().items()}
out = {}

out["dates"] = one(f"""SELECT count(*) AS rows,
  count(*) FILTER (WHERE {dmy} IS NOT NULL) AS fmt_dd_mm_yyyy,
  count(*) FILTER (WHERE {dmy} IS NULL AND {iso} IS NOT NULL) AS fmt_iso,
  count(*) FILTER (WHERE {dd} IS NULL) AS unparsed,
  count(*) FILTER (WHERE try_cast(split_part({D}, '/', 1) AS INTEGER) > 12) AS first_field_gt12,
  count(*) FILTER (WHERE try_cast(split_part({D}, '/', 2) AS INTEGER) > 12) AS second_field_gt12,
  min({dd}) AS date_min, max({dd}) AS date_max, count(DISTINCT {dd}) AS distinct_dates FROM {src}""")
per_year = con.execute(f"SELECT year({dd}) AS year, count(*) AS rows FROM {src} GROUP BY 1 ORDER BY 1").fetchdf()
out["rows_per_year"] = per_year.to_dict("records")
out["row_order"] = one(f"""WITH x AS (SELECT {dd} AS d, lag({dd}) OVER () AS prev FROM {src})
  SELECT count(*) AS rows, count(*) FILTER (WHERE d <> prev) AS date_changes_between_consecutive_rows FROM x""")
out["time"] = one(f"""SELECT
  count(*) FILTER (WHERE regexp_matches({T}, '^([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]')) AS pattern_hh_mm_ss,
  count(*) FILTER (WHERE regexp_matches({T}, '^[0-9]{{1,2}}:[0-9]{{2}}[.][0-9]+$')) AS pattern_mm_ss_fraction,
  count(*) FILTER (WHERE {T} = '00:00.0') AS exactly_00_00_0,
  max(try_cast(split_part({T}, ':', 1) AS INTEGER)) AS max_first_field FROM {src}""")
types = con.execute(f"""SELECT {Q['txn_type']} AS txn_type, count(*) AS rows,
  count(*) FILTER (WHERE {Q['start']} IS NULL OR trim({Q['start']}) = '') AS blank_start,
  count(*) FILTER (WHERE {Q['end']} IS NULL OR trim({Q['end']}) = '') AS blank_end
  FROM {src} GROUP BY 1 ORDER BY 2 DESC""").fetchdf()
out["txn_types"] = types.to_dict("records")
out["stations"] = one(f"""SELECT count(DISTINCT trim({Q['start']})) AS raw_start_names,
  count(DISTINCT {canon(Q['start'])}) AS canonical_start_names FROM {src}""")
variants = con.execute(f"""SELECT DISTINCT trim({Q['start']}) AS raw, {canon(Q['start'])} AS canonical FROM {src}
  WHERE {Q['start']} IS NOT NULL AND trim({Q['start']}) <> {canon(Q['start'])} LIMIT 10""").fetchdf()
out["name_variants_examples"] = variants.to_dict("records")
out["duplicates"] = one(f"""SELECT (SELECT count(*) FROM {src}) - (SELECT count(*) FROM (SELECT DISTINCT * FROM {src})) AS exact_duplicate_rows""")

for k, v in out.items():
    print(f"\n== {k} ==\n{json.dumps(v, indent=1, default=str)}")
rows = out["dates"]["rows"]
hh = out["time"]["pattern_hh_mm_ss"] / max(rows, 1)
print("\n" + "=" * 60)
print("HOUR RECOVERABLE?", "YES -> METRO_GRAIN=hour possible" if hh > 0.5 else
      f"NO ({hh:.1%} of rows are hh:mm:ss; max first field = {out['time']['max_first_field']}) -> use GRAIN='day'")
share = out["row_order"]["date_changes_between_consecutive_rows"] / max(rows, 1)
print(f"DATE ORDER: {out['dates']['distinct_dates']} distinct dates in this chunk; "
      f"date changes on {share:.1%} of consecutive rows -> "
      + ("rows are NOT date-sorted (a window needs many/all chunks)" if share > 0.2 else "rows look date-sorted in blocks"))
(C.REPORTS_DIR / f"chunk_profile_{path.name.split('.')[0]}.json").write_text(json.dumps(out, indent=1, default=str))
print(f"\nOutput saved to: {TXT_PATH}")
