"""Chunk -> running accumulator (DuckDB file) of counts by (date, hour, kind, station).

Why an accumulator: every chunk is a random ~1/3104 slice of ALL dates (2017-2026), so a
per-chunk table barely compresses. Instead each chunk's counts are appended to one table and
compacted (summed) every few chunks; a ledger (`chunks_done`) makes the job resumable and
prevents double counting.

Real-format handling: ISO dates; hh:mm:ss.000 times (rows at exactly 00:00:00.000, or with an
unparseable time, get hour = -1 = "time unknown"); 'Check in'/'Check out' casing; blank
start_location on Check out rows (station then comes from end_location); stray spaces in names.
"""
import sys
from pathlib import Path
import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from utils import resolve_cols

DB_PATH = C.INTERIM_DIR / "counts.duckdb"
COMPACT_EVERY = 20


def stem_of(path) -> str:
    return Path(path).name.split(".")[0]


def connect():
    C.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute("""CREATE TABLE IF NOT EXISTS counts
                   (date DATE, hour SMALLINT, kind VARCHAR, station VARCHAR, n BIGINT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS chunks_done
                   (stem VARCHAR PRIMARY KEY, rows_total BIGINT, rows_bad_date BIGINT,
                    rows_time_unknown BIGINT, rows_other_kind BIGINT, rows_no_station BIGINT)""")
    return con


def is_done(stem: str, con=None) -> bool:
    con = con or connect()
    return con.execute("SELECT count(*) FROM chunks_done WHERE stem = ?", [stem]).fetchone()[0] > 0


def canon(x: str) -> str:
    """SQL: trim, collapse whitespace, remove spaces around '/'."""
    return rf"regexp_replace(regexp_replace(trim({x}), '\s*/\s*', '/', 'g'), '\s+', ' ', 'g')"


def sniff_src(path) -> str:
    """read_csv() source that works whether or not the file is really gzipped."""
    p = Path(path)
    with open(p, "rb") as f:
        gz = f.read(2) == b"\x1f\x8b"
    return (f"read_csv('{p.as_posix()}', all_varchar=true, ignore_errors=true, "
            f"compression='{'gzip' if gz else 'none'}')")


def date_expr(col: str) -> str:
    parts = [f"CAST(try_strptime(trim({col}), '{f}') AS DATE)" for f in C.DATE_FORMATS]
    return "COALESCE(" + ", ".join(parts) + ")"


def hour_expr(col: str) -> str:
    t = f"trim({col})"
    return (f"COALESCE(CASE WHEN regexp_matches({t}, '^00:00:00([.]0+)?$') THEN -1 "
            f"WHEN regexp_matches({t}, '^([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]') "
            f"THEN CAST(substr({t}, 1, 2) AS INTEGER) END, -1)")


def compact(con):
    con.execute("BEGIN")
    con.execute("""CREATE TABLE counts_new AS
                   SELECT date, hour, kind, station, sum(n)::BIGINT AS n FROM counts GROUP BY ALL""")
    con.execute("DROP TABLE counts")
    con.execute("ALTER TABLE counts_new RENAME TO counts")
    con.execute("COMMIT")


def add_chunk(path, con=None) -> bool:
    """Aggregate one chunk into the accumulator. Returns False if it was already done."""
    con = con or connect()
    stem = stem_of(path)
    if is_done(stem, con):
        return False
    src = sniff_src(path)
    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()]
    c = resolve_cols(cols, C.COLS)
    q = {k: '"' + v + '"' for k, v in c.items()}
    typ = f"lower(trim({q['txn_type']}))"
    kind = f"CASE WHEN {typ} LIKE '%in' THEN 'in' WHEN {typ} LIKE '%out' THEN 'out' ELSE 'other' END"
    con.execute("BEGIN")
    try:
        con.execute("DROP TABLE IF EXISTS t")
        con.execute(f"""
          CREATE TEMP TABLE t AS
          WITH p AS (SELECT {date_expr(q['txn_date'])} AS d, {hour_expr(q['txn_time'])} AS h, {kind} AS kind,
                            {canon(q['start'])} AS s_in, {canon(q['end'])} AS s_out FROM {src})
          SELECT d, h, kind,
                 CASE WHEN kind = 'in' THEN COALESCE(NULLIF(s_in, ''), NULLIF(s_out, ''))
                      ELSE COALESCE(NULLIF(s_out, ''), NULLIF(s_in, '')) END AS station
          FROM p""")
        con.execute("""INSERT INTO counts
                       SELECT d, h::SMALLINT, kind, station, count(*)::BIGINT
                       FROM t WHERE d IS NOT NULL AND station IS NOT NULL GROUP BY 1, 2, 3, 4""")
        qual = con.execute("""SELECT count(*), count(*) FILTER (WHERE d IS NULL),
                                     count(*) FILTER (WHERE h = -1), count(*) FILTER (WHERE kind = 'other'),
                                     count(*) FILTER (WHERE d IS NOT NULL AND station IS NULL) FROM t""").fetchone()
        con.execute("INSERT INTO chunks_done VALUES (?, ?, ?, ?, ?, ?)", [stem, *[int(x) for x in qual]])
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    if con.execute("SELECT count(*) FROM chunks_done").fetchone()[0] % COMPACT_EVERY == 0:
        compact(con)
    return True
