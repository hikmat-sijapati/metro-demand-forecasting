"""Accumulate chunks, then build the modelling table.

  python src/03_aggregate.py add "data/raw/*.csv.gz"     # chunk(s) -> running accumulator (resumable)
  python src/03_aggregate.py status                      # progress + data-quality totals
  python src/03_aggregate.py combine --start 2024-01-01 --end 2026-06-30

`combine` writes data/interim/counts.parquet at the configured GRAIN (hour/day):
  * station-name variants merged; per-station grid zero-filled between its first and last period
  * dates with NO data (and, for hour grain, dates whose known-time share < MIN_KNOWN_TIME_SHARE)
    are set to NaN (missing), NOT zero, so gaps are never mistaken for zero demand
  * WARNING if fewer than all chunks are processed: counts are then a random SAMPLE at rate k/3104
    (use --scale-to-full for an estimate, and say so in the report)
"""
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import aggcore
from utils import norm_name, resolve_cols

ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
p1 = sub.add_parser("add"); p1.add_argument("files", nargs="+")
sub.add_parser("status")
p3 = sub.add_parser("combine")
p3.add_argument("--start", default=C.WINDOW_START); p3.add_argument("--end", default=C.WINDOW_END)
p3.add_argument("--kind", default="in", choices=["in", "out"])
p3.add_argument("--scale-to-full", action="store_true")
p3.add_argument("--out", default=str(C.INTERIM_DIR / "counts.parquet"))
a = ap.parse_args()

con = aggcore.connect()

if a.cmd == "add":
    files = [f for pat in a.files for f in (sorted(glob.glob(pat)) or [pat])]
    for i, f in enumerate(files, 1):
        t = time.time()
        new = aggcore.add_chunk(f, con)
        print(f"[{i}/{len(files)}] {Path(f).name}: {'added' if new else 'already done'} ({time.time() - t:.1f}s)")
    sys.exit(0)

ledger = con.execute("SELECT stem, rows_total, rows_bad_date, rows_time_unknown, rows_other_kind, rows_no_station "
                     "FROM chunks_done").fetchall()
done_stems = {r[0] for r in ledger}
done = len(done_stems)
q = [sum(r[i] for r in ledger) for i in range(1, 6)]
q = [int(x or 0) for x in q]
if a.cmd == "status":
    rng = con.execute("SELECT min(date), max(date), count(*) FROM counts").fetchone()
    index_file = C.INTERIM_DIR / "chunk_index.json"
    if index_file.exists():
        index = json.loads(index_file.read_text())
        indexed_stems = {str(item["file"]).split(".")[0] for item in index.values()}
        current_done = len(indexed_stems & done_stems)
        legacy = len(done_stems - indexed_stems)
        print(f"current indexed chunks processed: {current_done}/{len(indexed_stems)}"
              f" | older accumulator stems absent from index: {legacy}")
    else:
        print(f"chunks done: {done}/{C.total_chunks()}")
    print(f"accumulator rows: {rng[2]:,} | dates {rng[0]} .. {rng[1]}")
    print(f"rows read: {q[0]:,} | bad date: {q[1]:,} | time unknown (00:00:00 or unparseable): {q[2]:,} "
          f"({q[2] / max(q[0], 1):.1%}) | other txn_type: {q[3]:,} | no station: {q[4]:,}")
    sys.exit(0)

# ----- combine -----
if done == 0:
    sys.exit("No chunks processed yet. Run the `add` step (or 01_download.py --stream).")
index_file = C.INTERIM_DIR / "chunk_index.json"
if index_file.exists():
    index = json.loads(index_file.read_text())
    indexed_stems = {str(item["file"]).split(".")[0] for item in index.values()}
    export_prefixes = {str(item["folder"]).rsplit("_", 1)[0] for item in index.values()}
    current_done_stems = indexed_stems & done_stems
    legacy_stems = done_stems - indexed_stems
    missing_stems = indexed_stems - done_stems
    total = len(indexed_stems)
    current_done = len(current_done_stems)
else:
    indexed_stems, current_done_stems, legacy_stems, missing_stems = done_stems, done_stems, set(), set()
    export_prefixes = set()
    total = current_done = done
    index = {}

rows_by_stem = {r[0]: r[1] for r in ledger}
rows_read_current = sum(rows_by_stem[s] for s in current_done_stems)
rows_read_legacy = sum(rows_by_stem[s] for s in legacy_stems)
archive_range = con.execute("SELECT min(date), max(date), count(DISTINCT date) FROM counts").fetchone()
aggcore.compact(con)
df = con.execute("""SELECT date, hour, station, sum(n) AS n FROM counts
                    WHERE kind = ? AND date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
                    GROUP BY ALL""", [a.kind, a.start, a.end]).fetchdf()
if df.empty:
    sys.exit("No rows in the window. Check dates / processed chunks (python src/03_aggregate.py status).")
df["date"] = pd.to_datetime(df["date"])
df["hour"] = df["hour"].astype(int)

df["key"] = df["station"].map(norm_name)
disp = (df.groupby(["key", "station"])["n"].sum().reset_index()
          .sort_values("n", ascending=False).drop_duplicates("key").set_index("key")["station"])
df["station"] = df["key"].map(disp)

window = pd.date_range(a.start, a.end, freq="D")
net = df.groupby("date")["n"].sum()
missing_days = set(window) - set(net.index)
if C.GRAIN == "hour":
    known = df[df["hour"] >= 0].groupby("date")["n"].sum().reindex(net.index).fillna(0)
    low_known = set((known / net)[(known / net) < C.MIN_KNOWN_TIME_SHARE].index)
    bad_days = missing_days | low_known
    d2 = df[df["hour"] >= 0].copy()
    d2["ts"] = d2["date"] + pd.to_timedelta(d2["hour"], unit="h")
else:
    low_known, bad_days, d2 = set(), missing_days, df.copy()
    d2["ts"] = d2["date"]
# Outage days: some days hold only a handful of rows (e.g. 1-30 rows instead of ~500k). They are not "missing"
# (rows exist) but would look like zero demand, so treat them as missing too.
daily = net.reindex(window, fill_value=0.0)
med = daily.rolling(29, center=True, min_periods=7).median()
low_volume = set(daily.index[(daily > 0) & (daily < C.LOW_DAY_RATIO * med)])
bad_days = bad_days | low_volume
agg = d2.groupby(["station", "ts"], as_index=False)["n"].sum().rename(columns={"n": "checkins"})

bad_list = list(bad_days)
parts = []
for st, g in agg.groupby("station"):
    idx = pd.date_range(g["ts"].min(), g["ts"].max(), freq=C.P["freq"])
    s = g.set_index("ts")["checkins"].reindex(idx, fill_value=0).astype(float)
    s[idx.normalize().isin(bad_list)] = np.nan
    parts.append(pd.DataFrame({"station": st, "ts": idx, "checkins": s.values}))
out = pd.concat(parts, ignore_index=True)

rate = current_done / max(total, 1)
if missing_stems:
    print(f"WARNING: only {current_done}/{total} currently indexed chunks processed -> counts are a RANDOM SAMPLE at rate {rate:.3f}, "
          "not full counts." + (" Scaled by 1/rate (estimates)." if a.scale_to_full else " Use --scale-to-full for estimates."))
    if a.scale_to_full:
        out["checkins"] = out["checkins"] / rate
if legacy_stems:
    print(f"NOTE: the accumulator also contains {len(legacy_stems)} previously processed chunk stems "
          "that are absent from the current API index; their counts remain included.")
if C.REF_FILE.exists():
    ref = pd.read_csv(C.REF_FILE)
    rc = resolve_cols(ref.columns, C.REF_COLS)
    ref = ref.rename(columns={rc["name"]: "ref_name", rc["zone"]: "zone", rc["lat"]: "lat",
                              rc["lon"]: "lon", rc["line"]: "line"})
    ref["key"] = ref["ref_name"].map(norm_name)
    ref = ref.drop_duplicates("key")[["key", "zone", "lat", "lon", "line"]]
    out["key"] = out["station"].map(norm_name)
    out = out.merge(ref, on="key", how="left").drop(columns="key")
    un = out.loc[out["lat"].isna(), "station"].drop_duplicates()
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    un.to_csv(C.REPORTS_DIR / "unmatched_stations.csv", index=False)

qual = {"grain": C.GRAIN, "window": [a.start, a.end], "chunks_processed": current_done, "chunks_total": total,
        "chunk_exports_in_index": len(export_prefixes), "accumulator_chunks": done,
        "legacy_chunks_in_accumulator": len(legacy_stems),
        "sampling_rate": rate, "scaled_to_full": bool(a.scale_to_full), "rows_read": q[0],
        "rows_read_current_index": rows_read_current, "rows_read_legacy": rows_read_legacy,
        "archive_date_start": str(archive_range[0]), "archive_date_end": str(archive_range[1]),
        "archive_distinct_dates": int(archive_range[2]),
        "rows_bad_date": q[1],
        "rows_time_unknown": q[2], "time_unknown_share": q[2] / max(q[0], 1), "rows_other_kind": q[3],
        "rows_no_station": q[4], "days_in_window": len(window), "days_missing_entirely": len(missing_days),
        "days_low_known_time": len(low_known), "days_low_volume": len(low_volume), "stations": int(out["station"].nunique()),
        "periods": int(len(out)), "nan_share": float(out["checkins"].isna().mean()),
        "zero_share_of_known": float((out["checkins"].dropna() == 0).mean()),
        "total_checkins": float(out["checkins"].sum())}
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
out.to_parquet(a.out, index=False)
C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
(C.REPORTS_DIR / "data_quality.json").write_text(json.dumps(qual, indent=1))
print(json.dumps(qual, indent=1)); print("saved", a.out, out.shape)
