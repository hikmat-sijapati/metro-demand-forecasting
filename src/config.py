"""Central configuration. Edit after running 02_inspect_chunk.py on real chunks."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REF_DIR = ROOT / "data" / "reference"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIGS_DIR = REPORTS_DIR / "figures"

# Column names seen in chunk 0001 (matched case-insensitively).
COLS = {
    "txn_type": "txn_type", "txn_date": "txn_date", "txn_time": "txn_time",
    "start": "start_location", "end": "end_location",
}
# Real chunks (profiled): ISO dates "2020-11-01" and times "17:25:58.000". Tried in this order.
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y"]

# Granularity: the profiled chunks have real hh:mm:ss times, so the approved station-HOUR
# task is feasible. Rows whose time is exactly 00:00:00.000 are treated as "time unknown".
GRAIN = os.getenv("METRO_GRAIN", "hour")
MIN_KNOWN_TIME_SHARE = 0.9    # hour grain: a date with less known-time share is set to NaN, not zero
PARAMS = {
    "day":  dict(freq="D", horizon=1,  week=7,   day=1,  lags=[1, 2, 7, 14, 21, 28]),
    "hour": dict(freq="h", horizon=24, week=168, day=24, lags=[24, 48, 168, 336, 504, 672]),
}
P = PARAMS[GRAIN]
# UAE weekend: Friday-Saturday until 31 Dec 2021, Saturday-Sunday from 1 Jan 2022 (pandas dayofweek, Mon=0).
# Verify in the weekday profile of your window (09_db_diagnostics.py section 9).
WEEKEND_RULE = {"change_date": "2022-01-01", "before": [4, 5], "after": [5, 6]}
LOW_DAY_RATIO = 0.3              # a day whose network total is < 30% of its 29-day rolling median is treated as missing (outage)

# Bounded analysis window (set after you know which chunks cover which dates).
# Chosen from the data diagnostics: the longest stretch with real hourly times, complete coverage, stable
# volumes and a stable station set (pre-pandemic). See the report's Section 2 for the justification.
WINDOW_START = "2018-10-01"
WINDOW_END = "2020-01-31"

# Optional station reference table (separate dataset) -> data/reference/metro_stations.csv
REF_FILE = REF_DIR / "metro_stations.csv"
REF_COLS = {"name": "location_name_english", "zone": "zone_id",
            "lat": "station_location_latitude", "lon": "station_location_longitude",
            "line": "line_name"}

SPLIT = (0.70, 0.15, 0.15)       # chronological, by date: train / val / test
SEED = 42
TOTAL_CHUNKS = 3113               # fallback; total_chunks() prefers the size of data/interim/chunk_index.json

# ---- modelling / analysis (06, 07) ----
PEAK_HOURS = [7, 8, 9, 17, 18, 19]       # hour grain: used to split peak vs off-peak error
CV_SPLITS = 3                             # rolling-origin (expanding window) folds on train+val
TUNE_ROWS = 60_000                        # rows used for hyper-parameter search (time-ordered subsample)
SVR_RBF_TRAIN_ROWS = 30_000               # RBF-SVR is trained on a seeded subsample (cost grows ~quadratically)
RF_MAX_SAMPLES = 400_000                  # cap on rows per tree in the final forest fit
BOOTSTRAP_B = 2000                        # block bootstrap over test dates
PI_LEVEL = 0.90                           # Bayesian prediction-interval level


def total_chunks() -> int:
    """Number of chunks currently listed by the API (from the saved index), else TOTAL_CHUNKS."""
    import json
    f = INTERIM_DIR / "chunk_index.json"
    return len(json.loads(f.read_text())) if f.exists() else TOTAL_CHUNKS
