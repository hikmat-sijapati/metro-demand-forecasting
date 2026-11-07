"""SYNTHETIC data in the REAL chunk format, ONLY to test the pipeline offline: ISO dates,
hh:mm:ss.000 times, 'Check in'/'Check out', blank start_location on Check out, doubled spaces
in names, midnight-placeholder rows (some whole dates), and rows shuffled across all dates
then cut into equal chunks (like the real random slices). Never use outputs in the report.

  python src/make_synthetic.py --out data/synth_raw
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
ap.add_argument("--start", default="2025-09-01"); ap.add_argument("--days", type=int, default=75)
ap.add_argument("--chunks", type=int, default=6)
a = ap.parse_args()
rng = np.random.default_rng(0)
stations = {"Union": 1.0, "BurJuman": 0.9, "Al Rigga": 0.7, "Burj Khalifa/ Dubai Mall": 0.8,
            "Dubai Internet City": 0.5, "Rashidiya": 0.6, "Jebel Ali": 0.3, "Etisalat": 0.4}
commuter = np.array([0,0,0,0,0,2,8,20,26,14,8,8,9,9,8,9,12,18,16,9,5,3,2,1], float)
leisure = np.array([0,0,0,0,0,1,2,4,6,8,10,12,13,13,12,12,13,14,14,12,9,6,3,1], float)
loads = ["2025-09-23T22:11:56.000Z", "2025-10-14T00:09:51.000Z", "2026-07-06T13:05:10.000Z"]
days = pd.date_range(a.start, periods=a.days)
placeholder_days = set(pd.to_datetime(rng.choice(days, size=4, replace=False)))      # whole dates without real times
missing_days = {days[10], days[11]}
low_days = {days[40]}                                                  # outage: only ~3% of normal volume                                    # outage: no rows at all
frames = []
for day in days:
    if day in missing_days:
        continue
    weekend = day.dayofweek in (5, 6)
    for st, f in stations.items():
        lam = (leisure if weekend else commuter) * f * (0.8 if day.dayofweek == 4 else 1.0) * (0.03 if day in low_days else 1.0)
        for h, l in enumerate(lam):
            n = rng.poisson(l)
            if not n:
                continue
            if day in placeholder_days or rng.random() < 0.03:
                tm = ["00:00:00.000"] * n
            else:
                m, s = rng.integers(0, 60, n), rng.integers(0, 60, n)
                tm = [f"{h:02d}:{x:02d}:{y:02d}.000" for x, y in zip(m, s)]
            name = st + (" Metro Station" if rng.random() > 0.1 else "  Metro Station")
            for kind in ("Check in", "Check out"):
                frames.append(pd.DataFrame({
                    "end_location": name, "end_zone": "Zone 5", "line_name": "Red Metro Line",
                    "start_location": name if kind == "Check in" else None, "start_zone": "Zone 5",
                    "txn_date": day.date().isoformat(), "txn_time": tm, "txn_type": kind,
                    "load_timestamp": rng.choice(loads)}))
df = pd.concat(frames).sample(frac=1, random_state=0).reset_index(drop=True)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
bounds = np.linspace(0, len(df), a.chunks + 1).astype(int)
for i in range(1, a.chunks + 1):
    df.iloc[bounds[i - 1]:bounds[i]].to_csv(out / f"metro_ridership_SYNTH_{i:04d}.csv.gz", index=False)
print("wrote", a.chunks, "chunks,", len(df), "rows to", out,
      "| placeholder dates:", sorted(d.date().isoformat() for d in placeholder_days),
      "| missing dates:", sorted(d.date().isoformat() for d in missing_days))
