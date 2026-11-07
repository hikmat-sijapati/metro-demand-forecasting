"""Report-ready EDA figures (work for day or hour grain)."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

df = pd.read_parquet(C.INTERIM_DIR / "counts.parquet")
df["date"], df["dow"] = df["ts"].dt.normalize(), df["ts"].dt.dayofweek
C.FIGS_DIR.mkdir(parents=True, exist_ok=True)
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
daily = df.groupby("date")["checkins"].sum(min_count=1)

fig, ax = plt.subplots(figsize=(7, 2.8))
daily.plot(ax=ax, lw=0.6, label="daily"); daily.rolling(7, center=True).mean().plot(ax=ax, label="7-day mean")
ax.set_ylabel("Check-ins/day"); ax.set_title("Daily network check-ins (outages, holidays, breaks)"); ax.legend()
fig.tight_layout(); fig.savefig(C.FIGS_DIR / "eda_daily_total.png", dpi=200); plt.close(fig)

wd = daily.groupby(daily.index.dayofweek).mean()
fig, ax = plt.subplots(figsize=(5, 2.8)); ax.bar(days, wd.values)
ax.set_ylabel("Mean network check-ins/day"); ax.set_title("Weekday profile")
fig.tight_layout(); fig.savefig(C.FIGS_DIR / "eda_weekday_profile.png", dpi=200); plt.close(fig)

top = df.groupby("station")["checkins"].sum().nlargest(10).sort_values()
fig, ax = plt.subplots(figsize=(6.5, 3.2)); top.plot.barh(ax=ax)
ax.set_xlabel("Total check-ins in window"); ax.set_title("Ten busiest stations")
fig.tight_layout(); fig.savefig(C.FIGS_DIR / "eda_top_stations.png", dpi=200); plt.close(fig)

if C.GRAIN == "hour":
    hh = df.assign(hour=df["ts"].dt.hour).groupby(["date", "dow", "hour"])["checkins"].sum().reset_index()
    heat = hh.groupby(["dow", "hour"])["checkins"].mean().unstack()
    fig, ax = plt.subplots(figsize=(7, 3.2)); im = ax.imshow(heat.values, aspect="auto", cmap="viridis")
    ax.set_yticks(range(7), days); ax.set_xlabel("Hour of day"); fig.colorbar(im, label="check-ins/hour")
    ax.set_title("Mean network check-ins by weekday and hour")
    fig.tight_layout(); fig.savefig(C.FIGS_DIR / "eda_heatmap_hour_dow.png", dpi=200); plt.close(fig)
print("figures saved to", C.FIGS_DIR)
