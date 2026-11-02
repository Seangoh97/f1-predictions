"""Saves every live prediction with a timestamp so it can be scored after the race.
Replays of past races are never saved, so the log only holds genuine before-the-race predictions."""
import pandas as pd
from pathlib import Path

LOGS = Path("logs"); LOGS.mkdir(exist_ok=True)

def save(df, name, **meta):
    df = df.copy()
    df.insert(0, "run_time_utc", pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S"))
    for i, (k, v) in enumerate(meta.items(), start=1):
        df.insert(i, k, v)
    f = LOGS / f"{name}.csv"
    df.to_csv(f, mode="a", header=not f.exists(), index=False)
    print(f"\nSaved to {f}")
