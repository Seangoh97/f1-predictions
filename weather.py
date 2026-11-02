"""Rain forecasts for each race from Open-Meteo, plus which races were actually wet.

Historical races use Open-Meteo's archive of past forecasts (what the forecast said at the time,
not what actually fell), so the model is trained on the same kind of input it gets before a real race.
Wet races are taken from OpenF1: a race counts as wet if anyone ran intermediates or full wets."""
import time, requests, numpy as np, pandas as pd
from collect import get, BASE

HIST = "https://historical-forecast-api.open-meteo.com/v1/forecast"
LIVE = "https://api.open-meteo.com/v1/forecast"

def schedule(season):
    races = get(f"{BASE}/{season}/?limit=100")["MRData"]["RaceTable"]["Races"]
    return [dict(season=season, round=int(r["round"]), date=r["date"], time=r.get("time", "13:00:00Z"),
                 lat=float(r["Circuit"]["Location"]["lat"]), lon=float(r["Circuit"]["Location"]["long"]))
            for r in races]

def race_rain(race, live=False):
    """Forecast rain (mm) over the race window, and the highest hourly rain probability if given."""
    start = pd.Timestamp(f"{race['date']}T{race['time']}").tz_localize(None).floor("h")
    params = dict(latitude=race["lat"], longitude=race["lon"], hourly="precipitation,precipitation_probability",
                  timezone="UTC", start_date=str(start.date()), end_date=str((start + pd.Timedelta(hours=3)).date()))
    for attempt in range(6):
        r = requests.get(LIVE if live else HIST, params=params, timeout=30)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(10 * (attempt + 1)); continue
        r.raise_for_status(); break
    r.raise_for_status()
    h = pd.DataFrame(r.json()["hourly"]); h["time"] = pd.to_datetime(h["time"])
    w = h[(h.time >= start) & (h.time < start + pd.Timedelta(hours=2))]
    prob = pd.to_numeric(w.precipitation_probability, errors="coerce").max()
    return dict(rain_mm=float(w.precipitation.fillna(0).sum()), rain_prob=prob)

def wet_races(path="data"):
    """(season, round) of races where intermediates or wets were used, from OpenF1 (2023 onward)."""
    ses = pd.read_csv(f"{path}/of_sessions.csv"); st = pd.read_csv(f"{path}/of_stints.csv")
    res = pd.read_csv(f"{path}/results.csv")[["season", "round", "date"]].drop_duplicates()
    res["date"] = pd.to_datetime(res.date)
    race = ses[ses.session_name == "Race"].copy()
    race["date"] = pd.to_datetime(race.date_start).dt.tz_localize(None).dt.normalize()
    wet_keys = set(st[st.compound.isin(["INTERMEDIATE", "WET"])].session_key)
    out = []
    for r in race.itertuples():
        d = (res.date - r.date).dt.days
        hit = res[(d >= -1) & (d <= 1) & (res.season == r.year)]
        if len(hit):
            out.append(dict(season=r.year, round=int(hit["round"].iloc[0]), wet=int(r.session_key in wet_keys)))
    return pd.DataFrame(out)

def main(first=2022, last=None, path="data"):
    if last is None:
        last = pd.Timestamp.now(tz="UTC").year
    import os
    old = pd.read_csv(f"{path}/weather.csv") if os.path.exists(f"{path}/weather.csv") else None
    done = set() if old is None else set(zip(old.season, old["round"]))
    rows = [] if old is None else old.drop(columns="wet", errors="ignore").to_dict("records")
    for season in range(first, last + 1):
        for race in schedule(season):
            if pd.Timestamp(race["date"]) > pd.Timestamp.now() or (season, race["round"]) in done: continue
            rows.append(dict(season=season, round=race["round"], **race_rain(race)))
            pd.DataFrame(rows).to_csv(f"{path}/weather.csv", index=False)  # save as we go
            time.sleep(0.2)
        print("weather", season, flush=True)
    w = pd.DataFrame(rows).merge(wet_races(path), on=["season", "round"], how="left")
    w.to_csv(f"{path}/weather.csv", index=False)
    print(w.groupby("wet").rain_mm.describe())

if __name__ == "__main__":
    main()
