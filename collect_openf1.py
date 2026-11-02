"""Download practice and race laps plus tyre stints from OpenF1 (free historical data, 2023 onward).
The free tier allows 30 requests a minute, so this takes around 20 minutes the first time.
Everything is cached in data/openf1/, so later runs only fetch new sessions."""
import json, time, requests, pandas as pd
from pathlib import Path

API = "https://api.openf1.org/v1"
RAW = Path("data/openf1"); RAW.mkdir(parents=True, exist_ok=True)
WANTED = {"Practice 1", "Practice 2", "Practice 3", "Race", "Sprint"}

def get(endpoint, **params):
    key = endpoint + "_" + "_".join(f"{k}{v}" for k, v in params.items())
    f = RAW / f"{key}.json"
    # The current season's session list keeps growing, so always fetch it fresh.
    fresh = endpoint == "sessions" and int(params.get("year", 0)) >= pd.Timestamp.now().year
    if f.exists() and not fresh:
        return json.loads(f.read_text())
    for attempt in range(6):
        r = requests.get(f"{API}/{endpoint}", params=params, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(20 * (attempt + 1)); continue
        if r.status_code in (401, 403):
            # During a live session OpenF1's free tier is closed; the data opens 30 minutes after it ends.
            print(f"OpenF1 is not serving {endpoint} for this session yet (live window). Try again later.")
            time.sleep(2.1)
            return []
        data = [] if r.status_code == 404 else (r.raise_for_status() or r.json())
        # Don't cache empty answers: data for a session that just ended may not be published yet.
        if data:
            f.write_text(json.dumps(data))
        time.sleep(2.1)
        return data
    raise RuntimeError(f"gave up on {endpoint} {params}")

def main(first=2023, last=None, verbose=True):
    if last is None:
        last = pd.Timestamp.now(tz="UTC").year
    # Reuse completed-session tables; fetch only new sessions.
    def previous(name):
        f = Path("data") / name
        return pd.read_csv(f).to_dict("records") if f.exists() and f.stat().st_size else []
    old_sessions = previous("of_sessions.csv")
    old_laps = previous("of_laps.csv")
    old_stints = previous("of_stints.csv")
    lap_keys = {r["session_key"] for r in old_laps}
    stint_keys = {r["session_key"] for r in old_stints}
    complete = {r["session_key"] for r in old_sessions} & lap_keys & stint_keys
    sessions, laps, stints = [], [], []
    for year in range(first, last + 1):
        for s in get("sessions", year=year):
            if s["session_name"] not in WANTED or s.get("is_cancelled"):
                continue
            if pd.Timestamp(s["date_end"]) > pd.Timestamp.now(tz="UTC"):
                continue
            sessions.append(s)
            if s["session_key"] in complete:
                laps.extend(r for r in old_laps if r["session_key"] == s["session_key"])
                stints.extend(r for r in old_stints if r["session_key"] == s["session_key"])
                continue
            for row in get("laps", session_key=s["session_key"]):
                laps.append({k: row.get(k) for k in ["session_key", "driver_number", "lap_number",
                    "lap_duration", "is_pit_out_lap", "i1_speed", "i2_speed", "st_speed"]})
            for row in get("stints", session_key=s["session_key"]):
                stints.append(row)
            if verbose: print(year, s["location"], s["session_name"], flush=True)
    pd.DataFrame(sessions).to_csv("data/of_sessions.csv", index=False)
    pd.DataFrame(laps).to_csv("data/of_laps.csv", index=False)
    pd.DataFrame(stints).to_csv("data/of_stints.csv", index=False)
    if verbose: print("done", len(sessions), "sessions,", len(laps), "laps")

if __name__ == "__main__":
    main()
