"""Download race results and qualifying from Jolpica-F1 and save them as CSV."""
import time, json, requests, pandas as pd
from pathlib import Path

BASE = "https://api.jolpi.ca/ergast/f1"
OUT = Path("data"); OUT.mkdir(exist_ok=True)
CACHE = OUT / "cache"; CACHE.mkdir(exist_ok=True)

def get(url):
    # The current season keeps changing, so only cache past seasons.
    season = url.replace(BASE, "").strip("/").split("/")[0]
    use_cache = season.isdigit() and int(season) < pd.Timestamp.now().year
    f = CACHE / (url.replace(BASE, "").strip("/").replace("/", "_").replace("?", "_").replace("&", "_") + ".json")
    if use_cache and f.exists():
        return json.loads(f.read_text())
    for attempt in range(8):
        r = requests.get(url, timeout=30)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(10 * (attempt + 1), 60)); continue
        r.raise_for_status()
        data = r.json(); time.sleep(0.3)
        if use_cache: f.write_text(json.dumps(data))
        return data
    raise RuntimeError(url)

def fetch_all(season, kind):
    races, offset = {}, 0
    while True:
        d = get(f"{BASE}/{season}/{kind}/?limit=100&offset={offset}")["MRData"]
        for race in d["RaceTable"]["Races"]:
            key = race["round"]
            races.setdefault(key, {k: v for k, v in race.items() if not k.endswith("Results")})
            listkey = [k for k in race if k.endswith("Results")][0]
            races[key].setdefault("rows", []).extend(race[listkey])
        offset += 100
        if offset >= int(d["total"]): return list(races.values())

def qtime(s):
    if not s: return None
    try:
        m, rest = s.split(":") if ":" in s else ("0", s)
        return int(m) * 60 + float(rest)
    except ValueError:
        return None

def main(first=2018, last=None):
    if last is None:
        last = pd.Timestamp.now(tz="UTC").year
    res, qual, sprint = [], [], []
    for season in range(first, last + 1):
        print("season", season)
        for race in fetch_all(season, "results"):
            for r in race["rows"]:
                res.append(dict(season=season, round=int(race["round"]), date=race["date"], number=int(r["number"]),
                    circuit=race["Circuit"]["circuitId"], driver=r["Driver"]["driverId"],
                    team=r["Constructor"]["constructorId"], grid=int(r["grid"]),
                    position=int(r["position"]), status=r["status"],
                    points=float(r["points"]), laps=int(r["laps"])))
        for race in fetch_all(season, "qualifying"):
            for r in race["rows"]:
                qual.append(dict(season=season, round=int(race["round"]),
                    driver=r["Driver"]["driverId"], quali_pos=int(r["position"]),
                    q1=qtime(r.get("Q1")), q2=qtime(r.get("Q2")), q3=qtime(r.get("Q3"))))
        if season >= 2021:  # sprints started in 2021
            for race in fetch_all(season, "sprint"):
                for r in race["rows"]:
                    sprint.append(dict(season=season, round=int(race["round"]),
                        driver=r["Driver"]["driverId"], sprint_pos=int(r["position"]),
                        sprint_grid=int(r["grid"]), sprint_status=r["status"]))
    pd.DataFrame(sprint).to_csv(OUT / "sprint.csv", index=False)
    pd.DataFrame(res).to_csv(OUT / "results.csv", index=False)
    pd.DataFrame(qual).to_csv(OUT / "qualifying.csv", index=False)
    print(len(res), "result rows,", len(qual), "qualifying rows")

if __name__ == "__main__":
    main()
