"""One full update: new results and lap data, prediction logging, scoring and a fresh dashboard.

GitHub Actions runs this on a schedule (see .github/workflows/update.yml). It works the same on a
laptop: `python auto_update.py`.

What it does each time:
  1. Downloads new results, qualifying and sprints, new OpenF1 laps, and race weather.
  2. If qualifying for the next race is done and the race hasn't started, saves a race prediction
     to logs/ (so it can be scored afterwards).
  3. After each new race, saves the championship odds to logs/.
  4. Scores saved predictions and rebuilds dashboard.html.
Steps 1 to 3 carry on if one of the free data services is down; only a failed results download
or dashboard build stops the run."""
import subprocess, sys
import pandas as pd
from pathlib import Path
from collect import get, BASE

def run(args, required=False):
    print(f"\n== python {' '.join(args)}", flush=True)
    ok = subprocess.run([sys.executable, "-W", "ignore"] + args).returncode == 0
    if not ok:
        print(f"!! python {' '.join(args)} failed", flush=True)
        if required:
            sys.exit(1)
    return ok

def next_race():
    """(season, round, race start) of the next race to be run, or None if the calendar is done."""
    now = pd.Timestamp.now(tz="UTC")
    for season in (now.year, now.year + 1):
        races = get(f"{BASE}/{season}/?limit=100")["MRData"]["RaceTable"]["Races"]
        for r in races:
            start = pd.Timestamp(f"{r['date']}T{r.get('time', '13:00:00Z')}")
            if start > now:
                return season, int(r["round"]), start
    return None

def main():
    run(["collect.py"], required=True)
    run(["collect_openf1.py"])
    run(["weather.py"])

    # Race prediction once qualifying is in and before lights out.
    nxt = next_race()
    if nxt:
        season, rnd, start = nxt
        quali = get(f"{BASE}/{season}/{rnd}/qualifying/?limit=100")["MRData"]["RaceTable"]["Races"]
        if quali:
            run(["predict.py", str(season), str(rnd)])
        else:
            print(f"\nNo qualifying yet for {season} round {rnd}; race prediction will be saved after it.")

    # Championship odds, once per completed round of the current season.
    res = pd.read_csv("data/results.csv")
    year = pd.Timestamp.now().year
    if (res.season == year).any():
        last = int(res[res.season == year]["round"].max())
        log = Path("logs/season_predictions.csv")
        done = log.exists() and ((pd.read_csv(log).season == year) & (pd.read_csv(log).after_round == last)).any()
        if not done:
            run(["season.py", str(year)])

    run(["score_log.py"])
    run(["dashboard.py"], required=True)

if __name__ == "__main__":
    main()
