"""Score the saved predictions once the races have been run.

For each race, uses the last prediction saved before the race started and compares it with the
result and with the grid-slot baseline (historical rates for each starting position).
Run collect.py first so the latest results are in."""
import argparse, numpy as np, pandas as pd
from pathlib import Path
from features import classify
from evaluate_probs import grid_baseline, score
import weather

def history():
    r = pd.read_csv("data/results.csv")
    r["outcome"] = r.status.map(classify)
    r = r[r.outcome != "dns"].copy()
    n = r.groupby(["season", "round"]).driver.transform("count")
    r["grid_f"] = np.where(r.grid == 0, n, r.grid)
    r["not_finished"] = (r.outcome != "finished").astype(float)
    r["race_id"] = r.season * 100 + r["round"]
    return r

def score_races(log_file):
    log = pd.read_csv(log_file, parse_dates=["run_time_utc"])
    log["run_time_utc"] = log.run_time_utc.dt.tz_localize("UTC")
    hist = history()
    rows = []
    for (season, rnd), runs in log.groupby(["season", "round"]):
        actual = hist[(hist.season == season) & (hist["round"] == rnd)]
        if actual.empty:
            print(f"{season} round {rnd}: no result yet"); continue
        race = [r for r in weather.schedule(season) if r["round"] == rnd][0]
        start = pd.Timestamp(f"{race['date']}T{race['time']}")
        before = runs[runs.run_time_utc < start]
        if before.empty:
            print(f"{season} round {rnd}: no prediction saved before the start, skipped"); continue
        pred = before[before.run_time_utc == before.run_time_utc.max()]
        m = pred.merge(actual[["driver", "position", "not_finished", "grid_f"]], on="driver", how="inner")
        prob = {"win": m.p_win.to_numpy() / m.p_win.sum(), "podium": m.p_podium.to_numpy(),
                "points": m.p_points.to_numpy(), "dnf": m.p_dnf.to_numpy()}
        base = grid_baseline(hist[hist.race_id < season * 100 + rnd], m)
        base["win"] = base["win"] / base["win"].sum()
        name = race.get("name", f"round {rnd}")
        fav = m.loc[m.p_win.idxmax()]
        winner = m.loc[m.position == 1, "driver"].iloc[0]
        rows.append(dict(race=f"{season} R{rnd}", model_fav=f"{fav.driver} ({fav.p_win:.0%})", winner=winner,
                         winner_chance=f"{m.loc[m.position == 1, 'p_win'].iloc[0]:.0%}",
                         **{f"model_{k}": v for k, v in score(prob, m).items()},
                         **{f"grid_{k}": v for k, v in score(base, m).items()}))
    return pd.DataFrame(rows)

def _print_scores(res):
    print(res[["race", "model_fav", "winner", "winner_chance"]].to_string(index=False))
    cols = ["win", "podium", "points", "dnf", "winner_logloss"]
    tab = pd.DataFrame({"model": [res[f"model_{c}"].mean() for c in cols],
                        "grid slot": [res[f"grid_{c}"].mean() for c in cols]}, index=cols).round(4)
    print(f"\nAverage Brier scores over {len(res)} race(s), lower is better:")
    print(tab.T.to_string())
    better = (res.model_podium < res.grid_podium).sum()
    print(f"\nModel beat the grid baseline on podium chances in {better} of {len(res)} race(s).")

def _print_season():
    sf = Path("logs/season_predictions.csv")
    if sf.exists():
        s = pd.read_csv(sf)
        s = s.sort_values("run_time_utc").groupby(["season", "after_round", "driver"]).tail(1)
        for season, g in s.groupby("season"):
            top = g[g.after_round == g.after_round.max()].nlargest(5, "exp_pts").driver
            piv = g[g.driver.isin(top)].pivot(index="after_round", columns="driver", values="p_title") * 100
            print(f"\n{season} title chance (%) after each round:"); print(piv.round(1).to_string())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/race_predictions.csv")
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    res = score_races(a.log) if Path(a.log).exists() else pd.DataFrame()
    if not Path(a.log).exists():
        print("No race predictions saved yet. Run predict.py after qualifying, before the race.")
    elif res.empty:
        print("No saved race predictions can be scored yet.")
    if not res.empty:
        _print_scores(res)
    _print_season()
