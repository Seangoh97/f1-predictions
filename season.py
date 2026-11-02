"""Stage 3: championship odds. Simulates every remaining race (and sprint) of a season many times.

Before a weekend there is no qualifying or practice data, so this uses a separate "pre-weekend"
pace model built only on form, car pace from earlier races, and circuit history. Each simulated
season also gives every driver a random form shift that lasts for the rest of the year (an upgrade
that works, a car that falls back), so results in one race carry over to the next.

    python season.py                      # current season, after the latest race
    python season.py 2025 --after 19      # replay: odds as they stood after round 19 of 2025"""
import argparse, json, numpy as np, pandas as pd
from collect import get, BASE
from features import build
from evaluate import fit_predict
from probability import fit_dnf
from tuned import PARAMS, HALF_LIFE
import logbook

GP_POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1])
SPRINT_POINTS = np.array([8, 7, 6, 5, 4, 3, 2, 1])

PRE_FEATURES = ["drv_pos_5", "drv_gained_10", "drv_incident_20", "drv_qgap_5", "drv_starts",
                "team_pos_5", "team_gap_5", "team_mech_10", "circ_shuffle", "drv_circ_pos",
                "drv_rpace_3", "team_rpace_3", "field_size", "round", "era_2022", "era_2026"]
PRE_DNF = ["team_mech_10", "drv_incident_20", "circ_dnf", "drv_starts", "era_2026"]

def standings(season, rnd):
    d = get(f"{BASE}/{season}/{rnd}/driverstandings/?limit=100")["MRData"]["StandingsTable"]["StandingsLists"][0]
    c = get(f"{BASE}/{season}/{rnd}/constructorstandings/?limit=100")["MRData"]["StandingsTable"]["StandingsLists"][0]
    drv = pd.DataFrame([dict(driver=s["Driver"]["driverId"], points=float(s["points"]), wins=int(s["wins"]))
                        for s in d["DriverStandings"]])
    con = pd.DataFrame([dict(team=s["Constructor"]["constructorId"], points=float(s["points"]))
                        for s in c["ConstructorStandings"]])
    return drv, con

def remaining(season, rnd):
    races = get(f"{BASE}/{season}/?limit=100")["MRData"]["RaceTable"]["Races"]
    return [dict(round=int(r["round"]), date=r["date"], circuit=r["Circuit"]["circuitId"], name=r["raceName"],
                 sprint="Sprint" in r) for r in races if int(r["round"]) > rnd]

def future_rows(season, rnd, races):
    res = pd.read_csv("data/results.csv")
    lineup = res[(res.season == season) & (res["round"] == rnd)][["driver", "team", "number"]]
    rows = [dict(season=season, round=r["round"], date=r["date"], circuit=r["circuit"], status="Upcoming",
                 grid=np.nan, position=np.nan, points=np.nan, laps=np.nan, quali_pos=np.nan,
                 q1=np.nan, q2=np.nan, q3=np.nan, **l) for r in races for l in lineup.to_dict("records")]
    return pd.DataFrame(rows), lineup

def race_noise(exp_pos, a, b):
    return a + b * (exp_pos - 1)

def params_for(season):
    """Settings measured only on seasons before this one, so a replay of a past season is fair."""
    allp = json.load(open("season_params.json"))
    usable = [int(k) for k in allp if int(k) < season]
    if not usable:
        raise SystemExit(f"No simulator settings measured before {season}. Run tune_season.py.")
    return allp[str(max(usable))]

def run(season, rnd, n=20000, seed=0, quiet=False, log=False):
    cfg = params_for(season)
    races = remaining(season, rnd)
    drv_now, con_now = standings(season, rnd)
    if not races:
        raise SystemExit("No races left to simulate.")
    up, lineup = future_rows(season, rnd, races)
    df = build(upcoming=up, cutoff=season * 100 + rnd)
    train = df[df.position.notna()]
    fut = df[df.outcome == "upcoming"]

    _, model = fit_predict(train, fut[fut["round"] == races[0]["round"]], PRE_FEATURES, PARAMS, HALF_LIFE)
    dnf_model = fit_dnf(train, season, HALF_LIFE, PRE_DNF)

    drivers = list(lineup.driver)
    extra = [d for d in drv_now.driver if d not in drivers]  # drivers with points but no seat now
    all_drivers = drivers + extra
    teams = list(lineup.team)
    rng = np.random.default_rng(seed)
    D = len(drivers)
    pts = np.tile(drv_now.set_index("driver").points.reindex(all_drivers).fillna(0).to_numpy(), (n, 1))
    wins = np.tile(drv_now.set_index("driver").wins.reindex(all_drivers).fillna(0).to_numpy(), (n, 1))
    form = rng.normal(0, cfg["form_sd"], (n, D))
    p_wet = cfg["p_wet"]

    # Points still available after each race, to find when the title is mathematically decided.
    per_race = [25 + (8 if r["sprint"] else 0) for r in races]
    left_after = [sum(per_race[i + 1:]) for i in range(len(races))]
    clinch = np.full(n, -1)
    per_race = []

    for i, race in enumerate(races):
        t = fut[fut["round"] == race["round"]].set_index("driver").reindex(drivers)
        exp_pos = model.predict(t[PRE_FEATURES])
        p_dnf = dnf_model.predict_proba(t[PRE_DNF])[:, 1]
        for kind in (["sprint"] if race["sprint"] else []) + ["gp"]:
            wet = rng.random(n) < p_wet
            sd = race_noise(exp_pos, cfg["a"], cfg["b"])[None, :] * np.where(wet[:, None], cfg["wet_mult"], 1.0)
            score = exp_pos[None, :] + form + sd * rng.standard_normal((n, D))
            pdnf = p_dnf * (0.5 if kind == "sprint" else 1.0)  # a sprint is a third of the distance
            pdnf = np.where(wet[:, None], np.minimum(pdnf * cfg["wet_dnf_mult"], 0.9), pdnf)
            score = np.where(rng.random((n, D)) < pdnf, 1000 + rng.random((n, D)), score)
            pos = score.argsort(1).argsort(1)  # 0 = winner
            table = SPRINT_POINTS if kind == "sprint" else GP_POINTS
            lookup = np.zeros(D); lookup[:len(table)] = table
            pts[:, :D] += lookup[pos]
            if kind == "gp":
                wins[:, :D] += pos == 0
                per_race.append(dict(round=race["round"], name=race["name"], date=race["date"],
                    circuit=race["circuit"], sprint=race["sprint"], drivers=drivers, teams=teams,
                    exp_pos=(pos.mean(0) + 1).tolist(), win=(pos == 0).mean(0).tolist(),
                    podium=(pos < 3).mean(0).tolist(), points=(pos < 10).mean(0).tolist(),
                    dnf=(score >= 1000).mean(0).tolist(),
                    pos_dist=np.stack([(pos == k).mean(0) for k in range(D)], axis=1).tolist()))
        top2 = np.sort(pts, axis=1)[:, -2:]
        done = (clinch < 0) & (top2[:, 1] - top2[:, 0] > left_after[i])
        clinch[done] = i

    champ = (pts + wins * 1e-3).argmax(1)
    final_rank = (-(pts + wins * 1e-3)).argsort(1).argsort(1)
    d = pd.DataFrame({"driver": all_drivers,
                      "now": drv_now.set_index("driver").points.reindex(all_drivers).fillna(0).to_numpy(),
                      "title %": np.bincount(champ, minlength=len(all_drivers)) / n * 100,
                      "top 3 %": (final_rank < 3).mean(0) * 100,
                      "exp. pts": pts.mean(0),
                      "range (5-95%)": [f"{lo:.0f}-{hi:.0f}" for lo, hi in zip(*np.percentile(pts, [5, 95], axis=0))]})
    d = d.sort_values("exp. pts", ascending=False).round(1)

    team_ids = sorted(set(teams) | set(con_now.team))
    tpts = np.tile(con_now.set_index("team").points.reindex(team_ids).fillna(0).to_numpy(), (n, 1))
    now_d = drv_now.set_index("driver").points.reindex(drivers).fillna(0).to_numpy()
    for j, tm in enumerate(teams):
        tpts[:, team_ids.index(tm)] += pts[:, j] - now_d[j]
    c = pd.DataFrame({"team": team_ids, "now": con_now.set_index("team").points.reindex(team_ids).fillna(0).to_numpy(),
                      "title %": np.bincount(tpts.argmax(1), minlength=len(team_ids)) / n * 100,
                      "exp. pts": tpts.mean(0)}).sort_values("exp. pts", ascending=False).round(1)
    if not quiet:
        names = ", ".join(r["name"].replace(" Grand Prix", "") + (" (sprint)" if r["sprint"] else "") for r in races)
        print(f"\n{season} championship after round {rnd}. Still to run: {names}\n")
        print("Drivers"); print(d.head(10).to_string(index=False))
        decided = pd.Series(clinch).map(lambda i: races[i]["name"].replace(" Grand Prix", "") if i >= 0 else "level on points (decided by wins)")
        share = (decided.value_counts(normalize=True) * 100).round(1)
        order = [r["name"].replace(" Grand Prix", "") for r in races] + ["level on points (decided by wins)"]
        print("\nDrivers' title mathematically decided at:")
        for k in order:
            if k in share and share[k] >= 0.5: print(f"  {k}: {share[k]}%")
        print("\nConstructors"); print(c.to_string(index=False))
    if log:
        logbook.save(d.drop(columns="range (5-95%)").rename(columns={"title %": "p_title", "top 3 %": "p_top3",
                     "exp. pts": "exp_pts", "now": "points_now"}).assign(p_title=lambda x: x.p_title / 100,
                     p_top3=lambda x: x.p_top3 / 100), "season_predictions", season=season, after_round=rnd)
    if quiet == "full":
        decided = pd.Series(clinch).map(lambda i: races[i]["name"] if i >= 0 else "Level on points")
        return d, c, per_race, (decided.value_counts(normalize=True)).to_dict()
    return d, c

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("season", type=int, nargs="?", default=pd.Timestamp.now().year)
    ap.add_argument("--after", type=int, help="round to simulate from (default: latest completed)")
    a = ap.parse_args()
    if a.after is None:
        res = pd.read_csv("data/results.csv")
        a.after = int(res[res.season == a.season]["round"].max())
    latest = int(pd.read_csv("data/results.csv").query("season == @a.season")["round"].max())
    run(a.season, a.after, log=(a.after == latest and a.season == pd.Timestamp.now().year))
