"""Predict a Grand Prix once qualifying is done: win, podium, points and retirement chances.

    python predict.py 2026 19            # next race, after qualifying
    python predict.py 2026 18 --replay   # rerun a past race as if it hadn't happened yet

Grid is taken as the qualifying order. If grid penalties are announced, pass them like
    python predict.py 2026 19 --grid verstappen=15 hamilton=20"""
import argparse, pandas as pd, numpy as np
from collect import get, BASE, qtime
import collect_openf1
from features import build, FEATURES, PACE_FEATURES
import json
from tuned import PARAMS, HALF_LIFE
from probability import race_inputs, simulate
from evaluate_probs import blend, base_for
import weather
import logbook

def upcoming_rows(season, rnd, grid_overrides):
    race = get(f"{BASE}/{season}/{rnd}/qualifying/?limit=100")["MRData"]["RaceTable"]["Races"]
    if not race:
        raise SystemExit(f"No qualifying data yet for {season} round {rnd}. Try again after qualifying.")
    race = race[0]
    # Apply grid penalties: penalised drivers drop to their slot and everyone behind moves up.
    order = [q["Driver"]["driverId"] for q in sorted(race["QualifyingResults"], key=lambda q: int(q["position"]))]
    unknown = set(grid_overrides) - set(order)
    if unknown:
        raise SystemExit(f"Unknown driver id(s) in --grid: {', '.join(sorted(unknown))}. Use ids like {order[0]}.")
    grid_order = [d for d in order if d not in grid_overrides]
    for d, slot in sorted(grid_overrides.items(), key=lambda x: x[1]):
        grid_order.insert(min(max(slot, 1) - 1, len(grid_order)), d)
    grid_of = {d: i + 1 for i, d in enumerate(grid_order)}
    rows = []
    for q in race["QualifyingResults"]:
        d = q["Driver"]["driverId"]; pos = int(q["position"])
        rows.append(dict(season=season, round=rnd, date=race["date"], number=int(q["number"]),
                         circuit=race["Circuit"]["circuitId"], driver=d, team=q["Constructor"]["constructorId"],
                         grid=grid_of[d], position=np.nan, status="Upcoming",
                         points=np.nan, laps=np.nan, quali_pos=pos,
                         q1=qtime(q.get("Q1")), q2=qtime(q.get("Q2")), q3=qtime(q.get("Q3"))))
    return pd.DataFrame(rows), race["raceName"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("season", type=int); ap.add_argument("round", type=int)
    ap.add_argument("--replay", action="store_true", help="hide the real result and predict it")
    ap.add_argument("--grid", nargs="*", default=[], help="driver=grid_position overrides")
    a = ap.parse_args()
    overrides = {k: int(v) for k, v in (x.split("=") for x in a.grid)}

    collect_openf1.main(verbose=False)  # picks up this weekend's practice sessions (cached sessions are skipped)
    up, name = upcoming_rows(a.season, a.round, overrides)
    actual = None
    if a.replay:
        res = pd.read_csv("data/results.csv")
        actual = res[(res.season == a.season) & (res["round"] == a.round)].set_index("driver")["position"]

    df = build(upcoming=up)
    rid = a.season * 100 + a.round
    feats = FEATURES + PACE_FEATURES
    train, test = df[(df.race_id < rid) & df.position.notna()], df[df.race_id == rid].copy()

    # Rain: past races use the stored forecast; an upcoming race gets a fresh one.
    if not a.replay or test.rain_mm.isna().all():
        try:
            race = [r for r in weather.schedule(a.season) if r["round"] == a.round][0]
            test["rain_mm"] = weather.race_rain(race, live=not a.replay)["rain_mm"]
        except Exception as err:  # weather service busy: fall back to the usual rain rate
            print("Rain forecast unavailable, using the usual rate:", err)
            test["rain_mm"] = np.nan

    cfg = json.load(open("sim_params.json"))
    inp = race_inputs(train, test, feats, PARAMS, HALF_LIFE)
    prob = blend(simulate(**inp, **cfg["sim"], n=20000, seed=rid), base_for(train, test), cfg["blend"])

    out = test[["driver", "team", "grid_f"]].rename(columns={"grid_f": "grid"}).copy()
    out["grid"] = out["grid"].astype(int)
    for k, label in [("win", "win %"), ("podium", "podium %"), ("points", "points %"), ("dnf", "DNF %")]:
        out[label] = (prob[k] * 100).round(1)
    out["exp. pos"] = prob["exp_pos"].round(1)
    out = out.sort_values("exp. pos")
    if actual is not None:
        out["actual"] = out.driver.map(actual).astype("Int64")
    print(f"\n{a.season} {name}")
    rain = test.rain_mm.iloc[0]
    rain_txt = "no forecast available" if pd.isna(rain) else f"forecast rain in race window: {rain:.1f} mm"
    print(f"Chance of a wet race: {inp['p_wet'] * 100:.0f}%  ({rain_txt})\n")
    print(out.to_string(index=False))
    if not a.replay:
        log = pd.DataFrame({"driver": test.driver.values, "team": test.team.values, "grid": test.grid_f.values,
                            "p_win": prob["win"], "p_podium": prob["podium"], "p_points": prob["points"],
                            "p_dnf": prob["dnf"], "exp_pos": prob["exp_pos"]})
        logbook.save(log, "race_predictions", season=a.season, round=a.round, p_wet=round(inp["p_wet"], 3))

if __name__ == "__main__":
    main()
