"""Build dashboard.html: an interactive page with race predictions, position odds, driver form,
championship odds and model performance. Open the file in any browser.

    python dashboard.py            # current and previous season (takes a few minutes)

Rerun after each qualifying session or race weekend to refresh it."""
import json, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from collect import get, BASE
from features import build, FEATURES, PACE_FEATURES
from probability import race_inputs, simulate
from evaluate_probs import blend, base_for, score
from evaluate import fit_predict
from tuned import PARAMS, HALF_LIFE
import season as season_mod
import predict as predict_mod

_cur = int(pd.read_csv("data/results.csv").season.max())
SEASONS = (_cur - 1, _cur)
FEATS = FEATURES + PACE_FEATURES
SPRINT_POINTS = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}

def r3(x): return [round(float(v), 4) for v in x]

def schedule(season):
    races = get(f"{BASE}/{season}/?limit=100")["MRData"]["RaceTable"]["Races"]
    return {int(r["round"]): dict(name=r["raceName"], circuit_name=r["Circuit"]["circuitName"],
            locality=r["Circuit"]["Location"]["locality"], country=r["Circuit"]["Location"]["country"],
            date=r["date"], sprint="Sprint" in r) for r in races}

def names():
    """Driver display names from the latest standings of each season."""
    out = {}
    for s in SEASONS:
        for x in get(f"{BASE}/{s}/drivers/?limit=100")["MRData"]["DriverTable"]["Drivers"]:
            out[x["driverId"]] = dict(name=f'{x["givenName"]} {x["familyName"]}', code=x.get("code", x["familyName"][:3].upper()))
        for x in get(f"{BASE}/{s}/constructors/?limit=100")["MRData"]["ConstructorTable"]["Constructors"]:
            out[x["constructorId"]] = dict(name=x["name"])
    return out

def qualified_race(season, rnd, cfg):
    """Full race model for a race whose qualifying is done but which hasn't been run yet."""
    import weather
    up, _ = predict_mod.upcoming_rows(season, rnd, {})
    dfu = build(upcoming=up)
    rid = season * 100 + rnd
    train, test = dfu[(dfu.race_id < rid) & dfu.position.notna()], dfu[dfu.race_id == rid].copy()
    try:
        race_w = [r for r in weather.schedule(season) if r["round"] == rnd][0]
        test["rain_mm"] = weather.race_rain(race_w, live=True)["rain_mm"]
    except Exception as err:  # weather service busy: fall back to the usual rain rate
        print("Rain forecast unavailable, using the usual rate:", err)
        test["rain_mm"] = np.nan
    inp = race_inputs(train, test, FEATS, PARAMS, HALF_LIFE)
    raw = simulate(**inp, **cfg["sim"], n=20000, seed=rid, full=True)
    prob = blend(raw, base_for(train, test), cfg["blend"])
    rain = test.rain_mm.iloc[0]
    return dict(status="qualified", drivers=list(test.driver), teams=list(test.team),
                grid=[int(g) for g in test.grid_f], win=r3(prob["win"]), podium=r3(prob["podium"]),
                points=r3(prob["points"]), dnf=r3(prob["dnf"]), exp_pos=r3(prob["exp_pos"]),
                pos_dist=[r3(x) for x in raw["pos_dist"]], p_wet=round(inp["p_wet"], 3),
                rain_mm=None if pd.isna(rain) else round(float(rain), 1))

def main():
    cfg = json.load(open("sim_params.json"))
    df = build()
    sched = {s: schedule(s) for s in SEASONS}
    races = []

    # Completed races: replay each one using only races before it.
    done = df[df.season.isin(SEASONS) & df.position.notna()]
    for rid in sorted(done.race_id.unique()):
        train, test = df[df.race_id < rid], df[df.race_id == rid]
        s, r = int(test.season.iloc[0]), int(test["round"].iloc[0])
        inp = race_inputs(train, test, FEATS, PARAMS, HALF_LIFE)
        raw = simulate(**inp, **cfg["sim"], n=10000, seed=rid, full=True)
        base = base_for(train, test)
        prob = blend(raw, base, cfg["blend"])
        sm, sb = score(prob, test), score(base, test)
        rho_m = spearmanr(prob["exp_pos"], test.position).correlation
        rho_g = spearmanr(test.grid_f, test.position).correlation
        races.append(dict(season=s, round=r, status="done", **sched[s][r],
            circuit=test.circuit.iloc[0], p_wet=round(inp["p_wet"], 3),
            rain_mm=None if pd.isna(test.rain_mm.iloc[0]) else round(float(test.rain_mm.iloc[0]), 1),
            circ_shuffle=None if pd.isna(test.circ_shuffle.iloc[0]) else round(float(test.circ_shuffle.iloc[0]), 2),
            circ_dnf=None if pd.isna(test.circ_dnf.iloc[0]) else round(float(test.circ_dnf.iloc[0]), 3),
            drivers=list(test.driver), teams=list(test.team), grid=[int(g) for g in test.grid_f],
            actual=[int(p) for p in test.position], status_text=list(test.status),
            win=r3(prob["win"]), podium=r3(prob["podium"]), points=r3(prob["points"]), dnf=r3(prob["dnf"]),
            exp_pos=r3(prob["exp_pos"]), pos_dist=[r3(row) for row in raw["pos_dist"]],
            score=dict(model={k: round(float(v), 4) for k, v in sm.items()},
                       grid={k: round(float(v), 4) for k, v in sb.items()},
                       spearman_model=round(float(rho_m), 3), spearman_grid=round(float(rho_g), 3))))
        print("race", rid, flush=True)

    # Current season: championship simulation and pre-weekend odds for each race still to run.
    cur = max(SEASONS)
    cur_done = done[done.season == cur]
    standings_out, last = None, 0
    if len(cur_done):
        last = int(cur_done["round"].max())
        try:
            d, c, upcoming, decided = season_mod.run(cur, last, n=20000, quiet="full")
        except SystemExit:  # season finished, nothing left to simulate
            d = c = None; upcoming, decided = [], {}
        if d is not None:
            standings_out = dict(
                drivers=d.rename(columns={"title %": "title", "top 3 %": "top3", "exp. pts": "exp", "range (5-95%)": "range"}).to_dict("records"),
                teams=c.rename(columns={"title %": "title", "exp. pts": "exp"}).to_dict("records"),
                decided={k: round(v, 4) for k, v in decided.items()})
        p_wet_default = json.load(open("season_params.json")).get(str(cur - 1), {}).get("p_wet", 0.1)
        for u in upcoming:
            entry = dict(season=cur, round=u["round"], status="forecast", **sched[cur][u["round"]],
                         circuit=u["circuit"], drivers=u["drivers"], teams=u["teams"],
                         grid=None, actual=None, win=r3(u["win"]), podium=r3(u["podium"]), points=r3(u["points"]),
                         dnf=r3(u["dnf"]), exp_pos=r3(u["exp_pos"]), pos_dist=[r3(x) for x in u["pos_dist"]],
                         p_wet=p_wet_default, rain_mm=None)
            hist_c = df[(df.circuit == u["circuit"]) & df.position.notna()]
            entry["circ_shuffle"] = round(float(hist_c.abs_change.mean()), 2) if len(hist_c) else None
            entry["circ_dnf"] = round(float(hist_c.not_finished.mean()), 3) if len(hist_c) else None
            races.append(entry)
        # If qualifying for the next race is in, replace its pre-weekend odds with the full race model.
        nxt = last + 1
        e = next((x for x in races if x["season"] == cur and x["round"] == nxt), None)
        if e is not None and get(f"{BASE}/{cur}/{nxt}/qualifying/?limit=100")["MRData"]["RaceTable"]["Races"]:
            e.update(qualified_race(cur, nxt, cfg))
    # Driver form: one row per driver per race.
    form = df[df.season.isin(SEASONS) & df.position.notna()][
        ["season", "round", "driver", "team", "grid_f", "position", "status", "best_gap", "race_pace", "points"]]
    form = form.replace({np.nan: None})
    form_rows = [dict(season=int(r.season), round=int(r["round"]), driver=r.driver, team=r.team,
                      grid=int(r.grid_f), pos=int(r.position), status=r.status,
                      qgap=None if r.best_gap is None else round(float(r.best_gap), 3),
                      rpace=None if r.race_pace is None else round(float(r.race_pace), 3),
                      pts=float(r.points)) for _, r in form.iterrows()]

    # Points after each round, Grand Prix plus sprint.
    sp = pd.read_csv("data/sprint.csv")
    sp = sp[sp.season.isin(SEASONS)]
    sp["pts"] = sp.sprint_pos.map(SPRINT_POINTS).fillna(0)
    res = pd.read_csv("data/results.csv")
    res = res[res.season.isin(SEASONS)][["season", "round", "driver", "team", "points"]].rename(columns={"points": "pts"})
    allp = pd.concat([res, sp[["season", "round", "driver", "pts"]]]).groupby(["season", "round", "driver"]).pts.sum().reset_index()
    progression = {}
    for s, g in allp.groupby("season"):
        piv = g.pivot(index="round", columns="driver", values="pts").fillna(0).cumsum()
        progression[int(s)] = dict(rounds=[int(x) for x in piv.index], drivers={k: r3(v) for k, v in piv.items()})
    team_of = res.sort_values(["season", "round"]).groupby(["season", "driver"]).team.last()

    # Feature importance from the latest pace model.
    latest = df[df.position.notna()].race_id.max()
    _, model = fit_predict(df[df.race_id < latest], df[df.race_id == latest], FEATS, PARAMS, HALF_LIFE)
    imp = pd.Series(model.booster_.feature_importance("gain"), FEATS)
    imp = (imp / imp.sum()).sort_values(ascending=False).head(14)

    data = dict(generated=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"), current_season=cur,
        last_round=last, races=races, form=form_rows, progression=progression,
        team_of={f"{s}|{d}": t for (s, d), t in team_of.items()}, names=names(),
        standings=standings_out,
        importance=[dict(feature=k, share=round(float(v), 4)) for k, v in imp.items()],
        sim=cfg, season_params=json.load(open("season_params.json")))
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)): return None if not np.isfinite(x) else float(x)
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.bool_,)): return bool(x)
        return x
    payload = json.dumps(clean(data), allow_nan=False, default=str)
    Path("data/dashboard_data.json").write_text(payload, encoding="utf-8")
    html = Path("dashboard_template.html").read_text(encoding="utf-8").replace("/*__DATA__*/null", payload)
    Path("dashboard.html").write_text(html, encoding="utf-8")
    print("Wrote dashboard.html")

if __name__ == "__main__":
    main()
