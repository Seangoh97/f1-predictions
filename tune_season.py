"""Settings for the season simulator, measured on 2022 onward.

For each season Y, settings are measured only on seasons up to Y and saved under "Y" in
season_params.json. season.py uses the latest settings from before the season it simulates, so
replaying 2025 uses only 2022-2024, and the live 2026 run uses 2022-2025.

- a, b: how much a result varies around the pre-weekend prediction in a single race.
- form_sd: how much a driver's form drifts for weeks at a time, from how prediction errors for the
  same driver stick together over 5-race stretches. That part is taken out of the single-race noise.
- p_wet, wet_dnf_mult: share of wet races and how much rain raises retirements."""
import itertools, json, os, pickle, numpy as np, pandas as pd
from features import build
from evaluate import fit_predict
from probability import fit_dnf, simulate
from evaluate_probs import score
from season import PRE_FEATURES, PRE_DNF
from tuned import PARAMS, HALF_LIFE

FIRST = 2022
CACHE = "data/season_tuning_cache.pkl"

def walk_forward_cache(df, last):
    cache = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
    for rid in sorted(df.loc[df.season.between(FIRST, last), "race_id"].unique()):
        if rid in cache: continue
        train, test = df[df.race_id < rid], df[df.race_id == rid]
        season = int(test.season.iloc[0])
        exp_pos, _ = fit_predict(train, test, PRE_FEATURES, PARAMS, HALF_LIFE)
        p = fit_dnf(train, season, HALF_LIFE, PRE_DNF).predict_proba(test[PRE_DNF])[:, 1]
        cache[rid] = (test[["driver", "race_id", "season", "position", "outcome", "not_finished"]], np.asarray(exp_pos), p)
        pickle.dump(cache, open(CACHE, "wb"))
        print("fitted", rid, flush=True)
    return cache

def measure(items, df, through):
    best = None
    for a, b in itertools.product([1.0, 1.5, 2.0, 2.5, 3.0, 4.0], [0.0, 0.05, 0.1, 0.2]):
        sc = [score(simulate(e, p, p, 0.0, a, b, 1.0, n=3000, seed=1), t) for t, e, p in items]
        m = pd.DataFrame(sc)[["win", "podium", "points"]].mean().mean()
        if best is None or m < best[0]: best = (m, a, b)
    _, a, b = best
    res = pd.concat([pd.DataFrame({"driver": t.driver, "season": t.season, "race_id": t.race_id,
                                   "err": t.position - e, "fin": t.outcome == "finished"}) for t, e, _ in items])
    res = res[res.fin].sort_values("race_id")
    res["block"] = res.groupby(["driver", "season"]).cumcount() // 5
    blocks = res.groupby(["driver", "season", "block"]).err.agg(["mean", "var", "count"])
    blocks = blocks[blocks["count"] == 5]
    form_var = max(blocks["mean"].var() - blocks["var"].mean() / 5, 0.0)
    lab = df[df.season <= through].dropna(subset=["wet"])
    races = lab.drop_duplicates("race_id")
    ratio = lab[lab.wet == 1].not_finished.mean() / lab[lab.wet == 0].not_finished.mean()
    return dict(a=float(np.sqrt(max(a ** 2 - form_var, 0.25))), b=b, form_sd=float(np.sqrt(form_var)),
                p_wet=round(float(races.wet.mean()), 3) if len(races) else 0.1,
                wet_mult=json.load(open("sim_params.json"))["sim"]["wet_mult"],
                wet_dnf_mult=round(float(np.clip(ratio, 1.0, 2.5)), 2) if len(races) else 1.2,
                measured_on=f"{FIRST}-{through}")

if __name__ == "__main__":
    df = build()
    last = int(df[df.position.notna()].season.max()) - 1  # last fully finished season
    cache = walk_forward_cache(df, last)
    out = {}
    for through in range(FIRST + 1, last + 1):
        items = [v for k, v in cache.items() if k // 100 <= through]
        out[str(through)] = measure(items, df, through)
        print(through, out[str(through)])
    json.dump(out, open("season_params.json", "w"), indent=1)
