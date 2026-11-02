"""Stage 2 test. Tunes the simulator's noise on 2024, then scores 2025 and 2026 walk-forward.

Scores are Brier scores (lower is better): the average squared gap between the predicted chance and
what happened (1 or 0). The baseline gives every driver the historical rate for their grid slot,
e.g. the share of pole sitters who won, and the overall retirement rate."""
import itertools, json, pickle, os, numpy as np, pandas as pd
from features import build, FEATURES, PACE_FEATURES
from probability import race_inputs, simulate
from tuned import PARAMS, HALF_LIFE

EVENTS = {"win": lambda p: p == 1, "podium": lambda p: p <= 3, "points": lambda p: p <= 10}

def collect_inputs(df, seasons, feats):
    f = f"data/sim_inputs_{'_'.join(map(str, seasons))}.pkl"
    if os.path.exists(f) and os.environ.get("REUSE_INPUTS"):
        return pickle.load(open(f, "rb"))
    out = []
    for rid in sorted(df.loc[df.season.isin(seasons), "race_id"].unique()):
        train, test = df[df.race_id < rid], df[df.race_id == rid]
        inp = race_inputs(train, test, feats, PARAMS, HALF_LIFE)
        out.append((rid, train[["grid_f", "position", "not_finished"]], test, inp))
    pickle.dump(out, open(f, "wb"))
    return out

def grid_baseline(train, test):
    t = train[train.position.notna()].copy()
    t["slot"] = t.grid_f.clip(upper=20)
    rates = {e: t.groupby("slot").position.apply(lambda s: f(s).mean()) for e, f in EVENTS.items()}
    slot = test.grid_f.clip(upper=20)
    b = {e: slot.map(r).fillna(0).to_numpy() for e, r in rates.items()}
    b["dnf"] = np.full(len(test), t.not_finished.mean())
    return b

def score(prob, test):
    pos = test.position.to_numpy(); dnf = test.not_finished.to_numpy()
    s = {e: np.mean((prob[e] - f(pos)) ** 2) for e, f in EVENTS.items()}
    s["dnf"] = np.mean((prob["dnf"] - dnf) ** 2)
    s["winner_logloss"] = -np.log(max(prob["win"][pos == 1][0], 1e-3))
    return s

def tune(cache):
    grid = dict(a=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0], b=[0.0, 0.05, 0.1, 0.2, 0.3],
                wet_mult=[1.0, 1.5, 2.0], noise=["normal", "gumbel"])
    best = None
    for a, b, wm, nz in itertools.product(*grid.values()):
        sc = [score(simulate(**inp, a=a, b=b, wet_mult=wm, noise=nz, n=4000, seed=1), test) for _, _, test, inp in cache]
        m = pd.DataFrame(sc)[["win", "podium", "points"]].mean().mean()
        if best is None or m < best[0]: best = (m, dict(a=a, b=b, wet_mult=wm, noise=nz))
    return best[1]

def blend(prob, base, w):
    """Mix simulator and grid-slot chances. The grid slot is very strong at the front of the field,
    the simulator is better further back; the weights are picked on 2024."""
    out = dict(prob)
    for e in EVENTS:
        out[e] = w[e] * prob[e] + (1 - w[e]) * base[e]
    out["win"] = out["win"] / out["win"].sum()
    return out

def base_for(train, test):
    b = grid_baseline(train, test); b["win"] = b["win"] / b["win"].sum(); return b

def tune_blend(cache, sim):
    sims = [(simulate(**inp, **sim, n=4000, seed=1), base_for(train, test), test) for _, train, test, inp in cache]
    w = {}
    for e, f in EVENTS.items():
        errs = {x: np.mean([np.mean((x * p[e] + (1 - x) * b[e] - f(t.position.to_numpy())) ** 2) for p, b, t in sims])
                for x in np.round(np.arange(0, 1.01, 0.1), 1)}
        w[e] = min(errs, key=errs.get)
    return w

if __name__ == "__main__":
    df = build(); feats = FEATURES + PACE_FEATURES
    val = collect_inputs(df, (2024,), feats)
    sim = tune(val)
    w = tune_blend(val, sim)
    json.dump(dict(sim=sim, blend=w), open("sim_params.json", "w"))
    print("Simulator settings picked on 2024:", sim)
    print("Simulator weight in the blend (rest is grid slot):", w)

    rows, calib = [], []
    for rid, train, test, inp in collect_inputs(df, (2025, 2026), feats):
        raw = simulate(**inp, **sim, n=10000, seed=rid)
        base = base_for(train, test)
        prob = blend(raw, base, w)
        rows.append(dict(season=rid // 100, method="simulator only", **score(raw, test)))
        rows.append(dict(season=rid // 100, method="blend (final)", **score(prob, test)))
        rows.append(dict(season=rid // 100, method="baseline: grid slot", **score(base, test)))
        calib.append(pd.DataFrame({"p": prob["podium"], "hit": (test.position <= 3).to_numpy()}))
    res = pd.DataFrame(rows)
    cols = ["win", "podium", "points", "dnf", "winner_logloss"]
    print("\nBrier scores (lower is better), 2025 and 2026:")
    print(res.groupby("method")[cols].mean().round(4))
    print("\nBy season:"); print(res.groupby(["season", "method"])[cols].mean().round(4))
    c = pd.concat(calib); c["bin"] = pd.cut(c.p, [0, .05, .15, .3, .5, .7, .9, 1], include_lowest=True)
    print("\nPodium calibration (predicted chance vs how often it happened):")
    print(c.groupby("bin", observed=True).agg(predicted=("p", "mean"), actual=("hit", "mean"), drivers=("p", "size")).round(3))
