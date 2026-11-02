"""Stage 2: turn the pace model into probabilities by simulating each race many times.

Three parts:
  1. Pace model (from evaluate.py): expected finishing position if the driver finishes.
  2. DNF model: chance each driver retires, from car reliability, driver incident history,
     circuit retirement rate and grid slot.
  3. Simulator: in each run, every driver gets their expected position plus random noise,
     some drivers retire, and the order is set. Rain makes the noise wider and retirements likelier."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from evaluate import fit_predict

DNF_FEATURES = ["team_mech_10", "drv_incident_20", "circ_dnf", "grid_pct", "drv_starts", "era_2026"]

def recency(train, season, half_life):
    return (0.5 ** ((season - train.season) / half_life)).to_numpy()

def fit_dnf(train, season, half_life, feats=DNF_FEATURES):
    t = train[train.outcome != "upcoming"]
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.3, max_iter=1000))
    model.fit(t[feats], t.not_finished, logisticregression__sample_weight=recency(t, season, half_life))
    return model

def wet_effects(train):
    """How much rain raises retirement odds, and how likely a race is wet given the forecast.
    Both are learned from races before the one being predicted."""
    lab = train.dropna(subset=["wet"]).drop_duplicates(["race_id", "driver"])
    wet, dry = lab[lab.wet == 1], lab[lab.wet == 0]
    # Odds ratio for retiring in the wet, pulled toward 1 when there are only a few wet races.
    k = 20
    pw = (wet.not_finished.sum() + k * dry.not_finished.mean()) / (len(wet) + k)
    pd_ = dry.not_finished.mean()
    odds_ratio = (pw / (1 - pw)) / (pd_ / (1 - pd_)) if len(dry) else 1.0
    races = lab.drop_duplicates("race_id").dropna(subset=["rain_mm"])
    if races.wet.sum() >= 3 and (races.wet == 0).sum() >= 3:
        m = LogisticRegression(C=1.0).fit(np.log1p(races[["rain_mm"]]), races.wet)
    else:
        m = None
    return odds_ratio, m

def simulate(exp_pos, p_dnf_dry, p_dnf_wet, p_wet, a, b, wet_mult, noise="normal", n=10000, seed=0, full=False):
    rng = np.random.default_rng(seed)
    D = len(exp_pos)
    wet = rng.random(n) < p_wet
    sigma = a + b * (exp_pos - 1)
    scale = np.where(wet[:, None], wet_mult, 1.0) * sigma[None, :]
    eps = rng.standard_normal((n, D)) if noise == "normal" else -rng.gumbel(size=(n, D))
    score = exp_pos[None, :] + scale * eps
    dnf = rng.random((n, D)) < np.where(wet[:, None], p_dnf_wet[None, :], p_dnf_dry[None, :])
    score = np.where(dnf, 1000 + rng.random((n, D)), score)
    pos = score.argsort(axis=1).argsort(axis=1) + 1
    out = dict(win=(pos == 1).mean(0), podium=(pos <= 3).mean(0), points=(pos <= 10).mean(0),
               dnf=dnf.mean(0), exp_pos=pos.mean(0))
    if full:  # chance of each finishing position, one row per driver
        out["pos_dist"] = np.stack([(pos == k).mean(0) for k in range(1, D + 1)], axis=1)
    return out

def race_inputs(train, test, feats, params, half_life):
    """Everything the simulator needs for one race, from models trained on earlier races only."""
    season = int(test.season.iloc[0])
    exp_pos, _ = fit_predict(train, test, feats, params, half_life)
    p_dry = fit_dnf(train, season, half_life).predict_proba(test[DNF_FEATURES])[:, 1]
    odds_ratio, wet_model = wet_effects(train)
    odds = p_dry / (1 - p_dry) * odds_ratio
    p_wetdnf = odds / (1 + odds)
    rain = test.rain_mm.iloc[0] if "rain_mm" in test else np.nan
    if wet_model is not None and pd.notna(rain):
        p_wet = float(wet_model.predict_proba(np.log1p(pd.DataFrame({"rain_mm": [rain]})))[0, 1])
    else:
        p_wet = 0.08  # long-run share of wet races
    return dict(exp_pos=np.asarray(exp_pos), p_dnf_dry=p_dry, p_dnf_wet=p_wetdnf, p_wet=p_wet)

def predict_probs(train, test, feats, params, half_life, sim, n=10000):
    inp = race_inputs(train, test, feats, params, half_life)
    out = simulate(**inp, **sim, n=n)
    res = test[["driver", "team", "grid_f"]].copy()
    for k, v in out.items(): res[k] = v
    res["p_wet"] = inp["p_wet"]
    return res.sort_values("exp_pos")
