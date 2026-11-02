"""Walk-forward test: before each race, train on every earlier race, then predict that race.

Tuning is done on the 2024 season only. 2025 and 2026 are kept as the final test, so the
reported numbers aren't flattered by choices made while looking at them."""
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import spearmanr
from features import build, FEATURES, PACE_FEATURES

PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)

def race_metrics(pred_order, actual):
    rho = spearmanr(pred_order, actual).correlation
    win = float(actual[np.argmin(pred_order)] == 1)
    pod = len(set(np.argsort(pred_order)[:3]) & set(np.where(actual <= 3)[0])) / 3
    pts = len(set(np.argsort(pred_order)[:10]) & set(np.where(actual <= 10)[0])) / 10
    mae = np.mean(np.abs(pred_order - actual))
    return dict(spearman=rho, mae=mae, winner=win, podium=pod, top10=pts)

def fit_predict(train, test, feats, params=PARAMS, half_life=2.0):
    # Train on classified finishers only: DNFs are mostly random and teach the model noise.
    fin = train["outcome"] == "finished"
    w = 0.5 ** ((test.season.iloc[0] - train.season) / half_life)
    model = lgb.LGBMRegressor(**params).fit(train.loc[fin, feats], train.loc[fin, "position"],
                                            sample_weight=w[fin])
    return model.predict(test[feats]), model

def walk_forward(df, seasons, feats, params=PARAMS, half_life=2.0, baselines=True, start_rid=None):
    rows = []
    for rid in sorted(df.loc[df.season.isin(seasons), "race_id"].unique()):
        train, test = df[df.race_id < rid], df[df.race_id == rid]
        if start_rid: train = train[train.race_id >= start_rid]
        actual = test["position"].to_numpy()
        score, model = fit_predict(train, test, feats, params, half_life)
        cands = {"model": score}
        if baselines:
            cands["baseline: grid"] = test["grid_f"].to_numpy()
        for name, s in cands.items():
            order = pd.Series(s).rank(method="first").to_numpy()
            rows.append(dict(race_id=rid, season=rid // 100, method=name, **race_metrics(order, actual)))
    return pd.DataFrame(rows), model

COLS = ["spearman", "mae", "winner", "podium", "top10"]

if __name__ == "__main__":
    df = build()
    feats = FEATURES + [f for f in PACE_FEATURES if f in df.columns]
    from tuned import PARAMS as BEST, HALF_LIFE
    res, model = walk_forward(df, (2025, 2026), feats, BEST, HALF_LIFE)
    print(res.groupby(["season", "method"])[COLS].mean().round(3))
    print("\nAll test races:"); print(res.groupby("method")[COLS].mean().round(3))
    imp = pd.Series(model.booster_.feature_importance("gain"), feats).sort_values(ascending=False)
    print("\nFeature importance (latest model):"); print((imp / imp.sum()).round(3).head(15))
