"""Pick model settings using the 2024 season as validation."""
import itertools, pandas as pd
from features import build, FEATURES, PACE_FEATURES
from evaluate import walk_forward, PARAMS, COLS

df = build()
feats = FEATURES + [f for f in PACE_FEATURES if f in df.columns]
grid = dict(num_leaves=[7, 15], min_child_samples=[20, 50], n_estimators=[200, 500],
            learning_rate=[0.02], half_life=[1.0, 2.0, 4.0])
results = []
for nl, mcs, ne, lr, hl in itertools.product(*grid.values()):
    p = dict(PARAMS, num_leaves=nl, min_child_samples=mcs, n_estimators=ne, learning_rate=lr)
    r, _ = walk_forward(df, (2024,), feats, p, hl, baselines=False)
    m = r[COLS].mean()
    results.append(dict(num_leaves=nl, min_child_samples=mcs, n_estimators=ne, half_life=hl, **m))
    print(results[-1], flush=True)
res = pd.DataFrame(results)
# Rank on order accuracy, with winner and podium as tie-breakers.
res["score"] = res.spearman + 0.1 * res.winner + 0.1 * res.podium
print(res.sort_values("score", ascending=False).head(5).round(3))
best = res.sort_values("score", ascending=False).iloc[0]
with open("tuned.py", "w") as f:
    f.write("from evaluate import PARAMS as _P\n")
    f.write(f"PARAMS = dict(_P, num_leaves={int(best.num_leaves)}, min_child_samples={int(best.min_child_samples)}, "
            f"n_estimators={int(best.n_estimators)}, learning_rate=0.02)\nHALF_LIFE = {best.half_life}\n")
