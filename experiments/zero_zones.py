"""Zero-fraud zones (idea seen in public notebooks): rows in a region with no fraud are sent to the bottom.
Honest test: zone thresholds are searched on each TRAINING fold only, applied to its validation fold.
Zone 1: merchant_risk_score below the lowest fraud value in the training fold.
Zone 2: velocity_score < a and merchant_risk_score <= b with no training-fold fraud; b from a grid, the
        largest such box is kept.
Scored on the current blend (rank: 0.4 lgb_reg_x3 + 0.4 xgb + 0.2 mlp) per fold seed, same folds as the models.
Run: .venv/bin/python -m experiments.zero_zones
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold

from src.fraud.config import N_FOLDS, OUTPUTS, RAW, TARGET
from src.fraud.metrics import auc

t = pd.read_csv(RAW / "train.csv", usecols=["merchant_risk_score", "velocity_score", TARGET])
y = t[TARGET].astype(int).to_numpy()
mr, vel = t["merchant_risk_score"].to_numpy(), t["velocity_score"].to_numpy()
SEEDS = [42, 7, 2026]


def blend(s):
    r = lambda d, n: rankdata(np.load(OUTPUTS / d / f"oof_{n}_{s}.npy"))
    return 0.4 * r("variance_models", "lgb_reg_x3") + 0.4 * r("variance_models", "xgb") + 0.2 * r("nn_mlp", "mlp")


def find_zones(idx):
    f = idx[y[idx] == 1]
    t1 = mr[f].min()                                  # zone 1: mr < t1
    best = (0, None, None)
    for b in np.arange(t1, 40.01, 0.5):
        in_b = f[mr[f] <= b]
        a = vel[in_b].min() if len(in_b) else np.inf  # zone 2: vel < a and mr <= b
        size = ((vel[idx] < a) & (mr[idx] <= b)).sum()
        if size > best[0]:
            best = (size, a, b)
    return t1, best[1], best[2]


rows = []
for s in SEEDS:
    p = blend(s)
    new = p.copy()
    caught = 0
    for tr, va in StratifiedKFold(N_FOLDS, shuffle=True, random_state=s).split(mr, y):
        t1, a, b = find_zones(tr)
        z = (mr[va] < t1) | ((vel[va] < a) & (mr[va] <= b))
        new[va[z]] -= 1e9                             # bottom, original order kept inside the zone
        caught += y[va[z]].sum()
        rows.append(dict(fold_seed=s, t1=t1, a=a, b=b, zone_rows=int(z.sum()), zone_frauds=int(y[va[z]].sum())))
    print(f"fold_seed={s}: blend {auc(y, p):.5f} -> zones {auc(y, new):.5f} "
          f"(delta {auc(y, new) - auc(y, p):+.5f}); frauds inside zones on validation folds: {caught}")
print(pd.DataFrame(rows).round(2).to_string(index=False))
