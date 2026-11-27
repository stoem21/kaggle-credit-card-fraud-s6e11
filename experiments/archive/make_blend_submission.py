"""Rank-blend submission + zone-1 post-processing.
Members (weights chosen on 3-fold-seed-averaged OOF, see EXPERIMENTS.md 2026-11-26):
  0.4 lgb_l3 + 0.4 xgb_d2 (outputs/models_v2) + 0.2 mlp (outputs/nn_mlp)
Each member's test prediction = mean over its 3 fold seeds (each already a 5-fold average), then ranked.
Zone 1: merchant_risk_score below the lowest value seen among training frauds -> sent to the bottom
(in-fold test: +0.00008 on every fold seed, no fraud inside the zone).
Run: .venv/bin/python -m experiments.make_blend_submission <out name>
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.fraud.config import ID, OUTPUTS, RAW, SUBMISSION, TARGET
from src.fraud.metrics import auc

SEEDS = [42, 7, 2026]
MEMBERS = {("models_v2", "lgb_l3"): 0.4, ("models_v2", "xgb_d2"): 0.4, ("nn_mlp", "mlp"): 0.2}
out_name = sys.argv[1]

train = pd.read_csv(RAW / "train.csv", usecols=["merchant_risk_score", TARGET])
test = pd.read_csv(RAW / "test.csv", usecols=[ID, "merchant_risk_score"])
sample = pd.read_csv(RAW / "sample_submission.csv")
assert (sample[ID].to_numpy() == test[ID].to_numpy()).all()
y = train[TARGET].astype(int).to_numpy()


def blend(kind):
    return sum(w * rankdata(np.mean([np.load(OUTPUTS / d / f"{kind}_{n}_{s}.npy") for s in SEEDS], axis=0))
               for (d, n), w in MEMBERS.items())


oof, pred = blend("oof"), blend("test")
t1 = train.loc[y == 1, "merchant_risk_score"].min()
zone = test["merchant_risk_score"].to_numpy() < t1
pred = np.where(zone, pred - 1e9, pred)
final = (rankdata(pred) - 0.5) / len(pred)
sub = sample[[ID]].copy()
sub[TARGET] = final
sub.to_csv(SUBMISSION / f"{out_name}.csv", index=False)
print(f"OOF of the blend (3-fold-seed averaged, before zone): {auc(y, oof):.5f}")
print(f"zone 1 threshold {t1}; test rows in zone: {zone.sum():,}")
print(f"wrote submission/{out_name}.csv {sub.shape}")
