"""Rank-blend submission v3 (EBM blend) + zone-1 post-processing.
Members and weights (chosen on fold-seed-averaged OOF, see EXPERIMENTS.md 2026-11-26 round4-ebm):
  0.5 ebm (outputs/round4) + 0.2 lgb_l3_b1023_x3 (round5, max_bin 1023: user choice, gain within noise) + 0.2 xgb_d2_x3 (models_v2 for seeds 42/7/2026, round5 for extra seeds)
  + 0.05 mlp + 0.05 mlp_d (outputs/nn_mlp)
Every fold seed found on disk is used: member prediction = mean over its fold seeds (each a 5-fold average),
then ranked. OOF of each member = mean of its per-seed OOF ranks. Zone 1 as in make_blend_submission.py.
Run: .venv/bin/python -m experiments.make_blend_submission_v3 <out name>
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.fraud.config import ID, OUTPUTS, RAW, SUBMISSION, TARGET
from src.fraud.metrics import auc

SEEDS = [42, 7, 2026, 11, 123, 777, 31337]
MEMBERS = {"ebm": (["round4"], 0.5), "lgb_l3_b1023_x3": (["round5"], 0.2),
           "xgb_d2_x3": (["models_v2", "round5"], 0.2), "mlp": (["nn_mlp"], 0.05), "mlp_d": (["nn_mlp"], 0.05)}
out_name = sys.argv[1]

train = pd.read_csv(RAW / "train.csv", usecols=["merchant_risk_score", TARGET])
test = pd.read_csv(RAW / "test.csv", usecols=[ID, "merchant_risk_score"])
sample = pd.read_csv(RAW / "sample_submission.csv")
assert (sample[ID].to_numpy() == test[ID].to_numpy()).all()
y = train[TARGET].astype(int).to_numpy()


def files(kind, name, dirs):
    out = {}
    for s in SEEDS:
        for d in dirs:
            f = OUTPUTS / d / f"{kind}_{name}_{s}.npy"
            if f.exists() and s not in out:
                out[s] = f
    return out


oof, pred = 0, 0
for name, (dirs, w) in MEMBERS.items():
    fo, ft = files("oof", name, dirs), files("test", name, dirs)
    assert fo.keys() == ft.keys(), name
    print(f"{name}: weight {w}, fold seeds {sorted(fo)}")
    oof = oof + w * np.mean([rankdata(np.load(f)) for f in fo.values()], axis=0)
    pred = pred + w * rankdata(np.mean([np.load(f) for f in ft.values()], axis=0))
t1 = train.loc[y == 1, "merchant_risk_score"].min()
oof = np.where(train["merchant_risk_score"].to_numpy() < t1, oof - 1e9, oof)
zone = test["merchant_risk_score"].to_numpy() < t1
pred = np.where(zone, pred - 1e9, pred)
final = (rankdata(pred) - 0.5) / len(pred)
sub = sample[[ID]].copy()
sub[TARGET] = final
sub.to_csv(SUBMISSION / f"{out_name}.csv", index=False)
print(f"OOF of the blend (fold-seed averaged, with zone 1): {auc(y, oof):.5f}")
print(f"zone 1 threshold {t1}; test rows in zone: {zone.sum():,}")
print(f"wrote submission/{out_name}.csv {sub.shape}")
