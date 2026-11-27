"""Evaluate round-4 models (saved OOF only, no training). Per fold seed 42/7/2026:
1. alone: AUC overall / inside hard rows (0-1 strong signals) / inside quiet rows (0 signals), vs references
2. in the blend: base = current best blend (x3_both + zone 1). Global models are added with weight w to the
   rank blend. Specialists (seg_*) only change the ORDER INSIDE hard rows: hard rows keep the set of rank
   positions they have in the base blend and are re-sorted by (base rank + w * specialist rank) within hard.
Run: .venv/bin/python -m experiments.round4_eval
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata as R

from src.fraud.config import OUTPUTS
from src.fraud.data import load
from src.fraud.features import strong_signals
from src.fraud.metrics import auc

S = [42, 7, 2026]
train = load()[0]
y = train.is_fraud.to_numpy()
sig = strong_signals(train)
hard, quiet = sig <= 1, sig == 0
z = train.merchant_risk_score.to_numpy() < train.merchant_risk_score[y == 1].min()
ld = lambda d, n, s: np.load(OUTPUTS / d / f"oof_{n}_{s}.npy")
BASE = {("models_v2", "lgb_l3_x3"): .4, ("models_v2", "xgb_d2_x3"): .4, ("nn_mlp", "mlp"): .1, ("nn_mlp", "mlp_d"): .1}
REF = {"lgb_l3_b1023": ("models_v2", "lgb_l3"), "lgb_l3_w3": ("models_v2", "lgb_l3"),
       "lgb_l3_spw5": ("models_v2", "lgb_l3"), "xgb_d2_b1024": ("models_v2", "xgb_d2"),
       "seg_lgb_l3": ("models_v2", "lgb_l3"), "seg_lgb_l3_b1023": ("models_v2", "lgb_l3"),
       "seg_xgb_d2_b1024": ("models_v2", "xgb_d2"), "mlp_d_ple": ("nn_mlp", "mlp_d"),
       "mlp_d_seg": ("nn_mlp", "mlp_d"), "mlp_d_ple_seg": ("nn_mlp", "mlp_d"),
       "lr_spline": ("models_v2", "lgb_l3"), "ebm": ("models_v2", "lgb_l3")}


def base(s):
    p = sum(w * R(ld(d, n, s)) for (d, n), w in BASE.items())
    return R(np.where(z, p - 1e9, p))


def rerank_hard(b, m, w):
    out = b.copy()
    pos = np.sort(b[hard])
    order = np.argsort(R(b[hard]) + w * R(m[hard]), kind="stable")
    h = np.empty(len(order))
    h[order] = pos  # lowest combined score gets the lowest of the hard rows' positions
    out[hard] = h
    return out


rows = []
for name, (rd, rn) in REF.items():
    if not (OUTPUTS / "round4" / f"oof_{name}_2026.npy").exists():
        continue
    for s in S:
        m, r, b = ld("round4", name, s), ld(rd, rn, s), base(s)
        row = dict(model=name, ref=rn, seed=s)
        for g, mask in [("all", slice(None)), ("hard", hard), ("quiet", quiet)]:
            row[f"d_{g}"] = auc(y[mask], m[mask]) - auc(y[mask], r[mask])
        row["auc_hard"] = auc(y[hard], m[hard])
        b0 = auc(y, b)
        for w in (0.1, 0.25, 0.5):
            if name.startswith("seg") or name.endswith("_seg"):
                nb = rerank_hard(b, m, w)
            else:
                nb = R(np.where(z, b + w * R(m) - 1e9, b + w * R(m)))
            row[f"blend_w{w}"] = auc(y, nb) - b0
        rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(OUTPUTS / "round4" / "eval.csv", index=False)
pd.set_option("display.width", 220)
fmt = {c: "{:+.5f}".format for c in out.columns if c.startswith(("d_", "blend"))}
fmt["auc_hard"] = "{:.5f}".format
print(out.to_string(index=False, formatters=fmt))
print("\nmean over seeds:")
print(out.drop(columns=["ref", "seed"]).groupby("model", sort=False).mean().to_string(formatters=fmt))
