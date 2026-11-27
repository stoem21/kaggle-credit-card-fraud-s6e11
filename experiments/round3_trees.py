"""Round 3 (tree side), eivd features unless noted, same folds/TE/ES rows as experiments/models_v2.py.
Depth / bagging:
- lgb_l2: num_leaves 2 (pure additive model)    - xgb_d1: depth 1 (pure additive model)
- lgb_l3_x3: num_leaves 3, 3 model seeds         - xgb_d2_x3: depth 2, 3 model seeds
More digit keys (with lgb num_leaves 3, reference = lgb_l3 in outputs/models_v2):
- lgb_l3_dmv: + TE of last digit and first digit of merchant_risk_score and velocity_score
- lgb_l3_d3:  + TE of the last three characters ("d.dd") of the four 2-decimal floats
Run: .venv/bin/python -m experiments.round3_trees
"""
import numpy as np
import pandas as pd

from experiments.models_v2 import OUT, SEEDS, frames
from src.fraud.config import RAW
from src.fraud.cv import run_cv_multi
from src.fraud.features import DIGIT_FLOATS, feature_set

L3 = dict(num_leaves=3, min_child_samples=100, learning_rate=0.05)
D2 = dict(eta=0.05, max_depth=2, subsample=0.8, colsample_bytree=0.7, min_child_weight=5)
DEPTH = {
    "lgb_l2": ("lgb", dict(num_leaves=2, min_child_samples=100, learning_rate=0.05), 1),
    "xgb_d1": ("xgb", dict(D2, max_depth=1), 1),
    "lgb_l3_x3": ("lgb", dict(L3, bagging_fraction=0.8, feature_fraction=0.7), 3),
    "xgb_d2_x3": ("xgb", D2, 3),
}

y, X, X_test, base = frames()
for frame, f in [(X, "train.csv"), (X_test, "test.csv")]:
    txt = pd.read_csv(RAW / f, dtype=str, usecols=DIGIT_FLOATS + ["merchant_risk_score", "velocity_score"])
    for c in ["merchant_risk_score", "velocity_score"]:
        frame[f"{c}__last"] = txt[c].str[-1].to_numpy()
        frame[f"{c}__first"] = txt[c].str[0].to_numpy()
    for c in DIGIT_FLOATS:
        frame[f"{c}__tail3"] = txt[c].str[-3:].to_numpy()
raw, te = feature_set("eivd", base)
VARIANTS = [
    ("depth", te, DEPTH),
    ("dmv", te + [f"{c}__{k}" for c in ["merchant_risk_score", "velocity_score"] for k in ["last", "first"]],
     {"lgb_l3_dmv": ("lgb", L3, 1)}),
    ("d3", te + [f"{c}__tail3" for c in DIGIT_FLOATS], {"lgb_l3_d3": ("lgb", L3, 1)}),
]
rows = []
for s in SEEDS:
    for tag, te_cols, models in VARIANTS:
        res = run_cv_multi(X, y, X_test, raw, te_cols, fold_seed=s, models=models, log=lambda m: None)
        for name, r in res.items():
            np.save(OUT / f"oof_{name}_{s}.npy", r["oof"]); np.save(OUT / f"test_{name}_{s}.npy", r["test"])
            rows.append(dict(model=name, fold_seed=s, auc=r["auc"], mean_trees=np.mean(r["iters"])))
            print(f"{name} fold_seed={s} AUC={r['auc']:.5f} trees={np.mean(r['iters']):.0f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "results_round3.csv", index=False)
ref = pd.concat([pd.read_csv(OUT / "results_shallow.csv"), pd.DataFrame(rows)])
print(ref.pivot(index="fold_seed", columns="model", values="auc").round(5).to_string())
