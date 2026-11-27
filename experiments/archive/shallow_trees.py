"""How shallow is best? LightGBM num_leaves 3/4/15 and XGBoost depth 2/3 on eivd, same folds/TE/ES rows as
experiments/models_v2.py (whose lgb_shallow = num_leaves 7 is the reference).
Run: .venv/bin/python -m experiments.shallow_trees  -> outputs/models_v2/{oof,test}_<model>_<seed>.npy
"""
import numpy as np
import pandas as pd

from experiments.models_v2 import OUT, SEEDS, frames
from src.fraud.cv import run_cv_multi
from src.fraud.features import feature_set

MODELS = {
    "lgb_l3": ("lgb", dict(num_leaves=3, min_child_samples=100, learning_rate=0.05), 1),
    "lgb_l4": ("lgb", dict(num_leaves=4, min_child_samples=100, learning_rate=0.05), 1),
    "lgb_l15": ("lgb", dict(num_leaves=15, min_child_samples=100, learning_rate=0.05), 1),
    "xgb_d2": ("xgb", dict(eta=0.05, max_depth=2, subsample=0.8, colsample_bytree=0.7, min_child_weight=5), 1),
    "xgb_d3": ("xgb", dict(eta=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.7, min_child_weight=5), 1),
}
y, X, X_test, base = frames()
raw, te = feature_set("eivd", base)
rows = []
for s in SEEDS:
    res = run_cv_multi(X, y, X_test, raw, te, fold_seed=s, models=MODELS, log=lambda m: None)
    for name, r in res.items():
        np.save(OUT / f"oof_{name}_{s}.npy", r["oof"]); np.save(OUT / f"test_{name}_{s}.npy", r["test"])
        rows.append(dict(model=name, fold_seed=s, auc=r["auc"], mean_trees=np.mean(r["iters"])))
        print(f"{name} fold_seed={s} AUC={r['auc']:.5f} trees={np.mean(r['iters']):.0f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "results_shallow.csv", index=False)
ref = pd.read_csv(OUT / "results.csv")
res = pd.concat([ref[ref.model == "lgb_shallow"], pd.DataFrame(rows)])
print(res.pivot(index="fold_seed", columns="model", values="auc").round(5))
