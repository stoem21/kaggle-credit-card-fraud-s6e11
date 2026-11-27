"""Round 6 (trees): how finely shallow LightGBM reacts to values. eivd features, same folds as models_v2.
Reference: lgb_l3 (outputs/models_v2: num_leaves 3, min_child_samples 100, lr 0.05, one model seed).
- lgb_l3_mcs30 / lgb_l3_mcs300: min_child_samples 30 / 300
- lgb_l3_lr02: learning rate 0.02
Run: .venv/bin/python -m experiments.round6_trees
"""
import numpy as np
import pandas as pd

from experiments.models_v2 import SEEDS, frames
from src.fraud.config import OUTPUTS
from src.fraud.cv import run_cv_multi
from src.fraud.features import feature_set

OUT = OUTPUTS / "round6"
OUT.mkdir(parents=True, exist_ok=True)
L3 = dict(num_leaves=3, min_child_samples=100, learning_rate=0.05)
MODELS = {"lgb_l3_mcs30": ("lgb", dict(L3, min_child_samples=30), 1),
          "lgb_l3_mcs300": ("lgb", dict(L3, min_child_samples=300), 1),
          "lgb_l3_lr02": ("lgb", dict(L3, learning_rate=0.02), 1)}
y, X, X_test, base = frames()
raw, te = feature_set("eivd", base)
rows = []
for s in SEEDS:
    res = run_cv_multi(X, y, X_test, raw, te, fold_seed=s, models=MODELS, max_rounds=40000, log=lambda m: None)
    for name, r in res.items():
        np.save(OUT / f"oof_{name}_{s}.npy", r["oof"]); np.save(OUT / f"test_{name}_{s}.npy", r["test"])
        rows.append(dict(model=name, fold_seed=s, auc=r["auc"], mean_trees=np.mean(r["iters"])))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)
