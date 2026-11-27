"""Round 5 (trees), eivd features, same folds / TE / early-stopping rows as experiments/models_v2.py.
- lgb_l3_b1023_x3: num_leaves 3, max_bin 1023, 3 model seeds (bagging 0.8, feature_fraction 0.7, as lgb_l3_x3)
- lgb_l3_b1023_tx: num_leaves 3, max_bin 1023 + TE of txn_count_last_24h (reference: lgb_l3_b1023 in round4)
- xgb_d2_x3, lgb_l3_x3: as in round3 (default max_bin; only run for extra fold seeds)
Run: .venv/bin/python -m experiments.round5_trees [--seeds=11,123] [--only=name,name]
Writes outputs/round5/{results_<tag>.csv, oof_<model>_<seed>.npy, test_<model>_<seed>.npy}
"""
import sys

import numpy as np
import pandas as pd

from experiments.models_v2 import frames
from src.fraud.cli import check_argv, guard_outputs
from src.fraud.config import FOLD_SEEDS, OUTPUTS
from src.fraud.cv import run_cv_multi
from src.fraud.features import feature_set

check_argv(options=["seeds", "only"])
OUT = OUTPUTS / "round5"
OUT.mkdir(parents=True, exist_ok=True)
arg = lambda k: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), None)
SEEDS = [int(x) for x in arg("seeds").split(",")] if arg("seeds") else FOLD_SEEDS
ONLY = arg("only").split(",") if arg("only") else None
L3 = dict(num_leaves=3, min_child_samples=100, learning_rate=0.05, max_bin=1023)
D2 = dict(eta=0.05, max_depth=2, subsample=0.8, colsample_bytree=0.7, min_child_weight=5)
MODELS = {"lgb_l3_b1023_x3": ("lgb", dict(L3, bagging_fraction=0.8, feature_fraction=0.7), 3, False),
          "lgb_l3_b1023_tx": ("lgb", L3, 1, True),
          "xgb_d2_x3": ("xgb", D2, 3, False),
          "lgb_l3_x3": ("lgb", dict(L3, max_bin=255, bagging_fraction=0.8, feature_fraction=0.7), 3, False)}
MODELS = {k: v for k, v in MODELS.items() if (ONLY is None or k in ONLY)}
if ONLY and set(ONLY) - set(MODELS):
    raise SystemExit(f"unknown model(s) {sorted(set(ONLY) - set(MODELS))}")
tag = "_".join(MODELS) + "_" + "_".join(map(str, SEEDS))
guard_outputs(OUT, MODELS, SEEDS)

y, X, X_test, base = frames()
raw, te = feature_set("eivd", base)
rows = []
for s in SEEDS:
    for with_tx in (False, True):
        models = {k: v[:3] for k, v in MODELS.items() if v[3] == with_tx}
        if not models:
            continue
        te_cols = te + (["txn_count_last_24h"] if with_tx else [])
        res = run_cv_multi(X, y, X_test, raw, te_cols, fold_seed=s, models=models, log=lambda m: None)
        for name, r in res.items():
            np.save(OUT / f"oof_{name}_{s}.npy", r["oof"]); np.save(OUT / f"test_{name}_{s}.npy", r["test"])
            rows.append(dict(model=name, fold_seed=s, auc=r["auc"], mean_trees=np.mean(r["iters"])))
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(OUT / f"results_{tag}.csv", index=False)
