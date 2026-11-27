"""Confirm the feature screen: raw baseline vs "ei" (TE of numerics + drop no-signal raw columns).
LightGBM lr 0.05, 5-fold, fold seeds 42/7/2026, paired on the same folds.
Run: .venv/bin/python -m experiments.confirm_ei [set ...]   (default: raw ei; sets already in results are kept)
Writes outputs/confirm_ei/{results.csv, oof_<set>_<seed>.npy, test_<set>_<seed>.npy}
"""
import sys

import numpy as np
import pandas as pd

from src.fraud.config import OUTPUTS, TARGET
from src.fraud.cv import run_cv
from src.fraud.data import load
from src.fraud.features import align_categories, feature_set, model_frame

OUT = OUTPUTS / "confirm_ei"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 7, 2026]
SETS = ["raw", "ei"]
RUN = sys.argv[1:] or SETS

train, test, _ = load()
y = train[TARGET].to_numpy()
X, X_test = align_categories(model_frame(train), model_frame(test))

rows = []
for name in RUN:
    raw_cols, te_cols = feature_set(name, X.columns)
    for s in SEEDS:
        print(f"{name} fold_seed={s} ({len(raw_cols)} raw + {len(te_cols)} TE)", flush=True)
        r = run_cv(X, y, X_test, raw_cols, te_cols, fold_seed=s, log=lambda m: print(m, flush=True))
        np.save(OUT / f"oof_{name}_{s}.npy", r["oof"])
        np.save(OUT / f"test_{name}_{s}.npy", r["test"])
        rows.append(dict(set=name, fold_seed=s, auc=r["auc"], mean_trees=np.mean(r["iters"])))
        print(f"{name} fold_seed={s} AUC={r['auc']:.5f}", flush=True)

res = pd.DataFrame(rows)
if (OUT / "results.csv").exists():
    old = pd.read_csv(OUT / "results.csv")
    res = pd.concat([old[~old["set"].isin(RUN)], res], ignore_index=True)
res.to_csv(OUT / "results.csv", index=False)
wide = res.pivot(index="fold_seed", columns="set", values="auc")
if set(SETS) <= set(wide.columns):
    wide["delta_ei"] = wide["ei"] - wide["raw"]
print("\n", wide.round(5))
print("mean:", wide.mean().round(5).to_dict(), "| std over fold seeds:", wide.std().round(5).to_dict())
