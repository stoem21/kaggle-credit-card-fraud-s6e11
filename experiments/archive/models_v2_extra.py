"""More blend members on the "eivd" features, same folds / TE / early-stopping rows as experiments/models_v2.py:
- catboost: CatBoost (depth 6, lr 0.06, native categoricals)
- realmlp:  RealMLP_TD (pytabkit, tuned defaults); TE columns passed as logits
Run: .venv/bin/python -m experiments.models_v2_extra [model ...] [--quick] [--mps]
--mps: run RealMLP on the Apple GPU (default CPU).
Writes outputs/models_v2/{oof_<model>_<seed>.npy, test_<model>_<seed>.npy, results_extra.csv}
"""
import sys
import time

import numpy as np
import pandas as pd

from experiments.models_v2 import OUT, SEEDS, frames
from src.fraud.config import N_FOLDS
from src.fraud.cv import fold_frames
from src.fraud.data import CATEGORICAL
from src.fraud.features import feature_set
from src.fraud.metrics import auc

QUICK = "--quick" in sys.argv
DEVICE = "mps" if "--mps" in sys.argv else "cpu"
WHICH = [a for a in sys.argv[1:] if not a.startswith("--")] or ["catboost", "realmlp"]


def fit_catboost(Xf, yf, Xe, ye, seed):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=5000, learning_rate=0.06, depth=6, eval_metric="AUC", od_type="Iter",
                           od_wait=200, random_seed=seed, verbose=0, thread_count=-1, cat_features=CATEGORICAL)
    s = lambda d: d.assign(**{c: d[c].astype(str) for c in CATEGORICAL})
    m.fit(s(Xf), yf, eval_set=(s(Xe), ye), use_best_model=True)
    return lambda Z: m.predict_proba(s(Z))[:, 1], m.get_best_iteration()


def fit_realmlp(Xf, yf, Xe, ye, seed):
    from pytabkit import RealMLP_TD_Classifier
    te = [c for c in Xf.columns if c.startswith("te_")]
    lg = lambda d: d.assign(**{c: np.log(np.clip(d[c], 1e-5, 1 - 1e-5) / (1 - np.clip(d[c], 1e-5, 1 - 1e-5)))
                               for c in te})
    m = RealMLP_TD_Classifier(device=DEVICE, random_state=seed, val_metric_name="1-auc_ovr", verbosity=0)
    m.fit(lg(Xf), yf, X_val=lg(Xe), y_val=ye, cat_col_names=CATEGORICAL)
    return lambda Z: m.predict_proba(lg(Z))[:, 1], None


FIT = {"catboost": fit_catboost, "realmlp": fit_realmlp}

y, X, X_test, base = frames()
if QUICK:
    keep = np.random.RandomState(0).rand(len(y)) < 0.1
    X, y = X[keep].reset_index(drop=True), y[keep]
raw, te = feature_set("eivd", base)
rows = []
for name in WHICH:
    for s in (SEEDS[:1] if QUICK else SEEDS):
        oof, test_pred = np.zeros(len(y)), np.zeros(len(X_test))
        for k, tr, va, Xtr, Xva, Xte, fi, ei in fold_frames(X, y, X_test, raw, te, s):
            t0 = time.time()
            pred, it = FIT[name](Xtr.iloc[fi], y[tr][fi], Xtr.iloc[ei], y[tr][ei], seed=s)
            oof[va] = pred(Xva)
            test_pred += pred(Xte) / N_FOLDS
            print(f"  {name} seed {s} fold {k}: auc={auc(y[va], oof[va]):.5f} iters={it} {time.time() - t0:.0f}s",
                  flush=True)
            if QUICK:
                break
        if QUICK:
            continue
        np.save(OUT / f"oof_{name}_{s}.npy", oof)
        np.save(OUT / f"test_{name}_{s}.npy", test_pred)
        rows.append(dict(model=name, fold_seed=s, auc=auc(y, oof)))
        print(f"{name} fold_seed={s} AUC={rows[-1]['auc']:.5f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT / f"results_extra_{'_'.join(WHICH)}.csv", index=False)
