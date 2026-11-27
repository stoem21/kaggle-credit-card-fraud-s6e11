"""Round 4 (tree side): can trees see finer differences inside the hard (0-1 strong signal) rows?
eivd features, fold seeds 42/7/2026, same folds / early-stopping slice as experiments/models_v2.py.
Reference models: lgb_l3, xgb_d2 (outputs/models_v2).
All rows, global in-fold TE:
- lgb_l3_b1023: max_bin 1023 (default 255 merges values: merchant_risk has 950, velocity 666)
- xgb_d2_b1024: max_bin 1024
- lgb_l3_w3:    fit rows with 0 strong signals get weight 3
- lgb_l3_spw5:  scale_pos_weight 5 (class weight)
Specialists (seg_*): trained only on training-fold rows with 0-1 strong signals, TE fitted on those rows only,
early stopping on the in-group part of the same 10% slice. Their OOF is only meaningful for 0-1 signal rows
(other rows get a prediction too, but it is not used).
- seg_lgb_l3, seg_lgb_l3_b1023, seg_xgb_d2_b1024
Run: .venv/bin/python -m experiments.round4_trees
Writes outputs/round4/{results_trees.csv, oof_<model>_<seed>.npy, test_<model>_<seed>.npy}
"""
import time

import numpy as np
import pandas as pd

from experiments.models_v2 import SEEDS, frames
from src.fraud.config import OUTPUTS
from src.fraud.cv import _fit_predict, fold_frames
from src.fraud.data import load
from src.fraud.features import feature_set, strong_signals, target_encode, te_keys
from src.fraud.metrics import auc

OUT = OUTPUTS / "round4"
OUT.mkdir(parents=True, exist_ok=True)
L3 = dict(num_leaves=3, min_child_samples=100, learning_rate=0.05)
D2 = dict(eta=0.05, max_depth=2, subsample=0.8, colsample_bytree=0.7, min_child_weight=5)
GLOBAL = {
    "lgb_l3_b1023": ("lgb", dict(L3, max_bin=1023), None),
    "xgb_d2_b1024": ("xgb", dict(D2, max_bin=1024), None),
    "lgb_l3_w3": ("lgb", L3, "w3"),
    "lgb_l3_spw5": ("lgb", dict(L3, scale_pos_weight=5.0), None),
}
SEG = {
    "seg_lgb_l3": ("lgb", L3),
    "seg_lgb_l3_b1023": ("lgb", dict(L3, max_bin=1023)),
    "seg_xgb_d2_b1024": ("xgb", dict(D2, max_bin=1024)),
}

y, X, X_test, base = frames()
raw, te = feature_set("eivd", base)
train, test, _ = load()
sig = strong_signals(train)
hard = sig <= 1
print(f"hard rows (0-1 signals): {hard.mean():.1%}", flush=True)

rows = []
for s in SEEDS:
    oof = {n: np.zeros(len(y)) for n in list(GLOBAL) + list(SEG)}
    tst = {n: np.zeros(len(X_test)) for n in oof}
    for k, tr, va, Xtr, Xva, Xte, fit_idx, es_idx in fold_frames(X, y, X_test, raw, te, fold_seed=s):
        ytr = y[tr]
        Xfit, yfit, Xes, yes = Xtr.iloc[fit_idx], ytr[fit_idx], Xtr.iloc[es_idx], ytr[es_idx]
        for name, (kind, params, w) in GLOBAL.items():
            t0 = time.time()
            wfit = np.where(sig[tr][fit_idx] == 0, 3.0, 1.0) if w == "w3" else None
            (v, t), it = _fit_predict(kind, params, 0, Xfit, yfit, Xes, yes, [Xva, Xte], 20000, 300, wfit=wfit)
            oof[name][va], tst[name] = v, tst[name] + t / 5
            print(f"  seed {s} fold {k} {name}: auc={auc(y[va], v):.5f} trees={it} {time.time() - t0:.0f}s", flush=True)
        # specialists: TE refitted on the hard rows of the training fold only
        g_tr = tr[hard[tr]]
        Str, Sva, Ste = X.iloc[g_tr][raw].copy(), X.iloc[va][raw].copy(), X_test[raw].copy()
        for c in te:
            a, (b, t) = target_encode(te_keys(X.iloc[g_tr], c), y[g_tr], [te_keys(X.iloc[va], c), te_keys(X_test, c)],
                                      smoothing=20, seed=s)
            Str["te_" + c], Sva["te_" + c], Ste["te_" + c] = a, b, t
        pos = {r: i for i, r in enumerate(g_tr)}
        f_g = np.array([pos[r] for r in tr[fit_idx] if hard[r]])
        e_g = np.array([pos[r] for r in tr[es_idx] if hard[r]])
        for name, (kind, params) in SEG.items():
            t0 = time.time()
            (v, t), it = _fit_predict(kind, params, 0, Str.iloc[f_g], y[g_tr][f_g], Str.iloc[e_g], y[g_tr][e_g],
                                      [Sva, Ste], 20000, 300)
            oof[name][va], tst[name] = v, tst[name] + t / 5
            m = hard[va]
            print(f"  seed {s} fold {k} {name}: auc_hard={auc(y[va][m], v[m]):.5f} trees={it} "
                  f"{time.time() - t0:.0f}s", flush=True)
    for name in oof:
        np.save(OUT / f"oof_{name}_{s}.npy", oof[name]); np.save(OUT / f"test_{name}_{s}.npy", tst[name])
        rows.append(dict(model=name, fold_seed=s, auc=auc(y, oof[name]), auc_hard=auc(y[hard], oof[name][hard]),
                         auc_quiet=auc(y[sig == 0], oof[name][sig == 0])))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "results_trees.csv", index=False)
print(pd.DataFrame(rows).round(5).to_string(index=False))
