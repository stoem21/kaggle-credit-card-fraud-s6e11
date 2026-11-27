"""Round 7 (trees): eivd features + TE of the 42 numeric digit keys used by the plain GLM (src.fraud.features.k42)
and of the 3 exact-value numerics eivd leaves out. Same folds / early-stopping rows as every other model.
- lgb_l3_k42: LightGBM num_leaves 3, min_child_samples 100, lr 0.05 (as lgb_l3, which has no new keys)
- lgb_l7_k42 / lgb_l15_k42: num_leaves 7 / 15 (public stacks report LightGBM 15 leaves strong with value keys)
- lgb_l3_k42_x3: lgb_l3_k42 + bagging 0.8 / feature_fraction 0.7, 3 model seeds (0/1000/2000) averaged
  (as lgb_l3_b1023_x3 vs lgb_l3)
- cat_d2_k42: CatBoost depth 2, lr 0.1, l2 3 (public notebooks' setting), early stopping on the same 10% slice
Run: .venv/bin/python -m experiments.round7_trees [--only=cat_d2_k42] [--seeds=11,123] [--smoke]
Writes outputs/round7/{results_trees.csv, oof_<model>_<seed>.npy, test_<model>_<seed>.npy}
"""
import sys
import time

import numpy as np
import pandas as pd

from experiments.models_v2 import SEEDS, frames
from src.fraud.cli import check_argv, guard_outputs
from src.fraud.config import OUTPUTS
from src.fraud.cv import _fit_predict, fold_frames
from src.fraud.features import K42_COLS, feature_set, k42
from src.fraud.metrics import auc

check_argv(flags=["smoke"], options=["only", "seeds"])
SMOKE = "--smoke" in sys.argv
ONLY = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--only=")), None)
SEED_ARG = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--seeds=")), None)
OUT = OUTPUTS / "round7"
OUT.mkdir(parents=True, exist_ok=True)
L3 = dict(num_leaves=3, min_child_samples=100, learning_rate=0.05)
MODELS = ["lgb_l3_k42", "cat_d2_k42"]
LGB = {"lgb_l3_k42": L3, "lgb_l7_k42": dict(L3, num_leaves=7), "lgb_l15_k42": dict(L3, num_leaves=15),
       "lgb_l3_k42_x3": dict(L3, bagging_fraction=0.8, feature_fraction=0.7)}
BAGS = {"lgb_l3_k42_x3": 3}
CAT = ["cat_d2_k42"]
if ONLY:
    MODELS = ONLY
unknown = [m for m in MODELS if m not in LGB and m not in CAT]
if unknown:
    raise SystemExit(f"unknown model(s) {unknown}; known: {list(LGB) + CAT}")
y, X, X_test, base = frames()
raw, te = feature_set("eivd", base)
X, X_test = pd.concat([X, k42(X)], axis=1), pd.concat([X_test, k42(X_test)], axis=1)
extra_num = [c for c in ["prior_disputes", "card_age_months", "customer_age", "txn_count_last_24h", "cvv_retry_count",
                         "distance_from_home_km"] if c in X.columns and c not in te]
te = te + K42_COLS + extra_num
print(f"{len(raw)} raw, {len(te)} TE columns", flush=True)


def fit_cat(Xfit, yfit, Xes, yes, preds_on, seed):
    from catboost import CatBoostClassifier
    cats = [c for c in Xfit.columns if str(Xfit[c].dtype) == "category"]
    conv = lambda Z: Z.assign(**{c: Z[c].astype(str) for c in cats})
    m = CatBoostClassifier(iterations=20000, depth=2, learning_rate=0.1, l2_leaf_reg=3, eval_metric="AUC",
                           od_type="Iter", od_wait=300, random_seed=seed, thread_count=-1, verbose=False,
                           allow_writing_files=False, cat_features=cats)
    m.fit(conv(Xfit), yfit, eval_set=(conv(Xes), yes), use_best_model=True)
    return [m.predict_proba(conv(Z))[:, 1] for Z in preds_on], m.get_best_iteration()


rows = []
RUN_SEEDS = [42] if SMOKE else ([int(x) for x in SEED_ARG.split(",")] if SEED_ARG else SEEDS)
if not SMOKE:
    guard_outputs(OUT, MODELS, RUN_SEEDS)
for s in RUN_SEEDS:
    oof = {m: np.zeros(len(y)) for m in MODELS}
    tst = {m: np.zeros(len(X_test)) for m in MODELS}
    iters = {m: [] for m in MODELS}
    for k, tr, va, Xtr, Xva, Xte, fit_idx, es_idx in fold_frames(X, y, X_test, raw, te, fold_seed=s):
        ytr = y[tr]
        for name in MODELS:
            t0 = time.time()
            args = (Xtr.iloc[fit_idx], ytr[fit_idx], Xtr.iloc[es_idx], ytr[es_idx], [Xva, Xte])
            if name in LGB:
                nb = BAGS.get(name, 1)
                pv, pt, it = 0, 0, 0
                for b in range(nb):  # model seed 1000*b, as run_cv_multi
                    (bv, bt), bi = _fit_predict("lgb", LGB[name], 1000 * b, *args, max_rounds=20000, es_rounds=300)
                    pv, pt, it = pv + bv / nb, pt + bt / nb, it + bi / nb
            elif name in CAT:
                (pv, pt), it = fit_cat(*args, seed=0)
            oof[name][va] = pv; tst[name] += pt / 5; iters[name].append(it)
            print(f"  seed {s} fold {k} {name}: auc={auc(y[va], pv):.5f} iters={it} {time.time() - t0:.0f}s",
                  flush=True)
        if SMOKE:
            sys.exit(0)
    for name in MODELS:
        np.save(OUT / f"oof_{name}_{s}.npy", oof[name]); np.save(OUT / f"test_{name}_{s}.npy", tst[name])
        rows.append(dict(model=name, fold_seed=s, auc=auc(y, oof[name]), mean_iters=np.mean(iters[name])))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / ("results_trees" + ("_" + "_".join(ONLY) if ONLY else "")
                                     + ("_extra_" + SEED_ARG.replace(",", "_") if SEED_ARG else "") + ".csv"), index=False)
