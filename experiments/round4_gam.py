"""Round 4 (additive models): the target is close to additive in log-odds, so try models built for that.
eivd inputs from the same folds / in-fold TE as the tree models (src.fraud.cv.fold_frames).
- lr_spline: logistic regression on cubic splines of the numerics + one-hot categoricals + flags
             + standardized logit TE columns (a GAM with fixed smoothness)
- ebm_b4096 / ebm_leaf50 / ebm_ob8: one EBM setting changed (see EBM_VARIANTS)
- ebm:       Explainable Boosting Machine (interpret), max_bins 1024, 10 pairwise interactions, 4 outer bags;
             its early stopping uses its own split of the training fold.
Run: .venv/bin/python -m experiments.round4_gam lr_spline|ebm [--smoke] [--seeds=11,123]
--smoke: fold seed 42, fold 0 only, prints the time (to size the full run).
Writes outputs/round4/{results_<model>.csv, oof_<model>_<seed>.npy, test_<model>_<seed>.npy}
"""
import sys
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from experiments.models_v2 import SEEDS, frames
from src.fraud.cli import check_argv, guard_outputs
from src.fraud.config import OUTPUTS
from src.fraud.cv import fold_frames
from src.fraud.data import CATEGORICAL, load
from src.fraud.features import feature_set, strong_signals
from src.fraud.metrics import auc

check_argv(flags=["smoke"], options=["seeds"], n_positional=1)
MODEL = sys.argv[1]
SMOKE = "--smoke" in sys.argv
SEED_ARG = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--seeds=")), None)
OUT = OUTPUTS / "round4"
OUT.mkdir(parents=True, exist_ok=True)

y, X, X_test, base = frames()
raw, te = feature_set("eivd", base)
if MODEL.endswith("_k42"):  # + TE of the 42 numeric digit keys (src.fraud.features.k42), as round7 trees
    from src.fraud.features import K42_COLS, k42
    X, X_test = pd.concat([X, k42(X)], axis=1), pd.concat([X_test, k42(X_test)], axis=1)
    te = te + K42_COLS
sig = strong_signals(load()[0])
hard = sig <= 1
CATS = [c for c in raw if c in CATEGORICAL]
NUMS = [c for c in raw if c not in CATS and X[c].nunique() > 2]
FLAGS = [c for c in raw if c not in CATS + NUMS]
TECOLS = ["te_" + c for c in te]


def logit_te(Z):
    Z = Z.copy()
    p = np.clip(Z[TECOLS].to_numpy(float), 1e-5, 1 - 1e-5)
    Z[TECOLS] = np.log(p / (1 - p))
    return Z


EBM_BASE = dict(max_bins=1024, interactions=10, outer_bags=4, n_jobs=-1, random_state=0)
EBM_VARIANTS = {"ebm": {},
                "ebm_b4096": dict(max_bins=4096),          # finer bins (amount has 13k values)
                "ebm_leaf50": dict(min_samples_leaf=50),   # smoother curves (default 4)
                "ebm_ob8": dict(outer_bags=8),             # more internal bagging (less variance)
                "ebm_k42": {}}                             # ebm + TE of the 42 digit keys


def make_model():
    if MODEL == "lr_spline":
        ct = ColumnTransformer([
            ("spl", make_pipeline(StandardScaler(), SplineTransformer(n_knots=12, degree=3, knots="quantile",
                                                                    extrapolation="constant")), NUMS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATS),
            ("flag", "passthrough", FLAGS),
            ("te", StandardScaler(), TECOLS)])
        return make_pipeline(ct, LogisticRegression(C=1.0, max_iter=1000))
    if MODEL.startswith("ebm"):
        from interpret.glassbox import ExplainableBoostingClassifier
        return ExplainableBoostingClassifier(**dict(EBM_BASE, **EBM_VARIANTS[MODEL]))
    raise ValueError(MODEL)


rows = []
RUN_SEEDS = [42] if SMOKE else ([int(x) for x in SEED_ARG.split(",")] if SEED_ARG else SEEDS)
RESULTS = f"results_{MODEL}" + ("_" + "_".join(map(str, RUN_SEEDS)) if SEED_ARG else "") + ".csv"
if not SMOKE:
    guard_outputs(OUT, [MODEL], RUN_SEEDS)
for s in RUN_SEEDS:
    oof, tst = np.zeros(len(y)), np.zeros(len(X_test))
    for k, tr, va, Xtr, Xva, Xte, fit_idx, es_idx in fold_frames(X, y, X_test, raw, te, fold_seed=s):
        t0 = time.time()
        Xtr, Xva, Xte = logit_te(Xtr), logit_te(Xva), logit_te(Xte)
        if MODEL.startswith("ebm"):  # EBM wants plain columns
            for Z in (Xtr, Xva, Xte):
                Z[CATS] = Z[CATS].astype(str)
        m = make_model().fit(Xtr, y[tr])  # no early-stopping rows needed (LR) / EBM splits its own
        oof[va] = m.predict_proba(Xva)[:, 1]
        tst += m.predict_proba(Xte)[:, 1] / 5
        mh = hard[va]
        print(f"  seed {s} fold {k} {MODEL}: auc={auc(y[va], oof[va]):.5f} auc_hard={auc(y[va][mh], oof[va][mh]):.5f} "
              f"{time.time() - t0:.0f}s", flush=True)
        if SMOKE:
            sys.exit(0)
    np.save(OUT / f"oof_{MODEL}_{s}.npy", oof); np.save(OUT / f"test_{MODEL}_{s}.npy", tst)
    rows.append(dict(model=MODEL, fold_seed=s, auc=auc(y, oof), auc_hard=auc(y[hard], oof[hard]),
                     auc_quiet=auc(y[sig == 0], oof[sig == 0])))
    print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / RESULTS, index=False)
