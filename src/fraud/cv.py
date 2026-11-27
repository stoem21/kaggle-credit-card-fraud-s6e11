"""Cross-validation used by every model: stratified K-fold, in-fold target encoding, honest early stopping.

Early stopping uses a 10% stratified slice of the TRAINING fold, not the validation fold, so out-of-fold AUC is
not inflated by choosing the number of trees on the rows being scored.
"""
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import N_FOLDS
from .features import target_encode, te_keys
from .metrics import auc

LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, n_jobs=-1, seed=0)


def run_cv(X, y, X_test, raw_cols, te_cols, fold_seed, params=None, max_rounds=5000, es_rounds=200,
           smoothing=20, log=print, extra_X=None, extra_y=None):
    """Returns dict(auc, oof, test, iters). X/X_test: model frames (see features.model_frame).

    extra_X / extra_y: optional rows (e.g. the original dataset) appended to every TRAINING fold only.
    They are never scored, so out-of-fold AUC stays on competition rows.
    """
    import lightgbm as lgb
    p = dict(LGB_PARAMS, **(params or {}))
    y = np.asarray(y)
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_test))
    iters = []
    folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=fold_seed)
    for k, (tr, va) in enumerate(folds.split(X, y)):
        t0 = time.time()
        Xtr_full, ytr = X.iloc[tr], y[tr]
        if extra_X is not None:
            Xtr_full = pd.concat([Xtr_full, extra_X], ignore_index=True)
            ytr = np.r_[ytr, np.asarray(extra_y)]
        Xtr, Xva, Xte = Xtr_full[raw_cols].copy(), X.iloc[va][raw_cols].copy(), X_test[raw_cols].copy()
        for c in te_cols:
            a, (b, t) = target_encode(te_keys(Xtr_full, c), ytr, [te_keys(X.iloc[va], c), te_keys(X_test, c)],
                                      smoothing=smoothing, seed=fold_seed)
            Xtr["te_" + c], Xva["te_" + c], Xte["te_" + c] = a, b, t
        fit_idx, es_idx = train_test_split(np.arange(len(ytr)), test_size=0.1, stratify=ytr,
                                           random_state=fold_seed)
        dfit = lgb.Dataset(Xtr.iloc[fit_idx], ytr[fit_idx])
        des = lgb.Dataset(Xtr.iloc[es_idx], ytr[es_idx], reference=dfit)
        m = lgb.train(p, dfit, max_rounds, valid_sets=[des],
                      callbacks=[lgb.early_stopping(es_rounds, verbose=False)])
        oof[va] = m.predict(Xva, num_iteration=m.best_iteration)
        test_pred += m.predict(Xte, num_iteration=m.best_iteration) / N_FOLDS
        iters.append(m.best_iteration)
        log(f"  fold {k}: auc={auc(y[va], oof[va]):.5f} trees={m.best_iteration} {time.time() - t0:.0f}s")
    return dict(auc=auc(y, oof), oof=oof, test=test_pred, iters=iters)


def _fit_predict(kind, params, seed, Xfit, yfit, Xes, yes, preds_on, max_rounds, es_rounds, wfit=None):
    """wfit: optional sample weights for the fit rows (early-stopping rows stay unweighted)."""
    if kind == "lgb":
        import lightgbm as lgb  # imported lazily: loading it next to torch crashes on macOS
        p = dict(LGB_PARAMS, **params, seed=seed)
        dfit = lgb.Dataset(Xfit, yfit, weight=wfit)
        m = lgb.train(p, dfit, max_rounds, valid_sets=[lgb.Dataset(Xes, yes, reference=dfit)],
                      callbacks=[lgb.early_stopping(es_rounds, verbose=False)])
        return [m.predict(Z, num_iteration=m.best_iteration) for Z in preds_on], m.best_iteration
    if kind == "xgb":
        import xgboost as xgb
        p = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", nthread=-1, seed=seed)
        p.update(params)
        dfit = xgb.DMatrix(Xfit, yfit, weight=wfit, enable_categorical=True)
        des = xgb.DMatrix(Xes, yes, enable_categorical=True)
        m = xgb.train(p, dfit, max_rounds, evals=[(des, "es")], early_stopping_rounds=es_rounds, verbose_eval=False)
        rng = (0, m.best_iteration + 1)
        return [m.predict(xgb.DMatrix(Z, enable_categorical=True), iteration_range=rng) for Z in preds_on], \
            m.best_iteration
    raise ValueError(kind)


def run_cv_multi(X, y, X_test, raw_cols, te_cols, fold_seed, models, max_rounds=20000, es_rounds=300,
                 smoothing=20, log=print):
    """Like run_cv, but the fold's target encoding is computed once and shared by several models.

    models: {name: (kind "lgb"|"xgb", params dict, n_bags)}. Each bag uses a different model seed
    (seed = 1000*bag); the bag predictions are averaged. Returns {name: dict(auc, oof, test, iters,
    oof_bag0, test_bag0, auc_bag0)} -- bag 0 alone is the single-model result for the same settings.
    Early stopping: same 10% slice of the training fold for every model and bag.
    """
    y = np.asarray(y)
    out = {n: dict(oof=np.zeros(len(y)), test=np.zeros(len(X_test)), iters=[],
                   oof_bag0=np.zeros(len(y)), test_bag0=np.zeros(len(X_test))) for n in models}
    folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=fold_seed)
    for k, (tr, va) in enumerate(folds.split(X, y)):
        Xtr, Xva, Xte = X.iloc[tr][raw_cols].copy(), X.iloc[va][raw_cols].copy(), X_test[raw_cols].copy()
        for c in te_cols:
            a, (b, t) = target_encode(te_keys(X.iloc[tr], c), y[tr], [te_keys(X.iloc[va], c), te_keys(X_test, c)],
                                      smoothing=smoothing, seed=fold_seed)
            Xtr["te_" + c], Xva["te_" + c], Xte["te_" + c] = a, b, t
        fit_idx, es_idx = train_test_split(np.arange(len(tr)), test_size=0.1, stratify=y[tr],
                                           random_state=fold_seed)
        Xfit, yfit, Xes, yes = Xtr.iloc[fit_idx], y[tr][fit_idx], Xtr.iloc[es_idx], y[tr][es_idx]
        for name, (kind, params, n_bags) in models.items():
            t0 = time.time()
            pv, pt = np.zeros(len(va)), np.zeros(len(X_test))
            for b in range(n_bags):
                (v, t), it = _fit_predict(kind, params, 1000 * b, Xfit, yfit, Xes, yes, [Xva, Xte],
                                          max_rounds, es_rounds)
                pv += v / n_bags
                pt += t / n_bags
                out[name]["iters"].append(it)
                if b == 0:
                    out[name]["oof_bag0"][va] = v
                    out[name]["test_bag0"] += t / N_FOLDS
            out[name]["oof"][va] = pv
            out[name]["test"] += pt / N_FOLDS
            log(f"  fold {k} {name}: auc={auc(y[va], pv):.5f} trees={out[name]['iters'][-n_bags:]} "
                f"{time.time() - t0:.0f}s")
    for r in out.values():
        r["auc"] = auc(y, r["oof"])
        r["auc_bag0"] = auc(y, r["oof_bag0"])
    return out


def fold_frames(X, y, X_test, raw_cols, te_cols, fold_seed, smoothing=20):
    """Yield the per-fold model inputs used by every model: (k, tr, va, Xtr, Xva, Xte, fit_idx, es_idx).
    TE fitted on the training fold only; fit_idx / es_idx split the training fold (10% for early stopping),
    identical to run_cv_multi, so all models share folds and early-stopping rows."""
    y = np.asarray(y)
    for k, (tr, va) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=fold_seed).split(X, y)):
        Xtr, Xva, Xte = X.iloc[tr][raw_cols].copy(), X.iloc[va][raw_cols].copy(), X_test[raw_cols].copy()
        for c in te_cols:
            a, (b, t) = target_encode(te_keys(X.iloc[tr], c), y[tr], [te_keys(X.iloc[va], c), te_keys(X_test, c)],
                                      smoothing=smoothing, seed=fold_seed)
            Xtr["te_" + c], Xva["te_" + c], Xte["te_" + c] = a, b, t
        fit_idx, es_idx = train_test_split(np.arange(len(tr)), test_size=0.1, stratify=y[tr],
                                           random_state=fold_seed)
        yield k, tr, va, Xtr, Xva, Xte, fit_idx, es_idx
