"""Models on the "eivd" features (ei + vel_ex + TE of 12 digit keys), fold seeds 42/7/2026, shared in-fold TE:
- lgb_reg_x3: regularized LightGBM (as in variance_models), 3 model seeds
- lgb_shallow: shallow LightGBM (num_leaves 7, min_child_samples 100, lr 0.05) -- public notebooks report
               shallow trees beat deep ones on this data
- xgb: XGBoost hist depth 6 (as in variance_models)
Run: .venv/bin/python -m experiments.models_v2
Writes outputs/models_v2/{results.csv, oof_<model>_<seed>.npy, test_<model>_<seed>.npy}
"""
import numpy as np
import pandas as pd

from src.fraud.config import OUTPUTS, RAW, TARGET
from src.fraud.cv import run_cv_multi
from src.fraud.data import load
from src.fraud.features import add_derived, add_digit_keys, align_categories, feature_set, model_frame

OUT = OUTPUTS / "models_v2"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 7, 2026]
MODELS = {
    "lgb_reg_x3": ("lgb", dict(learning_rate=0.02, min_child_samples=200, feature_fraction=0.6,
                               bagging_fraction=0.7, lambda_l2=5.0), 3),
    "lgb_shallow": ("lgb", dict(num_leaves=7, min_child_samples=100, learning_rate=0.05), 1),
    "xgb": ("xgb", dict(eta=0.03, max_depth=6, subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
                        max_bin=256), 1),
}


def frames():
    train, test, _ = load()
    X, X_test = align_categories(model_frame(train), model_frame(test))
    base = list(X.columns)
    X, X_test = add_derived([X, X_test])
    X, X_test = add_digit_keys([X, X_test], [RAW / "train.csv", RAW / "test.csv"])
    return train[TARGET].to_numpy(), X, X_test, base


if __name__ == "__main__":
    y, X, X_test, base = frames()
    raw, te = feature_set("eivd", base)
    rows = []
    for s in SEEDS:
        res = run_cv_multi(X, y, X_test, raw, te, fold_seed=s, models=MODELS, log=lambda m: print(m, flush=True))
        for name, r in res.items():
            np.save(OUT / f"oof_{name}_{s}.npy", r["oof"])
            np.save(OUT / f"test_{name}_{s}.npy", r["test"])
            rows.append(dict(model=name, fold_seed=s, auc=r["auc"], mean_trees=np.mean(r["iters"])))
        pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)
        print(pd.DataFrame(rows).query("fold_seed == @s").round(5).to_string(index=False), flush=True)
    print(pd.DataFrame(rows).pivot(index="fold_seed", columns="model", values="auc").round(5))
