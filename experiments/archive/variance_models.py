"""Group A: variance reduction and a second model family on the "eiv" features (ei + vel_ex).
Models per fold (shared in-fold TE, same early-stopping slice), fold seeds 42/7/2026:
- lgb_ref:    current LightGBM settings (lr 0.05), 1 model  -> reference
- lgb_reg_x3: regularized LightGBM (lr 0.02, min_child_samples 200, feature_fraction 0.6,
              bagging_fraction 0.7, lambda_l2 5), 3 model seeds averaged. Its bag 0 = A2 alone.
- xgb:        XGBoost hist, depth 6, eta 0.03, subsample/colsample 0.7, min_child_weight 5
Blends: rank average of lgb_reg_x3 and xgb, weight chosen on the 3-fold-seed-averaged OOF.
Run: .venv/bin/python -m experiments.variance_models
Writes outputs/variance_models/{results.csv, oof_<model>_<seed>.npy, test_<model>_<seed>.npy}
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.fraud.config import OUTPUTS, TARGET
from src.fraud.cv import run_cv_multi
from src.fraud.data import load
from src.fraud.features import add_derived, align_categories, feature_set, model_frame
from src.fraud.metrics import auc

OUT = OUTPUTS / "variance_models"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 7, 2026]
MODELS = {
    "lgb_ref": ("lgb", {}, 1),
    "lgb_reg_x3": ("lgb", dict(learning_rate=0.02, min_child_samples=200, feature_fraction=0.6,
                               bagging_fraction=0.7, lambda_l2=5.0), 3),
    "xgb": ("xgb", dict(eta=0.03, max_depth=6, subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
                        max_bin=256), 1),
}

train, test, _ = load()
y = train[TARGET].to_numpy()
X, X_test = align_categories(model_frame(train), model_frame(test))
base_cols = list(X.columns)
X, X_test = add_derived([X, X_test])
raw, te = feature_set("eiv", base_cols)

rows = []
for s in SEEDS:
    res = run_cv_multi(X, y, X_test, raw, te, fold_seed=s, models=MODELS, log=lambda m: print(m, flush=True))
    for name, r in res.items():
        np.save(OUT / f"oof_{name}_{s}.npy", r["oof"])
        np.save(OUT / f"test_{name}_{s}.npy", r["test"])
        rows.append(dict(model=name, fold_seed=s, auc=r["auc"], mean_trees=np.mean(r["iters"])))
        if MODELS[name][2] > 1:
            np.save(OUT / f"oof_{name}_bag0_{s}.npy", r["oof_bag0"])
            np.save(OUT / f"test_{name}_bag0_{s}.npy", r["test_bag0"])
            rows.append(dict(model=name + "_bag0", fold_seed=s, auc=r["auc_bag0"], mean_trees=np.nan))
    pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)
    print(pd.DataFrame(rows).query("fold_seed == @s").round(5).to_string(index=False), flush=True)

res = pd.DataFrame(rows)
wide = res.pivot(index="fold_seed", columns="model", values="auc")
print("\nAUC per fold seed\n", wide.round(5), "\n\ndelta vs lgb_ref\n", wide.sub(wide["lgb_ref"], axis=0).round(5))

# seed-averaged OOF (what a 3-fold-seed submission behaves like) and rank blends
def avg(name):
    return np.mean([rankdata(np.load(OUT / f"oof_{name}_{s}.npy")) for s in SEEDS], axis=0)

A = {n: avg(n) for n in ["lgb_ref", "lgb_reg_x3", "lgb_reg_x3_bag0", "xgb"]}
print("\n3-fold-seed averaged OOF AUC:", {n: round(auc(y, v), 5) for n, v in A.items()})
grid = [(w, auc(y, w * A["lgb_reg_x3"] + (1 - w) * A["xgb"])) for w in np.arange(0, 1.01, 0.1)]
for w, a in grid:
    print(f"  blend w_lgb={w:.1f}: {a:.5f}")
