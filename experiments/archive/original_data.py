"""Does the original dataset help? Compared with "ei" on the same folds (fold seeds 42/7/2026).
- ei_plus_orig_rows: original rows appended to every training fold (never scored).
- ei_orig_stats: per-value fraud rate and count in the original, as extra columns (no rows added).
Run: .venv/bin/python -m experiments.original_data
Writes outputs/original_data/{results.csv, oof_*.npy, test_*.npy}
"""
import numpy as np
import pandas as pd

from src.fraud.config import OUTPUTS, TARGET
from src.fraud.cv import run_cv
from src.fraud.data import load, load_original
from src.fraud.features import TE_NUMERIC, align_categories, feature_set, model_frame, original_value_stats

OUT = OUTPUTS / "original_data"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 7, 2026]

train, test, _ = load()
orig = load_original()
y = train[TARGET].to_numpy()
X, X_test = align_categories(model_frame(train), model_frame(test))
Xo = model_frame(orig)
for c in X.select_dtypes("category").columns:
    Xo[c] = pd.Categorical(Xo[c], categories=X[c].cat.categories)
raw_cols, te_cols = feature_set("ei", X.columns)

Xs, Xs_test = original_value_stats([X.copy(), X_test.copy()], orig, TE_NUMERIC)
stat_cols = [c for c in Xs.columns if c.startswith("orig_")]

VARIANTS = {
    "ei_plus_orig_rows": dict(X=X, X_test=X_test, raw=raw_cols, extra_X=Xo, extra_y=orig[TARGET].to_numpy()),
    "ei_orig_stats": dict(X=Xs, X_test=Xs_test, raw=raw_cols + stat_cols, extra_X=None, extra_y=None),
}
rows = []
for name, v in VARIANTS.items():
    for s in SEEDS:
        r = run_cv(v["X"], y, v["X_test"], v["raw"], te_cols, fold_seed=s, log=lambda m: None,
                   extra_X=v["extra_X"], extra_y=v["extra_y"])
        np.save(OUT / f"oof_{name}_{s}.npy", r["oof"])
        np.save(OUT / f"test_{name}_{s}.npy", r["test"])
        rows.append(dict(set=name, fold_seed=s, auc=r["auc"], mean_trees=np.mean(r["iters"])))
        print(f"{name} fold_seed={s} AUC={r['auc']:.5f} trees={np.mean(r['iters']):.0f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)

ref = pd.read_csv(OUTPUTS / "confirm_ei" / "results.csv")
res = pd.concat([ref[ref["set"] == "ei"], pd.DataFrame(rows)], ignore_index=True)
wide = res.pivot(index="fold_seed", columns="set", values="auc")
delta = wide.sub(wide["ei"], axis=0)
print("\nAUC\n", wide.round(5), "\n\ndelta vs ei\n", delta.round(5), "\n\nmean delta vs ei\n", delta.mean().round(5))
