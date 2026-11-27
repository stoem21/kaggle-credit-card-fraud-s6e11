"""Target-encoding variants vs "ei" (smoothing 20), same CV as experiments/confirm_ei.py, paired fold seeds.
Run: .venv/bin/python -m experiments.te_tuning
Writes outputs/te_tuning/{results.csv, oof_<variant>_<seed>.npy, test_<variant>_<seed>.npy}
The ei reference is read from outputs/confirm_ei/results.csv (same code, same folds).
"""
import numpy as np
import pandas as pd

from src.fraud.config import OUTPUTS, TARGET
from src.fraud.cv import run_cv
from src.fraud.data import load
from src.fraud.features import align_categories, feature_set, model_frame

OUT = OUTPUTS / "te_tuning"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 7, 2026]
VARIANTS = {  # name -> (feature set, smoothing)
    "ei_s5": ("ei", 5),
    "ei_s100": ("ei", 100),
    "ei_te_all": ("ei_te_all", 20),
    "ei_no_noise": ("ei_no_noise", 20),
}

train, test, _ = load()
y = train[TARGET].to_numpy()
X, X_test = align_categories(model_frame(train), model_frame(test))

rows = []
for name, (fs, smoothing) in VARIANTS.items():
    raw_cols, te_cols = feature_set(fs, X.columns)
    for s in SEEDS:
        r = run_cv(X, y, X_test, raw_cols, te_cols, fold_seed=s, smoothing=smoothing, log=lambda m: None)
        np.save(OUT / f"oof_{name}_{s}.npy", r["oof"])
        np.save(OUT / f"test_{name}_{s}.npy", r["test"])
        rows.append(dict(set=name, fold_seed=s, auc=r["auc"], mean_trees=np.mean(r["iters"])))
        print(f"{name} fold_seed={s} AUC={r['auc']:.5f} trees={np.mean(r['iters']):.0f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)

ref = pd.read_csv(OUTPUTS / "confirm_ei" / "results.csv")
res = pd.concat([ref[ref["set"].isin(["raw", "ei"])], pd.DataFrame(rows)], ignore_index=True)
wide = res.pivot(index="fold_seed", columns="set", values="auc")
delta = wide.sub(wide["ei"], axis=0)
print("\nAUC\n", wide.round(5), "\n\ndelta vs ei\n", delta.round(5), "\n\nmean delta vs ei\n", delta.mean().round(5))
