"""Round 4 (NN side): MLP inputs as in experiments/nn_mlp.py --digits (reference: mlp_d in outputs/nn_mlp).
- mlp_d_ple:     + piecewise-linear encoding (48 quantile bins) of the 15 raw numerics
- mlp_d_seg:     trained only on training-fold rows with 0-1 strong signals (TE/scaling fitted there too)
- mlp_d_ple_seg: both
seg_* OOF is only meaningful for 0-1 signal rows. Fold seeds 42/7/2026, same folds as the tree models
(the early-stopping slice of seg models is drawn from the hard rows, so it differs from the trees').
Run in its own process (LightGBM + torch in one process crashes on macOS):
    .venv/bin/python -m experiments.round4_nn
Writes outputs/round4/{results_nn.csv, oof_<model>_<seed>.npy, test_<model>_<seed>.npy}
"""
import numpy as np
import pandas as pd

from src.fraud.config import OUTPUTS, RAW, TARGET
from src.fraud.data import CATEGORICAL, bool_columns, load
from src.fraud.features import (DIGIT_KEYS, TE_EXTRA, TE_NUMERIC, add_derived, add_digit_keys, align_categories,
                                 model_frame, strong_signals)
from src.fraud.metrics import auc
from src.fraud.nn import DEVICE, run_cv_nn

OUT = OUTPUTS / "round4"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 7, 2026]
train, test, _ = load()
y = train[TARGET].to_numpy()
sig = strong_signals(train)
hard = sig <= 1
BOOL = bool_columns(train)
X, X_test = align_categories(model_frame(train), model_frame(test))
X, X_test = add_derived([X, X_test])
X, X_test = add_digit_keys([X, X_test], [RAW / "train.csv", RAW / "test.csv"])
NUM = [c for c in model_frame(train).columns if c not in CATEGORICAL + BOOL] + ["vel_ex", "mr_ex"]
TE = TE_NUMERIC + TE_EXTRA + DIGIT_KEYS
MODELS = {"mlp_d_ple": dict(ple_bins=48), "mlp_d_seg": dict(train_mask=hard),
          "mlp_d_ple_seg": dict(ple_bins=48, train_mask=hard)}
print("device", DEVICE, "| numerics", len(NUM), flush=True)
rows = []
for s in SEEDS:
    for name, kw in MODELS.items():
        r = run_cv_nn(X, y, X_test, NUM, BOOL, TE, fold_seed=s, log=lambda m: print(name, m, flush=True), **kw)
        np.save(OUT / f"oof_{name}_{s}.npy", r["oof"]); np.save(OUT / f"test_{name}_{s}.npy", r["test"])
        rows.append(dict(model=name, fold_seed=s, auc=r["auc"], auc_hard=auc(y[hard], r["oof"][hard]),
                         auc_quiet=auc(y[sig == 0], r["oof"][sig == 0]), mean_epochs=np.mean(r["epochs"])))
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(OUT / "results_nn.csv", index=False)
print(pd.DataFrame(rows).round(5).to_string(index=False))
