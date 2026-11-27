"""MLP blend member on eiv-style inputs, fold seeds 42/7/2026 (same folds as the tree models).
TE (logit) of 13 numerics, rank-gauss numerics incl. vel_ex, 6 flags, embeddings of 5 categoricals.
Run: .venv/bin/python -m experiments.nn_mlp [--quick] [--digits] [--seeds=11,123]
--digits: also TE of the 12 digit keys (add_digit_keys); saved as model "mlp_d".
--k42: also TE of the 42 numeric digit keys (src.fraud.features.k42); saved as "mlp_k42" (or "mlp_d_k42" with --digits).
Writes outputs/nn_mlp/{results.csv, oof_mlp_<seed>.npy, test_mlp_<seed>.npy}
"""
import sys

import numpy as np
import pandas as pd

from src.fraud.cli import check_argv, guard_outputs
from src.fraud.config import FOLD_SEEDS, OUTPUTS, RAW, TARGET
from src.fraud.data import CATEGORICAL, bool_columns, load
from src.fraud.features import (DIGIT_KEYS, TE_EXTRA, TE_NUMERIC, add_derived, add_digit_keys, align_categories,
                                 model_frame)
from src.fraud.nn import DEVICE, run_cv_nn

check_argv(flags=["quick", "digits", "k42"], options=["seeds"])
QUICK = "--quick" in sys.argv
DIGITS = "--digits" in sys.argv
K42 = "--k42" in sys.argv
NAME = ("mlp_d" if DIGITS else "mlp") + ("_k42" if K42 else "")
OUT = OUTPUTS / ("nn_mlp_quick" if QUICK else "nn_mlp")
OUT.mkdir(parents=True, exist_ok=True)
SEED_ARG = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--seeds=")), None)
SEEDS = [42] if QUICK else ([int(x) for x in SEED_ARG.split(",")] if SEED_ARG else FOLD_SEEDS)
RESULTS = f"results_{NAME}" + ("_" + "_".join(map(str, SEEDS)) if SEED_ARG else "") + ".csv"
if not QUICK:
    guard_outputs(OUT, [NAME], SEEDS)

train, test, _ = load()
if QUICK:  # smoke test only: 20% of rows
    train = train.sample(frac=0.2, random_state=0).reset_index(drop=True)
y = train[TARGET].to_numpy()
BOOL = bool_columns(train)
X, X_test = align_categories(model_frame(train), model_frame(test))
X, X_test = add_derived([X, X_test])
if DIGITS:
    X, X_test = add_digit_keys([X, X_test], [RAW / "train.csv", RAW / "test.csv"])
NUM = [c for c in model_frame(train).columns if c not in CATEGORICAL + BOOL] + ["vel_ex", "mr_ex"]
TE = TE_NUMERIC + TE_EXTRA + (DIGIT_KEYS if DIGITS else [])
if K42:
    from src.fraud.features import K42_COLS, k42
    X, X_test = pd.concat([X, k42(X)], axis=1), pd.concat([X_test, k42(X_test)], axis=1)
    TE = TE + K42_COLS
print("device", DEVICE, "| dense inputs", len(NUM) + len(BOOL) + len(TE), flush=True)

rows = []
for s in SEEDS:
    r = run_cv_nn(X, y, X_test, NUM, BOOL, TE, fold_seed=s, log=lambda m: print(m, flush=True))
    np.save(OUT / f"oof_{NAME}_{s}.npy", r["oof"])
    np.save(OUT / f"test_{NAME}_{s}.npy", r["test"])
    rows.append(dict(model=NAME, fold_seed=s, auc=r["auc"], mean_epochs=np.mean(r["epochs"])))
    print(f"{NAME} fold_seed={s} AUC={r['auc']:.5f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / RESULTS, index=False)
