"""Definitive test of "how the number is written" keys with the tree model (the linear screen in
eda/digit_screen.py could miss keys whose signal depends on the value itself).
eiv + in-fold TE of last digit, number of decimals and fractional part of the four 2-decimal floats
(amount, hours, distance, balance) = 12 extra TE columns. Compared with lgb_ref of
experiments/variance_models.py (same function, settings and folds).
Run: .venv/bin/python -m experiments.digit_keys
"""
import numpy as np
import pandas as pd

from src.fraud.config import OUTPUTS, RAW, TARGET
from src.fraud.cv import run_cv_multi
from src.fraud.data import load
from src.fraud.features import add_derived, align_categories, feature_set, model_frame

OUT = OUTPUTS / "digit_keys"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 7, 2026]
FL = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd"]

train, test, _ = load()
y = train[TARGET].to_numpy()
X, X_test = align_categories(model_frame(train), model_frame(test))
base_cols = list(X.columns)
X, X_test = add_derived([X, X_test])
raw, te = feature_set("eiv", base_cols)
for frame, f in [(X, "train.csv"), (X_test, "test.csv")]:
    txt = pd.read_csv(RAW / f, dtype=str, usecols=FL)
    for c in FL:
        s = txt[c]
        frame[f"{c}__last"] = s.str[-1].to_numpy()
        frame[f"{c}__ndec"] = s.str.split(".").str[1].fillna("").str.len().to_numpy()
        frame[f"{c}__frac"] = s.str.split(".").str[1].fillna("").to_numpy()
te_digits = te + [f"{c}__{k}" for c in FL for k in ["last", "ndec", "frac"]]

rows = []
for s in SEEDS:
    r = run_cv_multi(X, y, X_test, raw, te_digits, fold_seed=s, models={"lgb_digits": ("lgb", {}, 1)},
                     log=lambda m: None)["lgb_digits"]
    np.save(OUT / f"oof_lgb_digits_{s}.npy", r["oof"]); np.save(OUT / f"test_lgb_digits_{s}.npy", r["test"])
    rows.append(dict(fold_seed=s, auc=r["auc"]))
    print(f"lgb_digits fold_seed={s} AUC={r['auc']:.5f}", flush=True)
ref = pd.read_csv(OUTPUTS / "variance_models" / "results.csv").query("model == 'lgb_ref'").set_index("fold_seed")["auc"]
res = pd.DataFrame(rows).set_index("fold_seed")
res["lgb_ref"] = ref
res["delta"] = res["auc"] - res["lgb_ref"]
res.to_csv(OUT / "results.csv")
print(res.round(5), "\nmean delta", round(res["delta"].mean(), 5))
