"""Round 7: plain GLM (idea from public notebooks by goodpjw2008 / lourenswalters; reimplemented on our folds).

Logistic regression on:
- standardized raw numerics (log1p of amount, balance, hours), flags, one-hot categoricals (+ hour, weekday);
- logit of in-fold smoothed target encoding (src.fraud.features.target_encode, inner 5-fold) of
  every column's exact value, and of digit keys of the 6 float columns, keys read from k = rint(100 * x):
  cent, integer, tens, tenths, leading1, leading2, length (6 x 7 = 42 keys).
Variants:
- glm_plain:     the recipe above (TE smoothing 50).
- glm_plain_k12: + our 12 text digit keys (last char / n decimals / fraction, src.fraud.features.add_digit_keys)
                 + vel_ex (raw and TE).
- glm_s20:       glm_plain with TE smoothing 20.
- glm_C01:       glm_plain with C 0.1.
- glm_pairs:     glm_plain + TE of all pairs of (5 categoricals, cvv_retry_count, txn_count_last_24h,
                 prior_disputes) = 28 pair keys.
Run: .venv/bin/python -m experiments.round7_glm glm_plain [--smoke] [--seeds=11,123]
Writes outputs/round7/{results_<model>.csv, oof_<model>_<seed>.npy, test_<model>_<seed>.npy}
"""
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.fraud.cli import check_argv, guard_outputs
from src.fraud.config import FOLD_SEEDS, OUTPUTS, RAW, TARGET
from src.fraud.data import CATEGORICAL, load
from src.fraud.features import DIGIT_KEYS, add_derived, add_digit_keys, k42, strong_signals, target_encode
from src.fraud.metrics import auc

check_argv(flags=["smoke"], options=["seeds"], n_positional=1)
MODEL = sys.argv[1]
SMOKE = "--smoke" in sys.argv
SEED_ARG = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--seeds=")), None)
OUT = OUTPUTS / "round7"
OUT.mkdir(parents=True, exist_ok=True)
SMOOTH = 20 if MODEL == "glm_s20" else 50
C = 0.1 if MODEL == "glm_C01" else 1.0

NUMERIC = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd", "velocity_score",
           "merchant_risk_score", "txn_count_last_24h", "cvv_retry_count", "prior_disputes", "card_age_months",
           "customer_age"]
LOG_COLS = ["amount_usd", "account_balance_usd", "hours_since_last_txn"]
CATS = CATEGORICAL + ["time_of_day_hour", "day_of_week"]

train, test, _ = load()
FLAGS = [c for c in train.columns if train[c].dtype == bool]
y = train[TARGET].to_numpy()
sig = strong_signals(train)
hard = sig <= 1


def build(df, extra=None):
    """Dense label-free part + key table (as strings) for TE."""
    dense = pd.DataFrame(index=df.index)
    for c in NUMERIC:
        dense[c] = np.log1p(df[c]) if c in LOG_COLS else df[c].astype(float)
    for c in FLAGS:
        dense[c] = df[c].astype(float)
    keys = pd.concat([df[NUMERIC + CATS + FLAGS], k42(df)], axis=1)
    if MODEL == "glm_pairs":
        from itertools import combinations
        pc = CATEGORICAL + ["cvv_retry_count", "txn_count_last_24h", "prior_disputes"]
        for a, b in combinations(pc, 2):
            keys[f"{a}|{b}"] = df[a].astype(str) + "|" + df[b].astype(str)
    if extra is not None:
        dense["vel_ex"] = extra["vel_ex"].to_numpy()
        keys["vel_ex"] = extra["vel_ex"].round(4).to_numpy()
        for c in DIGIT_KEYS:
            keys[c] = extra[c].to_numpy()
    return dense, keys.astype(str)


ex_tr = ex_te = None
if MODEL == "glm_plain_k12":
    ex_tr, ex_te = add_derived([train, test])
    ex_tr, ex_te = add_digit_keys([ex_tr, ex_te], [RAW / "train.csv", RAW / "test.csv"])
elif MODEL not in ("glm_plain", "glm_s20", "glm_C01", "glm_pairs"):
    raise ValueError(MODEL)
D_tr, K_tr = build(train, ex_tr)
D_te, K_te = build(test, ex_te)
onehot = pd.get_dummies(pd.concat([train[CATS], test[CATS]]).astype(str), dtype=float)
OH_tr, OH_te = onehot.iloc[:len(train)].to_numpy(), onehot.iloc[len(train):].to_numpy()
print(f"{MODEL}: {D_tr.shape[1]} dense, {OH_tr.shape[1]} one-hot, {K_tr.shape[1]} TE keys", flush=True)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


SEEDS = [42] if SMOKE else ([int(s) for s in SEED_ARG.split(",")] if SEED_ARG else FOLD_SEEDS)
RESULTS = f"results_{MODEL}" + ("_" + "_".join(map(str, SEEDS)) if SEED_ARG else "") + ".csv"
if not SMOKE:
    guard_outputs(OUT, [MODEL], SEEDS)
rows = []
for s in SEEDS:
    oof, tst = np.zeros(len(y)), np.zeros(len(test))
    for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=s).split(D_tr, y)):
        t0 = time.time()
        te_tr, te_va, te_te = [], [], []
        for c in K_tr.columns:
            a, (b, t) = target_encode(K_tr[c].to_numpy()[tr], y[tr], [K_tr[c].to_numpy()[va], K_te[c].to_numpy()],
                                      smoothing=SMOOTH, seed=s)
            te_tr.append(logit(a)); te_va.append(logit(b)); te_te.append(logit(t))
        dense_tr = np.c_[D_tr.to_numpy()[tr], np.column_stack(te_tr)]
        dense_va = np.c_[D_tr.to_numpy()[va], np.column_stack(te_va)]
        dense_te = np.c_[D_te.to_numpy(), np.column_stack(te_te)]
        sc = StandardScaler().fit(dense_tr)
        Xtr = np.c_[sc.transform(dense_tr), OH_tr[tr]]
        Xva = np.c_[sc.transform(dense_va), OH_tr[va]]
        Xte = np.c_[sc.transform(dense_te), OH_te]
        m = LogisticRegression(C=C, max_iter=2000).fit(Xtr, y[tr])
        oof[va] = m.decision_function(Xva)
        tst += m.decision_function(Xte) / 5
        mh = hard[va]
        print(f"  seed {s} fold {k} {MODEL}: auc={auc(y[va], oof[va]):.5f} "
              f"auc_hard={auc(y[va][mh], oof[va][mh]):.5f} {time.time() - t0:.0f}s", flush=True)
        if SMOKE:
            sys.exit(0)
    np.save(OUT / f"oof_{MODEL}_{s}.npy", oof); np.save(OUT / f"test_{MODEL}_{s}.npy", tst)
    rows.append(dict(model=MODEL, fold_seed=s, auc=auc(y, oof), auc_hard=auc(y[hard], oof[hard]),
                     auc_quiet=auc(y[sig == 0], oof[sig == 0])))
    print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / RESULTS, index=False)
