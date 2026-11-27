"""Screen "how the number is written" keys of the 6 float columns (read from the CSV as text).
For each key: (1) AUC of its out-of-fold target encoding alone, vs a shuffled-label control;
(2) gain over a logistic regression on OOF TE logits of all base columns (does the key add information?).
Cheap screen only; winners are confirmed with the full LightGBM CV afterwards.
Run: .venv/bin/python -m eda.digit_screen  -> outputs/eda/digit_screen.csv
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.fraud.config import OUTPUTS, RAW, TARGET
from src.fraud.data import CATEGORICAL
from src.fraud.features import target_encode
from src.fraud.metrics import auc

FLOATS = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd", "velocity_score",
          "merchant_risk_score"]
txt = pd.read_csv(RAW / "train.csv", dtype=str)
y = txt[TARGET].astype(float).astype(int).to_numpy()


def oof_te(keys, yy, seed=0):
    out = np.zeros(len(keys))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(keys, yy):
        _, (b,) = target_encode(keys[tr], yy[tr], [keys[va]], seed=seed)
        out[va] = b
    return out


logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

# base: OOF TE logits of every column's exact value
base_cols = [c for c in txt.columns if c not in ("id", TARGET)]
B = np.column_stack([logit(oof_te(txt[c].to_numpy(), y)) for c in base_cols])
lr = LogisticRegression(max_iter=2000, C=1.0)
cv = StratifiedKFold(5, shuffle=True, random_state=1)
base_auc = auc(y, cross_val_predict(lr, B, y, cv=cv, method="predict_proba")[:, 1])
print(f"base logistic on TE of {len(base_cols)} columns: {base_auc:.5f}", flush=True)

rows = []
rng = np.random.RandomState(0)
for c in FLOATS:
    s = txt[c]
    ip, fp = s.str.split(".").str[0], s.str.split(".").str[1].fillna("")
    keys = {
        "last_digit": s.str[-1],
        "second_last_digit": s.str[-2],
        "first_digit": s.str[0],
        "n_decimals": fp.str.len().astype(str),
        "n_int_digits": ip.str.len().astype(str),
        "frac_part": fp,
        "int_last_digit": ip.str[-1],
        "digit_sum_mod10": s.str.replace(".", "", regex=False).map(lambda d: str(sum(map(int, d)) % 10)),
    }
    for kname, k in keys.items():
        k = k.to_numpy()
        te = oof_te(k, y)
        a = auc(y, te)
        a_shuf = auc(y, oof_te(k, rng.permutation(y)))
        X = np.column_stack([B, logit(te)])
        gain = auc(y, cross_val_predict(lr, X, y, cv=cv, method="predict_proba")[:, 1]) - base_auc
        rows.append(dict(column=c, key=kname, n_levels=len(set(k)), auc_alone=a, auc_shuffled=a_shuf, lr_gain=gain))
        print(f"{c:24s} {kname:18s} levels={len(set(k)):6d} auc={a:.4f} shuffled={a_shuf:.4f} lr_gain={gain:+.5f}",
              flush=True)

res = pd.DataFrame(rows).sort_values("lr_gain", ascending=False)
res.to_csv(OUTPUTS / "eda" / "digit_screen.csv", index=False)
print("\n", res.head(15).round(5).to_string(index=False))
