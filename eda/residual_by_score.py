"""At the SAME model score, do fraud and legit rows still differ in any column? (hard rows = 0-1 strong signals)
Rows are put in 20 bins of the blend's OOF score (lgb_l3_x3, 3-seed average); inside each bin we compute, for
every column, the AUC of the column alone (fraud vs legit), then average over bins weighted by fraud count.
~0.5 = the model already used everything that column says; far from 0.5 = signal the model missed.
Also: frauds with low score (more than 20% of all legit ranked above) -- which columns are unusual
(outside the legit 5-95% range) for them, compared with legit rows in the same score bins."""
import numpy as np, pandas as pd
from scipy.stats import rankdata as R
from sklearn.metrics import roc_auc_score as A
from src.fraud.features import strong_signals

tr = pd.read_csv("data/raw/train.csv")
txt = pd.read_csv("data/raw/train.csv", dtype=str)
y = tr.is_fraud.astype(int).values
hard = strong_signals(tr) <= 1
p = np.mean([R(np.load(f"outputs/models_v2/oof_lgb_l3_x3_{s}.npy")) for s in [42, 7, 2026]], 0)
num = ["merchant_risk_score", "velocity_score", "customer_age", "amount_usd", "txn_count_last_24h",
       "hours_since_last_txn", "prior_disputes", "account_balance_usd", "distance_from_home_km",
       "card_age_months", "time_of_day_hour", "day_of_week"]
F = tr[num].astype(float).copy()
for c in ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd",
          "merchant_risk_score", "velocity_score"]:
    F[c + "__lastdigit"] = txt[c].str[-1].astype(int)
    F[c + "__ndec"] = txt[c].str.split(".").str[1].fillna("").str.len()
for c in ["merchant_category", "card_type", "auth_method", "channel", "device_type"]:
    F[c] = tr[c].astype("category").cat.codes  # order arbitrary: report |AUC-0.5|
F, yh, ph = F[hard].reset_index(drop=True), y[hard], p[hard]
bins = pd.qcut(R(ph), 20, labels=False)
res = {}
for c in F.columns:
    a, w = [], []
    for b in range(20):
        m = bins == b
        if 0 < yh[m].sum() < m.sum() and F[c][m].nunique() > 1:
            a.append(A(yh[m], F[c][m])); w.append(yh[m].sum())
    if w: res[c] = np.average(a, weights=w)
r = pd.Series(res)
print("column AUC at equal model score (hard rows; |AUC-0.5| sorted):")
print(r.reindex((r - 0.5).abs().sort_values(ascending=False).index).round(4).to_string())

# low-scored frauds: unusual columns vs legit in the same score bins
legit_all = np.sort(p[y == 0])
loss = 1 - np.searchsorted(legit_all, ph) / len(legit_all)
low = (yh == 1) & (loss > 0.2)
print(f"\nlow-scored frauds in hard rows: {low.sum()} (of {(yh == 1).sum()})")
rows = []
for c in num:
    lo, hi = np.percentile(F[c][yh == 0], [5, 95])
    same = (yh == 0) & np.isin(bins, np.unique(bins[low]))
    rows.append(dict(col=c, fraud_low_out=((F[c][low] < lo) | (F[c][low] > hi)).mean(),
                     legit_samebin_out=((F[c][same] < lo) | (F[c][same] > hi)).mean(),
                     fraud_low_mean=F[c][low].mean(), legit_samebin_mean=F[c][same].mean()))
print(pd.DataFrame(rows).round(3).to_string(index=False))
