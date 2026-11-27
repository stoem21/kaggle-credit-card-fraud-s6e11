"""How well does the current best blend (x3_both, 3-seed OOF) rank fraud inside each group of train rows."""
import numpy as np, pandas as pd
from scipy.stats import rankdata as R
from sklearn.metrics import roc_auc_score as A
from fraud.config import OUTPUTS

tr = pd.read_csv("data/raw/train.csv")
y = tr.is_fraud.astype(int).values
z = tr.merchant_risk_score.values < tr.merchant_risk_score[y == 1].min()
S = [42, 7, 2026]
M = {("models_v2", "lgb_l3_x3"): .4, ("models_v2", "xgb_d2_x3"): .4, ("nn_mlp", "mlp"): .1, ("nn_mlp", "mlp_d"): .1}
o = lambda d, n, s: R(np.load(f"outputs/{d}/oof_{n}_{s}.npy"))
def bl(s):
    p = sum(w * o(d, n, s) for (d, n), w in M.items()); return R(np.where(z, p - 1e9, p))
p = R(np.mean([bl(s) for s in S], 0))
print(f"overall AUC {A(y, p):.5f}")
legit = np.sort(p[y == 0])
loss = np.zeros(len(y)); f = y == 1
loss[f] = 1 - np.searchsorted(legit, p[f]) / len(legit)   # share of all legit ranked above each fraud

flags = ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
         "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
g = pd.DataFrame(index=tr.index)
g["n_flags"] = tr[flags].astype(bool).sum(axis=1).clip(upper=3).astype(str).replace({"3": "3+"})
g["cvv_retry"] = tr.cvv_retry_count.clip(upper=2).astype(str).replace({"2": "2+"})
g["no_auth"] = np.where(tr.auth_method == "No Authentication", "yes", "no")
strong = (tr.cvv_retry_count > 0).astype(int) + (tr.auth_method == "No Authentication") + tr[flags].astype(bool).sum(axis=1)
g["n_strong_signals"] = strong.clip(upper=3).astype(str).replace({"3": "3+"})
g["merchant_risk_band"] = pd.cut(tr.merchant_risk_score, [-1, 13.1, 30, 45, 60, 101],
                                 labels=["<13.1 (zone1)", "13-30", "30-45", "45-60", "60+"]).astype(str)
for c in ["merchant_category", "channel", "auth_method", "card_type", "device_type", "prior_disputes"]:
    g[c] = tr[c].astype(str)

rows = []
for col in g.columns:
    for lev, idx in g.groupby(col).groups.items():
        m = np.zeros(len(y), bool); m[idx] = True
        nf = y[m].sum()
        auc = A(y[m], p[m]) if 0 < nf < m.sum() else np.nan
        rows.append(dict(grouping=col, group=lev, rows=m.sum(), row_share=m.mean(), fraud_rate=y[m].mean(),
                         fraud_share=nf / f.sum(), loss_share=loss[m].sum() / loss.sum(), auc_inside=auc,
                         mean_loss_per_fraud=loss[m & f].mean() if nf else np.nan))
out = pd.DataFrame(rows)
(OUTPUTS / "eda").mkdir(parents=True, exist_ok=True)
out.to_csv(OUTPUTS / "eda" / "segments.csv", index=False)
pd.set_option("display.width", 200)
for col, d in out.groupby("grouping", sort=False):
    print(f"\n== {col}")
    print(d.drop(columns="grouping").to_string(index=False, formatters={
        "rows": "{:,}".format, "row_share": "{:.1%}".format, "fraud_rate": "{:.2%}".format,
        "fraud_share": "{:.1%}".format, "loss_share": "{:.1%}".format, "auc_inside": "{:.4f}".format,
        "mean_loss_per_fraud": "{:.3f}".format}))
