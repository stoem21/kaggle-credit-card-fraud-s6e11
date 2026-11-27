"""Where does the current best blend lose AUC? (read-only on saved OOF)."""
import itertools
import numpy as np, pandas as pd
from scipy.stats import rankdata as R
from sklearn.metrics import roc_auc_score as A

tr = pd.read_csv("data/raw/train.csv")
y = tr.is_fraud.astype(int).values
z = tr.merchant_risk_score.values < tr.merchant_risk_score[y == 1].min()
S = [42, 7, 2026]
M = {("models_v2", "lgb_l3_x3"): .4, ("models_v2", "xgb_d2_x3"): .4, ("nn_mlp", "mlp"): .1, ("nn_mlp", "mlp_d"): .1}
o = lambda d, n, s: R(np.load(f"outputs/{d}/oof_{n}_{s}.npy"))
def bl(s):
    p = sum(w * o(d, n, s) for (d, n), w in M.items()); return R(np.where(z, p - 1e9, p))
P = {s: bl(s) for s in S}

print("== 1. averaging over fold seeds (OOF of the blend)")
for k in (1, 2, 3):
    v = [A(y, np.mean([P[s] for s in c], 0)) for c in itertools.combinations(S, k)]
    print(f"  {k} seed(s): {np.mean(v):.5f}")
p = R(np.mean([P[s] for s in S], 0)); n = len(y)
print("  rank corr between seeds:", [round(np.corrcoef(P[a], P[b])[0, 1], 4) for a, b in itertools.combinations(S, 2)])

print("\n== 2. AUC loss by fraud row: loss_i = share of legit ranked above fraud i; 1-AUC = mean loss")
pos = np.sort(p[y == 0]); f = np.where(y == 1)[0]
loss = 1 - np.searchsorted(pos, p[f]) / len(pos)
print(f"  1-AUC = {loss.mean():.5f}  n_fraud = {len(f)}")
for q in (0.5, 0.2, 0.05, 0.01):
    m = loss > q; print(f"  frauds with >{q:.0%} legit above: {m.mean():6.1%} of frauds, {loss[m].sum()/loss.sum():6.1%} of the loss")

print("\n== 3. profile: hidden frauds (loss>0.2) vs caught frauds (loss<0.01) vs all legit")
hid, got = f[loss > 0.2], f[loss < 0.01]
num = ["cvv_retry_count", "merchant_risk_score", "velocity_score", "prior_disputes", "txn_count_last_24h", "amount_usd",
       "hours_since_last_txn", "customer_age", "is_foreign_transaction", "is_new_merchant", "used_vpn",
       "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
X = tr[num].astype(float)
print(pd.DataFrame({"hidden": X.iloc[hid].mean(), "caught": X.iloc[got].mean(), "legit": X[y == 0].mean()}).round(3).to_string())
nauth = (tr.auth_method == "No Authentication")
print(f"  No Authentication share: hidden {nauth.iloc[hid].mean():.3f} caught {nauth.iloc[got].mean():.3f} legit {nauth[y==0].mean():.3f}")

print("\n== 4. 'quiet' rows: cvv 0, all flags False, auth not 'No Authentication'")
flags = ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
q = (tr.cvv_retry_count == 0) & ~tr[flags].astype(bool).any(axis=1) & ~nauth
print(f"  quiet rows {q.mean():.1%}, fraud rate {y[q].mean():.4f} vs others {y[~q].mean():.4f}; "
      f"share of frauds {y[q].sum()/y.sum():.1%}; share of loss {loss[q.values[f]].sum()/loss.sum():.1%}")
print(f"  AUC inside quiet {A(y[q], p[q]):.4f}, inside others {A(y[~q], p[~q]):.4f}")

print("\n== 5. rank noise between seeds, for frauds by loss group (std of rank/n across seeds)")
sd = np.std([P[s] / n for s in S], 0)
for name, idx in [("caught", got), ("hidden", hid), ("middle", f[(loss >= 0.01) & (loss <= 0.2)])]:
    print(f"  {name:7s} n={len(idx):5d} mean rank {p[idx].mean()/n:.3f} seed std {sd[idx].mean():.4f}")
