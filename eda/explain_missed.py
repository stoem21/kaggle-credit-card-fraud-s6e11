"""Why are some frauds missed? Per-feature contributions from an EBM for a few out-of-fold frauds.

EBM (same settings and inputs as the `ebm` stack member, experiments/round4_gam.py) fit on fold 0's training rows
(fold seed 42); the explained rows are fold 0's validation rows, so the model never saw them. Examples: 3 frauds the
final stack misses (score below the 80th percentile, no strong signal) and 1 fraud it catches (top 1%), drawn with
a fixed random seed. A row's EBM score (logit) = intercept + sum of its term contributions; contributions of a
column's raw value, its target encoding and its digit keys are summed under the column name.
Run: .venv/bin/python -m eda.explain_missed   -> outputs/notebook/{ebm_examples.csv, ebm_example_rows.csv}
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from experiments.models_v2 import frames
from experiments.round7_stack import SETS
from src.fraud.config import OUTPUTS
from src.fraud.cv import fold_frames
from src.fraud.data import CATEGORICAL
from src.fraud.features import feature_set, strong_signals
from src.fraud.stack import feat, labels_and_zone, load, stack_oof

SEED = 42
OUT = OUTPUTS / "notebook"
OUT.mkdir(exist_ok=True)
y, X, X_test, base = frames()
raw, te = feature_set("eivd", base)
_, zone, _ = labels_and_zone()
members = SETS["v4_x3"]
score = stack_oof(np.column_stack([feat(m, load("oof", m, SEED)) for m in members]), y, SEED)
pct = rankdata(np.where(zone, score - 1e9, score)) / len(y)
from src.fraud.data import load as load_data
train_raw = load_data()[0]
sig = strong_signals(train_raw)

k, tr, va, Xtr, Xva, *_ = next(fold_frames(X, y, X_test, raw, te, fold_seed=SEED))
tecols = ["te_" + c for c in te]
cats = [c for c in raw if c in CATEGORICAL]


def prep(Z):
    Z = Z.copy()
    p = np.clip(Z[tecols].to_numpy(float), 1e-5, 1 - 1e-5)
    Z[tecols] = np.log(p / (1 - p))
    Z[cats] = Z[cats].astype(str)
    return Z


from interpret.glassbox import ExplainableBoostingClassifier
ebm = ExplainableBoostingClassifier(max_bins=1024, interactions=10, outer_bags=4, n_jobs=-1, random_state=0)
ebm.fit(prep(Xtr), y[tr])

rng = np.random.default_rng(0)
missed = va[(y[va] == 1) & (pct[va] < 0.80) & (sig[va] == 0)]
caught = va[(y[va] == 1) & (pct[va] >= 0.99)]
picks = [("missed", i) for i in rng.choice(missed, 3, replace=False)] + [("caught", rng.choice(caught))]
pos = {r: j for j, r in enumerate(va)}
Z = prep(Xva.iloc[[pos[i] for _, i in picks]])
contrib = ebm.eval_terms(Z)


def column(term):
    if " & " in term:
        return "pairs of columns"
    c = term[3:] if term.startswith("te_") else term
    c = c.split("__")[0]
    return "velocity_score" if c == "vel_ex" else c


rows, show = [], []
for j, (kind, i) in enumerate(picks):
    s = pd.Series(contrib[j], index=ebm.term_names_).groupby(column).sum()
    name = f"{kind} fraud #{j + 1}" if kind == "missed" else "caught fraud"
    for c, v in s.items():
        rows.append(dict(example=name, row=int(i), column=c, contribution=float(v)))
    rows.append(dict(example=name, row=int(i), column="(intercept)", contribution=float(ebm.intercept_[0])))
    show.append(dict(example=name, row=int(i), stack_score_percentile=round(float(pct[i]) * 100, 1),
                     ebm_probability=float(ebm.predict_proba(Z.iloc[[j]])[0, 1]),
                     **train_raw.iloc[i][["merchant_category", "auth_method", "channel", "merchant_risk_score",
                                     "velocity_score", "cvv_retry_count", "customer_age", "amount_usd"]].to_dict()))
pd.DataFrame(rows).to_csv(OUT / "ebm_examples.csv", index=False)
pd.DataFrame(show).to_csv(OUT / "ebm_example_rows.csv", index=False)
print(pd.DataFrame(show).to_string(index=False))
