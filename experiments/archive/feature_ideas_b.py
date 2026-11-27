"""Feature ideas from the data analysis (group B), each added to "ei", plus all together.
Same CV as experiments/confirm_ei.py, full data, paired fold seeds 42/7/2026.
- b1_vel_ex:   + vel_ex (raw column)
- b2_mr_ex:    + mr_ex (raw column)
- b3_amt_bin:  TE of amount on 50 quantile bins instead of exact amount values
- b4_cat_amt:  + TE of merchant_category x 20 amount bins
- b_all:       all four
Run: .venv/bin/python -m experiments.archive.feature_ideas_b
"""
import numpy as np
import pandas as pd

from src.fraud.config import OUTPUTS, TARGET
from src.fraud.cv import run_cv
from src.fraud.data import load
from src.fraud.features import add_derived, align_categories, feature_set, model_frame

OUT = OUTPUTS / "feature_ideas_b"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 7, 2026]

train, test, _ = load()
y = train[TARGET].to_numpy()
X, X_test = align_categories(model_frame(train), model_frame(test))
base_cols = list(X.columns)
X, X_test = add_derived([X, X_test])
EI_RAW, EI_TE = feature_set("ei", base_cols)


def variant(b1=False, b2=False, b3=False, b4=False):
    raw, te = list(EI_RAW), list(EI_TE)
    if b1: raw.append("vel_ex")
    if b2: raw.append("mr_ex")
    if b3: te = [c for c in te if c != "amount_usd"] + ["amt_bin50"]
    if b4: te.append("cat_amt20")
    return raw, te


VARIANTS = {
    "b1_vel_ex": variant(b1=True),
    "b2_mr_ex": variant(b2=True),
    "b3_amt_bin": variant(b3=True),
    "b4_cat_amt": variant(b4=True),
    "b_all": variant(True, True, True, True),
}
rows = []
for name, (raw, te) in VARIANTS.items():
    for s in SEEDS:
        r = run_cv(X, y, X_test, raw, te, fold_seed=s, log=lambda m: None)
        np.save(OUT / f"oof_{name}_{s}.npy", r["oof"])
        np.save(OUT / f"test_{name}_{s}.npy", r["test"])
        rows.append(dict(set=name, fold_seed=s, auc=r["auc"], mean_trees=np.mean(r["iters"])))
        print(f"{name} fold_seed={s} AUC={r['auc']:.5f} trees={np.mean(r['iters']):.0f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)

ref = pd.read_csv(OUTPUTS / "confirm_ei" / "results.csv")
res = pd.concat([ref[ref["set"] == "ei"], pd.DataFrame(rows)], ignore_index=True)
wide = res.pivot(index="fold_seed", columns="set", values="auc")
delta = wide.sub(wide["ei"], axis=0)
print("\nAUC\n", wide.round(5), "\n\ndelta vs ei\n", delta.round(5), "\n\nmean delta vs ei\n", delta.mean().round(5))
