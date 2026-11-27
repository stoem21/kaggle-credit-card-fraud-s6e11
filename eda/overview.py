"""First look at the data. Run: .venv/bin/python -m eda.overview  -> tables in outputs/eda/"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

from src.fraud.config import ID, TARGET, OUTPUTS, SEED
from src.fraud.data import load, CATEGORICAL, bool_columns
from src.fraud.metrics import auc

OUT = OUTPUTS / "eda"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 200)

train, test, sample = load()
BOOL = bool_columns(train)
NUM = [c for c in train.columns if c not in [ID, TARGET] + CATEGORICAL + BOOL]
y = train[TARGET]

print("== shapes", train.shape, test.shape, sample.shape)
print("ids: train", train[ID].min(), "-", train[ID].max(), "| test", test[ID].min(), "-", test[ID].max(),
      "| sample ids == test ids:", (sample[ID].values == test[ID].values).all())
print("target values:", y.value_counts().to_dict(), "rate %.4f" % y.mean())
print("bool cols:", BOOL)
print("cat cols:", CATEGORICAL)
print("num cols:", NUM)

# 1. columns: dtype, missing, unique values
cols = [c for c in train.columns if c not in [ID, TARGET]]
info = pd.DataFrame({
    "dtype": train[cols].dtypes.astype(str),
    "miss_train_%": train[cols].isna().mean() * 100,
    "miss_test_%": test[cols].isna().mean() * 100,
    "nunique_train": train[cols].nunique(),
    "nunique_test": test[cols].nunique(),
})
info.to_csv(OUT / "columns.csv")
print("\n== columns\n", info.round(3))

print("\nduplicate rows (features only): train", train[cols].duplicated().sum(),
      "| test", test[cols].duplicated().sum())

# 2. numeric: summary + single-feature AUC (how well the raw value ranks fraud on its own)
num = train[NUM].describe(percentiles=[.01, .5, .99]).T
num["test_mean"] = test[NUM].mean()
num["neg_share_%"] = (train[NUM] < 0).mean() * 100
num["whole_numbers_%"] = (train[NUM].dropna() % 1 == 0).mean() * 100
num["fraud_mean"] = train.loc[y == 1, NUM].mean()
num["legit_mean"] = train.loc[y == 0, NUM].mean()
num["auc_alone"] = [auc(y[train[c].notna()], train[c].dropna()) for c in NUM]
num["auc_strength"] = (num["auc_alone"] - 0.5).abs()
num = num.sort_values("auc_strength", ascending=False)
num.to_csv(OUT / "numeric.csv")
print("\n== numeric (sorted by single-feature AUC strength)\n", num.round(3))

# 3. booleans: fraud rate when True vs False
rows = []
for c in BOOL:
    rows.append({"col": c, "share_true_%": train[c].mean() * 100, "share_true_test_%": test[c].mean() * 100,
                 "fraud_rate_true_%": y[train[c]].mean() * 100, "fraud_rate_false_%": y[~train[c]].mean() * 100})
bools = pd.DataFrame(rows).set_index("col")
bools["lift"] = bools["fraud_rate_true_%"] / bools["fraud_rate_false_%"]
bools = bools.sort_values("lift", ascending=False)
bools.to_csv(OUT / "booleans.csv")
print("\n== booleans\n", bools.round(3))

# 4. categoricals: share and fraud rate per level, train vs test share
cat_rows = []
for c in CATEGORICAL:
    g = train.groupby(c, dropna=False)[TARGET].agg(["size", "mean"])
    g["share_test_%"] = test[c].value_counts(normalize=True, dropna=False) * 100
    for lvl, r in g.iterrows():
        cat_rows.append({"col": c, "level": lvl, "n": int(r["size"]), "share_%": r["size"] / len(train) * 100,
                         "share_test_%": r["share_test_%"], "fraud_rate_%": r["mean"] * 100})
cats = pd.DataFrame(cat_rows)
cats.to_csv(OUT / "categoricals.csv", index=False)
print("\n== categoricals\n", cats.round(3).to_string(index=False))
print("\nlevels in test not in train:",
      {c: sorted(set(test[c].dropna()) - set(train[c].dropna())) for c in CATEGORICAL})

# 5. adversarial validation: can a model tell train rows from test rows? AUC ~0.5 = same distribution
X = pd.concat([train[cols], test[cols]], ignore_index=True)
for c in CATEGORICAL:
    X[c] = X[c].astype("category")
X[BOOL] = X[BOOL].astype(int)
is_test = np.r_[np.zeros(len(train)), np.ones(len(test))]
oof = np.zeros(len(X))
for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, is_test):
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, verbose=-1, random_state=SEED)
    m.fit(X.iloc[tr], is_test[tr])
    oof[va] = m.predict_proba(X.iloc[va])[:, 1]
print("\n== adversarial validation AUC (train vs test): %.4f" % auc(is_test, oof))

# 6. quick model check: 5-fold LightGBM with default-ish settings, only to see how much signal there is
Xtr = X.iloc[:len(train)]
oof = np.zeros(len(train))
for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, y):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=SEED)
    m.fit(Xtr.iloc[tr], y.iloc[tr])
    oof[va] = m.predict_proba(Xtr.iloc[va])[:, 1]
print("== quick LightGBM 5-fold AUC (not tuned): %.4f" % auc(y, oof))
imp = pd.Series(m.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
imp.to_csv(OUT / "quick_lgbm_gain_last_fold.csv")
print("top gain (last fold):\n", (imp / imp.sum() * 100).round(1).head(15))
