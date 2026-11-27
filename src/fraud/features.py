"""Feature sets and in-fold target encoding.

A feature set is described by two lists:
- raw: columns passed to the model as they are,
- te:  columns whose individual values are target-encoded (numeric values treated as categories).
Target encoding uses labels, so it is computed inside each CV training fold (see cv.py), never on all of train.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import ID, TARGET
from .data import CATEGORICAL

# Weak on their own in EDA (single-feature AUC 0.50-0.51); dropped as raw columns in "ei", kept as TE.
NO_SIGNAL = ["time_of_day_hour", "day_of_week", "card_age_months", "distance_from_home_km", "account_balance_usd"]
# Numeric columns target-encoded as categories (feature screen 2026-11-25, see EXPERIMENTS.md).
TE_NUMERIC = ["merchant_risk_score", "velocity_score", "amount_usd", "hours_since_last_txn", "account_balance_usd",
              "distance_from_home_km", "customer_age", "card_age_months"]


def model_frame(df):
    """Model-ready copy: booleans -> int, categoricals -> pandas category (same levels in train and test)."""
    out = df.drop(columns=[c for c in (ID, TARGET) if c in df.columns]).copy()
    for c in out.columns:
        if out[c].dtype == bool:
            out[c] = out[c].astype(int)
    return out


def align_categories(train_x, test_x):
    for c in CATEGORICAL:
        levels = sorted(set(train_x[c]) | set(test_x[c]))
        train_x[c] = pd.Categorical(train_x[c], categories=levels)
        test_x[c] = pd.Categorical(test_x[c], categories=levels)
    return train_x, test_x


# Low-cardinality numerics not in TE_NUMERIC.
TE_EXTRA = ["txn_count_last_24h", "cvv_retry_count", "prior_disputes", "time_of_day_hour", "day_of_week"]


def feature_set(name, columns):
    """Return (raw columns, te columns) for a named feature set."""
    columns = list(columns)
    if name == "raw":
        return columns, []
    if name == "ei":
        return [c for c in columns if c not in NO_SIGNAL], list(TE_NUMERIC)
    if name == "eiv":  # ei + vel_ex (needs add_derived)
        return [c for c in columns if c not in NO_SIGNAL] + ["vel_ex"], list(TE_NUMERIC)
    if name == "eivd":  # eiv + TE of digit keys (needs add_derived and add_digit_keys)
        return [c for c in columns if c not in NO_SIGNAL] + ["vel_ex"], TE_NUMERIC + DIGIT_KEYS
    if name == "ei_te_all":  # ei + TE of the remaining numerics
        return [c for c in columns if c not in NO_SIGNAL], TE_NUMERIC + TE_EXTRA
    if name == "ei_no_noise":  # no-signal columns removed entirely (no raw, no TE)
        return [c for c in columns if c not in NO_SIGNAL], [c for c in TE_NUMERIC if c not in NO_SIGNAL]
    raise ValueError(f"unknown feature set {name!r}")


def _te_map(keys, y, prior, smoothing):
    g = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + prior * smoothing) / (g["count"] + smoothing)


def target_encode(train_keys, y_train, other_keys, smoothing=20, n_inner=5, seed=0):
    """Smoothed mean-target encoding of `train_keys` (values as strings).

    Training rows get out-of-fold values from an inner K-fold, so no row sees its own label.
    Every array in `other_keys` (validation fold, test) gets the mapping fitted on all training rows.
    Unseen values get the prior.
    """
    train_keys = np.asarray(train_keys)
    y_train = np.asarray(y_train)
    prior = y_train.mean()
    out_train = np.full(len(train_keys), prior)
    for a, b in StratifiedKFold(n_inner, shuffle=True, random_state=seed).split(train_keys, y_train):
        m = _te_map(train_keys[a], y_train[a], prior, smoothing)
        out_train[b] = pd.Series(train_keys[b]).map(m).fillna(prior).to_numpy()
    m = _te_map(train_keys, y_train, prior, smoothing)
    return out_train, [pd.Series(np.asarray(k)).map(m).fillna(prior).to_numpy(dtype=float) for k in other_keys]


def te_keys(df, col):
    return df[col].astype(str).to_numpy()


def original_value_stats(frames, original, cols, smoothing=20):
    """For each value of each column: smoothed fraud rate and row count in the ORIGINAL dataset.
    The original's labels are external, so these can be computed once, outside CV, without leakage."""
    prior = original[TARGET].mean()
    for c in cols:
        keys = original[c].astype(str)
        g = original.groupby(keys)[TARGET].agg(["sum", "count"])
        rate = (g["sum"] + prior * smoothing) / (g["count"] + smoothing)
        for f in frames:
            k = f[c].astype(str)
            f["orig_rate_" + c] = k.map(rate).fillna(prior).astype(float).to_numpy()
            f["orig_n_" + c] = k.map(g["count"]).fillna(0).astype(float).to_numpy()
    return frames


def add_derived(frames):
    """Label-free derived columns, computed on train+test together (no target used).

    vel_ex:   velocity_score minus its mean within (txn_count_last_24h clipped at 9, 20 quantile bins of
              hours_since_last_txn): the part of velocity not explained by count and recency.
    mr_ex:    merchant_risk_score minus its mean within merchant_category (risk relative to peer merchants).
    amt_bin50: amount_usd in 50 quantile bins (key for a lower-variance TE than exact amount values).
    cat_amt20: merchant_category x amount_usd in 20 quantile bins (key for a pair TE).
    """
    allx = pd.concat(frames, ignore_index=True)
    cnt = allx["txn_count_last_24h"].clip(upper=9).astype(str)
    hrs = pd.qcut(allx["hours_since_last_txn"], 20, labels=False, duplicates="drop").astype(str)
    cell = cnt + "|" + hrs
    allx["vel_ex"] = allx["velocity_score"] - allx.groupby(cell)["velocity_score"].transform("mean")
    cat = allx["merchant_category"].astype(str)
    allx["mr_ex"] = allx["merchant_risk_score"] - allx.groupby(cat)["merchant_risk_score"].transform("mean")
    allx["amt_bin50"] = pd.qcut(allx["amount_usd"], 50, labels=False, duplicates="drop")
    allx["cat_amt20"] = cat + "|" + pd.qcut(allx["amount_usd"], 20, labels=False, duplicates="drop").astype(str)
    out, start = [], 0
    for f in frames:
        part = allx.iloc[start:start + len(f)].set_index(f.index)
        g = f.copy()
        for c in ["vel_ex", "mr_ex", "amt_bin50", "cat_amt20"]:
            g[c] = part[c].to_numpy()
        out.append(g)
        start += len(f)
    return out


DIGIT_FLOATS = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd"]
DIGIT_KEYS = [f"{c}__{k}" for c in DIGIT_FLOATS for k in ["last", "ndec", "frac"]]


def add_digit_keys(frames, csv_paths):
    """How each 2-decimal float is written in the CSV (read as text): last character, number of decimals,
    fractional digits. Used only as target-encoding keys (in-fold). frames and csv_paths in the same order."""
    out = []
    for f, path in zip(frames, csv_paths):
        g = f.copy()
        txt = pd.read_csv(path, dtype=str, usecols=DIGIT_FLOATS)
        for c in DIGIT_FLOATS:
            s = txt[c]
            frac = s.str.split(".").str[1].fillna("")
            g[f"{c}__last"] = s.str[-1].to_numpy()
            g[f"{c}__ndec"] = frac.str.len().to_numpy()
            g[f"{c}__frac"] = frac.to_numpy()
        out.append(g)
    return out


SIGNAL_FLAGS = ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
                "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]


def strong_signals(df):
    """Number of strong fraud signals in a raw row: cvv retry > 0, No Authentication, each True flag.
    Uses no labels (see eda/segments.py: 89% of the AUC loss is in rows with 0-1 signals)."""
    return ((df["cvv_retry_count"] > 0).astype(int) + (df["auth_method"] == "No Authentication").astype(int)
            + df[SIGNAL_FLAGS].astype(bool).sum(axis=1)).to_numpy()


# --- 42 numeric digit keys (idea from public notebooks; used by the plain GLM and the round 7 trees / EBM / MLP) ---
K42_FLOATS = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd", "velocity_score",
              "merchant_risk_score"]
K42_PARTS = ["cent", "integer", "tens", "tenths", "leading1", "leading2", "length"]
K42_COLS = [f"k_{p}_{c}" for c in K42_FLOATS for p in K42_PARTS]


def k42(df):
    """7 keys per float column, read from k = rint(100 * x): cent (k % 100), integer part, tens (k // 1000),
    tenths (k // 10), first digit and first two digits of the integer part, number of integer digits."""
    out = {}
    for c in K42_FLOATS:
        k = np.rint(100 * df[c].to_numpy(float)).astype(np.int64)
        integer = k // 100
        length = np.ones(len(k), dtype=np.int64)
        pos = integer > 0
        length[pos] = np.floor(np.log10(integer[pos])).astype(np.int64) + 1
        out[f"k_cent_{c}"] = k % 100
        out[f"k_integer_{c}"] = integer
        out[f"k_tens_{c}"] = k // 1000
        out[f"k_tenths_{c}"] = k // 10
        out[f"k_leading1_{c}"] = integer // 10 ** (length - 1)
        out[f"k_leading2_{c}"] = integer // 10 ** np.maximum(length - 2, 0)
        out[f"k_length_{c}"] = length
    return pd.DataFrame(out, index=df.index)[K42_COLS]
