"""Stacking helpers shared by experiments/round7_stack.py and experiments/make_stack_submission.py.

Member predictions live in outputs/<dir>/{oof,test}_<member>_<fold seed>.npy. Every member, and the stacker, use the
same StratifiedKFold(5, shuffle, fold seed) splits of train.csv, so a member's OOF score for a row comes from a model
that never saw that row, and the stacker is scored only on rows outside the folds it was fit on.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .config import N_FOLDS, OUTPUTS, RAW, TARGET

# member -> output directories searched (exactly one must hold a given fold seed's file)
DIRS = {"ebm": ["round4"], "lr_spline": ["round4"], "lgb_l3_b1023_x3": ["round5"],
        "xgb_d2_x3": ["models_v2", "round5"],  # seeds 42/7/2026 from round3_trees, extra seeds from round5_trees
        "lgb_l3_x3": ["models_v2", "round5"], "mlp": ["nn_mlp"], "mlp_d": ["nn_mlp"], "glm_plain": ["round7"],
        "glm_plain_k12": ["round7"], "lgb_l3_k42": ["round7"], "cat_d2_k42": ["round7"],
        "ebm_b4096": ["round4"], "ebm_leaf50": ["round4"], "ebm_ob8": ["round4"], "ebm_k42": ["round4"],
        "lgb_l3_k42_x3": ["round7"], "mlp_k42": ["nn_mlp"], "mlp_d_k42": ["nn_mlp"]}
LOGIT_FREE = {"glm_plain", "glm_plain_k12"}  # already decision values (logits); the rest are probabilities
C_STACK = 0.05


def member_file(kind, name, seed, required=True):
    """Path of outputs/<dir>/<kind>_<name>_<seed>.npy; error if missing (when required) or found in two dirs."""
    hits = [OUTPUTS / d / f"{kind}_{name}_{seed}.npy" for d in DIRS[name]]
    hits = [f for f in hits if f.exists()]
    if len(hits) > 1:
        raise RuntimeError(f"{kind}_{name}_{seed} found in several dirs: {hits}")
    if not hits:
        if required:
            raise FileNotFoundError(f"{kind}_{name}_{seed}.npy not found in {DIRS[name]} under {OUTPUTS}")
        return None
    return hits[0]


def load(kind, name, seed, required=True):
    f = member_file(kind, name, seed, required)
    return None if f is None else np.load(f)


def feat(name, v):
    """Stacker input: logit of a probability (GLM decision values are passed through)."""
    if name in LOGIT_FREE:
        return v
    p = np.clip(v, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def labels_and_zone():
    """y and the zone-1 mask: merchant_risk_score below the lowest value seen on any fraud in train
    (13.1; experiments/zero_zones.py checks that per-fold thresholds give the same value in every fold)."""
    train = pd.read_csv(RAW / "train.csv", usecols=["merchant_risk_score", TARGET])
    y = train[TARGET].astype(int).to_numpy()
    t1 = train.loc[y == 1, "merchant_risk_score"].min()
    return y, train["merchant_risk_score"].to_numpy() < t1, t1


def z1(score, zone):
    """Zone 1 post-processing: rows in the zone go to the bottom of the ranking."""
    return np.where(zone, score - 1e9, score)


def fit_stacker(F, y, C=C_STACK):
    sc = StandardScaler().fit(F)
    return sc, LogisticRegression(C=C, max_iter=2000).fit(sc.transform(F), y)


def stack_oof(F, y, seed, C=C_STACK):
    """Nested stack OOF on the members' own folds: fit on 4 folds' OOF, score the 5th."""
    out = np.zeros(len(y))
    for tr, va in StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(F, y):
        sc, m = fit_stacker(F[tr], y[tr], C)
        out[va] = m.decision_function(sc.transform(F[va]))
    return out
