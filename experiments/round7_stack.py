"""Round 7 stacking: logistic regression on member OOF scores vs the v3 fixed-weight rank blend.

For each fold seed s, members' OOF for seed s come from that seed's 5 folds. The stacker is scored on the SAME 5 folds
(fit on 4 folds' OOF, predict the 5th), so no row is scored by a stacker that saw its label. Member inputs: logit of
the score (GLM decision values as they are), standardized. Zone 1 applied after. Helpers: src/fraud/stack.py.
Run: .venv/bin/python -m experiments.round7_stack [set name ...] [--seeds=42,7,2026,11,123,31337,777]
(default: every set, seeds 42/7/2026). A member missing for a seed is an error, except in the set "all".
Prints per-seed AUC and delta vs the v3 blend; writes outputs/round7/stack_eval[_<sets>][_<seeds>].csv
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.fraud.cli import check_argv
from src.fraud.config import FOLD_SEEDS, OUTPUTS
from src.fraud.metrics import auc
from src.fraud.stack import DIRS, feat, labels_and_zone, load, stack_oof, z1

V3 = {"ebm": 0.5, "lgb_l3_b1023_x3": 0.2, "xgb_d2_x3": 0.2, "mlp": 0.05, "mlp_d": 0.05}  # submission 3 weights
SETS = {
    "v3_members": list(V3),
    "v3+glm": list(V3) + ["glm_plain"],
    "v3+glm+k42": list(V3) + ["glm_plain", "lgb_l3_k42", "cat_d2_k42"],              # submission 4
    "v4_x3": list(V3) + ["glm_plain", "lgb_l3_k42_x3", "cat_d2_k42"],                # submission 5 (final)
    "v4_x3+k42nn": list(V3) + ["glm_plain", "lgb_l3_k42_x3", "cat_d2_k42", "ebm_k42", "mlp_k42", "mlp_d_k42"],
    "k42_core": ["glm_plain", "lgb_l3_k42_x3", "cat_d2_k42", "ebm_k42", "mlp_k42", "mlp_d_k42", "xgb_d2_x3"],
    "all": list(DIRS),
}
CS = [0.01, 0.05, 1.0]

if __name__ == "__main__":
    check_argv(options=["seeds"], n_positional=len([a for a in sys.argv[1:] if not a.startswith("--")]))
    seed_arg = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--seeds=")), None)
    seeds = [int(x) for x in seed_arg.split(",")] if seed_arg else FOLD_SEEDS
    sets = [a for a in sys.argv[1:] if not a.startswith("--")] or list(SETS)
    y, zone, _ = labels_and_zone()
    rows = []
    for s in seeds:
        base = z1(sum(w * rankdata(load("oof", n, s)) for n, w in V3.items()), zone)
        b = auc(y, base)
        rows.append(dict(fold_seed=s, method="v3_rank_blend", auc=b, delta=0.0))
        for name in sets:
            members = [m for m in SETS[name] if load("oof", m, s, required=(name != "all")) is not None]
            F = np.column_stack([feat(m, load("oof", m, s)) for m in members])
            for C in CS:
                a = auc(y, z1(stack_oof(F, y, s, C), zone))
                rows.append(dict(fold_seed=s, method=f"stack_{name}_C{C}", auc=a, delta=a - b, n=len(members)))
        for r in rows[-(1 + len(sets) * len(CS)):]:
            print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    df = pd.DataFrame(rows)
    tag = ("_" + "_".join(sets) if sys.argv[1:] else "") + ("_" + seed_arg.replace(",", "_") if seed_arg else "")
    df.to_csv(OUTPUTS / "round7" / f"stack_eval{tag}.csv", index=False)
    print(df.pivot(index="method", columns="fold_seed", values="delta").assign(
        mean=lambda d: d.mean(axis=1)).sort_values("mean").round(5).to_string())
