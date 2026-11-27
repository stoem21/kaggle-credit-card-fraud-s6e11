"""Stacked submission: logistic-regression stacker on member scores (src/fraud/stack.py) + zone 1.

For each fold seed: fit the stacker (C 0.05) on all members' OOF of that seed, apply it to the members' test
predictions of that seed (each a 5-fold average). Test score = mean over seeds of the stacker's decision values.
Reported OOF = mean over seeds of the per-seed nested stack OOF ranks (stacker fit on 4 folds, scored on the 5th).
Also writes submission/<out name>.manifest.json: members, seeds, sha256 of every input file, stacker weights, OOF.
Run: .venv/bin/python -m experiments.make_stack_submission <set name> <out name> [--seeds=...]
Final submission: make_stack_submission v4_x3 stack_v5_x3_7seeds_zone1 --seeds=42,7,2026,11,123,31337,777
"""
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from experiments.round7_stack import SETS
from src.fraud.cli import check_argv
from src.fraud.config import FOLD_SEEDS, ID, RAW, ROOT, SUBMISSION, TARGET
from src.fraud.metrics import auc
from src.fraud.stack import C_STACK, feat, fit_stacker, labels_and_zone, member_file, stack_oof, z1

check_argv(options=["seeds"], n_positional=2)
set_name, out_name = [a for a in sys.argv[1:] if not a.startswith("--")]
seed_arg = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--seeds=")), None)
seeds = [int(x) for x in seed_arg.split(",")] if seed_arg else FOLD_SEEDS
y, zone, t1 = labels_and_zone()
test = pd.read_csv(RAW / "test.csv", usecols=[ID, "merchant_risk_score"])
sample = pd.read_csv(RAW / "sample_submission.csv")
assert (sample[ID].to_numpy() == test[ID].to_numpy()).all()

members = SETS[set_name]
print(f"members ({len(members)}): {members}")
manifest = dict(set=set_name, members=members, fold_seeds=seeds, C=C_STACK, zone1_threshold=float(t1), inputs={},
                per_seed={})
oof_ranks, test_scores = [], []
for s in seeds:
    files = {(k, m): member_file(k, m, s) for k in ("oof", "test") for m in members}
    for f in files.values():
        manifest["inputs"][str(f.relative_to(f.parents[1]))] = hashlib.sha256(f.read_bytes()).hexdigest()
    F = np.column_stack([feat(m, np.load(files["oof", m])) for m in members])
    T = np.column_stack([feat(m, np.load(files["test", m])) for m in members])
    o = stack_oof(F, y, s)
    a = auc(y, z1(o, zone))
    print(f"  seed {s}: nested stack OOF {a:.5f}")
    oof_ranks.append(rankdata(o))
    sc, m = fit_stacker(F, y)
    w = dict(zip(members, np.round(m.coef_[0], 4).tolist()))
    print(f"    weights: {w}")
    manifest["per_seed"][s] = dict(nested_oof_auc=round(a, 6), weights=w)
    test_scores.append(m.decision_function(sc.transform(T)))
oof = np.mean(oof_ranks, axis=0)
manifest["oof_auc_seed_averaged"] = round(auc(y, z1(oof, zone)), 6)
print(f"OOF (seed-averaged ranks, zone 1): {manifest['oof_auc_seed_averaged']:.5f}")
tz = test["merchant_risk_score"].to_numpy() < t1
pred = z1(np.mean(test_scores, axis=0), tz)
sub = sample[[ID]].copy()
sub[TARGET] = (rankdata(pred) - 0.5) / len(pred)
out = SUBMISSION / f"{out_name}.csv"
if out.exists() and "--force" not in sys.argv:
    raise SystemExit(f"{out} exists; refusing to overwrite (use another name or --force)")
sub.to_csv(out, index=False)
manifest["submission_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
(SUBMISSION / f"{out_name}.manifest.json").write_text(json.dumps(manifest, indent=1))
print(f"zone 1 test rows: {tz.sum():,}; wrote {out.relative_to(ROOT)} {sub.shape} + manifest")
