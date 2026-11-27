"""Compare a reproducibility re-run (FRAUD_OUTPUTS=<dir> ./reproduce.sh <seed> <name>) with the saved predictions.
For each stack member and fold seed: OOF AUC of both runs, max |difference| of OOF and test predictions, Spearman
correlation of test predictions. Then the stack submission built from the re-run vs the one from saved files.
Run: .venv/bin/python -m experiments.compare_repro <re-run dir> <fold seed> <re-run submission name>
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.round7_stack import SETS
from src.fraud.cli import check_argv
from src.fraud.config import OUTPUTS, ROOT, SUBMISSION
from src.fraud.metrics import auc
from src.fraud.stack import DIRS, labels_and_zone

check_argv(n_positional=3)
rerun, seed, sub_name = ROOT / sys.argv[1], int(sys.argv[2]), sys.argv[3]
y, _, _ = labels_and_zone()


def find(base, kind, name):
    hits = [base / d / f"{kind}_{name}_{seed}.npy" for d in DIRS[name]]
    hits = [f for f in hits if f.exists()]
    assert len(hits) == 1, (base, kind, name, hits)
    return np.load(hits[0])


rows = []
for m in SETS["v4_x3"]:
    o0, o1 = find(OUTPUTS, "oof", m), find(rerun, "oof", m)
    t0, t1 = find(OUTPUTS, "test", m), find(rerun, "test", m)
    rows.append(dict(member=m, auc_saved=auc(y, o0), auc_rerun=auc(y, o1), auc_diff=auc(y, o1) - auc(y, o0),
                     max_abs_oof=np.abs(o1 - o0).max(), max_abs_test=np.abs(t1 - t0).max(),
                     spearman_test=spearmanr(t0, t1)[0], bitwise=bool((o0 == o1).all() and (t0 == t1).all())))
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.to_string(index=False, float_format=lambda v: f"{v:.6g}"))
out = OUTPUTS / "repro" / f"compare_seed{seed}.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
a = pd.read_csv(SUBMISSION / f"{sub_name}.csv")["is_fraud"].to_numpy()
ref = SUBMISSION / f"ref_v5_seed{seed}.csv"
if ref.exists():
    b = pd.read_csv(ref)["is_fraud"].to_numpy()
    print(f"submission {sub_name} vs {ref.name}: max |rank diff| {np.abs(a - b).max():.3g}, "
          f"rows differing {(a != b).sum():,}, Spearman {spearmanr(a, b)[0]:.6f}")
print(f"wrote {out.relative_to(ROOT)}")
