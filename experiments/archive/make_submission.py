"""Average saved test predictions into a submission file.
Run: .venv/bin/python -m experiments.make_submission <outputs subdir> <set> <out name>
e.g. .venv/bin/python -m experiments.make_submission confirm_ei ei ei_lgbm_3seeds
"""
import sys

import numpy as np

from src.fraud.config import ID, OUTPUTS, SUBMISSION, TARGET
from src.fraud.data import load

subdir, name, out = sys.argv[1:4]
files = sorted((OUTPUTS / subdir).glob(f"test_{name}_*.npy"))
pred = np.mean([np.load(f) for f in files], axis=0)
_, test, sample = load()
assert (sample[ID].to_numpy() == test[ID].to_numpy()).all()
sub = sample[[ID]].copy()
sub[TARGET] = pred
assert len(sub) == 299_984 and sub[TARGET].between(0, 1).all() and sub[TARGET].notna().all()
path = SUBMISSION / f"{out}.csv"
sub.to_csv(path, index=False)
print("averaged:", [f.name for f in files])
print(path, sub.shape, sub[TARGET].describe().round(4).to_dict())
