"""Paths and column names shared by every script. Run scripts from the repo root: python -m src.fraud.<module>"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
# FRAUD_OUTPUTS: write (and read) predictions somewhere else, e.g. for a reproducibility re-run
OUTPUTS = Path(os.environ.get("FRAUD_OUTPUTS", ROOT / "outputs"))
SUBMISSION = ROOT / "submission"

ID = "id"
TARGET = "is_fraud"
SEED = 42
N_FOLDS = 5
FOLD_SEEDS = [42, 7, 2026]               # main fold seeds (every model)
EXTRA_FOLD_SEEDS = [11, 123, 31337, 777]  # added later for the final members (7 seeds in all)
