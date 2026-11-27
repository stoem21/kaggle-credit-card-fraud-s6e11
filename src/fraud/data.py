"""Load the competition CSVs from data/raw/."""
import pandas as pd

from .config import RAW, TARGET

CATEGORICAL = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]


def load():
    train = pd.read_csv(RAW / "train.csv")
    test = pd.read_csv(RAW / "test.csv")
    sample = pd.read_csv(RAW / "sample_submission.csv")
    train[TARGET] = train[TARGET].astype(int)  # stored as 0.0 / 1.0
    return train, test, sample


def bool_columns(df):
    return [c for c in df.columns if df[c].dtype == bool]


def load_original():
    """Original dataset the competition data was generated from (Kaggle uditjain13/credit-card-fraud-detection-2026,
    CC0, 20,000 rows). Downloaded to data/external/. Same columns; `transaction_id` instead of `id`."""
    from .config import ROOT
    o = pd.read_csv(ROOT / "data" / "external" / "credit_card_fraud_2026.csv").drop(columns=["transaction_id"])
    o[TARGET] = o[TARGET].astype(int)
    return o
