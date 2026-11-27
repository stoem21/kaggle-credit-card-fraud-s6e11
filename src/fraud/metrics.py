"""One evaluation function for every model and baseline (competition metric: ROC AUC on probabilities)."""
import numpy as np
from sklearn.metrics import roc_auc_score


def auc(y_true, y_prob) -> float:
    return float(roc_auc_score(np.asarray(y_true), np.asarray(y_prob)))
